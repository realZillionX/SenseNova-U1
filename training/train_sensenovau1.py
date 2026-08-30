# Copyright (c) SenseNovaLM contributors. Licensed under Apache-2.0.
# Main training entry point for SenseNova-U1.5.
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from sensenovalm.accelerator import get_accelerator
from sensenovalm.core.context import ParallelMode
from sensenovalm.core.context import global_context as gpc
from sensenovalm.core.trainer_builder import TrainerBuilder
from sensenovalm.initialize import initialize_distributed_env
from sensenovalm.utils.common import get_current_device, parse_args
from sensenovavl.data import build_train_loader_with_data_type
from sensenovavl.train.pipeline import get_model
from sensenovavl.utils.utils import init_pil

# global llm logger
logger = logging.getLogger(__file__)
sensenovalm_accelerator = get_accelerator()


class FixedBatchLoader:
    """Delegate loader metadata while replaying sealed optimizer batches."""

    def __init__(self, base_loader, path):
        fixed_path = Path(path).resolve()
        sidecar = fixed_path.with_suffix(fixed_path.suffix + ".json")
        if not sidecar.is_file():
            raise ValueError("fixed SFT batch artifact has no identity sidecar")
        identity = json.loads(sidecar.read_text(encoding="utf-8"))
        observed = None
        if dist.get_rank() == 0:
            digest = hashlib.sha256()
            with fixed_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            observed = digest.hexdigest()
        values = [observed]
        dist.broadcast_object_list(values, src=0)
        if values[0] != identity.get("sha256"):
            raise ValueError("fixed SFT batch artifact differs from its sealed identity")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "sensenova_u15.sft_ablation_batches.v1":
            raise ValueError("unsupported fixed SFT batch artifact")
        if int(payload["sequence_length"]) != int(gpc.config.data.seq_len):
            raise ValueError("fixed SFT batch sequence length differs from the live config")
        data_parallel_size = gpc.get_world_size(ParallelMode.WEIGHT_DATA)
        local_microbatches = int(gpc.config.data.micro_num)
        global_microbatches = local_microbatches * data_parallel_size
        if int(payload["microbatches_per_optimizer_step"]) != global_microbatches:
            raise ValueError(
                "fixed SFT batch width differs from grad_accm * weight-data parallel size"
            )
        if Path(payload["data_meta"]).resolve() != Path(gpc.config.data.meta_path).resolve():
            raise ValueError("fixed SFT batch data meta differs from the live config")
        data_parallel_rank = gpc.get_local_rank(ParallelMode.WEIGHT_DATA)
        start = data_parallel_rank * local_microbatches
        stop = start + local_microbatches

        def select(value):
            if isinstance(value, torch.Tensor):
                return value[start:stop]
            if isinstance(value, (list, tuple)):
                return value[start:stop]
            if isinstance(value, bool):
                return value
            raise TypeError(f"unsupported fixed SFT batch value: {type(value)!r}")

        selected_batches = []
        for data, labels in payload["batches"]:
            selected_data = {
                name: select(value)
                for name, value in data.items()
                if name not in {"num_samples", "num_padding_tokens"}
            }
            input_ids = selected_data["input_ids"]
            cu_seqlens = selected_data["cu_seqlens"]
            valid_samples = 0
            padding_tokens = 0
            for row, offsets in zip(input_ids, cu_seqlens):
                for left, right in zip(offsets[:-1], offsets[1:]):
                    segment = row[int(left) : int(right)]
                    if bool((segment != 0).any()):
                        valid_samples += 1
                    else:
                        padding_tokens += int(right) - int(left)
            selected_data["num_samples"] = valid_samples
            selected_data["num_padding_tokens"] = padding_tokens
            if isinstance(labels, torch.Tensor):
                selected_labels = labels[start:stop]
            elif isinstance(labels, dict):
                selected_labels = {
                    name: value[start:stop] if value.dim() else value
                    for name, value in labels.items()
                }
            else:
                selected_labels = labels
            selected_batches.append((selected_data, selected_labels))

        self.base_loader = base_loader
        self.batches = selected_batches
        self.ordered_microbatch_sha256 = payload["ordered_microbatch_sha256"]
        self.identity = {
            "path": str(fixed_path),
            "bytes": int(identity["bytes"]),
            "sha256": str(identity["sha256"]),
            "ordered_microbatch_sha256": self.ordered_microbatch_sha256,
            "weight_data_parallel_size": data_parallel_size,
            "weight_data_parallel_rank": data_parallel_rank,
        }

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

    def __getattr__(self, name):
        return getattr(self.base_loader, name)


def patch_inductor_triton_max_block():
    max_block_x = os.environ.get("TORCHINDUCTOR_TRITON_MAX_BLOCK_X")
    if not max_block_x:
        return

    try:
        max_block_x = int(max_block_x)
    except ValueError:
        logger.warning("Invalid TORCHINDUCTOR_TRITON_MAX_BLOCK_X=%s, skip patch.", max_block_x)
        return

    try:
        from torch._inductor.runtime.hints import TRITON_MAX_BLOCK
    except Exception:
        logger.warning("Failed to import torch._inductor.runtime.hints.TRITON_MAX_BLOCK.", exc_info=True)
        return

    old_max_block_x = TRITON_MAX_BLOCK.get("X")
    if old_max_block_x is None or max_block_x > old_max_block_x:
        TRITON_MAX_BLOCK["X"] = max_block_x
        logger.info("Patched TRITON_MAX_BLOCK['X'] from %s to %s.", old_max_block_x, max_block_x)


def main(args):
    if gpc.config.get("MP_SPAWN", False):
        torch.multiprocessing.set_start_method("spawn")

    very_beginning_time = time.time()

    # initialize the train and validation data loader
    train_dl, dataset_types = build_train_loader_with_data_type()
    fixed_batches = os.environ.get("SFT_ABLATION_BATCHES")
    if fixed_batches:
        train_dl = FixedBatchLoader(train_dl, fixed_batches)
        if gpc.is_rank_for_log():
            print(
                "SFT_FIXED_BATCHES "
                + str(
                    {
                        "path": str(Path(fixed_batches).resolve()),
                        "ordered_microbatch_sha256": train_dl.ordered_microbatch_sha256,
                    }
                ),
                flush=True,
            )
    val_dls = None

    # get sensenovavl model
    model = get_model(gpc.config.model, gpc.config.data).to(get_current_device())

    # build trainer
    merged_args = {
        **vars(args),
        "dataset_types": dataset_types,
        "very_begining_time": very_beginning_time,
    }

    trainer = TrainerBuilder(
        model,
        train_dl,
        val_dls,
        **merged_args,
    )

    # training
    trainer.fit()


if __name__ == "__main__":
    patch_inductor_triton_max_block()

    args = parse_args()

    # initialize distributed environment
    initialize_distributed_env(config=args.config, launcher=args.launcher, master_port=args.port, seed=args.seed)
    assert hasattr(gpc, "config") and gpc.config is not None

    init_pil()

    # Run the main function with parsed arguments
    main(args)
