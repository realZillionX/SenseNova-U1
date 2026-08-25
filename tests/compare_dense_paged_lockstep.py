"""Lockstep diagnosis for dense versus paged Flash KV attention."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from sensenova_u1 import set_attn_backend
from sensenova_u1.batch_inference import (
    ContiguousTextBatchSession,
    ContinuousTextBatchEngine,
    TextBatchRequest,
    _apply_repetition_penalty,
)
from sensenova_u1.utils import load_model_and_tokenizer


BABYVISION_DIR = (
    Path(__file__).resolve().parents[1] / "evaluation" / "interleave" / "BabyVision"
)
sys.path.insert(0, str(BABYVISION_DIR))
from infer_babyvision_native_batch import _prepare_item  # noqa: E402


def _penalize(logits, seen_tokens, penalty):
    if penalty == 1.0:
        return logits
    seen = torch.zeros_like(logits, dtype=torch.bool)
    if seen_tokens:
        indexes = torch.tensor(
            tuple(seen_tokens), device=logits.device, dtype=torch.long
        )
        seen[0, indexes] = True
    return _apply_repetition_penalty(logits, seen, penalty)


def _paged_row(layer, block_table, length, page_size, name):
    source = getattr(layer, name)
    pieces = []
    for logical_page in range((length + page_size - 1) // page_size):
        block = int(block_table[0, logical_page].item())
        count = min(page_size, length - logical_page * page_size)
        pieces.append(source[block, :count])
    return torch.cat(pieces, dim=0)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--task-id", type=int, default=5133)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    args = parser.parse_args()

    item = next(
        (
            row
            for row in map(json.loads, Path(args.data).open(encoding="utf-8"))
            if int(row["taskId"]) == args.task_id
        ),
        None,
    )
    if item is None:
        raise ValueError(f"BabyVision task {args.task_id} was not found")
    question, _answer, image_path = _prepare_item(item, Path(args.image_root))
    request = TextBatchRequest(
        prompt=question, images=(image_path,), system_message=""
    )

    set_attn_backend("flash")
    model, tokenizer = load_model_and_tokenizer(
        args.model, dtype=torch.bfloat16, device="cuda"
    )
    dense = ContiguousTextBatchSession(
        model,
        tokenizer,
        (request,),
        device="cuda",
        dtype=torch.bfloat16,
        prefix_sharing=False,
        flash_decode_tokens=args.steps + 1,
    )
    paged = ContinuousTextBatchEngine(
        model,
        tokenizer,
        device="cuda",
        dtype=torch.bfloat16,
        max_batch_size=1,
        max_model_len=65536,
        prefill_chunk_size=2048,
        default_max_new_tokens=args.steps + 1,
        max_kv_tokens=8192,
        kv_page_size=args.page_size,
        prefix_sharing=False,
    )
    paged.submit(request)
    batch = paged.schedule()
    assert batch is not None and paged._cache is not None

    prefix = int(dense.cache.get_seq_length())
    dense_layer = dense.cache.layers[0]
    paged_layer = paged._cache.layers[0]
    paged_prefix_k = _paged_row(
        paged_layer,
        paged._cache.flash_decode_block_table,
        prefix,
        args.page_size,
        "flash_decode_k_cache",
    )
    paged_prefix_v = _paged_row(
        paged_layer,
        paged._cache.flash_decode_block_table,
        prefix,
        args.page_size,
        "flash_decode_v_cache",
    )
    prefix_k_diff = float(
        (dense_layer.flash_decode_k_cache[0, :prefix] - paged_prefix_k)
        .abs()
        .max()
        .item()
    )
    prefix_v_diff = float(
        (dense_layer.flash_decode_v_cache[0, :prefix] - paged_prefix_v)
        .abs()
        .max()
        .item()
    )

    seen = set()
    rows = []
    for step in range(args.steps):
        dense_logits = _penalize(
            dense.constrained_logits(), seen, args.repetition_penalty
        )
        paged_logits = _penalize(
            batch.logits, seen, args.repetition_penalty
        )
        dense_top = torch.topk(dense_logits[0], k=2)
        paged_top = torch.topk(paged_logits[0], k=2)
        token = int(dense_top.indices[0].item())
        row = {
            "step": step,
            "forced_token": token,
            "dense_top": dense_top.indices.detach().cpu().tolist(),
            "paged_top": paged_top.indices.detach().cpu().tolist(),
            "dense_margin": float((dense_top.values[0] - dense_top.values[1]).item()),
            "paged_margin": float((paged_top.values[0] - paged_top.values[1]).item()),
            "logit_max_abs": float((dense_logits - paged_logits).abs().max().item()),
            "logit_mean_abs": float((dense_logits - paged_logits).abs().mean().item()),
        }
        dense.commit(
            torch.tensor([token], device="cuda", dtype=torch.long),
            torch.ones(1, device="cuda", dtype=torch.bool),
        )
        paged.advance(
            batch,
            torch.tensor([token], device="cuda", dtype=torch.long),
        )
        position = prefix + step
        block = int(
            paged._cache.flash_decode_block_table[
                0, position // args.page_size
            ].item()
        )
        offset = position % args.page_size
        row["appended_k_max_abs"] = float(
            (
                dense_layer.flash_decode_k_cache[0, position]
                - paged_layer.flash_decode_k_cache[block, offset]
            )
            .abs()
            .max()
            .item()
        )
        row["appended_v_max_abs"] = float(
            (
                dense_layer.flash_decode_v_cache[0, position]
                - paged_layer.flash_decode_v_cache[block, offset]
            )
            .abs()
            .max()
            .item()
        )
        rows.append(row)
        seen.add(token)
        batch = paged.schedule()
        assert batch is not None

    report = {
        "task_id": args.task_id,
        "prefix_length": prefix,
        "page_size": args.page_size,
        "prefix_k_max_abs": prefix_k_diff,
        "prefix_v_max_abs": prefix_v_diff,
        "steps": rows,
    }
    print("DENSE_PAGED_LOCKSTEP_RESULT=" + json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
