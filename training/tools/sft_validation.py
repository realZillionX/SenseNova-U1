"""Paired, fixed-noise authored-sample validation for the FSDP2 SFT runner."""
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist


@contextmanager
def validation_state(model, devices):
    """Validation must not advance training RNGs or leave dropout disabled."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    training = model.training
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            yield
    finally:
        model.train(training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def aggregate_samples(records, expected_ids, seeds):
    expected = set(expected_ids)
    if len(expected) != len(expected_ids) or not expected:
        raise ValueError("validation requires distinct sample IDs")
    by_seed = {seed: {} for seed in seeds}
    for row in records:
        bucket = by_seed[row['noise_seed']]
        identity = row['sample_id']
        if identity in bucket:
            raise ValueError("duplicate validation sample")
        if not all(math.isfinite(row[key]) for key in ('text_loss', 'image_loss')):
            raise ValueError("non-finite validation loss")
        bucket[identity] = row
    if any(set(bucket) != expected for bucket in by_seed.values()):
        raise ValueError("validation coverage differs from the fixed sample set")
    examples = []
    for identity in expected_ids:
        item = {'sample_id': identity}
        for key in ('text_loss', 'image_loss'):
            item[key] = sum(bucket[identity][key] for bucket in by_seed.values()) / len(seeds)
        item['loss'] = item['text_loss'] + item['image_loss']
        examples.append(item)
    means = {key: sum(row[key] for row in examples) / len(examples)
             for key in ('text_loss', 'image_loss', 'loss')}
    return dict(samples=len(examples), noise_seeds=list(seeds), means=means, examples=examples)


class SampleValidation:
    """Evaluate one raw sample per rank, with the same noise at every checkpoint.

    The callback uses SFT's native per-sample CE and image objectives. It is an
    optimization diagnostic, not a generation-quality or verifier score.
    """
    def __init__(self, meta, sample_ids, output, *, noise_seeds=(901, 902)):
        self.meta = Path(meta)
        self.sample_ids = tuple(sample_ids)
        self.output = Path(output)
        self.noise_seeds = tuple(noise_seeds)
        self.batches = None

    def __call__(self, model, criterion, progress, training_seconds):
        import train_sensenovau1_fsdp2 as trainer
        from sensenovavl.data.build_dataloader import get_multimodal_streaming_train_loader_items

        rank, world = dist.get_rank(), dist.get_world_size()
        started = time.perf_counter()
        with validation_state(model, [torch.cuda.current_device()]):
            if self.batches is None:
                config = copy.deepcopy(trainer.gpc.config.data)
                for name, value in dict(meta_path=str(self.meta), batch_samples=world,
                                        max_samples=len(self.sample_ids), samples_per_epoch=len(self.sample_ids),
                                        seed=417, data_augment=False).items():
                    config._add_item(name, value)
                dataset, _, _, _ = get_multimodal_streaming_train_loader_items(config)
                self.batches = list(dataset)
            collected = []
            trainer.gpc.config.batch_count = progress.optimizer_updates
            for noise_seed in self.noise_seeds:
                for batch in self.batches:
                    raw_batches = batch['microbatches']
                    if len(raw_batches) != 1:
                        raise ValueError('one-sample validation unexpectedly required packing')
                    data, labels = trainer.move_to_device(raw_batches[0])
                    identities = data.pop('sample_ids')
                    count = int(data.pop('num_samples'))
                    if count not in (0, 1) or len(identities) != count:
                        raise ValueError('validation must assign at most one real sample per rank')
                    for name in ('samples_per_microbatch', 'worker_state_key_list',
                                 'worker_state_dict_list', 'worker_state_custom_infos_list',
                                 'num_padding_tokens', 'is_empty_data_list'):
                        data.pop(name, None)
                    micro_data, micro_labels = trainer._prepare_microbatch((data, labels), 0)
                    denominator = batch['sample_count'] / world
                    micro_data['sample_loss_denominator'] = denominator
                    seed = noise_seed * 100000 + batch['sample_start'] + rank
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.default_generator.manual_seed(seed)
                    torch.cuda.default_generators[torch.cuda.current_device()].manual_seed(seed)
                    output, _mtp, *extras = model(**micro_data)
                    text_loss = criterion(output, micro_labels,
                                          loss_weight=micro_data.pop('loss_weight', None),
                                          sample_denominator=denominator)
                    image_loss = trainer._numeric_extra_losses(tuple(extras))
                    # Undo the distributed update denominator to retain one
                    # independently comparable loss value for each raw sample.
                    local = None if not count else dict(
                        sample_id=identities[0], noise_seed=noise_seed,
                        text_loss=float(text_loss) * denominator,
                        image_loss=float(image_loss) * denominator,
                    )
                    gathered = [None] * world
                    dist.all_gather_object(gathered, local)
                    if rank == 0:
                        collected.extend(row for row in gathered if row is not None)
                    del output, _mtp, extras, text_loss, image_loss, micro_data, micro_labels, data, labels
            torch.cuda.synchronize()
        seconds = trainer._distributed_max(time.perf_counter() - started)
        if rank == 0:
            result = aggregate_samples(collected, self.sample_ids, self.noise_seeds)
            result.update(consumed_samples=progress.consumed_samples,
                          optimizer_updates=progress.optimizer_updates,
                          training_seconds=training_seconds, validation_seconds=seconds)
            with self.output.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(result, sort_keys=True) + '\n')
            print(json.dumps(dict(event='validation_complete', consumed_samples=progress.consumed_samples,
                                  training_seconds=training_seconds, validation_seconds=seconds,
                                  **result['means']), sort_keys=True), flush=True)
        dist.barrier()
