"""Decode and pack each rank's share of a fixed global sample batch."""
from bisect import bisect_right
from contextlib import ExitStack
import mmap
import json
from torch.utils.data import IterableDataset, get_worker_info
from sensenovalm.data.sample_batch import sample_batches, balance_rows
from sensenovavl.data.dataset_interleaved_iterable import PackedDataset


def estimate_sample_work(line):
    """Cheap metadata-only placement estimate; never used for truncation or loss."""
    item = json.loads(line)
    text = sum(len(turn.get("value", "")) for turn in item.get("conversations", []))
    images = item.get("image", item.get("images", []))
    image_count = len(images) if isinstance(images, list) else int(bool(images))
    return max(1, text // 4 + 512 * image_count, len(line) // 8)


class SampleBatchDataset(IterableDataset):
    def __init__(self, *, datasets, batch_samples, max_samples, seed, rank,
                 world_size, max_tokens, max_images, collate, start_samples=0):
        self.start_samples = start_samples
        self.datasets = datasets
        self.batch_samples, self.max_samples, self.seed = batch_samples, max_samples, seed
        self.rank, self.world_size = rank, world_size
        self.max_tokens, self.max_images, self.collate = max_tokens, max_images, collate
        self.ends = []
        total = 0
        for dataset in datasets:
            if not hasattr(dataset, '_annotation_offsets') or dataset.repeat_time != 1:
                raise ValueError('sample-batched SFT requires finite indexed annotation datasets')
            count = len(dataset._annotation_offsets) - 1
            if count != dataset.meta['length']:
                raise ValueError('SFT annotation row count differs from its sealed meta')
            total += count
            self.ends.append(total)
        self.rows = total
        if not self.rows:
            raise ValueError('sample-batched SFT cannot use an empty dataset')

    def _pack(self, samples):
        # Best-fit decreasing is confined to this optimizer batch. It neither
        # splits a sample nor takes a sample from the next batch.
        packer = object.__new__(PackedDataset)
        packer.max_packed_tokens = self.max_tokens
        packer.num_images_expected = self.max_images
        packer.allow_overflow = False
        buffers = []
        for sample in sorted(samples, key=lambda item: len(item['input_ids']), reverse=True):
            if len(sample['input_ids']) > self.max_tokens:
                raise ValueError('SFT sample exceeds the declared sequence length')
            if sample['pixel_values'] is not None and len(sample['pixel_values']) > self.max_images:
                raise ValueError('SFT sample exceeds the declared image slot limit')
            buffer = packer.find_buffer(buffers, sample)
            buffers.append(packer.update_buffer(buffer, sample))
        batches = []
        # Align long physical sequences across ranks; otherwise a different
        # rank can become the straggler on every accumulation microbatch.
        for buffer in sorted(buffers, key=lambda item: len(item['input_ids']), reverse=True):
            buffer['worker_state_key'] = ''
            buffer['worker_state_dict'] = b''
            # Pad to a small kernel-alignment boundary, not the full sequence
            # capacity. This matters for sample-limited and final microbatches.
            length = min(self.max_tokens, ((len(buffer['input_ids']) + 127) // 128) * 128)
            batches.append(self.collate([buffer], max_item_length=length))
        return batches

    def __iter__(self):
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        with ExitStack() as stack:
            maps = []
            for dataset in self.datasets:
                stream = stack.enter_context(open(dataset.annotation_file, 'rb'))
                maps.append(stack.enter_context(mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)))

            def locate(index):
                dataset_index = bisect_right(self.ends, index)
                start = self.ends[dataset_index - 1] if dataset_index else 0
                row = index - start
                dataset = self.datasets[dataset_index]
                offsets = dataset._annotation_offsets
                line = maps[dataset_index][offsets[row]:offsets[row + 1]]
                return dataset, row, line

            def estimate(index):
                return estimate_sample_work(locate(index)[2])

            def decode(index):
                dataset, row, line = locate(index)
                dataset._state_dict['line_shift'] = row + 1
                sample = dataset.get_sample(line)
                sample.pop('meta_info', None)
                return sample

            for batch in sample_batches(rows=self.rows, batch_samples=self.batch_samples,
                                        max_samples=self.max_samples, seed=self.seed,
                                        rank=self.rank, world_size=self.world_size,
                                        worker_id=worker_id, num_workers=workers, start_samples=self.start_samples):
                assignments = balance_rows(batch.global_rows, [estimate(index) for index in batch.global_rows],
                                           self.world_size)
                samples = [decode(index) for index in assignments[self.rank]]
                # A partial final batch can have no real sample on a rank. A
                # legal zero-loss example keeps the common FSDP hook sequence.
                padded_rank = not samples
                if padded_rank:
                    samples = [decode(0)]
                microbatches = self._pack(samples)
                if padded_rank:
                    for data, _labels in microbatches:
                        data['num_samples'] = 0
                        data['sample_ids'] = []
                        data['samples_per_microbatch'] = [0]
                yield dict(epoch=batch.epoch, sample_start=batch.sample_start,
                           sample_count=batch.sample_count, microbatches=microbatches)


def identity_collate(value):
    """DataLoader already receives a complete sample batch from each worker."""
    return value
