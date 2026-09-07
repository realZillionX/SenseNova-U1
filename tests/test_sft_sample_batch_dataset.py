import json
import tempfile
import unittest
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sensenovalm.data.annotation_order import annotation_offsets
from sensenovavl.data.sample_batch_dataset import SampleBatchDataset, identity_collate


class Reader:
    def __init__(self, path, factor):
        self.annotation_file = path
        self._annotation_offsets = annotation_offsets(path)
        self.meta = {'length': len(self._annotation_offsets) - 1}
        self.repeat_time, self._state_dict, self.factor = 1, {}, factor
    def get_sample(self, row):
        value = json.loads(row)
        length = (value['sample_id'] % 7 + 2) * self.factor
        return {'sample_ids': [str(value['sample_id'])], 'input_ids': torch.full((length,), 7),
                'pixel_values': None, 'labels': torch.ones(length, dtype=torch.long)}


def collate(rows, *, max_item_length):
    row, = rows
    count = len(row['sample_ids'])
    padding = max_item_length - len(row['input_ids'])
    data = {'input_ids': torch.nn.functional.pad(row['input_ids'], (0, padding)).unsqueeze(0),
            'sample_ids': row['sample_ids'], 'num_samples': count, 'samples_per_microbatch': [count],
            'num_padding_tokens': padding}
    return data, data['input_ids'].clone()


class SampleBatchDatasetTest(unittest.TestCase):
    def test_worker_order_and_two_modalities_have_identical_optimizer_batches(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rows.jsonl'
            path.write_text(''.join(json.dumps({'sample_id': i}) + '\n' for i in range(101)))
            baseline = None
            for factor in (2, 40):
                combined, physical = {}, 0
                for rank in range(4):
                    dataset = SampleBatchDataset(datasets=[Reader(path, factor)], batch_samples=16,
                                                 max_samples=101, seed=42, rank=rank, world_size=4,
                                                 max_tokens=512, max_images=144, collate=collate)
                    loader = DataLoader(dataset, batch_size=None, num_workers=3, collate_fn=identity_collate)
                    starts = []
                    for batch in loader:
                        starts.append(batch['sample_start'])
                        ids = [i for data, _labels in batch['microbatches'] for i in data['sample_ids']]
                        combined.setdefault(batch['sample_start'], []).extend(ids)
                        physical += len(batch['microbatches'])
                        self.assertTrue(all(data['input_ids'].shape[1] <= 512 for data, _ in batch['microbatches']))
                    self.assertEqual(starts, list(range(0, 101, 16)))
                membership = {k: sorted(v, key=int) for k, v in combined.items()}
                self.assertEqual([len(v) for v in membership.values()], [16] * 6 + [5])
                if baseline is None:
                    baseline, first_physical = membership, physical
                else:
                    self.assertEqual(membership, baseline)
                    self.assertGreater(physical, first_physical)

    def test_empty_rank_in_last_batch_is_padding_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rows.jsonl'
            path.write_text(''.join(json.dumps({'sample_id': i}) + '\n' for i in range(17)))
            dataset = SampleBatchDataset(datasets=[Reader(path, 2)], batch_samples=16,
                                         max_samples=17, seed=42, rank=3, world_size=4,
                                         max_tokens=512, max_images=144, collate=collate)
            batches = list(dataset)
            self.assertEqual(batches[-1]['sample_count'], 1)
            self.assertEqual(sum(data['num_samples'] for data, _ in batches[-1]['microbatches']), 0)
            self.assertEqual(batches[-1]['microbatches'][0][0]['sample_ids'], [])


if __name__ == '__main__':
    unittest.main()
