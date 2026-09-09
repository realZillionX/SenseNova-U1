import unittest

import numpy as np
from sensenovalm.data.sample_batch import balance_rows, optimizer_updates, sample_batches


class SampleBatchesTest(unittest.TestCase):
    def collect(self, rows, budget, batch, world, workers):
        by_start = {}
        for rank in range(world):
            for worker in range(workers):
                for item in sample_batches(
                    rows=rows,
                    max_samples=budget,
                    batch_samples=batch,
                    seed=42,
                    rank=rank,
                    world_size=world,
                    worker_id=worker,
                    num_workers=workers,
                ):
                    self.assertNotIn(rank, by_start.setdefault(item.sample_start, {}))
                    by_start[item.sample_start][rank] = item
        result = []
        for start, ranks in sorted(by_start.items()):
            count = ranks[0].sample_count
            self.assertEqual(len(ranks), world)
            ordered = [ranks[i % world].rows[i // world] for i in range(count)]
            result.append((start, ranks[0].epoch, count, ordered))
        return result

    def test_global_membership_does_not_depend_on_rank_or_worker_counts(self):
        baseline = self.collect(101, 201, 32, 1, 1)
        for world, workers in ((4, 2), (8, 3), (32, 7)):
            self.assertEqual(self.collect(101, 201, 32, world, workers), baseline)
        self.assertEqual(sum(item[2] for item in baseline), 201)
        self.assertEqual(len(baseline), optimizer_updates(rows=101, batch_samples=32, max_samples=201))
        for start, epoch, count, rows in baseline:
            expected = np.random.default_rng(np.random.SeedSequence([42, epoch])).permutation(101)
            offset = start - epoch * 101
            self.assertEqual(rows, expected[offset : offset + count].tolist())

    def test_sample_budget_is_exact_even_for_a_partial_first_batch(self):
        batches = self.collect(101, 5, 32, 8, 3)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][2], 5)
        self.assertEqual(len(set(batches[0][3])), 5)

    def test_full_epochs_and_final_remainder_have_declared_update_counts(self):
        for rows, budget, batch in ((3792, 3792, 128), (101, 202, 16), (101, 301, 32)):
            actual = self.collect(rows, budget, batch, 8, 2)
            self.assertEqual(len(actual), optimizer_updates(rows=rows, batch_samples=batch, max_samples=budget))
            self.assertEqual(sum(item[2] for item in actual), budget)
            self.assertTrue(all(0 < item[2] <= batch for item in actual))
        self.assertEqual(optimizer_updates(rows=3792, batch_samples=128, max_samples=3792), 30)

    def test_cost_balancing_keeps_the_batch_and_reduces_uneven_rank_work(self):
        rows = list(range(8))
        costs = [9, 1, 8, 1, 7, 1, 6, 1]
        assignments = balance_rows(rows, costs, 2)
        self.assertEqual(sorted(i for rank in assignments for i in rank), rows)
        loads = [sum(costs[i] for i in rank) for rank in assignments]
        old_loads = [sum(costs[rank::2]) for rank in range(2)]
        self.assertLess(max(loads), max(old_loads))
        self.assertEqual(sum(loads), sum(costs))


if __name__ == "__main__":
    unittest.main()
