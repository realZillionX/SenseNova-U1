import random
import unittest

from sensenova_u1.rl.sampling import prompt_batch_indices


class PromptSamplingTest(unittest.TestCase):
    def epoch(self, epoch, *, count=24, batch=8, seed=42):
        return [index for step in range(epoch * (count // batch), (epoch + 1) * (count // batch))
                for index in prompt_batch_indices(step, count=count, prompts_per_batch=batch, seed=seed)]

    def test_full_pool_is_shuffled_without_replacement_each_epoch(self):
        first, second = self.epoch(0), self.epoch(1)
        self.assertEqual(sorted(first), list(range(24)))
        self.assertEqual(sorted(second), list(range(24)))
        self.assertNotEqual(first, list(range(24)))
        self.assertNotEqual(first, second)
        # Canonical blocks represent different stages; the first batch mixes them.
        self.assertGreater(len({index // 8 for index in first[:8]}), 1)

    def test_arms_and_ranks_do_not_depend_on_global_rng_or_call_order(self):
        first = self.epoch(3)
        random.seed(123)
        for _ in range(100):
            random.random()
        self.epoch(7)
        self.epoch(8)
        self.assertEqual(first, self.epoch(3))
        self.assertNotEqual(first, self.epoch(3, seed=99))

    def test_tail_is_not_wrapped_into_duplicate_groups(self):
        for epoch in range(3):
            rows = self.epoch(epoch, count=19)
            self.assertEqual(len(rows), 16)
            self.assertEqual(len(set(rows)), 16)
            self.assertTrue(all(0 <= index < 19 for index in rows))
        with self.assertRaises(ValueError):
            prompt_batch_indices(0, count=7, prompts_per_batch=8, seed=42)


if __name__ == "__main__":
    unittest.main()
