import unittest

from train_sensenovau1_fsdp2 import _aggregate


class SftTimingTest(unittest.TestCase):
    def test_p95_uses_nearest_rank_even_for_short_probes(self):
        records = [
            dict(seconds=seconds, physical_tokens=100, supervised_tokens=50, samples=2) for seconds in (1.0, 2.0, 10.0)
        ]
        result = _aggregate(records)
        self.assertEqual(result["update_seconds_p95"], 10.0)
        self.assertAlmostEqual(result["samples_per_second"], 6 / 13)

    def test_empty_measured_window_has_an_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "no measured updates"):
            _aggregate([])


if __name__ == "__main__":
    unittest.main()
