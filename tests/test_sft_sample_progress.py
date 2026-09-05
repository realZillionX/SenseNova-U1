from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import unittest

path = Path(__file__).parents[1] / "training/sensenovalm/data/sample_progress.py"
spec = importlib.util.spec_from_file_location("forge_sft_sample_progress", path)
progress_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = progress_module
spec.loader.exec_module(progress_module)
SampleProgress = progress_module.SampleProgress


class SampleProgressTest(unittest.TestCase):
    def test_packing_counts_and_resume_preserve_five_targets_per_epoch(self):
        progress = SampleProgress(103, 206)
        targets = []
        for batch_samples in [3, 7, 2, 6, 9] * 10:
            target = progress.advance(batch_samples)
            if target:
                targets.append(target)
                self.assertLess(progress.consumed_samples - target, batch_samples)
                progress = SampleProgress(**asdict(progress))
            if progress.done:
                break
        self.assertEqual(targets, [21, 42, 62, 83, 103, 124, 145, 165, 186, 206])

    def test_large_batch_cannot_fabricate_checkpoint_points(self):
        progress = SampleProgress(100, 100)
        with self.assertRaisesRegex(ValueError, "multiple sample checkpoints"):
            progress.advance(50)
        self.assertEqual(progress.consumed_samples, 0)

    def test_formal_dataset_and_partial_final_epoch(self):
        self.assertEqual(SampleProgress(1185000, 1185000).targets, (237000, 474000, 711000, 948000, 1185000))
        self.assertEqual(SampleProgress(100, 60).targets, (20, 40, 60))
        self.assertEqual(SampleProgress(100, 65).targets, (20, 40, 60, 65))


if __name__ == "__main__":
    unittest.main()
