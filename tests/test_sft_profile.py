import os
import unittest
from unittest.mock import patch

import torch
from sensenovalm.data.sample_progress import SampleProgress
from sensenovalm.train.fsdp import sample_profile_schedule


class SampleProfileTest(unittest.TestCase):
    def test_profile_actions_follow_samples_not_callback_count(self):
        progress = SampleProgress(1000, 512)
        with patch.dict(os.environ, {}, clear=True):
            schedule = sample_profile_schedule(progress, 64)
        action = torch.profiler.ProfilerAction
        for position, expected in (
            (0, action.NONE),
            (64, action.NONE),
            (128, action.WARMUP),
            (192, action.RECORD_AND_SAVE),
            (256, action.NONE),
        ):
            progress.consumed_samples = position
            self.assertEqual(schedule(999), expected)

    def test_legacy_step_windows_are_rejected(self):
        with patch.dict(os.environ, {"SFT_PROFILE_ACTIVE": "1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "sample counts"):
                sample_profile_schedule(SampleProgress(1000, 512), 64)


if __name__ == "__main__":
    unittest.main()
