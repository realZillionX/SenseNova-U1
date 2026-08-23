from __future__ import annotations

import io
import time
import unittest
from contextlib import redirect_stdout

from sensenova_u1.utils.profiler import InferenceProfiler


class InferenceProfilerTest(unittest.TestCase):
    def test_text_only_generation_reports_without_dividing_by_zero(self) -> None:
        profiler = InferenceProfiler(enabled=True, device="cpu")
        with profiler.time_generate(width=256, height=256):
            time.sleep(0.001)
        profiler.update_last_batch(0)
        profiler.update_last_text_tokens(3)

        output = io.StringIO()
        with redirect_stdout(output):
            profiler.report()

        report = output.getvalue()
        self.assertIn("0 image(s), 3 text token(s)", report)
        self.assertIn("image tokens        : 0 (text-only generation)", report)
        self.assertIn("text throughput", report)

    def test_mixed_generation_reports_both_token_domains(self) -> None:
        profiler = InferenceProfiler(enabled=True, device="cpu")
        with profiler.time_generate(width=64, height=64, batch=2):
            time.sleep(0.001)
        profiler.update_last_text_tokens(4)

        output = io.StringIO()
        with redirect_stdout(output):
            profiler.report()

        report = output.getvalue()
        self.assertIn("2 image(s), 4 text token(s)", report)
        self.assertIn("image throughput", report)
        self.assertIn("text throughput", report)

    def test_negative_text_token_count_is_rejected(self) -> None:
        profiler = InferenceProfiler(enabled=True, device="cpu")
        with profiler.time_generate(width=64, height=64):
            pass
        with self.assertRaisesRegex(ValueError, "non-negative"):
            profiler.update_last_text_tokens(-1)


if __name__ == "__main__":
    unittest.main()
