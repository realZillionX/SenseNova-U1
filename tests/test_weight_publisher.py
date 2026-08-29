from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

import torch

from sensenova_u1.rl.weight_publisher import ServingWeightPublisher


class WeightPublisherTest(unittest.TestCase):
    def test_close_destroys_both_ends_concurrently(self) -> None:
        publisher = ServingWeightPublisher(
            base_url="http://127.0.0.1:8000",
            master_address="127.0.0.1",
            base_port=29680,
            backend="nccl",
            bucket_bytes=1024,
            default_dtype=torch.bfloat16,
            device=torch.device("cpu"),
            group_name="test-group",
        )
        publisher.groups = {name: object() for name in ("language", "vision", "x2v")}
        local_destroyed = threading.Event()
        destroy_count = 0

        def request(*args, **kwargs):
            self.assertTrue(local_destroyed.wait(timeout=2), "HTTP teardown serialized before local destroy")
            return {"receipts": {}}

        def destroy(group):
            nonlocal destroy_count
            destroy_count += 1
            if destroy_count == len(publisher.groups):
                local_destroyed.set()

        with (
            patch("sensenova_u1.rl.weight_publisher._request", side_effect=request),
            patch("sensenova_u1.rl.weight_publisher.dist.destroy_process_group", side_effect=destroy),
        ):
            publisher.close()

        self.assertEqual(destroy_count, 3)
        self.assertEqual(publisher.groups, {})


if __name__ == "__main__":
    unittest.main()
