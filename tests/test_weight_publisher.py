from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

import torch

from sensenova_u1.rl.weight_publisher import ServingWeightPublisher


class WeightPublisherTest(unittest.TestCase):
    @staticmethod
    def _publisher() -> ServingWeightPublisher:
        return ServingWeightPublisher(
            base_urls=("http://127.0.0.1:8000", "http://127.0.0.1:8001"),
            master_address="127.0.0.1",
            base_port=29680,
            backend="nccl",
            bucket_bytes=1024,
            default_dtype=torch.bfloat16,
            device=torch.device("cpu"),
            group_name="test-group",
        )

    def test_initialize_joins_all_replicas_to_one_group_per_owner(self) -> None:
        publisher = self._publisher()
        payloads = []
        group_world_sizes = []

        def request(_method, _url, *, payload=None, **_kwargs):
            payloads.append(dict(payload))
            return {
                "receipts": {
                    "language": {"ranks": [{"closure_names": ["language"], "closure_specs": {}}]},
                    "vision": {"ranks": [{"closure_names": ["vision"], "closure_specs": {}}]},
                    "x2v": {"closure_names": ["x2v"], "closure_specs": {}},
                }
            }

        def init_group(**kwargs):
            group_world_sizes.append(kwargs["world_size"])
            return object()

        with (
            patch("sensenova_u1.rl.weight_publisher._request", side_effect=request),
            patch("sensenova_u1.rl.weight_publisher._init_custom_process_group", side_effect=init_group),
        ):
            closures = publisher.prepare()

        self.assertEqual({payload["replica_index"] for payload in payloads}, {0, 1})
        self.assertTrue(all(payload["replica_count"] == 2 for payload in payloads))
        self.assertTrue(all(payload["world_size"] == 7 for payload in payloads))
        self.assertEqual(group_world_sizes, [3, 3, 3])
        self.assertEqual(set(closures), {"language", "vision", "x2v"})

    def test_close_destroys_both_ends_concurrently(self) -> None:
        publisher = self._publisher()
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

    @staticmethod
    def _transfer_receipt(version: str) -> dict:
        return {
            "policy_version": version,
            "committed": False,
            "receipts": {
                "language": {"ranks": [{"received_names": []}]},
                "vision": {"ranks": [{"received_names": []}]},
                "x2v": {"received_names": []},
            },
        }

    def test_commit_uses_replica_set_pending_barrier(self) -> None:
        publisher = self._publisher()
        requests = []

        def request(method, url, *, payload=None, **_kwargs):
            requests.append((method, url, payload))
            if url.endswith("/commit_weights_update"):
                return {"policy_version": "v2", "committed": True}
            if url.endswith("/v1/rl/status"):
                return {
                    "active_policy_version": "v2",
                    "pending_policy_version": None,
                    "paused": False,
                }
            raise AssertionError(url)

        with patch("sensenova_u1.rl.weight_publisher._request", side_effect=request):
            result = publisher.commit(
                [self._transfer_receipt("v2"), self._transfer_receipt("v2")],
                entries=[],
                policy_version="v2",
            )

        self.assertEqual(result["policy_version"], "v2")
        self.assertEqual(
            sum(url.endswith("/commit_weights_update") for _method, url, _payload in requests),
            2,
        )

    def test_partial_commit_repauses_every_replica(self) -> None:
        publisher = self._publisher()
        paused = []

        def request(_method, url, *, payload=None, **_kwargs):
            if url.endswith("/commit_weights_update"):
                if ":8001/" in url:
                    raise RuntimeError("injected commit failure")
                return {"policy_version": "v2", "committed": True}
            if url.endswith("/pause_generation"):
                paused.append(url)
                return {}
            raise AssertionError(url)

        with (
            patch("sensenova_u1.rl.weight_publisher._request", side_effect=request),
            self.assertRaisesRegex(RuntimeError, "injected commit failure"),
        ):
            publisher.commit(
                [self._transfer_receipt("v2"), self._transfer_receipt("v2")],
                entries=[],
                policy_version="v2",
            )

        self.assertEqual(len(paused), 2)


if __name__ == "__main__":
    unittest.main()
