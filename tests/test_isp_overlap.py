from __future__ import annotations

import unittest

from sensenovalm.core.parallel.comm.isp import ISPCommunicator, ISPOverlapState


class _Handle:
    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> None:
        self.waited = True


class ISPOverlapTest(unittest.TestCase):
    def test_drain_prefetch_state_waits_and_clears_every_chunk(self) -> None:
        communicator = object.__new__(ISPCommunicator)
        first = ISPOverlapState()
        second = ISPOverlapState()
        handles = [_Handle(), _Handle(), _Handle()]
        first.weight_global_handle["weight"] = handles[0]
        first.bias_global_handle["bias"] = handles[1]
        first.weight_global_output["weight"] = object()
        first.bias_global_output["bias"] = object()
        second.weight_global_handle["weight"] = handles[2]
        second.weight_global_output["weight"] = object()
        communicator._overlap_states = {
            0: {"llm": first, "vision": second},
        }

        communicator.drain_prefetch_state()

        self.assertTrue(all(handle.waited for handle in handles))
        for state in (first, second):
            self.assertEqual(state.weight_global_handle, {})
            self.assertEqual(state.bias_global_handle, {})
            self.assertEqual(state.weight_global_output, {})
            self.assertEqual(state.bias_global_output, {})


if __name__ == "__main__":
    unittest.main()
