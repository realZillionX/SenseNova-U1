from __future__ import annotations

import unittest

import torch

from sensenova_u1.batch_inference import _left_pad_prefills
from sensenova_u1.models.neo_unify.modeling_qwen3 import batch_axis_indexes


class BatchPrefillTest(unittest.TestCase):
    def test_left_padding_preserves_each_real_attention_block(self) -> None:
        first = torch.tensor([[[1.0], [2.0]]])
        second = torch.tensor([[[3.0], [4.0], [5.0], [6.0]]])
        first_indexes = torch.tensor([[0, 1], [0, 0], [0, 0]])
        second_indexes = torch.tensor(
            [[0, 1, 1, 2], [0, 0, 1, 0], [0, 0, 1, 0]]
        )
        first_mask = torch.tensor(
            [[[[0.0, -torch.inf], [0.0, 0.0]]]]
        )
        second_mask = torch.tensor(
            [
                [
                    [
                        [0.0, -torch.inf, -torch.inf, -torch.inf],
                        [0.0, 0.0, 0.0, -torch.inf],
                        [0.0, 0.0, 0.0, -torch.inf],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )

        embeds, indexes, mask, valid = _left_pad_prefills(
            [first, second],
            [first_indexes, second_indexes],
            [first_mask, second_mask],
        )

        self.assertEqual(tuple(embeds.shape), (2, 4, 1))
        self.assertEqual(embeds[0, :, 0].tolist(), [0.0, 0.0, 1.0, 2.0])
        self.assertTrue(torch.equal(indexes[0, :, 2:], first_indexes))
        self.assertEqual(valid.tolist(), [[False, False, True, True], [True] * 4])
        self.assertTrue(torch.equal(mask[0, 0, 2:, 2:], first_mask[0, 0]))
        self.assertTrue(torch.equal(mask[1, 0], second_mask[0, 0]))
        self.assertEqual(mask[0, 0, 0, 0].item(), 0.0)
        self.assertEqual(mask[0, 0, 1, 1].item(), 0.0)
        self.assertTrue(torch.isneginf(mask[0, 0, 2, :2]).all())

    def test_axis_helper_accepts_shared_and_per_row_indexes(self) -> None:
        shared = torch.arange(12, dtype=torch.long).reshape(3, 4)
        batched = torch.stack((shared, shared + 20))

        self.assertTrue(torch.equal(batch_axis_indexes(shared, 1), shared[1:2]))
        self.assertTrue(torch.equal(batch_axis_indexes(batched, 2), batched[:, 2]))


if __name__ == "__main__":
    unittest.main()
