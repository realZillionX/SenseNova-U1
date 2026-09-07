from __future__ import annotations

import unittest

import torch
from sensenovavl.data.dataset_interleaved_iterable import u15_packed_collate_fn
from sensenovalm.model.losses.ce_loss import FlashGPTLMLoss
from sensenovalm.model.losses.sample_mean import sample_mean_sum


class SampleMeanTest(unittest.TestCase):
    def test_collator_excludes_padding_copies_from_sample_count_and_weights(self):
        feature = {
            "already_packed": True,
            "sample_ids": ["first", "second"],
            "input_ids": torch.tensor([3, 4, 5, 3, 6, 7, 8, 9]),
            "labels": torch.tensor([-100, 4, 5, -100, 6, 7, 8, 9]),
            "type_ids": torch.zeros(8, dtype=torch.long),
            "cu_seqlens": [0, 3, 8], "indexes": [0, 1, 2, 0, 1, 2, 3, 4],
            "worker_state_key": "worker0", "worker_state_dict": {},
        }
        data, labels = u15_packed_collate_fn(
            [feature], max_item_length=16, img_start_token_id=100,
            img_token_id=101, img_end_token_id=102, ignored_token_ids=[],
            micro_num=2, len2weight=lambda count: 1 / count if count else 0,
        )
        self.assertEqual(data["num_samples"], 2)
        self.assertEqual(data["sample_ids"], ["first", "second"])
        self.assertEqual(data["samples_per_microbatch"], [2, 0])
        self.assertEqual(sum(data["loss_weight"][0]), 2)
        self.assertEqual(sum(data["loss_weight"][1]), 0)
        self.assertTrue((labels[1] == -100).all())

    def test_text_gradients_ignore_lengths_and_rank_microbatch_partition(self):
        # The same three examples are represented with different token counts
        # and packed into uneven ranks / accumulation microbatches.
        criterion = FlashGPTLMLoss(parallel_output=False, ce_loss_weight=0.1)
        gradients = []
        for lengths, partitions in (
            ([2, 8, 3], [[[0, 1]], [[2]]]),
            ([4, 16, 6], [[[0], [1]], [[2], []]]),
        ):
            logits = torch.tensor([[0.2, 0.7], [0.5, -0.3], [-0.8, 0.1]], requires_grad=True)
            rank_losses = []
            for rank in partitions:
                rank_loss = logits.sum() * 0
                for batch in rank:
                    if not batch:
                        # Padding copies cannot contribute a sample.
                        continue
                    packed = torch.cat([logits[i].expand(lengths[i], -1) for i in batch])
                    labels = torch.cat([torch.full((lengths[i],), i % 2) for i in batch])
                    weights = [1 / lengths[i] for i in batch for _ in range(lengths[i])]
                    rank_loss = rank_loss + criterion(
                        packed, labels, loss_weight=weights,
                        sample_denominator=3 / len(partitions),
                    )
                rank_losses.append(rank_loss)
            result = torch.stack(rank_losses).mean()
            expected = 0.1 * torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0]))
            torch.testing.assert_close(result, expected)
            result.backward()
            gradients.append(logits.grad)
        torch.testing.assert_close(gradients[0], gradients[1])

    def test_images_and_empty_image_samples_share_global_sample_denominator(self):
        gradients = []
        for lengths in ([256, 768], [1024, 3072]):
            values = torch.tensor([2.0, 5.0], requires_grad=True)
            # Sample 0 has one image; sample 1 has three images. Sample 2 has
            # no image and remains in the denominator, with zero image loss.
            losses = torch.cat([values[i].expand(lengths[i]) for i in range(2)])
            ids = torch.cat([torch.full((lengths[i],), i) for i in range(2)])
            result = sample_mean_sum(losses, ids, 3)
            torch.testing.assert_close(result, torch.tensor(7 / 3))
            result.backward()
            gradients.append(values.grad)
        torch.testing.assert_close(gradients[0], torch.full((2,), 1 / 3))
        torch.testing.assert_close(gradients[0], gradients[1])

    def test_empty_visual_branch_and_masked_text_remain_differentiable_zero(self):
        dummy = torch.ones(8, requires_grad=True)
        result = sample_mean_sum(dummy[:0], torch.empty(0, dtype=torch.long), 2)
        result.backward()
        torch.testing.assert_close(dummy.grad, torch.zeros_like(dummy))
        logits = torch.randn(4, 3, requires_grad=True)
        criterion = FlashGPTLMLoss(parallel_output=False)
        loss = criterion(logits, torch.full((4,), -100), loss_weight=[0] * 4, sample_denominator=2)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0)
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


if __name__ == "__main__":
    unittest.main()
