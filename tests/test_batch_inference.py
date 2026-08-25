from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sensenova_u1.batch_inference import (
    ContiguousTextBatchSession,
    ContinuousTextBatchEngine,
    NativeTextBatchSession,
    TextBatchRequest,
    _apply_repetition_penalty,
    _chunk_block_causal_mask,
    _chunk_end_without_splitting_block,
    _left_pad_prefills,
    batch_text_gen,
)
from sensenova_u1.models.neo_unify.modeling_qwen3 import (
    batch_axis_indexes,
    create_block_causal_mask,
)


class BatchPrefillTest(unittest.TestCase):
    def test_native_session_name_remains_a_compatibility_alias(self) -> None:
        self.assertIs(NativeTextBatchSession, ContiguousTextBatchSession)

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

    def test_prefill_chunk_does_not_split_bidirectional_image_block(self) -> None:
        temporal = torch.tensor([0, 1, 2, 2, 2, 3, 4], dtype=torch.long)

        end = _chunk_end_without_splitting_block(temporal, 0, 3)

        self.assertEqual(end, 5)

    def test_chunk_mask_matches_official_full_mask_rows(self) -> None:
        temporal = torch.tensor([0, 1, 2, 2, 2, 3, 4], dtype=torch.long)
        full = create_block_causal_mask(temporal)

        chunk = _chunk_block_causal_mask(temporal, 2, 5)

        self.assertTrue(torch.equal(chunk, full[:, :, 2:5, :5]))

    def test_repetition_penalty_only_changes_seen_generated_tokens(self) -> None:
        logits = torch.tensor([[4.0, -3.0, 2.0], [-6.0, 5.0, 1.0]])
        seen = torch.tensor([[True, True, False], [False, True, False]])

        penalized = _apply_repetition_penalty(logits, seen, 2.0)

        expected = torch.tensor([[2.0, -6.0, 2.0], [-6.0, 2.5, 1.0]])
        self.assertTrue(torch.equal(penalized, expected))

    def test_constrained_logits_does_not_mutate_float32_source(self) -> None:
        session = ContiguousTextBatchSession.__new__(ContiguousTextBatchSession)
        session.batch_size = 1
        session.allow_image_actions = False
        session.image_start_token_id = 1
        session.next_logits = torch.tensor([[2.0, 7.0, 3.0]])

        constrained = session.constrained_logits()

        self.assertEqual(session.next_logits.tolist(), [[2.0, 7.0, 3.0]])
        self.assertLess(constrained[0, 1].item(), -1e30)

    def test_continuous_prefill_can_clamp_generation_to_context_limit(self) -> None:
        engine = ContinuousTextBatchEngine.__new__(ContinuousTextBatchEngine)
        engine.model = object()
        engine.tokenizer = object()
        engine.device = torch.device("cpu")
        engine.dtype = torch.float32
        engine.image_start_token = "<img>"
        engine.image_context_token = "<IMG_CONTEXT>"
        engine.image_end_token = "</img>"
        engine.max_model_len = 128
        engine.truncate_to_max_model_len = True
        state = SimpleNamespace(
            request=TextBatchRequest(prompt="question"),
            max_new_tokens=64,
            inputs_embeds=None,
            indexes=None,
        )
        prepared = (
            torch.zeros(1, 96, 4),
            torch.zeros(3, 96, dtype=torch.long),
        )

        with patch(
            "sensenova_u1.batch_inference._prepare_text_request",
            return_value=prepared,
        ):
            engine._prepare_prefill(state)

        self.assertEqual(state.max_new_tokens, 32)
        self.assertIs(state.inputs_embeds, prepared[0])


class _FakeTokenizer:
    _tokens = {"<eos>": 0, "<img>": 1}
    _text = {10: "A", 20: "C", 21: "D"}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._tokens[token]

    def decode(self, tokens, *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(self._text[int(token)] for token in tokens)


class _FakeContiguousSession:
    scripts = ((10, 0), (20, 21))
    latest = None

    def __init__(self, model, tokenizer, requests, **kwargs) -> None:
        del model, tokenizer
        self.requests = tuple(requests)
        self.kwargs = kwargs
        self.device = torch.device(kwargs["device"])
        self.positions = [0] * len(self.requests)
        self.next_logits = torch.zeros(len(self.requests), 32)
        self.commits: list[list[bool]] = []
        type(self).latest = self

    def constrained_logits(self) -> torch.Tensor:
        logits = torch.full((len(self.requests), 32), -1000.0)
        for row, script in enumerate(self.scripts):
            logits[row, script[self.positions[row]]] = 1.0
        return logits

    def commit(self, token_ids: torch.Tensor, accepted: torch.Tensor) -> None:
        del token_ids
        flags = accepted.tolist()
        self.commits.append(flags)
        for row, is_active in enumerate(flags):
            if is_active:
                self.positions[row] += 1


class BatchTextGenerationTest(unittest.TestCase):
    def test_rows_stop_independently_without_final_wasted_decode(self) -> None:
        model = SimpleNamespace(device=torch.device("cpu"), template="fake")
        requests = (TextBatchRequest(prompt="first"), TextBatchRequest(prompt="second"))
        with (
            patch(
                "sensenova_u1.batch_inference.NativeTextBatchSession",
                _FakeContiguousSession,
            ),
            patch(
                "sensenova_u1.batch_inference.get_conv_template",
                return_value=SimpleNamespace(sep="<eos>"),
            ),
        ):
            results = batch_text_gen(
                model,
                _FakeTokenizer(),
                requests,
                generation_config=SimpleNamespace(
                    max_new_tokens=2, repetition_penalty=1.0
                ),
                dtype=torch.float32,
            )

        self.assertEqual([result.text for result in results], ["A", "CD"])
        self.assertEqual(
            [result.finish_reason for result in results], ["eos", "max_new_tokens"]
        )
        self.assertEqual(_FakeContiguousSession.latest.commits, [[True, True]])


if __name__ == "__main__":
    unittest.main()
