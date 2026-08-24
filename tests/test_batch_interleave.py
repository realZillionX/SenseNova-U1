from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import torch

from sensenova_u1.batch_inference import (
    InterleaveBatchRequest,
    NativeTextBatchSession,
    TextBatchRequest,
    _repeat_cache_batch,
    _select_cache_batch,
    batch_interleave_gen,
)


class _Layer:
    def __init__(self, batch: int = 1) -> None:
        self.keys = torch.arange(batch * 6, dtype=torch.float32).reshape(
            batch, 1, 3, 2
        )
        self.values = self.keys + 100


class _Cache:
    def __init__(self, batch: int = 1) -> None:
        self.layers = [_Layer(batch), _Layer(batch)]


class CacheBatchTest(unittest.TestCase):
    def test_repeat_then_select_prefix_cache(self) -> None:
        cache = _repeat_cache_batch(_Cache(), 4)
        self.assertEqual(tuple(cache.layers[0].keys.shape), (4, 1, 3, 2))
        self.assertTrue(torch.equal(cache.layers[0].keys[0], cache.layers[0].keys[3]))

        selected = _select_cache_batch(cache, torch.tensor([3, 1]))
        self.assertEqual(tuple(selected.layers[0].keys.shape), (2, 1, 3, 2))
        self.assertTrue(torch.equal(selected.layers[0].keys[0], cache.layers[0].keys[3]))

    def test_session_computes_shared_prefix_once(self) -> None:
        class LanguageModel:
            def __init__(self) -> None:
                self.prefill_batch_sizes: list[int] = []

            def __call__(self, **kwargs):
                embeds = kwargs["inputs_embeds"]
                self.prefill_batch_sizes.append(int(embeds.shape[0]))
                batch, sequence = embeds.shape[:2]
                return types.SimpleNamespace(
                    past_key_values=_Cache(batch),
                    logits=torch.zeros(batch, sequence, 32),
                )

        class Tokenizer:
            @staticmethod
            def convert_tokens_to_ids(token: str) -> int:
                return {"<img>": 1, "<IMG_CONTEXT>": 2, "</img>": 3}[token]

        language_model = LanguageModel()
        model = types.SimpleNamespace(language_model=language_model)
        request = TextBatchRequest(prompt="same")
        prepared = (
            torch.zeros(1, 2, 4),
            torch.tensor([[0, 1], [0, 0], [0, 0]]),
            torch.zeros(1, 1, 2, 2),
        )
        with patch.object(
            NativeTextBatchSession, "_prepare_request", return_value=prepared
        ) as prepare:
            session = NativeTextBatchSession(
                model,
                Tokenizer(),
                [request] * 4,
                device="cpu",
                dtype=torch.float32,
                prefix_sharing=True,
            )

        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(language_model.prefill_batch_sizes, [1])
        self.assertEqual(tuple(session.next_logits.shape), (4, 32))
        self.assertEqual(tuple(session.cache.layers[0].keys.shape), (4, 1, 3, 2))


class _FakeTokenizer:
    _tokens = {"<eos>": 0, "<img>": 1, "</img>": 2, "<IMG_CONTEXT>": 3}
    _text = {10: "A", 11: "B", 20: "C", 21: "D"}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._tokens[token]

    def decode(self, tokens, *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(self._text[int(token)] for token in tokens)


class _FakeSession:
    scripts = ((10, 1, 11, 0), (20, 21, 0))
    latest = None

    def __init__(self, model, tokenizer, requests, **kwargs) -> None:
        del model, tokenizer
        self.requests = tuple(requests)
        self.kwargs = kwargs
        self.device = torch.device(kwargs["device"])
        self.positions = [0] * len(self.requests)
        self.commits: list[list[bool]] = []
        self.appended: list[list[int]] = []
        type(self).latest = self

    def constrained_logits(self) -> torch.Tensor:
        logits = torch.full((len(self.requests), 32), -1000.0)
        for row, script in enumerate(self.scripts):
            logits[row, script[self.positions[row]]] = 1.0
        return logits

    def commit(self, token_ids: torch.Tensor, accepted: torch.Tensor) -> None:
        del token_ids
        self.commits.append(accepted.tolist())
        for row, value in enumerate(accepted.tolist()):
            if value:
                self.positions[row] += 1

    def selected_cache(self, indices: torch.Tensor):
        count = int(indices.numel())
        return object(), torch.ones(count, 1, dtype=torch.bool), torch.zeros(
            count, dtype=torch.long
        )

    def append_generated_images(
        self, predictions: torch.Tensor, indices: torch.Tensor, **kwargs
    ) -> None:
        del predictions, kwargs
        self.appended.append(indices.tolist())


class InterleaveSchedulerTest(unittest.TestCase):
    def test_text_then_image_batches_and_rejoins_rows(self) -> None:
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(t_eps=0.0),
            device=torch.device("cpu"),
            template="fake",
        )
        requests = (
            InterleaveBatchRequest(prompt="same", seed=7),
            InterleaveBatchRequest(prompt="same", seed=8),
        )
        with (
            patch(
                "sensenova_u1.batch_inference.NativeTextBatchSession",
                _FakeSession,
            ),
            patch(
                "sensenova_u1.batch_inference.get_conv_template",
                return_value=types.SimpleNamespace(sep="<eos>"),
            ),
            patch(
                "sensenova_u1.batch_inference._run_image_sde_batch",
                return_value=torch.zeros(1, 3, 8, 8),
            ) as image_batch,
        ):
            results = batch_interleave_gen(
                model,
                _FakeTokenizer(),
                requests,
                generation_config=types.SimpleNamespace(max_new_tokens=8),
                image_size=(8, 8),
                num_steps=1,
                prefix_sharing=True,
                dtype=torch.float32,
            )

        self.assertEqual([result.text for result in results], ["A<image>B", "CD"])
        self.assertEqual([result.finish_reason for result in results], ["eos", "eos"])
        self.assertEqual([len(result.images) for result in results], [1, 0])
        self.assertEqual(_FakeSession.latest.appended, [[0]])
        self.assertTrue(_FakeSession.latest.kwargs["prefix_sharing"])
        image_batch.assert_called_once()

    def test_cfg_is_explicitly_rejected(self) -> None:
        model = types.SimpleNamespace(config=types.SimpleNamespace(), device="cpu")
        with self.assertRaisesRegex(NotImplementedError, "no-CFG"):
            batch_interleave_gen(
                model,
                _FakeTokenizer(),
                [InterleaveBatchRequest(prompt="x")],
                cfg_scale=2.0,
            )


if __name__ == "__main__":
    unittest.main()
