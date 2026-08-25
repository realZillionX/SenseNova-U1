from __future__ import annotations

import types
import unittest
from collections import deque
from unittest.mock import patch

import torch

from sensenova_u1.batch_inference import (
    ContinuousInterleaveBatchEngine,
    ContinuousTextBatch,
    InterleaveBatchRequest,
    NativeTextBatchSession,
    TextBatchRequest,
    _ContinuousInterleavePhase,
    _ContinuousInterleaveState,
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
            patch(
                "sensenova_u1.batch_inference._run_paged_image_sde_batch",
            ) as paged_image_batch,
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
        paged_image_batch.assert_not_called()

    def test_cfg_is_explicitly_rejected(self) -> None:
        model = types.SimpleNamespace(config=types.SimpleNamespace(), device="cpu")
        with self.assertRaisesRegex(NotImplementedError, "no-CFG"):
            batch_interleave_gen(
                model,
                _FakeTokenizer(),
                [InterleaveBatchRequest(prompt="x")],
                cfg_scale=2.0,
            )


class _FakePagedCache:
    def __init__(self) -> None:
        self.flash_decode_seqlens = torch.tensor([5, 5], dtype=torch.int32)
        self.active = None
        self.active_tokens = 0
        self.activations: list[tuple[list[int], int]] = []
        self.released: list[int] = []

    def activate(self, slots: torch.Tensor, *, token_count: int = 1) -> None:
        self.active = slots.clone()
        self.active_tokens = token_count
        self.activations.append((slots.tolist(), token_count))

    def commit_active(self) -> None:
        self.flash_decode_seqlens.index_add_(
            0,
            self.active,
            torch.full_like(self.active, self.active_tokens, dtype=torch.int32),
        )
        self.active = None

    def cancel_active(self) -> None:
        self.active = None

    def can_reserve(self, slots: torch.Tensor, token_count: int) -> bool:
        del slots, token_count
        return True

    def release(self, slot: int) -> None:
        self.released.append(slot)


class ContinuousInterleaveSchedulerTest(unittest.TestCase):
    def _engine(self, count: int = 2):
        class LanguageModel:
            def __call__(self, *, input_ids, **kwargs):
                del kwargs
                batch = int(input_ids.shape[0])
                logits = torch.arange(batch * 32, dtype=torch.float32).reshape(
                    batch, 1, 32
                )
                return types.SimpleNamespace(logits=logits)

        tokenizer = _FakeTokenizer()
        model = types.SimpleNamespace(
            language_model=LanguageModel(),
            patch_size=1,
            downsample_ratio=1.0,
        )
        engine = ContinuousInterleaveBatchEngine.__new__(
            ContinuousInterleaveBatchEngine
        )
        engine.model = model
        engine.tokenizer = tokenizer
        engine.device = torch.device("cpu")
        engine.eos_token_id = 0
        engine.image_start_token_id = 1
        engine.image_end_token_id = 2
        engine.image_size = (2, 2)
        engine.max_images = 1
        engine.max_model_len = 128
        engine.max_image_batch_size = 8
        engine.image_wait_steps = 8
        engine._text_decode_step = 0
        engine.num_steps = 1
        engine.enable_timestep_shift = True
        engine.timestep_shift = 1.0
        engine.cfg_interval = (0.0, 1.0)
        engine._cache = _FakePagedCache()
        engine._free_slots = deque()
        engine._completed = deque()
        engine._waiting = deque()
        engine._prefilling = None
        engine._scheduled_request_ids = tuple(range(count))
        engine._scheduled_image_request_ids = None
        request = InterleaveBatchRequest(prompt="same", seed=7)
        states = {
            request_id: _ContinuousInterleaveState(
                request_id=request_id,
                request=TextBatchRequest(prompt="same"),
                interleave_request=request,
                max_new_tokens=8,
                slot=request_id,
                t_index=3,
                next_logits=torch.zeros(1, 32),
                generator=torch.Generator().manual_seed(7 + request_id),
            )
            for request_id in range(count)
        }
        engine._active = states
        return engine, states

    def test_image_action_leaves_text_batch_without_waiting_for_other_row(self) -> None:
        engine, states = self._engine()
        batch = ContinuousTextBatch(
            request_ids=(0, 1), logits=torch.zeros(2, 32)
        )

        engine.advance_text(batch, torch.tensor([1, 10], dtype=torch.long))

        self.assertIs(
            states[0].phase, _ContinuousInterleavePhase.IMAGE_READY
        )
        self.assertIs(
            states[1].phase, _ContinuousInterleavePhase.TEXT_READY
        )
        self.assertEqual(states[0].image_ready_step, 1)
        self.assertEqual(engine._text_decode_step, 1)
        self.assertEqual(states[1].generated, [10])
        self.assertEqual(engine._cache.flash_decode_seqlens.tolist(), [6, 6])

    def test_image_batch_rejoins_text_queue_after_one_paged_sde(self) -> None:
        engine, states = self._engine()
        states[0].phase = _ContinuousInterleavePhase.IMAGE_READY
        states[0].image_ready_step = 0
        engine._active = {0: states[0]}
        engine._scheduled_request_ids = None
        image_batch = engine.schedule_images()
        self.assertEqual(image_batch.request_ids, (0,))
        prediction = torch.zeros(1, 3, 2, 2)

        with (
            patch(
                "sensenova_u1.batch_inference._run_paged_image_sde_batch",
                return_value=prediction,
            ) as image_sde,
            patch.object(
                engine, "_append_generated_images_paged"
            ) as append_images,
        ):
            output = engine.run_images(image_batch)

        self.assertIs(output, prediction)
        self.assertIs(
            states[0].phase, _ContinuousInterleavePhase.TEXT_READY
        )
        self.assertEqual(states[0].parts, ["<image>"])
        self.assertEqual(len(states[0].generated_images), 1)
        self.assertIsNone(states[0].image_ready_step)
        self.assertEqual(engine._cache.activations[-1], ([0], 5))
        image_sde.assert_called_once()
        append_images.assert_called_once()

    def test_image_batch_waits_at_most_eight_text_decode_steps(self) -> None:
        engine, states = self._engine()
        states[0].phase = _ContinuousInterleavePhase.IMAGE_READY
        states[0].image_ready_step = 0
        engine._scheduled_request_ids = None
        engine._text_decode_step = 7

        self.assertIsNone(engine.schedule_images())

        engine._text_decode_step = 8
        image_batch = engine.schedule_images()
        self.assertEqual(image_batch.request_ids, (0,))

    def test_full_image_queue_flushes_while_text_can_progress(self) -> None:
        engine, states = self._engine(count=9)
        for request_id in range(8):
            states[request_id].phase = _ContinuousInterleavePhase.IMAGE_READY
            states[request_id].image_ready_step = 0
        engine._scheduled_request_ids = None

        image_batch = engine.schedule_images()

        self.assertEqual(image_batch.request_ids, tuple(range(8)))


if __name__ == "__main__":
    unittest.main()
