"""One-GPU throughput benchmark for native TI2T and TI2TI batching."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Callable
from typing import Any

import torch
from transformers import GenerationConfig

from sensenova_u1 import set_attn_backend
from sensenova_u1.batch_inference import InterleaveBatchRequest, TextBatchRequest
from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_INTERLEAVE
from sensenova_u1.utils import load_model_and_tokenizer


TEXT_PROMPT = (
    "Write a detailed technical explanation of why dynamic batching improves "
    "autoregressive inference throughput. Continue for at least 1000 words."
)
INTERLEAVE_PROMPT = (
    "I want to learn how to cook tomato and egg stir-fry. "
    "Please give me a beginner-friendly illustrated tutorial."
)


def _measure(run: Callable[[], Any]) -> tuple[Any, float, int, int]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return (
        result,
        elapsed,
        int(torch.cuda.max_memory_allocated()),
        int(torch.cuda.max_memory_reserved()),
    )


def _rate(numerator: int, elapsed: float) -> float:
    return float(numerator) / elapsed


def _text_benchmark(
    model: Any,
    tokenizer: Any,
    *,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    requests = tuple(TextBatchRequest(prompt=TEXT_PROMPT) for _ in range(batch_size))
    generation_config = GenerationConfig(max_new_tokens=max_new_tokens)

    # Warm the one-row prefill/decode kernels without affecting measurements.
    model.batch_text_gen(
        tokenizer,
        requests[:1],
        generation_config=GenerationConfig(max_new_tokens=8),
        prefix_sharing=False,
        device="cuda",
        dtype=torch.bfloat16,
    )

    def run_serial() -> tuple[Any, ...]:
        completed = []
        print("TI2T_SERIAL_START", flush=True)
        for index, request in enumerate(requests, start=1):
            completed.append(
                model.batch_text_gen(
                    tokenizer,
                    (request,),
                    generation_config=generation_config,
                    prefix_sharing=False,
                    device="cuda",
                    dtype=torch.bfloat16,
                )[0]
            )
            print(f"TI2T_SERIAL_PROGRESS={index}/{batch_size}", flush=True)
        return tuple(completed)

    serial, serial_seconds, serial_allocated, serial_reserved = _measure(run_serial)

    def run_batch() -> tuple[Any, ...]:
        print(f"TI2T_BATCH_START={batch_size}", flush=True)
        return model.batch_text_gen(
            tokenizer,
            requests,
            generation_config=generation_config,
            prefix_sharing=True,
            device="cuda",
            dtype=torch.bfloat16,
        )

    batch, batch_seconds, batch_allocated, batch_reserved = _measure(run_batch)
    serial_tokens = sum(result.generated_tokens for result in serial)
    batch_tokens = sum(result.generated_tokens for result in batch)
    exact_matches = sum(
        left.text == right.text and left.finish_reason == right.finish_reason
        for left, right in zip(serial, batch)
    )
    return {
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "serial": {
            "seconds": serial_seconds,
            "samples_per_second": _rate(batch_size, serial_seconds),
            "tokens": serial_tokens,
            "tokens_per_second": _rate(serial_tokens, serial_seconds),
            "peak_allocated_bytes": serial_allocated,
            "peak_reserved_bytes": serial_reserved,
        },
        "batch": {
            "seconds": batch_seconds,
            "samples_per_second": _rate(batch_size, batch_seconds),
            "tokens": batch_tokens,
            "tokens_per_second": _rate(batch_tokens, batch_seconds),
            "peak_allocated_bytes": batch_allocated,
            "peak_reserved_bytes": batch_reserved,
        },
        "speedup": serial_seconds / batch_seconds,
        "exact_match_rows": exact_matches,
    }


def _summarize_interleave(results: tuple[Any, ...]) -> dict[str, Any]:
    images_cpu: list[torch.Tensor | None] = []
    for result in results:
        if result.images:
            images_cpu.append(result.images[0].detach().float().cpu())
        else:
            images_cpu.append(None)
    return {
        "texts": [result.text for result in results],
        "finish_reasons": [result.finish_reason for result in results],
        "generated_tokens": [result.generated_tokens for result in results],
        "image_counts": [len(result.images) for result in results],
        "image_shapes": [
            list(image.shape) if image is not None else None for image in images_cpu
        ],
        "images_finite": [
            bool(torch.isfinite(image).all().item()) if image is not None else False
            for image in images_cpu
        ],
        "images": images_cpu,
    }


def _interleave_benchmark(
    model: Any,
    tokenizer: Any,
    *,
    batch_size: int,
    max_new_tokens: int,
    image_size: int,
    image_steps: int,
) -> dict[str, Any]:
    requests = tuple(
        InterleaveBatchRequest(
            prompt=INTERLEAVE_PROMPT,
            system_message=SYSTEM_MESSAGE_FOR_INTERLEAVE,
            seed=100 + row,
        )
        for row in range(batch_size)
    )
    generation_config = GenerationConfig(max_new_tokens=max_new_tokens)
    common = {
        "generation_config": generation_config,
        "cfg_scale": 1.0,
        "img_cfg_scale": 1.0,
        "cfg_norm": "none",
        "max_images": 1,
        "image_size": (image_size, image_size),
        "num_steps": image_steps,
        "think_mode": True,
        "device": "cuda",
        "dtype": torch.bfloat16,
    }

    # Warm all text/image/text phases with one cheap SDE step.
    warmup = dict(common)
    warmup["num_steps"] = 1
    model.batch_interleave_gen(
        tokenizer,
        requests[:1],
        prefix_sharing=False,
        **warmup,
    )

    def run_serial() -> tuple[Any, ...]:
        completed = []
        print("TI2TI_SERIAL_START", flush=True)
        for index, request in enumerate(requests, start=1):
            completed.append(
                model.batch_interleave_gen(
                    tokenizer,
                    (request,),
                    prefix_sharing=False,
                    **common,
                )[0]
            )
            print(f"TI2TI_SERIAL_PROGRESS={index}/{batch_size}", flush=True)
        return tuple(completed)

    serial_raw, serial_seconds, serial_allocated, serial_reserved = _measure(run_serial)
    serial = _summarize_interleave(serial_raw)
    del serial_raw

    def run_batch() -> tuple[Any, ...]:
        print(f"TI2TI_BATCH_START={batch_size}", flush=True)
        return model.batch_interleave_gen(
            tokenizer,
            requests,
            prefix_sharing=True,
            **common,
        )

    batch_raw, batch_seconds, batch_allocated, batch_reserved = _measure(run_batch)
    batch = _summarize_interleave(batch_raw)
    del batch_raw

    image_mae: list[float | None] = []
    for serial_image, batch_image in zip(serial["images"], batch["images"]):
        if serial_image is None or batch_image is None:
            image_mae.append(None)
        else:
            image_mae.append(float((serial_image - batch_image).abs().mean().item()))
    exact_matches = sum(
        left == right for left, right in zip(serial["texts"], batch["texts"])
    )
    serial_tokens = sum(serial["generated_tokens"])
    batch_tokens = sum(batch["generated_tokens"])
    for summary in (serial, batch):
        del summary["images"]

    return {
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "image_size": [image_size, image_size],
        "image_steps": image_steps,
        "serial": {
            "seconds": serial_seconds,
            "samples_per_second": _rate(batch_size, serial_seconds),
            "tokens": serial_tokens,
            "tokens_per_second": _rate(serial_tokens, serial_seconds),
            "peak_allocated_bytes": serial_allocated,
            "peak_reserved_bytes": serial_reserved,
            **{key: value for key, value in serial.items() if key != "texts"},
        },
        "batch": {
            "seconds": batch_seconds,
            "samples_per_second": _rate(batch_size, batch_seconds),
            "tokens": batch_tokens,
            "tokens_per_second": _rate(batch_tokens, batch_seconds),
            "peak_allocated_bytes": batch_allocated,
            "peak_reserved_bytes": batch_reserved,
            **{key: value for key, value in batch.items() if key != "texts"},
        },
        "speedup": serial_seconds / batch_seconds,
        "exact_match_rows": exact_matches,
        "first_image_mean_absolute_error": image_mae,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("all", "ti2t", "ti2ti"), default="all")
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--text-max-new-tokens", type=int, default=16384)
    parser.add_argument("--interleave-batch-size", type=int, default=8)
    parser.add_argument("--interleave-max-new-tokens", type=int, default=16384)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--image-steps", type=int, default=30)
    args = parser.parse_args()

    set_attn_backend("flash")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype=torch.bfloat16,
        device="cuda",
    )
    report: dict[str, Any] = {
        "model": args.model,
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(),
    }
    if args.mode in ("all", "ti2t"):
        report["ti2t"] = _text_benchmark(
            model,
            tokenizer,
            batch_size=args.text_batch_size,
            max_new_tokens=args.text_max_new_tokens,
        )
        print("TI2T_RESULT=" + json.dumps(report["ti2t"], sort_keys=True), flush=True)
    if args.mode in ("all", "ti2ti"):
        report["ti2ti"] = _interleave_benchmark(
            model,
            tokenizer,
            batch_size=args.interleave_batch_size,
            max_new_tokens=args.interleave_max_new_tokens,
            image_size=args.image_size,
            image_steps=args.image_steps,
        )
        print("TI2TI_RESULT=" + json.dumps(report["ti2ti"], sort_keys=True), flush=True)
    print("BENCHMARK_RESULT=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
