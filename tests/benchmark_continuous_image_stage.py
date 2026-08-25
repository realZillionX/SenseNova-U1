"""Measure only the continuous TI2TI image stage on one GPU."""

from __future__ import annotations

import argparse
import gc
import json
import time

import torch

from sensenova_u1 import set_attn_backend
from sensenova_u1.batch_inference import (
    ContinuousInterleaveBatchEngine,
    InterleaveBatchRequest,
)
from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_INTERLEAVE
from sensenova_u1.utils import load_model_and_tokenizer


def _run_image_stage(model, tokenizer, *, batch_size: int, num_steps: int) -> dict:
    engine = ContinuousInterleaveBatchEngine(
        model,
        tokenizer,
        device="cuda",
        dtype=torch.bfloat16,
        max_batch_size=batch_size,
        max_image_batch_size=batch_size,
        max_model_len=16384,
        prefill_chunk_size=2048,
        default_max_new_tokens=8192,
        max_kv_tokens=131072,
        kv_page_size=256,
        prefix_sharing=True,
        max_images=10,
        image_size=(256, 256),
        num_steps=num_steps,
        think_mode=True,
    )
    request_ids = [
        engine.submit(
            InterleaveBatchRequest(
                prompt="Generate one visual reasoning state.",
                system_message=SYSTEM_MESSAGE_FOR_INTERLEAVE,
                seed=1000 + row,
            )
        )
        for row in range(batch_size)
    ]
    text_batch = engine.schedule_text()
    if text_batch is None or tuple(request_ids) != text_batch.request_ids:
        raise RuntimeError("shared-prefix image benchmark did not admit one full batch")
    image_tokens = torch.full(
        (batch_size,),
        engine.image_start_token_id,
        device="cuda",
        dtype=torch.long,
    )
    engine.advance_text(text_batch, image_tokens)
    image_batch = engine.schedule_images(max_batch_size=batch_size)
    if image_batch is None or len(image_batch.request_ids) != batch_size:
        raise RuntimeError("image benchmark did not assemble the requested batch")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    predictions = engine.run_images(image_batch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if tuple(predictions.shape) != (batch_size, 3, 256, 256):
        raise RuntimeError("image benchmark returned an unexpected tensor shape")
    result = {
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / batch_size,
        "images_per_second": batch_size / elapsed,
        "sde_steps_per_second": batch_size * num_steps / elapsed,
        "num_steps": num_steps,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    del predictions, image_batch, text_batch, engine
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image-steps", type=int, default=30)
    args = parser.parse_args()
    if args.image_steps < 1:
        parser.error("--image-steps must be positive")

    set_attn_backend("flash")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype=torch.bfloat16,
        device="cuda",
    )
    _run_image_stage(model, tokenizer, batch_size=1, num_steps=2)
    results = [
        _run_image_stage(
            model,
            tokenizer,
            batch_size=batch_size,
            num_steps=args.image_steps,
        )
        for batch_size in (1, 2, 4, 8)
    ]
    print(
        "CONTINUOUS_IMAGE_STAGE_RESULT="
        + json.dumps(
            {
                "backend": "flash_kv_paged_continuous",
                "image_size": 256,
                "results": results,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
