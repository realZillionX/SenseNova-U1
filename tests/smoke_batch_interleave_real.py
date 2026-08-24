"""One-GPU smoke for the public U1.5 batched TI2TI interface."""

from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import GenerationConfig

from sensenova_u1 import set_attn_backend
from sensenova_u1.batch_inference import InterleaveBatchRequest
from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_INTERLEAVE
from sensenova_u1.utils import load_model_and_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--image-steps", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 2:
        raise ValueError("real batch smoke requires at least two rows")

    set_attn_backend("flash")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype=torch.bfloat16,
        device="cuda",
    )
    prompt = (
        "I want to learn how to cook tomato and egg stir-fry. "
        "Please give me a beginner-friendly illustrated tutorial."
    )
    requests = tuple(
        InterleaveBatchRequest(
            prompt=prompt,
            system_message=SYSTEM_MESSAGE_FOR_INTERLEAVE,
            seed=100 + row,
        )
        for row in range(args.batch_size)
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    results = model.batch_interleave_gen(
        tokenizer,
        requests,
        generation_config=GenerationConfig(max_new_tokens=64),
        cfg_scale=1.0,
        img_cfg_scale=1.0,
        max_images=1,
        image_size=(args.image_size, args.image_size),
        num_steps=args.image_steps,
        think_mode=True,
        prefix_sharing=True,
        device="cuda",
        dtype=torch.bfloat16,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    diagnostics = {
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "finish_reasons": [result.finish_reason for result in results],
        "generated_tokens": [result.generated_tokens for result in results],
        "image_shapes": [
            [list(image.shape) for image in result.images]
            for result in results
        ],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "texts": [result.text for result in results],
    }
    print(json.dumps(diagnostics, ensure_ascii=False, sort_keys=True), flush=True)
    if not all(result.images for result in results):
        raise RuntimeError(
            "TI2TI smoke did not enter the image batch for every shared-prefix row"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
