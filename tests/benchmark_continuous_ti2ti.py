"""Benchmark native SenseNova continuous TI2TI batching on canonical rows."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from sensenova_u1 import set_attn_backend
from sensenova_u1.batch_inference import (
    ContinuousInterleaveBatchEngine,
    InterleaveBatchRequest,
    _apply_repetition_penalty,
)
from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_INTERLEAVE
from sensenova_u1.utils import load_model_and_tokenizer


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _load_rows(
    path: Path,
    *,
    limit: int,
    min_authored_images: int,
    input_image_count: int,
) -> list[dict]:
    selected = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row["TI2TI"]["prompt"]
            authored = row["TI2TI"]["answer"]["interleaved_output"]
            authored_images = sum(item.get("type") == "image" for item in authored)
            if input_image_count and len(prompt["image_list"]) != input_image_count:
                continue
            if authored_images < min_authored_images:
                continue
            selected.append(row)
            if len(selected) == limit:
                break
    if len(selected) != limit:
        raise ValueError(
            f"requested {limit} matching rows, found {len(selected)} in {path}"
        )
    return selected


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=_positive_int, default=64)
    parser.add_argument("--min-authored-images", type=_positive_int, default=2)
    parser.add_argument(
        "--input-image-count",
        type=_nonnegative_int,
        default=1,
        help="required input image count; zero accepts any count",
    )
    parser.add_argument("--max-batch-size", type=_positive_int, default=32)
    parser.add_argument("--max-image-batch-size", type=_positive_int, default=8)
    parser.add_argument("--image-wait-steps", type=_nonnegative_int, default=8)
    parser.add_argument("--prefill-chunk-size", type=_positive_int, default=2048)
    parser.add_argument("--max-model-len", type=_positive_int, default=16384)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=8192)
    parser.add_argument("--max-kv-tokens", type=_positive_int, default=131072)
    parser.add_argument("--kv-page-size", type=_positive_int, default=256)
    parser.add_argument("--max-images", type=_positive_int, default=10)
    parser.add_argument("--image-size", type=_positive_int, default=256)
    parser.add_argument("--image-steps", type=_positive_int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--progress-every", type=_positive_int, default=8)
    args = parser.parse_args()
    if args.repetition_penalty < 1.0:
        parser.error("--repetition-penalty must be at least 1.0")

    rows = _load_rows(
        Path(args.manifest),
        limit=args.limit,
        min_authored_images=args.min_authored_images,
        input_image_count=args.input_image_count,
    )
    requests = tuple(
        InterleaveBatchRequest(
            prompt=row["TI2TI"]["phase_prompts"]["rl"],
            images=tuple(row["TI2TI"]["prompt"]["image_list"]),
            system_message=SYSTEM_MESSAGE_FOR_INTERLEAVE,
            seed=10000 + index,
        )
        for index, row in enumerate(rows)
    )

    set_attn_backend("flash")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype=torch.bfloat16,
        device="cuda",
    )
    engine = ContinuousInterleaveBatchEngine(
        model,
        tokenizer,
        device="cuda",
        dtype=torch.bfloat16,
        max_batch_size=args.max_batch_size,
        max_image_batch_size=args.max_image_batch_size,
        image_wait_steps=args.image_wait_steps,
        max_model_len=args.max_model_len,
        prefill_chunk_size=args.prefill_chunk_size,
        default_max_new_tokens=args.max_new_tokens,
        max_kv_tokens=args.max_kv_tokens,
        kv_page_size=args.kv_page_size,
        prefix_sharing=False,
        max_images=args.max_images,
        image_size=(args.image_size, args.image_size),
        num_steps=args.image_steps,
        think_mode=True,
    )
    request_ids = [engine.submit(request) for request in requests]
    seen_tokens: dict[int, set[int]] = {
        request_id: set() for request_id in request_ids
    }
    completed = {}
    completion_seconds: dict[int, float] = {}
    text_batch_sizes: list[int] = []
    image_batch_sizes: list[int] = []
    scheduler_steps = 0
    text_row_steps = 0
    image_row_steps = 0
    next_progress = args.progress_every

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    while engine.has_unfinished_requests:
        scheduler_steps += 1
        text_batch = engine.schedule_text()
        if text_batch is not None:
            text_batch_sizes.append(len(text_batch.request_ids))
            text_row_steps += len(text_batch.request_ids)
            logits = text_batch.logits
            if args.repetition_penalty != 1.0:
                seen = torch.zeros_like(logits, dtype=torch.bool)
                for batch_row, request_id in enumerate(text_batch.request_ids):
                    tokens = seen_tokens[request_id]
                    if tokens:
                        indexes = torch.tensor(
                            tuple(tokens), device="cuda", dtype=torch.long
                        )
                        seen[batch_row, indexes] = True
                logits = _apply_repetition_penalty(
                    logits, seen, args.repetition_penalty
                )
            next_tokens = torch.argmax(logits, dim=-1).to(dtype=torch.long)
            for request_id, token in zip(
                text_batch.request_ids,
                next_tokens.detach().cpu().tolist(),
                strict=True,
            ):
                if token not in (engine.eos_token_id, engine.image_start_token_id):
                    seen_tokens[request_id].add(int(token))
            engine.advance_text(text_batch, next_tokens)
            for request_id, result in engine.pop_completed():
                completed[request_id] = result
                completion_seconds[request_id] = time.perf_counter() - started

        image_batch = engine.schedule_images()
        if image_batch is not None:
            image_batch_sizes.append(len(image_batch.request_ids))
            image_row_steps += len(image_batch.request_ids)
            engine.run_images(image_batch)
        for request_id, result in engine.pop_completed():
            completed[request_id] = result
            completion_seconds[request_id] = time.perf_counter() - started

        if len(completed) >= next_progress:
            elapsed = time.perf_counter() - started
            print(
                f"TI2TI_PROGRESS={len(completed)}/{len(requests)} "
                f"elapsed={elapsed:.3f} active={engine.active_size} "
                f"waiting={engine.waiting_size} free_kv_tokens={engine.free_kv_tokens}",
                flush=True,
            )
            while next_progress <= len(completed):
                next_progress += args.progress_every

    for request_id, result in engine.pop_completed():
        completed[request_id] = result
        completion_seconds[request_id] = time.perf_counter() - started
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    results = [completed[request_id] for request_id in request_ids]
    latencies = [completion_seconds[request_id] for request_id in request_ids]
    total_tokens = sum(result.generated_tokens for result in results)
    total_images = sum(len(result.images) for result in results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row, result, latency in zip(rows, results, latencies, strict=True):
            handle.write(
                json.dumps(
                    {
                        "sample_id": row["id"],
                        "task_name": row["task_name"],
                        "text": result.text,
                        "finish_reason": result.finish_reason,
                        "generated_tokens": result.generated_tokens,
                        "generated_images": len(result.images),
                        "latency_seconds": latency,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = {
        "backend": engine.decode_backend,
        "rows": len(results),
        "selection": {
            "input_images": args.input_image_count or "any",
            "min_authored_reasoning_images": args.min_authored_images,
            "tasks": dict(Counter(row["task_name"] for row in rows)),
        },
        "max_batch_size": args.max_batch_size,
        "max_image_batch_size": args.max_image_batch_size,
        "image_wait_steps": args.image_wait_steps,
        "prefill_chunk_size": args.prefill_chunk_size,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "max_kv_tokens": args.max_kv_tokens,
        "kv_page_size": args.kv_page_size,
        "max_images": args.max_images,
        "image_size": args.image_size,
        "image_steps": args.image_steps,
        "repetition_penalty": args.repetition_penalty,
        "elapsed_seconds": elapsed,
        "samples_per_second": len(results) / elapsed,
        "generated_tokens": total_tokens,
        "generated_tokens_per_second": total_tokens / elapsed,
        "generated_images": total_images,
        "generated_images_per_second": total_images / elapsed,
        "image_sde_steps_per_second": total_images * args.image_steps / elapsed,
        "scheduler_steps": scheduler_steps,
        "text_row_steps": text_row_steps,
        "image_row_steps": image_row_steps,
        "text_batches": len(text_batch_sizes),
        "image_batches": len(image_batch_sizes),
        "mean_text_batch": _mean(text_batch_sizes),
        "median_text_batch": statistics.median(text_batch_sizes)
        if text_batch_sizes
        else 0.0,
        "max_text_batch": max(text_batch_sizes, default=0),
        "mean_image_batch": _mean(image_batch_sizes),
        "median_image_batch": statistics.median(image_batch_sizes)
        if image_batch_sizes
        else 0.0,
        "max_image_batch": max(image_batch_sizes, default=0),
        "finish_reasons": dict(Counter(result.finish_reason for result in results)),
        "generated_image_count_distribution": dict(
            sorted(Counter(len(result.images) for result in results).items())
        ),
        "latency_seconds": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies, default=0.0),
        },
        "free_kv_tokens_at_end": engine.free_kv_tokens,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "output": str(output),
    }
    report_path = output.with_suffix(output.suffix + ".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("CONTINUOUS_TI2TI_RESULT=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
