"""Benchmark native SenseNova continuous TI2T batching on BabyVision rows."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sensenova_u1 import set_attn_backend
from sensenova_u1.batch_inference import (
    ContinuousTextBatchEngine,
    TextBatchRequest,
    _apply_repetition_penalty,
)
from sensenova_u1.utils import load_model_and_tokenizer


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _format_choices(options) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join(
        f"{labels[index]}. {option}" for index, option in enumerate(options)
    )


def _prepare_item(item: dict, root: Path) -> TextBatchRequest:
    image = root / item["image"]
    if not image.is_file():
        raise FileNotFoundError(image)
    question = str(item["question"])
    if item["ansType"] != "blank":
        question += "\nChoices:\n" + _format_choices(item["options"])
    question += "\nPut your final answer inside <answer></answer>."
    return TextBatchRequest(
        prompt=question,
        images=(str(image),),
        system_message="",
    )


def _load_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) == limit:
                    break
    if len(rows) != limit:
        raise ValueError(f"requested {limit} BabyVision rows, found {len(rows)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=_positive_int, default=64)
    parser.add_argument("--max-batch-size", type=_positive_int, default=32)
    parser.add_argument("--prefill-chunk-size", type=_positive_int, default=2048)
    parser.add_argument("--max-model-len", type=_positive_int, default=32768)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=16384)
    parser.add_argument("--max-kv-tokens", type=_positive_int, default=98304)
    parser.add_argument("--kv-page-size", type=_positive_int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    args = parser.parse_args()
    if args.repetition_penalty < 1.0:
        parser.error("--repetition-penalty must be at least 1.0")

    rows = _load_rows(Path(args.data), args.limit)
    requests = tuple(
        _prepare_item(item, Path(args.image_root)) for item in rows
    )
    set_attn_backend("flash")
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        dtype=torch.bfloat16,
        device="cuda",
    )
    engine = ContinuousTextBatchEngine(
        model,
        tokenizer,
        device="cuda",
        dtype=torch.bfloat16,
        max_batch_size=args.max_batch_size,
        max_model_len=args.max_model_len,
        prefill_chunk_size=args.prefill_chunk_size,
        default_max_new_tokens=args.max_new_tokens,
        max_kv_tokens=args.max_kv_tokens,
        kv_page_size=args.kv_page_size,
        prefix_sharing=False,
    )
    request_ids = [engine.submit(request) for request in requests]
    seen_tokens: dict[int, set[int]] = {
        request_id: set() for request_id in request_ids
    }
    completed = {}
    active_sizes = []
    decode_row_steps = 0
    scheduler_steps = 0
    next_progress = 8
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    while engine.has_unfinished_requests:
        batch = engine.schedule()
        scheduler_steps += 1
        if batch is None:
            continue
        active_sizes.append(len(batch.request_ids))
        decode_row_steps += len(batch.request_ids)
        logits = batch.logits
        if args.repetition_penalty != 1.0:
            seen = torch.zeros_like(logits, dtype=torch.bool)
            for row, request_id in enumerate(batch.request_ids):
                tokens = seen_tokens[request_id]
                if tokens:
                    indexes = torch.tensor(
                        tuple(tokens), device="cuda", dtype=torch.long
                    )
                    seen[row, indexes] = True
            logits = _apply_repetition_penalty(
                logits, seen, args.repetition_penalty
            )
        token_ids = torch.argmax(logits, dim=-1).to(dtype=torch.long)
        for request_id, token in zip(
            batch.request_ids, token_ids.detach().cpu().tolist(), strict=True
        ):
            if int(token) != engine.eos_token_id:
                seen_tokens[request_id].add(int(token))
        engine.advance(batch, token_ids)
        newly_completed = engine.pop_completed()
        completed.update(newly_completed)
        if newly_completed and len(completed) >= next_progress:
            elapsed = time.perf_counter() - started
            print(
                f"CONTINUOUS_PROGRESS={len(completed)}/{len(requests)} "
                f"elapsed={elapsed:.3f} active={engine.active_size} "
                f"waiting={engine.waiting_size} "
                f"free_kv_tokens={engine.free_kv_tokens}",
                flush=True,
            )
            while next_progress <= len(completed):
                next_progress += 8
    completed.update(engine.pop_completed())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    results = [completed[request_id] for request_id in request_ids]
    total_tokens = sum(result.generated_tokens for result in results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item, result in zip(rows, results, strict=True):
            handle.write(
                json.dumps(
                    {
                        "taskId": item["taskId"],
                        "model_response": result.text,
                        "finish_reason": result.finish_reason,
                        "generated_tokens": result.generated_tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    report = {
        "backend": engine.decode_backend,
        "rows": len(results),
        "max_batch_size": args.max_batch_size,
        "prefill_chunk_size": args.prefill_chunk_size,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "max_kv_tokens": args.max_kv_tokens,
        "kv_page_size": args.kv_page_size,
        "repetition_penalty": args.repetition_penalty,
        "elapsed_seconds": elapsed,
        "samples_per_second": len(results) / elapsed,
        "generated_tokens": total_tokens,
        "generated_tokens_per_second": total_tokens / elapsed,
        "decode_row_steps": decode_row_steps,
        "scheduler_steps": scheduler_steps,
        "mean_active_batch": (
            sum(active_sizes) / len(active_sizes) if active_sizes else 0.0
        ),
        "max_active_batch": max(active_sizes, default=0),
        "free_kv_tokens_at_end": engine.free_kv_tokens,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "output": str(output),
    }
    print("CONTINUOUS_BABYVISION_RESULT=" + json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
