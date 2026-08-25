"""Run BabyVision with native SenseNova continuous TI2T batching.

This is a local evaluation adapter.  It deliberately reuses the official
BabyVision prompt construction, decode policy, and JSONL schema while replacing
the HTTP request fan-out with the in-process paged-KV scheduler.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from infer_babyvision import (
    DEFAULT_DATA_PATH,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPETITION_PENALTY,
    _safe_model_name,
    extract_boxed_answer,
    format_choices,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BabyVision with SenseNova continuous TI2T batching."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="SenseNova-U1.5-8B-MoT")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--prefill-chunk-size", type=_positive_int, default=2048)
    parser.add_argument("--max-model-len", type=_positive_int, default=65536)
    parser.add_argument("--max-kv-tokens", type=_positive_int, default=131072)
    parser.add_argument("--kv-page-size", type=_positive_int, default=256)
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Evaluate only the first N pending rows (for a real-batch smoke).",
    )
    parser.add_argument(
        "--max-new-tokens", type=_positive_int, default=DEFAULT_MAX_NEW_TOKENS
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_REPETITION_PENALTY,
    )
    return parser.parse_args()


def _prepare_item(item: dict, image_root: Path) -> tuple[str, str, str]:
    """Mirror the official HTTP inference prompt and answer preparation."""

    image_path = image_root / item["image"]
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    if item["ansType"] == "blank":
        question = item["question"]
        answer = item["blankAns"]
    else:
        question = item["question"] + "\nChoices:\n" + format_choices(item["options"])
        choice_answer = str(item["choiceAns"]).strip()
        if choice_answer.isdigit():
            index = int(choice_answer)
            answer = chr(64 + index) if 1 <= index <= 26 else chr(65 + index)
        else:
            answer = choice_answer

    question = question + "\nPut your final answer inside <answer></answer>."
    return question, answer, str(image_path)


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_processed_ids(path: Path) -> set:
    if not path.exists():
        return set()
    processed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                processed.add(json.loads(line)["taskId"])
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid JSONL row at {path}:{line_number}: {exc}"
                ) from exc
    return processed


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    if args.repetition_penalty < 1.0:
        raise ValueError("--repetition-penalty must be >= 1.0")

    import torch
    from sensenova_u1 import set_attn_backend
    from sensenova_u1.batch_inference import (
        ContinuousTextBatchEngine,
        TextBatchRequest,
        _apply_repetition_penalty,
    )
    from sensenova_u1.utils import load_model_and_tokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"babyvision_{_safe_model_name(args.model_name)}.jsonl"
    rows = _load_rows(data_path)
    processed_ids = _load_processed_ids(output_path)
    pending = [row for row in rows if row["taskId"] not in processed_ids]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        json.dumps(
            {
                "backend": "flash_kv_paged_continuous",
                "model": args.model_name,
                "max_batch_size": args.batch_size,
                "prefill_chunk_size": args.prefill_chunk_size,
                "max_model_len": args.max_model_len,
                "max_kv_tokens": args.max_kv_tokens,
                "kv_page_size": args.kv_page_size,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "temperature": 0,
                "top_p": 0.95,
                "repetition_penalty": args.repetition_penalty,
                "rows": len(rows),
                "already_processed": len(processed_ids),
                "pending": len(pending),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not pending:
        return 0

    total_started = time.perf_counter()
    set_attn_backend("flash")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        dtype=torch.bfloat16,
        device="cuda",
    )
    model_load_seconds = time.perf_counter() - total_started
    engine = ContinuousTextBatchEngine(
        model,
        tokenizer,
        device="cuda",
        dtype=torch.bfloat16,
        max_batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        prefill_chunk_size=args.prefill_chunk_size,
        default_max_new_tokens=args.max_new_tokens,
        max_kv_tokens=args.max_kv_tokens,
        kv_page_size=args.kv_page_size,
        prefix_sharing=False,
        truncate_to_max_model_len=True,
    )

    success_count = 0
    failed = []
    prepared = {}
    seen_tokens = {}
    for item in pending:
        try:
            question, answer, image_path = _prepare_item(item, image_root)
            request_id = engine.submit(
                TextBatchRequest(
                    prompt=question,
                    images=(image_path,),
                    system_message="",
                )
            )
            prepared[request_id] = (item, question, answer)
            seen_tokens[request_id] = set()
        except Exception as exc:
            failed.append((item["taskId"], str(exc)))

    if not prepared:
        raise RuntimeError("no valid BabyVision requests were prepared")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    generated_tokens = 0
    decode_row_steps = 0
    with output_path.open("a", encoding="utf-8") as output:
        progress = tqdm(total=len(prepared), desc=args.model_name)
        while engine.has_unfinished_requests:
            batch = engine.schedule()
            if batch is None:
                continue
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
                batch.request_ids,
                token_ids.detach().cpu().tolist(),
                strict=True,
            ):
                if int(token) != engine.eos_token_id:
                    seen_tokens[request_id].add(int(token))
            engine.advance(batch, token_ids)

            completed = engine.pop_completed()
            for request_id, result in completed:
                item, question, answer = prepared[request_id]
                record = {
                    "taskId": item["taskId"],
                    "type": item["type"],
                    "subtype": item["subtype"],
                    "ansType": item["ansType"],
                    "question": question,
                    "answer": answer,
                    "model": args.model_name,
                    "model_response": result.text,
                    "extracted_answer": extract_boxed_answer(result.text),
                    "reasoning": "",
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                success_count += 1
                generated_tokens += result.generated_tokens
            if completed:
                output.flush()
                progress.update(len(completed))
                elapsed = time.perf_counter() - started
                progress.set_postfix(
                    active=engine.active_size,
                    waiting=engine.waiting_size,
                    free_kv=engine.free_kv_tokens,
                    tok_s=f"{generated_tokens / elapsed:.1f}",
                )
        progress.close()

    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started

    summary = {
        "backend": engine.decode_backend,
        "output": str(output_path),
        "success_count": success_count,
        "failed_count": len(failed),
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens / inference_seconds,
        "decode_row_steps": decode_row_steps,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "max_kv_tokens": args.max_kv_tokens,
        "free_kv_tokens_at_end": engine.free_kv_tokens,
        "failures": failed[:20],
    }
    print("BABYVISION_NATIVE_BATCH_RESULT=" + json.dumps(summary, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise
