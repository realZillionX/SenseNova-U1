"""Compare native dense and paged Flash KV decode on one BabyVision row."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

from sensenova_u1 import set_attn_backend
from sensenova_u1.batch_inference import (
    ContiguousTextBatchSession,
    ContinuousTextBatchEngine,
    TextBatchRequest,
    _apply_repetition_penalty,
)
from sensenova_u1.models.neo_unify.conversation import get_conv_template
from sensenova_u1.utils import load_model_and_tokenizer


BABYVISION_DIR = (
    Path(__file__).resolve().parents[1] / "evaluation" / "interleave" / "BabyVision"
)
sys.path.insert(0, str(BABYVISION_DIR))
from infer_babyvision_native_batch import _prepare_item  # noqa: E402


def _select(
    logits: torch.Tensor,
    seen_tokens: set[int],
    repetition_penalty: float,
) -> tuple[int, float]:
    if repetition_penalty != 1.0:
        seen = torch.zeros_like(logits, dtype=torch.bool)
        if seen_tokens:
            indexes = torch.tensor(
                tuple(seen_tokens), device=logits.device, dtype=torch.long
            )
            seen[0, indexes] = True
        logits = _apply_repetition_penalty(logits, seen, repetition_penalty)
    top = torch.topk(logits[0], k=2)
    return int(top.indices[0].item()), float((top.values[0] - top.values[1]).item())


@torch.no_grad()
def _run_dense(model, tokenizer, request, *, limit: int, penalty: float):
    session = ContiguousTextBatchSession(
        model,
        tokenizer,
        (request,),
        device="cuda",
        dtype=torch.bfloat16,
        prefix_sharing=False,
        flash_decode_tokens=limit,
    )
    prefix_length = int(session.cache.get_seq_length())
    eos = int(
        tokenizer.convert_tokens_to_ids(get_conv_template(model.template).sep.strip())
    )
    generated = []
    margins = []
    seen = set()
    started = time.perf_counter()
    for _ in range(limit):
        token, margin = _select(session.constrained_logits(), seen, penalty)
        margins.append(margin)
        if token == eos:
            reason = "eos"
            break
        generated.append(token)
        seen.add(token)
        if len(generated) == limit:
            reason = "max_new_tokens"
            break
        session.commit(
            torch.tensor([token], device="cuda", dtype=torch.long),
            torch.ones(1, device="cuda", dtype=torch.bool),
        )
    elapsed = time.perf_counter() - started
    return generated, margins, reason, elapsed, prefix_length


@torch.no_grad()
def _run_paged(
    model, tokenizer, request, *, limit: int, penalty: float, page_size: int
):
    engine = ContinuousTextBatchEngine(
        model,
        tokenizer,
        device="cuda",
        dtype=torch.bfloat16,
        max_batch_size=1,
        max_model_len=65536,
        prefill_chunk_size=2048,
        default_max_new_tokens=limit,
        max_kv_tokens=131072,
        kv_page_size=page_size,
        prefix_sharing=False,
    )
    request_id = engine.submit(request)
    generated = []
    margins = []
    seen = set()
    result = None
    started = time.perf_counter()
    while engine.has_unfinished_requests:
        batch = engine.schedule()
        if batch is None:
            continue
        token, margin = _select(batch.logits, seen, penalty)
        margins.append(margin)
        if token != engine.eos_token_id:
            generated.append(token)
            seen.add(token)
        engine.advance(
            batch,
            torch.tensor([token], device="cuda", dtype=torch.long),
        )
        for completed_id, completed in engine.pop_completed():
            if completed_id == request_id:
                result = completed
    elapsed = time.perf_counter() - started
    assert result is not None
    return generated, margins, result.finish_reason, elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--task-id", type=int, default=5133)
    parser.add_argument("--limit", type=int, default=1024)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    args = parser.parse_args()

    item = next(
        (
            row
            for row in map(json.loads, Path(args.data).open(encoding="utf-8"))
            if int(row["taskId"]) == args.task_id
        ),
        None,
    )
    if item is None:
        raise ValueError(f"BabyVision task {args.task_id} was not found")
    question, _answer, image_path = _prepare_item(item, Path(args.image_root))
    request = TextBatchRequest(
        prompt=question,
        images=(image_path,),
        system_message="",
    )

    set_attn_backend("flash")
    model, tokenizer = load_model_and_tokenizer(
        args.model, dtype=torch.bfloat16, device="cuda"
    )
    dense = _run_dense(
        model,
        tokenizer,
        request,
        limit=args.limit,
        penalty=args.repetition_penalty,
    )
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    paged = _run_paged(
        model,
        tokenizer,
        request,
        limit=args.limit,
        penalty=args.repetition_penalty,
        page_size=args.page_size,
    )
    torch.cuda.synchronize()

    dense_tokens, dense_margins, dense_reason, dense_seconds, prefix_length = dense
    paged_tokens, paged_margins, paged_reason, paged_seconds = paged
    common = 0
    while (
        common < len(dense_tokens)
        and common < len(paged_tokens)
        and dense_tokens[common] == paged_tokens[common]
    ):
        common += 1
    report = {
        "task_id": args.task_id,
        "limit": args.limit,
        "page_size": args.page_size,
        "prefix_length": prefix_length,
        "prefix_mod_page": prefix_length % args.page_size,
        "common_token_prefix": common,
        "dense_tokens": len(dense_tokens),
        "paged_tokens": len(paged_tokens),
        "dense_reason": dense_reason,
        "paged_reason": paged_reason,
        "dense_seconds": dense_seconds,
        "paged_seconds": paged_seconds,
        "dense_divergence_token": (
            dense_tokens[common] if common < len(dense_tokens) else None
        ),
        "paged_divergence_token": (
            paged_tokens[common] if common < len(paged_tokens) else None
        ),
        "dense_margin_at_divergence": (
            dense_margins[common] if common < len(dense_margins) else None
        ),
        "paged_margin_at_divergence": (
            paged_margins[common] if common < len(paged_margins) else None
        ),
        "dense_text_tail": tokenizer.decode(
            dense_tokens[-128:], skip_special_tokens=True
        ),
        "paged_text_tail": tokenizer.decode(
            paged_tokens[-128:], skip_special_tokens=True
        ),
    }
    print("DENSE_PAGED_PRECISION_RESULT=" + json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
