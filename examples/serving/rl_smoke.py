#!/usr/bin/env python3
"""End-to-end TI2T/TI2TI rollout and policy-update smoke test.

The two-GPU development topology uses Gloo for the checkpoint publisher: a
third NCCL publisher rank cannot share either physical GPU with a serving
consumer.  The serving control plane also supports NCCL unchanged for the
future trainer topology where publisher GPUs are distinct.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import math
import os
import time
from pathlib import Path

import requests
import torch
import torch.distributed as dist
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save

from lightllm.utils.dist_utils import init_custom_process_group
from lightllm.utils.rl_weight_update import tensor_checksum
from client import INTERLEAVE_SYSTEM_PROMPT


def _request(method, url, *, expected=200, timeout=1800, **kwargs):
    response = requests.request(method, url, timeout=timeout, **kwargs)
    if response.status_code != expected:
        raise RuntimeError(f"{method} {url} returned {response.status_code}: {response.text[:1000]}")
    if response.headers.get("content-type", "").startswith("application/json"):
        return response.json()
    return response


def _checkpoint_files(model_path: Path) -> dict[str, Path]:
    index_files = sorted(model_path.glob("*.safetensors.index.json"))
    if index_files:
        index = json.loads(index_files[0].read_text())
        return {name: model_path / filename for name, filename in index["weight_map"].items()}
    result = {}
    for path in sorted(model_path.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            result.update({name: path for name in handle.keys()})
    if not result:
        raise FileNotFoundError(f"no safetensors checkpoint found below {model_path}")
    return result


def _load_tensor(path: Path, name: str) -> torch.Tensor:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _scan_manifest(
    model_path: Path,
    language_closure: set[str],
    vision_closure: set[str],
    x2v_closure: set[str],
):
    locations = _checkpoint_files(model_path)
    entries = []
    smallest = {}
    closures = {
        "language": language_closure,
        "vision": vision_closure,
        "x2v": x2v_closure,
    }
    seen_closures = {consumer: set() for consumer in closures}
    for name, path in sorted(locations.items()):
        owners = [consumer for consumer, closure in closures.items() if name in closure]
        if not owners:
            continue
        if len(owners) != 1:
            raise RuntimeError(f"checkpoint tensor has ambiguous ownership: {name} -> {owners}")
        owner = owners[0]
        seen_closures[owner].add(name)
        tensor = _load_tensor(path, name)
        entry = {
            "name": name,
            "path": str(path),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "checksum": tensor_checksum(tensor),
            "owner": owner,
            "numel": tensor.numel(),
            "element_size": tensor.element_size(),
        }
        entries.append(entry)
        if tensor.is_floating_point() and (owner not in smallest or tensor.numel() < smallest[owner]["numel"]):
            smallest[owner] = entry
    missing = {
        consumer: sorted(closure - seen_closures[consumer])
        for consumer, closure in closures.items()
    }
    if any(missing.values()):
        raise RuntimeError(f"checkpoint closure mismatch: {missing}")
    if set(smallest) != {"language", "vision", "x2v"}:
        raise RuntimeError(f"could not select one floating smoke tensor per consumer: {sorted(smallest)}")
    return entries, smallest


def _init_publish_groups(base_url: str, port: int, group_name: str):
    ports = {"language": port, "vision": port + 1, "x2v": port + 2}
    body = {
        "master_address": "127.0.0.1",
        "master_port": port,
        "master_ports": ports,
        "world_size": 4,
        "group_name": group_name,
        "backend": "gloo",
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(_request, "POST", f"{base_url}/init_weights_update_group", json=body)
        groups = {}
        for consumer, consumer_port in ports.items():
            groups[consumer] = init_custom_process_group(
                backend="gloo",
                init_method=f"tcp://127.0.0.1:{consumer_port}",
                world_size=2,
                rank=0,
                group_name=f"{group_name}:{consumer}",
            )
        receipt = pending.result()
    return groups, receipt


def _build_buckets(entries, max_bytes=256 * 1024 * 1024):
    buckets = []
    current = []
    current_dtype = None
    current_bytes = 0
    for index, entry in enumerate(entries):
        element_size = _load_tensor(Path(entry["path"]), entry["name"]).element_size()
        size = entry["numel"] * element_size
        if current and (entry["dtype"] != current_dtype or current_bytes + size > max_bytes):
            buckets.append(current)
            current, current_bytes = [], 0
        current_dtype = entry["dtype"]
        current.append(index)
        current_bytes += size
    if current:
        buckets.append(current)
    result = []
    for bucket_index, indices in enumerate(buckets):
        flat = torch.cat(
            [_load_tensor(Path(entries[index]["path"]), entries[index]["name"]).reshape(-1) for index in indices]
        ).contiguous()
        result.append(
            {
                "id": f"bucket-{bucket_index:05d}",
                "dtype": str(flat.dtype),
                "numel": flat.numel(),
                "entry_indices": indices,
                "checksum": tensor_checksum(flat),
            }
        )
    return result


def _manifest_payload(entries, buckets, policy_version, group_name):
    return {
        "names": [entry["name"] for entry in entries],
        "dtypes": [entry["dtype"] for entry in entries],
        "shapes": [entry["shape"] for entry in entries],
        "checksums": [entry["checksum"] for entry in entries],
        "assignments": {entry["name"]: [entry["owner"]] for entry in entries},
        "required": {
            consumer: [entry["name"] for entry in entries if entry["owner"] == consumer]
            for consumer in ("language", "vision", "x2v")
        },
        "policy_version": policy_version,
        "group_name": group_name,
        "buckets": buckets,
        "full_update": True,
    }


def _publish_full_checkpoint(base_url, groups, entries, buckets, policy_version, group_name):
    payload = _manifest_payload(entries, buckets, policy_version, group_name)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            _request,
            "POST",
            f"{base_url}/update_weights_from_distributed",
            json=payload,
            timeout=7200,
        )
        for group in groups.values():
            for bucket in buckets:
                tensor = torch.cat(
                    [
                        _load_tensor(Path(entries[index]["path"]), entries[index]["name"]).reshape(-1)
                        for index in bucket["entry_indices"]
                    ]
                ).contiguous()
                dist.broadcast(tensor, src=0, group=group)
        receipt = pending.result()
    return receipt, time.perf_counter() - started


def _closure_from_init(receipt):
    language = receipt["receipts"]["language"]["ranks"][0]["closure_names"]
    vision = receipt["receipts"]["vision"]["ranks"][0]["closure_names"]
    x2v = receipt["receipts"]["x2v"]["closure_names"]
    return set(language), set(vision), set(x2v)


def _verify_update_receipts(response, expected_checksums, assignments):
    receipts = response["receipts"]
    consumer_receipts = {
        "language": receipts["language"]["ranks"],
        "vision": receipts["vision"]["ranks"],
        "x2v": [receipts["x2v"]],
    }
    for consumer, rank_receipts in consumer_receipts.items():
        expected = {
            name: checksum
            for name, checksum in expected_checksums.items()
            if consumer in assignments[name]
        }
        for rank_receipt in rank_receipts:
            actual = rank_receipt.get("checksums", {})
            if actual != expected:
                raise AssertionError(
                    f"{consumer} transport checksum ACK mismatch: "
                    f"expected={len(expected)}, actual={len(actual)}"
                )
    return response


def _assert_text_event(event, tokenizer):
    if event["type"] != "text" or not event["token_ids"]:
        raise AssertionError("invalid text event")
    if len(event["token_ids"]) != len(event["selected_token_logprobs"]):
        raise AssertionError("token/logprob lengths differ")
    if not all(math.isfinite(value) for value in event["selected_token_logprobs"]):
        raise AssertionError("non-finite token logprob")
    decoded = tokenizer.decode(event["token_ids"], skip_special_tokens=False)
    if event.get("decoded_tokens") != decoded:
        raise AssertionError("token decode round-trip differs from the serving tokenizer")


def _check_trace(path: Path):
    tensors = load_file(str(path))
    required = {
        "samples",
        "next_samples",
        "old_means",
        "old_log_probs",
        "timesteps",
        "next_timesteps",
        "scales",
        "indices",
        "final_latent",
    }
    if not required <= set(tensors):
        raise AssertionError(f"trace is missing {sorted(required - set(tensors))}")
    log_prob = (
        -((tensors["next_samples"].float() - tensors["old_means"].float()).square())
        / (2 * tensors["scales"].float().square())
        - torch.log(tensors["scales"].float())
        - 0.5 * torch.log(torch.tensor(2.0 * torch.pi))
    ).mean(dim=tuple(range(2, tensors["samples"].ndim)))
    torch.testing.assert_close(log_prob, tensors["old_log_probs"].float(), atol=1e-5, rtol=1e-5)
    if not all(bool(torch.isfinite(value).all()) for value in tensors.values() if value.is_floating_point()):
        raise AssertionError("trace contains non-finite values")
    return {name: list(value.shape) for name, value in tensors.items()}


def _tensor_bundle(entries, mutate=False):
    tensors = {}
    assignments = {}
    originals = {}
    for consumer, entry in entries.items():
        tensor = _load_tensor(Path(entry["path"]), entry["name"]).contiguous()
        originals[entry["name"]] = tensor.clone()
        if mutate:
            tensor = tensor.clone()
            flat = tensor.view(-1)
            flat[0] = torch.nextafter(flat[0], torch.full_like(flat[0], float("inf")))
        tensors[entry["name"]] = tensor
        assignments[entry["name"]] = [consumer]
    return tensors, originals, assignments


def _post_tensor_update(base_url, tensors, assignments, version, required=None, checksums=None, expected=200):
    checksums = checksums or {name: tensor_checksum(tensor) for name, tensor in tensors.items()}
    body = {
        "serialized_safetensors": base64.b64encode(save(tensors)).decode("ascii"),
        "checksums": checksums,
        "assignments": assignments,
        "required": required or {
            consumer: [name for name, owners in assignments.items() if consumer in owners]
            for consumer in ("language", "vision", "x2v")
        },
        "policy_version": version,
        "full_update": False,
    }
    return _request("POST", f"{base_url}/update_weights_from_tensor", json=body, expected=expected)


def _assert_failed_barrier(base_url, active_version):
    status = _request("GET", f"{base_url}/v1/rl/status")
    if not status["paused"] or status["active_policy_version"] != active_version:
        raise AssertionError("failed update crossed the policy barrier")
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="/tmp/mova-u15-rl-smoke")
    parser.add_argument("--publisher-port", type=int, default=29680)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = {"started_at": time.time(), "transport": "gloo-cpu-two-gpu-smoke", "stages": {}}

    receipt["health"] = _request("GET", f"{base_url}/health")
    receipt["initial_status"] = _request("GET", f"{base_url}/v1/rl/status")
    # Tokenizer registration imports GPU-specific LightLLM model modules. Keep
    # it lazy so CLI/help and static smoke validation work in the CPU builder.
    from lightllm.server.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(str(Path(args.model_path)), "auto", trust_remote_code=True)
    groups, init_receipt = _init_publish_groups(base_url, args.publisher_port, "smoke-update")
    receipt["stages"]["init_groups"] = init_receipt
    language_closure, vision_closure, x2v_closure = _closure_from_init(init_receipt)
    entries, smallest = _scan_manifest(
        Path(args.model_path), language_closure, vision_closure, x2v_closure
    )
    buckets = _build_buckets(entries)
    receipt["manifest"] = {
        "tensor_count": len(entries),
        "bytes": sum(entry["numel"] * entry["element_size"] for entry in entries),
        "consumer_counts": {
            consumer: sum(entry["owner"] == consumer for entry in entries)
            for consumer in ("language", "vision", "x2v")
        },
        "bucket_count": len(buckets),
    }
    update_receipt, seconds = _publish_full_checkpoint(
        base_url, groups, entries, buckets, "smoke-v1", "smoke-update"
    )
    _verify_update_receipts(
        update_receipt,
        {entry["name"]: entry["checksum"] for entry in entries},
        {entry["name"]: [entry["owner"]] for entry in entries},
    )
    receipt["stages"]["full_weight_sync"] = {"seconds": seconds, "receipt": update_receipt}

    ti2t_request = {
        "expected_policy_version": "smoke-v1",
        "modality": "ti2t",
        "messages": [{"role": "user", "content": "What is 17 + 25? Answer briefly."}],
        "seeds": [11, 12],
        "max_new_tokens": 256,
        "max_images": 0,
    }
    ti2t = _request("POST", f"{base_url}/v1/rl/rollouts", json=ti2t_request)
    for rollout in ti2t["rollouts"]:
        for event in rollout["events"]:
            _assert_text_event(event, tokenizer)
        if any(event["type"] == "image" for event in rollout["events"]):
            raise AssertionError("TI2T exposed an image action")
    receipt["stages"]["ti2t"] = {"rollouts": len(ti2t["rollouts"]), "usage": [r["usage"] for r in ti2t["rollouts"]]}

    ti2ti_request = {
        "expected_policy_version": "smoke-v1",
        "modality": "ti2ti",
        "messages": [
            {"role": "system", "content": INTERLEAVE_SYSTEM_PROMPT},
            {"role": "user", "content": "Create an image of a red cube on a white table, then describe it."},
        ],
        "seeds": [21, 22],
        "max_new_tokens": 1024,
        "max_images": 1,
        "image_policy": {
            "height": 512,
            "width": 512,
            "image_steps": 8,
            "timestep_shift": 3.0,
            "t_eps": 0.02,
            "image_noise_level": 0.7,
            "sde_window_start": 0,
            "sde_window_end": 8,
            "sde_selected_steps": 4,
        },
    }
    ti2ti = _request("POST", f"{base_url}/v1/rl/rollouts", json=ti2ti_request, timeout=3600)
    bundle_ids = []
    interleaved = False
    trace_shapes = {}
    for rollout in ti2ti["rollouts"]:
        types = [event["type"] for event in rollout["events"]]
        interleaved = interleaved or any(types[index : index + 3] == ["text", "image", "text"] for index in range(max(0, len(types) - 2)))
        for event in rollout["events"]:
            if event["type"] == "text":
                _assert_text_event(event, tokenizer)
            else:
                bundle_ids.append(event["trace_bundle_key"])
                image_data = base64.b64decode(event["image"].split(",", 1)[1])
                image_path = output_dir / f"{event['trace_bundle_key']}.jpg"
                image_path.write_bytes(image_data)
                Image.open(image_path).verify()
                trace_path = output_dir / f"{event['trace_bundle_key']}.safetensors"
                response = _request("GET", f"{base_url}/v1/rl/traces/{event['trace_bundle_key']}")
                trace_path.write_bytes(response.content)
                trace_shapes[event["trace_bundle_key"]] = _check_trace(trace_path)
    if not interleaved:
        raise AssertionError("no TI2TI rollout completed text->image->text")
    receipt["stages"]["ti2ti"] = {"rollouts": len(ti2ti["rollouts"]), "trace_shapes": trace_shapes}

    stale = dict(ti2t_request, expected_policy_version="stale")
    _request("POST", f"{base_url}/v1/rl/rollouts", json=stale, expected=409)

    changed, originals, assignments = _tensor_bundle(smallest, mutate=True)
    _request("POST", f"{base_url}/pause_generation")
    controlled_receipt = _post_tensor_update(
        base_url, changed, assignments, "smoke-controlled"
    )
    receipt["stages"]["controlled_change"] = _verify_update_receipts(
        controlled_receipt,
        {name: tensor_checksum(tensor) for name, tensor in changed.items()},
        assignments,
    )
    _request("POST", f"{base_url}/pause_generation")
    restore_v2 = _post_tensor_update(
        base_url, originals, assignments, "smoke-v2"
    )
    receipt["stages"]["restore_v2"] = _verify_update_receipts(
        restore_v2,
        {name: tensor_checksum(tensor) for name, tensor in originals.items()},
        assignments,
    )

    # Failure protection: every failure must leave the active version intact
    # and the server paused until a valid restore is committed.
    _request("POST", f"{base_url}/pause_generation")
    bad_checksums = {name: tensor_checksum(tensor) for name, tensor in originals.items()}
    first_name = next(iter(bad_checksums))
    bad_checksums[first_name] = "0" * 64
    _post_tensor_update(base_url, originals, assignments, "bad-checksum", checksums=bad_checksums, expected=409)
    receipt["stages"]["bad_checksum"] = _assert_failed_barrier(base_url, "smoke-v2")
    receipt["stages"]["restore_v3"] = _post_tensor_update(
        base_url, originals, assignments, "smoke-v3"
    )

    # Missing closure tensor: advertise the complete controlled closure while
    # omitting one tensor from the bundle.
    _request("POST", f"{base_url}/pause_generation")
    omitted_name = next(iter(originals))
    subset = {name: tensor for name, tensor in originals.items() if name != omitted_name}
    subset_assignments = {name: assignments[name] for name in subset}
    required = {
        consumer: [name for name, owners in assignments.items() if consumer in owners]
        for consumer in ("language", "vision", "x2v")
    }
    _post_tensor_update(
        base_url,
        subset,
        subset_assignments,
        "bad-missing",
        required=required,
        expected=409,
    )
    receipt["stages"]["missing_tensor"] = _assert_failed_barrier(base_url, "smoke-v3")
    receipt["stages"]["restore_v4"] = _post_tensor_update(
        base_url, originals, assignments, "smoke-v4"
    )

    # Correct bytes/checksum with a wrong model shape must fail at the owning
    # consumer rather than being accepted by the transport layer.
    _request("POST", f"{base_url}/pause_generation")
    malformed = {name: tensor.clone() for name, tensor in originals.items()}
    shape_name = smallest["x2v"]["name"]
    source = malformed[shape_name]
    malformed[shape_name] = (
        source.reshape(-1) if source.ndim != 1 else source.reshape(source.numel(), 1)
    )
    _post_tensor_update(
        base_url, malformed, assignments, "bad-shape", expected=409
    )
    receipt["stages"]["wrong_shape"] = _assert_failed_barrier(base_url, "smoke-v4")
    receipt["stages"]["final_restore"] = _post_tensor_update(
        base_url, originals, assignments, "smoke-v5"
    )

    for bundle_id in bundle_ids:
        _request("DELETE", f"{base_url}/v1/rl/traces/{bundle_id}")
        _request("GET", f"{base_url}/v1/rl/traces/{bundle_id}", expected=404)

    destroy_body = {"group_name": "smoke-update"}
    receipt["stages"]["destroy_groups"] = _request(
        "POST", f"{base_url}/destroy_weights_update_group", json=destroy_body
    )
    for group in groups.values():
        dist.destroy_process_group(group)
    receipt["final_status"] = _request("GET", f"{base_url}/v1/rl/status")
    receipt["finished_at"] = time.time()
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False))
    print(receipt_path)


if __name__ == "__main__":
    main()
