"""Deterministic NDJSON reward provider used only by Forge integration tests."""

from __future__ import annotations

import json
import sys


def score(request: dict[str, object]) -> dict[str, object]:
    rollouts = request.get("rollouts")
    group = request.get("rollout_group_key")
    if not isinstance(rollouts, list) or not isinstance(group, str):
        raise ValueError("invalid Forge reward request")
    count = len(rollouts)
    return {
        "schema": "sensenova.u15.forge.reward.response.v1",
        "reward": {
            "matrix": [[float(index % 2)] for index in range(count)],
            "dimension_names": ["synthetic_correct"],
            "availability": [[True] for _ in range(count)],
            "group_ids": [group for _ in range(count)],
            "errors": [None for _ in range(count)],
            "diagnostics": [{} for _ in range(count)],
        },
        "budget": {},
    }


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            response = score(json.loads(line))
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
            raise
        print(json.dumps(response, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
