#!/usr/bin/env python3
"""Register compact baseline metrics from an evaluation result JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY_PATH = Path("data/evaluation/results/baseline_registry.json")


def build_baseline_record(result_path: str | Path, label: str) -> dict[str, Any]:
    path = Path(result_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    manifest = data.get("benchmark_manifest") or {}
    metrics = data.get("metrics") or {}
    if not manifest:
        raise ValueError(f"evaluation result has no benchmark_manifest: {path}")

    return {
        "label": label,
        "result_path": str(path),
        "input_path": manifest.get("input_path"),
        "hit_at_5": metrics.get("hit_at_5"),
        "mrr": metrics.get("mrr"),
        "coarse_hit_count": metrics.get("coarse_hit_count"),
        "coarse_hit_in_rule_top20_count": metrics.get("coarse_hit_in_rule_top20_count"),
        "acceptable_journal_hit_at_5": metrics.get("acceptable_journal_hit_at_5"),
        "app_config_hash": manifest.get("app_config_hash"),
        "prompt_hash": manifest.get("prompt_hash"),
        "minimax_model": manifest.get("minimax_model"),
    }


def register_baseline_record(
    record: dict[str, Any],
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    replace: bool = False,
) -> dict[str, Any]:
    path = Path(registry_path)
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
    else:
        registry = {"baselines": []}

    baselines = registry.setdefault("baselines", [])
    existing_idx = next(
        (idx for idx, item in enumerate(baselines) if item.get("label") == record.get("label")),
        None,
    )
    if existing_idx is not None:
        if not replace:
            raise ValueError(f"baseline label already exists: {record.get('label')}")
        baselines[existing_idx] = record
    else:
        baselines.append(record)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Register an evaluation result as a baseline.")
    parser.add_argument("--result", required=True, help="Path to evaluation result JSON")
    parser.add_argument("--label", required=True, help="Unique baseline label")
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Baseline registry path",
    )
    parser.add_argument("--replace", action="store_true", help="Replace an existing label")
    args = parser.parse_args()

    record = build_baseline_record(args.result, args.label)
    register_baseline_record(record, registry_path=args.registry, replace=args.replace)
    print(f"Registered baseline: {args.label}")
    print(f"Registry: {args.registry}")


if __name__ == "__main__":
    main()
