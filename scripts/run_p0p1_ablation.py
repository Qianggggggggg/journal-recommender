#!/usr/bin/env python3
"""P0+P1 dual-track ablation runner.

Diagnostic (2026-06-16): the 4 negative experiments on holdout240 hit a plateau
at 65-66% hit@5 because the LLM evidence extractor under-scored CCF-C
application-oriented journals (HCI, security engineering, education) and
the rule_scorer had a strength-blind CCF multiplier (A=1.05, C=1.0 always).

P0: prompts.yaml adds llm_evidence_extractor_system_v2 / _user_v2 with CCF-tier
    calibration clauses. Selected via ranking.evidence_role.prompt_version.
P1: src/utils/text.py adds quality_adjustment_multiplier(strength, ccf_rating)
    that flips to C>A for strength<0.5. Pipeline._apply_quality_adjustment
    now calls it.

P4: every ablation cell must be run with and without the accepted-paper route
    to disentangle algorithmic gains from corpus leakage signal.

This script is a thin wrapper around scripts/run_evaluation.py that:
  1. Overrides configs/app.yaml::candidate_generator.accepted_paper_weight and
     ::ranking.evidence_role.prompt_version via a temp yaml patch
  2. Optionally re-precomputes the evidence snapshot when prompt_version
     changes (v2 needs a fresh snapshot to take effect at recommend-time)
  3. Runs run_evaluation.py and registers the result to baseline_registry.json
     with a cell-specific label

Usage:
  # Run all 4 cells (resumable; skips cells whose result JSON exists)
  python scripts/run_p0p1_ablation.py --all

  # Speed up with 10 workers (M2.7 typically tolerates 10 concurrent calls;
  # CLAUDE.md default of 1 is for *formal* baseline registration, not for
  # diagnostic ablations. Use --workers 10 to ~10x throughput at the cost
  # of higher rate-limit / format-error risk.)
  python scripts/run_p0p1_ablation.py --all --workers 10

  # Run a single cell
  python scripts/run_p0p1_ablation.py --prompt-version v2 --no-corpus
  python scripts/run_p0p1_ablation.py --prompt-version v2 --with-corpus

  # Re-precompute v2 evidence only (then re-run the cells that use it)
  python scripts/run_p0p1_ablation.py --precompute-only --prompt-version v2

  # Print summary of registered cells
  python scripts/run_p0p1_ablation.py --summary
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_YAML = REPO_ROOT / "configs" / "app.yaml"
REGISTRY_PATH = REPO_ROOT / "data/evaluation/results/baseline_registry.json"
RESULTS_DIR = REPO_ROOT / "data/evaluation/results"
EVIDENCE_DIR = REPO_ROOT / "data/evaluation/evidence"
HOLDOUT_INPUT = REPO_ROOT / "data/evaluation/papers_metadata_holdout240.jsonl"
EVAL_SCRIPT = REPO_ROOT / "scripts/run_evaluation.py"
PRECOMPUTE_SCRIPT = REPO_ROOT / "scripts/precompute_evidence.py"


# ---- 4 ablation cells ----
# Each cell: (prompt_version, accepted_weight, label_suffix)
CELLS = [
    ("v1", 0.20, "v1_with_corpus"),
    ("v1", 0.00, "v1_no_corpus"),
    ("v2", 0.20, "v2_with_corpus"),
    ("v2", 0.00, "v2_no_corpus"),
]


def _find_latest_snapshot_for_version(prompt_version: str) -> Path | None:
    """Find the most recent evidence snapshot for a given prompt version.

    v1 snapshots are stored as ``holdout240_evidence_*.json`` (legacy naming).
    v2 snapshots are stored under ``holdout240_v2_<ts>/holdout240_v2_*.json``
    (new naming from ``_run_precompute``).

    Locked by tests/test_run_p0p1_ablation.py::TestFindLatestSnapshotForVersion.
    """
    if not EVIDENCE_DIR.exists():
        return None
    if prompt_version == "v1":
        candidates = list(EVIDENCE_DIR.glob("holdout240_evidence_*.json"))
        # v2 snapshots live under holdout240_v2_*/ subdirs, exclude those
        candidates = [p for p in candidates if "_v2_" not in p.name]
    else:
        # v2 snapshots live in subdirs named holdout240_v2_<ts>
        subdirs = sorted(p for p in EVIDENCE_DIR.glob(f"holdout240_{prompt_version}_*") if p.is_dir())
        candidates = []
        for subdir in subdirs:
            candidates.extend(subdir.glob(f"holdout240_{prompt_version}_*.json"))
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _ensure_snapshot_for_version(prompt_version: str, workers: int = 10) -> Path:
    """Return an evidence snapshot for ``prompt_version``; precompute v2 if missing.

    Locked by tests/test_run_p0p1_ablation.py::TestEnsureSnapshotForVersion.
    """
    existing = _find_latest_snapshot_for_version(prompt_version)
    if existing is not None:
        return existing
    if prompt_version == "v1":
        raise FileNotFoundError(
            "No v1 evidence snapshot found under data/evaluation/evidence/. "
            "Run scripts/precompute_evidence.py first (default v1)."
        )
    # v2: trigger precompute
    return _run_precompute(prompt_version, workers=workers)


def _patch_app_yaml(
    prompt_version: str,
    accepted_weight: float,
    snapshot_path: Path | None = None,
) -> Path:
    """Backup app.yaml, apply cell overrides, return backup path for restoration.

    Overrides applied:
      1. candidate_generator.accepted_paper_weight
      2. ranking.evidence_role.prompt_version
      3. ranking.evidence_role.snapshot_path (only when ``snapshot_path`` is given)
    """
    import shutil
    backup = APP_YAML.with_suffix(".yaml.bak")
    shutil.copy(APP_YAML, backup)
    text = APP_YAML.read_text(encoding="utf-8")

    def _replace_first_key(text: str, key: str, new_value: str) -> str:
        """Replace the first occurrence of ``key: ...`` with ``new_value``,
        preserving the leading indent of the original line.
        """
        lines = text.splitlines()
        out = []
        replaced = False
        for line in lines:
            if line.strip().startswith(f"{key}:") and not replaced:
                indent = line[: len(line) - len(line.lstrip())]
                out.append(f"{indent}{key}: {new_value}")
                replaced = True
            else:
                out.append(line)
        result = "\n".join(out)
        if text.endswith("\n"):
            result += "\n"
        return result

    text = _replace_first_key(text, "accepted_paper_weight", str(accepted_weight))
    text = _replace_first_key(text, "prompt_version", f"\"{prompt_version}\"")
    if snapshot_path is not None:
        # snapshot_path value is a quoted string in app.yaml
        text = _replace_first_key(text, "snapshot_path", f"\"{snapshot_path}\"")

    APP_YAML.write_text(text, encoding="utf-8")
    return backup


def _restore_app_yaml(backup: Path) -> None:
    import shutil
    if backup.exists():
        shutil.copy(backup, APP_YAML)
        backup.unlink()


def _run_precompute(prompt_version: str, workers: int = 10) -> Path:
    """Run precompute_evidence.py with the given prompt version; return snapshot path."""
    if not PRECOMPUTE_SCRIPT.exists():
        raise FileNotFoundError(f"precompute script not found: {PRECOMPUTE_SCRIPT}")
    if not HOLDOUT_INPUT.exists():
        raise FileNotFoundError(f"holdout input not found: {HOLDOUT_INPUT}")

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = EVIDENCE_DIR / f"holdout240_{prompt_version}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # precompute_evidence.py needs a real eval JSON for --baseline-eval.
    # Pick the most recent eval result that exists on disk.
    eval_candidates = sorted(RESULTS_DIR.glob("eval_abstract_top5_*.json"))
    if not eval_candidates:
        raise FileNotFoundError(
            "No eval_abstract_top5_*.json found under data/evaluation/results/. "
            "Run scripts/run_evaluation.py at least once to seed the precompute."
        )
    baseline_eval = eval_candidates[-1]

    cmd = [
        sys.executable,
        str(PRECOMPUTE_SCRIPT),
        "--benchmark-profile", "custom",
        "--input", str(HOLDOUT_INPUT),
        "--baseline-eval", str(baseline_eval),
        "--mode", "abstract",
        "--workers", str(workers),
        "--output-dir", str(out_dir),
        "--prompt-version", prompt_version,
    ]
    print(f"[precompute {prompt_version}] running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=REPO_ROOT)
    snapshots = sorted(out_dir.glob(f"holdout240_{prompt_version}_*.json"))
    if not snapshots:
        raise FileNotFoundError(f"precompute produced no snapshot under {out_dir}")
    return snapshots[-1]


def _run_eval(prompt_version: str, accepted_weight: float, output_suffix: str, workers: int = 1) -> Path:
    """Run scripts/run_evaluation.py on holdout240; return result JSON path.

    ``workers`` defaults to 1 (CLAUDE.md baseline discipline). Diagnostic
    ablations can pass a higher value (e.g. 10) to trade reliability for
    throughput. M2.7 typically tolerates 10 concurrent LLM calls without
    rate-limit issues.
    """
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_holdout240_p0p1_{output_suffix}_{ts}.json"

    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--benchmark-profile", "holdout240",
        "--mode", "abstract",
        "--top-k", "5",
        "--workers", str(workers),
        "--output", str(out_path),
    ]
    print(f"[eval workers={workers}] running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=REPO_ROOT)
    if not out_path.exists():
        raise FileNotFoundError(f"evaluation produced no output at {out_path}")
    return out_path


def _register(result_path: Path, label: str) -> None:
    cmd = [
        sys.executable,
        "scripts/register_baseline_result.py",
        "--result", str(result_path),
        "--label", label,
    ]
    print(f"[register] {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=REPO_ROOT)


def run_cell(prompt_version: str, accepted_weight: float, *, label_suffix: str,
             workers: int = 1, precompute_workers: int = 10) -> Path:
    """Run a single (prompt_version, accepted_weight) cell end-to-end.

    Steps:
      1. Ensure an evidence snapshot exists for ``prompt_version``
         (auto-precompute v2 snapshot if missing; v1 must already exist)
      2. Patch configs/app.yaml (prompt_version, accepted_paper_weight, snapshot_path)
      3. Run scripts/run_evaluation.py with ``workers`` (default 1)
      4. Register result to baseline_registry.json
      5. Restore configs/app.yaml (always, even on failure)
    """
    snapshot = _ensure_snapshot_for_version(prompt_version, workers=precompute_workers)
    print(f"[cell {label_suffix}] using snapshot: {snapshot}")
    backup = _patch_app_yaml(
        prompt_version, accepted_weight, snapshot_path=snapshot
    )
    try:
        result = _run_eval(
            prompt_version, accepted_weight, label_suffix, workers=workers
        )
        label = f"holdout240_p0p1_{label_suffix}"
        _register(result, label)
        return result
    finally:
        _restore_app_yaml(backup)


def print_summary() -> None:
    """Print comparison table of all 4 p0p1 cells from baseline_registry.json."""
    if not REGISTRY_PATH.exists():
        print("No baseline registry found.")
        return
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    rows = [
        r for r in registry.get("baselines", [])
        if r.get("label", "").startswith("holdout240_p0p1_")
    ]
    if not rows:
        print("No p0p1 cells registered yet.")
        return

    print(f"\n{'label':<40} {'hit@5':>6} {'MRR':>6} {'coarse':>7} {'r20':>4}")
    print("-" * 70)
    for r in rows:
        print(
            f"{r['label']:<40} "
            f"{r.get('hit_at_5', '?'):>6} "
            f"{r.get('mrr', 0):>6.3f} "
            f"{r.get('coarse_hit_count', '?'):>7} "
            f"{r.get('coarse_hit_in_rule_top20_count', '?'):>4}"
        )

    # Side-by-side comparison: v1 vs v2, with vs without corpus
    by_cell = {r["label"].replace("holdout240_p0p1_", ""): r for r in rows}
    if all(c in by_cell for c in ("v1_with_corpus", "v1_no_corpus",
                                    "v2_with_corpus", "v2_no_corpus")):
        print("\n--- 2x2 comparison ---")
        for v in ("v1", "v2"):
            for c in ("with_corpus", "no_corpus"):
                r = by_cell[f"{v}_{c}"]
                print(f"  {v:>2} × {c:<12} hit@5={r['hit_at_5']}/240 "
                      f"({r['hit_at_5']/240*100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P0+P1 dual-track ablation runner on holdout240.",
    )
    parser.add_argument("--prompt-version", choices=["v1", "v2"], help="Evidence prompt version")
    parser.add_argument(
        "--with-corpus", action="store_true",
        help="Set accepted_paper_weight=0.20 (default prod)",
    )
    parser.add_argument(
        "--no-corpus", action="store_true",
        help="Set accepted_paper_weight=0.0 (leakage control)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all 4 cells; skips cells whose labelled result already exists",
    )
    parser.add_argument(
        "--precompute-only", action="store_true",
        help="Only pre-compute evidence snapshot (for v2); do not run eval",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print comparison table of all registered p0p1 cells",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Concurrent LLM workers passed to run_evaluation.py and "
            "precompute_evidence.py. Default 1 (CLAUDE.md baseline discipline). "
            "Diagnostic ablations can use 5-10 for ~Nx throughput at the "
            "cost of slightly higher rate-limit / format-error risk. M2.7 "
            "typically tolerates 10 concurrent calls."
        ),
    )
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    if args.precompute_only:
        if not args.prompt_version:
            parser.error("--precompute-only requires --prompt-version")
        snap = _run_precompute(args.prompt_version)
        print(f"Snapshot: {snap}")
        return

    if args.all:
        for pv, aw, suffix in CELLS:
            label = f"holdout240_p0p1_{suffix}"
            # Skip if already registered
            if REGISTRY_PATH.exists():
                registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
                if any(
                    r.get("label") == label
                    for r in registry.get("baselines", [])
                ):
                    print(f"[skip] {label} already registered")
                    continue
            print(f"\n=== cell: {pv} × accepted_weight={aw} (workers={args.workers}) ===")
            try:
                run_cell(
                    pv, aw, label_suffix=suffix,
                    workers=args.workers, precompute_workers=args.workers,
                )
            except subprocess.CalledProcessError as exc:
                print(f"[fail] cell {suffix} failed: {exc}")
                print("  restoring app.yaml and continuing with next cell")
        print_summary()
        return

    if not (args.prompt_version and (args.with_corpus or args.no_corpus)):
        parser.error("Specify --prompt-version and (--with-corpus|--no-corpus), or use --all/--summary")

    aw = 0.20 if args.with_corpus else 0.00
    suffix = f"{args.prompt_version}_{'with_corpus' if args.with_corpus else 'no_corpus'}"
    result = run_cell(
        args.prompt_version, aw, label_suffix=suffix,
        workers=args.workers, precompute_workers=args.workers,
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
