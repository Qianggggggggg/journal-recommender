#!/usr/bin/env python3
"""Pre-pass: extract LLM evidence for every paper ONCE, save snapshot for reuse.

Task 6.3 Fix #1 — evidence snapshot sharing.

Why this exists
---------------
``run_llm_role_ablation.py`` runs three LLM-role variants per paper. Without a
pre-pass, each evidence variant (rule, learned) calls the LLM independently,
producing two different evidence scores (LLM non-determinism). Comparing the
two variants is then unfair because the "evidence" itself differs.

This script runs candidate generation + rule scoring + LTR rerank (when
enabled) + LLM evidence extraction exactly once per paper, then writes a
JSON snapshot keyed by paper key. The ablation runner reads the snapshot
and feeds identical evidence to every evidence variant, making the
rule-vs-learned comparison causally interpretable.

Output schema
-------------
``data/evaluation/evidence/<benchmark>_<timestamp>.json``::

    {
      "schema_version": 1,
      "benchmark_profile": "light30",
      "benchmark_path": "...",
      "baseline_eval": "...",
      "mode": "abstract",
      "papers": {
        "<paper_key>": {
          "title": "...",
          "venue": "...",
          "rule_ranks": {"jid1": 1, "jid2": 2, ...},
          "learned_ranks": {"jid1": 3, "jid2": 1, ...} | null,
          "candidates": [
            {"journal_id": "...", "rule_score": 0.9, "reasons": [...]},
            ...
          ],
          "evidence": {
            "jid1": {"scope_fit": 0.7, ..., "evidence": ["..."]},
            ...
          },
          "evidence_coverage": 1.0,
          "status": "ok" | "neutral_fallback",
          "fallback_reason": "",
        },
        ...
      }
    }

Coverage requirement
--------------------
The script FAILS (exit code 2) if any paper has ``evidence_coverage < 1.0``
(i.e., any candidate is missing a non-neutral evidence entry). Run with
``--allow-partial`` to relax this for debugging.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from datetime import datetime
from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.run_evaluation import (  # noqa: E402
    attach_baseline_profile_snapshots,
    init_pipeline,
    paper_profile_from_snapshot,
    resolve_benchmark_input,
)
from scripts.run_retrieval_ablation import (  # noqa: E402
    load_comparable_eval_papers,
    paper_profile_from_metadata,
)
from src.ranker.llm_evidence_extractor import (  # noqa: E402
    LLMEvidenceExtractor,
    LLMEvidenceExtractorError,
)

logger = logging.getLogger("precompute_evidence")

SCHEMA_VERSION = 1


def _paper_key(title: str, venue: str) -> str:
    t = " ".join(str(title or "").casefold().split())
    v = " ".join(str(venue or "").casefold().split())
    return f"{t} | {v}"


def select_snapshot_candidates(rec_result: dict) -> list:
    """Return the exact candidate pool that the LLM-role variants will rank."""
    return list(rec_result.get("llm_candidates") or [])


# Module-level print lock for --workers > 1 to avoid interleaved progress
# output. (Created lazily in main() to keep this module import-safe.)
_PRINT_LOCK = None


def _safe_print(*args, **kwargs):
    """Thread-safe print that uses a module-level lock when workers > 1."""
    if _PRINT_LOCK is not None:
        with _PRINT_LOCK:
            print(*args, **kwargs)
            return
    print(*args, **kwargs)


def _process_one_paper(
    paper: dict,
    pkey: str,
    prior_entry: Optional[dict],
    pipeline,
    extractor: LLMEvidenceExtractor,
    show_progress: bool,
) -> tuple:
    """Process a single paper: full extract + up to 2 focused retry rounds.

    Returns (pkey, entry, copied_increment, retried_increment, failed_increment).
    """
    title = paper.get("title", "")
    venue = paper.get("venue", "")

    # Incremental repair: copy complete papers verbatim from prior snapshot.
    if prior_entry is not None and prior_entry.get("evidence_coverage", 0.0) >= 1.0:
        if show_progress:
            _safe_print(
                f"  ↻ {title[:50]:50s} | copied from prior (coverage=100%)"
            )
        return pkey, dict(prior_entry), 1, 0, 0

    # Full extract for the paper (or for the missing subset on retry).
    try:
        entry = _extract_evidence_for_paper(
            paper=paper,
            pipeline=pipeline,
            extractor=extractor,
            show_progress=show_progress,
        )
    except Exception as exc:
        logger.warning(
            "[%s] evidence pre-pass failed entirely: %s",
            title[:40], exc,
        )
        entry = {
            "title": title,
            "venue": venue,
            "rule_ranks": {},
            "learned_ranks": None,
            "candidates": [],
            "evidence": {},
            "evidence_coverage": 0.0,
            "status": "pre_pass_error",
            "fallback_reason": f"{type(exc).__name__}: {exc}"[:500],
        }
        return pkey, entry, 0, 0, 1

    retried_local = 0

    # Incremental repair: merge with prior evidence, then up to 2
    # focused retry rounds targeting only the still-missing candidates.
    if prior_entry is not None and prior_entry.get("evidence"):
        merged = {**prior_entry.get("evidence", {}), **entry["evidence"]}
        entry["evidence"] = merged
        cand_ids = {c["journal_id"] for c in entry["candidates"]}
        entry["evidence_coverage"] = (
            len(merged) / len(cand_ids) if cand_ids else 1.0
        )

    # Focused retry rounds: max 2 attempts on the still-missing IDs.
    for round_idx in (1, 2):
        if entry.get("evidence_coverage", 0.0) >= 1.0:
            break
        cand_ids = {c["journal_id"] for c in entry["candidates"]}
        missing = sorted(cand_ids - set(entry["evidence"].keys()))
        if not missing:
            break
        try:
            # Use paper_profile_from_metadata (accepts paper dict directly,
            # falls back to paper_profile_snapshot internally if present).
            # This works whether or not the source ablation JSON included
            # a paper_profile_snapshot field.
            profile = paper_profile_from_metadata(paper)
            focused_candidates = _build_focused_candidates(
                entry, missing, pipeline
            )
            new_ev = extractor.extract_focused(
                focused_candidates,
                profile,
                focus_journal_ids=missing,
                already_covered_ids=sorted(entry["evidence"].keys()),
                rule_ranks=entry.get("rule_ranks", {}),
            )
            entry["evidence"].update(new_ev)
            entry["evidence_coverage"] = (
                len(entry["evidence"]) / len(cand_ids) if cand_ids else 1.0
            )
            retried_local += 1
            if show_progress:
                _safe_print(
                    f"     round {round_idx}: +{len(new_ev)} new "
                    f"→ coverage={entry['evidence_coverage']:.0%}"
                )
        except Exception as exc:
            logger.warning(
                "[%s] focused round %s failed: %s",
                title[:40], round_idx, exc,
            )
            if show_progress:
                _safe_print(
                    f"     round {round_idx}: failed ({type(exc).__name__})"
                )
            break  # don't retry if extractor raised

    if entry.get("evidence_coverage", 0.0) >= 1.0:
        if entry.get("status") not in ("pre_pass_error",):
            entry["status"] = "ok_repaired" if prior_entry else "ok"
    return pkey, entry, 0, retried_local, 0


def _extract_evidence_for_paper(
    paper: dict,
    pipeline,
    extractor: LLMEvidenceExtractor,
    show_progress: bool = True,
) -> dict:
    """Run one paper through candidate gen + rule scoring + evidence extraction."""
    title = paper.get("title", "")
    venue = paper.get("venue", "")
    # Build PaperProfile directly from paper fields. We previously required
    # a paper_profile_snapshot field, but ablation JSONs written by
    # run_retrieval_ablation.py don't always include one. The
    # paper_profile_from_metadata helper accepts a paper dict and falls
    # back to a snapshot if present.
    profile = paper_profile_from_metadata(paper)

    from src.papers.paper_model import PaperInput
    paper_input = PaperInput(
        title=title,
        abstract=paper.get("abstract", ""),
        mode="abstract",
    )

    # Fix A: precompute vs ablation candidate-set alignment.
    # run_evaluation.py passes ``diagnostic_journal_ids=[target_journal.jid]``
    # so the gold venue is force-included in candidates. Mirror that here so
    # the snapshot's candidate set is bit-identical to the ablation runner's.
    target_journal = _find_journal_by_venue(venue, pipeline)
    diagnostic_ids = (
        [target_journal.journal_id] if target_journal is not None else None
    )

    # LLM ranker is not needed here; we only need candidate gen + rule scoring.
    # Temporarily disable the LLM ranker so recommend() returns at the rule step.
    saved_ranker = pipeline.llm_ranker
    pipeline.llm_ranker = None
    try:
        rec_result = pipeline.recommend(
            paper_input,
            profile,
            top_k=5,
            mode="abstract",
            diagnostic_journal_ids=diagnostic_ids,
        )
    finally:
        pipeline.llm_ranker = saved_ranker

    rule_ranked = rec_result.get("rule_ranked") or []
    retrieval_trace = rec_result.get("retrieval_trace") or {}
    learned_diag = rec_result.get("learned_diagnostics") or {}

    rule_ranks = {
        jid: idx + 1
        for idx, (j, _s, _r) in enumerate(rule_ranked)
        for jid in [j.journal_id]
    }
    rule_scores = {
        j.journal_id: float(s)
        for j, s, _r in rule_ranked
    }
    learned_ranks = (
        dict(learned_diag.get("learned_rank") or {})
        if learned_diag.get("status") == "ok"
        else None
    )

    # Extract evidence only for the exact LLM candidate pool. The pool may be
    # LTR-ordered, but the extractor receives explicit real Rule ranks.
    candidates = [
        (j, s, list(r or []))
        for j, s, r in select_snapshot_candidates(rec_result)
    ]

    if not candidates:
        return {
            "title": title,
            "venue": venue,
            "rule_ranks": {},
            "learned_ranks": learned_ranks,
            "candidates": [],
            "evidence": {},
            "evidence_coverage": 1.0,
            "status": "ok",
            "fallback_reason": "",
        }

    try:
        evidence_by_id = extractor.extract(
            candidates, profile, rule_ranks=rule_ranks
        )
        status = "ok"
        fallback_reason = ""
    except LLMEvidenceExtractorError as exc:
        logger.warning(
            "[%s] evidence extraction failed: %s", title[:40], exc
        )
        evidence_by_id = {}
        status = "neutral_fallback"
        fallback_reason = f"{type(exc).__name__}: {exc}"[:500]
    except Exception as exc:
        logger.warning(
            "[%s] evidence extraction crashed: %s", title[:40], exc
        )
        evidence_by_id = {}
        status = "neutral_fallback"
        fallback_reason = f"{type(exc).__name__}: {exc}"[:500]

    coverage = (
        len(evidence_by_id) / len(candidates) if candidates else 1.0
    )

    if show_progress:
        print(
            f"  ✓ {title[:50]:50s} | candidates={len(candidates):2d} | "
            f"coverage={coverage:.0%} | status={status}"
        )

    return {
        "title": title,
        "venue": venue,
        "rule_ranks": rule_ranks,
        "rule_scores": rule_scores,
        "learned_ranks": learned_ranks,
        "candidates": [
            {
                "journal_id": j.journal_id,
                "journal_name": j.journal_name,
                "rule_score": float(s),
                "reasons": list(r or []),
            }
            for j, s, r in candidates
        ],
        "evidence": evidence_by_id,
        "evidence_coverage": coverage,
        "status": status,
        "fallback_reason": fallback_reason,
    }


def _find_journal_by_venue(venue_name: str, pipeline) -> "Journal | None":
    """Mirror of run_evaluation._find_journal_by_venue: lookup the gold journal
    by name so we can pass ``diagnostic_journal_ids`` to ``recommend()`` and
    keep the candidate set aligned with the ablation runner."""
    store = getattr(getattr(pipeline, "candidate_generator", None), "store", None)
    if store is None or not venue_name:
        return None
    venue_normalized = venue_name.strip().lower()
    for journal in getattr(store, "journals", []):
        if (journal.journal_name or "").strip().lower() == venue_normalized:
            return journal
    return None


def _build_focused_candidates(entry, missing_journal_ids, pipeline):
    """Build the (Journal, score, reasons) tuples for the missing candidates
    in rule-rank order, by looking them up in the journal store."""
    store = getattr(
        getattr(pipeline, "candidate_generator", None), "store", None
    )
    if store is None:
        return []
    by_id = {j.journal_id: j for j in store.journals}
    rule_ranks = entry.get("rule_ranks", {}) or {}
    candidates_meta = entry.get("candidates", [])
    meta_by_id = {c["journal_id"]: c for c in candidates_meta}
    # Preserve the rule-rank order from the prior entry.
    ordered_ids = sorted(
        missing_journal_ids, key=lambda jid: rule_ranks.get(jid, 10**6)
    )
    out = []
    for jid in ordered_ids:
        j = by_id.get(jid)
        if j is None:
            continue
        meta = meta_by_id.get(jid, {})
        out.append(
            (j, float(meta.get("rule_score", 0.0)), list(meta.get("reasons", [])))
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-profile",
        choices=["light30", "full-v2-90", "custom"],
        default="light30",
    )
    parser.add_argument("--input", "-i", default=None)
    parser.add_argument(
        "--baseline-eval",
        required=True,
        help="Completed evaluation JSON used to freeze denominator and snapshots.",
    )
    parser.add_argument("--mode", choices=["title", "abstract", "full"], default="abstract")
    parser.add_argument("--papers", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default="data/evaluation/evidence",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow evidence_coverage < 1.0 (debug only).",
    )
    parser.add_argument(
        "--no-progress", action="store_true"
    )
    parser.add_argument(
        "--retry-incomplete-from",
        default=None,
        help=(
            "Path to a previous evidence snapshot. Papers with coverage=1.0 are "
            "copied verbatim (no LLM call). Papers with coverage<1.0 are re-run "
            "with up to 2 focused retry rounds (only the missing journal_ids "
            "are sent to the LLM). Final result is written to --output-dir with "
            "a new timestamp. Strict coverage=100pct gate is enforced unless "
            "--allow-partial is also set."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent papers processed in parallel (default 1, sequential).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    input_path = resolve_benchmark_input(args.benchmark_profile, args.input)
    papers = load_comparable_eval_papers(input_path, args.baseline_eval, limit=args.papers)
    if not papers:
        raise RuntimeError("No comparable papers loaded from baseline evaluation")

    with open("configs/app.yaml", "r", encoding="utf-8") as fh:
        app_config = yaml.safe_load(fh)
    with open("configs/prompts.yaml", "r", encoding="utf-8") as fh:
        prompts = yaml.safe_load(fh)

    pipeline = init_pipeline()
    # When evidence_role.enabled=true, pipeline.llm_ranker is wrapped in
    # LLMEvidenceRoleRanker which holds the LLM client as
    # ``evidence_extractor.llm``. Walk through wrappers to find the raw
    # LLM client.
    raw_llm = None
    ranker = pipeline.llm_ranker
    if ranker is not None:
        if hasattr(ranker, "llm") and ranker.llm is not None:
            raw_llm = ranker.llm
        elif hasattr(ranker, "evidence_extractor") and ranker.evidence_extractor is not None:
            raw_llm = getattr(ranker.evidence_extractor, "llm", None)
        elif hasattr(ranker, "direct_ranker") and ranker.direct_ranker is not None:
            inner = ranker.direct_ranker
            if hasattr(inner, "llm") and inner.llm is not None:
                raw_llm = inner.llm
    if raw_llm is None:
        raise RuntimeError("Pipeline has no LLM client; cannot run evidence extractor")
    extractor = LLMEvidenceExtractor(
        llm=raw_llm,
        system_prompt=prompts["llm_evidence_extractor_system"],
        user_prompt_template=prompts["llm_evidence_extractor_user"],
        focused_user_prompt_template=prompts.get(
            "llm_evidence_extractor_user_focused"
        ),
        timeout_seconds=app_config.get("ranking", {}).get(
            "llm_ranker_timeout_seconds", 420
        ),
    )

    # Load prior snapshot for incremental repair.
    prior_snapshot: dict = {}
    if args.retry_incomplete_from:
        prior_path = Path(args.retry_incomplete_from)
        if not prior_path.exists():
            raise RuntimeError(
                f"--retry-incomplete-from file not found: {prior_path}"
            )
        prior_snapshot = json.loads(prior_path.read_text(encoding="utf-8"))
        prior_papers = prior_snapshot.get("papers", {})
        n_complete = sum(
            1 for p in prior_papers.values() if p.get("evidence_coverage", 0) >= 1.0
        )
        n_partial = sum(
            1 for p in prior_papers.values() if p.get("evidence_coverage", 0) < 1.0
        )
        print(
            f"Incremental repair: {len(prior_papers)} papers carried over from "
            f"{prior_path.name} ({n_complete} complete, {n_partial} to repair)"
        )

    print(
        f"Pre-pass: benchmark={args.benchmark_profile}, papers={len(papers)}, "
        f"baseline={args.baseline_eval}, workers={args.workers}"
    )

    # Set up thread-safe printing when concurrent
    global _PRINT_LOCK
    if args.workers > 1:
        _PRINT_LOCK = threading.Lock()
    else:
        _PRINT_LOCK = None

    # Prepare per-paper inputs (paper + pkey + prior_entry) so workers
    # only need to be called with the resolved inputs.
    work_items = []
    for paper in papers:
        title = paper.get("title", "")
        venue = paper.get("venue", "")
        pkey = _paper_key(title, venue)
        prior_entry = (
            prior_snapshot.get("papers", {}).get(pkey) if prior_snapshot else None
        )
        work_items.append((paper, pkey, prior_entry))

    papers_out: dict = {}
    failed = 0
    copied = 0
    retried = 0
    started = time.time()
    show_progress = not args.no_progress

    if args.workers <= 1:
        # Sequential path: same behavior as before
        for paper, pkey, prior_entry in work_items:
            key, entry, c_inc, r_inc, f_inc = _process_one_paper(
                paper, pkey, prior_entry, pipeline, extractor, show_progress
            )
            papers_out[key] = entry
            copied += c_inc
            retried += r_inc
            failed += f_inc
    else:
        # Concurrent path: each worker gets its own (paper, pkey, prior_entry).
        # The pipeline + extractor are shared (LLM clients are thread-safe).
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _process_one_paper,
                    paper, pkey, prior_entry,
                    pipeline, extractor, show_progress,
                ): pkey
                for paper, pkey, prior_entry in work_items
            }
            for fut in as_completed(futures):
                try:
                    key, entry, c_inc, r_inc, f_inc = fut.result()
                except Exception as exc:  # belt-and-suspenders
                    logger.warning(
                        "[%s] worker raised: %s",
                        futures[fut][:40], exc,
                    )
                    failed += 1
                    continue
                papers_out[key] = entry
                copied += c_inc
                retried += r_inc
                failed += f_inc

    # Coverage gate.
    partial = [
        (k, v) for k, v in papers_out.items()
        if v.get("evidence_coverage", 0.0) < 1.0
    ]
    elapsed = time.time() - started
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": timestamp,
        "benchmark_profile": args.benchmark_profile,
        "benchmark_path": input_path,
        "baseline_eval": args.baseline_eval,
        "mode": args.mode,
        "paper_count": len(papers_out),
        "failed_pre_pass_count": failed,
        "partial_coverage_count": len(partial),
        "copied_from_prior_count": copied,
        "focused_retry_rounds": retried,
        "incremental_repair_source": (
            str(args.retry_incomplete_from) if args.retry_incomplete_from else None
        ),
        "papers": papers_out,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.benchmark_profile}_evidence_{timestamp}.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"\nEvidence snapshot: {out_path} ({elapsed:.0f}s, "
        f"failed={failed}, partial={len(partial)})"
    )
    if partial:
        print(
            f"[warn] {len(partial)} papers have evidence_coverage < 1.0:"
        )
        for k, v in partial[:5]:
            cov = v.get("evidence_coverage", 0.0)
            print(f"  - {k[:60]}... coverage={cov:.0%}")
        if not args.allow_partial:
            print(
                f"\nFAIL: {len(partial)} papers have partial coverage. "
                "Re-run extractor or pass --allow-partial to silence."
            )
            raise SystemExit(2)


if __name__ == "__main__":
    main()
