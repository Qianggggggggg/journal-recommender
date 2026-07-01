#!/usr/bin/env python3
"""Augment accepted-paper corpus by adding missing venues to fill benchmark coverage.

This script now delegates the heavy lifting (S2 fetching, leak checking, title
normalization) to ``scripts/expand_accepted_paper_corpus.py`` so the leak-blacklist
and normalization rules stay consistent across the project.

Usage::

    python scripts/augment_accepted_corpus.py --limit 10
    python scripts/augment_accepted_corpus.py --journals tocl,tos,apal --limit 6

Why this rewrite:
    The original (commit 15959b0) wrote 28 holdout240 papers into the corpus
    because it did not consult the leak blacklist. We now import
    ``build_blacklist``, ``fetch_semantic_scholar_candidates``,
    ``filter_candidates`` and ``merge_profile`` from
    ``expand_accepted_paper_corpus``. Any new accepted-paper write path should
    use the same helpers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.expand_accepted_paper_corpus import (
    DEFAULT_BENCHMARK_INPUTS,
    accepted_profile_names,
    build_blacklist,
    fetch_semantic_scholar_candidates,
    filter_candidates,
    load_journal_index,
    merge_profile,
    _profile_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Augment accepted-paper corpus with leak-safe S2 papers."
    )
    p.add_argument(
        "--journals",
        default=None,
        help="Comma-separated journal_ids to fill. Defaults to all venues present in the store "
             "but not yet in the corpus.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Target paper count per journal (default 10).",
    )
    p.add_argument(
        "--year-min",
        type=int,
        default=2020,
    )
    p.add_argument(
        "--year-max",
        type=int,
        default=2025,
    )
    p.add_argument(
        "--max-candidates",
        type=int,
        default=25,
        help="S2 search limit per journal.",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Semantic Scholar API key. Also reads from S2_API_KEY env var.",
    )
    p.add_argument(
        "--accepted-dir",
        default="data/accepted_papers",
    )
    p.add_argument(
        "--journals-store",
        default="data/processed/journals.jsonl",
    )
    p.add_argument(
        "--benchmarks",
        default=None,
        help=(
            "Comma-separated benchmark JSONL paths used to build the leak blacklist. "
            f"Defaults to the canonical 3 ({', '.join(str(p) for p in DEFAULT_BENCHMARK_INPUTS)})."
        ),
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.3,
        help="Sleep between S2 requests (sec).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    accepted_dir = Path(args.accepted_dir)
    journal_index = load_journal_index(Path(args.journals_store))
    accepted_names = accepted_profile_names(accepted_dir)

    if args.benchmarks:
        benchmark_paths = [Path(p) for p in args.benchmarks.split(",")]
    else:
        benchmark_paths = list(DEFAULT_BENCHMARK_INPUTS)
    blacklist = build_blacklist(benchmark_paths)

    if args.journals:
        target_ids = [j.strip() for j in args.journals.split(",") if j.strip()]
    else:
        # journal_index is keyed by normalized journal_name; use the store's
        # canonical journal_id for the on-disk file path.  This is the bug
        # commit 15959b0's script (and the earlier 5c4a00e path) had:
        # writing e.g. "medical image analysis.json" instead of
        # "medicalimageanalysis.json", producing 100+ orphan files.
        target_ids = [
            jrow["journal_id"]
            for jrow in journal_index.values()
            if jrow["journal_name"] not in accepted_names
            and jrow.get("journal_id")
        ]
    target_ids = sorted(set(target_ids))

    print(f"=== augment_accepted_corpus ===")
    print(f"store={len(journal_index)} | already_in_corpus={len(accepted_names)} | targets={len(target_ids)}")
    print(f"blacklist: titles={len(blacklist.titles)}, snippets={len(blacklist.abstract_snippets)}")
    print()

    # Build journal_id -> journal_row map for lookup, since journal_index is
    # keyed by normalized journal_name (load_journal_index quirk).
    id_to_row: dict[str, dict] = {}
    for jrow in journal_index.values():
        jid_store = jrow.get("journal_id", "")
        if jid_store:
            id_to_row[jid_store] = jrow

    summary = []
    for idx, jid in enumerate(target_ids, 1):
        jinfo = id_to_row.get(jid)
        if not jinfo:
            print(f"[{idx}/{len(target_ids)}] {jid} -> journal not in store, skip")
            continue
        jname = jinfo["journal_name"]
        profile_path = _profile_path(accepted_dir, jid)

        try:
            cands = fetch_semantic_scholar_candidates(
                venue=jname,
                max_candidates=args.max_candidates,
                year=f"{args.year_min}-{args.year_max}",
                timeout=30,
                api_key=args.api_key,
            )
        except Exception as exc:
            print(f"[{idx}/{len(target_ids)}] {jid} S2 error: {exc}")
            continue

        kept = filter_candidates(
            candidates=cands,
            target_venue=jname,
            blacklist=blacklist,
            existing_titles=set(),  # merge_profile dedupes on its own
            limit=args.limit,
        )

        if not kept:
            print(f"[{idx}/{len(target_ids)}] {jid} ({jname[:30]:<30}) 0 papers after filter")
            summary.append((jid, jname, 0, "filtered_to_zero"))
            time.sleep(args.sleep)
            continue

        try:
            added = merge_profile(
                profile_path=profile_path,
                journal_id=jid,
                journal_name=jname,
                new_papers=kept,
                target_total=args.limit,
                source="semantic_scholar",
            )
        except Exception as exc:
            print(f"[{idx}/{len(target_ids)}] {jid} write error: {exc}")
            summary.append((jid, jname, 0, f"write_error: {exc}"))
            time.sleep(args.sleep)
            continue

        print(
            f"[{idx}/{len(target_ids)}] {jid:30s} ({jname[:30]:<30}) "
            f"added={added} of {len(kept)} kept"
        )
        summary.append((jid, jname, added, ""))
        time.sleep(args.sleep)

    print()
    print("=== summary ===")
    total_added = sum(s[2] for s in summary)
    success = sum(1 for s in summary if s[2] > 0)
    zero = sum(1 for s in summary if s[2] == 0 and not s[3])
    err = sum(1 for s in summary if s[3] and s[3] != "filtered_to_zero")
    print(f"  total_added: {total_added}")
    print(f"  successful journals: {success}")
    print(f"  zero-paper (filtered): {zero}")
    print(f"  errors: {err}")


if __name__ == "__main__":
    main()
