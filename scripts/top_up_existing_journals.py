#!/usr/bin/env python3
"""Top up existing accepted-paper profiles to a target minimum.

Unlike expand_accepted_paper_corpus.py (which only targets NEW journals),
this script tops up journals that already have some papers but fewer than
a target minimum. It reuses the exact same blacklist, filter, and merge
logic from expand_accepted_paper_corpus.py.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expand_accepted_paper_corpus import (
    build_blacklist,
    filter_candidates,
    fetch_openalex_candidates,
    fetch_semantic_scholar_candidates,
    load_journal_index,
    merge_profile,
    normalize_venue,
)


MIN_TARGET = 10
MAX_CANDIDATES = 50
YEAR = "2018-2026"
TIMEOUT = 30
SLEEP = 1.0
BENCHMARKS = [
    Path("data/evaluation/papers_metadata_660_balanced.jsonl"),
    Path("data/evaluation/papers_metadata_holdout240.jsonl"),
    Path("data/evaluation/papers_metadata_full_v2_90.jsonl"),
    Path("data/evaluation/papers_metadata_light_30.jsonl"),
]


def load_profile_titles(path: Path) -> set[str]:
    from scripts.expand_accepted_paper_corpus import normalize_text
    titles: set[str] = set()
    if not path.exists():
        return titles
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return titles
    for paper in data.get("papers", []):
        if isinstance(paper, dict) and paper.get("title"):
            titles.add(normalize_text(paper["title"]))
    return titles


def main() -> int:
    accepted_dir = Path("data/accepted_papers")
    journal_index = load_journal_index(Path("data/processed/journals.jsonl"))
    blacklist = build_blacklist(BENCHMARKS)

    # Find journals with < MIN_TARGET papers
    thin: list[tuple[str, str, int]] = []
    for path in sorted(accepted_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        jid = data.get("journal_id", "")
        n = len(data.get("papers", []))
        if 0 < n < MIN_TARGET:
            thin.append((jid, data.get("journal_name", ""), n))

    thin.sort(key=lambda x: x[1])
    print(f"Topping up {len(thin)} journals to ≥ {MIN_TARGET} papers")
    use_openalex = os.environ.get("USE_OPENALEX", "1") == "1"

    ok, fail, total_added = 0, 0, 0
    for idx, (jid, jname, current) in enumerate(thin, 1):
        needed = MIN_TARGET - current
        profile_path = Path(f"data/accepted_papers/{jid}.json")
        existing_titles = load_profile_titles(profile_path)

        try:
            if use_openalex:
                candidates = fetch_openalex_candidates(
                    jname, max_candidates=MAX_CANDIDATES, year=YEAR, timeout=TIMEOUT
                )
                source = "openalex"
            else:
                api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
                candidates = fetch_semantic_scholar_candidates(
                    jname, max_candidates=MAX_CANDIDATES, year=YEAR, timeout=TIMEOUT, api_key=api_key
                )
                source = "semantic_scholar"

            filtered = filter_candidates(
                candidates,
                target_venue=jname,
                blacklist=blacklist,
                existing_titles=existing_titles,
                limit=needed + 10,  # fetch extra as buffer
            )
            added = merge_profile(
                profile_path=profile_path,
                journal_id=jid,
                journal_name=jname,
                new_papers=filtered,
                target_total=MIN_TARGET,
                source=source,
            )
            total_added += added
            if added >= needed:
                ok += 1
                print(f"  [{idx}/{len(thin)}] {jid}: {current}→{current+added} ✓")
            elif added > 0:
                ok += 1
                print(f"  [{idx}/{len(thin)}] {jid}: {current}→{current+added} (wanted +{needed}) ⚠️")
            else:
                fail += 1
                print(f"  [{idx}/{len(thin)}] {jid}: {current}→{current+added} (no new papers) ✗")
        except Exception as exc:
            fail += 1
            print(f"  [{idx}/{len(thin)}] {jid}: ERROR {exc}")

        if idx < len(thin) and SLEEP > 0:
            time.sleep(SLEEP)

    print(f"\nDone: {ok} OK, {fail} failed, {total_added} papers added")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
