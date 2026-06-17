"""Batch 1: Augment 10 zero-coverage C-tier journals via OpenAlex + ISSN.

Strict zero-leakage batch: all 10 target journals are NOT in holdout240 test
set, so any improvement is purely from retrieval coverage, not test-set fit.

Strategy:
  1. Use OpenAlex /works endpoint with ISSN filter (exact venue matching).
  2. Filter candidates via existing build_blacklist() to ensure no test papers leak.
  3. Write data/accepted_papers/<jid>.json files with fetched papers.

Usage:
  python scripts/augment_batch1_c_tier.py
  python scripts/augment_batch1_c_tier.py --limit 5 --dry-run
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

import urllib.request
import urllib.parse

from scripts.expand_accepted_paper_corpus import (
    DEFAULT_BENCHMARK_INPUTS,
    CandidatePaper,
    build_blacklist,
    merge_profile,
)


# Hardcoded ISSNs for the 10 zero-coverage C-tier journals.
# Sourced from CrossRef / publisher websites.
# All 10 are confirmed NOT in holdout240 test set (zero leakage).
BATCH1_ISSN = {
    "integration": "0167-9260",                          # Integration, the VLSI Journal
    "jetta": "0923-8174",                                # J. Electronic Testing
    "jgc": "1570-7873",                                  # Journal of Grid Computing
    "sttt": "1433-2779",                                 # Int J Software Tools for Tech Transfer
    "jsl": "0022-4812",                                  # Journal of Symbolic Logic
    "tqc": "2643-6817",                                  # ACM Trans. on Quantum Computing
    "cgta": "0925-7721",                                 # Computational Geometry
    "fuzzysetsandsystems": "0165-0114",                  # Fuzzy Sets and Systems
    "eitee": "1009-3093",                                # J. Info. Tech. & Electronic Eng.
    "acm_dlt": "2688-0084",                              # ACM Distributed Ledger Tech.
}


def fetch_openalex_works_by_issn(
    issn: str,
    *,
    max_results: int,
    year_from: int,
    year_to: int,
    timeout: int,
) -> list[CandidatePaper]:
    """Fetch papers from OpenAlex filtered by ISSN.

    Uses /works endpoint with `primary_location.source.issn:<issn>` filter.
    """
    url = (
        "https://api.openalex.org/works?"
        + urllib.parse.urlencode(
            {
                "filter": f"primary_location.source.issn:{issn},type:article,"
                          f"from_publication_date:{year_from}-01-01,"
                          f"to_publication_date:{year_to}-12-31",
                "sort": "publication_date:desc",
                "per-page": str(max_results),
                "select": "id,doi,title,publication_year,abstract_inverted_index,primary_location",
            }
        )
    )
    req = urllib.request.Request(
        url, headers={"User-Agent": "jr-corpus-batch1/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    candidates: list[CandidatePaper] = []
    for work in payload.get("results", []):
        title = (work.get("title") or "").strip()
        if not title:
            continue
        # Reconstruct abstract from OpenAlex's inverted index format
        abstract_inverted = work.get("abstract_inverted_index") or {}
        if abstract_inverted:
            abstract = _reconstruct_inverted_abstract(abstract_inverted)
        else:
            abstract = ""
        if not abstract:
            continue  # OpenAlex papers without abstracts are useless for retrieval
        primary_loc = work.get("primary_location") or {}
        source = primary_loc.get("source") or {}
        venue = (source.get("display_name") or "").strip()
        doi_url = (work.get("doi") or "").strip()
        candidates.append(
            CandidatePaper(
                title=title,
                abstract=abstract,
                venue=venue,
                year=work.get("publication_year"),
                doi=doi_url.replace("https://doi.org/", "") if doi_url else "",
                url=doi_url or work.get("id", ""),
            )
        )
    return candidates


def _reconstruct_inverted_abstract(inverted: dict[str, list[int]]) -> str:
    """Convert OpenAlex inverted-index abstract back to plain text."""
    positions: list[tuple[int, str]] = []
    for word, locs in inverted.items():
        for pos in locs:
            positions.append((pos, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10, help="Papers per journal (default 10)")
    parser.add_argument("--year-from", type=int, default=2020)
    parser.add_argument("--year-to", type=int, default=2025)
    parser.add_argument("--max-results", type=int, default=30)
    parser.add_argument("--accepted-dir", default="data/accepted_papers")
    parser.add_argument(
        "--journals",
        default=",".join(BATCH1_ISSN.keys()),
        help="Comma-separated journal_ids (default: all 10 in BATCH1_ISSN)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and print stats without writing profile JSON",
    )
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    blacklist = build_blacklist(list(DEFAULT_BENCHMARK_INPUTS))
    accepted_dir = Path(args.accepted_dir)
    accepted_dir.mkdir(parents=True, exist_ok=True)

    target_ids = [j.strip() for j in args.journals.split(",") if j.strip()]
    print(f"=== augment_batch1_c_tier ===")
    print(f"targets={len(target_ids)} | year={args.year_from}-{args.year_to} | limit={args.limit}/journal")
    print(f"blacklist: titles={len(blacklist.titles)}, snippets={len(blacklist.abstract_snippets)}")
    print()

    summary = []
    for idx, jid in enumerate(target_ids, 1):
        issn = BATCH1_ISSN.get(jid)
        if not issn:
            print(f"[{idx}/{len(target_ids)}] {jid} -> no ISSN in BATCH1_ISSN, skip")
            summary.append((jid, "", 0, "no_issn"))
            continue

        try:
            cands = fetch_openalex_works_by_issn(
                issn,
                max_results=args.max_results,
                year_from=args.year_from,
                year_to=args.year_to,
                timeout=args.timeout,
            )
        except Exception as exc:
            print(f"[{idx}/{len(target_ids)}] {jid} OA error: {exc}")
            summary.append((jid, issn, 0, f"oa_error: {exc}"))
            time.sleep(args.sleep)
            continue

        if not cands:
            print(f"[{idx}/{len(target_ids)}] {jid} (ISSN {issn}) 0 candidates from OpenAlex")
            summary.append((jid, issn, 0, "oa_zero"))
            time.sleep(args.sleep)
            continue

        # Filter against blacklist (test set leak protection)
        kept: list[CandidatePaper] = []
        for c in cands:
            from scripts.expand_accepted_paper_corpus import _candidate_has_leak
            if _candidate_has_leak(c, blacklist):
                continue
            kept.append(c)
            if len(kept) >= args.limit:
                break

        # Filter by ISSN match (defensive — OpenAlex already filtered, but verify)
        # (skip — trust OpenAlex source.issn filter)

        if args.dry_run:
            print(f"[{idx}/{len(target_ids)}] {jid} (ISSN {issn}) fetched={len(cands)}, kept={len(kept)} (DRY RUN)")
            summary.append((jid, issn, len(kept), "dry_run"))
            time.sleep(args.sleep)
            continue

        # Load existing profile or create new
        profile_path = accepted_dir / f"{jid}.json"
        journal_name = cands[0].venue if cands else jid

        try:
            added = merge_profile(
                profile_path=profile_path,
                journal_id=jid,
                journal_name=journal_name,
                new_papers=kept,
                target_total=args.limit,
                source="openalex_batch1_20260617",
            )
        except Exception as exc:
            print(f"[{idx}/{len(target_ids)}] {jid} write error: {exc}")
            summary.append((jid, issn, 0, f"write_error: {exc}"))
            time.sleep(args.sleep)
            continue

        print(
            f"[{idx}/{len(target_ids)}] {jid:30s} (ISSN {issn}) "
            f"fetched={len(cands)}, kept={len(kept)}, added={added}"
        )
        summary.append((jid, issn, added, ""))
        time.sleep(args.sleep)

    print()
    print("=== summary ===")
    total_added = sum(s[2] for s in summary)
    success = sum(1 for s in summary if s[2] > 0)
    print(f"  total_added: {total_added}")
    print(f"  successful journals: {success}/{len(summary)}")
    for jid, issn, n, note in summary:
        if note:
            print(f"    {jid:<30} ISSN={issn:<10} added={n}  ({note})")


if __name__ == "__main__":
    main()
