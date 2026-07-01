"""Fill all journals with corpus < N papers to reach the threshold.

Generalized version of augment_batch1_c_tier.py that handles any subset of
journals (by corpus count, by journal_id list, or by CCF tier). Uses OpenAlex
as primary source, falls back to fuzzy name search, applies leak blacklist.

Usage:
    # Fill all <10 papers to 10
    python scripts/fill_corpus_to_threshold.py --target-count 10

    # Only A-tier and B-tier
    python scripts/fill_corpus_to_threshold.py --target-count 10 --ccf-tiers A,B

    # Only specific journals
    python scripts/fill_corpus_to_threshold.py --journals c&g,apal,ijseke --target-count 10

    # Dry run
    python scripts/fill_corpus_to_threshold.py --dry-run

Output: writes data/accepted_papers/<jid>.json with merged papers.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.expand_accepted_paper_corpus import (
    DEFAULT_BENCHMARK_INPUTS,
    CandidatePaper,
    build_blacklist,
    merge_profile,
)


def normalize_venue(s: str) -> str:
    """Match the convention used in scripts/expand_accepted_paper_corpus.py."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def openalex_search_source(name: str, timeout: int = 30) -> Optional[str]:
    """Find OpenAlex source_id for a journal by name search.

    Returns source_id (e.g. 'https://openalex.org/S123') or None.
    Picks the FIRST result whose normalized display_name matches the query exactly;
    falls back to first result if no exact match.
    """
    url = (
        "https://api.openalex.org/sources?"
        + urllib.parse.urlencode({"search": name, "per-page": "5"})
    )
    req = urllib.request.Request(url, headers={"User-Agent": "jr-fill/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    results = payload.get("results") or []
    if not results:
        return None
    target_norm = normalize_venue(name)
    # Try exact match first
    for r in results:
        if normalize_venue(r.get("display_name", "")) == target_norm:
            sid = r.get("id")
            if sid:
                return sid
    # Fall back to first result
    sid = results[0].get("id")
    return sid if sid else None


def openalex_fetch_works_by_source(
    source_id: str,
    *,
    max_results: int,
    year_from: int,
    year_to: int,
    timeout: int,
) -> list[CandidatePaper]:
    url = (
        "https://api.openalex.org/works?"
        + urllib.parse.urlencode(
            {
                "filter": f"primary_location.source.id:{source_id},type:article,"
                          f"from_publication_date:{year_from}-01-01,"
                          f"to_publication_date:{year_to}-12-31",
                "sort": "publication_date:desc",
                "per-page": str(max_results),
                "select": "id,doi,title,publication_year,abstract_inverted_index,primary_location",
            }
        )
    )
    req = urllib.request.Request(url, headers={"User-Agent": "jr-fill/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    candidates: list[CandidatePaper] = []
    for work in payload.get("results", []):
        title = (work.get("title") or "").strip()
        if not title:
            continue
        inverted = work.get("abstract_inverted_index") or {}
        abstract = _reconstruct_abstract(inverted) if inverted else ""
        if not abstract:
            continue
        primary = work.get("primary_location") or {}
        source = primary.get("source") or {}
        venue = (source.get("display_name") or "").strip()
        doi = (work.get("doi") or "").strip()
        candidates.append(
            CandidatePaper(
                title=title,
                abstract=abstract,
                venue=venue,
                year=work.get("publication_year"),
                doi=doi.replace("https://doi.org/", "") if doi else "",
                url=doi or work.get("id", ""),
            )
        )
    return candidates


def _reconstruct_abstract(inverted: dict) -> str:
    positions: list[tuple[int, str]] = []
    for word, locs in inverted.items():
        for pos in locs:
            positions.append((pos, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def count_papers(journal_id: str, accepted_dir: Path) -> int:
    p = accepted_dir / f"{journal_id}.json"
    if not p.exists():
        return 0
    try:
        d = json.loads(p.read_text())
        return len(d.get("papers", []))
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-count", type=int, default=10)
    parser.add_argument("--year-from", type=int, default=2020)
    parser.add_argument("--year-to", type=int, default=2025)
    parser.add_argument("--max-results", type=int, default=30,
                       help="Max candidates to fetch per journal")
    parser.add_argument("--accepted-dir", default="data/accepted_papers")
    parser.add_argument("--journals-store", default="data/processed/journals.jsonl")
    parser.add_argument("--journals", default=None,
                       help="Comma-separated journal_ids to fill (default: all <target-count)")
    parser.add_argument("--ccf-tiers", default=None,
                       help="Comma-separated CCF tiers to restrict (e.g. 'A,B' or 'C')")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--year-range-wider", action="store_true",
                       help="On low fetch, retry with year_from=2015")
    args = parser.parse_args()

    blacklist = build_blacklist(list(DEFAULT_BENCHMARK_INPUTS))
    accepted_dir = Path(args.accepted_dir)
    accepted_dir.mkdir(parents=True, exist_ok=True)

    # Load journal store
    journals = []
    with open(args.journals_store) as f:
        for line in f:
            journals.append(json.loads(line))
    journal_by_id = {j["journal_id"]: j for j in journals}

    # Determine target list
    if args.journals:
        target_ids = [j.strip() for j in args.journals.split(",") if j.strip()]
    else:
        target_ids = [j["journal_id"] for j in journals]
    if args.ccf_tiers:
        tiers = {t.strip().upper() for t in args.ccf_tiers.split(",") if t.strip()}
        target_ids = [j for j in target_ids if journal_by_id.get(j, {}).get("ccf_rating") in tiers]

    # Filter to those below threshold
    target_ids = [j for j in target_ids if count_papers(j, accepted_dir) < args.target_count]

    print(f"=== fill_corpus_to_threshold ===")
    print(f"target_count={args.target_count} | year={args.year_from}-{args.year_to}")
    print(f"targets={len(target_ids)} | blacklist: titles={len(blacklist.titles)}, "
          f"snippets={len(blacklist.abstract_snippets)}")
    print(f"dry_run={args.dry_run}")
    print()

    summary = []
    for idx, jid in enumerate(target_ids, 1):
        jinfo = journal_by_id.get(jid)
        if not jinfo:
            print(f"[{idx}/{len(target_ids)}] {jid} -> not in journal store, skip")
            continue
        jname = jinfo["journal_name"]
        current = count_papers(jid, accepted_dir)
        needed = args.target_count - current

        # OpenAlex source lookup
        source_id = openalex_search_source(jname, timeout=args.timeout)
        if not source_id:
            print(f"[{idx}/{len(target_ids)}] {jid} ({jname[:30]:<30})  no OpenAlex source")
            summary.append((jid, jname, current, 0, "no_source"))
            time.sleep(args.sleep)
            continue

        # Fetch works
        try:
            cands = openalex_fetch_works_by_source(
                source_id,
                max_results=args.max_results,
                year_from=args.year_from,
                year_to=args.year_to,
                timeout=args.timeout,
            )
        except Exception as exc:
            print(f"[{idx}/{len(target_ids)}] {jid} OA fetch error: {exc}")
            summary.append((jid, jname, current, 0, f"fetch_error"))
            time.sleep(args.sleep)
            continue

        if not cands and args.year_range_wider:
            # Retry with wider year range
            try:
                cands = openalex_fetch_works_by_source(
                    source_id,
                    max_results=args.max_results,
                    year_from=2015,
                    year_to=args.year_to,
                    timeout=args.timeout,
                )
            except Exception:
                pass

        if not cands:
            print(f"[{idx}/{len(target_ids)}] {jid} ({jname[:30]:<30})  0 candidates")
            summary.append((jid, jname, current, 0, "oa_zero"))
            time.sleep(args.sleep)
            continue

        # Apply blacklist filter
        from scripts.expand_accepted_paper_corpus import _candidate_has_leak
        kept: list[CandidatePaper] = []
        for c in cands:
            if _candidate_has_leak(c, blacklist):
                continue
            kept.append(c)

        # Cap to needed
        if len(kept) > needed:
            kept = kept[:needed]

        if args.dry_run:
            print(f"[{idx}/{len(target_ids)}] {jid} ({jname[:30]:<30})  "
                  f"current={current}, fetched={len(cands)}, kept={len(kept)}, target={args.target_count} (DRY)")
            summary.append((jid, jname, current, len(kept), "dry_run"))
            time.sleep(args.sleep)
            continue

        # Merge into existing profile
        profile_path = accepted_dir / f"{jid}.json"
        try:
            added = merge_profile(
                profile_path=profile_path,
                journal_id=jid,
                journal_name=jname,
                new_papers=kept,
                target_total=args.target_count,
                source="openalex_fill_20260617",
            )
        except Exception as exc:
            print(f"[{idx}/{len(target_ids)}] {jid} write error: {exc}")
            summary.append((jid, jname, current, 0, f"write_error"))
            time.sleep(args.sleep)
            continue

        new_total = current + added
        flag = "✓" if new_total >= args.target_count else "partial"
        print(f"[{idx}/{len(target_ids)}] {jid} ({jname[:30]:<30})  "
              f"{current}→{new_total} (+{added})  {flag}")
        summary.append((jid, jname, new_total, added, ""))
        time.sleep(args.sleep)

    print()
    print("=== summary ===")
    total_added = sum(s[3] for s in summary)
    reached_target = sum(1 for s in summary if s[2] >= args.target_count)
    no_source = sum(1 for s in summary if s[4] == "no_source")
    oa_zero = sum(1 for s in summary if s[4] == "oa_zero")
    print(f"  total processed: {len(summary)}")
    print(f"  total_added: {total_added}")
    print(f"  reached target ({args.target_count}+): {reached_target}")
    print(f"  no OpenAlex source: {no_source}")
    print(f"  OpenAlex zero candidates: {oa_zero}")


if __name__ == "__main__":
    main()
