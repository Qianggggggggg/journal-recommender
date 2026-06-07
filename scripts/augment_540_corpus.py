#!/usr/bin/env python3
"""Augment 540 papers corpus via Semantic Scholar search.

For each (area, ccf) bucket (10 areas × 3 ccf = 30 buckets), we need
18 papers. Each bucket has N journals. We aim for ⌈18/N⌉ papers per
journal, evenly distributed. Search S2 for each journal's papers
in 2022-2025, filter by abstract/title/venue match, output 18 per
bucket.

Outputs:
  - data/evaluation/papers_metadata_540_raw.jsonl
  - data/evaluation/papers_metadata_540_pool_candidates.json (intermediate cache)

S2 API limits:
  - Free tier: ~100 req/sec with API key, ~1 req/sec without
  - We sleep 0.3s between requests to stay safe with key

Filters per candidate:
  1. title not empty, not in seen_titles
  2. abstract not empty, ≥ 160 chars
  3. publicationVenue.name casefold-matches the queried journal name
  4. year ∈ [2022, 2025]
  5. externalIds has DOI or arXiv or CorpusId
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

POOL_INDEX_PATH = project_root / "data/evaluation/papers_metadata_540_pool_index.json"
RAW_OUT_PATH = project_root / "data/evaluation/papers_metadata_540_raw.jsonl"
CACHE_DIR = project_root / "data/evaluation/papers_metadata_540_s2_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
TARGET_PER_BUCKET = 18
YEAR_MIN = 2022
YEAR_MAX = 2025
SLEEP_BETWEEN = 0.7  # seconds between S2 calls (with API key, conservative)


def s2_get(url: str, max_retries: int = 5) -> dict | None:
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {"User-Agent": "540-augment/1.0"}
    if key:
        headers["x-api-key"] = key
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 10 * (attempt + 1)  # 10, 20, 30, 40, 50s
                print(f"  [429] backoff {wait}s", flush=True)
                time.sleep(wait)
            elif e.code == 404:
                return None
            else:
                print(f"  [S2 {e.code}]", flush=True)
                return None
        except Exception as e:
            print(f"  [net] {e}", flush=True)
            time.sleep(2)
    return None


def normalize_title(t: str) -> str:
    return " ".join((t or "").casefold().split())


def search_journal_papers(
    journal_name: str, year_min: int, year_max: int, limit: int = 50
) -> list[dict]:
    """S2 search by exact journal name + year range, with venue filter.

    Returns list of S2 paper records. Each record has title, abstract,
    year, venue, publicationVenue, externalIds, paperId.

    IMPORTANT: S2's `query` param is fuzzy text search; it returns papers
    that MENTION the journal name in title/abstract even if they were
    actually published in a different journal. Adding `venue=<exact name>`
    restricts the search to that publicationVenue, so we don't get noise
    from sibling journals (e.g. searching "ACM Transactions on Embedded
    Computing Systems" without venue= returns papers from TECS, TACO,
    TIST, TOCE, etc.). Filter on publicationVenue.name is still applied
    defensively in paper_passes_filter.
    """
    url = (
        f"{S2_SEARCH}?query={urllib.parse.quote(journal_name)}"
        f"&venue={urllib.parse.quote(journal_name)}"
        f"&limit={limit}&year={year_min}-{year_max}"
        f"&fields=title,venue,publicationVenue,year,abstract,externalIds,"
        f"authors,paperId"
    )
    data = s2_get(url)
    if not data or "data" not in data:
        return []
    return data["data"]


def paper_matches_journal(paper: dict, journal_name: str) -> bool:
    """publicationVenue.name or venue matches queried journal (casefold)."""
    pv = paper.get("publicationVenue") or {}
    pv_name = (pv.get("name") or "").casefold().strip()
    venue = (paper.get("venue") or "").casefold().strip()
    target = journal_name.casefold().strip()
    return target == pv_name or target == venue


def paper_passes_filter(paper: dict, journal_name: str) -> bool:
    if not paper_matches_journal(paper, journal_name):
        return False
    if not paper.get("title"):
        return False
    abstract = paper.get("abstract") or ""
    if len(abstract) < 160:
        return False
    year = paper.get("year")
    if not (isinstance(year, int) and YEAR_MIN <= year <= YEAR_MAX):
        return False
    ext = paper.get("externalIds") or {}
    if not (ext.get("DOI") or ext.get("ArXiv") or ext.get("CorpusId")):
        return False
    return True


def collect_bucket(
    area: str, ccf: str, journals: list[str], seen_titles: set[str]
) -> tuple[list[dict], dict]:
    """Search S2 for each journal in this bucket, return up to TARGET_PER_BUCKET
    papers plus per-journal diagnostic counts.
    """
    target_pp = math.ceil(TARGET_PER_BUCKET / len(journals))
    # Distribute target with at least 1 per journal where possible; allow over
    # for first journals if last ones are short.
    per_journal_targets = [target_pp] * len(journals)
    remainder = TARGET_PER_BUCKET - target_pp * len(journals)
    for i in range(remainder):
        per_journal_targets[i] += 1

    papers: list[dict] = []
    diagnostics: dict = {
        "area": area,
        "ccf": ccf,
        "journals": journals,
        "per_journal_target": per_journal_targets,
        "per_journal_collected": [0] * len(journals),
        "per_journal_s2_searched": [0] * len(journals),
        "per_journal_s2_passed_filter": [0] * len(journals),
    }
    for j_idx, jname in enumerate(journals):
        need = per_journal_targets[j_idx]
        if need <= 0:
            continue
        # Use j_idx in cache filename (NOT the journal name with '/' or
        # non-ASCII chars from the area string) so the path is portable.
        cache_path = CACHE_DIR / f"bucket_{j_idx}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            diagnostics["per_journal_s2_searched"][j_idx] = len(cached)
        else:
            print(f"  [S2] searching {jname[:60]} ...", flush=True)
            s2_papers = search_journal_papers(jname, YEAR_MIN, YEAR_MAX, limit=50)
            diagnostics["per_journal_s2_searched"][j_idx] = len(s2_papers)
            cache_path.write_text(json.dumps(s2_papers, ensure_ascii=False))
            time.sleep(SLEEP_BETWEEN)

        cached = json.loads(cache_path.read_text())
        passed = 0
        for cand in cached:
            if not paper_passes_filter(cand, jname):
                continue
            t_norm = normalize_title(cand.get("title", ""))
            if t_norm in seen_titles:
                continue
            ext = cand.get("externalIds", {}) or {}
            paper = {
                "title": cand["title"],
                "abstract": cand.get("abstract", ""),
                "venue": jname,
                "year": cand.get("year"),
                "source": "semantic_scholar",
                "doi": ext.get("DOI", ""),
                "arxiv": ext.get("ArXiv", ""),
                "corpus_id": str(ext.get("CorpusId", "") or ""),
                "url": (
                    f"https://www.semanticscholar.org/paper/{cand.get('paperId','')}"
                    if cand.get("paperId")
                    else ""
                ),
                "research_area": [area],
                "ccf_level": ccf,
                "audit_status": "raw",
            }
            papers.append(paper)
            seen_titles.add(t_norm)
            passed += 1
            diagnostics["per_journal_collected"][j_idx] += 1
            if diagnostics["per_journal_collected"][j_idx] >= need:
                break
        diagnostics["per_journal_s2_passed_filter"][j_idx] = passed
        print(
            f"    {jname[:50]:50s} searched={diagnostics['per_journal_s2_searched'][j_idx]:>3d} "
            f"passed={passed:>2d} collected={diagnostics['per_journal_collected'][j_idx]:>2d}/{need}",
            flush=True,
        )
    return papers, diagnostics


def main() -> int:
    if not POOL_INDEX_PATH.exists():
        print(f"Missing {POOL_INDEX_PATH}; run scripts/build_540_pool_index.py first")
        return 1
    pool_doc = json.loads(POOL_INDEX_PATH.read_text())
    buckets = pool_doc["buckets"]
    print(f"=== Augmenting 540 corpus: {len(buckets)} buckets ===\n", flush=True)

    seen_titles: set[str] = set()
    # Also load titles from existing datasets to enforce no overlap
    for path in [
        project_root / "data/evaluation/papers_metadata_light_30.jsonl",
        project_root / "data/evaluation/papers_metadata_full_v2_90.jsonl",
        project_root / "data/evaluation/papers_metadata_holdout240.jsonl",
    ]:
        if path.exists():
            for line in path.open():
                d = json.loads(line)
                seen_titles.add(normalize_title(d.get("title", "")))
    print(f"Excluded titles (from light30/full-v2-90/holdout240): {len(seen_titles)}",
          flush=True)

    all_papers: list[dict] = []
    all_diags: list[dict] = []
    for bucket_key, journals in buckets.items():
        area, ccf = bucket_key.split("|")
        print(f"\n[{area}/{ccf}] {len(journals)} journals × ⌈{TARGET_PER_BUCKET}/{len(journals)}⌉",
              flush=True)
        papers, diag = collect_bucket(area, ccf, journals, seen_titles)
        all_papers.extend(papers)
        all_diags.append(diag)
        print(f"  -> collected {len(papers)}/{TARGET_PER_BUCKET}", flush=True)

    # Write raw jsonl
    with RAW_OUT_PATH.open("w", encoding="utf-8") as f:
        for p in all_papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_papers)} papers to {RAW_OUT_PATH}", flush=True)

    # Summary
    total_target = sum(d["per_journal_target"] for d in all_diags)
    total_collected = sum(
        sum(d["per_journal_collected"]) for d in all_diags
    )
    short_buckets = [d for d in all_diags
                     if sum(d["per_journal_collected"]) < TARGET_PER_BUCKET]
    print(f"\n=== Summary ===")
    print(f"Target total: {total_target}")
    print(f"Collected total: {total_collected}")
    print(f"Short buckets (need re-search): {len(short_buckets)}")
    for d in short_buckets:
        n_collected = sum(d["per_journal_collected"])
        print(f"  {d['area']}/{d['ccf']}: {n_collected}/{TARGET_PER_BUCKET}")

    diag_path = project_root / "data/evaluation/papers_metadata_540_augment_diagnostics.json"
    diag_path.write_text(json.dumps(all_diags, ensure_ascii=False, indent=2))
    print(f"\nWrote diagnostics to {diag_path}")
    return 0 if not short_buckets else 2


if __name__ == "__main__":
    sys.exit(main())
