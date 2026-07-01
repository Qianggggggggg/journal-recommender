#!/usr/bin/env python3
"""One-shot batch expansion: expand all missing/thin journals to ≥10 papers.

Handles rate limiting (retry with backoff), S2 short abstracts, and
venue name normalization. Uses the expand_accepted_paper_corpus blacklist.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expand_accepted_paper_corpus import (
    ABSTRACT_SNIPPET_LENGTH,
    Blacklist,
    build_blacklist,
    merge_profile,
    normalize_text,
    normalize_venue,
    venue_matches,
)

TARGET = 10
MAX_CANDIDATES = 50
YEAR = "2018-2026"
BENCHMARKS = [
    Path("data/evaluation/papers_metadata_660_balanced.jsonl"),
    Path("data/evaluation/papers_metadata_holdout240.jsonl"),
    Path("data/evaluation/papers_metadata_full_v2_90.jsonl"),
    Path("data/evaluation/papers_metadata_light_30.jsonl"),
]
MIN_ABSTRACT_LEN = 100  # S2 often has short abstracts; accept ≥100 chars


def _s2_url(venue: str, limit: int) -> str:
    params = {
        "query": venue, "venue": venue, "limit": str(limit),
        "year": YEAR,
        "fields": "title,abstract,venue,year,externalIds,url",
    }
    return "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params)


def fetch_s2_with_retry(venue: str, api_key: str, max_retries: int = 3) -> list[dict]:
    """Fetch S2 papers with exponential backoff on 429."""
    for attempt in range(max_retries):
        try:
            url = _s2_url(venue, MAX_CANDIDATES)
            headers = {"User-Agent": "batch-expand/1.0"}
            if api_key:
                headers["x-api-key"] = api_key
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode())
            results = []
            for item in payload.get("data", []):
                title = (item.get("title") or "").strip()
                abstract = (item.get("abstract") or "").strip()
                if not title:
                    continue
                results.append({
                    "title": title,
                    "abstract": abstract,
                    "venue": (item.get("venue") or "").strip(),
                    "year": item.get("year") if isinstance(item.get("year"), int) else None,
                    "doi": str((item.get("externalIds") or {}).get("DOI") or "").strip(),
                    "url": str(item.get("url") or "").strip(),
                })
            return results
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait = (attempt + 1) * 30
                print(f"    429, retry in {wait}s...", end="", flush=True)
                time.sleep(wait)
                print(" retrying", flush=True)
                continue
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(10)
                continue
            raise


def _has_leak(title: str, abstract: str, blacklist: Blacklist) -> bool:
    if normalize_text(title) in blacklist.titles:
        return True
    snippet = normalize_text(abstract)[:ABSTRACT_SNIPPET_LENGTH]
    return len(snippet) >= ABSTRACT_SNIPPET_LENGTH and snippet in blacklist.abstract_snippets


def is_non_research_title(title: str) -> bool:
    """Filter out corrigendum/erratum/editorial/index/retraction."""
    import re
    t = title.lower()
    patterns = [
        r'\bcorrigendum\b', r'\berratum\b', r'\bretraction\b', r'\bretracted\b',
        r'\bwithdraw(n|al)\b', r'\beditorial\b', r'\bguest\s+editor', r'\bpreface\b',
        r'\bforeword\b', r'^\d{4}\s+index\b', r'^(?:author|subject|keyword|volume)\s+index\b',
        r'\bin\s+memoriam\b', r'\bobituary\b',
    ]
    return any(re.search(p, t) for p in patterns)


def expand_journal(jid: str, jname: str, blacklist: Blacklist, api_key: str) -> int:
    """Expand one journal to TARGET papers. Returns number of papers added."""
    profile_path = Path(f"data/accepted_papers/{jid}.json")

    # Load existing
    existing_papers: list[dict] = []
    if profile_path.exists():
        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))
            existing_papers = data.get("papers", [])
        except Exception:
            pass

    existing_n = len(existing_papers)
    if existing_n >= TARGET:
        return 0

    existing_titles = {normalize_text(p.get("title", "")) for p in existing_papers if p.get("title")}

    # Fetch from S2
    candidates = fetch_s2_with_retry(jname, api_key)
    print(f"    S2: {len(candidates)} candidates", end="", flush=True)

    # Filter
    new_papers: list[dict] = []
    seen = set(existing_titles)
    for c in candidates:
        if len(new_papers) + existing_n >= TARGET:
            break
        title_norm = normalize_text(c["title"])
        if not title_norm or title_norm in seen:
            continue
        if not venue_matches(c["venue"], jname):
            continue
        if len(c["abstract"]) < MIN_ABSTRACT_LEN:
            continue
        if _has_leak(c["title"], c["abstract"], blacklist):
            continue
        if is_non_research_title(c["title"]):
            continue
        seen.add(title_norm)
        new_papers.append({
            "title": c["title"],
            "abstract": c["abstract"],
            "year": c["year"],
            "source": "semantic_scholar",
            "doi": c["doi"],
            "url": c["url"],
        })

    print(f" → {len(new_papers)} accepted", end="", flush=True)
    if new_papers and existing_papers:
        merge_profile(
            profile_path=profile_path, journal_id=jid, journal_name=jname,
            new_papers=[type('C', (), {
                'title': p['title'], 'abstract': p['abstract'],
                'venue': jname, 'year': p['year'],
                'doi': p['doi'], 'url': p['url'],
            })() for p in new_papers],
            target_total=TARGET, source="semantic_scholar",
        )
    elif new_papers and not existing_papers:
        # New journal: write directly
        merged = existing_papers + new_papers
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(json.dumps({
            "journal_id": jid, "journal_name": jname, "papers": merged,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f" ({existing_n}→{existing_n + len(new_papers)})")
    return len(new_papers)


def main() -> int:
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if not api_key:
        print("WARNING: SEMANTIC_SCHOLAR_API_KEY not set. Rate limits will be severe.", file=sys.stderr)

    blacklist = build_blacklist(BENCHMARKS)
    print(f"Blacklist: {len(blacklist.titles)} titles, {len(blacklist.abstract_snippets)} snippets")

    # Find journals to expand: new (0 papers) + thin (1-9 papers)
    # Load all from journal store
    journal_map = {}
    with open("data/processed/journals.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            journal_map[j["journal_id"]] = j

    # Current state
    current = {}
    for p in Path("data/accepted_papers").glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            current[d.get("journal_id", "")] = len(d.get("papers", []))
        except Exception:
            pass

    # Build priority queue: new first, then thin
    queue = []
    for jid, jinfo in sorted(journal_map.items(), key=lambda x: x[1]["journal_name"]):
        n = current.get(jid, 0)
        need = TARGET - n
        if need > 0:
            priority = 0 if n == 0 else 1  # 0=new, 1=thin
            queue.append((priority, jid, jinfo["journal_name"], n, need))

    queue.sort()
    print(f"\nJournals to expand: {len(queue)} ({sum(1 for p,_,_,n,_ in queue if n==0)} new, {sum(1 for p,_,_,n,_ in queue if n>0)} thin)")
    print()

    ok = fail = total_added = 0
    for idx, (_, jid, jname, current_n, need) in enumerate(queue, 1):
        print(f"[{idx}/{len(queue)}] {jid}: {current_n}→{TARGET} (need +{need})")
        try:
            added = expand_journal(jid, jname, blacklist, api_key)
            total_added += added
            if added > 0:
                ok += 1
            else:
                fail += 1
                print(f"    ✗ No papers added")
        except Exception as e:
            fail += 1
            print(f"    ✗ ERROR: {e}")

        # Sleep between calls (S2 allows ~1/sec with key, 0.1/sec without)
        if idx < len(queue):
            sleep = 1.0 if api_key else 10.0
            time.sleep(sleep)

    print(f"\n{'='*60}")
    print(f"Done: {ok} OK, {fail} failed, {total_added} papers added")
    return 0 if fail < len(queue) * 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
