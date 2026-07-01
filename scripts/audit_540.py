#!/usr/bin/env python3
"""Audit 540 candidate papers via Semantic Scholar + DBLP + arXiv.

For each candidate paper:
  1. S2 search by exact title; get canonical venue + abstract + year.
  2. Compare canonical venue with jsonl venue.
  3. Check venue exists in journals.jsonl.
  4. Check CCF rating matches.
  5. Check year matches.
  6. Check abstract exists and is non-trivial (≥ 160 chars).
  7. Cross-check via arXiv journal-ref (if arXiv ID present).
  8. Cross-check via DBLP title search.
  9. Mark valid / suspect / invalid with reason.

Outputs:
  - data/evaluation/results/540_audit_report.json
  - data/evaluation/results/540_invalid_titles.json (just invalid papers, for
    replacement)
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

RAW_PATH = project_root / "data/evaluation/papers_metadata_540_raw.jsonl"
JOURNALS_PATH = project_root / "data/processed/journals.jsonl"
REPORT_PATH = project_root / "data/evaluation/results/540_audit_report.json"
INVALID_PATH = project_root / "data/evaluation/results/540_invalid_titles.json"

S2_BASE = "https://api.semanticscholar.org/graph/v1/paper"
S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API = "http://export.arxiv.org/api/query"
DBLP_SEARCH = "https://dblp.org/search/publ/api"


def s2_get(url: str, max_retries: int = 3) -> dict | None:
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {"User-Agent": "540-audit/1.0"}
    if key:
        headers["x-api-key"] = key
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)
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


def arxiv_journal_ref(arxiv_id: str) -> str | None:
    """Fetch arXiv paper's journal-ref field. Returns None on any error."""
    if not arxiv_id:
        return None
    url = f"{ARXIV_API}?id_list={urllib.parse.quote(arxiv_id)}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            xml = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    # crude extraction
    import re
    m = re.search(r"<arxiv:journal_ref[^>]*>(.*?)</arxiv:journal_ref>", xml)
    if not m:
        m = re.search(r"<journal_ref[^>]*>(.*?)</journal_ref>", xml)
    return m.group(1).strip() if m else None


def dblp_search_title(title: str) -> dict | None:
    """DBLP title search. Returns first hit's venue/year or None."""
    q = urllib.parse.quote(title)
    url = f"{DBLP_SEARCH}?q={q}&format=json&h=3"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None
    hits = data.get("result", {}).get("hits", {}).get("hit", [])
    if not hits:
        return None
    info = hits[0].get("info", {})
    return {
        "venue": info.get("venue", ""),
        "year": info.get("year", ""),
        "title": info.get("title", ""),
    }


def normalize_title(t: str) -> str:
    return " ".join((t or "").casefold().split())


def load_journal_lookup() -> dict[str, dict]:
    """journal_name (casefold) → {ccf_rating, subject_tags, journal_id}"""
    out: dict[str, dict] = {}
    with JOURNALS_PATH.open() as f:
        for line in f:
            j = json.loads(line)
            ccf = (
                j.get("ccf_rating")
                or j.get("ccf_rank")
                or j.get("ccf")
                or j.get("cf_rating")
            )
            if ccf is None:
                continue
            key = j["journal_name"].casefold().strip()
            out[key] = {
                "ccf_rating": ccf,
                "subject_tags": j.get("subject_tags", []),
                "journal_id": j.get("journal_id"),
            }
    return out


def audit_paper(paper: dict, journal_lookup: dict) -> dict:
    """Audit a single paper, return verdict dict."""
    title = paper.get("title", "")
    expected_venue = paper.get("venue", "")
    expected_ccf = paper.get("ccf_level", "")
    expected_year = paper.get("year")
    arxiv_id = paper.get("arxiv", "")

    result: dict = {
        "title": title,
        "expected_venue": expected_venue,
        "expected_ccf": expected_ccf,
        "expected_year": expected_year,
        "verdict": "invalid",
        "reasons": [],
        "sources": {},
    }

    # 1. S2 search by title
    s2_url = (
        f"{S2_SEARCH}?query={urllib.parse.quote(title)}&limit=5"
        f"&fields=title,venue,publicationVenue,year,abstract,externalIds"
    )
    s2 = s2_get(s2_url)
    time.sleep(0.3)

    s2_match = None
    if s2 and s2.get("data"):
        target = normalize_title(title)
        for cand in s2["data"]:
            if normalize_title(cand.get("title", "")) == target:
                s2_match = cand
                break
        if s2_match is None and s2["data"]:
            # No exact match; record closest
            result["sources"]["s2_closest_title"] = s2["data"][0].get("title")
    result["sources"]["s2_found"] = s2_match is not None

    # 2. Check S2 venue
    s2_venue = ""
    if s2_match:
        pv = s2_match.get("publicationVenue") or {}
        s2_venue = (pv.get("name") or s2_match.get("venue") or "").casefold().strip()
    if s2_match and s2_venue and s2_venue != expected_venue.casefold().strip():
        result["reasons"].append(
            f"venue_mismatch:s2='{s2_venue}' expected='{expected_venue.casefold()}'"
        )

    # 3. Check venue exists in journals.jsonl
    j_match = journal_lookup.get(expected_venue.casefold().strip())
    if not j_match:
        result["reasons"].append(f"venue_not_in_journals:{expected_venue}")
    else:
        if j_match["ccf_rating"] != expected_ccf:
            result["reasons"].append(
                f"ccf_mismatch:journals={j_match['ccf_rating']} expected={expected_ccf}"
            )

    # 4. Check year
    if s2_match and s2_match.get("year") and expected_year:
        if int(s2_match["year"]) != int(expected_year):
            result["reasons"].append(
                f"year_mismatch:s2={s2_match['year']} expected={expected_year}"
            )

    # 5. Check abstract
    abstract = paper.get("abstract", "") or ""
    if len(abstract) < 160:
        result["reasons"].append(f"abstract_too_short:{len(abstract)}")

    # 6. arXiv journal-ref cross-check (only if arXiv ID present)
    if arxiv_id:
        jr = arxiv_journal_ref(arxiv_id)
        result["sources"]["arxiv_journal_ref"] = jr
        if jr and expected_venue:
            if expected_venue.casefold() not in jr.casefold():
                result["reasons"].append(f"arxiv_journal_ref_mismatch:{jr}")

    # 7. DBLP title search
    dblp = dblp_search_title(title)
    result["sources"]["dblp"] = dblp
    # DBLP is informational, not a hard reject (some S2-only papers have no DBLP)

    # Decide verdict
    if not result["reasons"]:
        result["verdict"] = "valid"
    elif any(
        r.startswith("venue_") or r.startswith("ccf_") or r == "venue_not_in_journals"
        for r in result["reasons"]
    ):
        result["verdict"] = "invalid"
    else:
        result["verdict"] = "suspect"
    return result


def main() -> int:
    if not RAW_PATH.exists():
        print(f"Missing {RAW_PATH}; run scripts/augment_540_corpus.py first")
        return 1

    journal_lookup = load_journal_lookup()
    print(f"Loaded {len(journal_lookup)} journals from {JOURNALS_PATH}", flush=True)

    papers = [json.loads(line) for line in RAW_PATH.open() if line.strip()]
    print(f"Auditing {len(papers)} papers ...\n", flush=True)

    audits = []
    verdict_counter: Counter = Counter()
    for i, p in enumerate(papers):
        print(f"[{i+1}/{len(papers)}] {p.get('title','')[:60]}", flush=True)
        result = audit_paper(p, journal_lookup)
        audits.append(result)
        verdict_counter[result["verdict"]] += 1
        print(f"   verdict={result['verdict']} reasons={result['reasons']}", flush=True)

    report = {
        "schema_version": 1,
        "input_path": str(RAW_PATH),
        "total_audited": len(papers),
        "verdict_counts": dict(verdict_counter),
        "audits": audits,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {REPORT_PATH}")

    invalid = [a for a in audits if a["verdict"] == "invalid"]
    suspect = [a for a in audits if a["verdict"] == "suspect"]
    INVALID_PATH.write_text(json.dumps(
        {"invalid": invalid, "suspect": suspect}, ensure_ascii=False, indent=2
    ))
    print(f"Wrote {INVALID_PATH} ({len(invalid)} invalid, {len(suspect)} suspect)")

    print(f"\n=== Summary ===")
    print(f"  total: {len(papers)}")
    print(f"  valid: {verdict_counter['valid']}")
    print(f"  suspect: {verdict_counter['suspect']}")
    print(f"  invalid: {verdict_counter['invalid']}")
    return 0 if verdict_counter["invalid"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
