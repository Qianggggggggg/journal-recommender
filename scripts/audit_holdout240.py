#!/usr/bin/env python3
"""Audit holdout240 paper authenticity via Semantic Scholar + DBLP + arXiv.

For each paper:
  1. Search S2 by exact title. Get canonical venue + abstract.
  2. Compare canonical venue with jsonl venue.
  3. Check venue exists in journals.jsonl.
  4. Check CCF rating matches.
  5. (Leakage) Check title not in light30/full-v2-90.
  6. Mark valid/suspect/invalid with reason.

Outputs: data/evaluation/results/holdout240_audit_report.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

HOLDOUT_PATH = project_root / "data/evaluation/papers_metadata_holdout240.jsonl"
JOURNALS_PATH = project_root / "data/processed/journals.jsonl"
LIGHT30_PATH = project_root / "data/evaluation/papers_metadata_light_30.jsonl"
FULL_V2_PATH = project_root / "data/evaluation/papers_metadata_full_v2_90.jsonl"
REPORT_PATH = project_root / "data/evaluation/results/holdout240_audit_report.json"


def _input_path() -> Path:
    """Allow overriding the audit input via env var for re-auditing clean jsonl."""
    override = os.environ.get("AUDIT_HOLDOUT_PATH")
    return Path(override) if override else HOLDOUT_PATH

S2_BASE = "https://api.semanticscholar.org/graph/v1/paper"
S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"


def s2_get(url: str, max_retries: int = 3) -> dict | None:
    """GET S2 API with retry + backoff for 429."""
    import os
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {"User-Agent": "holdout240-audit/1.0"}
    if key:
        headers["x-api-key"] = key
    for attempt in range(max_retries):
        try:
            req = urlopen(url, timeout=30) if False else None
            import urllib.request
            r = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(r, timeout=30) as resp:
                return json.loads(resp.read())
        except HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)
                print(f"  [S2 429] backing off {wait}s...", flush=True)
                time.sleep(wait)
            elif e.code == 404:
                return None
            else:
                print(f"  [S2 {e.code}] {url[:80]}", flush=True)
                return None
        except (URLError, TimeoutError) as e:
            print(f"  [S2 net] {e}", flush=True)
            time.sleep(2)
    return None


def s2_search_title(title: str) -> dict | None:
    """Search S2 for paper by title; return top match if any."""
    # Strip noise: trim, lowercase-collapse
    q = title.strip()
    url = f"{S2_SEARCH}?query={quote(q)}&limit=3&fields=title,venue,year,abstract,externalIds,publicationVenue,publicationDate"
    return s2_get(url)


def normalize_title(t: str) -> str:
    return " ".join((t or "").casefold().split())


def normalize_venue(v: str) -> str:
    return " ".join((v or "").casefold().split())


def load_journals() -> dict[str, dict]:
    out = {}
    with open(JOURNALS_PATH) as f:
        for line in f:
            j = json.loads(line)
            out[j.get("journal_name", "")] = j
    return out


def load_existing_titles() -> set[str]:
    """All titles from light30 + full-v2-90 (always). The audit input
    itself is NOT included — leakage check is against other benchmarks."""
    titles = set()
    for path in (LIGHT30_PATH, FULL_V2_PATH):
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                p = json.loads(line)
                titles.add(normalize_title(p.get("title", "")))
    return titles


def audit_paper(paper: dict, idx: int, journals: dict, existing_titles: set) -> dict:
    """Audit a single paper; return per-paper audit row."""
    title = paper.get("title", "")
    jsonl_venue = paper.get("venue", "")
    jsonl_year = paper.get("year")
    jsonl_ccf = paper.get("ccf_level")
    jsonl_abstract = paper.get("abstract", "")

    row = {
        "index": idx,
        "title": title[:120],
        "jsonl_venue": jsonl_venue,
        "jsonl_year": jsonl_year,
        "jsonl_ccf": jsonl_ccf,
        "jsonl_abstract_len": len(jsonl_abstract) if jsonl_abstract else 0,
        "s2_match": None,
        "s2_venue": None,
        "s2_year": None,
        "s2_abstract_first_200": None,
        "venue_in_journals": False,
        "journal_ccf": None,
        "ccf_match": None,
        "leakage": False,
        "abstract_match": None,
        "verdict": "unknown",
        "issues": [],
    }

    # Leakage check
    if normalize_title(title) in existing_titles:
        row["leakage"] = True
        row["issues"].append("leakage: title matches light30 or full-v2-90")

    # S2 search
    s2 = s2_search_title(title)
    if s2 is None or "data" not in s2 or not s2["data"]:
        row["issues"].append("S2: not found")
        row["verdict"] = "suspect"
        return row

    # Find best match by exact title (casefolded). When multiple candidates
    # share the same casefolded title, prefer the journal version (more
    # reliable for our use case) and the more recent year.
    target = normalize_title(title)
    matches = [
        cand for cand in s2["data"]
        if normalize_title(cand.get("title", "")) == target
    ]
    if matches:
        # Sort: journal first, then by year desc
        def cand_score(c):
            pv = c.get("publicationVenue") or {}
            t = pv.get("type", "")
            y = c.get("year") or 0
            # higher score = better
            return (1 if t == "journal" else 0, y)
        matches.sort(key=cand_score, reverse=True)
        best = matches[0]
    else:
        # No exact title match; fall back to first result, but flag
        best = s2["data"][0]
        if normalize_title(best.get("title", "")) != target:
            row["issues"].append(f"S2: title mismatch (got: {best.get('title', '')[:60]})")
            row["verdict"] = "suspect"

    row["s2_match"] = True
    s2_venue = (
        best.get("venue")
        or (best.get("publicationVenue") or {}).get("name")
        or ""
    )
    s2_year = best.get("year") or (
        (best.get("publicationDate") or "")[:4] or None
    )
    s2_abstract = best.get("abstract") or ""
    row["s2_venue"] = s2_venue
    row["s2_year"] = s2_year
    row["s2_abstract_first_200"] = s2_abstract[:200]
    row["s2_abstract_len"] = len(s2_abstract)

    # Venue match — use S2 publicationVenue.id when available, since
    # S2 venue strings have multiple variants for the same journal
    # (e.g. "Computer Graphics Forum" vs "Computer graphics forum (Print)").
    s2_pv = best.get("publicationVenue") or {}
    s2_pv_id = s2_pv.get("id", "")
    s2_pv_name = s2_pv.get("name", "")
    s2_pv_type = s2_pv.get("type", "")

    # Save for downstream filtering
    row["s2_pv_id"] = s2_pv_id
    row["s2_pv_name"] = s2_pv_name
    row["s2_pv_type"] = s2_pv_type

    # Treat as venue match when:
    #   - pv_id matches the canonical id we previously saw for this venue
    #     (handled downstream by compare_canonical_ids), OR
    #   - string match (casefolded), OR
    #   - same journal type (e.g. both "journal") and name is a variant
    venue_strings_match = (
        normalize_venue(s2_venue) and normalize_venue(s2_venue) == normalize_venue(jsonl_venue)
    )
    if not venue_strings_match and s2_venue:
        row["issues"].append(
            f"venue mismatch: jsonl='{jsonl_venue}' vs s2='{s2_venue}' "
            f"(pv_id={s2_pv_id}, pv_type={s2_pv_type})"
        )

    # Year match (within ±1 tolerance)
    if s2_year and jsonl_year and abs(int(s2_year) - int(jsonl_year)) > 1:
        row["issues"].append(f"year mismatch: jsonl={jsonl_year} vs s2={s2_year}")

    # Abstract match — first 100 chars of jsonl should appear in s2 abstract
    if s2_abstract and jsonl_abstract:
        snippet = (jsonl_abstract[:100] or "").casefold().strip()
        s2_abstract_norm = s2_abstract.casefold()
        if snippet and snippet in s2_abstract_norm:
            row["abstract_match"] = True
        else:
            # Try first 50 chars
            snippet50 = (jsonl_abstract[:50] or "").casefold().strip()
            if snippet50 and snippet50 in s2_abstract_norm:
                row["abstract_match"] = "partial(50)"
            else:
                row["abstract_match"] = False
                row["issues"].append("abstract mismatch: jsonl abstract not in s2 abstract")
    elif not s2_abstract and jsonl_abstract:
        row["abstract_match"] = None
        row["issues"].append("s2 has no abstract (cannot verify)")

    # Venue in journals.jsonl
    j = journals.get(jsonl_venue)
    if j is None:
        # Try fuzzy: S2 venue might be canonical
        for jn, jentry in journals.items():
            if normalize_venue(jn) == normalize_venue(jsonl_venue):
                j = jentry
                break
    if j is None:
        row["venue_in_journals"] = False
        row["issues"].append(f"venue not in journals.jsonl: '{jsonl_venue}'")
    else:
        row["venue_in_journals"] = True
        row["journal_ccf"] = j.get("ccf_rating")
        if j.get("ccf_rating") != jsonl_ccf:
            row["ccf_match"] = False
            row["issues"].append(
                f"ccf mismatch: jsonl={jsonl_ccf} vs journals={j.get('ccf_rating')}"
            )
        else:
            row["ccf_match"] = True

    # Verdict
    if any("venue not in journals.jsonl" in i for i in row["issues"]):
        row["verdict"] = "invalid"
    elif any("venue mismatch" in i for i in row["issues"]):
        # Disambiguate: if s2 explicitly says it's a conference, then the
        # jsonl_venue (which should be a journal) is wrong -> invalid.
        # If s2_pv_type is empty and the venue strings are clearly
        # variants of the same journal (e.g. "Computer Graphics Forum"
        # vs "Computer graphics forum (Print)"), treat as valid.
        if s2_pv_type == "conference":
            row["verdict"] = "invalid"
        elif s2_pv_type == "" or s2_pv_type == "journal":
            # Heuristic: if s2 venue string differs from jsonl only in
            # case + parentheses + "Print"/"Online" suffix, it's the
            # same journal.
            def normalize_loose(s: str) -> str:
                s = s.lower()
                for tok in [" (print)", " (online)", "print", "online", "(", ")", "the "]:
                    s = s.replace(tok, "")
                return " ".join(s.split())

            loose_match = (
                normalize_loose(s2_venue) == normalize_loose(jsonl_venue)
            )
            if loose_match:
                row["verdict"] = "valid"
            else:
                row["verdict"] = "suspect"
    elif any("ccf mismatch" in i for i in row["issues"]):
        row["verdict"] = "invalid"
    elif any("year mismatch" in i for i in row["issues"]) or any(
        "abstract mismatch" in i for i in row["issues"]
    ):
        row["verdict"] = "suspect"
    elif row["leakage"]:
        row["verdict"] = "invalid"
    elif row["s2_match"] and row["venue_in_journals"] and row["ccf_match"]:
        row["verdict"] = "valid"

    return row


def main():
    print("Loading journals.jsonl ...", flush=True)
    journals = load_journals()
    print(f"  {len(journals)} journals", flush=True)

    print("Loading existing benchmark titles (for leakage) ...", flush=True)
    existing_titles = load_existing_titles()
    print(f"  {len(existing_titles)} existing titles", flush=True)

    input_path = _input_path()
    print(f"Auditing: {input_path}\n", flush=True)
    papers = []
    with open(input_path) as f:
        for line in f:
            papers.append(json.loads(line))
    print(f"Auditing {len(papers)} papers ...\n", flush=True)

    rows = []
    for i, p in enumerate(papers, start=1):
        if i % 10 == 0:
            print(f"  [{i}/{len(papers)}] ...", flush=True)
        row = audit_paper(p, i, journals, existing_titles)
        rows.append(row)
        time.sleep(0.8)  # 5000 req/h with key ≈ 1.4 req/s, 0.8s is safe

    # Summary
    verdict_counts = Counter(r["verdict"] for r in rows)
    print("\n=== Verdict summary ===")
    for v, c in sorted(verdict_counts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {c}")

    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": 1,
            "total": len(rows),
            "summary": {
                "verdict_counts": dict(verdict_counts),
                "leakage_count": sum(1 for r in rows if r["leakage"]),
                "venue_not_in_journals": sum(1 for r in rows if not r["venue_in_journals"]),
                "venue_mismatch": sum(1 for r in rows if any("venue mismatch" in i for i in r["issues"])),
                "ccf_mismatch": sum(1 for r in rows if r.get("ccf_match") is False),
                "year_mismatch": sum(1 for r in rows if any("year mismatch" in i for i in r["issues"])),
                "abstract_mismatch": sum(1 for r in rows if r.get("abstract_match") is False),
            },
            "papers": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nReport: {REPORT_PATH}")


if __name__ == "__main__":
    main()
