#!/usr/bin/env python3
"""Fill thin accepted-paper profiles using Crossref + publisher APIs + CORE.

Strategy per the data-source guide:
  1. Crossref REST API → get ISSN, DOI candidates, verify journal
  2. Springer/Elsevier API → get abstracts (premium quality)
  3. CORE API → fallback abstracts for papers without them
  4. Filter: benchmark blacklist, venue verification, abstract quality

Works for the 9 stubborn journals that S2/OpenAlex couldn't fill.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expand_accepted_paper_corpus import (
    build_blacklist,
    merge_profile,
    normalize_text,
    normalize_venue,
    venue_matches,
)

TARGETS = {
    # Elsevier: ScienceDirect
    "cad": "Computer-Aided Design",
    "cl": "Computer Languages, Systems and Structures",
    "fuzzysetsandsystems": "Fuzzy Sets and Systems",
    "hcc": "High-Confidence Computing",
    # Springer
    "cscw": "Computer Supported Cooperative Work",
    "geoinformatica": "GeoInformatica",
    "ijcv": "International Journal of Computer Vision",
    "tjsc": "The Journal of Supercomputing",
    # Wiley
    "computationalintelligence": "Computational Intelligence",
}

TARGET_TOTAL = 10
YEAR_MIN = 2018
YEAR_MAX = 2026
MIN_ABSTRACT_LEN = 150

BENCHMARKS = [
    Path("data/evaluation/papers_metadata_660_balanced.jsonl"),
    Path("data/evaluation/papers_metadata_holdout240.jsonl"),
    Path("data/evaluation/papers_metadata_full_v2_90.jsonl"),
    Path("data/evaluation/papers_metadata_light_30.jsonl"),
]

UA = "journal-recommender-crossref/1.0 (mailto:paper-recommender@example.com)"
CROSSREF_BASE = "https://api.crossref.org"


def _fetch_json(url: str, *, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def lookup_issn_by_name(name: str) -> str | None:
    """Search Crossref for a journal's ISSN."""
    params = urllib.parse.urlencode({"query": name, "rows": "3"})
    url = f"{CROSSREF_BASE}/journals?{params}"
    try:
        data = _fetch_json(url)
    except Exception as e:
        print(f"    Crossref journal search error: {e}")
        return None
    for item in data.get("message", {}).get("items", []):
        if venue_matches(item.get("title", ""), name):
            return item.get("ISSN", [None])[0]
    # Fallback: first result
    items = data.get("message", {}).get("items", [])
    if items:
        return items[0].get("ISSN", [None])[0]
    return None


def fetch_crossref_papers(issn: str, *, rows: int = 100) -> list[dict]:
    """Fetch recent papers for a journal ISSN from Crossref."""
    all_papers = []
    cursor = "*"
    for _ in range(5):  # max 5 pages (500 papers)
        params = {
            "filter": f"issn:{issn},from-pub-date:{YEAR_MIN}-01-01,until-pub-date:{YEAR_MAX}-12-31,type:journal-article",
            "rows": str(rows),
            "cursor": cursor,
            "select": "DOI,title,abstract,published-print,container-title,author",
        }
        url = f"{CROSSREF_BASE}/works?{urllib.parse.urlencode(params)}"
        try:
            data = _fetch_json(url)
        except Exception as e:
            print(f"    Crossref works error: {e}")
            break
        msg = data.get("message", {})
        items = msg.get("items", [])
        for item in items:
            title = item.get("title", [""])[0] if item.get("title") else ""
            abstract = extract_abstract(item)
            doi = item.get("DOI", "")
            year = None
            pp = item.get("published-print")
            if pp and "date-parts" in pp and pp["date-parts"]:
                year = pp["date-parts"][0][0]
            container = item.get("container-title", [""])[0] if item.get("container-title") else ""
            if title:
                all_papers.append({
                    "title": title.strip(),
                    "abstract": abstract.strip(),
                    "doi": doi,
                    "year": year,
                    "container": container.strip(),
                    "source": "crossref",
                })
        cursor = msg.get("next-cursor", "")
        if not cursor or len(all_papers) >= 200:
            break
        time.sleep(0.5)
    return all_papers


def extract_abstract(item: dict) -> str:
    """Extract abstract from Crossref work item."""
    # Direct abstract field
    abstract = item.get("abstract", "")
    if abstract and len(abstract.strip()) > 50:
        # Strip HTML tags
        abstract = re.sub(r"<[^>]+>", " ", abstract)
        abstract = re.sub(r"\s+", " ", abstract).strip()
        # Remove "Abstract" prefix
        abstract = re.sub(r"^\s*Abstract\s*[:.\-]?\s*", "", abstract, flags=re.IGNORECASE)
        return abstract
    return ""


def fetch_core_by_doi(doi: str) -> str | None:
    """Try to get abstract from CORE API by DOI."""
    url = f"https://api.core.ac.uk/v3/search/works?doi={urllib.parse.quote(doi)}&limit=1"
    try:
        data = _fetch_json(url, timeout=15)
    except Exception:
        return None
    results = data.get("results", [])
    if results:
        abstract = results[0].get("abstract", "")
        if abstract and len(abstract.strip()) > 100:
            return abstract.strip()
    return None


def fetch_s2_abstract_by_doi(doi: str) -> str | None:
    """Get abstract from Semantic Scholar by DOI (exact paper lookup)."""
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi)}?fields=title,abstract,year"
    headers = {"User-Agent": UA}
    if api_key:
        headers["x-api-key"] = api_key
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception:
        return None
    abstract = data.get("abstract", "") or ""
    if abstract and len(abstract.strip()) > 100:
        return abstract.strip()
    return None


def fetch_springer_abstract(doi: str) -> str | None:
    """Try Springer Nature Metadata API for abstract."""
    api_key = os.environ.get("SPRINGER_API_KEY", "")
    if not api_key:
        return None
    url = f"https://api.springernature.com/metadata/json?q=doi:{urllib.parse.quote(doi)}&api_key={api_key}"
    try:
        data = _fetch_json(url, timeout=15)
    except Exception:
        return None
    records = data.get("records", [])
    if records:
        abstract = records[0].get("abstract", "")
        if abstract and len(abstract.strip()) > 100:
            return abstract.strip()
    return None


def is_non_research(title: str) -> bool:
    """Filter out editorials, corrigenda, indexes, etc."""
    t = title.lower()
    patterns = [
        r"\bcorrigendum\b", r"\berratum\b", r"\bretraction\b", r"\bretracted\b",
        r"\bwithdraw(n|al)\b", r"\beditorial\b", r"\bguest\s+editor",
        r"\bpreface\b", r"\bforeword\b", r"\bspecial\s+issue\b",
        r"^\d{4}\s+index\b", r"^(?:author|subject|keyword|volume)\s+index\b",
        r"\bin\s+memoriam\b", r"\bobituary\b",
    ]
    return any(re.search(p, t) for p in patterns)


def fill_journal(
    jid: str,
    jname: str,
    blacklist: Any,
    *,
    issn: str | None = None,
) -> int:
    """Fill one journal to TARGET_TOTAL papers. Returns number added."""
    profile_path = Path(f"data/accepted_papers/{jid}.json")

    existing_papers: list[dict] = []
    existing_titles: set[str] = set()
    if profile_path.exists():
        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))
            existing_papers = data.get("papers", [])
            existing_titles = {normalize_text(p.get("title", "")) for p in existing_papers if p.get("title")}
        except Exception:
            pass

    current_n = len(existing_papers)
    need = TARGET_TOTAL - current_n
    if need <= 0:
        return 0

    # Step 1: Look up ISSN via Crossref if not provided
    if not issn:
        print(f"    Looking up ISSN for '{jname[:50]}'...")
        issn = lookup_issn_by_name(jname)
    if not issn:
        print(f"    ✗ No ISSN found")
        return 0
    print(f"    ISSN: {issn}")

    # Step 2: Fetch papers from Crossref
    print(f"    Fetching Crossref papers ({YEAR_MIN}-{YEAR_MAX})...")
    candidates = fetch_crossref_papers(issn)
    print(f"    Crossref returned {len(candidates)} candidates")

    # Step 3: Filter and collect
    new_papers = []
    for c in candidates:
        if len(new_papers) >= need:
            break

        title_norm = normalize_text(c["title"])
        if not title_norm or title_norm in existing_titles:
            continue

        # Venue verification
        if c["container"] and not venue_matches(c["container"], jname):
            # Allow loose match for journal abbreviations
            container_norm = normalize_venue(c["container"])
            target_norm = normalize_venue(jname)
            if container_norm not in target_norm and target_norm not in container_norm:
                # Check token overlap
                tokens_c = set(container_norm.split())
                tokens_t = set(target_norm.split())
                overlap = tokens_c & tokens_t
                if len(overlap) < min(len(tokens_c), len(tokens_t)) * 0.4:
                    continue

        # Quality filters
        if is_non_research(c["title"]):
            continue

        # Benchmark leak check
        if normalize_text(c["title"]) in blacklist.titles:
            continue

        abstract = c["abstract"]
        # Try to enrich abstract from Springer API if applicable
        if not abstract and c["doi"] and jid in ("cscw", "geoinformatica", "ijcv", "tjsc"):
            abstract = fetch_springer_abstract(c["doi"]) or ""
            if abstract:
                print(f"      Springer abstract: {len(abstract)} chars")

        # CORE fallback (often unreliable, skip)
        if not abstract and c["doi"]:
            abstract = fetch_s2_abstract_by_doi(c["doi"]) or ""
            if abstract:
                print(f"      S2 abstract: {len(abstract)} chars")

        if not abstract or len(abstract.strip()) < MIN_ABSTRACT_LEN:
            continue

        new_papers.append({
            "title": c["title"],
            "abstract": abstract,
            "year": c["year"],
            "source": "crossref",
            "doi": c["doi"],
            "url": f"https://doi.org/{c['doi']}" if c["doi"] else "",
        })
        existing_titles.add(title_norm)

    print(f"    After filtering: {len(new_papers)} new papers")

    if not new_papers:
        return 0

    # Step 4: Write using merge_profile
    from scripts.expand_accepted_paper_corpus import CandidatePaper
    cpapers = [
        CandidatePaper(
            title=p["title"], abstract=p["abstract"], venue=jname,
            year=p["year"], doi=p["doi"], url=p["url"],
        )
        for p in new_papers
    ]
    added = merge_profile(
        profile_path=profile_path, journal_id=jid, journal_name=jname,
        new_papers=cpapers, target_total=TARGET_TOTAL, source="crossref",
    )
    return added


def main() -> int:
    blacklist = build_blacklist(BENCHMARKS)
    print(f"Blacklist: {len(blacklist.titles)} titles")
    print()

    ok, fail, total = 0, 0, 0
    for idx, (jid, jname) in enumerate(TARGETS.items(), 1):
        # Check current count
        profile_path = Path(f"data/accepted_papers/{jid}.json")
        current_n = 0
        if profile_path.exists():
            try:
                d = json.loads(profile_path.read_text(encoding="utf-8"))
                current_n = len(d.get("papers", []))
            except Exception:
                pass

        need = TARGET_TOTAL - current_n
        if need <= 0:
            print(f"[{idx}/{len(TARGETS)}] {jid}: {current_n} papers ✓ (done)")
            ok += 1
            continue

        print(f"[{idx}/{len(TARGETS)}] {jid}: {current_n}→{TARGET_TOTAL} (need +{need})")
        try:
            added = fill_journal(jid, jname, blacklist)
            total += added
            if added > 0:
                ok += 1
                print(f"    ✓ +{added} ({current_n}→{current_n + added})")
            else:
                fail += 1
                print(f"    ✗ No papers added")
        except Exception as e:
            fail += 1
            print(f"    ✗ ERROR: {e}")
        print()

        if idx < len(TARGETS):
            time.sleep(2)  # Polite delay

    print(f"{'='*60}")
    print(f"Done: +{total} papers, {ok} OK, {fail} failed")
    return 0 if fail < len(TARGETS) * 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
