#!/usr/bin/env python3
"""Expand accepted-paper profiles from external scholarly metadata.

The script targets benchmark gold venues that are not yet represented under
``data/accepted_papers``. It keeps benchmark papers out of the accepted corpus
by blacklisting both normalized titles and abstract snippets from the selected
evaluation JSONL files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ABSTRACT_SNIPPET_LENGTH = 160
DEFAULT_BENCHMARK_INPUTS = [
    Path("data/evaluation/papers_metadata_full_v2_90.jsonl"),
    Path("data/evaluation/papers_metadata_light_30.jsonl"),
]


@dataclass(frozen=True)
class CandidatePaper:
    title: str
    abstract: str
    venue: str
    year: int | None
    doi: str
    url: str


@dataclass(frozen=True)
class Blacklist:
    titles: set[str]
    abstract_snippets: set[str]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).lower()
    chars = [ch if ch.isalnum() else " " for ch in text]
    return " ".join("".join(chars).split())


def normalize_venue(venue: str) -> str:
    return " ".join(str(venue).strip().lower().split())


def abstract_snippet(abstract: str) -> str:
    return normalize_text(abstract)[:ABSTRACT_SNIPPET_LENGTH]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_journal_index(path: Path) -> dict[str, dict[str, Any]]:
    journals: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        name = row.get("journal_name", "")
        if name:
            journals[normalize_venue(name)] = row
    return journals


def build_blacklist(benchmark_inputs: Sequence[Path]) -> Blacklist:
    titles: set[str] = set()
    abstract_snippets: set[str] = set()
    for path in benchmark_inputs:
        for row in iter_jsonl(path):
            title = normalize_text(row.get("title", ""))
            if title:
                titles.add(title)
            snippet = abstract_snippet(row.get("abstract", ""))
            if len(snippet) >= ABSTRACT_SNIPPET_LENGTH:
                abstract_snippets.add(snippet)
    return Blacklist(titles=titles, abstract_snippets=abstract_snippets)


def accepted_profile_names(accepted_dir: Path) -> set[str]:
    names: set[str] = set()
    if not accepted_dir.exists():
        return names
    for path in accepted_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        papers = data.get("papers", [])
        if not isinstance(papers, list) or len(papers) == 0:
            continue
        name = data.get("journal_name", "")
        if name:
            names.add(normalize_venue(name))
    return names


def target_uncovered_venues(
    benchmark_inputs: Sequence[Path],
    accepted_dir: Path,
    journal_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    accepted_names = accepted_profile_names(accepted_dir)
    counts: dict[str, int] = {}
    for path in benchmark_inputs:
        for row in iter_jsonl(path):
            venue_norm = normalize_venue(row.get("venue", ""))
            if not venue_norm or venue_norm in accepted_names:
                continue
            if venue_norm not in journal_index:
                continue
            counts[venue_norm] = counts.get(venue_norm, 0) + 1

    targets = []
    for venue_norm, count in sorted(counts.items(), key=lambda item: journal_index[item[0]]["journal_name"]):
        journal = journal_index[venue_norm]
        targets.append(
            {
                "journal_id": journal["journal_id"],
                "journal_name": journal["journal_name"],
                "benchmark_count": count,
            }
        )
    return targets


def _candidate_has_leak(candidate: CandidatePaper, blacklist: Blacklist) -> bool:
    if normalize_text(candidate.title) in blacklist.titles:
        return True
    snippet = abstract_snippet(candidate.abstract)
    return len(snippet) >= ABSTRACT_SNIPPET_LENGTH and snippet in blacklist.abstract_snippets


def filter_candidates(
    candidates: Sequence[CandidatePaper],
    *,
    target_venue: str,
    blacklist: Blacklist,
    existing_titles: set[str],
    limit: int,
) -> list[CandidatePaper]:
    accepted: list[CandidatePaper] = []
    seen = set(existing_titles)
    target_norm = normalize_venue(target_venue)
    for candidate in candidates:
        title_norm = normalize_text(candidate.title)
        if not title_norm or title_norm in seen:
            continue
        if normalize_venue(candidate.venue) != target_norm:
            continue
        if len(candidate.abstract.strip()) < 300:
            continue
        if _candidate_has_leak(candidate, blacklist):
            continue
        seen.add(title_norm)
        accepted.append(candidate)
        if len(accepted) >= limit:
            break
    return accepted


def _load_profile(path: Path, journal_id: str, journal_name: str) -> dict[str, Any]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("journal_id", journal_id)
                data.setdefault("journal_name", journal_name)
                data.setdefault("papers", [])
                if isinstance(data["papers"], list):
                    return data
        except (OSError, json.JSONDecodeError):
            pass
    return {"journal_id": journal_id, "journal_name": journal_name, "papers": []}


def merge_profile(
    *,
    profile_path: Path,
    journal_id: str,
    journal_name: str,
    new_papers: Sequence[CandidatePaper],
    target_total: int,
    source: str,
) -> int:
    data = _load_profile(profile_path, journal_id, journal_name)
    papers = data["papers"]
    existing_titles = {
        normalize_text(paper.get("title", "")) for paper in papers if isinstance(paper, dict)
    }
    added = 0
    for paper in new_papers:
        if len(papers) >= target_total:
            break
        title_norm = normalize_text(paper.title)
        if title_norm in existing_titles:
            continue
        papers.append(
            {
                "title": paper.title,
                "abstract": paper.abstract,
                "year": paper.year,
                "source": source,
                "doi": paper.doi,
                "url": paper.url,
            }
        )
        existing_titles.add(title_norm)
        added += 1

    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return added


def _semantic_scholar_url(venue: str, *, limit: int, year: str) -> str:
    params = {
        "query": venue,
        "venue": venue,
        "limit": str(limit),
        "year": year,
        "fields": "title,abstract,venue,year,externalIds,url",
    }
    return "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params)


def build_semantic_scholar_request(url: str, *, api_key: str = "") -> urllib.request.Request:
    headers = {"User-Agent": "journal-recommender-corpus-expander/1.0"}
    if api_key:
        headers["x-api-key"] = api_key
    return urllib.request.Request(url, headers=headers)


def reconstruct_openalex_abstract(inverted_index: Any) -> str:
    if not isinstance(inverted_index, dict):
        return ""
    positions: dict[int, str] = {}
    for token, indexes in inverted_index.items():
        if not isinstance(indexes, list):
            continue
        for index in indexes:
            if isinstance(index, int):
                positions[index] = str(token)
    return " ".join(positions[index] for index in sorted(positions))


def _strip_doi_url(doi: str) -> str:
    return str(doi or "").replace("https://doi.org/", "").replace("http://doi.org/", "").strip()


def candidate_from_openalex_work(
    work: dict[str, Any], *, target_venue: str
) -> CandidatePaper | None:
    source = (
        ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
        or ""
    )
    if normalize_venue(source) != normalize_venue(target_venue):
        return None

    title = (work.get("title") or "").strip()
    abstract = reconstruct_openalex_abstract(work.get("abstract_inverted_index"))
    if not title or not abstract:
        return None
    return CandidatePaper(
        title=title,
        abstract=abstract,
        venue=source.strip(),
        year=work.get("publication_year")
        if isinstance(work.get("publication_year"), int)
        else None,
        doi=_strip_doi_url(work.get("doi") or ""),
        url=str(work.get("id") or "").strip(),
    )


def fetch_semantic_scholar_candidates(
    venue: str,
    *,
    max_candidates: int,
    year: str,
    timeout: int,
    api_key: str = "",
) -> list[CandidatePaper]:
    url = _semantic_scholar_url(venue, limit=max_candidates, year=year)
    req = build_semantic_scholar_request(url, api_key=api_key)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    candidates: list[CandidatePaper] = []
    for item in payload.get("data", []):
        title = (item.get("title") or "").strip()
        abstract = (item.get("abstract") or "").strip()
        item_venue = (item.get("venue") or "").strip()
        if not title or not abstract:
            continue
        external_ids = item.get("externalIds") or {}
        candidates.append(
            CandidatePaper(
                title=title,
                abstract=abstract,
                venue=item_venue,
                year=item.get("year") if isinstance(item.get("year"), int) else None,
                doi=str(external_ids.get("DOI") or "").strip(),
                url=str(item.get("url") or "").strip(),
            )
        )
    return candidates


def _openalex_sources_url(venue: str) -> str:
    return "https://api.openalex.org/sources?" + urllib.parse.urlencode(
        {"search": venue, "per-page": "10"}
    )


def _openalex_works_url(source_id: str, *, max_candidates: int, year: str) -> str:
    start_year, _, end_year = year.partition("-")
    filters = [f"primary_location.source.id:{source_id}", "type:article"]
    if start_year:
        filters.append(f"from_publication_date:{start_year}-01-01")
    if end_year:
        filters.append(f"to_publication_date:{end_year}-12-31")
    return "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {
            "filter": ",".join(filters),
            "sort": "publication_date:desc",
            "per-page": str(max_candidates),
            "select": "id,doi,title,publication_year,abstract_inverted_index,primary_location",
        }
    )


def _fetch_json(url: str, *, timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "journal-recommender-corpus-expander/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _find_openalex_source_id(venue: str, *, timeout: int) -> str:
    payload = _fetch_json(_openalex_sources_url(venue), timeout=timeout)
    for source in payload.get("results", []):
        if normalize_venue(source.get("display_name", "")) == normalize_venue(venue):
            return str(source.get("id") or "").strip()
    return ""


def fetch_openalex_candidates(
    venue: str,
    *,
    max_candidates: int,
    year: str,
    timeout: int,
) -> list[CandidatePaper]:
    source_id = _find_openalex_source_id(venue, timeout=timeout)
    if not source_id:
        return []
    payload = _fetch_json(
        _openalex_works_url(source_id, max_candidates=max_candidates, year=year),
        timeout=timeout,
    )
    candidates = []
    for work in payload.get("results", []):
        candidate = candidate_from_openalex_work(work, target_venue=venue)
        if candidate:
            candidates.append(candidate)
    return candidates


def _profile_path(accepted_dir: Path, journal_id: str) -> Path:
    return accepted_dir / f"{journal_id}.json"


def _existing_titles(profile_path: Path) -> set[str]:
    if not profile_path.exists():
        return set()
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {
        normalize_text(paper.get("title", ""))
        for paper in data.get("papers", [])
        if isinstance(paper, dict) and paper.get("title")
    }


def expand_corpus(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_inputs = [Path(path) for path in args.benchmark_input]
    accepted_dir = Path(args.accepted_dir)
    journal_index = load_journal_index(Path(args.journal_store_path))
    blacklist = build_blacklist(benchmark_inputs)
    targets = target_uncovered_venues(benchmark_inputs, accepted_dir, journal_index)
    if args.limit_venues is not None:
        targets = targets[: args.limit_venues]

    report: dict[str, Any] = {
        "benchmark_inputs": [str(path) for path in benchmark_inputs],
        "accepted_dir": str(accepted_dir),
        "target_total_per_new_journal": args.target_total,
        "target_venue_count": len(targets),
        "targets": [],
    }
    if args.dry_run:
        report["targets"] = targets
        return report

    for idx, target in enumerate(targets, 1):
        journal_id = target["journal_id"]
        journal_name = target["journal_name"]
        profile_path = _profile_path(accepted_dir, journal_id)
        existing_titles = _existing_titles(profile_path)
        status = {
            **target,
            "candidate_count": 0,
            "accepted_candidate_count": 0,
            "added_count": 0,
            "error": "",
        }
        try:
            if args.source == "openalex":
                candidates = fetch_openalex_candidates(
                    journal_name,
                    max_candidates=args.max_candidates,
                    year=args.year,
                    timeout=args.timeout,
                )
                source = "openalex"
            else:
                candidates = fetch_semantic_scholar_candidates(
                    journal_name,
                    max_candidates=args.max_candidates,
                    year=args.year,
                    timeout=args.timeout,
                    api_key=os.environ.get(args.api_key_env, ""),
                )
                source = "semantic_scholar"
            filtered = filter_candidates(
                candidates,
                target_venue=journal_name,
                blacklist=blacklist,
                existing_titles=existing_titles,
                limit=args.target_total,
            )
            added = merge_profile(
                profile_path=profile_path,
                journal_id=journal_id,
                journal_name=journal_name,
                new_papers=filtered,
                target_total=args.target_total,
                source=source,
            )
            status.update(
                {
                    "candidate_count": len(candidates),
                    "accepted_candidate_count": len(filtered),
                    "added_count": added,
                }
            )
        except Exception as exc:  # pragma: no cover - exercised via real CLI
            status["error"] = str(exc)
        report["targets"].append(status)
        if idx < len(targets) and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
    report["added_total"] = sum(item["added_count"] for item in report["targets"])
    report["successful_venue_count"] = sum(1 for item in report["targets"] if item["added_count"] > 0)
    report["failed_venue_count"] = sum(1 for item in report["targets"] if item.get("error"))
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--benchmark-input", action="append", default=None)
    parser.add_argument("--accepted-dir", default="data/accepted_papers")
    parser.add_argument("--journal-store-path", default="data/processed/journals.jsonl")
    parser.add_argument("--target-total", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--year", default="2020-2026")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--source", choices=["semantic-scholar", "openalex"], default="semantic-scholar")
    parser.add_argument("--limit-venues", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--api-key-env",
        default="SEMANTIC_SCHOLAR_API_KEY",
        help="Environment variable containing the Semantic Scholar API key.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.benchmark_input is None:
        args.benchmark_input = [str(path) for path in DEFAULT_BENCHMARK_INPUTS]
    report = expand_corpus(args)
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
