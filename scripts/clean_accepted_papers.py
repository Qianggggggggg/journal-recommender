#!/usr/bin/env python3
"""Audit and clean accepted-paper profiles without modifying the benchmark.

The cleaner removes:

1. papers overlapping the configured benchmark by normalized title, abstract
   snippet, or DOI;
2. retractions, index pages, editorials, and other issue front matter;
3. duplicate papers occurring under more than one ``journal_id``.

Dry-run is the default. Pass ``--apply`` to rewrite only changed JSON profiles.
Every removal and duplicate winner is written to a machine-readable report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import os
import re
import statistics
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_BENCHMARK = Path("data/evaluation/papers_metadata_660_balanced.jsonl")
DEFAULT_ACCEPTED_DIR = Path("data/accepted_papers")
DEFAULT_REPORT = Path(
    "data/evaluation/results/accepted_papers_cleaning_report.json"
)
ABSTRACT_SNIPPET_LENGTH = 160
MIN_BENCHMARK_TITLE_CHARS = 24

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WHITESPACE_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MATHML_BLOCK_RE = re.compile(
    r"<mml:math\b[^>]*>.*?</mml:math\s*>", re.IGNORECASE | re.DOTALL
)
_PROCESSING_INSTRUCTION_RE = re.compile(r"<\?.*?\?>", re.DOTALL)
_LEADING_ABSTRACT_RE = re.compile(
    r"^\s*abstract(?:\s*[:.\u2014-]\s*|\s+)", re.IGNORECASE
)
_NUMERIC_SUFFIX_RE = re.compile(r"_\d+$")

_NON_RESEARCH_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "retraction",
        re.compile(
            r"\b(?:retraction|retracted(?:\s+article)?|withdrawal\s+notice|"
            r"withdrawn(?:\s+article)?)\b"
        ),
    ),
    (
        "correction",
        re.compile(
            r"\b(?:corrigendum|erratum|publisher\s+correction|"
            r"author\s+correction|expression\s+of\s+concern)\b"
        ),
    ),
    (
        "index",
        re.compile(
            r"(?:^\d{4}\s+index\b.*\b(?:vol|volume)\b|"
            r"^(?:author|subject|keyword|annual|cumulative|volume)\s+index\b|"
            r"^index\s+(?:to|for|of)\b)"
        ),
    ),
    (
        "editorial",
        re.compile(
            r"(?:\beditorial\b|"
            r"\bguest\s+editor(?:ial)?\b|"
            r"^editors?\s+(?:note|introduction)\b|"
            r"^introduction\s+to\s+(?:the\s+)?(?:virtual\s+)?special\s+issue\b|"
            r"\bspecial\s+issue\b|"
            r"\bpreface\b|"
            r"\bforeword\b|"
            r"^call\s+for\s+papers\b|"
            r"\bmessage\s+from\b.*\beditor(?:s|\s+in\s+chief)?\b|"
            r"\bcommunication\s+from\s+(?:the\s+)?editor(?:s|\s+in\s+chief)?\b)"
        ),
    ),
    (
        "front_matter",
        re.compile(
            r"(?:\bin\s+memoriam\b|"
            r"\bobituary\b|"
            r"^recognition\s+of\b.*\breviewers?\b|"
            r"^\d{4}\s+reviewers?\s+for\b|"
            r"^\d{4}\s+.*\bpaper\s+award\b)"
        ),
    ),
)


@dataclass
class BenchmarkIndex:
    path: Path
    sha256: str
    paper_count: int
    exact_titles: set[str]
    title_needles: list[str]
    abstract_needles: list[str]
    dois: set[str]


@dataclass
class Entry:
    path: Path
    payload: dict[str, Any]
    journal_id: str
    journal_name: str
    index: int
    paper: dict[str, Any]
    removed: bool = False

    @property
    def title(self) -> str:
        return str(self.paper.get("title") or "").strip()

    @property
    def normalized_title(self) -> str:
        return normalize_text(self.title)

    @property
    def normalized_abstract(self) -> str:
        return normalize_text(self.paper.get("abstract") or "")

    @property
    def normalized_doi(self) -> str:
        return normalize_doi(self.paper.get("doi") or "")


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_doi(value: Any) -> str:
    doi = str(value or "").strip().lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.rstrip(" .;,")


def sanitize_metadata_text(value: Any, *, abstract: bool = False) -> str:
    text = html.unescape(str(value or ""))
    if abstract:
        # OpenAlex/Springer records sometimes contain a LaTeX rendering followed
        # by a duplicate MathML rendering. Keep the readable text/LaTeX and drop
        # only the duplicate MathML block.
        text = _MATHML_BLOCK_RE.sub(" ", text)
    text = _PROCESSING_INSTRUCTION_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if abstract:
        text = _LEADING_ABSTRACT_RE.sub("", text).strip()
    return text


def classify_abstract_quality(abstract: str) -> str | None:
    cleaned = sanitize_metadata_text(abstract, abstract=True)
    if len(cleaned) < 300:
        return "short_abstract"
    normalized = normalize_text(cleaned)
    if re.match(
        r"^(?:no abstract|abstract unavailable|not available)\b", normalized
    ):
        return "placeholder_abstract"
    return None


def _row_doi(row: dict[str, Any]) -> str:
    direct = normalize_doi(row.get("doi") or "")
    if direct:
        return direct
    external_ids = row.get("external_ids") or {}
    if isinstance(external_ids, dict):
        return normalize_doi(
            external_ids.get("doi") or external_ids.get("DOI") or ""
        )
    return ""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield row


def load_benchmark(path: Path) -> BenchmarkIndex:
    rows = list(iter_jsonl(path))
    exact_titles: set[str] = set()
    title_needles: set[str] = set()
    abstract_needles: set[str] = set()
    dois: set[str] = set()
    for row in rows:
        title = normalize_text(row.get("title") or "")
        if title:
            exact_titles.add(title)
            if len(title) >= MIN_BENCHMARK_TITLE_CHARS:
                title_needles.add(title)
        abstract = normalize_text(row.get("abstract") or "")
        if len(abstract) >= ABSTRACT_SNIPPET_LENGTH:
            abstract_needles.add(abstract[:ABSTRACT_SNIPPET_LENGTH])
        doi = _row_doi(row)
        if doi:
            dois.add(doi)
    return BenchmarkIndex(
        path=path,
        sha256=file_sha256(path),
        paper_count=len(rows),
        exact_titles=exact_titles,
        title_needles=sorted(title_needles),
        abstract_needles=sorted(abstract_needles),
        dois=dois,
    )


def load_entries(
    accepted_dir: Path,
) -> tuple[list[Entry], dict[Path, dict[str, Any]]]:
    entries: list[Entry] = []
    payloads: dict[Path, dict[str, Any]] = {}
    for path in sorted(accepted_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read accepted profile {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"accepted profile must be an object: {path}")
        papers = payload.get("papers")
        if not isinstance(papers, list):
            raise ValueError(f"accepted profile papers must be a list: {path}")
        payloads[path] = payload
        journal_id = str(payload.get("journal_id") or path.stem)
        journal_name = str(payload.get("journal_name") or "")
        for index, paper in enumerate(papers):
            if not isinstance(paper, dict):
                raise ValueError(f"paper must be an object: {path} papers[{index}]")
            entries.append(
                Entry(
                    path=path,
                    payload=payload,
                    journal_id=journal_id,
                    journal_name=journal_name,
                    index=index,
                    paper=paper,
                )
            )
    return entries, payloads


def benchmark_match_reasons(
    entry: Entry, benchmark: BenchmarkIndex
) -> list[str]:
    reasons: list[str] = []
    title = entry.normalized_title
    abstract = entry.normalized_abstract
    haystack = f"{title} {abstract}".strip()
    doi = entry.normalized_doi
    if title and title in benchmark.exact_titles:
        reasons.append("benchmark_title")
    elif any(needle in haystack for needle in benchmark.title_needles):
        reasons.append("benchmark_title")
    if any(needle in haystack for needle in benchmark.abstract_needles):
        reasons.append("benchmark_abstract")
    if doi and doi in benchmark.dois:
        reasons.append("benchmark_doi")
    return reasons


def classify_non_research_title(title: str) -> str | None:
    without_tags = _HTML_TAG_RE.sub(" ", str(title or ""))
    normalized = normalize_text(without_tags)
    for reason, pattern in _NON_RESEARCH_RULES:
        if pattern.search(normalized):
            return reason
    return None


def _entry_identity_keys(entry: Entry) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    if entry.normalized_doi:
        keys.append(("doi", entry.normalized_doi))
    if entry.normalized_title:
        keys.append(("title", entry.normalized_title))
    abstract = entry.normalized_abstract
    if len(abstract) >= ABSTRACT_SNIPPET_LENGTH:
        keys.append(
            ("abstract", hashlib.sha256(abstract.encode("utf-8")).hexdigest())
        )
    return keys


def _completeness(entry: Entry) -> int:
    fields = ("abstract", "year", "source", "doi", "url")
    return sum(bool(entry.paper.get(field)) for field in fields)


def _winner_sort_key(entry: Entry, *, same_journal_name: bool) -> tuple[Any, ...]:
    has_numeric_suffix = bool(_NUMERIC_SUFFIX_RE.search(entry.journal_id))
    source = str(entry.paper.get("source") or "").lower()
    source_rank = {"openalex": 2, "semantic_scholar": 1}.get(source, 0)
    quality = (
        -_completeness(entry),
        -source_rank,
        -len(str(entry.paper.get("abstract") or "")),
    )
    if same_journal_name:
        return (
            has_numeric_suffix,
            *quality,
            entry.journal_id,
            entry.path.name,
            entry.index,
        )
    return (
        *quality,
        has_numeric_suffix,
        entry.journal_id,
        entry.path.name,
        entry.index,
    )


def find_cross_journal_duplicate_groups(
    entries: Sequence[Entry],
) -> list[list[Entry]]:
    active = [entry for entry in entries if not entry.removed]
    union_find = UnionFind(len(active))
    first_by_key: dict[tuple[str, str], int] = {}
    for position, entry in enumerate(active):
        for key in _entry_identity_keys(entry):
            first = first_by_key.setdefault(key, position)
            union_find.union(first, position)

    grouped: dict[int, list[Entry]] = defaultdict(list)
    for position, entry in enumerate(active):
        grouped[union_find.find(position)].append(entry)
    return [
        group
        for group in grouped.values()
        if len({entry.journal_id for entry in group}) > 1
    ]


def _entry_report(entry: Entry) -> dict[str, Any]:
    return {
        "file": entry.path.name,
        "journal_id": entry.journal_id,
        "journal_name": entry.journal_name,
        "entry_index": entry.index,
        "title": entry.title,
        "doi": str(entry.paper.get("doi") or ""),
        "source": str(entry.paper.get("source") or ""),
    }


def _coverage(
    payloads: dict[Path, dict[str, Any]],
    kept_by_path: dict[Path, list[dict[str, Any]]],
) -> dict[str, Any]:
    counts = [len(kept_by_path.get(path, [])) for path in sorted(payloads)]
    nonempty = [count for count in counts if count > 0]
    return {
        "profile_file_count": len(counts),
        "nonempty_journal_count": len(nonempty),
        "empty_journal_count": len(counts) - len(nonempty),
        "paper_count": sum(counts),
        "papers_per_nonempty_journal": {
            "min": min(nonempty) if nonempty else 0,
            "median": statistics.median(nonempty) if nonempty else 0,
            "mean": round(statistics.mean(nonempty), 3) if nonempty else 0,
            "max": max(nonempty) if nonempty else 0,
        },
        "journal_count_buckets": {
            "0": sum(count == 0 for count in counts),
            "1-4": sum(1 <= count <= 4 for count in counts),
            "5-9": sum(5 <= count <= 9 for count in counts),
            "10+": sum(count >= 10 for count in counts),
        },
    }


def _quality_metrics(entries: Sequence[Entry]) -> dict[str, Any]:
    abstract_lengths = [
        len(str(entry.paper.get("abstract") or "")) for entry in entries
    ]
    years = [
        entry.paper.get("year")
        for entry in entries
        if isinstance(entry.paper.get("year"), int)
    ]
    sources = Counter(str(entry.paper.get("source") or "") for entry in entries)
    invalid_doi = sum(
        bool(entry.normalized_doi)
        and re.fullmatch(r"10\.\d{4,9}/\S+", entry.normalized_doi) is None
        for entry in entries
    )
    invalid_url = sum(
        bool(entry.paper.get("url"))
        and re.fullmatch(r"https?://\S+", str(entry.paper.get("url"))) is None
        for entry in entries
    )
    return {
        "paper_count": len(entries),
        "missing_fields": {
            field: sum(not bool(entry.paper.get(field)) for entry in entries)
            for field in ("title", "abstract", "year", "source", "doi", "url")
        },
        "invalid_doi_count": invalid_doi,
        "invalid_url_count": invalid_url,
        "short_abstract_count": sum(length < 300 for length in abstract_lengths),
        "title_markup_count": sum(
            bool(_HTML_TAG_RE.search(str(entry.paper.get("title") or "")))
            for entry in entries
        ),
        "abstract_markup_count": sum(
            bool(_HTML_TAG_RE.search(str(entry.paper.get("abstract") or "")))
            for entry in entries
        ),
        "leading_abstract_label_count": sum(
            bool(
                _LEADING_ABSTRACT_RE.search(
                    str(entry.paper.get("abstract") or "")
                )
            )
            for entry in entries
        ),
        "abstract_length": {
            "min": min(abstract_lengths) if abstract_lengths else 0,
            "median": statistics.median(abstract_lengths)
            if abstract_lengths
            else 0,
            "mean": round(statistics.mean(abstract_lengths), 3)
            if abstract_lengths
            else 0,
            "max": max(abstract_lengths) if abstract_lengths else 0,
        },
        "year_range": {
            "min": min(years) if years else None,
            "max": max(years) if years else None,
        },
        "sources": dict(sorted(sources.items())),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(rendered)
    try:
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def clean_accepted_papers(
    *,
    benchmark_path: Path,
    accepted_dir: Path,
    apply: bool,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    benchmark = load_benchmark(benchmark_path)
    entries, payloads = load_entries(accepted_dir)
    quality_before = _quality_metrics(entries)
    before_by_path = {
        path: copy.deepcopy(payload.get("papers") or [])
        for path, payload in payloads.items()
    }
    removals: list[dict[str, Any]] = []
    normalizations: list[dict[str, Any]] = []

    for entry in entries:
        reasons = benchmark_match_reasons(entry, benchmark)
        non_research = classify_non_research_title(entry.title)
        if non_research:
            reasons.append(non_research)
        if reasons:
            entry.removed = True
            removal = _entry_report(entry)
            removal.update(
                {
                    "primary_reason": reasons[0],
                    "matched_reasons": sorted(set(reasons)),
                }
            )
            removals.append(removal)

    for entry in entries:
        if entry.removed:
            continue
        changed_fields: list[str] = []
        clean_title = sanitize_metadata_text(entry.paper.get("title") or "")
        clean_abstract = sanitize_metadata_text(
            entry.paper.get("abstract") or "", abstract=True
        )
        if clean_title != entry.paper.get("title"):
            entry.paper["title"] = clean_title
            changed_fields.append("title")
        if clean_abstract != entry.paper.get("abstract"):
            entry.paper["abstract"] = clean_abstract
            changed_fields.append("abstract")
        if changed_fields:
            normalization = _entry_report(entry)
            normalization["changed_fields"] = changed_fields
            normalizations.append(normalization)

        abstract_issue = classify_abstract_quality(clean_abstract)
        if abstract_issue:
            entry.removed = True
            removal = _entry_report(entry)
            removal.update(
                {
                    "primary_reason": abstract_issue,
                    "matched_reasons": [abstract_issue],
                }
            )
            removals.append(removal)

    duplicate_groups_report: list[dict[str, Any]] = []
    duplicate_groups = find_cross_journal_duplicate_groups(entries)
    for group in duplicate_groups:
        normalized_names = {
            normalize_text(entry.journal_name) for entry in group if entry.journal_name
        }
        winner = sorted(
            group,
            key=lambda entry: _winner_sort_key(
                entry, same_journal_name=len(normalized_names) == 1
            ),
        )[0]
        removed_group: list[dict[str, Any]] = []
        for entry in group:
            if entry is winner:
                continue
            entry.removed = True
            removal = _entry_report(entry)
            removal.update(
                {
                    "primary_reason": "cross_journal_duplicate",
                    "matched_reasons": ["cross_journal_duplicate"],
                    "kept": _entry_report(winner),
                }
            )
            removals.append(removal)
            removed_group.append(_entry_report(entry))
        duplicate_groups_report.append(
            {
                "journal_ids": sorted({entry.journal_id for entry in group}),
                "identity_keys": sorted(
                    {
                        f"{kind}:{value}"
                        for entry in group
                        for kind, value in _entry_identity_keys(entry)
                    }
                ),
                "kept": _entry_report(winner),
                "removed": removed_group,
            }
        )

    kept_by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        if not entry.removed:
            kept_by_path[entry.path].append(entry.paper)

    changed_paths = [
        path
        for path in sorted(payloads)
        if before_by_path[path] != kept_by_path.get(path, [])
    ]
    reason_counts = Counter(
        reason
        for removal in removals
        for reason in removal.get("matched_reasons", [])
    )
    primary_reason_counts = Counter(
        removal["primary_reason"] for removal in removals
    )
    after_entries: list[Entry] = []
    for path, payload in payloads.items():
        for index, paper in enumerate(kept_by_path.get(path, [])):
            after_entries.append(
                Entry(
                    path=path,
                    payload=payload,
                    journal_id=str(payload.get("journal_id") or path.stem),
                    journal_name=str(payload.get("journal_name") or ""),
                    index=index,
                    paper=paper,
                )
            )

    remaining_benchmark_matches = sum(
        bool(benchmark_match_reasons(entry, benchmark)) for entry in after_entries
    )
    remaining_non_research = sum(
        classify_non_research_title(entry.title) is not None
        for entry in after_entries
    )
    remaining_low_quality_abstracts = sum(
        classify_abstract_quality(str(entry.paper.get("abstract") or "")) is not None
        for entry in after_entries
    )
    remaining_duplicate_groups = find_cross_journal_duplicate_groups(after_entries)

    before_kept = {path: papers for path, papers in before_by_path.items()}
    report: dict[str, Any] = {
        "schema_version": 1,
        "mode": "apply" if apply else "dry-run",
        "benchmark": {
            "path": str(benchmark.path),
            "sha256_before": benchmark.sha256,
            "paper_count": benchmark.paper_count,
        },
        "accepted_dir": str(accepted_dir),
        "summary": {
            "removed_entry_count": len(removals),
            "normalized_entry_count": len(normalizations),
            "changed_file_count": len(changed_paths),
            "removal_primary_counts": dict(sorted(primary_reason_counts.items())),
            "removal_match_counts": dict(sorted(reason_counts.items())),
            "duplicate_group_count": len(duplicate_groups_report),
            "remaining_benchmark_overlap_count": remaining_benchmark_matches,
            "remaining_non_research_count": remaining_non_research,
            "remaining_low_quality_abstract_count": (
                remaining_low_quality_abstracts
            ),
            "remaining_cross_journal_duplicate_group_count": len(
                remaining_duplicate_groups
            ),
        },
        "quality_before": quality_before,
        "quality_after": _quality_metrics(after_entries),
        "coverage_before": _coverage(payloads, before_kept),
        "coverage_after": _coverage(payloads, kept_by_path),
        "changed_files": [path.name for path in changed_paths],
        "duplicate_groups": duplicate_groups_report,
        "normalizations": normalizations,
        "removals": removals,
    }

    if apply:
        for path in changed_paths:
            payload = payloads[path]
            payload["papers"] = kept_by_path.get(path, [])
            _atomic_write_json(path, payload)
        benchmark_hash_after = file_sha256(benchmark.path)
        report["benchmark"]["sha256_after"] = benchmark_hash_after
        report["benchmark"]["unchanged"] = benchmark_hash_after == benchmark.sha256
        if benchmark_hash_after != benchmark.sha256:
            raise RuntimeError(f"benchmark changed unexpectedly: {benchmark.path}")

    return report, payloads


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--accepted-dir", type=Path, default=DEFAULT_ACCEPTED_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite changed accepted-paper profiles. Default is report-only.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report, _ = clean_accepted_papers(
        benchmark_path=args.benchmark,
        accepted_dir=args.accepted_dir,
        apply=args.apply,
    )
    write_report(report, args.report)
    summary = report["summary"]
    print(f"mode={report['mode']}")
    print(f"benchmark_papers={report['benchmark']['paper_count']}")
    print(f"accepted_before={report['coverage_before']['paper_count']}")
    print(f"removed={summary['removed_entry_count']}")
    print(f"normalized={summary['normalized_entry_count']}")
    print(f"accepted_after={report['coverage_after']['paper_count']}")
    print(f"changed_files={summary['changed_file_count']}")
    print(
        "remaining="
        f"benchmark:{summary['remaining_benchmark_overlap_count']},"
        f"non_research:{summary['remaining_non_research_count']},"
        f"low_quality_abstracts:"
        f"{summary['remaining_low_quality_abstract_count']},"
        "cross_journal_duplicates:"
        f"{summary['remaining_cross_journal_duplicate_group_count']}"
    )
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
