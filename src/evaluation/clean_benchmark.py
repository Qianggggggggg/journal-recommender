"""Leakage checks and clean typical-abstract snapshots for fair evaluation."""
import json
import re
import shutil
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class LeakageReport:
    """Structured leakage report for typical abstracts."""

    summary: dict
    matches: list[dict]

    def to_dict(self) -> dict:
        return {"summary": self.summary, "matches": self.matches}


def load_papers_jsonl(path: str | Path) -> list[dict]:
    """Load evaluation papers from JSONL."""
    papers = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))
    return papers


def detect_typical_leakage(
    papers: Sequence[dict],
    typical_dir: str | Path,
    *,
    min_title_chars: int = 24,
    min_abstract_chars: int = 220,
) -> LeakageReport:
    """Find test paper text that appears inside typical abstracts."""
    typical_files = list(_iter_typical_files(typical_dir))
    paper_needles = _paper_needles(
        papers,
        min_title_chars=min_title_chars,
        min_abstract_chars=min_abstract_chars,
    )
    matches = []
    leaked_entries = set()
    leaked_papers = set()

    for path in typical_files:
        data = _read_json(path)
        for idx, item in enumerate(data.get("abstracts", [])):
            haystack = _normalize_text(_entry_text(item))
            if not haystack:
                continue
            for needle in paper_needles:
                if needle["text"] in haystack:
                    leaked_entries.add((path.name, idx))
                    leaked_papers.add(needle["paper_index"])
                    matches.append(
                        {
                            "paper_index": needle["paper_index"],
                            "paper_title": needle["paper_title"],
                            "paper_venue": needle["paper_venue"],
                            "match_type": needle["match_type"],
                            "typical_file": path.name,
                            "journal_id": data.get("journal_id") or path.stem,
                            "entry_index": idx,
                            "method_type": item.get("method_type", ""),
                            "novelty_level": item.get("novelty_level", ""),
                        }
                    )
                    break

    return LeakageReport(
        summary={
            "paper_count": len(papers),
            "typical_file_count": len(typical_files),
            "match_count": len(matches),
            "leaked_entry_count": len(leaked_entries),
            "leaked_paper_count": len(leaked_papers),
        },
        matches=matches,
    )


def build_clean_typical_snapshot(
    papers: Sequence[dict],
    typical_dir: str | Path,
    output_dir: str | Path,
    *,
    min_title_chars: int = 24,
    min_abstract_chars: int = 220,
    overwrite: bool = True,
) -> LeakageReport:
    """Write a clean typical-abstract directory with leaked entries removed."""
    typical_dir = Path(typical_dir)
    output_dir = Path(output_dir)
    report = detect_typical_leakage(
        papers,
        typical_dir,
        min_title_chars=min_title_chars,
        min_abstract_chars=min_abstract_chars,
    )
    leaked_entries = {
        (match["typical_file"], match["entry_index"])
        for match in report.matches
    }

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    removed = 0
    for path in _iter_typical_files(typical_dir):
        data = _read_json(path)
        clean_abstracts = []
        for idx, item in enumerate(data.get("abstracts", [])):
            if (path.name, idx) in leaked_entries:
                removed += 1
                continue
            clean_abstracts.append(item)
        data["abstracts"] = clean_abstracts
        (output_dir / path.name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        written += 1

    summary = dict(report.summary)
    summary.update(
        {
            "removed_entry_count": removed,
            "written_file_count": written,
            "source_typical_dir": str(typical_dir),
            "clean_typical_dir": str(output_dir),
        }
    )
    return LeakageReport(summary=summary, matches=report.matches)


def write_report(report: LeakageReport, output_path: str | Path) -> None:
    """Persist a leakage report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _paper_needles(
    papers: Sequence[dict],
    *,
    min_title_chars: int,
    min_abstract_chars: int,
) -> list[dict]:
    needles = []
    for idx, paper in enumerate(papers):
        title = (paper.get("title") or "").strip()
        normalized_title = _normalize_text(title)
        if len(normalized_title) >= min_title_chars:
            needles.append(
                {
                    "paper_index": idx,
                    "paper_title": title,
                    "paper_venue": paper.get("venue", ""),
                    "match_type": "title",
                    "text": normalized_title,
                }
            )

        abstract = _normalize_text(paper.get("abstract") or "")
        if len(abstract) >= min_abstract_chars:
            needles.append(
                {
                    "paper_index": idx,
                    "paper_title": title,
                    "paper_venue": paper.get("venue", ""),
                    "match_type": "abstract_snippet",
                    "text": abstract[:min_abstract_chars],
                }
            )
    return needles


def _iter_typical_files(typical_dir: str | Path) -> Iterable[Path]:
    base = Path(typical_dir)
    if not base.exists():
        return []
    return sorted(base.glob("*.json"))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _entry_text(item: dict) -> str:
    return " ".join(
        str(item.get(field) or "")
        for field in ("title", "paper_title", "source_title", "abstract", "text")
    )


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
