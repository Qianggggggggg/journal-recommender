#!/usr/bin/env python3
"""540 leakage detection vs light30 / full-v2-90 / holdout240.

Rules (consistent with plan 1.2):
  - title: casefold + whitespace collapse exact match
  - abstract: ≥ 160-char fragment match (any contiguous 160+ chars)
  - source_type: light30 | full_v2_90 | holdout240

Outputs:
  - data/evaluation/results/540_leakage_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

RAW_PATH = project_root / "data/evaluation/papers_metadata_540_raw.jsonl"
LIGHT30 = project_root / "data/evaluation/papers_metadata_light_30.jsonl"
FULL_V2 = project_root / "data/evaluation/papers_metadata_full_v2_90.jsonl"
HOLDOUT = project_root / "data/evaluation/papers_metadata_holdout240.jsonl"
REPORT_PATH = project_root / "data/evaluation/results/540_leakage_report.json"


def normalize_title(t: str) -> str:
    return " ".join((t or "").casefold().split())


def has_abstract_overlap(a: str, b: str, min_chars: int = 160) -> bool:
    """Check if a and b share a contiguous fragment of ≥ min_chars (case-insensitive)."""
    a_norm = (a or "").casefold()
    b_norm = (b or "").casefold()
    if not a_norm or not b_norm:
        return False
    if len(a_norm) < min_chars or len(b_norm) < min_chars:
        return False
    # Sliding window in a; check if any window appears in b
    for i in range(0, len(a_norm) - min_chars + 1, 80):
        frag = a_norm[i : i + min_chars]
        if frag in b_norm:
            return True
    return False


def main() -> int:
    if not RAW_PATH.exists():
        print(f"Missing {RAW_PATH}; run augment_540_corpus.py first")
        return 1

    raw_papers = [json.loads(line) for line in RAW_PATH.open() if line.strip()]
    print(f"Loaded {len(raw_papers)} raw 540 candidates", flush=True)

    sources = {
        "light30": LIGHT30,
        "full_v2_90": FULL_V2,
        "holdout240": HOLDOUT,
    }
    source_titles: dict[str, set[str]] = {}
    source_abstracts: dict[str, list[str]] = {}
    for name, p in sources.items():
        if not p.exists():
            print(f"  WARN: {p} missing, skipping", flush=True)
            source_titles[name] = set()
            source_abstracts[name] = []
            continue
        rows = [json.loads(line) for line in p.open() if line.strip()]
        source_titles[name] = {normalize_title(r.get("title", "")) for r in rows}
        source_abstracts[name] = [r.get("abstract", "") for r in rows]
        print(f"  {name}: {len(rows)} papers", flush=True)

    leaks = []
    for i, p in enumerate(raw_papers):
        t = normalize_title(p.get("title", ""))
        abstract = p.get("abstract", "")
        for src_name in sources:
            if t in source_titles[src_name]:
                leaks.append({
                    "raw_index": i,
                    "title": p.get("title"),
                    "source_type": src_name,
                    "match_type": "title_exact",
                })
                continue
            for ref_abstract in source_abstracts[src_name]:
                if has_abstract_overlap(abstract, ref_abstract):
                    leaks.append({
                        "raw_index": i,
                        "title": p.get("title"),
                        "source_type": src_name,
                        "match_type": "abstract_fragment",
                    })
                    break

    report = {
        "schema_version": 1,
        "input_path": str(RAW_PATH),
        "total_checked": len(raw_papers),
        "leak_count": len(leaks),
        "leak_by_source": {
            src: sum(1 for l in leaks if l["source_type"] == src)
            for src in sources
        },
        "leaks": leaks,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {REPORT_PATH}")
    print(f"\n=== Summary ===")
    print(f"  total_checked: {len(raw_papers)}")
    print(f"  leak_count: {len(leaks)}")
    for src, n in report["leak_by_source"].items():
        print(f"  {src}: {n}")
    return 0 if not leaks else 2


if __name__ == "__main__":
    sys.exit(main())
