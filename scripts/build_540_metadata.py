#!/usr/bin/env python3
"""Build final 540 metadata jsonl from raw + audit + leakage reports.

Steps:
  1. Load raw 540 candidates.
  2. Mark audit_status from 540_audit_report.json (valid / suspect / invalid).
  3. Drop invalid papers and any with leakage.
  4. If short of 18 per (area, ccf) bucket, log a warning.
  5. Write papers_metadata_540.jsonl with final audit_status.
  6. Write papers_metadata_540_report.json with summary.

Outputs:
  - data/evaluation/papers_metadata_540.jsonl
  - data/evaluation/papers_metadata_540_report.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

RAW_PATH = project_root / "data/evaluation/papers_metadata_540_raw.jsonl"
AUDIT_PATH = project_root / "data/evaluation/results/540_audit_report.json"
LEAK_PATH = project_root / "data/evaluation/results/540_leakage_report.json"
OUT_PATH = project_root / "data/evaluation/papers_metadata_540.jsonl"
REPORT_PATH = project_root / "data/evaluation/papers_metadata_540_report.json"


def normalize_title(t: str) -> str:
    return " ".join((t or "").casefold().split())


def main() -> int:
    if not RAW_PATH.exists():
        print(f"Missing {RAW_PATH}")
        return 1

    raw_papers = [json.loads(line) for line in RAW_PATH.open() if line.strip()]
    print(f"Loaded {len(raw_papers)} raw candidates")

    # Audit map
    audit_status: dict[str, str] = {}  # title → verdict
    if AUDIT_PATH.exists():
        audit = json.loads(AUDIT_PATH.read_text())
        for a in audit["audits"]:
            audit_status[normalize_title(a["title"])] = a["verdict"]
    print(f"Audit verdicts loaded: {Counter(audit_status.values())}")

    # Leak set
    leak_titles: set[str] = set()
    if LEAK_PATH.exists():
        leak = json.loads(LEAK_PATH.read_text())
        for l in leak["leaks"]:
            leak_titles.add(normalize_title(l["title"]))
    print(f"Leak titles: {len(leak_titles)}")

    # Apply filters
    final = []
    bucket_counts: dict[str, int] = defaultdict(int)
    dropped = Counter()
    for p in raw_papers:
        t = normalize_title(p.get("title", ""))
        if t in leak_titles:
            dropped["leak"] += 1
            continue
        verdict = audit_status.get(t, "missing")
        if verdict == "invalid":
            dropped["invalid"] += 1
            continue
        # Suspect: keep but tag; user may want to inspect
        p["audit_status"] = verdict if verdict != "missing" else "unaudited"
        final.append(p)
        bucket_key = f"{p.get('research_area', ['?'])[0]}|{p.get('ccf_level', '?')}"
        bucket_counts[bucket_key] += 1

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for p in final:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(final)} papers to {OUT_PATH}")

    # Bucket report
    target = 18
    short_buckets = []
    bucket_report = []
    for area in sorted({k.split("|")[0] for k in bucket_counts}):
        for ccf in ["A", "B", "C"]:
            key = f"{area}|{ccf}"
            n = bucket_counts.get(key, 0)
            bucket_report.append({
                "area": area, "ccf": ccf, "count": n, "target": target,
                "short": n < target,
            })
            if n < target:
                short_buckets.append((area, ccf, n))

    report = {
        "schema_version": 1,
        "input_raw": str(RAW_PATH),
        "total_raw": len(raw_papers),
        "total_final": len(final),
        "dropped": dict(dropped),
        "audit_status_counts": Counter(p["audit_status"] for p in final),
        "buckets": bucket_report,
        "short_buckets": [
            {"area": a, "ccf": c, "have": n, "need": target - n}
            for a, c, n in short_buckets
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {REPORT_PATH}")

    print(f"\n=== Summary ===")
    print(f"  raw: {len(raw_papers)}")
    print(f"  final: {len(final)}")
    print(f"  dropped: {dict(dropped)}")
    print(f"  short buckets: {len(short_buckets)}")
    for a, c, n in short_buckets:
        print(f"    {a}/{c}: {n}/{target}")
    return 0 if not short_buckets else 2


if __name__ == "__main__":
    sys.exit(main())
