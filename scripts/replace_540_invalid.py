#!/usr/bin/env python3
"""Replace invalid/short papers in 540 raw jsonl with new S2 searches.

Two modes:
  --mode invalid:  replace papers marked invalid by audit_540.py
                   (in 540_audit_report.json)
  --mode short:    replace papers in short buckets (where collected < 18)
                   (in papers_metadata_540_report.json)

Reuses augment_540_corpus.search_journal_papers + paper_passes_filter
to find replacement candidates from the same (area, ccf) pool.

Outputs:
  - data/evaluation/papers_metadata_540_replaced.jsonl (final, after replace)
  - data/evaluation/papers_metadata_540_replace_diagnostics.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Reuse search + filter from augment
spec = importlib.util.spec_from_file_location(
    "aug", project_root / "scripts/augment_540_corpus.py"
)
aug = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aug)

RAW_PATH = project_root / "data/evaluation/papers_metadata_540_raw.jsonl"
AUDIT_PATH = project_root / "data/evaluation/results/540_audit_report.json"
META_REPORT_PATH = project_root / "data/evaluation/papers_metadata_540_report.json"
POOL_INDEX_PATH = project_root / "data/evaluation/papers_metadata_540_pool_index.json"
REPLACED_PATH = project_root / "data/evaluation/papers_metadata_540_replaced.jsonl"
DIAG_PATH = project_root / "data/evaluation/papers_metadata_540_replace_diagnostics.json"


def normalize_title(t: str) -> str:
    return " ".join((t or "").casefold().split())


def load_pool() -> dict[str, list[str]]:
    doc = json.loads(POOL_INDEX_PATH.read_text())
    return doc["buckets"]


def collect_seen_titles(papers: list[dict]) -> set[str]:
    return {normalize_title(p.get("title", "")) for p in papers}


def find_replacement(
    area: str,
    ccf: str,
    journals: list[str],
    seen_titles: set[str],
    per_journal_needed: dict[str, int],
    papers_per_search: int = 100,
) -> list[dict]:
    """Search each journal in the bucket, collect up to per_journal_needed
    new papers that aren't in seen_titles.
    """
    new_papers = []
    new_seen = set()
    for jname, need in per_journal_needed.items():
        if need <= 0:
            continue
        print(f"  [S2] replacing {jname[:60]} ×{need} ...", flush=True)
        try:
            s2_papers = aug.search_journal_papers(
                jname, aug.YEAR_MIN, aug.YEAR_MAX, limit=papers_per_search,
            )
        except Exception as e:
            print(f"    [err] {e}", flush=True)
            continue
        accepted = 0
        for cand in s2_papers:
            if not aug.paper_passes_filter(cand, jname):
                continue
            t_norm = normalize_title(cand.get("title", ""))
            if t_norm in seen_titles or t_norm in new_seen:
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
            new_papers.append(paper)
            new_seen.add(t_norm)
            accepted += 1
            if accepted >= need:
                break
        print(f"    accepted {accepted}/{need}", flush=True)
        time.sleep(aug.SLEEP_BETWEEN)
    return new_papers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["invalid", "short"], required=True)
    args = parser.parse_args()

    if not RAW_PATH.exists():
        print(f"Missing {RAW_PATH}")
        return 1
    raw_papers = [json.loads(line) for line in RAW_PATH.open() if line.strip()]
    print(f"Loaded {len(raw_papers)} raw papers")

    pool = load_pool()

    # Determine which titles to drop
    drop_titles: set[str] = set()
    if args.mode == "invalid":
        if not AUDIT_PATH.exists():
            print(f"Missing {AUDIT_PATH}; run audit_540.py first")
            return 1
        audit = json.loads(AUDIT_PATH.read_text())
        for a in audit["audits"]:
            if a["verdict"] == "invalid":
                drop_titles.add(normalize_title(a["title"]))
        print(f"Invalid titles to replace: {len(drop_titles)}")
    elif args.mode == "short":
        if not META_REPORT_PATH.exists():
            print(f"Missing {META_REPORT_PATH}; run build_540_metadata.py first")
            return 1
        report = json.loads(META_REPORT_PATH.read_text())
        for b in report["buckets"]:
            if b["short"]:
                # drop last (target - count) papers in this bucket
                bucket_papers = [
                    p for p in raw_papers
                    if (p.get("research_area") or [None])[0] == b["area"]
                    and p.get("ccf_level") == b["ccf"]
                ]
                # drop newest (assume end of bucket is overflow/empty)
                need = b["target"] - b["count"]
                if need > 0 and bucket_papers:
                    # drop the last `need` papers in this bucket
                    for p in bucket_papers[-need:]:
                        drop_titles.add(normalize_title(p["title"]))
        print(f"Short-bucket titles to replace: {len(drop_titles)}")

    # Build kept + drop
    kept = []
    for p in raw_papers:
        if normalize_title(p.get("title", "")) in drop_titles:
            continue
        kept.append(p)
    print(f"Kept {len(kept)} (was {len(raw_papers)}, dropped {len(drop_titles)})")

    # Now figure out which buckets need more
    seen = collect_seen_titles(kept)
    bucket_need: dict[tuple[str, str], int] = defaultdict(int)
    for t in drop_titles:
        # find this paper in raw to know its (area, ccf)
        for p in raw_papers:
            if normalize_title(p.get("title", "")) == t:
                area = (p.get("research_area") or [None])[0]
                ccf = p.get("ccf_level")
                if area and ccf:
                    bucket_need[(area, ccf)] += 1
                break

    # For each bucket, distribute need across journals
    per_bucket_diag = []
    for (area, ccf), need in bucket_need.items():
        key = f"{area}|{ccf}"
        journals = pool.get(key, [])
        if not journals:
            print(f"  WARN: no journals in pool for {key}")
            continue
        per_journal = max(1, need // len(journals))
        per_journal_targets = {j: per_journal for j in journals}
        # spread remainder
        remainder = need - per_journal * len(journals)
        for i, j in enumerate(journals):
            if remainder <= 0:
                break
            per_journal_targets[j] += 1
            remainder -= 1
        print(f"\n[replace {key}] need {need} across {len(journals)} journals")
        new_papers = find_replacement(area, ccf, journals, seen,
                                        per_journal_targets)
        kept.extend(new_papers)
        seen.update(normalize_title(p["title"]) for p in new_papers)
        per_bucket_diag.append({
            "area": area, "ccf": ccf, "needed": need,
            "replaced": len(new_papers), "short": len(new_papers) < need,
        })

    # Write final
    with REPLACED_PATH.open("w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(kept)} papers to {REPLACED_PATH}")

    DIAG_PATH.write_text(json.dumps(per_bucket_diag, ensure_ascii=False, indent=2))
    print(f"Wrote {DIAG_PATH}")
    return 0 if all(not d["short"] for d in per_bucket_diag) else 2


if __name__ == "__main__":
    sys.exit(main())
