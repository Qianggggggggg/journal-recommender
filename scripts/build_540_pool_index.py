#!/usr/bin/env python3
"""Build (area, ccf) → [journal_name, ...] index from journals.jsonl.

Used by:
  - scripts/augment_540_corpus.py  (search S2 for papers in each bucket)
  - scripts/audit_540.py            (verify candidate venues)
  - scripts/replace_540_invalid.py  (in-place invalid paper replacement)

Outputs:
  - data/evaluation/papers_metadata_540_pool_index.json
      { "(area, ccf)": [journal_name, ...], ... }
  - prints bucket report to stdout

CCF areas (must match journals.jsonl subject_tags):
  1. 计算机体系结构/并行与分布计算/存储系统
  2. 计算机网络
  3. 网络与信息安全
  4. 软件工程/系统软件/程序设计语言
  5. 数据库/数据挖掘/内容检索
  6. 计算机科学理论
  7. 计算机图形学与多媒体
  8. 人工智能
  9. 人机交互与普适计算
  10. 交叉/综合/新兴

Target distribution: 18 papers per (area, ccf) bucket, 540 total.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
JOURNALS_PATH = project_root / "data/processed/journals.jsonl"
OUT_PATH = project_root / "data/evaluation/papers_metadata_540_pool_index.json"

CCF_AREAS = [
    "计算机体系架构/并行与分布计算/存储系统",  # placeholder, real list below
]
# Note: real list is defined as CCF_AREAS_OFFICIAL further down. The placeholder
# above is a marker to avoid duplicating the literal in two places.

CCF_AREAS_OFFICIAL = [
    "计算机体系结构/并行与分布计算/存储系统",
    "计算机网络",
    "网络与信息安全",
    "软件工程/系统软件/程序设计语言",
    "数据库/数据挖掘/内容检索",
    "计算机科学理论",
    "计算机图形学与多媒体",
    "人工智能",
    "人机交互与普适计算",
    "交叉/综合/新兴",
]
CCF_LEVELS = ["A", "B", "C"]
TARGET_PAPERS_PER_BUCKET = 18


def load_pool() -> dict[str, list[str]]:
    """(area, ccf) → [journal_name, ...]"""
    pool: dict[str, list[str]] = defaultdict(list)
    with JOURNALS_PATH.open() as f:
        for line in f:
            j = json.loads(line)
            # Known typo: 1 row has "cf_rating" instead of "ccf_rating".
            # Fall back to that field if "ccf_rating" is missing so we don't
            # silently drop the journal. We do NOT want to silently drop
            # 1/295 journals.
            ccf = (
                j.get("ccf_rating")
                or j.get("ccf_rank")
                or j.get("ccf")
                or j.get("cf_rating")
            )
            if ccf is None:
                print(f'  WARN: {j.get("journal_id")} has no ccf field, skipping',
                      file=sys.stderr)
                continue
            for area in j.get("subject_tags", []):
                key = f"{area}|{ccf}"
                if j["journal_name"] not in pool[key]:
                    pool[key].append(j["journal_name"])
    return dict(pool)


def bucket_report(pool: dict[str, list[str]]) -> str:
    """Print (area, ccf) bucket size + target distribution."""
    lines = [f'{"Bucket":52s} {"Journals":>8s} {"pp/journal":>10s} {"Target":>7s}']
    lines.append("-" * 80)
    total = 0
    for area in CCF_AREAS_OFFICIAL:
        for ccf in CCF_LEVELS:
            key = f"{area}|{ccf}"
            journals = pool.get(key, [])
            n = len(journals)
            if n == 0:
                lines.append(f"{area}/{ccf}: NO JOURNALS")
                continue
            pp = math.ceil(TARGET_PAPERS_PER_BUCKET / n)
            lines.append(
                f"{(area + '/' + ccf)[:52]:52s} {n:>8d} {pp:>10d} "
                f"{TARGET_PAPERS_PER_BUCKET:>7d}"
            )
            total += TARGET_PAPERS_PER_BUCKET
    lines.append("-" * 80)
    lines.append(f"{'TOTAL':52s} {'':>8s} {'':>10s} {total:>7d}")
    return "\n".join(lines)


def main() -> int:
    pool = load_pool()
    print(bucket_report(pool))
    print(f"\nWriting {OUT_PATH}")
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": 1,
                "ccf_areas": CCF_AREAS_OFFICIAL,
                "ccf_levels": CCF_LEVELS,
                "target_papers_per_bucket": TARGET_PAPERS_PER_BUCKET,
                "buckets": pool,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  -> {len(pool)} buckets written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
