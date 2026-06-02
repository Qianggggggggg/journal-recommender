#!/usr/bin/env python3
"""Build LTR training data from retrieval ablation output (Task 4.1.f).

输入:``data/evaluation/results/<ablation>.json``(由 ``run_retrieval_ablation.py`` 产出,
含 ``feature_names`` 与 ``paper_results[i].candidate_features``)。
输出:JSONL,每行一个 paper-candidate pair,字段::

    {
        "paper_id": str,         # 论文标题 (or paper index)
        "journal_id": str,
        "label": 0|1,            # 1=gold venue, 0=其他
        "features": list[float], # 长度 == len(feature_names)
        "feature_names": list[str],
        "negative_type": "gold" | "hard_rule_top20" | "same_area" | "easy_other",
        "variant": str           # 来源 variant (e.g. "hybrid", "full_hybrid")
    }

每篇 paper 保留 1 个正样本(若 gold 期刊在 candidate_features 中)与至多
``max_negatives`` 个负样本(按优先级:hard > same_area > easy)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ranker.feature_builder import FEATURE_NAMES


# 负样本类型优先级(从前到后填,直到达到 max_negatives)
NEGATIVE_PRIORITY: tuple = ("hard_rule_top20", "same_area", "easy_other")


def _classify_negative(
    jid: str,
    target_jid: str,
    rule_top20: List[str],
    journals_by_id: Dict[str, dict],
) -> Optional[str]:
    """给候选期刊 jid 分配负样本类型。返回 None 表示是 gold(正样本)。"""
    if jid == target_jid:
        return None
    if jid in rule_top20:
        return "hard_rule_top20"
    target_area = (journals_by_id.get(target_jid) or {}).get("subject_tags") or []
    cand_area = (journals_by_id.get(jid) or {}).get("subject_tags") or []
    if target_area and cand_area and set(target_area) & set(cand_area):
        return "same_area"
    return "easy_other"


def _build_negatives(
    candidate_jids: List[str],
    target_jid: str,
    rule_top20: List[str],
    journals_by_id: Dict[str, dict],
    max_negatives: int,
) -> List[tuple]:
    """返回 [(jid, negative_type), ...] 按 NEGATIVE_PRIORITY 排序,总数 ≤ max_negatives。"""
    by_type: dict = {t: [] for t in NEGATIVE_PRIORITY}
    for jid in candidate_jids:
        neg_type = _classify_negative(jid, target_jid, rule_top20, journals_by_id)
        if neg_type is None:
            continue
        by_type[neg_type].append(jid)
    selected: List[tuple] = []
    for neg_type in NEGATIVE_PRIORITY:
        for jid in by_type[neg_type]:
            if len(selected) >= max_negatives:
                return selected
            selected.append((jid, neg_type))
    return selected


def build_training_rows(
    ablation_data: dict,
    journals_by_id: Dict[str, dict],
    max_negatives: int = 10,
    only_variants: Optional[Iterable[str]] = None,
) -> Iterable[dict]:
    """从 ablation JSON 产出训练样本。

    仅在 paper 的 ``candidate_features[target_jid]`` 存在时产正样本;
    负样本来自同 paper 的其他候选期刊,按 NEGATIVE_PRIORITY 分类。
    """
    variants = ablation_data.get("variants") or {}
    for variant_name, variant_data in variants.items():
        if only_variants is not None and variant_name not in set(only_variants):
            continue
        feature_names = variant_data.get("feature_names") or list(FEATURE_NAMES)
        for paper_idx, paper_result in enumerate(variant_data.get("paper_results") or []):
            target_jid = paper_result.get("target_journal_id")
            if not target_jid:
                continue
            candidate_features = paper_result.get("candidate_features") or {}
            if not candidate_features:
                continue
            paper_id = paper_result.get("title") or f"paper_{paper_idx}"

            # 1. 正样本(若 gold 期刊在 candidate_features)
            if target_jid in candidate_features:
                yield {
                    "paper_id": paper_id,
                    "journal_id": target_jid,
                    "label": 1,
                    "features": candidate_features[target_jid],
                    "feature_names": feature_names,
                    "negative_type": "gold",
                    "variant": variant_name,
                }

            # 2. 负样本
            rule_top20 = paper_result.get("rule_top5") or []  # ablation 只存 top5;近似硬负样本
            neg_list = _build_negatives(
                candidate_jids=list(candidate_features.keys()),
                target_jid=target_jid,
                rule_top20=rule_top20,
                journals_by_id=journals_by_id,
                max_negatives=max_negatives,
            )
            for jid, neg_type in neg_list:
                yield {
                    "paper_id": paper_id,
                    "journal_id": jid,
                    "label": 0,
                    "features": candidate_features[jid],
                    "feature_names": feature_names,
                    "negative_type": neg_type,
                    "variant": variant_name,
                }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-json", required=True, help="run_retrieval_ablation.py 产出")
    parser.add_argument("--journals-jsonl", required=True, help="期刊元数据 jsonl(用于 area 判定)")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--max-negatives", type=int, default=10, help="每篇 paper 最多负样本数")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="只处理指定 variant (默认:所有);例如 --variants full_hybrid",
    )
    args = parser.parse_args()

    ablation_data = json.loads(Path(args.ablation_json).read_text(encoding="utf-8"))
    journals_by_id: Dict[str, dict] = {}
    for line in Path(args.journals_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        jid = rec.get("journal_id")
        if jid:
            journals_by_id[jid] = rec

    rows = list(
        build_training_rows(
            ablation_data=ablation_data,
            journals_by_id=journals_by_id,
            max_negatives=args.max_negatives,
            only_variants=args.variants,
        )
    )
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 简要诊断
    pos = sum(1 for r in rows if r["label"] == 1)
    neg = sum(1 for r in rows if r["label"] == 0)
    by_type: dict = {}
    for r in rows:
        if r["label"] == 0:
            by_type[r["negative_type"]] = by_type.get(r["negative_type"], 0) + 1
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"  positives: {pos}, negatives: {neg}")
    print(f"  negatives by type: {by_type}")


if __name__ == "__main__":
    main()
