#!/usr/bin/env python3
"""Build LTR training data from retrieval ablation output (Task 4.1.f + 4.3).

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

sidecar report(4.3):
- 正样本缺失 route 特征的数量
- < 80% 正样本 retrieval_rank ≤ 50 时输出 warning
- route combination 分布
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ranker.feature_builder import FEATURE_NAMES, MISSING_RANK_SENTINEL


# 负样本类型优先级(从前到后填,直到达到 max_negatives)
NEGATIVE_PRIORITY: tuple = ("hard_rule_top20", "same_area", "easy_other")

# 80% warning 阈值(per plan 4.3):< 该比例的 positive 在 retrieval_top50 内时报警
RETRIEVAL_TOPK_80_THRESHOLD: float = 0.8

# route 名称:对应的 features 中的 rank 字段名(必须以 _rank 结尾、与 FEATURE_NAMES 对齐)
ROUTE_RANK_FEATURES: tuple = (
    "scope_bm25_rank",
    "scope_vector_rank",
    "typical_bm25_rank",
    "typical_vector_rank",
    "accepted_bm25_rank",
    "accepted_vector_rank",
)


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


def _extract_route_combination(features: List[float]) -> str:
    """从 features 向量中提取"实际出现"(rank != 哨兵)的 route 集合。

    返回 sorted 后用 + 连接的字符串。例如 ``"scope_bm25+typical_bm25"``。
    全缺失时返回空字符串。
    """
    if not features:
        return ""
    present: List[str] = []
    for route_field in ROUTE_RANK_FEATURES:
        if route_field not in FEATURE_NAMES:
            continue
        idx = FEATURE_NAMES.index(route_field)
        if idx >= len(features):
            continue
        if features[idx] != MISSING_RANK_SENTINEL:
            # 字段名去掉 _rank 后缀,即 "scope_bm25"
            present.append(route_field[: -len("_rank")])
    return "+".join(present)


def _build_lookup_for_retrieval_rank(
    ablation_data: dict, only_variants: Optional[Iterable[str]] = None
) -> Dict[tuple, Optional[int]]:
    """构造 (variant, paper_id) → retrieval_rank 的查找表。

    paper_id = paper_result["title"] (与 build_training_rows 一致)。
    """
    lookup: Dict[tuple, Optional[int]] = {}
    variants = ablation_data.get("variants") or {}
    for variant_name, variant_data in variants.items():
        if only_variants is not None and variant_name not in set(only_variants):
            continue
        for paper_idx, paper_result in enumerate(variant_data.get("paper_results") or []):
            paper_id = paper_result.get("title") or f"paper_{paper_idx}"
            lookup[(variant_name, paper_id)] = paper_result.get("retrieval_rank")
    return lookup


def build_training_report(
    ablation_data: dict,
    positive_rows: List[dict],
    only_variants: Optional[Iterable[str]] = None,
) -> dict:
    """为训练数据生成 sidecar report(per plan 4.3)。

    包含:
    - ``positives_total`` / ``positives_by_variant``
    - ``positives_with_target_in_top50_count`` / ``positives_with_target_in_top50_ratio``
    - ``retrieval_topk_80_warning``(< RETRIEVAL_TOPK_80_THRESHOLD 触发)
    - ``positives_missing_route_features``:每个 route 字段在正样本中为哨兵的次数
    - ``route_combination_counts``:route 组合分布
    """
    if only_variants is not None:
        only_variants_set = set(only_variants)
    else:
        only_variants_set = None

    retrieval_lookup = _build_lookup_for_retrieval_rank(ablation_data, only_variants=only_variants_set)

    positives_by_variant: Dict[str, int] = {}
    positives_in_top50 = 0
    positives_total = 0
    missing_per_feature: Dict[str, int] = {f: 0 for f in ROUTE_RANK_FEATURES}
    combination_counts: Dict[str, int] = {}

    for row in positive_rows:
        positives_total += 1
        variant = row.get("variant", "")
        positives_by_variant[variant] = positives_by_variant.get(variant, 0) + 1
        paper_id = row.get("paper_id", "")
        rank = retrieval_lookup.get((variant, paper_id))
        if rank is not None and isinstance(rank, (int, float)) and 0 < rank <= 50:
            positives_in_top50 += 1
        # 统计缺失的 route 特征
        feats = row.get("features") or []
        for route_field in ROUTE_RANK_FEATURES:
            if route_field not in FEATURE_NAMES:
                continue
            idx = FEATURE_NAMES.index(route_field)
            if idx < len(feats) and feats[idx] == MISSING_RANK_SENTINEL:
                missing_per_feature[route_field] += 1
        # 统计 route combination
        combination = _extract_route_combination(feats)
        if combination:
            combination_counts[combination] = combination_counts.get(combination, 0) + 1

    ratio = (positives_in_top50 / positives_total) if positives_total else 0.0
    warning = positives_total > 0 and ratio < RETRIEVAL_TOPK_80_THRESHOLD

    return {
        "positives_total": positives_total,
        "positives_by_variant": positives_by_variant,
        "positives_with_target_in_top50_count": positives_in_top50,
        "positives_with_target_in_top50_ratio": round(ratio, 6),
        "retrieval_topk_80_warning": warning,
        "retrieval_topk_80_threshold": RETRIEVAL_TOPK_80_THRESHOLD,
        "positives_missing_route_features": missing_per_feature,
        "route_combination_counts": combination_counts,
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
    parser.add_argument(
        "--report",
        default=None,
        help="可选:sidecar report JSON 路径 (per plan 4.3)",
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

    # 4.3 sidecar report
    if args.report:
        report = build_training_report(
            ablation_data=ablation_data,
            positive_rows=[r for r in rows if r["label"] == 1],
            only_variants=args.variants,
        )
        # 加一些元信息
        report["input_ablation_json"] = str(args.ablation_json)
        report["output_jsonl"] = str(args.output)
        report["max_negatives"] = args.max_negatives
        report["negatives_by_type"] = by_type
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if report["retrieval_topk_80_warning"]:
            print(
                f"[warn] {report['positives_with_target_in_top50_ratio']:.1%} positives have "
                f"retrieval_rank <= 50 (below {RETRIEVAL_TOPK_80_THRESHOLD:.0%}); see {args.report}",
                file=sys.stderr,
            )
        print(f"Wrote sidecar report to {args.report}")


if __name__ == "__main__":
    main()
