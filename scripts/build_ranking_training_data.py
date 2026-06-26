#!/usr/bin/env python3
"""Build LTR training data from retrieval ablation output (Task 4.1.f + 4.3).

注意:本脚本走 ``--ablation-json`` 入口,不是 plan 原文写的 ``--eval-json`` 入口。
该改线由 ADR 0002 (`docs/adr/0002-ltr-v1-ablation-input.md`) 决定:
LTR v1 不学 LLM 介入后的信号,evaluation 路径延期到阶段 6。

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
from typing import Dict, Iterable, List, Optional, Set

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ranker.feature_builder import (
    FEATURE_NAMES,
    FEATURE_NAMES_WITH_LLM_EVIDENCE,
    FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY,
    LLM_EVIDENCE_FEATURE_NAMES,
    MISSING_RANK_SENTINEL,
    _tier_weight_value,
    _area_exclusivity_value,
)


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


# Default values for the 6 LLM evidence fields when the snapshot has no
# evidence for a (paper, candidate) pair. Keys use the SAME names as the
# LLM extractor prompt output (no ``llm_`` prefix) so the snapshot JSON
# can be consumed directly without remapping.
EVIDENCE_FIT_DEFAULTS = {
    "scope_fit": 0.5,
    "method_fit": 0.5,
    "application_fit": 0.5,
    "journal_position_fit": 0.5,
    "too_broad_penalty": 0.0,
    "too_narrow_penalty": 0.0,
}

# Map snapshot/raw evidence field name → 26-dim feature vector field name.
# The LLM extractor returns ``scope_fit`` etc. (per the prompt), but
# ``LLM_EVIDENCE_FEATURE_NAMES`` uses the ``llm_`` prefix because these
# features are stored in a single feature vector alongside the 20 base
# features. Without this remap, the 26-dim vector would have all six
# new fields at their default values regardless of the snapshot content.
EVIDENCE_RAW_TO_FEATURE = {
    "scope_fit": "llm_scope_fit",
    "method_fit": "llm_method_fit",
    "application_fit": "llm_application_fit",
    "journal_position_fit": "llm_journal_position_fit",
    "too_broad_penalty": "llm_too_broad_penalty",
    "too_narrow_penalty": "llm_too_narrow_penalty",
}


def _title_key(title: str) -> str:
    """Normalize a paper title for snapshot lookup.

    Mirrors precompute_evidence._title_key exactly so the same paper
    resolves to the same key whether we are reading from
    ``precompute_evidence.py`` output or the ablation runner.
    """
    return " ".join(str(title or "").casefold().split())


def _snapshot_paper_key(title: str, venue: str) -> str:
    """Build the exact paper key used by ``precompute_evidence._paper_key``
    (``title | venue``, both casefold-normalized). This is the only key
    form stored in the snapshot's ``papers`` dict, so any lookup here
    must use the same format.

    Note: this differs from ``_title_key`` (which only normalizes the
    title). Using ``_title_key`` alone to look up in a snapshot would
    silently miss every paper.
    """
    t = " ".join(str(title or "").casefold().split())
    v = " ".join(str(venue or "").casefold().split())
    return f"{t} | {v}"


def _load_evidence_lookup(snapshot_path: str) -> Dict[str, Dict[str, dict]]:
    """Load an evidence snapshot JSON and return
    ``{title_key: {journal_id: evidence_item}}`` for fast per-row lookup.
    Papers with no evidence (e.g. pre_pass_error) are simply missing
    from the inner dict; the caller is expected to fall back to neutral
    defaults via ``_evidence_vector_for_row``.
    """
    payload = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    papers = payload.get("papers") or {}
    lookup: Dict[str, Dict[str, dict]] = {}
    for paper_key, entry in papers.items():
        evidence = entry.get("evidence") or {}
        if evidence:
            lookup[paper_key] = evidence
    return lookup


def _evidence_vector_for_row(
    paper_title: str, paper_venue: str, journal_id: str,
    evidence_lookup: Dict[str, Dict[str, dict]],
) -> List[float]:
    """Return the 6-element evidence vector for one training row, ordered
    to match ``LLM_EVIDENCE_FEATURE_NAMES`` (the 26-dim feature vector
    schema). Reads raw evidence field names (``scope_fit`` etc.) from
    the snapshot and maps them to the prefixed feature names.

    Uses ``_snapshot_paper_key`` (title + venue, normalized) so the
    lookup matches the exact key format written by precompute_evidence.
    """
    evidence = evidence_lookup.get(
        _snapshot_paper_key(paper_title, paper_venue), {}
    ).get(journal_id)
    out: List[float] = []
    for feature_name in LLM_EVIDENCE_FEATURE_NAMES:
        # Reverse-lookup: feature name like ``llm_scope_fit`` → raw name
        # like ``scope_fit`` for reading from the snapshot.
        raw_name = next(
            (k for k, v in EVIDENCE_RAW_TO_FEATURE.items() if v == feature_name),
            feature_name,
        )
        if not evidence or not isinstance(evidence, dict):
            out.append(EVIDENCE_FIT_DEFAULTS[raw_name])
            continue
        value = evidence.get(raw_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            out.append(EVIDENCE_FIT_DEFAULTS[raw_name])
        elif not 0 <= value <= 1:
            out.append(EVIDENCE_FIT_DEFAULTS[raw_name])
        else:
            out.append(float(value))
    return out


def build_training_rows(
    ablation_data: dict,
    journals_by_id: Dict[str, dict],
    max_negatives: int = 10,
    only_variants: Optional[Iterable[str]] = None,
    evidence_lookup: Optional[Dict[str, Dict[str, dict]]] = None,
    papers_by_title: Optional[Dict[str, dict]] = None,
    accepted_jid_set: Optional[Set[str]] = None,
) -> Iterable[dict]:
    """从 ablation JSON 产出训练样本。

    仅在 paper 的 ``candidate_features[target_jid]`` 存在时产正样本;
    负样本来自同 paper 的其他候选期刊,按 NEGATIVE_PRIORITY 分类。

    ``evidence_lookup`` (Task 6.4, 2026-06-26 → 25-dim schema): when supplied,
    each row's 19-dim base features are extended with the 6 LLM-evidence
    fields looked up by (paper title, journal_id) and the row's
    ``feature_names`` is set to ``FEATURE_NAMES_WITH_LLM_EVIDENCE``
    (25-dim). When omitted, output is the legacy 19-dim schema.

    ``papers_by_title`` (阶段 6.5, 2026-06-26 → 27-dim schema): when
    supplied with ``evidence_lookup``, each row's 25-dim features are
    further extended with 2 tier/area features (journal_tier_weight +
    area_exclusivity), and feature_names is set to
    FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY (27-dim). Used for P2-mini
    27-dim LTR retrain.

    ``accepted_jid_set`` (2026-06-26): set of journal_ids that have at
    least one paper in the AcceptedPaperStore. Used inside ``_row()`` to
    recompute ``candidate_in_accepted_corpus`` from real corpus coverage
    instead of reading the (always 0.0) dead-feature value from the
    ablation JSON. Also used together with ``journals_by_id`` to recompute
    ``same_gold_area`` / ``same_parsed_ccf_area`` / ``same_ccf_level`` from
    the gold venue's subject_tags + the paper's research_area.

    2026-06-26 schema changes: paper_strength removed (was 0.0 in all v4
    training rows; dead feature). Base dim 20→19, evidence schema 26→25,
    tier+exclusivity schema 28→27. Old models incompatible; retrain
    required.
    """
    variants = ablation_data.get("variants") or {}
    for variant_name, variant_data in variants.items():
        if only_variants is not None and variant_name not in set(only_variants):
            continue
        # Decide feature schema once per variant
        use_evidence_schema = evidence_lookup is not None
        # 2026-06-26: tier+exclusivity schema is 27-dim (was 28); paper_strength
        # removed. Variable name kept for minimal-diff with prior code paths.
        use_27_dim_schema = use_evidence_schema and papers_by_title is not None
        if use_27_dim_schema:
            feature_names = list(FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY)
        elif use_evidence_schema:
            feature_names = list(FEATURE_NAMES_WITH_LLM_EVIDENCE)
        else:
            feature_names = variant_data.get("feature_names") or list(FEATURE_NAMES)
        for paper_idx, paper_result in enumerate(variant_data.get("paper_results") or []):
            target_jid = paper_result.get("target_journal_id")
            if not target_jid:
                continue
            candidate_features = paper_result.get("candidate_features") or {}
            if not candidate_features:
                continue
            paper_id = paper_result.get("title") or f"paper_{paper_idx}"

            # Per-paper venue (for snapshot key lookup). Default to empty
            # so the snapshot key becomes just the title, which is what
            # the role ranker does when paper_profile has no venue.
            paper_venue = paper_result.get("venue", "") or ""

            # 阶段 6.5 (P2-mini): paper 锚 area + 同领域候选数。
            # 从 papers_by_title (joined from papers metadata jsonl) 读。
            paper_anchor_area: Optional[str] = None
            paper_meta = papers_by_title.get(paper_id, {}) if papers_by_title else {}
            pra = paper_meta.get("research_area") or []
            if isinstance(pra, list) and pra:
                paper_anchor_area = pra[0]
            elif isinstance(pra, str) and pra:
                paper_anchor_area = pra

            # 算 n_matching_in_pool (同领域候选数)。
            n_matching_in_pool: Optional[int] = None
            if paper_anchor_area:
                n_matching_in_pool = sum(
                    1
                    for jid in candidate_features.keys()
                    if paper_anchor_area
                    in (journals_by_id.get(jid, {}).get("subject_tags") or [])
                )

            # 2026-06-26: 提取 gold venue + paper 上下文,用于重算 4 个
            # dead 特征 (same_gold_area / same_parsed_ccf_area /
            # same_ccf_level / candidate_in_accepted_corpus)。旧 ablation
            # JSON 里 candidate_features 的这 4 个位置全是 0.0,必须用真实
            # 元数据覆写才能让 LR 学到信号。
            gold_journal_meta = journals_by_id.get(target_jid) or {}
            gold_subject_tags = set(gold_journal_meta.get("subject_tags") or [])
            paper_research_area = set(paper_meta.get("research_area") or [])
            paper_ccf_research_area = (
                set(paper_meta.get("ccf_research_area") or [])
                or paper_research_area
            )
            paper_ccf_target_level = (
                (gold_journal_meta.get("ccf_rating") or "").upper()
            )
            accepted_jid_set_resolved: Set[str] = (
                accepted_jid_set if accepted_jid_set is not None else set()
            )

            # 2026-06-26: 用 feature_names 动态查 idx (19/25/27 维 schema
            # 中 dead 特征位置一致 — 都是 base 19 维里的 [14..18] 区间)。
            # 这样 schema 调整不需要改这里。
            base_feature_names = list(FEATURE_NAMES)
            idx_same_gold_area = (
                base_feature_names.index("same_gold_area")
                if "same_gold_area" in base_feature_names
                else None
            )
            idx_same_parsed_ccf_area = (
                base_feature_names.index("same_parsed_ccf_area")
                if "same_parsed_ccf_area" in base_feature_names
                else None
            )
            idx_same_ccf_level = (
                base_feature_names.index("same_ccf_level")
                if "same_ccf_level" in base_feature_names
                else None
            )
            idx_candidate_in_accepted_corpus = (
                base_feature_names.index("candidate_in_accepted_corpus")
                if "candidate_in_accepted_corpus" in base_feature_names
                else None
            )

            def _row(label: int, jid: str, neg_type: str) -> dict:
                feats = candidate_features.get(jid) or []
                # 2026-06-26: ablation JSON 里 candidate_features 是 20 维
                # (含 paper_strength 占位在 idx 18),新 schema 是 19 维。
                # trim 掉 idx 18 保持 base 19 维(防御性,旧数据混进新管道时)。
                if len(feats) > 19:
                    feats = list(feats[:18]) + list(feats[19:])
                elif len(feats) < 19:
                    feats = list(feats) + [0.0] * (19 - len(feats))
                else:
                    feats = list(feats)
                # 2026-06-26: 用真实元数据覆写 4 个 dead 特征(覆盖 ablation
                # JSON 里全 0 的占位)。这些值之前是 dead 的,现在接通后能
                # 让 LR 学到 "gold subject_tags ∩ paper research_area"、
                # "paper gold ccf level"、以及 "candidate 在 accepted corpus
                # 中" 三类强信号。
                if idx_same_gold_area is not None:
                    feats[idx_same_gold_area] = (
                        1.0 if (paper_research_area & gold_subject_tags) else 0.0
                    )
                if idx_same_parsed_ccf_area is not None:
                    feats[idx_same_parsed_ccf_area] = (
                        1.0
                        if (paper_ccf_research_area & gold_subject_tags)
                        else 0.0
                    )
                if idx_same_ccf_level is not None:
                    cand_meta = journals_by_id.get(jid) or {}
                    cand_ccf = (cand_meta.get("ccf_rating") or "").upper()
                    feats[idx_same_ccf_level] = (
                        1.0
                        if (
                            paper_ccf_target_level
                            and cand_ccf
                            and paper_ccf_target_level == cand_ccf
                        )
                        else 0.0
                    )
                if idx_candidate_in_accepted_corpus is not None:
                    feats[idx_candidate_in_accepted_corpus] = (
                        1.0 if jid in accepted_jid_set_resolved else 0.0
                    )
                if use_evidence_schema:
                    feats = list(feats) + _evidence_vector_for_row(
                        paper_id, paper_venue, jid, evidence_lookup
                    )
                # 阶段 6.5 (P2-mini): 27-dim schema 时附加 2 维 tier/area。
                if use_27_dim_schema:
                    journal_meta = journals_by_id.get(jid, {})
                    feats = list(feats) + [
                        _tier_weight_value(journal_meta.get("ccf_rating")),
                        _area_exclusivity_value(
                            candidate_subject_tags=journal_meta.get("subject_tags") or [],
                            paper_anchor_area=paper_anchor_area,
                            n_matching_in_pool=n_matching_in_pool,
                        ),
                    ]
                return {
                    "paper_id": paper_id,
                    "journal_id": jid,
                    "label": label,
                    "features": feats,
                    "feature_names": feature_names,
                    "negative_type": neg_type,
                    "variant": variant_name,
                }

            # 1. 正样本(若 gold 期刊在 candidate_features)
            if target_jid in candidate_features:
                yield _row(1, target_jid, "gold")

            # 2. 负样本
            # 优先用 paper_result["rule_top20"](per plan 4.2);缺省时回退到 rule_top5。
            rule_top20 = paper_result.get("rule_top20") or paper_result.get("rule_top5") or []
            neg_list = _build_negatives(
                candidate_jids=list(candidate_features.keys()),
                target_jid=target_jid,
                rule_top20=rule_top20,
                journals_by_id=journals_by_id,
                max_negatives=max_negatives,
            )
            for jid, neg_type in neg_list:
                yield _row(0, jid, neg_type)


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



def _extract_route_combination(features: List[float]) -> str:  # noqa: F811
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

    # 2026-06-26: 4 dead features nonzero 计数(证明 augmentation 真生效)。
    # 旧 ablation JSON 里这些位置全 0;新 build_training_rows._row()
    # 会用真实元数据覆写,所以 nonzero > 0 是数据接通的标志。
    dead_feature_names = [
        "same_gold_area",
        "same_parsed_ccf_area",
        "same_ccf_level",
        "candidate_in_accepted_corpus",
    ]
    dead_feature_nonzero: Dict[str, int] = {f: 0 for f in dead_feature_names}

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
        # 2026-06-26: 4 dead features nonzero 计数(用 row 自身 feature_names
        # 动态查 idx — schema 改 19/25/27 维时位置可能不同)。
        fns = row.get("feature_names") or []
        for fname in dead_feature_names:
            if fname in fns:
                idx = fns.index(fname)
                if idx < len(feats) and feats[idx] > 0.0:
                    dead_feature_nonzero[fname] += 1

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
        "dead_feature_nonzero": dead_feature_nonzero,  # 2026-06-26 新加
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
    parser.add_argument(
        "--evidence-snapshot",
        default=None,
        help=(
            "Task 6.4 (2026-06-26: 25-dim LTR retrain): path to a "
            "precompute_evidence.py snapshot JSON. When supplied, each "
            "training row's 19-dim base features are extended with the "
            "6 LLM-evidence fields for the (paper, journal_id) pair, "
            "and feature_names is set to FEATURE_NAMES_WITH_LLM_EVIDENCE. "
            "Without this flag, output is the legacy 19-dim schema."
        ),
    )
    parser.add_argument(
        "--papers-jsonl",
        default=None,
        help=(
            "阶段 6.5 (P2-mini, 2026-06-26: 27-dim schema): path to papers "
            "metadata jsonl (e.g. papers_metadata_540.jsonl). Used to join "
            "paper.research_area for area_exclusivity feature. Required "
            "when --evidence-snapshot is also passed (27-dim path)."
        ),
    )
    parser.add_argument(
        "--accepted-corpus-dir",
        default="data/accepted_papers",
        help=(
            "2026-06-26: path to AcceptedPaperStore directory. When supplied, "
            "the build script loads the corpus and uses the set of journal_ids "
            "with papers to compute candidate_in_accepted_corpus. Default: "
            "data/accepted_papers (the project's standard corpus location)."
        ),
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

    # 阶段 6.5: 读 papers metadata 拿 research_area,join 到 ablation JSON。
    papers_by_title: Dict[str, dict] = {}
    if args.papers_jsonl:
        for line in Path(args.papers_jsonl).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            title = rec.get("title")
            if title:
                papers_by_title[title] = rec
        print(f"Loaded research_area for {len(papers_by_title)} papers from {args.papers_jsonl}")

    evidence_lookup = None
    if args.evidence_snapshot:
        evidence_lookup = _load_evidence_lookup(args.evidence_snapshot)
        if not evidence_lookup:
            print(
                f"[warn] --evidence-snapshot {args.evidence_snapshot} has no "
                "per-paper evidence; output will use 19-dim defaults."
            )
            evidence_lookup = None
        else:
            print(
                f"Loaded evidence lookup for {len(evidence_lookup)} papers "
                f"from {args.evidence_snapshot}"
            )

    # 2026-06-26: 加载 AcceptedPaperStore 构造 accepted_jid_set,用于计算
    # candidate_in_accepted_corpus。set 取 _by_journal.keys() (只含至少
    # 有一篇 paper 的 jid;空 journal 不计入)。
    accepted_jid_set: set = set()
    if args.accepted_corpus_dir:
        try:
            from src.journals.accepted_paper_store import AcceptedPaperStore
            accepted_store = AcceptedPaperStore(accepted_dir=args.accepted_corpus_dir)
            accepted_store.load()
            accepted_jid_set = set(accepted_store._by_journal.keys())
            print(
                f"Loaded {accepted_store.journal_count} journals "
                f"({accepted_store.count} papers) from {args.accepted_corpus_dir}"
            )
        except Exception as e:
            print(
                f"[warn] failed to load AcceptedPaperStore from "
                f"{args.accepted_corpus_dir}: {e}; accepted_jid_set will be empty",
                file=sys.stderr,
            )

    rows = list(
        build_training_rows(
            ablation_data=ablation_data,
            journals_by_id=journals_by_id,
            max_negatives=args.max_negatives,
            only_variants=args.variants,
            evidence_lookup=evidence_lookup,
            papers_by_title=papers_by_title,
            accepted_jid_set=accepted_jid_set,
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
        report["feature_schema"] = (
            "25_dim_with_llm_evidence"
            if evidence_lookup is not None
            else "19_dim_base"
        )
        report["evidence_snapshot"] = (
            str(args.evidence_snapshot) if args.evidence_snapshot else None
        )
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
