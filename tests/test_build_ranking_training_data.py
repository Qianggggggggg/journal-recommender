"""Tests for scripts/build_ranking_training_data.py (Task 4.1.f + 4.3)."""
import json
from pathlib import Path

import pytest

from scripts.build_ranking_training_data import (
    _build_negatives,
    _classify_negative,
    _extract_route_combination,
    build_training_report,
    build_training_rows,
)
from src.ranker.feature_builder import FEATURE_NAMES, MISSING_RANK_SENTINEL


def _make_features(value: float) -> list:
    return [float(value)] * len(FEATURE_NAMES)


def test_classify_negative_returns_none_for_gold():
    """jid == target_jid → None(表示这是正样本,不属于负样本分类)。"""
    journals = {"a": {"subject_tags": ["ai"]}, "b": {"subject_tags": ["ai"]}}
    assert _classify_negative("a", "a", ["a", "b"], journals) is None


def test_classify_negative_returns_hard_when_in_rule_top20():
    """出现在 rule_top20 的非 gold 期刊 → hard_rule_top20。"""
    journals = {"a": {"subject_tags": ["ai"]}, "b": {"subject_tags": ["cv"]}}
    assert _classify_negative("b", "a", ["a", "b"], journals) == "hard_rule_top20"


def test_classify_negative_returns_same_area_when_subject_tags_overlap():
    """非 rule_top20 但 subject_tags 与 gold 重叠 → same_area。"""
    journals = {"a": {"subject_tags": ["ai"]}, "b": {"subject_tags": ["ai", "ml"]}}
    assert _classify_negative("b", "a", ["a"], journals) == "same_area"


def test_classify_negative_returns_easy_other_otherwise():
    """既不在 rule_top20 也不在同领域 → easy_other。"""
    journals = {"a": {"subject_tags": ["ai"]}, "b": {"subject_tags": ["graphics"]}}
    assert _classify_negative("b", "a", ["a"], journals) == "easy_other"


def test_build_negatives_respects_max_negatives_cap():
    """负样本数不能超过 max_negatives。"""
    journals = {}
    cands = [f"j{i}" for i in range(20)]
    result = _build_negatives(cands, "gold", [], journals, max_negatives=5)
    assert len(result) == 5
    # 全是 easy_other(没 journals 信息,也没 rule_top20)
    assert all(t == "easy_other" for _, t in result)


def test_build_negatives_prioritizes_hard_then_same_area_then_easy():
    """hard > same_area > easy_other 的优先级必须实现。"""
    journals = {
        "gold": {"subject_tags": ["ai"]},
        "hard1": {"subject_tags": ["cv"]},
        "hard2": {"subject_tags": ["graphics"]},
        "same1": {"subject_tags": ["ai"]},
        "easy1": {"subject_tags": ["security"]},
        "easy2": {"subject_tags": ["theory"]},
    }
    cands = ["hard1", "hard2", "same1", "easy1", "easy2"]
    rule_top20 = ["gold", "hard1", "hard2"]
    result = _build_negatives(cands, "gold", rule_top20, journals, max_negatives=2)
    jids = [jid for jid, _ in result]
    assert jids == ["hard1", "hard2"]  # 优先级最高先填


def test_build_training_rows_emits_positive_and_negatives():
    """每篇 paper 应产出 1 个正样本(若 gold 在 features 中)+ 至多 10 个负样本。"""
    ablation = {
        "variants": {
            "hybrid": {
                "feature_names": list(FEATURE_NAMES),
                "paper_results": [
                    {
                        "title": "Paper 1",
                        "target_journal_id": "gold",
                        "candidate_features": {
                            "gold": _make_features(0.9),
                            "neg_hard": _make_features(0.5),
                            "neg_easy": _make_features(0.1),
                        },
                        "rule_top5": ["gold", "neg_hard"],
                    }
                ],
            }
        }
    }
    journals = {
        "gold": {"subject_tags": ["ai"]},
        "neg_hard": {"subject_tags": ["cv"]},
        "neg_easy": {"subject_tags": ["graphics"]},
    }
    rows = list(build_training_rows(ablation, journals, max_negatives=10))
    # 1 positive + 2 negatives = 3
    assert len(rows) == 3
    pos_rows = [r for r in rows if r["label"] == 1]
    neg_rows = [r for r in rows if r["label"] == 0]
    assert len(pos_rows) == 1
    assert pos_rows[0]["journal_id"] == "gold"
    assert pos_rows[0]["negative_type"] == "gold"
    assert pos_rows[0]["features"] == _make_features(0.9)
    assert neg_rows[0]["negative_type"] == "hard_rule_top20"
    assert neg_rows[1]["negative_type"] == "easy_other"


def test_build_training_rows_skips_paper_without_gold_in_features():
    """如果 gold 期刊不在 candidate_features 中(完全没召回),该 paper 只产负样本。"""
    ablation = {
        "variants": {
            "hybrid": {
                "feature_names": list(FEATURE_NAMES),
                "paper_results": [
                    {
                        "title": "Paper missed",
                        "target_journal_id": "gold",
                        "candidate_features": {
                            "neg1": _make_features(0.3),
                        },
                        "rule_top5": ["neg1"],
                    }
                ],
            }
        }
    }
    journals = {"gold": {"subject_tags": ["ai"]}, "neg1": {"subject_tags": ["cv"]}}
    rows = list(build_training_rows(ablation, journals))
    # 没有正样本,只有 1 个负样本
    assert all(r["label"] == 0 for r in rows)
    assert len(rows) == 1


def test_build_training_rows_respects_max_negatives_per_paper():
    """max_negatives=3 时,每篇 paper 最多 1 正 + 3 负 = 4 行。"""
    ablation = {
        "variants": {
            "hybrid": {
                "feature_names": list(FEATURE_NAMES),
                "paper_results": [
                    {
                        "title": f"P{i}",
                        "target_journal_id": "gold",
                        "candidate_features": {
                            "gold": _make_features(0.9),
                            **{f"n{j}": _make_features(0.1) for j in range(10)},
                        },
                        "rule_top5": ["gold"],
                    }
                    for i in range(2)
                ],
            }
        }
    }
    rows = list(build_training_rows(ablation, {}, max_negatives=3))
    # 2 papers × (1 positive + 3 negatives) = 8
    assert len(rows) == 8
    # 每篇 paper 恰好 4 行
    from collections import Counter
    counts = Counter(r["paper_id"] for r in rows)
    assert counts["P0"] == 4
    assert counts["P1"] == 4


def test_build_training_rows_filters_variants():
    """only_variants=['full_hybrid'] 时,只处理 full_hybrid,跳过 hybrid。"""
    ablation = {
        "variants": {
            "hybrid": {
                "feature_names": list(FEATURE_NAMES),
                "paper_results": [
                    {"title": "P", "target_journal_id": "g", "candidate_features": {"g": _make_features(0.5)}, "rule_top5": []}
                ],
            },
            "full_hybrid": {
                "feature_names": list(FEATURE_NAMES),
                "paper_results": [
                    {"title": "P", "target_journal_id": "g", "candidate_features": {"g": _make_features(0.9)}, "rule_top5": []}
                ],
            },
        }
    }
    rows = list(build_training_rows(ablation, {}, only_variants=["full_hybrid"]))
    assert len(rows) == 1
    assert rows[0]["variant"] == "full_hybrid"
    assert rows[0]["features"] == _make_features(0.9)


# ---- Task 4.3: sidecar report + 80% warning ----


def _ablation_with_paper_features(papers: list) -> dict:
    """构造 ablation-like 数据,每条 paper 有 retrieval_rank + candidate_features。"""
    return {
        "variants": {
            "hybrid": {
                "feature_names": list(FEATURE_NAMES),
                "paper_results": papers,
            }
        }
    }


def test_extract_route_combination_returns_present_routes_sorted():
    """_extract_route_combination 从 features 向量中提取"实际出现"的 route 集合,排序后用 + 连接。"""
    # 构造 features: scope_bm25=3 (有), scope_vector=999 (缺), typical_bm25=5 (有), accepted_bm25=999 (缺)
    feats = [MISSING_RANK_SENTINEL] * len(FEATURE_NAMES)
    feats[FEATURE_NAMES.index("scope_bm25_rank")] = 3.0
    feats[FEATURE_NAMES.index("typical_bm25_rank")] = 5.0
    combination = _extract_route_combination(feats)
    # 应当返回 sorted 的非空 routes: "scope_bm25+typical_bm25"
    assert combination == "scope_bm25+typical_bm25"


def test_extract_route_combination_returns_empty_when_no_routes():
    """所有 route 都缺失时,返回空字符串(防止组合成空标签)。"""
    feats = [MISSING_RANK_SENTINEL] * len(FEATURE_NAMES)
    assert _extract_route_combination(feats) == ""


def test_build_training_report_warns_when_fewer_than_80pct_positives_in_top50():
    """如果 < 80% 正样本的 retrieval_rank <= 50,retrieval_topk_80_warning=True。"""
    papers = [
        {"title": "P1", "retrieval_rank": 10, "target_journal_id": "g1", "candidate_features": {"g1": _make_features(0.9)}, "rule_top5": []},
        {"title": "P2", "retrieval_rank": 5, "target_journal_id": "g2", "candidate_features": {"g2": _make_features(0.9)}, "rule_top5": []},
        {"title": "P3", "retrieval_rank": 30, "target_journal_id": "g3", "candidate_features": {"g3": _make_features(0.9)}, "rule_top5": []},
        {"title": "P4", "retrieval_rank": 999, "target_journal_id": "g4", "candidate_features": {"g4": _make_features(0.9)}, "rule_top5": []},
    ]
    ablation = _ablation_with_paper_features(papers)
    pos_rows = [r for r in build_training_rows(ablation, {}) if r["label"] == 1]
    report = build_training_report(ablation, pos_rows)
    assert report["positives_with_target_in_top50_count"] == 3
    assert abs(report["positives_with_target_in_top50_ratio"] - 0.75) < 1e-6
    assert report["retrieval_topk_80_warning"] is True


def test_build_training_report_no_warning_when_above_80pct_threshold():
    """≥ 80% 正样本 retrieval_rank <= 50 时,warning=False。"""
    papers = [
        {"title": "P1", "retrieval_rank": 10, "target_journal_id": "g1", "candidate_features": {"g1": _make_features(0.9)}, "rule_top5": []},
        {"title": "P2", "retrieval_rank": 5, "target_journal_id": "g2", "candidate_features": {"g2": _make_features(0.9)}, "rule_top5": []},
        {"title": "P3", "retrieval_rank": 30, "target_journal_id": "g3", "candidate_features": {"g3": _make_features(0.9)}, "rule_top5": []},
        {"title": "P4", "retrieval_rank": 40, "target_journal_id": "g4", "candidate_features": {"g4": _make_features(0.9)}, "rule_top5": []},
        {"title": "P5", "retrieval_rank": 999, "target_journal_id": "g5", "candidate_features": {"g5": _make_features(0.9)}, "rule_top5": []},
    ]
    ablation = _ablation_with_paper_features(papers)
    pos_rows = [r for r in build_training_rows(ablation, {}) if r["label"] == 1]
    report = build_training_report(ablation, pos_rows)
    # 4/5 = 80% (>= 80%,不报警)
    assert report["positives_with_target_in_top50_count"] == 4
    assert abs(report["positives_with_target_in_top50_ratio"] - 0.8) < 1e-6
    assert report["retrieval_topk_80_warning"] is False


def test_build_training_report_counts_positives_missing_route_features():
    """统计正样本中 route 特征为哨兵(999)的数量;按 feature 分桶计数。"""
    # 3 个正样本: P1 没有 accepted_bm25,P2 没有 accepted_bm25,P3 有 accepted_bm25
    papers = [
        {
            "title": "P1", "retrieval_rank": 5, "target_journal_id": "g1",
            "candidate_features": {"g1": [999.0] * len(FEATURE_NAMES)},  # 全缺失
            "rule_top5": [],
        },
        {
            "title": "P2", "retrieval_rank": 5, "target_journal_id": "g2",
            "candidate_features": {"g2": [999.0] * len(FEATURE_NAMES)},  # 全缺失
            "rule_top5": [],
        },
        {
            "title": "P3", "retrieval_rank": 5, "target_journal_id": "g3",
            "candidate_features": {"g3": _make_features(0.9)},  # 全部非哨兵
            "rule_top5": [],
        },
    ]
    ablation = _ablation_with_paper_features(papers)
    pos_rows = [r for r in build_training_rows(ablation, {}) if r["label"] == 1]
    report = build_training_report(ablation, pos_rows)
    # P1, P2 在 scope_bm25_rank 上是 999(哨兵),P3 不是
    scope_bm25_idx = FEATURE_NAMES.index("scope_bm25_rank")
    # 检查每个 route feature 的缺失计数
    missing = report["positives_missing_route_features"]
    assert missing["scope_bm25_rank"] == 2  # P1, P2 缺失
    assert missing["accepted_bm25_rank"] == 2  # P1, P2 也缺失
    # P3 的所有 route 都存在,所以对于 P3 涉及的所有 route 缺失计数应为 2
    # 但 accepted_vector_rank 也是 2(P1, P2 缺失)
    assert missing["accepted_vector_rank"] == 2


def test_build_training_report_counts_route_combinations():
    """route combination 分布:对正样本统计"哪些 route 命中",按 string 计数。"""
    # P1: scope_bm25 命中 → "scope_bm25"
    feats_p1 = [MISSING_RANK_SENTINEL] * len(FEATURE_NAMES)
    feats_p1[FEATURE_NAMES.index("scope_bm25_rank")] = 3.0
    # P2: scope_bm25 + typical_bm25 → "scope_bm25+typical_bm25"
    feats_p2 = [MISSING_RANK_SENTINEL] * len(FEATURE_NAMES)
    feats_p2[FEATURE_NAMES.index("scope_bm25_rank")] = 2.0
    feats_p2[FEATURE_NAMES.index("typical_bm25_rank")] = 7.0
    # P3: 仅 typical_bm25 → "typical_bm25"
    feats_p3 = [MISSING_RANK_SENTINEL] * len(FEATURE_NAMES)
    feats_p3[FEATURE_NAMES.index("typical_bm25_rank")] = 4.0

    papers = [
        {"title": "P1", "retrieval_rank": 5, "target_journal_id": "g1", "candidate_features": {"g1": feats_p1}, "rule_top5": []},
        {"title": "P2", "retrieval_rank": 5, "target_journal_id": "g2", "candidate_features": {"g2": feats_p2}, "rule_top5": []},
        {"title": "P3", "retrieval_rank": 5, "target_journal_id": "g3", "candidate_features": {"g3": feats_p3}, "rule_top5": []},
    ]
    ablation = _ablation_with_paper_features(papers)
    pos_rows = [r for r in build_training_rows(ablation, {}) if r["label"] == 1]
    report = build_training_report(ablation, pos_rows)
    combinations = report["route_combination_counts"]
    assert combinations["scope_bm25"] == 1
    assert combinations["scope_bm25+typical_bm25"] == 1
    assert combinations["typical_bm25"] == 1


def test_build_training_report_includes_positive_and_negative_counts():
    """report 必须包含正负样本计数(per variant 与 overall)。"""
    papers = [
        {"title": "P1", "retrieval_rank": 5, "target_journal_id": "g1", "candidate_features": {"g1": _make_features(0.9), "n1": _make_features(0.1)}, "rule_top5": []},
    ]
    ablation = _ablation_with_paper_features(papers)
    rows = list(build_training_rows(ablation, {}))
    pos_rows = [r for r in rows if r["label"] == 1]
    report = build_training_report(ablation, pos_rows)
    assert report["positives_total"] == 1
    assert report["positives_by_variant"] == {"hybrid": 1}
