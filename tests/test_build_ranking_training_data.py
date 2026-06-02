"""Tests for scripts/build_ranking_training_data.py (Task 4.1.f)."""
import json
from pathlib import Path

import pytest

from scripts.build_ranking_training_data import (
    _build_negatives,
    _classify_negative,
    build_training_rows,
)
from src.ranker.feature_builder import FEATURE_NAMES


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
