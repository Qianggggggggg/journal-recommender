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
    # 2026-06-26: dead features (idx 14, 15, 16, 18) now recomputed.
    # No papers_by_title → all 4 → 0.0 (no paper metadata, no accepted corpus).
    expected_pos_features = _make_features(0.9)
    for dead_idx in (14, 15, 16, 18):
        expected_pos_features[dead_idx] = 0.0
    assert pos_rows[0]["features"] == expected_pos_features
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
    # 2026-06-26: dead features (idx 14, 15, 16, 18) are now recomputed from
    # metadata — with empty journals + no papers_by_title, all 4 → 0.0.
    expected_features = _make_features(0.9)
    for dead_idx in (14, 15, 16, 18):
        expected_features[dead_idx] = 0.0
    assert rows[0]["features"] == expected_features


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


# ---------------------------------------------------------------------------
# Task 6.4 — 26-dim schema with LLM evidence lookup
# ---------------------------------------------------------------------------


def test_build_training_rows_outputs_19_dim_by_default():
    """No evidence_lookup → 19-dim schema (2026-06-26: paper_strength removed, was 20).

    The ablation JSON may carry legacy 20-dim candidate_features (paper_strength
    position); build_training_rows must trim idx 18 (paper_strength) before
    writing the row so output is always 19-dim.
    """
    papers = [
        {
            "title": "P1", "retrieval_rank": 5,
            "target_journal_id": "g1",
            "candidate_features": {"g1": _make_features(0.9), "n1": _make_features(0.1)},
            "rule_top5": [],
        }
    ]
    ablation = _ablation_with_paper_features(papers)
    rows = list(build_training_rows(ablation, {}))
    assert all(len(r["features"]) == 19 for r in rows)
    assert all(r["feature_names"] == FEATURE_NAMES for r in rows)


def test_build_training_rows_appends_6_evidence_fields_when_lookup_supplied():
    """With evidence_lookup, each row's features become 26-dim and
    feature_names switches to FEATURE_NAMES_WITH_LLM_EVIDENCE.

    The snapshot key is ``title | venue`` (both casefold-normalized),
    mirroring ``precompute_evidence._paper_key``. Lookup is keyed off
    the paper's (title, venue) pair, not the title alone.
    """
    from src.ranker.feature_builder import FEATURE_NAMES_WITH_LLM_EVIDENCE

    papers = [
        {
            "title": "Paper One", "venue": "Journal of Foo",
            "retrieval_rank": 5,
            "target_journal_id": "g1",
            "candidate_features": {"g1": _make_features(0.9), "n1": _make_features(0.1)},
            "rule_top5": [],
        }
    ]
    ablation = _ablation_with_paper_features(papers)
    evidence_lookup = {
        "paper one | journal of foo": {
            "g1": {
                "scope_fit": 0.9, "method_fit": 0.8,
                "application_fit": 0.7, "journal_position_fit": 0.85,
                "too_broad_penalty": 0.1, "too_narrow_penalty": 0.05,
            },
            "n1": {
                "scope_fit": 0.2, "method_fit": 0.3,
                "application_fit": 0.4, "journal_position_fit": 0.25,
                "too_broad_penalty": 0.0, "too_narrow_penalty": 0.0,
            },
        }
    }
    rows = list(build_training_rows(ablation, {}, evidence_lookup=evidence_lookup))
    assert all(len(r["features"]) == 25 for r in rows)
    assert all(r["feature_names"] == FEATURE_NAMES_WITH_LLM_EVIDENCE for r in rows)
    # The 6 appended evidence values match the snapshot
    g1_row = next(r for r in rows if r["journal_id"] == "g1" and r["label"] == 1)
    assert g1_row["features"][19:] == [0.9, 0.8, 0.7, 0.85, 0.1, 0.05]
    n1_row = next(r for r in rows if r["journal_id"] == "n1")
    assert n1_row["features"][19:] == [0.2, 0.3, 0.4, 0.25, 0.0, 0.0]


def test_build_training_rows_uses_neutral_defaults_for_missing_evidence():
    """When the snapshot has no entry for (paper, journal_id), the row
    uses neutral defaults (fit=0.5, penalty=0.0)."""
    from src.ranker.feature_builder import LLM_EVIDENCE_FEATURE_NAMES

    papers = [
        {
            "title": "P1", "venue": "V1",
            "retrieval_rank": 5,
            "target_journal_id": "g1",
            "candidate_features": {"g1": _make_features(0.9), "n1": _make_features(0.1)},
            "rule_top5": [],
        }
    ]
    ablation = _ablation_with_paper_features(papers)
    # Lookup only has g1, not n1
    evidence_lookup = {
        "p1 | v1": {
            "g1": {
                "scope_fit": 0.9, "method_fit": 0.8, "application_fit": 0.7,
                "journal_position_fit": 0.85, "too_broad_penalty": 0.1, "too_narrow_penalty": 0.05,
            },
        }
    }
    rows = list(build_training_rows(ablation, {}, evidence_lookup=evidence_lookup))
    g1_row = next(r for r in rows if r["journal_id"] == "g1")
    n1_row = next(r for r in rows if r["journal_id"] == "n1")
    assert g1_row["features"][19:] == [0.9, 0.8, 0.7, 0.85, 0.1, 0.05]
    # n1 falls back to neutral: 4 fits at 0.5, 2 penalties at 0.0
    expected_neutral = [
        0.5, 0.5, 0.5, 0.5,  # fits
        0.0, 0.0,  # penalties
    ]
    assert n1_row["features"][19:] == expected_neutral
    # Sanity: 6 evidence values in the order declared in LLM_EVIDENCE_FEATURE_NAMES
    assert len(n1_row["features"][19:]) == len(LLM_EVIDENCE_FEATURE_NAMES)


def test_build_training_rows_uses_neutral_defaults_for_invalid_evidence_values():
    """Out-of-range or non-numeric evidence values must NOT be passed through
    to training data; they fall back to neutral defaults (matches the
    ranker's runtime behavior, so train/inference agree on bad-input policy)."""
    papers = [
        {
            "title": "P1", "venue": "V1",
            "retrieval_rank": 5,
            "target_journal_id": "g1",
            "candidate_features": {"g1": _make_features(0.9)},
            "rule_top5": [],
        }
    ]
    ablation = _ablation_with_paper_features(papers)
    # scope_fit=1.5 is out of [0,1]; method_fit="bad" is non-numeric
    evidence_lookup = {
        "p1 | v1": {
            "g1": {
                "scope_fit": 1.5, "method_fit": "bad", "application_fit": 0.6,
                "journal_position_fit": -0.1, "too_broad_penalty": "x",
                "too_narrow_penalty": None,
            },
        }
    }
    rows = list(build_training_rows(ablation, {}, evidence_lookup=evidence_lookup))
    g1 = next(r for r in rows if r["journal_id"] == "g1")
    # All 6 fields fell back to defaults (only application_fit=0.6 was valid)
    assert g1["features"][19:] == [0.5, 0.5, 0.6, 0.5, 0.0, 0.0]


def test_build_training_rows_uses_title_only_key_when_venue_empty():
    """When ``venue`` is empty, the snapshot's paper_key is ``title | ''
    (empty venue, trailing separator). The lookup mirrors this exact
    format so it stays in sync with how precompute_evidence stores keys.
    """
    from src.ranker.feature_builder import FEATURE_NAMES_WITH_LLM_EVIDENCE

    papers = [
        {
            "title": "Standalone Paper", "venue": "",
            "retrieval_rank": 1,
            "target_journal_id": "g1",
            "candidate_features": {"g1": _make_features(0.9)},
            "rule_top5": [],
        }
    ]
    ablation = _ablation_with_paper_features(papers)
    evidence_lookup = {
        "standalone paper | ": {  # mirrors precompute_evidence._paper_key
            "g1": {
                "scope_fit": 0.7, "method_fit": 0.6, "application_fit": 0.5,
                "journal_position_fit": 0.4, "too_broad_penalty": 0.0, "too_narrow_penalty": 0.0,
            },
        }
    }
    rows = list(build_training_rows(ablation, {}, evidence_lookup=evidence_lookup))
    g1 = next(r for r in rows if r["journal_id"] == "g1")
    assert g1["features"][19:] == [0.7, 0.6, 0.5, 0.4, 0.0, 0.0]
    # Schema is 25-dim (2026-06-26: was 26)
    assert len(g1["features"]) == 25
    assert g1["feature_names"] == FEATURE_NAMES_WITH_LLM_EVIDENCE


# ---------------------------------------------------------------------------
# 2026-06-26: Accepted Corpus LTR — 4 dead features recompute
# (same_gold_area, same_parsed_ccf_area, same_ccf_level, candidate_in_accepted_corpus)
# ---------------------------------------------------------------------------


def _make_19_dim_base_features(
    journal_ids: list, accepted_jid_set: set = None
) -> dict:
    """Build a 19-dim base feature vector per jid (no paper_strength).

    Schema (per src/ranker/feature_builder.py):
      0:retrieval_rank, 1:rule_rank, 2:rule_score,
      3-8: scope/typical/accepted bm25/vector rank (999 sentinel = missing)
      9:route_count, 10:has_scope_route, 11:has_typical_route,
      12:has_accepted_route, 13:has_identity_anchor,
      14:same_gold_area, 15:same_parsed_ccf_area, 16:same_ccf_level,
      17:journal_ccf_numeric, 18:candidate_in_accepted_corpus
    """
    accepted_jid_set = accepted_jid_set or set()
    out = {}
    for jid in journal_ids:
        feats = [999.0] * 19
        feats[0] = 1.0   # retrieval_rank
        feats[1] = 999.0  # rule_rank (missing)
        feats[2] = 0.0   # rule_score
        # 3-8 stay 999
        feats[9] = 0.0   # route_count
        feats[10] = 0.0  # has_scope_route
        feats[11] = 0.0  # has_typical_route
        feats[12] = 0.0  # has_accepted_route
        feats[13] = 0.0  # has_identity_anchor
        # 14-17 dead placeholders (will be overwritten by build_training_rows)
        feats[14] = 0.0
        feats[15] = 0.0
        feats[16] = 0.0
        feats[17] = 0.0
        feats[18] = 1.0 if jid in accepted_jid_set else 0.0
        out[jid] = feats
    return out


def _make_journal_meta(jid: str, subject_tags: list, ccf_rating: str) -> dict:
    return {
        "journal_id": jid,
        "journal_name": jid.upper(),
        "subject_tags": subject_tags,
        "ccf_rating": ccf_rating,
    }


def _make_paper_meta(title: str, research_area: list, ccf_research_area: list = None) -> dict:
    return {
        "title": title,
        "research_area": research_area,
        "ccf_research_area": ccf_research_area if ccf_research_area is not None else research_area,
    }


def _ablation_with_one_paper(
    paper_title: str, target_jid: str, candidate_jids: list,
    candidate_features: dict, rule_top20: list = None,
) -> dict:
    return {
        "variants": {
            "full_hybrid": {
                "feature_names": None,  # 让 build_training_rows 走 FEATURE_NAMES
                "paper_results": [
                    {
                        "title": paper_title,
                        "venue": "",
                        "retrieval_rank": 1,
                        "target_journal_id": target_jid,
                        "rule_top20": rule_top20 or [target_jid],
                        "candidate_features": candidate_features,
                    }
                ],
            }
        }
    }


def test_same_gold_area_computed_when_research_area_overlaps():
    """same_gold_area=1.0 when paper.research_area ∩ gold.subject_tags ≠ ∅.

    Gold journal subject_tags=["AI","ML"], paper.research_area=["AI"] → overlap → 1.0
    for both the gold row and the neg row (same_gold_area is a paper×gold signal,
    not a candidate-level signal — it's 1.0 for all candidates of this paper when
    paper.research_area matches the gold venue's tags).

    Counter-test: a paper whose research_area=["Databases"] with same gold subject_tags
    gets same_gold_area=0.0 (no overlap).
    """
    paper_title = "Test Paper A"
    target_jid = "gold_j"
    neg_jid = "neg_j"
    candidate_jids = [target_jid, neg_jid]
    candidate_features = _make_19_dim_base_features(candidate_jids)

    papers_by_title = {
        paper_title: _make_paper_meta(paper_title, research_area=["AI"]),
    }
    journals_by_id = {
        target_jid: _make_journal_meta(target_jid, ["AI", "ML"], "A"),
        neg_jid: _make_journal_meta(neg_jid, ["Databases"], "B"),
    }
    ablation_data = _ablation_with_one_paper(
        paper_title, target_jid, candidate_jids, candidate_features,
        rule_top20=[target_jid],
    )

    rows = list(build_training_rows(
        ablation_data=ablation_data,
        journals_by_id=journals_by_id,
        max_negatives=1,
        accepted_jid_set=set(),
        papers_by_title=papers_by_title,
    ))

    pos_rows = [r for r in rows if r["label"] == 1]
    assert len(pos_rows) == 1
    pos = pos_rows[0]
    same_gold_area_idx = pos["feature_names"].index("same_gold_area")
    # paper.research_area=["AI"] ∩ gold.subject_tags=["AI","ML"] → 1.0
    assert pos["features"][same_gold_area_idx] == 1.0

    neg_rows = [r for r in rows if r["label"] == 0]
    assert len(neg_rows) == 1
    neg = neg_rows[0]
    # same_gold_area is paper×gold, so it's also 1.0 for the neg candidate
    # (it's a property of the paper, not the candidate).
    assert neg["features"][same_gold_area_idx] == 1.0

    # Counter-test: a paper with research_area=["Databases"] for the same gold
    # journal → same_gold_area=0.0.
    paper_title_b = "Test Paper A-bis"
    papers_by_title_b = {
        paper_title_b: _make_paper_meta(paper_title_b, research_area=["Databases"]),
    }
    ablation_data_b = _ablation_with_one_paper(
        paper_title_b, target_jid, [target_jid], _make_19_dim_base_features([target_jid]),
        rule_top20=[target_jid],
    )
    rows_b = list(build_training_rows(
        ablation_data=ablation_data_b,
        journals_by_id=journals_by_id,
        max_negatives=0,
        accepted_jid_set=set(),
        papers_by_title=papers_by_title_b,
    ))
    pos_b = next(r for r in rows_b if r["label"] == 1)
    assert pos_b["features"][same_gold_area_idx] == 0.0


def test_same_parsed_ccf_area_computed_when_ccf_area_overlaps():
    """same_parsed_ccf_area=1.0 when paper.ccf_research_area ∩ gold.subject_tags ≠ ∅.

    paper.ccf_research_area=["人工智能"], gold.subject_tags=["人工智能","机器学习"] → 1.0.
    """
    paper_title = "Test Paper B"
    target_jid = "gold_j"
    candidate_features = _make_19_dim_base_features([target_jid])

    papers_by_title = {
        paper_title: _make_paper_meta(
            paper_title,
            research_area=["machine learning"],
            ccf_research_area=["人工智能"],
        ),
    }
    journals_by_id = {
        target_jid: _make_journal_meta(target_jid, ["人工智能", "机器学习"], "A"),
    }
    ablation_data = _ablation_with_one_paper(
        paper_title, target_jid, [target_jid], candidate_features,
        rule_top20=[target_jid],
    )

    rows = list(build_training_rows(
        ablation_data=ablation_data,
        journals_by_id=journals_by_id,
        max_negatives=0,
        accepted_jid_set=set(),
        papers_by_title=papers_by_title,
    ))
    pos = next(r for r in rows if r["label"] == 1)
    idx = pos["feature_names"].index("same_parsed_ccf_area")
    assert pos["features"][idx] == 1.0


def test_same_ccf_level_computed_when_levels_match():
    """same_ccf_level=1.0 when paper gold ccf=A matches candidate ccf=A; 0.0 when ccf=B."""
    paper_title = "Test Paper C"
    target_jid = "gold_j"   # ccf A
    other_a_jid = "other_a"  # ccf A
    other_b_jid = "other_b"  # ccf B
    candidate_jids = [target_jid, other_a_jid, other_b_jid]
    candidate_features = _make_19_dim_base_features(candidate_jids)

    papers_by_title = {paper_title: _make_paper_meta(paper_title, research_area=["AI"])}
    journals_by_id = {
        target_jid: _make_journal_meta(target_jid, ["AI"], "A"),
        other_a_jid: _make_journal_meta(other_a_jid, ["AI"], "A"),
        other_b_jid: _make_journal_meta(other_b_jid, ["AI"], "B"),
    }
    ablation_data = _ablation_with_one_paper(
        paper_title, target_jid, candidate_jids, candidate_features,
        rule_top20=[target_jid],
    )

    rows = list(build_training_rows(
        ablation_data=ablation_data,
        journals_by_id=journals_by_id,
        max_negatives=2,
        accepted_jid_set=set(),
        papers_by_title=papers_by_title,
    ))
    feature_names = rows[0]["feature_names"]
    idx = feature_names.index("same_ccf_level")

    by_jid = {r["journal_id"]: r for r in rows}
    assert by_jid[target_jid]["features"][idx] == 1.0   # gold ccf=A
    assert by_jid[other_a_jid]["features"][idx] == 1.0  # other A
    assert by_jid[other_b_jid]["features"][idx] == 0.0  # other B


def test_candidate_in_accepted_corpus_set_when_jid_in_corpus():
    """candidate_in_accepted_corpus=1.0 when jid ∈ AcceptedPaperStore, else 0.0."""
    paper_title = "Test Paper D"
    target_jid = "in_corpus_j"
    other_jid = "not_in_corpus_j"
    candidate_jids = [target_jid, other_jid]
    candidate_features = _make_19_dim_base_features(candidate_jids)

    papers_by_title = {paper_title: _make_paper_meta(paper_title, research_area=["AI"])}
    journals_by_id = {
        target_jid: _make_journal_meta(target_jid, ["AI"], "A"),
        other_jid: _make_journal_meta(other_jid, ["AI"], "A"),
    }
    accepted_jid_set = {target_jid}  # only target in corpus

    ablation_data = _ablation_with_one_paper(
        paper_title, target_jid, candidate_jids, candidate_features,
        rule_top20=[target_jid],
    )

    rows = list(build_training_rows(
        ablation_data=ablation_data,
        journals_by_id=journals_by_id,
        max_negatives=1,
        accepted_jid_set=accepted_jid_set,
        papers_by_title=papers_by_title,
    ))
    feature_names = rows[0]["feature_names"]
    idx = feature_names.index("candidate_in_accepted_corpus")
    by_jid = {r["journal_id"]: r for r in rows}
    assert by_jid[target_jid]["features"][idx] == 1.0
    assert by_jid[other_jid]["features"][idx] == 0.0


def test_base_features_19_dim_not_20():
    """2026-06-26: paper_strength removed → base features are 19-dim, not 20."""
    paper_title = "Test Paper E"
    target_jid = "gold_j"
    candidate_features = _make_19_dim_base_features([target_jid])
    papers_by_title = {paper_title: _make_paper_meta(paper_title, research_area=["AI"])}
    journals_by_id = {target_jid: _make_journal_meta(target_jid, ["AI"], "A")}
    ablation_data = _ablation_with_one_paper(
        paper_title, target_jid, [target_jid], candidate_features,
        rule_top20=[target_jid],
    )

    rows = list(build_training_rows(
        ablation_data=ablation_data,
        journals_by_id=journals_by_id,
        max_negatives=0,
        accepted_jid_set=set(),
        papers_by_title=papers_by_title,
    ))
    pos = next(r for r in rows if r["label"] == 1)
    # 19 base + 6 evidence = 25 (when evidence_lookup is None → just 19 base)
    # Without evidence_lookup the schema is 19-dim.
    assert len(pos["features"]) == 19
    assert len(pos["feature_names"]) == 19


def test_no_paper_strength_in_feature_names():
    """2026-06-26: paper_strength is gone from feature_names."""
    paper_title = "Test Paper F"
    target_jid = "gold_j"
    candidate_features = _make_19_dim_base_features([target_jid])
    papers_by_title = {paper_title: _make_paper_meta(paper_title, research_area=["AI"])}
    journals_by_id = {target_jid: _make_journal_meta(target_jid, ["AI"], "A")}
    ablation_data = _ablation_with_one_paper(
        paper_title, target_jid, [target_jid], candidate_features,
        rule_top20=[target_jid],
    )

    rows = list(build_training_rows(
        ablation_data=ablation_data,
        journals_by_id=journals_by_id,
        max_negatives=0,
        accepted_jid_set=set(),
        papers_by_title=papers_by_title,
    ))
    pos = next(r for r in rows if r["label"] == 1)
    assert "paper_strength" not in pos["feature_names"]
