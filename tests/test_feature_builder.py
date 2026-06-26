"""Tests for src/ranker/feature_builder.py (Task 4.1)."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.journals.accepted_paper_store import AcceptedPaperStore
from src.journals.journal_model import Journal
from src.journals.journal_store import JournalStore
from src.papers.paper_model import PaperProfile
from src.ranker.feature_builder import (
    FEATURE_NAMES,
    FEATURE_NAMES_WITH_LLM_EVIDENCE,
    FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY,
    LLM_EVIDENCE_FEATURE_NAMES,
    PaperCandidateFeatures,
    TIER_EXCLUSIVITY_FEATURE_NAMES,
    TIER_WEIGHT_BY_CCF,
    _area_exclusivity_value,
    _tier_weight_value,
    attach_features_to_trace,
    build_features,
    ccf_level_to_numeric,
)


def test_feature_names_is_a_locked_list_of_19_strings():
    """FEATURE_NAMES 是锁定 schema:19 个字符串特征名(per plan 4.1 + ADR 0001)。"""
    assert isinstance(FEATURE_NAMES, list)
    assert len(FEATURE_NAMES) == 19
    assert all(isinstance(n, str) for n in FEATURE_NAMES)
    # 防止重复
    assert len(set(FEATURE_NAMES)) == 19


def test_llm_evidence_feature_names_extend_locked_schema_without_changing_v1():
    """6.2 evidence schema 必须显式扩展,不能改变现有 19 维 LTR schema。"""
    assert LLM_EVIDENCE_FEATURE_NAMES == [
        "llm_scope_fit",
        "llm_method_fit",
        "llm_application_fit",
        "llm_journal_position_fit",
        "llm_too_broad_penalty",
        "llm_too_narrow_penalty",
    ]
    assert len(FEATURE_NAMES) == 19
    assert FEATURE_NAMES_WITH_LLM_EVIDENCE[:19] == FEATURE_NAMES
    assert FEATURE_NAMES_WITH_LLM_EVIDENCE[19:] == LLM_EVIDENCE_FEATURE_NAMES
    assert len(FEATURE_NAMES_WITH_LLM_EVIDENCE) == 25


def test_feature_names_includes_accepted_route_features():
    """accepted route 信号必须可学习(per ADR 0001)。"""
    assert "accepted_bm25_rank" in FEATURE_NAMES
    assert "accepted_vector_rank" in FEATURE_NAMES
    assert "has_accepted_route" in FEATURE_NAMES
    # candidate-level 信号(可推理)
    assert "candidate_in_accepted_corpus" in FEATURE_NAMES


def test_feature_names_excludes_gold_oracle_feature():
    """gold_in_accepted_corpus 严禁进 feature(per ADR 0001 防 oracle leakage)。"""
    assert "gold_in_accepted_corpus" not in FEATURE_NAMES
    # 其他可能的 oracle 命名也要排除
    for forbidden in [
        "gold_venue_in_corpus",
        "target_journal_in_corpus",
        "is_gold_covered",
    ]:
        assert forbidden not in FEATURE_NAMES, f"{forbidden} must not be a feature"


def test_paper_candidate_features_default_rank_is_sentinel_999():
    """缺失 rank 用 999 哨兵值(per plan 4.1);不能默认成 0(会扭曲排序)。"""
    f = PaperCandidateFeatures()
    assert f.retrieval_rank == 999.0
    assert f.rule_rank == 999.0
    assert f.scope_bm25_rank == 999.0
    assert f.scope_vector_rank == 999.0
    assert f.typical_bm25_rank == 999.0
    assert f.typical_vector_rank == 999.0
    assert f.accepted_bm25_rank == 999.0
    assert f.accepted_vector_rank == 999.0


def test_paper_candidate_features_to_vector_returns_one_value_per_feature_name():
    """to_vector() 返回的向量长度必须等于 FEATURE_NAMES 长度,顺序一致。"""
    f = PaperCandidateFeatures()
    vec = f.to_vector()
    assert len(vec) == len(FEATURE_NAMES)
    # 全部为 float
    assert all(isinstance(v, float) for v in vec)


def test_paper_candidate_features_to_vector_preserves_explicit_values_in_order():
    """显式传入的值必须出现在 FEATURE_NAMES 对应位置上。"""
    f = PaperCandidateFeatures(
        retrieval_rank=3.0,
        rule_rank=2.0,
        has_accepted_route=1.0,
        candidate_in_accepted_corpus=1.0,
    )
    vec = f.to_vector()
    assert vec[FEATURE_NAMES.index("retrieval_rank")] == 3.0
    assert vec[FEATURE_NAMES.index("rule_rank")] == 2.0
    assert vec[FEATURE_NAMES.index("has_accepted_route")] == 1.0
    assert vec[FEATURE_NAMES.index("candidate_in_accepted_corpus")] == 1.0


def test_paper_candidate_features_only_emits_evidence_for_explicit_v2_schema():
    """默认保持 19 维;显式选择 evidence schema 才输出 25 维。"""
    f = PaperCandidateFeatures(llm_scope_fit=0.9, llm_too_narrow_penalty=0.2)

    assert len(f.to_vector()) == 19

    evidence_vector = f.to_vector(FEATURE_NAMES_WITH_LLM_EVIDENCE)
    assert len(evidence_vector) == 25
    assert evidence_vector[FEATURE_NAMES_WITH_LLM_EVIDENCE.index("llm_scope_fit")] == 0.9
    assert (
        evidence_vector[
            FEATURE_NAMES_WITH_LLM_EVIDENCE.index("llm_too_narrow_penalty")
        ]
        == 0.2
    )


def test_paper_candidate_features_boolean_features_are_floats():
    """所有布尔/二元特征必须以 0.0/1.0 形式存储(per plan 4.1)。"""
    f = PaperCandidateFeatures(
        has_scope_route=1.0,
        has_typical_route=0.0,
        has_accepted_route=1.0,
        has_identity_anchor=0.0,
        same_gold_area=1.0,
        same_parsed_ccf_area=0.0,
        same_ccf_level=1.0,
        candidate_in_accepted_corpus=0.0,
    )
    assert f.has_scope_route == 1.0
    assert f.has_typical_route == 0.0
    assert isinstance(f.has_scope_route, float)


# ---- 4.1.b: build_features 边界与缺失值 ----


def _make_journal(journal_id: str = "tpds", ccf_rating: str | None = "A") -> Journal:
    return Journal(journal_id=journal_id, journal_name=journal_id.upper(), ccf_rating=ccf_rating)


def _make_paper_profile(paper_strength: float | None = 0.7) -> PaperProfile:
    return PaperProfile(
        title="T", abstract="A", paper_strength=paper_strength, ccf_research_area=[]
    )


def test_build_features_extracts_per_route_ranks_from_trace():
    """trace 中的 scope_bm25/accepted_bm25 rank 正确抽取;缺失 route 用 999 哨兵。"""
    trace_entry = {
        "routes": {
            "scope_bm25": {"rank": 3, "raw_score": 10.0, "normalized_score": 0.5, "weighted_score": 0.2},
            "accepted_bm25": {"rank": 5, "raw_score": 8.0, "normalized_score": 0.4, "weighted_score": 0.1},
        }
    }
    f = build_features(
        paper_profile=_make_paper_profile(),
        journal=_make_journal(),
        trace_entry=trace_entry,
        rule_rank=2,
        rule_score=0.5,
        candidate_in_accepted_corpus=True,
    )
    assert f.scope_bm25_rank == 3.0
    assert f.accepted_bm25_rank == 5.0
    assert f.scope_vector_rank == 999.0  # 缺失 route
    assert f.typical_bm25_rank == 999.0
    assert f.has_scope_route == 1.0
    assert f.has_accepted_route == 1.0
    assert f.has_typical_route == 0.0
    assert f.route_count == 2.0


def test_build_features_rule_rank_none_uses_sentinel():
    """rule_rank=None(rule_scorer 没把期刊列入 Top20)用哨兵 999。"""
    f = build_features(
        paper_profile=_make_paper_profile(),
        journal=_make_journal(),
        trace_entry={"routes": {}},
        rule_rank=None,
        rule_score=0.0,
        candidate_in_accepted_corpus=False,
    )
    assert f.rule_rank == 999.0


def test_build_features_sets_candidate_in_accepted_corpus_to_1_when_passed_true():
    """调用者传入 candidate_in_accepted_corpus=True → 1.0(per ADR 0001)。"""
    f = build_features(
        paper_profile=_make_paper_profile(),
        journal=_make_journal(),
        trace_entry={"routes": {}},
        rule_rank=None,
        rule_score=0.0,
        candidate_in_accepted_corpus=True,
    )
    assert f.candidate_in_accepted_corpus == 1.0


def test_build_features_sets_candidate_in_accepted_corpus_to_0_when_passed_false():
    """调用者传入 candidate_in_accepted_corpus=False → 0.0。"""
    f = build_features(
        paper_profile=_make_paper_profile(),
        journal=_make_journal(journal_id="uncovered_journal"),
        trace_entry={"routes": {}},
        rule_rank=None,
        rule_score=0.0,
        candidate_in_accepted_corpus=False,
    )
    assert f.candidate_in_accepted_corpus == 0.0


def test_build_features_ccf_rating_a_maps_to_3():
    """期刊 CCF 等级 A → journal_ccf_numeric=3.0。"""
    f = build_features(
        paper_profile=_make_paper_profile(),
        journal=_make_journal(ccf_rating="A"),
        trace_entry={"routes": {}},
        rule_rank=None,
        rule_score=0.0,
        candidate_in_accepted_corpus=False,
    )
    assert f.journal_ccf_numeric == 3.0


def test_build_features_ccf_rating_none_maps_to_0():
    """期刊 CCF 等级缺失 → journal_ccf_numeric=0.0。"""
    f = build_features(
        paper_profile=_make_paper_profile(),
        journal=_make_journal(ccf_rating=None),
        trace_entry={"routes": {}},
        rule_rank=None,
        rule_score=0.0,
        candidate_in_accepted_corpus=False,
    )
    assert f.journal_ccf_numeric == 0.0


def test_build_features_uses_neutral_defaults_when_llm_evidence_is_missing():
    """缺失 evidence 不应被当成负信号:fit=0.5, penalty=0.0。"""
    f = build_features(
        paper_profile=_make_paper_profile(),
        journal=_make_journal(),
        trace_entry={"routes": {}},
        rule_rank=None,
        rule_score=0.0,
        candidate_in_accepted_corpus=False,
    )

    assert f.llm_scope_fit == 0.5
    assert f.llm_method_fit == 0.5
    assert f.llm_application_fit == 0.5
    assert f.llm_journal_position_fit == 0.5
    assert f.llm_too_broad_penalty == 0.0
    assert f.llm_too_narrow_penalty == 0.0


def test_build_features_extracts_valid_llm_evidence_scores():
    """合法 evidence 分数应进入 25 维特征对象。"""
    evidence = {
        "scope_fit": 0.91,
        "method_fit": 0.82,
        "application_fit": 0.73,
        "journal_position_fit": 0.64,
        "too_broad_penalty": 0.15,
        "too_narrow_penalty": 0.26,
    }
    f = build_features(
        paper_profile=_make_paper_profile(),
        journal=_make_journal(),
        trace_entry={"routes": {}},
        rule_rank=None,
        rule_score=0.0,
        candidate_in_accepted_corpus=False,
        llm_evidence=evidence,
    )

    assert f.llm_scope_fit == 0.91
    assert f.llm_method_fit == 0.82
    assert f.llm_application_fit == 0.73
    assert f.llm_journal_position_fit == 0.64
    assert f.llm_too_broad_penalty == 0.15
    assert f.llm_too_narrow_penalty == 0.26


@pytest.mark.parametrize("invalid_value", [True, "0.9", -0.1, 1.1, None])
def test_build_features_replaces_invalid_llm_evidence_with_neutral_defaults(invalid_value):
    """非法 evidence 不截断也不惩罚,而是回到对应字段的中性值。"""
    f = build_features(
        paper_profile=_make_paper_profile(),
        journal=_make_journal(),
        trace_entry={"routes": {}},
        rule_rank=None,
        rule_score=0.0,
        candidate_in_accepted_corpus=False,
        llm_evidence={
            "scope_fit": invalid_value,
            "too_broad_penalty": invalid_value,
        },
    )

    assert f.llm_scope_fit == 0.5
    assert f.llm_too_broad_penalty == 0.0


def test_build_features_retrieval_rank_is_read_from_trace_top_level():
    """retrieval_rank 是 LTR 最重要的排序特征之一;必须从 trace 顶层读,不是哨兵。"""
    f = build_features(
        paper_profile=_make_paper_profile(),
        journal=_make_journal(),
        trace_entry={"routes": {}, "retrieval_rank": 5},
        rule_rank=None,
        rule_score=0.0,
        candidate_in_accepted_corpus=False,
    )
    assert f.retrieval_rank == 5.0
    # vector 里也能取到
    assert f.to_vector()[FEATURE_NAMES.index("retrieval_rank")] == 5.0


def test_build_features_retrieval_rank_missing_uses_sentinel_999():
    """trace 没有 retrieval_rank 字段时(防御性兜底),仍用哨兵 999(不能默认 0)。"""
    f = build_features(
        paper_profile=_make_paper_profile(),
        journal=_make_journal(),
        trace_entry={"routes": {}},  # 没有 retrieval_rank
        rule_rank=None,
        rule_score=0.0,
        candidate_in_accepted_corpus=False,
    )
    assert f.retrieval_rank == 999.0


def test_build_features_retrieval_rank_zero_or_negative_uses_sentinel():
    """retrieval_rank 非法值(0、负数、None)→ 哨兵 999,不能误读成"排名第一"。"""
    for bad in [0, -1, None, "abc"]:
        f = build_features(
            paper_profile=_make_paper_profile(),
            journal=_make_journal(),
            trace_entry={"routes": {}, "retrieval_rank": bad},
            rule_rank=None,
            rule_score=0.0,
            candidate_in_accepted_corpus=False,
        )
        assert f.retrieval_rank == 999.0, f"failed for bad value {bad!r}"


def test_build_features_detects_has_identity_anchor_route():
    """trace 中存在 identity_anchor → has_identity_anchor=1.0。"""
    f = build_features(
        paper_profile=_make_paper_profile(),
        journal=_make_journal(),
        trace_entry={"routes": {"identity_anchor": {"rank": 1, "raw_score": 1.0, "normalized_score": 1.0, "weighted_score": 0.05}}},
        rule_rank=None,
        rule_score=0.0,
        candidate_in_accepted_corpus=False,
    )
    assert f.has_identity_anchor == 1.0


def test_ccf_level_to_numeric_returns_zero_for_unknown():
    """未识别的 CCF 等级字符串 → 0.0。"""
    assert ccf_level_to_numeric("X") == 0.0
    assert ccf_level_to_numeric(None) == 0.0
    assert ccf_level_to_numeric("") == 0.0


def test_ccf_level_to_numeric_returns_correct_values():
    """A=3, B=2, C=1(per plan 4.1)。"""
    assert ccf_level_to_numeric("A") == 3.0
    assert ccf_level_to_numeric("B") == 2.0
    assert ccf_level_to_numeric("C") == 1.0
    assert ccf_level_to_numeric("a") == 3.0  # 大小写不敏感


# ---- 4.1.d: attach_features_to_trace ----


def _setup_store_with_journals(journal_specs: list[tuple[str, str | None]]) -> JournalStore:
    """journal_specs = [(jid, ccf_rating), ...]"""
    store = JournalStore()
    for jid, ccf in journal_specs:
        store.add_journal(Journal(journal_id=jid, journal_name=jid.upper(), ccf_rating=ccf))
    return store


def _write_corpus_journal(dir_path: Path, jid: str) -> None:
    payload = {"journal_id": jid, "journal_name": jid, "papers": [{"title": "x", "abstract": "y" * 50}]}
    (dir_path / f"{jid}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_attach_features_to_trace_adds_features_to_each_journal():
    """对 trace 中每本期刊,都应注入 'features' 与 'feature_names'。"""
    store = _setup_store_with_journals([("a", "A"), ("b", "B")])
    trace = {
        "a": {"total_score": 0.5, "routes": {"scope_bm25": {"rank": 2, "raw_score": 1.0, "normalized_score": 0.5, "weighted_score": 0.2}}},
        "b": {"total_score": 0.3, "routes": {"typical_bm25": {"rank": 3, "raw_score": 0.8, "normalized_score": 0.4, "weighted_score": 0.1}}},
    }
    paper = _make_paper_profile()

    attach_features_to_trace(
        trace=trace,
        paper_profile=paper,
        journal_store=store,
        rule_ranks={"a": 1, "b": 2},
        rule_scores={"a": 0.7, "b": 0.4},
        accepted_paper_store=None,  # 没有任何期刊在 corpus 中
    )

    assert "features" in trace["a"]
    assert "feature_names" in trace["a"]
    assert len(trace["a"]["features"]) == 19
    assert trace["a"]["feature_names"] == FEATURE_NAMES
    assert len(trace["b"]["features"]) == 19
    # a 在 scope_bm25 rank=2
    assert trace["a"]["features"][FEATURE_NAMES.index("scope_bm25_rank")] == 2.0
    assert trace["a"]["features"][FEATURE_NAMES.index("rule_rank")] == 1.0
    # b 在 typical_bm25 rank=3, accepted_bm25 缺失 → 哨兵
    assert trace["b"]["features"][FEATURE_NAMES.index("typical_bm25_rank")] == 3.0
    assert trace["b"]["features"][FEATURE_NAMES.index("accepted_bm25_rank")] == 999.0


def test_attach_features_to_trace_marks_candidate_in_corpus_correctly(tmp_path: Path):
    """candidate_in_accepted_corpus 必须从 accepted_paper_store 读取,不能写死。"""
    _write_corpus_journal(tmp_path, "a")
    # b 不在 corpus
    store = _setup_store_with_journals([("a", "A"), ("b", "B")])
    store_path = tmp_path
    accepted_store = AcceptedPaperStore(str(store_path))
    accepted_store.load()

    trace = {
        "a": {"total_score": 0.5, "routes": {}},
        "b": {"total_score": 0.4, "routes": {}},
    }
    paper = _make_paper_profile()

    attach_features_to_trace(
        trace=trace,
        paper_profile=paper,
        journal_store=store,
        rule_ranks=None,
        rule_scores=None,
        accepted_paper_store=accepted_store,
    )

    corpus_feat_idx = FEATURE_NAMES.index("candidate_in_accepted_corpus")
    assert trace["a"]["features"][corpus_feat_idx] == 1.0
    assert trace["b"]["features"][corpus_feat_idx] == 0.0


def test_attach_features_to_trace_handles_journal_missing_from_store():
    """trace 中出现 store 找不到的 journal_id(理论不会发生,但要安全跳过)。"""
    store = _setup_store_with_journals([("a", "A")])
    trace = {
        "a": {"total_score": 0.5, "routes": {}},
        "ghost_journal": {"total_score": 0.1, "routes": {}},
    }
    paper = _make_paper_profile()

    attach_features_to_trace(
        trace=trace,
        paper_profile=paper,
        journal_store=store,
        rule_ranks=None,
        rule_scores=None,
        accepted_paper_store=None,
    )

    # a 正常注入 features
    assert "features" in trace["a"]
    # ghost_journal 应被静默跳过(不抛异常,不污染 features)
    assert "features" not in trace["ghost_journal"]


def test_attach_features_to_trace_handles_missing_rule_ranks_gracefully():
    """rule_ranks=None 或某 jid 缺失 → rule_rank 用哨兵 999。"""
    store = _setup_store_with_journals([("a", "A")])
    trace = {"a": {"total_score": 0.5, "routes": {}}}
    paper = _make_paper_profile()

    attach_features_to_trace(
        trace=trace,
        paper_profile=paper,
        journal_store=store,
        rule_ranks=None,  # 完全不传
        rule_scores=None,
        accepted_paper_store=None,
    )

    rule_idx = FEATURE_NAMES.index("rule_rank")
    assert trace["a"]["features"][rule_idx] == 999.0


def test_attach_features_to_trace_can_emit_explicit_25_dim_evidence_schema():
    """显式选择 v2 schema 时,按 journal_id 注入六维 evidence。"""
    store = _setup_store_with_journals([("a", "A"), ("b", "B")])
    trace = {
        "a": {"total_score": 0.5, "routes": {}},
        "b": {"total_score": 0.4, "routes": {}},
    }

    attach_features_to_trace(
        trace=trace,
        paper_profile=_make_paper_profile(),
        journal_store=store,
        rule_ranks=None,
        rule_scores=None,
        accepted_paper_store=None,
        llm_evidence_by_journal={
            "a": {
                "scope_fit": 0.9,
                "method_fit": 0.8,
                "application_fit": 0.7,
                "journal_position_fit": 0.6,
                "too_broad_penalty": 0.2,
                "too_narrow_penalty": 0.1,
            }
        },
        feature_names=FEATURE_NAMES_WITH_LLM_EVIDENCE,
    )

    assert trace["a"]["feature_names"] == FEATURE_NAMES_WITH_LLM_EVIDENCE
    assert len(trace["a"]["features"]) == 25
    assert (
        trace["a"]["features"][
            FEATURE_NAMES_WITH_LLM_EVIDENCE.index("llm_scope_fit")
        ]
        == 0.9
    )
    # b 没有 evidence,使用中性值。
    assert (
        trace["b"]["features"][
            FEATURE_NAMES_WITH_LLM_EVIDENCE.index("llm_scope_fit")
        ]
        == 0.5
    )
    assert (
        trace["b"]["features"][
            FEATURE_NAMES_WITH_LLM_EVIDENCE.index("llm_too_broad_penalty")
        ]
        == 0.0
    )


def test_attach_features_to_trace_default_schema_remains_19_dim_when_evidence_exists():
    """即使传入 evidence,未显式选择 v2 schema 时仍保持旧模型的 19 维输入。"""
    store = _setup_store_with_journals([("a", "A")])
    trace = {"a": {"total_score": 0.5, "routes": {}}}

    attach_features_to_trace(
        trace=trace,
        paper_profile=_make_paper_profile(),
        journal_store=store,
        rule_ranks=None,
        rule_scores=None,
        accepted_paper_store=None,
        llm_evidence_by_journal={"a": {"scope_fit": 0.9}},
    )

    assert trace["a"]["feature_names"] == FEATURE_NAMES
    assert len(trace["a"]["features"]) == 19


# ---------------------------------------------------------------------------
# 阶段 6.5 (P2-mini): 27-dim schema — journal_tier_weight + area_exclusivity
# 2026-06-26: 从 28 维降到 27 维 (paper_strength 已删除)
# ---------------------------------------------------------------------------


def test_feature_names_with_tier_and_exclusivity_is_27_dim():
    """27 维 schema = 19 base + 6 LLM evidence + 2 tier/area features."""
    assert len(FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY) == 27
    # 前 19 项 == FEATURE_NAMES
    assert FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY[:19] == FEATURE_NAMES
    # 20-25 == LLM_EVIDENCE_FEATURE_NAMES
    assert FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY[19:25] == LLM_EVIDENCE_FEATURE_NAMES
    # 26-27 == ["journal_tier_weight", "area_exclusivity"]
    assert FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY[25:] == [
        "journal_tier_weight",
        "area_exclusivity",
    ]
    assert TIER_EXCLUSIVITY_FEATURE_NAMES == ["journal_tier_weight", "area_exclusivity"]


def test_tier_weight_value_returns_expected_discrete_values():
    """A=0.7, B=1.0, C=1.5; unknown/missing → 1.0 (中性)。"""
    assert _tier_weight_value("A") == 0.7
    assert _tier_weight_value("B") == 1.0
    assert _tier_weight_value("C") == 1.5
    assert _tier_weight_value(None) == 1.0
    assert _tier_weight_value("X") == 1.0
    # 大小写不敏感
    assert _tier_weight_value("a") == 0.7
    assert _tier_weight_value("c") == 1.5


def test_area_exclusivity_value_zero_when_no_anchor():
    """paper_anchor_area=None → 0.0(没有锚就没法算互斥度)。"""
    assert _area_exclusivity_value(["AI"], None, 3) == 0.0
    assert _area_exclusivity_value(["AI", "DB"], None, 5) == 0.0


def test_area_exclusivity_value_zero_when_no_match():
    """paper_anchor_area 不在 candidate.subject_tags → 0.0。"""
    assert _area_exclusivity_value(["DB"], "AI", 3) == 0.0
    assert _area_exclusivity_value([], "AI", 3) == 0.0
    assert _area_exclusivity_value(None, "AI", 3) == 0.0


def test_area_exclusivity_value_one_over_n_when_match():
    """match + n_matching=N → 1/N。"""
    assert _area_exclusivity_value(["AI"], "AI", 3) == 1.0 / 3
    assert _area_exclusivity_value(["AI", "DB"], "AI", 1) == 1.0
    # 防御:n_matching=None 或 <=0 → 1.0 (单点最优)
    assert _area_exclusivity_value(["AI"], "AI", None) == 1.0
    assert _area_exclusivity_value(["AI"], "AI", 0) == 1.0
    assert _area_exclusivity_value(["AI"], "AI", -1) == 1.0


def test_build_features_populates_tier_weight_from_journal_ccf_rating():
    """build_features 内置 _tier_weight_value(journal.ccf_rating) → tier_weight 字段。"""
    journal_c = Journal(journal_id="j_c", journal_name="C", ccf_rating="C", subject_tags=["AI"])
    paper = PaperProfile(title="t", research_area=["AI"])
    feats = build_features(
        paper_profile=paper,
        journal=journal_c,
        trace_entry={"retrieval_rank": 1, "routes": {}},
        rule_rank=1,
        rule_score=0.9,
        candidate_in_accepted_corpus=False,
    )
    assert feats.journal_tier_weight == 1.5  # C → 1.5

    journal_a = Journal(journal_id="j_a", journal_name="A", ccf_rating="A", subject_tags=["AI"])
    feats_a = build_features(
        paper_profile=paper,
        journal=journal_a,
        trace_entry={"retrieval_rank": 1, "routes": {}},
        rule_rank=1,
        rule_score=0.9,
        candidate_in_accepted_corpus=False,
    )
    assert feats_a.journal_tier_weight == 0.7  # A → 0.7


def test_build_features_populates_area_exclusivity_from_paper_anchor():
    """build_features 接受 paper_anchor_area + n_matching_in_pool,产出 area_exclusivity。"""
    journal = Journal(journal_id="j", journal_name="J", ccf_rating="B", subject_tags=["AI"])
    paper = PaperProfile(title="t", research_area=["AI"])
    feats = build_features(
        paper_profile=paper,
        journal=journal,
        trace_entry={"retrieval_rank": 1, "routes": {}},
        rule_rank=1,
        rule_score=0.9,
        candidate_in_accepted_corpus=False,
        paper_anchor_area="AI",
        n_matching_in_pool=4,
    )
    assert feats.area_exclusivity == 0.25  # 1/4

    # 不匹配:journal 没有 "AI" tag
    journal_no_match = Journal(journal_id="j2", journal_name="J2", ccf_rating="B", subject_tags=["DB"])
    feats2 = build_features(
        paper_profile=paper,
        journal=journal_no_match,
        trace_entry={"retrieval_rank": 1, "routes": {}},
        rule_rank=1,
        rule_score=0.9,
        candidate_in_accepted_corpus=False,
        paper_anchor_area="AI",
        n_matching_in_pool=4,
    )
    assert feats2.area_exclusivity == 0.0  # 不匹配


def test_build_features_default_tier_weight_and_area_exclusivity():
    """不传 paper_anchor_area + n_matching → area_exclusivity=0.0, tier_weight=1.0 (中性)。"""
    journal = Journal(journal_id="j", journal_name="J")  # ccf_rating=None
    paper = PaperProfile(title="t")
    feats = build_features(
        paper_profile=paper,
        journal=journal,
        trace_entry={"retrieval_rank": 1, "routes": {}},
        rule_rank=1,
        rule_score=0.9,
        candidate_in_accepted_corpus=False,
    )
    assert feats.journal_tier_weight == 1.0  # unknown → 中性
    assert feats.area_exclusivity == 0.0  # 无 anchor


def test_paper_candidate_features_to_vector_27_dim():
    """to_vector(FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY) → 27 长。"""
    feats = PaperCandidateFeatures(
        retrieval_rank=1, rule_rank=1, rule_score=0.5,
        journal_ccf_numeric=2,
        journal_tier_weight=1.0, area_exclusivity=0.5,
    )
    vec = feats.to_vector(FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY)
    assert len(vec) == 27
    assert vec[-2] == 1.0  # tier_weight
    assert vec[-1] == 0.5  # area_exclusivity


def test_attach_features_to_trace_27_dim_schema_writes_27_features():
    """显式 FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY + paper_anchor_area
    → trace[jid]['features'] 长 27。"""
    from src.ranker.feature_builder import attach_features_to_trace

    journal = Journal(
        journal_id="ji",
        journal_name="Journal I",
        ccf_rating="C",
        subject_tags=["AI"],
    )
    journal_store = MagicMock(spec=JournalStore)
    journal_store.get_journal.return_value = journal
    paper = PaperProfile(title="t", research_area=["AI"])
    trace = {"ji": {"retrieval_rank": 1, "routes": {}}}

    attach_features_to_trace(
        trace, paper, journal_store,
        rule_ranks={"ji": 1},
        rule_scores={"ji": 0.5},
        accepted_paper_store=None,
        feature_names=FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY,
        paper_anchor_area="AI",
        n_matching_in_pool=2,
    )

    assert len(trace["ji"]["features"]) == 27
    assert trace["ji"]["features"][-2] == 1.5  # tier C
    assert trace["ji"]["features"][-1] == 0.5  # 1/2


def test_attach_features_19_dim_path_unaffected_by_27_dim_constants():
    """默认 19 维 schema 仍 19 长 (不破旧测试)。"""
    journal = Journal(journal_id="j", journal_name="J", ccf_rating="A", subject_tags=["AI"])
    journal_store = MagicMock(spec=JournalStore)
    journal_store.get_journal.return_value = journal
    paper = PaperProfile(title="t", research_area=["AI"])
    trace = {"j": {"retrieval_rank": 1, "routes": {}}}

    attach_features_to_trace(
        trace, paper, journal_store,
        rule_ranks={"j": 1}, rule_scores={"j": 0.5},
        accepted_paper_store=None,
        feature_names=None,  # 默认 19 维
    )

    assert len(trace["j"]["features"]) == 19
    assert trace["j"]["feature_names"] == FEATURE_NAMES


# ---------------------------------------------------------------------------
# 2026-06-25: 修复 same_* 三个 dead 特征 (from 0.0 占位 → 真实计算)
# ---------------------------------------------------------------------------


def test_build_features_same_gold_area_set_when_gold_and_paper_areas_overlap():
    """paper.research_area ∩ gold.subject_tags ≠ ∅ → same_gold_area=1.0。"""
    gold = Journal(journal_id="gold", journal_name="G", subject_tags=["人工智能"])
    paper = PaperProfile(
        title="T", research_area=["人工智能"], ccf_research_area=[]
    )
    cand = Journal(journal_id="c", journal_name="C", subject_tags=["计算机网络"])
    f = build_features(
        paper_profile=paper,
        journal=cand,
        trace_entry={"routes": {}},
        rule_rank=1,
        rule_score=0.5,
        candidate_in_accepted_corpus=False,
        gold_journal=gold,
    )
    assert f.same_gold_area == 1.0


def test_build_features_same_gold_area_zero_on_mismatch():
    """paper.research_area ∩ gold.subject_tags == ∅ → 0.0。"""
    gold = Journal(journal_id="gold", journal_name="G", subject_tags=["人工智能"])
    paper = PaperProfile(
        title="T", research_area=["计算机网络"], ccf_research_area=[]
    )
    cand = Journal(journal_id="c", journal_name="C", subject_tags=["软件工程"])
    f = build_features(
        paper_profile=paper,
        journal=cand,
        trace_entry={"routes": {}},
        rule_rank=1,
        rule_score=0.5,
        candidate_in_accepted_corpus=False,
        gold_journal=gold,
    )
    assert f.same_gold_area == 0.0


def test_build_features_same_gold_area_proxy_when_no_gold():
    """gold_journal=None → proxy 用 candidate journal 自身 subject_tags。
    paper.research_area=["AI"], candidate.subject_tags=["AI"] → 1.0。
    """
    paper = PaperProfile(
        title="T", research_area=["人工智能"], ccf_research_area=[]
    )
    cand = Journal(journal_id="c", journal_name="C", subject_tags=["人工智能"])
    f = build_features(
        paper_profile=paper,
        journal=cand,
        trace_entry={"routes": {}},
        rule_rank=1,
        rule_score=0.5,
        candidate_in_accepted_corpus=False,
        gold_journal=None,  # inference proxy
    )
    assert f.same_gold_area == 1.0


def test_build_features_same_parsed_ccf_area_set_when_overlap():
    """paper.ccf_research_area ∩ gold.subject_tags ≠ ∅ → same_parsed_ccf_area=1.0。"""
    gold = Journal(journal_id="gold", journal_name="G", subject_tags=["数据库/数据挖掘/内容检索"])
    paper = PaperProfile(
        title="T",
        research_area=["人工智能"],
        ccf_research_area=["数据库/数据挖掘/内容检索"],
    )
    cand = Journal(journal_id="c", journal_name="C", subject_tags=["计算机网络"])
    f = build_features(
        paper_profile=paper,
        journal=cand,
        trace_entry={"routes": {}},
        rule_rank=1,
        rule_score=0.5,
        candidate_in_accepted_corpus=False,
        gold_journal=gold,
    )
    assert f.same_parsed_ccf_area == 1.0
    assert f.same_gold_area == 0.0  # research_area 和 gold tags 不重叠


def test_build_features_same_ccf_level_set_when_levels_match():
    """paper_ccf_target_level="A" + journal.ccf_rating="A" → same_ccf_level=1.0。"""
    paper = PaperProfile(title="T", ccf_target_level="A")
    cand = Journal(journal_id="c", journal_name="C", ccf_rating="A")
    f = build_features(
        paper_profile=paper,
        journal=cand,
        trace_entry={"routes": {}},
        rule_rank=1,
        rule_score=0.5,
        candidate_in_accepted_corpus=False,
        paper_ccf_target_level="A",
    )
    assert f.same_ccf_level == 1.0


def test_build_features_same_ccf_level_case_insensitive():
    """大小写不敏感:"a" vs "A" → 1.0。"""
    paper = PaperProfile(title="T", ccf_target_level="a")
    cand = Journal(journal_id="c", journal_name="C", ccf_rating="A")
    f = build_features(
        paper_profile=paper,
        journal=cand,
        trace_entry={"routes": {}},
        rule_rank=1,
        rule_score=0.5,
        candidate_in_accepted_corpus=False,
        paper_ccf_target_level="a",
    )
    assert f.same_ccf_level == 1.0


def test_build_features_same_ccf_level_zero_when_paper_target_none():
    """paper_ccf_target_level=None → 0.0 (inference 不可知时退化)。"""
    paper = PaperProfile(title="T")
    cand = Journal(journal_id="c", journal_name="C", ccf_rating="A")
    f = build_features(
        paper_profile=paper,
        journal=cand,
        trace_entry={"routes": {}},
        rule_rank=1,
        rule_score=0.5,
        candidate_in_accepted_corpus=False,
        paper_ccf_target_level=None,
    )
    assert f.same_ccf_level == 0.0


def test_attach_features_to_trace_passes_gold_journal_through():
    """attach_features_to_trace 把 gold_journal 透传给 build_features。"""
    gold = Journal(journal_id="gold", journal_name="GOLD", subject_tags=["人工智能"])
    cand = Journal(journal_id="c", journal_name="C", subject_tags=["计算机网络"])
    journal_store = MagicMock(spec=JournalStore)
    journal_store.get_journal.return_value = cand

    paper = PaperProfile(
        title="T", research_area=["人工智能"], ccf_research_area=[]
    )
    trace = {"c": {"retrieval_rank": 1, "routes": {}}}

    attach_features_to_trace(
        trace, paper, journal_store,
        rule_ranks={"c": 1}, rule_scores={"c": 0.5},
        accepted_paper_store=None,
        feature_names=FEATURE_NAMES,
        gold_journal=gold,
    )

    same_gold_idx = FEATURE_NAMES.index("same_gold_area")
    assert trace["c"]["features"][same_gold_idx] == 1.0


# ---------------------------------------------------------------------------
# 2026-06-26: Accepted Corpus LTR — paper_strength 移除 + schema 维度调整
# ---------------------------------------------------------------------------


def test_feature_names_excludes_paper_strength():
    """2026-06-26: paper_strength removed from schema (was 20-dim, now 19-dim)."""
    import dataclasses

    assert "paper_strength" not in FEATURE_NAMES


def test_feature_names_19_dim():
    """2026-06-26: base schema is 19-dim (was 20)."""
    assert len(FEATURE_NAMES) == 19
    assert len(set(FEATURE_NAMES)) == 19  # 无重复


def test_feature_names_with_llm_evidence_25_dim():
    """2026-06-26: 19 base + 6 evidence = 25-dim (was 26)."""
    assert len(FEATURE_NAMES_WITH_LLM_EVIDENCE) == 25
    assert "paper_strength" not in FEATURE_NAMES_WITH_LLM_EVIDENCE


def test_feature_names_with_tier_and_exclusivity_27_dim():
    """2026-06-26: 25 + 2 tier/area = 27-dim (was 28)."""
    assert len(FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY) == 27
    assert "paper_strength" not in FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY


def test_paper_candidate_features_no_paper_strength():
    """2026-06-26: PaperCandidateFeatures dataclass no longer has paper_strength field."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(PaperCandidateFeatures)}
    assert "paper_strength" not in field_names
