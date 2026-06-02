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
    PaperCandidateFeatures,
    attach_features_to_trace,
    build_features,
    ccf_level_to_numeric,
)


def test_feature_names_is_a_locked_list_of_20_strings():
    """FEATURE_NAMES 是锁定 schema:20 个字符串特征名(per plan 4.1 + ADR 0001)。"""
    assert isinstance(FEATURE_NAMES, list)
    assert len(FEATURE_NAMES) == 20
    assert all(isinstance(n, str) for n in FEATURE_NAMES)
    # 防止重复
    assert len(set(FEATURE_NAMES)) == 20


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


def test_build_features_paper_strength_none_defaults_to_0():
    """paper_profile.paper_strength=None → 0.0(无信号比哨兵 999 更合理,这是连续值)。"""
    f = build_features(
        paper_profile=_make_paper_profile(paper_strength=None),
        journal=_make_journal(),
        trace_entry={"routes": {}},
        rule_rank=None,
        rule_score=0.0,
        candidate_in_accepted_corpus=False,
    )
    assert f.paper_strength == 0.0


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
    assert len(trace["a"]["features"]) == 20
    assert trace["a"]["feature_names"] == FEATURE_NAMES
    assert len(trace["b"]["features"]) == 20
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
