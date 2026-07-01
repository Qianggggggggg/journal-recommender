"""Tests for src/ranker/ltr_adapter.py (Task 5.3)."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ranker.ltr_adapter import LTRAdapter  # noqa: E402
from src.papers.paper_model import PaperProfile  # noqa: E402

# 测试用的最小有效 journal store 桩
# 不依赖真实 JournalStore/AcceptedPaperStore,直接 mock 调用接口
class _StubJournal:
    def __init__(self, jid: str) -> None:
        self.journal_id = jid
        self.journal_name = jid
        self.ccf_rating = "B"
        self.ccf_research_area: List[str] = []
        self.research_area: List[str] = []
        self.subject_tags: List[str] = []
        self.scope_text = ""
        self.keywords: List[str] = []


class _StubJournalStore:
    def get_journal(self, jid: str) -> Optional[_StubJournal]:
        if jid.startswith("J"):
            return _StubJournal(jid)
        return None


def _make_paper_profile() -> PaperProfile:
    """构造一个最小可用的 PaperProfile(paper_strength=None 让 build_features 取 0.0)。"""
    return PaperProfile(title="test", paper_strength=None)


def _make_stub_candidates(n: int) -> List[Tuple[Any, float, List[str]]]:
    """构造 N 本桩候选期刊,rule_score 顺序为 n, n-1, ..., 1。"""
    candidates = []
    for i in range(n):
        jid = f"J{i+1}"
        candidates.append((_StubJournal(jid), float(n - i), [f"reason-{i}"]))
    return candidates


def _make_trace_for_candidates(candidates: List[Tuple[Any, float, List[str]]]) -> Dict[str, dict]:
    """构造一个最简单的 trace,retrieval_rank 按候选顺序。"""
    trace: Dict[str, dict] = {}
    for i, (j, _, _) in enumerate(candidates):
        trace[j.journal_id] = {
            "retrieval_rank": i + 1,
            "routes": {},
            "primary_routes": [],
        }
    return trace


# ---------------------------------------------------------------------------
# 初始化 / 失败降级
# ---------------------------------------------------------------------------


def test_disabled_when_enabled_false_in_config():
    """config.enabled=False → LTRAdapter 永远禁用。"""
    adapter = LTRAdapter(config={"enabled": False, "model_path": "/tmp/whatever"}, journal_store=_StubJournalStore())
    assert adapter.enabled is False
    assert adapter.disable_reason is not None
    # compute_scores 立即返回原 list
    candidates = _make_stub_candidates(3)
    out, diag = adapter.compute_scores(
        paper_profile=None,
        llm_candidates=candidates,
        retrieval_trace=_make_trace_for_candidates(candidates),
        rule_ranks={},
        rule_scores={},
    )
    assert out == candidates
    assert diag["status"] == "fallback_disabled"


def test_disabled_when_model_path_missing(tmp_path: Path):
    """model_path 指向不存在的文件 → 禁用,disable_reason 含 'not found'。"""
    cfg = {"enabled": True, "model_path": str(tmp_path / "no_such_model.json")}
    adapter = LTRAdapter(config=cfg, journal_store=_StubJournalStore())
    assert adapter.enabled is False
    assert adapter.disable_reason is not None
    assert "not found" in adapter.disable_reason.lower()


def test_disabled_when_model_corrupt_json(tmp_path: Path):
    """model 存在但 JSON 无效 → 禁用,disable_reason 含 'load' 或 'json'。"""
    bad = tmp_path / "bad.json"
    bad.write_text("not a valid json {{{", encoding="utf-8")
    cfg = {"enabled": True, "model_path": str(bad)}
    adapter = LTRAdapter(config=cfg, journal_store=_StubJournalStore())
    assert adapter.enabled is False
    assert adapter.disable_reason is not None
    assert any(k in adapter.disable_reason.lower() for k in ("load", "json", "schema", "key"))


def test_disabled_when_model_missing_required_fields(tmp_path: Path):
    """model JSON 缺 coef/feature_dim 等关键字段 → 禁用。"""
    bad = tmp_path / "incomplete.json"
    bad.write_text(json.dumps({"schema_version": 1, "coef": []}), encoding="utf-8")
    cfg = {"enabled": True, "model_path": str(bad)}
    adapter = LTRAdapter(config=cfg, journal_store=_StubJournalStore())
    assert adapter.enabled is False
    assert adapter.disable_reason is not None


def test_disabled_when_model_unconverged(tmp_path: Path):
    """model.convergence_info.converged=False → 禁用,disable_reason 含 'converge'。

    2026-06-26: feature_dim 从 20 改到 16 (paper_strength + 3 noise 删)。
    """
    unconverged = {
        "schema_version": 1,
        "model_type": "logistic_regression",
        "backend": "sklearn",
        "feature_dim": 16,
        "coef": [0.0] * 16,
        "intercept": 0.0,
        "use_standardization": False,
        "scaler_mean": None,
        "scaler_scale": None,
        "convergence_info": {
            "converged": False,
            "n_iter": 5000,
            "max_iter": 5000,
            "warning_message": "TOTAL NO. of ITERATIONS REACHED LIMIT",
        },
        "seed": 42,
    }
    p = tmp_path / "unconv.json"
    p.write_text(json.dumps(unconverged), encoding="utf-8")
    cfg = {"enabled": True, "model_path": str(p)}
    adapter = LTRAdapter(config=cfg, journal_store=_StubJournalStore())
    assert adapter.enabled is False
    assert adapter.disable_reason is not None
    assert "converge" in adapter.disable_reason.lower()


def test_enabled_with_real_model_file(tmp_path: Path):
    """用 16-dim stub 模型文件能成功初始化 (2026-06-26: 16-dim, was 19)."""
    model_path = tmp_path / "stub_ltr.json"
    model_path.write_text(json.dumps({
        "schema_version": 1, "model_type": "logistic_regression", "backend": "sklearn",
        "feature_dim": 16, "feature_names": ["f"] * 16,
        "coef": [0.0] * 16, "intercept": 0.0,
        "use_standardization": False, "scaler_mean": None, "scaler_scale": None,
        "convergence_info": {"converged": True, "n_iter": 1, "max_iter": 100, "warning_message": None},
        "seed": 42, "max_iter": 100,
    }))
    cfg = {"enabled": True, "model_path": str(model_path)}
    adapter = LTRAdapter(config=cfg, journal_store=_StubJournalStore())
    assert adapter.enabled is True
    assert adapter.feature_dim == 16
    assert adapter.disable_reason is None


# ---------------------------------------------------------------------------
# compute_scores 行为契约
# ---------------------------------------------------------------------------


def _write_minimal_converged_model(tmp_path: Path, feature_dim: int = 16) -> Path:
    """写一个最小可用的 converged 模型文件(系数全 0,scaler None,无 journal 依赖)。"""
    payload = {
        "schema_version": 1,
        "model_type": "logistic_regression",
        "backend": "sklearn",
        "feature_dim": feature_dim,
        "coef": [0.0] * feature_dim,
        "intercept": 0.0,
        "use_standardization": False,
        "scaler_mean": None,
        "scaler_scale": None,
        "convergence_info": {
            "converged": True,
            "n_iter": 1,
            "max_iter": 100,
            "warning_message": None,
        },
        "seed": 42,
    }
    p = tmp_path / "stub_model.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_compute_scores_returns_empty_diag_on_empty_candidates(tmp_path: Path):
    """llm_candidates=[] → 返回 [],diag 字段为空 dict。"""
    p = _write_minimal_converged_model(tmp_path)
    adapter = LTRAdapter(
        config={"enabled": True, "model_path": str(p)},
        journal_store=_StubJournalStore(),
    )
    assert adapter.enabled
    out, diag = adapter.compute_scores(
        paper_profile=None,
        llm_candidates=[],
        retrieval_trace={},
        rule_ranks={},
        rule_scores={},
    )
    assert out == []
    assert diag["learned_score"] == {}
    assert diag["learned_rank"] == {}
    assert diag["status"] == "ok"


def test_compute_scores_reranks_and_provides_diagnostics(tmp_path: Path):
    """3 本期刊 + valid trace + valid store → reranked 长度==3,learned_score/rank 1..N,status='ok'。

    注:模型系数全 0,所以所有候选的 LTR 分数相等,rerank 应该是 stable sort
    (按输入顺序 tiebreak),输出顺序应当和输入一致。
    """
    p = _write_minimal_converged_model(tmp_path)
    adapter = LTRAdapter(
        config={"enabled": True, "model_path": str(p)},
        journal_store=_StubJournalStore(),
    )
    candidates = _make_stub_candidates(3)
    trace = _make_trace_for_candidates(candidates)

    out, diag = adapter.compute_scores(
        paper_profile=_make_paper_profile(),
        llm_candidates=candidates,
        retrieval_trace=trace,
        rule_ranks=None,
        rule_scores=None,
    )

    assert len(out) == 3
    assert diag["status"] == "ok", f"expected ok, got {diag}"
    # learned_score 是 dict, key 是 jid
    assert isinstance(diag["learned_score"], dict)
    assert isinstance(diag["learned_rank"], dict)
    # learned_rank 应当是 1..N
    ranks = sorted(diag["learned_rank"].values())
    assert ranks == [1, 2, 3]


def test_compute_scores_falls_back_on_feature_dim_mismatch(tmp_path: Path):
    """特征缺失 / 维度不对 → 降级回原序,status 含 'feature_dim'。

    实现:用空 journal store(对所有 jid 返回 None),让 attach_features_to_trace
    在每本期刊的 entry 上跳过(features 不会被注入)。adapter 拿到 None → 触发 fallback。
    """
    p = _write_minimal_converged_model(tmp_path, feature_dim=16)
    empty_store = _EmptyJournalStore()  # get_journal 永远返回 None
    adapter = LTRAdapter(
        config={"enabled": True, "model_path": str(p)},
        journal_store=empty_store,
    )
    candidates = _make_stub_candidates(2)
    trace = _make_trace_for_candidates(candidates)

    out, diag = adapter.compute_scores(
        paper_profile=_make_paper_profile(),
        llm_candidates=candidates,
        retrieval_trace=trace,
        rule_ranks=None,
        rule_scores=None,
    )
    # 应当 fallback 到原序
    assert out == candidates
    assert "feature_dim" in diag["status"]


class _EmptyJournalStore:
    """永远返回 None 的 journal store,用来模拟 feature build 失败的场景。"""

    def get_journal(self, jid: str) -> None:
        return None


def test_compute_scores_does_not_mutate_caller_trace(tmp_path: Path):
    """compute_scores 不得修改 caller 的 trace(防止污染 evaluation diagnostics)。"""
    p = _write_minimal_converged_model(tmp_path)
    adapter = LTRAdapter(
        config={"enabled": True, "model_path": str(p)},
        journal_store=_StubJournalStore(),
    )
    candidates = _make_stub_candidates(2)
    trace = _make_trace_for_candidates(candidates)
    # 保存原始 trace 快照
    original_trace = {jid: dict(entry) for jid, entry in trace.items()}

    out, diag = adapter.compute_scores(
        paper_profile=_make_paper_profile(),
        llm_candidates=candidates,
        retrieval_trace=trace,
        rule_ranks=None,
        rule_scores=None,
    )

    # caller 的 trace 不应被加 features/feature_names
    for jid, entry in trace.items():
        assert "features" not in entry, f"caller trace[{jid}] was mutated to include features"
        assert "feature_names" not in entry
    # 应当与原 snapshot 一致
    assert trace == original_trace


# ---------------------------------------------------------------------------
# 阶段 6.5 (P2-mini): 27-dim schema support in LTRAdapter
# 2026-06-26: tier+exclusivity schema 从 28 维降到 27 维 (paper_strength 移除)。
# ---------------------------------------------------------------------------


def test_feature_schema_lookup_table_includes_23():
    """LTRAdapter.compute_scores 必须能识别 23-dim model 并选对应 schema (2026-06-26: was 27)。"""
    from src.ranker.feature_builder import (
        FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY,
    )
    from src.ranker.ltr_adapter import LTRAdapter

    # 反射地验证 compute_scores 源码包含正确的 lookup table
    import inspect
    src = inspect.getsource(LTRAdapter.compute_scores)
    assert "FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY" in src, (
        "LTRAdapter must reference the 23-dim schema constant"
    )
    assert "_FEATURE_SCHEMA_BY_DIM" in src, (
        "LTRAdapter must use a lookup table (not if-else) for schema selection"
    )


def test_compute_scores_unknown_feature_dim_returns_fallback(tmp_path: Path):
    """当 model 的 feature_dim 不在 lookup table 中(如 20-dim)→ fallback 不崩。"""
    from src.ranker.ltr_adapter import LTRAdapter
    from src.papers.paper_model import PaperProfile
    from src.journals.journal_model import Journal
    from src.journals.journal_store import JournalStore

    # 写一个 20-dim stub model (not in lookup table: 16/22/23)
    model_path = tmp_path / "stub_20dim.json"
    model_path.write_text(json.dumps({
        "schema_version": 1,
        "model_type": "logistic_regression",
        "backend": "sklearn",
        "feature_names": ["f"] * 20,
        "feature_dim": 20,
        "coef": [0.0] * 20,
        "intercept": 0.0,
        "use_standardization": False,
        "scaler_mean": None,
        "scaler_scale": None,
        "metrics": {"n_train": 0, "n_positive": 0, "n_negative": 0, "pairwise_accuracy": 0.0, "positive_mean_score": 0.0, "hard_negative_mean_score": 0.0},
        "convergence_info": {"converged": True, "n_iter": 1, "max_iter": 100, "warning_message": None},
        "seed": 42,
        "max_iter": 100,
    }))
    adapter = LTRAdapter(
        config={"enabled": True, "model_path": str(model_path)},
        journal_store=MagicMock(spec=JournalStore),
        accepted_paper_store=None,
    )
    paper = PaperProfile(title="t", research_area=["AI"])
    j = Journal(journal_id="j", journal_name="J")
    candidates = [(j, 0.5, [])]
    trace = {"j": {"retrieval_rank": 1, "routes": {}}}

    out, diag = adapter.compute_scores(paper, candidates, trace, {}, {})
    # 20-dim 不在 lookup → fallback_feature_dim,但 candidates 顺序保留
    assert diag["status"] == "fallback_feature_dim"
    assert [c[0].journal_id for c in out] == ["j"]


def test_compute_scores_23_dim_reranks(tmp_path: Path):
    """23-dim stub model + attach_features_to_trace 输出 23 维 → compute_scores 跑通。

    2026-06-26: tier+exclusivity schema 从 27 维降到 23 维 (paper_strength + journal_tier_weight 移除)。
    """
    from src.ranker.ltr_adapter import LTRAdapter
    from src.papers.paper_model import PaperProfile
    from src.journals.journal_model import Journal
    from src.journals.journal_store import JournalStore

    # 写一个 23-dim stub model(coef 全 0,predict 不会崩)
    model_path = tmp_path / "stub_23dim.json"
    model_path.write_text(json.dumps({
        "schema_version": 1,
        "model_type": "logistic_regression",
        "backend": "sklearn",
        "feature_names": ["f"] * 23,
        "feature_dim": 23,
        "coef": [0.0] * 23,
        "intercept": 0.0,
        "use_standardization": False,
        "scaler_mean": None,
        "scaler_scale": None,
        "metrics": {"n_train": 0, "n_positive": 0, "n_negative": 0, "pairwise_accuracy": 0.0, "positive_mean_score": 0.0, "hard_negative_mean_score": 0.0},
        "convergence_info": {"converged": True, "n_iter": 1, "max_iter": 100, "warning_message": None},
        "seed": 42,
        "max_iter": 100,
    }))
    journal = Journal(
        journal_id="j", journal_name="J",
        ccf_rating="C", subject_tags=["AI"],
    )
    journal_store = MagicMock(spec=JournalStore)
    journal_store.get_journal.return_value = journal
    adapter = LTRAdapter(
        config={"enabled": True, "model_path": str(model_path)},
        journal_store=journal_store,
        accepted_paper_store=None,
    )
    paper = PaperProfile(title="t", research_area=["AI"])
    candidates = [(journal, 0.5, [])]
    trace = {"j": {"retrieval_rank": 1, "routes": {}}}

    out, diag = adapter.compute_scores(paper, candidates, trace, {"j": 1}, {"j": 0.5})
    # model coef 全 0 → 不会崩
    assert diag["status"] == "ok", f"got {diag}"
    assert "j" in diag["learned_rank"]


def test_compute_scores_passes_paper_anchor_to_attach_features(tmp_path: Path):
    """23-dim LTRAdapter 必须算 paper_anchor_area + n_matching_in_pool 传给 attach_features。

    2026-06-26: tier+exclusivity schema 从 27 维降到 23 维。
    """
    from src.ranker.ltr_adapter import LTRAdapter
    from src.papers.paper_model import PaperProfile
    from src.journals.journal_model import Journal
    from src.journals.journal_store import JournalStore
    from src.ranker import feature_builder as fb

    # 写 23-dim stub
    model_path = tmp_path / "stub_23dim.json"
    model_path.write_text(json.dumps({
        "schema_version": 1, "model_type": "logistic_regression", "backend": "sklearn",
        "feature_names": ["f"] * 23, "feature_dim": 23, "coef": [0.0] * 23, "intercept": 0.0,
        "use_standardization": False, "scaler_mean": None, "scaler_scale": None,
        "metrics": {"n_train": 0, "n_positive": 0, "n_negative": 0, "pairwise_accuracy": 0.0, "positive_mean_score": 0.0, "hard_negative_mean_score": 0.0},
        "convergence_info": {"converged": True, "n_iter": 1, "max_iter": 100, "warning_message": None},
        "seed": 42, "max_iter": 100,
    }))
    captured_kwargs = {}
    orig_attach = fb.attach_features_to_trace

    def spy_attach(trace, paper_profile, journal_store, *args, **kwargs):
        captured_kwargs.update(kwargs)
        return orig_attach(trace, paper_profile, journal_store, *args, **kwargs)

    # Monkey-patch for this test only
    import src.ranker.ltr_adapter as ltr_module
    monkey = __import__("pytest").MonkeyPatch()
    monkey.setattr(ltr_module, "attach_features_to_trace", spy_attach)

    journal = Journal(journal_id="j", journal_name="J", ccf_rating="C", subject_tags=["AI"])
    journal_store = MagicMock(spec=JournalStore)
    journal_store.get_journal.return_value = journal
    adapter = LTRAdapter(
        config={"enabled": True, "model_path": str(model_path)},
        journal_store=journal_store, accepted_paper_store=None,
    )
    paper = PaperProfile(title="t", research_area=["AI"])
    candidates = [(journal, 0.5, [])]
    trace = {"j": {"retrieval_rank": 1, "routes": {}}}

    adapter.compute_scores(paper, candidates, trace, {"j": 1}, {"j": 0.5})
    monkey.undo()

    assert "paper_anchor_area" in captured_kwargs, (
        f"paper_anchor_area not passed to attach_features; kwargs={list(captured_kwargs.keys())}"
    )
    assert captured_kwargs["paper_anchor_area"] == "AI"
    assert "n_matching_in_pool" in captured_kwargs
    assert captured_kwargs["n_matching_in_pool"] == 1  # 只有 1 个候选
