"""LTRAdapter: thin wrapper around LearningToRanker (Task 5.3).

职责:
- 加载模型(plan 5.2 训练的 ``data/models/learning_to_ranker.json``),并集中
  处理 6 类失败(enabled=False / model_path 缺失 / JSON 损坏 / 缺字段 /
  未收敛 / feature_dim 不匹配),任一失败都安静降级 + 设 ``disable_reason``。
- 对 caller 的 ``retrieval_trace`` 做**副本**调
  ``attach_features_to_trace``,绝不动 caller 的 trace(避免污染 evaluation
  diagnostics 的 ``retrieval_sources`` 等字段)。
- 稳定排序(stable sort)输出 reranked list;ties 按输入顺序 tiebreak。
- 任何异常都**不传播**到 pipeline,避免破坏推荐主流程。

设计原则:
- v1 = pure rerank:不消费 ``blend_with_rule_score`` / ``blend_with_llm_score``
  字段(plan 5.3 标 reserved for 5.4+ ablation)。
- 默认 OFF(yaml ``enabled: false``)→ LTRAdapter 实例化但 ``enabled=False``,
  pipeline 跳过所有 LTR 路径,baseline bit-equal。
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.journals.accepted_paper_store import AcceptedPaperStore
from src.journals.journal_model import Journal
from src.journals.journal_store import JournalStore
from src.papers.paper_model import PaperProfile
from src.ranker.feature_builder import attach_features_to_trace
from src.ranker.learning_to_rank import LearningToRanker

logger = logging.getLogger(__name__)


# 输入/输出类型:与 RuleScorer / LLMRanker pipeline 内部一致
LTRCandidate = Tuple[Journal, float, List[str]]


def _empty_diag(status: str = "fallback_disabled") -> Dict[str, Any]:
    return {
        "learned_score": {},
        "learned_rank": {},
        "status": status,
    }


class LTRAdapter:
    """Thin wrapper around LearningToRanker with failure isolation.

    配置 (config dict):
        enabled: bool              # default False
        model_path: str            # 指向 train_learning_to_rank.py 输出的 JSON
        blend_with_rule_score: float  # reserved; 5.3 v1 不消费
        blend_with_llm_score: float   # reserved; 5.3 v1 不消费

    初始化失败 (任一) → self._enabled=False, self._ranker=None,
    self._disable_reason=<人类可读原因>。不抛异常。
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]],
        journal_store: JournalStore,
        accepted_paper_store: Optional[AcceptedPaperStore] = None,
    ) -> None:
        self._config: Dict[str, Any] = dict(config or {})
        self._journal_store = journal_store
        self._accepted_paper_store = accepted_paper_store
        self._ranker: Optional[LearningToRanker] = None
        self._enabled: bool = False
        self._disable_reason: Optional[str] = None
        self._initialize()

    # ---- public API ----

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def disable_reason(self) -> Optional[str]:
        return self._disable_reason

    @property
    def feature_dim(self) -> Optional[int]:
        return getattr(self._ranker, "_feature_dim", None) if self._ranker else None

    @property
    def model_converged(self) -> Optional[bool]:
        if self._ranker is None:
            return None
        info = getattr(self._ranker, "convergence_info", None) or {}
        return info.get("converged")

    def compute_scores(
        self,
        paper_profile: Optional[PaperProfile],
        llm_candidates: List[LTRCandidate],
        retrieval_trace: Dict[str, dict],
        rule_ranks: Optional[Dict[str, int]] = None,
        rule_scores: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[LTRCandidate], Dict[str, Any]]:
        """对 llm_candidates 内部重排,返回 (reranked, diagnostics)。

        diagnostics schema::

            {
                "learned_score": {jid: float, ...},     # LTR 给的概率(类 1)
                "learned_rank":  {jid: int, ...},        # 1-indexed 在 rerank 后的位置
                "status":        "ok" | "fallback_*",
            }

        失败语义(任一发生 → 返回原 list + status=fallback_*):
        - adapter 初始化时已被禁用 → status="fallback_disabled"
        - llm_candidates 为空 → status="ok",空 dict
        - attach_features 抛异常 → status="fallback_feature_build"
        - 任一候选 features 维度 != self._feature_dim → status="fallback_feature_dim"
        - predict_scores 抛异常 → status="fallback_predict"
        - 任何其他异常 → status="fallback_unknown"
        """
        if not self._enabled:
            return list(llm_candidates), _empty_diag("fallback_disabled")
        if not llm_candidates:
            return list(llm_candidates), _empty_diag("ok")

        # 1. 复制 caller 的 trace,绝不修改原 dict
        trace_copy: Dict[str, dict] = {
            jid: dict(entry) for jid, entry in (retrieval_trace or {}).items()
        }

        # 2. 在副本上注入 features
        # 6.4 fix: detect the schema (20-dim base vs 26-dim with-llm-evidence)
        # from the existing trace entry's "features" list length. If the
        # caller (pipeline) already attached features of the correct
        # schema, reuse them; otherwise re-attach with the right schema.
        expected_dim = self._ranker._feature_dim  # type: ignore[union-attr]
        existing_schema_dim = None
        for entry in trace_copy.values():
            feats = entry.get("features")
            if isinstance(feats, list) and feats:
                existing_schema_dim = len(feats)
                break

        if existing_schema_dim == expected_dim:
            # Caller already wrote features at the right dim. Skip re-attach
            # to avoid silently downgrading 26-dim features to 20-dim.
            pass
        else:
            # Pick feature_names that match expected_dim so re-attach
            # produces the right shape.
            from src.ranker.feature_builder import (
                FEATURE_NAMES,
                FEATURE_NAMES_WITH_LLM_EVIDENCE,
            )
            feature_names = (
                FEATURE_NAMES_WITH_LLM_EVIDENCE
                if expected_dim == 26
                else FEATURE_NAMES
            )
            try:
                attach_features_to_trace(
                    trace_copy,
                    paper_profile,
                    self._journal_store,
                    rule_ranks,
                    rule_scores,
                    self._accepted_paper_store,
                    feature_names=feature_names,
                )
            except Exception as e:  # 防御:任何 attach 异常都降级
                logger.warning("LTRAdapter: attach_features_to_trace failed, fallback to identity: %s", e)
                return list(llm_candidates), _empty_diag("fallback_feature_build")

        # 3. 抽 feature rows,逐候选校验 feature_dim
        # (expected_dim was already set above from self._ranker._feature_dim)
        rows: List[Dict[str, Any]] = []
        jids: List[str] = []
        for journal, _score, _reasons in llm_candidates:
            entry = trace_copy.get(journal.journal_id) or {}
            feats = entry.get("features")
            if not isinstance(feats, list) or len(feats) != expected_dim:
                logger.warning(
                    "LTRAdapter: feature dim mismatch for %s (got %s, expected %s); "
                    "falling back to identity order",
                    journal.journal_id,
                    None if feats is None else len(feats),
                    expected_dim,
                )
                return list(llm_candidates), _empty_diag("fallback_feature_dim")
            rows.append({"features": list(feats)})
            jids.append(journal.journal_id)

        # 4. 推理
        try:
            scores = self._ranker.predict_scores(rows)  # type: ignore[union-attr]
        except Exception as e:
            logger.warning("LTRAdapter: predict_scores failed, fallback to identity: %s", e)
            return list(llm_candidates), _empty_diag("fallback_predict")

        # 5. stable sort:learned_score desc,输入顺序 tiebreak
        score_by_id: Dict[str, float] = {jid: float(s) for jid, s in zip(jids, scores)}
        order = sorted(
            range(len(llm_candidates)),
            key=lambda i: (-score_by_id[jids[i]], i),
        )
        reranked = [llm_candidates[i] for i in order]
        rank_by_id: Dict[str, int] = {
            jids[i]: rk for rk, i in enumerate(order, start=1)
        }

        return reranked, {
            "learned_score": score_by_id,
            "learned_rank": rank_by_id,
            "status": "ok",
        }

    # ---- internals ----

    def _initialize(self) -> None:
        if not self._config.get("enabled", False):
            self._disable_reason = "disabled in config (enabled=False)"
            return
        model_path_raw = self._config.get("model_path")
        if not model_path_raw:
            self._disable_reason = "model_path is empty or missing"
            logger.warning("LTRAdapter: %s", self._disable_reason)
            return
        model_path = Path(model_path_raw)
        if not model_path.exists():
            self._disable_reason = f"model file not found: {model_path}"
            logger.warning("LTRAdapter: %s", self._disable_reason)
            return
        try:
            self._ranker = LearningToRanker.load(str(model_path))
        except Exception as e:
            self._disable_reason = f"failed to load model ({type(e).__name__}): {e}"
            logger.warning("LTRAdapter: %s", self._disable_reason)
            return
        info = getattr(self._ranker, "convergence_info", None) or {}
        if info.get("converged") is False:
            n_iter = info.get("n_iter")
            max_iter = info.get("max_iter")
            warn = info.get("warning_message")
            self._disable_reason = (
                f"model did not converge (n_iter={n_iter}, max_iter={max_iter}): {warn}"
            )
            logger.warning("LTRAdapter: %s", self._disable_reason)
            self._ranker = None
            return
        # 成功路径
        self._enabled = True
        self._disable_reason = None
        logger.info(
            "LTRAdapter enabled (model=%s, feature_dim=%s, converged=%s)",
            model_path,
            self._ranker._feature_dim,
            info.get("converged"),
        )
