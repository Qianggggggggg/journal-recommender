"""推荐流程编排"""
import logging

import yaml
from typing import Callable, List, Optional, Dict, Any, Tuple

# 2026-06-25 / 2026-06-26: import order matters for lightgbm — lightgbm 必须在
# src.journals.* (内部 import pandas via journal_store) 之前 import。
# 否则 pandas 触发 lightgbm C 库 state 污染,_lgb.Booster() segfault。
# 实测顺序: lightgbm → src.journals.* → feature_builder → Booster OK (500 trees)
#           src.journals.* → lightgbm → feature_builder → Booster segfault
# 2026-06-26: 把 sklearn 从 early-init 拿掉(过度防御;venv 没装 sklearn 会卡,
# 真正的修复点只是「lightgbm < pandas」这一个顺序约束)。
try:  # pragma: no cover
    import lightgbm as _lgb_early  # noqa: F401  # early init
    _HAS_LIGHTGBM_EARLY = True
except ImportError:  # pragma: no cover
    _HAS_LIGHTGBM_EARLY = False
from ..utils.text import quality_adjustment_factor, quality_adjustment_multiplier
from ..journals.journal_model import Journal, JournalMatch
from ..papers.paper_model import PaperInput, PaperProfile
from ..papers.quality_assessor import PaperQualityAssessor
from ..retriever.candidate_generator import CandidateGenerator
from ..ranker.feature_builder import FEATURE_NAMES_WITH_LLM_EVIDENCE  # 6.4
from ..ranker.rule_scorer import RuleScorer
from ..ranker.llm_ranker import LLMRanker
from ..ranker.ltr_adapter import LTRAdapter  # 5.3

logger = logging.getLogger(__name__)


class RecommenderPipeline:
    """推荐流程编排器"""

    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        rule_scorer: RuleScorer,
        llm_ranker: Optional[LLMRanker] = None,
        quality_assessor: Optional[PaperQualityAssessor] = None,
        llm_anchor_guard: Optional[Dict[str, Any]] = None,
        learned_reranker: Optional[LTRAdapter] = None,
        # 6.4: per-paper LLM evidence snapshot, keyed by normalized title.
        # When set and the paper title is present, attach_features uses the
        # 26-dim schema (FEATURE_NAMES_WITH_LLM_EVIDENCE).
        evidence_lookup: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
        feature_schema: str = "19_dim_base",  # "19_dim_base" | "25_dim_with_llm_evidence"
        # 2026-06-26: paper_strength removed (was 20/26)
        # 6.5: optional LLMEvidenceExtractor for the per-request online
        # evidence path. The API layer can call extractor.extract() to
        # backfill the in-memory evidence cache when a paper is not in
        # the offline snapshot. None by default to keep tests
        # backwards-compatible.
        evidence_extractor: Optional["LLMEvidenceExtractor"] = None,
        # Ablation knob: max size of the LLM精排 candidate pool. Default 30
        # preserves the original补位 behavior (top-20 + up to 10 by 4 conditions).
        # Set to 20 to disable补位 entirely and feed only RuleScorer top-20 to LLM.
        llm_max_candidates: int = 30,
        # P-counterfactual 2026-06-16: pass through app.yaml so the
        # pipeline can read experimental flags like
        # ``ranking.experimental.skip_quality_assessment`` without coupling
        # to any specific caller. Optional; default empty dict preserves
        # backward-compat with tests that construct the pipeline without
        # a config.
        app_config: Optional[Dict[str, Any]] = None,
    ):
        self.candidate_generator = candidate_generator
        self.rule_scorer = rule_scorer
        self.llm_ranker = llm_ranker
        self.quality_assessor = quality_assessor
        self.llm_anchor_guard = llm_anchor_guard or {}
        self.learned_reranker = learned_reranker  # 5.3: 默认 None ⇒ 完全跳过 LTR 路径
        # 6.4: evidence_lookup + feature_schema
        self.evidence_lookup = evidence_lookup or {}
        self.feature_schema = feature_schema
        # 决定 attach_features 的 expected feature dim
        # 25-dim 仅在 feature_schema="25_dim_with_llm_evidence" 时启用,
        # 且 paper 标题必须命中 evidence_lookup(否则回退 19 维,见 recommend())。
        # 2026-06-26: paper_strength 移除 → 20/26 → 19/25。
        self._expected_feature_dim = (
            25 if feature_schema == "25_dim_with_llm_evidence" else 19
        )
        # 6.5: optional LLMEvidenceExtractor for the per-request online
        # evidence path. If set, the API layer can call
        # ``extractor.extract(candidates, paper_profile)`` to backfill the
        # in-memory evidence cache when a paper is not in the offline
        # snapshot. The role ranker itself can also call this directly
        # (it already does so as the final fallback in
        # ``rank_with_diagnostics``).
        self.evidence_extractor = evidence_extractor
        # Ablation knob: max LLM精排 candidate pool size.
        self.llm_max_candidates = llm_max_candidates
        # P-counterfactual 2026-06-16: optional app_config so the pipeline
        # can read experimental flags without coupling to a specific
        # caller. Used by ``ranking.experimental.skip_quality_assessment``.
        self._app_config: Dict[str, Any] = app_config or {}

    def recommend(
        self,
        paper_input: PaperInput,
        paper_profile: PaperProfile,
        top_k: int = 5,
        mode: str = "abstract",
        oa_preference: str = "any",
        quality_prompts: Optional[Dict[str, str]] = None,
        diagnostic_journal_ids: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        precomputed_evidence: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """执行推荐流程"""
        # 0. 评估论文质量
        # P-counterfactual 2026-06-16: ranking.experimental.skip_quality_assessment
        # bypasses the LLM quality_assessor call. ``paper_strength = None``
        # triggers the early-exit branch in ``_apply_quality_adjustment``
        # (no tier bias applied). ccf_research_area is forwarded from
        # research_area so domain signals survive. This is the experiment
        # knob for the "is quality assessment actually doing anything?"
        # counterfactual — see tests/test_skip_quality_assessment.py.
        skip_quality = bool(
            self._app_config
            .get("ranking", {})
            .get("experimental", {})
            .get("skip_quality_assessment", False)
        )
        if skip_quality:
            # Forward domain signal without LLM call; clear strength so
            # _apply_quality_adjustment returns the ranked list unchanged.
            paper_profile.ccf_research_area = paper_profile.research_area
            paper_profile.paper_strength = None
            paper_profile.quality_level = None
            paper_profile.readiness = None
            paper_profile.quality_confidence = None
            paper_profile.quality_reasons = []
        elif self.quality_assessor and quality_prompts:
            self._emit_progress(
                progress_callback,
                stage="quality",
                percent=22,
                message="正在评估论文质量...",
            )
            quality = self.quality_assessor.assess(
                paper_input,
                paper_profile,
                quality_prompts.get("system", ""),
                quality_prompts.get("user", ""),
            )
            paper_profile.paper_strength = quality.paper_strength
            paper_profile.readiness = quality.readiness
            paper_profile.quality_level = quality.quality_level
            paper_profile.quality_confidence = quality.confidence
            paper_profile.quality_reasons = quality.reasons
            paper_profile.ccf_research_area = quality.ccf_research_area
            self._emit_progress(
                progress_callback,
                stage="quality",
                percent=25,
                message=f"论文质量评估完成: {quality.quality_level}",
                quality_level=quality.quality_level,
                paper_strength=quality.paper_strength,
            )

        # 1. 候选召回
        query_text = paper_input.title
        if paper_input.abstract:
            query_text += " " + paper_input.abstract

        self._emit_progress(
            progress_callback,
            stage="retrieval",
            percent=30,
            message="正在召回候选期刊...",
        )
        candidates, retrieval_trace = self.candidate_generator.generate_with_trace(
            query_text,
            paper_profile,
            top_k=50,
            mode=mode,
            diagnostic_journal_ids=diagnostic_journal_ids,
        )
        self._emit_progress(
            progress_callback,
            stage="retrieval",
            percent=45,
            message=f"候选召回完成: {len(candidates)} 个候选",
            candidate_count=len(candidates),
        )

        if not candidates:
            return {"recommendations": [], "warning": "未找到合适的候选期刊"}

        # 2. 阶段一：规则打分（只做主题匹配，不含质量调整）
        # 覆盖完整粗召回池，便于评估定位真实 venue 在规则排序中的位置。
        self._emit_progress(
            progress_callback,
            stage="rule_ranking",
            percent=52,
            message="正在进行规则排序...",
            candidate_count=len(candidates),
        )
        rule_ranked_all = self.rule_scorer.rank(
            candidates,
            paper_profile,
            oa_preference=oa_preference,
            top_k=len(candidates),
            retrieval_trace=retrieval_trace,
        )

        # 2.5 质量调整软权重（在 Pipeline 中统一应用，解耦）
        rule_ranked_all = self._apply_quality_adjustment(rule_ranked_all, paper_profile)
        rule_ranks_map = {
            journal.journal_id: index + 1
            for index, (journal, _score, _reasons) in enumerate(rule_ranked_all)
        }
        rule_scores_map = {
            journal.journal_id: float(score)
            for journal, score, _reasons in rule_ranked_all
        }
        self._emit_progress(
            progress_callback,
            stage="rule_ranking",
            percent=62,
            message=f"规则排序完成: {len(rule_ranked_all)} 个候选",
            rule_candidate_count=len(rule_ranked_all),
        )

        # 2.6 (6.4) 把 LTR 训练特征注入 retrieval_trace(per journal)
        # 默认 19 维;当 feature_schema="25_dim_with_llm_evidence" 且
        # paper 标题命中 evidence_lookup 时,输出 25 维 evidence schema。
        # 缺 paper_evidence 时**自动回退** 19 维(防御性,不破坏 baseline)。
        # 2026-06-26: paper_strength 移除 → 20/26 → 19/25。
        paper_title_key = " ".join((paper_profile.title or "").casefold().split())
        paper_evidence_entry = self.evidence_lookup.get(paper_title_key, {})
        # evidence_lookup stores whole paper entries (title/venue/rule_ranks/
        # learned_ranks/candidates/evidence/...). attach_features wants
        # the per-journal-id evidence dict ({jid: ev_item}), so unwrap.
        paper_evidence = (
            paper_evidence_entry.get("evidence", {})
            if isinstance(paper_evidence_entry, dict)
            else {}
        )
        feature_names: Optional[List[str]] = None
        if self._expected_feature_dim == 26 and paper_evidence:
            feature_names = FEATURE_NAMES_WITH_LLM_EVIDENCE
        # accepted_paper_store 暂未在 pipeline 持有,传 None(attach_features
        # 内部会按 store=None 处理,candidate_in_accepted_corpus 全 0)。
        # 2026-06-25: 透传 paper_ccf_target_level 让 same_ccf_level 在
        # inference 也能计算(前提是 paper metadata 含 ccf_level)。
        self.candidate_generator.attach_features(
            trace=retrieval_trace,
            paper_profile=paper_profile,
            rule_ranks=rule_ranks_map,
            rule_scores=rule_scores_map,
            accepted_paper_store=None,
            feature_names=feature_names,
            llm_evidence_by_journal=paper_evidence,
            paper_ccf_target_level=getattr(paper_profile, "ccf_target_level", None),
        )

        # 构建 LLM 候选集：top20 + scope 边界强候选 + 同领域参考候选 + 受控 typical-only 候选
        llm_candidates = rule_ranked_all[:20]  # top20 必选
        seen_ids = {j.journal_id for j, _, _ in llm_candidates}
        top20_floor = llm_candidates[-1][1] if llm_candidates else 0.0

        # 优先补入 scope 边界证据强、但被规则分压到 top20 外的候选。
        for journal, score, reasons in rule_ranked_all[20:]:
            if journal.journal_id in seen_ids:
                continue
            if self._has_scope_boundary_evidence(retrieval_trace.get(journal.journal_id)):
                if len(llm_candidates) < self.llm_max_candidates:
                    llm_candidates.append((journal, score, reasons))
                    seen_ids.add(journal.journal_id)

        # 同领域但未入选的高分候选（从 top20 之后找）
        if paper_profile.research_area:
            research_areas = set(paper_profile.research_area)
            for journal, score, reasons in rule_ranked_all[20:]:
                if journal.subject_tags and journal.journal_id not in seen_ids:
                    matched = set(journal.subject_tags) & research_areas
                    if matched and len(llm_candidates) < self.llm_max_candidates:
                        llm_candidates.append((journal, score, reasons))
                        seen_ids.add(journal.journal_id)

        # subject_tags 与 ccf_research_area 匹配的候选（兜底）
        if paper_profile.ccf_research_area:
            ccf_areas = set(paper_profile.ccf_research_area)
            for journal, score, reasons in rule_ranked_all[20:]:
                if journal.subject_tags and journal.journal_id not in seen_ids:
                    matched = set(journal.subject_tags) & ccf_areas
                    if matched and len(llm_candidates) < self.llm_max_candidates:
                        llm_candidates.append((journal, score, reasons))
                        seen_ids.add(journal.journal_id)

        # 最后才允许 pure typical 候选补位，且要求规则分接近 top20 门槛。
        for journal, score, reasons in rule_ranked_all[20:]:
            if journal.journal_id in seen_ids:
                continue
            if not self._is_typical_only(retrieval_trace.get(journal.journal_id)):
                continue
            if score >= top20_floor * 0.8 and len(llm_candidates) < self.llm_max_candidates:
                llm_candidates.append((journal, score, reasons))
                seen_ids.add(journal.journal_id)

        # 2.7 (5.3) LTR 纯 rerank:在 llm_candidates 内部重排,不改 score 语义。
        # 默认 OFF (learned_reranker=None) → 整个 if 块跳过,llm_candidates 顺序不变。
        learned_diag: Dict[str, Any] = {
            "learned_score": {},
            "learned_rank": {},
            "status": "fallback_disabled",
        }
        if self.learned_reranker and self.learned_reranker.enabled:
            try:
                reranked, learned_diag = self.learned_reranker.compute_scores(
                    paper_profile=paper_profile,
                    llm_candidates=llm_candidates,
                    retrieval_trace=retrieval_trace,
                    rule_ranks=rule_ranks_map,
                    rule_scores=rule_scores_map,
                )
                llm_candidates = reranked  # 纯 rerank:替换候选顺序,集合不变
            except Exception as e:  # 防御:LTR 任何异常都不能破坏推荐主流程
                logger.warning("LTR rerank skipped: %s", e)
                learned_diag = {
                    "learned_score": {},
                    "learned_rank": {},
                    "status": "fallback_pipeline_error",
                }

        # 3. 阶段二：LLM 精排；重试耗尽后使用 Rule TopK，避免返回空结果。
        rank_method = "rule"
        fallback_reason = ""
        llm_role_diag: Dict[str, Any] = {}
        if self.llm_ranker:
            try:
                self._emit_progress(
                    progress_callback,
                    stage="llm_ranking",
                    percent=68,
                    message=f"正在进行 LLM 精排: {len(llm_candidates)} 个候选",
                    llm_candidate_count=len(llm_candidates),
                )
                rank_with_diagnostics = getattr(
                    self.llm_ranker, "rank_with_diagnostics", None
                )
                if callable(rank_with_diagnostics):
                    # 6.5: only thread precomputed_evidence when the ranker
                    # actually accepts it. Older rankers (and test mocks)
                    # may not have this kwarg in their signature.
                    import inspect as _inspect
                    _rwd_params = _inspect.signature(rank_with_diagnostics).parameters
                    _rwd_kwargs = {
                        "candidates": llm_candidates,
                        "paper_profile": paper_profile,
                        "top_k": len(llm_candidates),
                        "retrieval_trace": retrieval_trace,
                        "rule_ranks": rule_ranks_map,
                        "rule_scores": rule_scores_map,
                        "learned_ranks": learned_diag.get("learned_rank") or {},
                        # 6.4: thread the actual LTR score so the role ranker
                        # can use it as a 3rd formula component when
                        # ltr_score_weight > 0.
                        "learned_scores": learned_diag.get("learned_score") or {},
                    }
                    if "precomputed_evidence" in _rwd_params:
                        _rwd_kwargs["precomputed_evidence"] = precomputed_evidence
                    llm_ranked_all, rank_method, llm_role_diag = (
                        rank_with_diagnostics(**_rwd_kwargs)
                    )
                else:
                    llm_ranked_all, rank_method = self.llm_ranker.rank(
                        llm_candidates,
                        paper_profile,
                        top_k=len(llm_candidates),
                        retrieval_trace=retrieval_trace,
                    )
                llm_ranked = self._select_final_llm_ranked(
                    llm_ranked_all,
                    llm_candidates,
                    top_k,
                )
            except Exception as e:
                logger.warning("LLM ranking failed; using Rule TopK fallback: %s", e)
                rank_method = "rule_fallback"
                fallback_reason = f"{type(e).__name__}: {e}"[:500]
                llm_ranked = [
                    (journal, score, reasons, 0.5)
                    for journal, score, reasons in rule_ranked_all[:top_k]
                ]
            self._emit_progress(
                progress_callback,
                stage="llm_ranking",
                percent=78,
                message=(
                    "LLM 精排失败，已使用规则排序结果"
                    if rank_method == "rule_fallback"
                    else "LLM 精排完成"
                ),
                llm_candidate_count=len(llm_candidates),
            )
        else:
            llm_ranked = [(j, s, r, 0.5) for j, s, r in llm_candidates[:top_k]]

        # 5.3: final_rank_source 记录最终 Top5 由哪一层决定。
        # v1 纯 rerank:LTR 只重排 llm_candidates 输入,最终选择仍由 LLM score + anchor guard 完成。
        # 故 LTR ON 但 LLM 正常时记为 "llm_after_learned_rerank"(而不是 "learned")。
        final_rank_source = "llm"
        if not self.llm_ranker:
            final_rank_source = "rule"
        elif rank_method == "rule_fallback":
            final_rank_source = "rule_fallback"
        elif rank_method.startswith("llm_evidence_"):
            final_rank_source = rank_method
        if (
            rank_method == "llm"
            and self.learned_reranker
            and self.learned_reranker.enabled
            and learned_diag.get("status") == "ok"
        ):
            final_rank_source = "llm_after_learned_rerank"

        if llm_role_diag and learned_diag.get("status") == "ok":
            learned_scores = learned_diag.get("learned_score") or {}
            learned_ranks = learned_diag.get("learned_rank") or {}
            for journal_id, detail in (
                llm_role_diag.get("candidates") or {}
            ).items():
                detail["learned_score"] = learned_scores.get(journal_id)
                detail["learned_rank"] = learned_ranks.get(journal_id)

        # 4. 构建推荐结果（直接使用 LLMRanker 输出的 reasons，不再单独调用 Explainer）
        self._emit_progress(
            progress_callback,
            stage="finalizing",
            percent=82,
            message="正在整理推荐结果...",
            recommendation_count=len(llm_ranked),
        )
        recommendations = []
        for journal, score, reasons, confidence in llm_ranked:
            recommendations.append(JournalMatch(
                journal=journal,
                score=score,
                confidence=confidence,
                match_reasons=reasons if reasons else [],
                matched_fields=["research_area", "method_type"],
            ))

        # 5. 构建响应
        result = {
            "recommendations": recommendations,
            "candidates": candidates,  # 粗排候选（用于调试分析）
            "rule_ranked": rule_ranked_all,  # RuleScorer 排序结果（用于调试分析）
            "llm_candidates": llm_candidates,  # LLM 精排候选池（用于评估诊断）
            "llm_candidate_ids": [j.journal_id for j, _, _ in llm_candidates],
            "retrieval_trace": retrieval_trace,  # 候选召回来源（用于评估噪声定位）
            "paper_profile": paper_profile,
            "mode_used": mode,
            "rank_method": rank_method,
        }
        if rank_method == "rule_fallback":
            result.update({
                "fallback_used": True,
                "fallback_stage": "llm_ranking",
                "fallback_reason": fallback_reason,
            })
        if llm_role_diag:
            result["llm_role_diagnostics"] = llm_role_diag
            result["final_rank_source"] = final_rank_source

        # 5.3: 默认 OFF 时**完全不写**新 key（bit-equal baseline 强约束）。
        # 6.4: 同时把 learned_score / learned_rank 顶层暴露给下游消费者
        # (例如 6.4 ablation runner 需要把 learned_score 注入 role ranker)。
        if (
            self.learned_reranker
            and self.learned_reranker.enabled
            and learned_diag.get("status") == "ok"
        ):
            result["learned_diagnostics"] = learned_diag
            result["final_rank_source"] = final_rank_source
            result["learned_score"] = learned_diag.get("learned_score", {})
            result["learned_rank"] = learned_diag.get("learned_rank", {})

        # 标题模式加警告
        if mode == "title":
            result["warning"] = "置信度较低，建议补充摘要以获得更精确的推荐"

        return result

    def _emit_progress(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        **payload: Any,
    ) -> None:
        """Emit a best-effort progress event without affecting recommendation results."""
        if not progress_callback:
            return
        try:
            progress_callback(payload)
        except Exception:
            return

    def _has_scope_boundary_evidence(self, trace: Optional[dict]) -> bool:
        if not trace:
            return False
        routes = trace.get("routes", {})
        scope_routes = [
            data
            for route, data in routes.items()
            if route.startswith("scope_")
        ]
        if not scope_routes:
            return False
        return any(
            float(data.get("weighted_score") or 0.0) > 0.0
            or int(data.get("rank") or 9999) <= 12
            for data in scope_routes
        )

    def _is_typical_only(self, trace: Optional[dict]) -> bool:
        if not trace:
            return False
        routes = trace.get("routes", {})
        has_scope = any(route.startswith("scope_") for route in routes)
        has_typical = any(route.startswith("typical_") or route == "identity_anchor" for route in routes)
        return has_typical and not has_scope

    def _apply_quality_adjustment(
        self,
        ranked: List[Tuple[Journal, float, List[str]]],
        paper_profile: PaperProfile,
    ) -> List[Tuple[Journal, float, List[str]]]:
        """应用质量软权重调整（解耦：质量评估结果不再直接流入 RuleScorer）

        P1 (2026-06-16): 改用 strength-aware CCF-tier multiplier
        (quality_adjustment_multiplier)，弱论文（strength<0.5）偏好 C 级
        期刊，强论文保持 A>B>C。详见 src/utils/text.py 和
        tests/test_quality_adjustment.py。
        """
        if paper_profile.paper_strength is None:
            return ranked

        strength = paper_profile.paper_strength

        base_adjustment = quality_adjustment_factor(strength)

        adjusted = []
        for journal, score, reasons in ranked:
            ccf_multiplier = quality_adjustment_multiplier(
                strength, journal.ccf_rating
            )
            adjustment = base_adjustment * ccf_multiplier

            adjusted_score = score * adjustment
            new_reasons = reasons.copy()
            if strength >= 0.65:
                new_reasons.append(f"强论文调整(+{(adjustment-1)*100:.0f}%)")
            elif strength < 0.35:
                new_reasons.append(f"弱论文调整({(adjustment-1)*100:.0f}%)")

            adjusted.append((journal, adjusted_score, new_reasons))

        # 重新排序
        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted

    def _select_final_llm_ranked(
        self,
        llm_ranked: List[Tuple[Journal, float, List[str], float]],
        llm_candidates: List[Tuple[Journal, float, List[str]]],
        top_k: int,
    ) -> List[Tuple[Journal, float, List[str], float]]:
        """Select final Top-K while preserving close high-rule-rank anchors."""
        selected = list(llm_ranked[:top_k])
        guard = self.llm_anchor_guard or {}
        if not guard.get("enabled", False) or top_k <= 0:
            return selected

        protect_rule_rank = int(guard.get("protect_rule_rank", 5))
        max_score_gap = float(guard.get("max_score_gap", 0.08))
        rule_rank_by_id = {
            journal.journal_id: idx + 1
            for idx, (journal, _, _) in enumerate(llm_candidates)
        }
        llm_by_id = {item[0].journal_id: item for item in llm_ranked}
        selected_ids = {journal.journal_id for journal, _, _, _ in selected}

        protected_ids = [
            journal.journal_id
            for journal, _, _ in llm_candidates[:protect_rule_rank]
            if journal.journal_id not in selected_ids
        ]

        for journal_id in protected_ids:
            protected = llm_by_id.get(journal_id)
            if not protected:
                continue
            if len(selected) < top_k:
                selected.append(protected)
                selected_ids.add(journal_id)
                continue

            replaceable = [
                idx
                for idx, (journal, _, _, _) in enumerate(selected)
                if rule_rank_by_id.get(journal.journal_id, 9999) > protect_rule_rank
            ]
            if not replaceable:
                replaceable = list(range(len(selected)))

            replace_idx = min(replaceable, key=lambda idx: selected[idx][1])
            if selected[replace_idx][1] - protected[1] <= max_score_gap:
                removed_id = selected[replace_idx][0].journal_id
                selected[replace_idx] = protected
                selected_ids.discard(removed_id)
                selected_ids.add(journal_id)

        selected.sort(key=lambda item: item[1], reverse=True)
        return selected[:top_k]

    @classmethod
    def from_config(cls, config_path: str = "configs/app.yaml") -> "RecommenderPipeline":
        """从配置文件创建"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 加载 prompts
        with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        # 这里需要传入已初始化的组件
        # 实际使用时通过依赖注入
        return cls(
            candidate_generator=None,  # 外部注入
            rule_scorer=RuleScorer(),
            llm_ranker=None,
        )
