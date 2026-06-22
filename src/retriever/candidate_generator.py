"""混合召回"""
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..journals.accepted_paper_store import AcceptedPaperStore
from ..journals.journal_model import Journal
from ..journals.journal_store import JournalStore
from ..papers.paper_model import PaperProfile
from ..ranker.feature_builder import attach_features_to_trace
from .accepted_paper_retriever import (
    AcceptedPaperBM25Retriever,
    AcceptedPaperEmbeddingRetriever,
)
from .bm25_retriever import BM25Retriever
from .embedding_retriever import EmbeddingRetriever
from .typical_abstract_retriever import (
    TypicalAbstractBM25Retriever,
    TypicalAbstractEmbeddingRetriever,
    TypicalAbstractTextRetriever,
)


class CandidateGenerator:
    """候选召回生成器（混合召回）"""

    def __init__(
        self,
        store: JournalStore,
        bm25_retriever: BM25Retriever,
        embedding_retriever: Optional[EmbeddingRetriever] = None,
        merge_weights: Optional[Dict[str, float]] = None,
        retrieval_target: str = "scope_text",
        typical_bm25_retriever: Optional[TypicalAbstractBM25Retriever] = None,
        typical_embedding_retriever: Optional[TypicalAbstractEmbeddingRetriever] = None,
        typical_text_retriever: Optional[TypicalAbstractTextRetriever] = None,
        accepted_bm25_retriever: Optional[AcceptedPaperBM25Retriever] = None,
        accepted_embedding_retriever: Optional[AcceptedPaperEmbeddingRetriever] = None,
        weight_gater: Optional[object] = None,
        use_gating: bool = False,
        hybrid_scope_weight: float = 0.75,
        hybrid_typical_weight: float = 0.25,
        identity_anchor_weight: float = 0.03,
        accepted_paper_weight: float = 0.20,
        fusion_strategy: str = "weighted_minmax",
        rrf_k: int = 60,
        route_top_k: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        self.store = store
        self.bm25_retriever = bm25_retriever
        self.embedding_retriever = embedding_retriever
        self.merge_weights = merge_weights or {"bm25": 0.45, "vector": 0.35, "text": 0.20}
        self.retrieval_target = retrieval_target
        self.typical_bm25_retriever = typical_bm25_retriever
        self.typical_embedding_retriever = typical_embedding_retriever
        self.typical_text_retriever = typical_text_retriever
        self.accepted_bm25_retriever = accepted_bm25_retriever
        self.accepted_embedding_retriever = accepted_embedding_retriever
        self.weight_gater = weight_gater
        self.use_gating = use_gating
        self.hybrid_scope_weight = hybrid_scope_weight
        self.hybrid_typical_weight = hybrid_typical_weight
        self.identity_anchor_weight = identity_anchor_weight
        self.accepted_paper_weight = accepted_paper_weight
        self.fusion_strategy = fusion_strategy
        self.rrf_k = rrf_k
        self.route_top_k = self._merge_route_top_k(route_top_k)

    def generate(
        self,
        query_text: str,
        paper_profile: PaperProfile,
        top_k: int = 40,
        mode: str = "abstract",
    ) -> List[Journal]:
        """生成候选期刊"""
        candidates, _ = self.generate_with_trace(query_text, paper_profile, top_k=top_k, mode=mode)
        return candidates

    def attach_features(
        self,
        trace: Dict[str, dict],
        paper_profile: PaperProfile,
        rule_ranks: Optional[Dict[str, int]],
        rule_scores: Optional[Dict[str, float]],
        accepted_paper_store: Optional[AcceptedPaperStore] = None,
        feature_names: Optional[List[str]] = None,
        llm_evidence_by_journal: Optional[Dict[str, Dict[str, Any]]] = None,
        paper_anchor_area: Optional[str] = None,
        n_matching_in_pool: Optional[int] = None,
    ) -> None:
        """把 LTR 训练特征注入 trace(per Task 4.1.d + ADR 0001)。

        必须在 rule_scorer.rank(...) 之后调用,因为 features 需要 rule_rank/rule_score。
        ``accepted_paper_store`` 可为 None(此时 candidate_in_accepted_corpus 全为 0)。

        显式传入 ``feature_names``(例如 ``FEATURE_NAMES_WITH_LLM_EVIDENCE`` 或
        ``FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY``)与 ``llm_evidence_by_journal``
        时,会输出对应的 schema(20/26/28 维)。
        不传则保持默认 20 维 ``FEATURE_NAMES``。

        阶段 6.5 (P2-mini):``paper_anchor_area`` + ``n_matching_in_pool`` 透传
        给 attach_features_to_trace,仅当 feature_names 含 ``area_exclusivity``
        (即 28 维 schema)时才有意义。

        原地修改 trace:每本期刊 entry 增加 ``features`` (list[float]) 与
        ``feature_names`` (list[str])。
        """
        attach_features_to_trace(
            trace=trace,
            paper_profile=paper_profile,
            journal_store=self.store,
            rule_ranks=rule_ranks,
            rule_scores=rule_scores,
            accepted_paper_store=accepted_paper_store,
            llm_evidence_by_journal=llm_evidence_by_journal,
            feature_names=feature_names,
            paper_anchor_area=paper_anchor_area,
            n_matching_in_pool=n_matching_in_pool,
        )

    def generate_with_trace(
        self,
        query_text: str,
        paper_profile: PaperProfile,
        top_k: int = 40,
        mode: str = "abstract",
        diagnostic_journal_ids: Optional[Iterable[str]] = None,
    ) -> Tuple[List[Journal], Dict[str, dict]]:
        """生成候选期刊，并返回每本期刊的召回来源。"""
        cfg = self._route_config_for_mode(mode)
        diagnostic_ids = set(diagnostic_journal_ids or [])

        # 构建丰富的检索 query，包含 paper_profile 的所有关键字段
        rich_query = self._build_rich_query(query_text, paper_profile)

        weights = self._weights_for_profile(paper_profile)

        if self._use_typical_abstracts():
            route_results = self._hybrid_route_results(rich_query, paper_profile, cfg, weights)
        else:
            route_results = self._scope_route_results(rich_query, paper_profile, cfg, weights)

        candidates, trace = self._merge_route_results(
            route_results,
            top_k=top_k,
        )
        if diagnostic_ids:
            wide_cfg = {key: value * 4 for key, value in cfg.items()}
            if self._use_typical_abstracts():
                wide_route_results = self._hybrid_route_results(rich_query, paper_profile, wide_cfg, weights)
            else:
                wide_route_results = self._scope_route_results(rich_query, paper_profile, wide_cfg, weights)
            self._attach_diagnostic_trace(trace, wide_route_results, diagnostic_ids)
        return candidates, trace

    def _scope_text_search(self, paper_profile: PaperProfile, top_k: int = 10) -> List[Tuple[Journal, float]]:
        """基于 scope/profile 的关键词交集文本搜索。"""
        # 构建检索文本：从 paper_profile 的多个字段提取关键词
        query_parts = [paper_profile.title]
        if paper_profile.abstract:
            query_parts.append(paper_profile.abstract)
        if paper_profile.keywords:
            query_parts.extend(paper_profile.keywords)
        if paper_profile.research_area:
            query_parts.extend(paper_profile.research_area)
        if paper_profile.techniques:
            query_parts.extend(paper_profile.techniques)

        query_text = " ".join(query_parts)
        return self.store.search_by_text(query_text, top_k=top_k)

    def _text_search(
        self,
        paper_profile: PaperProfile,
        rich_query: str = "",
        top_k: int = 10,
    ) -> List[Tuple[Journal, float]]:
        """兼容旧实验脚本：返回当前目标对应的文本召回路由。"""
        if self._use_typical_abstracts() and self.typical_text_retriever:
            return self.typical_text_retriever.retrieve(rich_query, top_k=top_k)
        return self._scope_text_search(paper_profile, top_k=top_k)

    def _scope_route_results(
        self,
        rich_query: str,
        paper_profile: PaperProfile,
        cfg: Dict[str, int],
        weights: Dict[str, float],
    ) -> Dict[str, Tuple[List[Tuple[Journal, float]], float]]:
        """scope_text 模式保持原有三路召回。"""
        route_results: Dict[str, Tuple[List[Tuple[Journal, float]], float]] = {
            "scope_bm25": (self.bm25_retriever.retrieve(rich_query, top_k=cfg["bm25"]), weights["bm25"]),
        }
        if self.embedding_retriever:
            route_results["scope_vector"] = (
                self.embedding_retriever.retrieve(rich_query, top_k=cfg["vector"]),
                weights["vector"],
            )
        route_results["scope_text"] = (self._scope_text_search(paper_profile, top_k=cfg["text"]), weights["text"])
        return route_results

    def _hybrid_route_results(
        self,
        rich_query: str,
        paper_profile: PaperProfile,
        cfg: Dict[str, int],
        weights: Dict[str, float],
    ) -> Dict[str, Tuple[List[Tuple[Journal, float]], float]]:
        """典型摘要模式：scope 作为身份边界，typical 作为语义扩展。"""
        route_results = self._scope_route_results(
            rich_query,
            paper_profile,
            cfg,
            {key: value * self.hybrid_scope_weight for key, value in weights.items()},
        )

        if self.typical_bm25_retriever:
            route_results["typical_bm25"] = (
                self.typical_bm25_retriever.retrieve(rich_query, top_k=cfg["bm25"]),
                weights["bm25"] * self.hybrid_typical_weight,
            )
        if self.typical_embedding_retriever:
            route_results["typical_vector"] = (
                self.typical_embedding_retriever.retrieve(rich_query, top_k=cfg["vector"]),
                weights["vector"] * self.hybrid_typical_weight,
            )
        if self.typical_text_retriever:
            route_results["typical_text"] = (
                self.typical_text_retriever.retrieve(rich_query, top_k=cfg["text"]),
                weights["text"] * self.hybrid_typical_weight,
            )
        route_results["identity_anchor"] = (
            self._identity_anchor_search(rich_query, paper_profile, top_k=max(cfg.values())),
            self.identity_anchor_weight,
        )
        # accepted-paper route:基于真实发表论文画像的 BM25/向量召回。
        # 若对应 retriever 未注入 (例如索引未构建),自动跳过,不影响其他路由。
        if self.accepted_bm25_retriever:
            accepted_bm25_top_k = cfg.get("accepted_bm25", cfg.get("bm25", 28))
            route_results["accepted_bm25"] = (
                self.accepted_bm25_retriever.retrieve(rich_query, top_k=accepted_bm25_top_k),
                weights["bm25"] * self.accepted_paper_weight,
            )
        if self.accepted_embedding_retriever:
            accepted_vector_top_k = cfg.get("accepted_vector", cfg.get("vector", 28))
            route_results["accepted_vector"] = (
                self.accepted_embedding_retriever.retrieve(rich_query, top_k=accepted_vector_top_k),
                weights["vector"] * self.accepted_paper_weight,
            )
        return route_results

    def _build_rich_query(self, query_text: str, paper_profile: PaperProfile) -> str:
        """构建丰富的检索 query，整合 paper_profile 的所有关键字段"""
        core_terms = " ".join([
            " ".join(paper_profile.keywords),
            " ".join(paper_profile.techniques),
            " ".join(paper_profile.application_domain),
            " ".join(paper_profile.datasets),
        ])
        parts = [
            query_text,  # 论文标题+摘要
            query_text,  # 重复一次以提升权重
            paper_profile.title,
            paper_profile.title,
            core_terms,
            core_terms,
            " ".join(paper_profile.keywords),
            " ".join(paper_profile.techniques),
            " ".join(paper_profile.application_domain),
            " ".join(paper_profile.keywords),      # 再次重复
            " ".join(paper_profile.techniques),    # 再次重复
            " ".join(paper_profile.datasets),
            " ".join(paper_profile.evaluation_metrics),
            paper_profile.novelty_type or "",
        ]
        return " ".join(parts)

    def _identity_anchor_search(
        self,
        query_text: str,
        paper_profile: PaperProfile,
        top_k: int = 30,
    ) -> List[Tuple[Journal, float]]:
        """使用期刊自身身份文本做 deterministic anchor 召回。"""
        query_terms = self._token_set(" ".join([
            query_text,
            paper_profile.title,
            " ".join(paper_profile.keywords),
            " ".join(paper_profile.techniques),
            " ".join(paper_profile.application_domain),
        ]))
        if not query_terms:
            return []

        scored: List[Tuple[Journal, float]] = []
        for journal in self.store.journals:
            identity_text = " ".join([
                journal.journal_name,
                journal.scope_text,
                " ".join(journal.keywords),
                " ".join(journal.subject_tags),
                journal.journal_profile,
            ])
            identity_terms = self._token_set(identity_text)
            if not identity_terms:
                continue
            overlap = query_terms & identity_terms
            if not overlap:
                continue
            score = len(overlap) / max(len(query_terms), 1)
            scored.append((journal, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _token_set(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in text.replace("-", " ").replace("/", " ").split()
            if len(token) > 2
        }

    def _normalize_scores(self, results: List[Tuple[Journal, float]]) -> List[Tuple[Journal, float]]:
        """对单路结果做 min-max 归一化到 [0,1]"""
        if not results:
            return []

        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # 所有分数相同，归一化为 0
            return [(journal, 0.0) for journal, _ in results]

        normalized = [(journal, (score - min_score) / (max_score - min_score)) for journal, score in results]
        return normalized

    def _merge_route_results(
        self,
        route_results: Dict[str, Tuple[List[Tuple[Journal, float]], float]],
        top_k: int,
    ) -> Tuple[List[Journal], Dict[str, dict]]:
        """合并任意数量召回路由，并记录来源贡献。"""
        trace, sorted_all_ids = self._collect_route_trace(route_results)
        sorted_ids = sorted_all_ids[:top_k]
        for retrieval_rank, jid in enumerate(sorted_ids, start=1):
            if jid in trace:
                trace[jid]["retrieval_rank"] = retrieval_rank

        journal_map = {j.journal_id: j for j in self.store._journals}
        candidates = [journal_map[jid] for jid in sorted_ids if jid in journal_map]
        candidate_ids = {journal.journal_id for journal in candidates}
        return candidates, {
            jid: trace[jid]
            for jid in candidate_ids
            if jid in trace
        }

    def _collect_route_trace(
        self,
        route_results: Dict[str, Tuple[List[Tuple[Journal, float]], float]],
    ) -> Tuple[Dict[str, dict], List[str]]:
        """收集完整路由 trace，不裁剪候选。"""
        score_map: Dict[str, float] = {}
        trace: Dict[str, dict] = {}

        for route_name, (results, route_weight) in route_results.items():
            normalized = self._normalize_scores(results)
            for rank, ((journal, raw_score), (_, normalized_score)) in enumerate(
                zip(results, normalized),
                start=1,
            ):
                weighted_score = self._route_contribution(
                    normalized_score=normalized_score,
                    route_weight=route_weight,
                    rank=rank,
                )
                jid = journal.journal_id
                score_map[jid] = score_map.get(jid, 0.0) + weighted_score
                item = trace.setdefault(jid, {"total_score": 0.0, "routes": {}})
                item["routes"][route_name] = {
                    "rank": rank,
                    "raw_score": raw_score,
                    "normalized_score": normalized_score,
                    "weighted_score": weighted_score,
                }

        for jid, total_score in score_map.items():
            item = trace.setdefault(jid, {"routes": {}})
            routes = item.get("routes", {})
            has_scope_boundary = any(route.startswith("scope_") for route in routes)
            has_typical = any(route.startswith("typical_") for route in routes)
            boundary_bonus = 0.04 if has_scope_boundary and has_typical else 0.0
            item["base_score"] = total_score
            item["boundary_bonus"] = boundary_bonus
            item["total_score"] = total_score + boundary_bonus
            item["primary_routes"] = [
                route
                for route, _ in sorted(
                    item["routes"].items(),
                    key=lambda kv: kv[1]["weighted_score"],
                    reverse=True,
                )
            ]

        sorted_all_ids = sorted(
            trace.keys(),
            key=lambda x: trace[x].get("total_score", 0.0),
            reverse=True,
        )
        return trace, sorted_all_ids

    def _route_contribution(
        self,
        normalized_score: float,
        route_weight: float,
        rank: int,
    ) -> float:
        if self.fusion_strategy == "rrf":
            return 1.0 / (self.rrf_k + rank)
        if self.fusion_strategy == "weighted_rrf":
            return route_weight / (self.rrf_k + rank)
        return normalized_score * route_weight

    def _attach_diagnostic_trace(
        self,
        trace: Dict[str, dict],
        wide_route_results: Dict[str, Tuple[List[Tuple[Journal, float]], float]],
        diagnostic_ids: set[str],
    ) -> None:
        """把宽召回诊断信息附加到 trace，不影响真实候选排序。"""
        wide_trace, wide_sorted_ids = self._collect_route_trace(wide_route_results)
        for wide_rank, jid in enumerate(wide_sorted_ids, start=1):
            if jid not in diagnostic_ids:
                continue
            wide_item = wide_trace.get(jid, {})
            item = trace.setdefault(
                jid,
                {
                    "total_score": None,
                    "primary_routes": [],
                    "routes": {},
                },
            )
            item["wide_retrieval_rank"] = wide_rank
            item["wide_total_score"] = wide_item.get("total_score")
            item["wide_primary_routes"] = wide_item.get("primary_routes", [])
            item["wide_routes"] = wide_item.get("routes", {})

    def _use_typical_abstracts(self) -> bool:
        return self.retrieval_target in {"typical_abstracts", "semantic_anchors"}

    def _active_bm25_retriever(self):
        """兼容旧实验脚本：返回当前目标的主 BM25 召回器。"""
        if self._use_typical_abstracts() and self.typical_bm25_retriever:
            return self.typical_bm25_retriever
        return self.bm25_retriever

    def _active_embedding_retriever(self):
        """兼容旧实验脚本：返回当前目标的主向量召回器。"""
        if self._use_typical_abstracts() and self.typical_embedding_retriever:
            return self.typical_embedding_retriever
        return self.embedding_retriever

    def _merge_route_top_k(
        self,
        route_top_k: Optional[Dict[str, Dict[str, int]]],
    ) -> Dict[str, Dict[str, int]]:
        defaults = {
            "title": {"bm25": 22, "vector": 22, "text": 16, "accepted_bm25": 22, "accepted_vector": 44},
            "abstract": {"bm25": 28, "vector": 28, "text": 14, "accepted_bm25": 28, "accepted_vector": 56},
            "full": {"bm25": 32, "vector": 32, "text": 16, "accepted_bm25": 32, "accepted_vector": 64},
        }
        if not route_top_k:
            return defaults
        merged = {mode: cfg.copy() for mode, cfg in defaults.items()}
        for mode, overrides in route_top_k.items():
            if not isinstance(overrides, dict):
                continue
            base = merged.setdefault(mode, defaults["abstract"].copy())
            for key in ("bm25", "vector", "text", "accepted_bm25", "accepted_vector"):
                if key in overrides:
                    base[key] = int(overrides[key])
        return merged

    def _route_config_for_mode(self, mode: str) -> Dict[str, int]:
        return self.route_top_k.get(mode, self.route_top_k["abstract"]).copy()

    def _weights_for_profile(self, paper_profile: PaperProfile) -> Dict[str, float]:
        if self.use_gating and self.weight_gater:
            weights = self.weight_gater.predict_weights(paper_profile)
            return {
                "bm25": float(weights.get("bm25", self.merge_weights["bm25"])),
                "vector": float(weights.get("vector", self.merge_weights["vector"])),
                "text": float(weights.get("text", self.merge_weights["text"])),
            }
        return self.merge_weights
