"""规则打分（阶段一）"""
from typing import Dict, List, Tuple, Optional

from rank_bm25 import BM25Plus

from ..journals.journal_model import Journal
from ..papers.paper_model import PaperProfile


class RuleScorer:
    """规则打分器"""

    DEFAULT_WEIGHTS = {
            # 文本匹配特征（新增/调整）
            "bm25_title_scope": 3.0,    # BM25 标题-scope 相似度（归一化）
            "title_journal_name": 1.5,   # 标题词在期刊名中匹配（权重提升）
            "journal_name_keyword": 1.0,  # 期刊名关键词与论文 keywords/techniques 匹配（新增）
            "technique_match": 2.0,      # 技术词重叠
            "keyword_overlap": 1.5,     # 关键词重叠
            # 已有特征（保留）
            "method_type_match": 0,  # 期刊 target_paper_type 全为空，暂不启用
            "paper_type_match": 1.0,
            "dataset_match": 1.0,
            "metric_match": 0.8,
            "novelty_match": 0.7,
            "oa_preference_match": 0.3,
            "scope_boundary_evidence": 0.35,
            "typical_scope_synergy": 0.08,
            "typical_only_penalty": 0.05,
            "retrieval_rank_prior": 0.0,
            "strong_scope_rank_bonus": 0.0,
            "strong_typical_rank_bonus": 0.0,
            "scope_typical_confirm_bonus": 0.0,
            "multi_route_bonus": 0.0,
            # 领域仲裁信号（权重为0，不参与计分，仅作为理由传递给LLM）
            "research_area_match": 0.0,
            "ccf_research_area_match": 0.0,
    }

    def __init__(
        self,
        journals: Optional[List[Journal]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        # 权重配置
        self.weights = self.DEFAULT_WEIGHTS.copy()
        if weights:
            self.weights.update(weights)

        # 预建 BM25 索引（期刊 scope）
        self._bm25_index: Optional[BM25Plus] = None
        self._journal_scopeTexts: List[str] = []
        self._max_bm25_score: float = 1.0  # 用于归一化

        if journals:
            self._build_bm25_index(journals)

    def _build_bm25_index(self, journals: List[Journal]):
        """预建期刊 scope 的 BM25 索引"""
        self._journal_scopeTexts = [j.scope_text for j in journals]
        if self._journal_scopeTexts:
            # 使用 BM25Plus（更鲁棒，对长文本友好）
            tokenized_corpus = [text.split() for text in self._journal_scopeTexts]
            self._bm25_index = BM25Plus(tokenized_corpus)
            # 预计算一个基准分数用于归一化（使用 "machine learning" 作为查询）
            dummy_query = "machine learning deep neural network".split()
            dummy_scores = self._bm25_index.get_scores(dummy_query)
            self._max_bm25_score = max(dummy_scores) if max(dummy_scores) > 0 else 1.0

    def _compute_bm25_title_scope(self, paper_profile: PaperProfile) -> List[float]:
        """计算论文标题（+关键词）与所有期刊 scope 的 BM25 分数"""
        if not self._bm25_index or not self._journal_scopeTexts:
            return [0.0] * len(self._journal_scopeTexts) if self._journal_scopeTexts else []

        # 构建查询文本：标题 + 关键词
        query_parts = [paper_profile.title]
        if paper_profile.keywords:
            query_parts.extend(paper_profile.keywords)
        query = " ".join(query_parts)
        tokenized_query = query.split()

        scores = self._bm25_index.get_scores(tokenized_query)
        # 归一化到 0~1
        normalized = []
        for s in scores:
            normalized.append(s / self._max_bm25_score if self._max_bm25_score > 0 else 0.0)
        return normalized

    def _compute_keyword_overlap(self, text1: str, text2: str) -> float:
        """计算关键词重叠度（0-1）"""
        words1 = set(w.lower() for w in text1.split() if len(w) > 2)
        words2 = set(w.lower() for w in text2.split() if len(w) > 2)
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        return len(intersection) / max(len(words1), len(words2))

    def _compute_journal_keyword_overlap(self, paper_profile: PaperProfile, journal: Journal) -> float:
        """计算论文关键词与期刊 scope 的重叠度"""
        paper_text = " ".join([
            paper_profile.title,
            paper_profile.abstract,
            " ".join(paper_profile.keywords),
            " ".join(paper_profile.techniques),
        ])
        journal_text = " ".join([
            journal.scope_text,
            " ".join(journal.keywords),
            journal.journal_name,
        ])
        return self._compute_keyword_overlap(paper_text, journal_text)

    def _compute_title_journal_name_match(self, paper_profile: PaperProfile, journal: Journal) -> float:
        """标题词与期刊名称的匹配（长度>1的词，排除停用词）"""
        stop_words = {"the", "a", "an", "of", "and", "in", "on", "for", "to", "with", "by", "from", "is", "as", "at"}
        title_words = set(w.lower() for w in paper_profile.title.split() if len(w) > 1 and w.lower() not in stop_words)
        journal_name_words = set(w.lower() for w in journal.journal_name.split() if len(w) > 1 and w.lower() not in stop_words)
        if not title_words or not journal_name_words:
            return 0.0
        intersection = title_words & journal_name_words
        return 1.0 if intersection else 0.0

    def _compute_journal_name_keyword_match(self, paper_profile: PaperProfile, journal: Journal) -> float:
        """期刊名关键词与论文 keywords/techniques 的匹配"""
        # 停用词
        stop_words = {"the", "a", "an", "of", "and", "in", "on", "for", "to", "with", "by", "from", "is", "as", "at", "journal", "transactions", "ieee", "acm", "international", "proceedings"}
        journal_name_words = set(w.lower() for w in journal.journal_name.split() if len(w) > 1 and w.lower() not in stop_words)
        if not journal_name_words:
            return 0.0

        # 合并论文的 keywords 和 techniques
        paper_terms = set()
        if paper_profile.keywords:
            paper_terms.update(w.lower() for w in paper_profile.keywords)
        if paper_profile.techniques:
            paper_terms.update(w.lower() for w in paper_profile.techniques)

        if not paper_terms:
            return 0.0

        # 匹配：期刊名词中的词是否出现在论文关键词/技术词中
        overlap = journal_name_words & paper_terms
        return 1.0 if overlap else 0.0

    def _retrieval_strengths(self, trace: Optional[dict]) -> Tuple[float, float, bool, bool]:
        """返回 scope 边界强度与 typical 扩展强度。"""
        if not trace:
            return 0.0, 0.0, False, False

        routes = trace.get("routes", {})
        scope_scores = [
            float(data.get("weighted_score") or 0.0)
            for route, data in routes.items()
            if route.startswith("scope_")
        ]
        typical_scores = []
        for route, data in routes.items():
            score = float(data.get("weighted_score") or 0.0)
            if route.startswith("typical_"):
                typical_scores.append(score)
            elif route == "identity_anchor":
                # identity_anchor 作为补召回证据，强度略低于 typical 摘要语义。
                #
                # Note (2026-06-18, audit finding #6): identity_anchor has
                # THREE distinct meanings across modules — keep them in sync
                # if you tune any one:
                #   1. candidate_generator.py  : independent route, weight 0.10
                #      (``identity_anchor_weight``) in fusion
                #   2. rule_scorer.py here     : merged into ``typical_strength``
                #      with 0.5 multiplier (this site)
                #   3. feature_builder.py:236   : binary ``has_identity_anchor``
                #      feature for LTR (not the score)
                # This is concept leakage — modifying the route weight in
                # candidate_generator does NOT change LTR feature semantics.
                typical_scores.append(score * 0.5)
        has_scope = bool(scope_scores)
        has_typical = bool(typical_scores)

        # 单路结果只有一个候选时 min-max 会归零，保留“出现过”的边界证据。
        scope_strength = sum(scope_scores) if has_scope else 0.0
        typical_strength = sum(typical_scores) if has_typical else 0.0
        if has_scope and scope_strength == 0.0:
            scope_strength = 0.03
        if has_typical and typical_strength == 0.0:
            typical_strength = 0.02

        return scope_strength, typical_strength, has_scope, has_typical

    def _apply_retrieval_evidence(
        self,
        score: float,
        reasons: List[str],
        trace: Optional[dict],
    ) -> Tuple[float, List[str]]:
        """把召回证据作为软信号带入规则排序。"""
        scope_strength, typical_strength, has_scope, has_typical = self._retrieval_strengths(trace)
        adjusted_score = score
        adjusted_reasons = reasons.copy()
        routes = trace.get("routes", {}) if trace else {}

        if has_scope:
            bonus = min(
                self.weights["scope_boundary_evidence"],
                scope_strength * self.weights["scope_boundary_evidence"],
            )
            adjusted_score += bonus
            adjusted_reasons.append("期刊范围文本提供边界匹配证据")

        if has_scope and has_typical:
            bonus = min(
                self.weights["typical_scope_synergy"],
                typical_strength * self.weights["typical_scope_synergy"],
            )
            adjusted_score += bonus
            adjusted_reasons.append("补充语义证据与期刊范围互相支持")
        elif has_typical and not has_scope:
            penalty = min(
                self.weights["typical_only_penalty"],
                max(typical_strength, 0.02) * self.weights["typical_only_penalty"],
            )
            adjusted_score -= penalty
            adjusted_reasons.append("仅有补充语义证据，缺少期刊范围边界支撑")

        retrieval_rank = trace.get("retrieval_rank") if trace else None
        rank_prior_weight = float(self.weights.get("retrieval_rank_prior", 0.0) or 0.0)
        if retrieval_rank and rank_prior_weight > 0:
            rank_bonus = rank_prior_weight * max(0.0, (51 - int(retrieval_rank)) / 50)
            adjusted_score += rank_bonus
            adjusted_reasons.append(f"粗排排名证据: top{int(retrieval_rank)}")

        scope_ranks = [
            int(data.get("rank"))
            for route, data in routes.items()
            if route.startswith("scope_") and data.get("rank")
        ]
        strong_scope_weight = float(self.weights.get("strong_scope_rank_bonus", 0.0) or 0.0)
        if scope_ranks and strong_scope_weight > 0:
            best_scope_rank = min(scope_ranks)
            if best_scope_rank <= 20:
                scope_rank_bonus = strong_scope_weight * max(0.0, (21 - best_scope_rank) / 20)
                adjusted_score += scope_rank_bonus
                adjusted_reasons.append(f"强scope召回证据: top{best_scope_rank}")

        typical_ranks = [
            int(data.get("rank"))
            for route, data in routes.items()
            if route.startswith("typical_") and data.get("rank")
        ]
        strong_typical_weight = float(self.weights.get("strong_typical_rank_bonus", 0.0) or 0.0)
        if has_scope and typical_ranks and strong_typical_weight > 0:
            best_typical_rank = min(typical_ranks)
            if best_typical_rank <= 10:
                typical_rank_bonus = strong_typical_weight * max(0.0, (11 - best_typical_rank) / 10)
                adjusted_score += typical_rank_bonus
                adjusted_reasons.append(f"强典型摘要召回证据: top{best_typical_rank}")

        confirm_weight = float(self.weights.get("scope_typical_confirm_bonus", 0.0) or 0.0)
        if has_scope and has_typical and confirm_weight > 0:
            confirm_strength = min(1.0, 0.5 + scope_strength + typical_strength)
            adjusted_score += confirm_weight * confirm_strength
            adjusted_reasons.append("scope与典型摘要召回互相确认")

        multi_route_weight = float(self.weights.get("multi_route_bonus", 0.0) or 0.0)
        if len(routes) >= 3 and multi_route_weight > 0:
            route_bonus = multi_route_weight * min(1.0, (len(routes) - 2) / 3)
            adjusted_score += route_bonus
            adjusted_reasons.append(f"多路召回一致: {len(routes)}条证据")

        return adjusted_score, adjusted_reasons

    def score(
        self, journal: Journal, paper_profile: PaperProfile, oa_preference: str = "any"
    ) -> Tuple[float, List[str]]:
        """计算规则分数"""
        score = 0.0
        reasons = []

        # 预计算 BM25 标题-scope（只计算一次）
        journal_idx = self._journal_scopeTexts.index(journal.scope_text) if journal.scope_text in self._journal_scopeTexts else -1
        bm25_scores = self._compute_bm25_title_scope(paper_profile)
        bm25_score = bm25_scores[journal_idx] if journal_idx >= 0 else 0.0

        # BM25 标题-scope 匹配
        if bm25_score > 0.1:
            score += self.weights["bm25_title_scope"] * bm25_score
            reasons.append(f"标题-领域BM25匹配度: {bm25_score:.2f}")

        # 标题-期刊名匹配
        title_name_match = self._compute_title_journal_name_match(paper_profile, journal)
        if title_name_match > 0:
            score += self.weights["title_journal_name"]
            reasons.append("标题词命中期刊名")

        # 期刊名关键词与论文 keywords/techniques 匹配（新增）
        jn_keyword_match = self._compute_journal_name_keyword_match(paper_profile, journal)
        if jn_keyword_match > 0:
            score += self.weights["journal_name_keyword"]
            reasons.append("期刊名专有词命中论文关键词/技术词")

        # 具体技术匹配
        if paper_profile.techniques:
            tech_overlap = self._compute_keyword_overlap(
                " ".join(paper_profile.techniques),
                journal.scope_text
            )
            if tech_overlap > 0.2:
                score += self.weights["technique_match"] * tech_overlap
                matched_techs = [t for t in paper_profile.techniques
                               if t.lower() in journal.scope_text.lower()]
                if matched_techs:
                    reasons.append(f"技术契合: {', '.join(matched_techs[:3])} (重叠度: {tech_overlap:.2f})")

        # 方法类型匹配（需要 journal.target_paper_type 非空）
        if paper_profile.method_type and journal.target_paper_type:
            if paper_profile.method_type in journal.target_paper_type:
                score += self.weights["method_type_match"]
                reasons.append(f"方法类型匹配: {paper_profile.method_type}")

        # 论文类型匹配
        if paper_profile.paper_type:
            if journal.scope_text.lower().find(paper_profile.paper_type) >= 0:
                score += self.weights["paper_type_match"]

        # 关键词重叠度
        keyword_overlap = self._compute_journal_keyword_overlap(paper_profile, journal)
        if keyword_overlap > 0.1:
            score += self.weights["keyword_overlap"] * keyword_overlap
            reasons.append(f"关键词重叠度: {keyword_overlap:.2f}")

        # 数据集匹配（专项）
        if paper_profile.datasets:
            known_datasets = {
                "pubmed": ["pubmed", "biomedical", "生物医学"],
                "imagenet": ["imagenet", "image net"],
                "coco": ["coco", "ms coco", "common objects in context"],
                "mnist": ["mnist", "digit recognition"],
                "wikitext": ["wikitext", "wikipedia", "wiki"],
                "glue": ["glue", "glue benchmark"],
                "squad": ["squad", "question answering"],
                "arxiv": ["arxiv", "cs.", "computer science"],
                "github": ["github", "code generation", "program synthesis"],
                "freebase": ["freebase", "knowledge graph"],
                "dbpedia": ["dbpedia", "knowledge base"],
                "wikidata": ["wikidata", "knowledge graph"],
                "wordnet": ["wordnet", "lexical"],
                "voc": ["voc", "pascal voc", "object detection"],
                "visual_genome": ["visual genome", "scene graph"],
                "flickr": ["flickr", "image caption"],
                "sst": ["sst", "sentiment", "情感分析"],
                "snli": ["snli", "natural language inference", "entailment"],
                "multinli": ["multinli", "multi-genre nli"],
                "llm": ["llm", "large language model", "language model"],
                "rag": ["rag", "retrieval-augmented", "retrieval augmented"],
                "kg": ["knowledge graph", "knowledge base"],
                "ner": ["ner", "named entity recognition", "命名实体"],
                "relation extraction": ["relation extraction", "relex"],
                "citation network": ["citation network", "bibliographic", "co-citation"],
            }
            matched_datasets = []
            scope_lower = journal.scope_text.lower()
            for ds in paper_profile.datasets:
                ds_lower = ds.lower()
                if ds_lower in scope_lower:
                    matched_datasets.append(ds)
                else:
                    for known, aliases in known_datasets.items():
                        if ds_lower in aliases or any(alias in ds_lower for alias in aliases):
                            if any(alias in scope_lower for alias in aliases):
                                matched_datasets.append(ds)
                                break

            if matched_datasets:
                score += self.weights["dataset_match"]
                reasons.append(f"数据集匹配: {', '.join(matched_datasets[:3])}")
            else:
                dataset_overlap = self._compute_keyword_overlap(
                    " ".join(paper_profile.datasets), journal.scope_text
                )
                if dataset_overlap > 0.1:
                    score += self.weights["dataset_match"] * dataset_overlap * 0.5

        # 评估指标匹配（专项）
        if paper_profile.evaluation_metrics:
            known_metrics = {
                "accuracy": ["accuracy", "acc"],
                "f1": ["f1", "f1-score", "f1 score"],
                "map": ["map", "mean average precision", "mAP"],
                "mrr": ["mrr", "mean reciprocal rank"],
                "ndcg": ["ndcg", "normalized dcg"],
                "bleu": ["bleu", "bilingual evaluation understudy"],
                "rouge": ["rouge", "recall-oriented understudy"],
                "perplexity": ["perplexity", "ppl"],
                "latency": ["latency", "inference time", "response time"],
                "throughput": ["throughput", "tokens per second", "tps"],
                "hit rate": ["hit rate", "hits@k"],
                "auc": ["auc", "area under curve"],
                "recall": ["recall", "sensitivity"],
                "precision": ["precision"],
                "em": ["em", "exact match"],
                "bertscore": ["bertscore", "bert score"],
            }
            matched_metrics = []
            scope_lower = journal.scope_text.lower()
            for metric in paper_profile.evaluation_metrics:
                metric_lower = metric.lower()
                if metric_lower in scope_lower:
                    matched_metrics.append(metric)
                else:
                    for known, aliases in known_metrics.items():
                        if any(alias in metric_lower for alias in aliases):
                            if any(alias in scope_lower for alias in aliases):
                                matched_metrics.append(metric)
                                break

            if matched_metrics:
                score += self.weights["metric_match"]
                reasons.append(f"评估指标: {', '.join(matched_metrics[:3])}")

        # 创新类型匹配（支持中英文枚举）
        if paper_profile.novelty_type:
            novelty_keywords = {
                "new_method": ["novel", "new method", "新的方法", "创新方法"],
                "new_application": ["application", "应用", "场景"],
                "benchmark": ["benchmark", "基准", "dataset", "数据集"],
                "performance": ["performance", "improvement", "提升", "性能"],
                "efficiency": ["efficiency", "fast", "efficient", "高效", "加速"],
            }
            # 标准化映射（中英文 -> 英文）
            novelty_normalize = {
                "新方法": "new_method", "新应用": "new_application", "新基准": "benchmark",
                "性能提升": "performance", "效率优化": "efficiency",
            }
            normalized_type = novelty_normalize.get(paper_profile.novelty_type, paper_profile.novelty_type)
            if normalized_type in novelty_keywords:
                for kw in novelty_keywords[normalized_type]:
                    if kw.lower() in journal.scope_text.lower():
                        score += self.weights["novelty_match"]
                        reasons.append(f"创新类型契合: {normalized_type}")
                        break

        # 领域仲裁信号：默认只作理由；配置权重后作为正向软信号，不命中不扣分。
        matched_area_labels = []
        if paper_profile.research_area and journal.subject_tags:
            matched_area_labels.extend(
                ra for ra in paper_profile.research_area if ra in journal.subject_tags
            )
        if paper_profile.ccf_research_area and journal.subject_tags:
            matched_area_labels.extend(
                ra for ra in paper_profile.ccf_research_area if ra in journal.subject_tags
            )
        matched_area_labels = list(dict.fromkeys(matched_area_labels))
        if matched_area_labels:
            area_weight = max(
                float(self.weights.get("research_area_match", 0.0) or 0.0),
                float(self.weights.get("ccf_research_area_match", 0.0) or 0.0),
            )
            if area_weight:
                score += area_weight
            reasons.append(f"领域标签对齐: {', '.join(matched_area_labels)}")

        # OA 偏好匹配
        if oa_preference != "any":
            if (oa_preference == "full_oa" and journal.oa_type == "full_oa") or \
               (oa_preference == "hybrid" and journal.oa_type in ["full_oa", "hybrid"]):
                score += self.weights["oa_preference_match"]
                reasons.append(f"OA类型匹配: {journal.oa_type}")

        return score, reasons

    def rank(
        self,
        journals: List[Journal],
        paper_profile: PaperProfile,
        oa_preference: str = "any",
        top_k: int = 10,
        retrieval_trace: Optional[Dict[str, dict]] = None,
    ) -> List[Tuple[Journal, float, List[str]]]:
        """排序候选期刊"""
        scored = []
        for journal in journals:
            score, reasons = self.score(journal, paper_profile, oa_preference)
            if retrieval_trace is not None:
                score, reasons = self._apply_retrieval_evidence(
                    score,
                    reasons,
                    retrieval_trace.get(journal.journal_id),
                )
            scored.append((journal, score, reasons))

        # 按分数排序
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
