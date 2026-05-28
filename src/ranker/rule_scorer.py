"""规则打分（阶段一）"""
from typing import List, Tuple, Optional

from rank_bm25 import BM25Plus

from ..journals.journal_model import Journal
from ..papers.paper_model import PaperProfile


class RuleScorer:
    """规则打分器"""

    def __init__(self, journals: Optional[List[Journal]] = None):
        # 权重配置
        self.weights = {
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
            # 领域仲裁信号（权重为0，不参与计分，仅作为理由传递给LLM）
            "research_area_match": 0.0,
        }

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

        # 领域仲裁信号（research_area 与 subject_tags 精确匹配，不加分仅作理由）
        if paper_profile.research_area and journal.subject_tags:
            matched_areas = [ra for ra in paper_profile.research_area if ra in journal.subject_tags]
            if matched_areas:
                reasons.append(f"领域标签对齐: {', '.join(matched_areas)}")

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
    ) -> List[Tuple[Journal, float, List[str]]]:
        """排序候选期刊"""
        scored = []
        for journal in journals:
            score, reasons = self.score(journal, paper_profile, oa_preference)
            scored.append((journal, score, reasons))

        # 按分数排序
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]