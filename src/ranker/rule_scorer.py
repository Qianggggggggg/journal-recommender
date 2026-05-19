"""规则打分（阶段一）"""
from typing import List, Tuple, Set

from ..journals.journal_model import Journal
from ..papers.paper_model import PaperProfile


class RuleScorer:
    """规则打分器"""

    def __init__(self):
        # 权重配置
        self.weights = {
            "research_area_match": 2.0,
            "technique_match": 1.8,
            "method_type_match": 1.5,
            "paper_type_match": 1.0,
            "quartile_q1": 1.0,    # Q1 加分
            "quartile_q2": 0.6,    # Q2 加分
            "quartile_q3": 0.2,    # Q3 加分
            "oa_preference_match": 0.3,
            "keyword_overlap": 1.2,
            "dataset_match": 1.0,
            "metric_match": 0.8,
            "novelty_match": 0.7,
        }

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

    def _get_quartile_weight(self, quartile: str) -> int:
        """获取分区权重（1-4）"""
        weights = {"Q1": 4, "Q2": 3, "Q3": 2, "Q4": 1}
        return weights.get(quartile or "", 2)  # 默认 Q3

    def _get_quality_level_weight(self, quality_level: str) -> int:
        """获取论文质量等级权重（1-4）"""
        weights = {"Q1": 4, "Q2": 3, "Q3": 2, "Q4": 1}
        return weights.get(quality_level or "", 2)

    def score(
        self, journal: Journal, paper_profile: PaperProfile, oa_preference: str = "any"
    ) -> Tuple[float, List[str]]:
        """计算规则分数"""
        score = 0.0
        reasons = []

        # 研究领域匹配
        if paper_profile.research_area:
            matched_areas = [a for a in paper_profile.research_area if a in journal.subject_tags]
            if matched_areas:
                score += self.weights["research_area_match"]
                reasons.append(f"研究领域匹配: {', '.join(matched_areas)}")

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

        # 方法类型匹配
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
                # 通用数据集
                "pubmed": ["pubmed", "biomedical", "生物医学"],
                "imagenet": ["imagenet", "image net"],
                "coco": ["coco", "ms coco", "common objects in context"],
                "mnist": ["mnist", "digit recognition"],
                "wikitext": ["wikitext", "wikipedia", "wiki"],
                "glue": ["glue", "glue benchmark"],
                "squad": ["squad", "question answering"],
                "arxiv": ["arxiv", "cs.", "computer science"],
                "github": ["github", "code generation", "program synthesis"],
                # 知识图谱
                "freebase": ["freebase", "knowledge graph"],
                "dbpedia": ["dbpedia", "knowledge base"],
                "wikidata": ["wikidata", "knowledge graph"],
                "wordnet": ["wordnet", "lexical"],
                # 视觉
                "voc": ["voc", "pascal voc", "object detection"],
                "visual_genome": ["visual genome", "scene graph"],
                "flickr": ["flickr", "image caption"],
                # NLP
                "sst": ["sst", "sentiment", "情感分析"],
                "snli": ["snli", "natural language inference", "entailment"],
                "multinli": ["multinli", "multi-genre nli"],
                "llm": ["llm", "large language model", "language model"],
                "rag": ["rag", "retrieval-augmented", "retrieval augmented"],
                "kg": ["knowledge graph", "knowledge base"],
                "ner": ["ner", "named entity recognition", "命名实体"],
                "relation extraction": ["relation extraction", "relex"],
                # 图/网络
                "citation network": ["citation network", "bibliographic", "co-citation"],
                "arxiv": ["arxiv", "citation", "academic"],
            }
            matched_datasets = []
            scope_lower = journal.scope_text.lower()
            for ds in paper_profile.datasets:
                ds_lower = ds.lower()
                # 精确匹配
                if ds_lower in scope_lower:
                    matched_datasets.append(ds)
                else:
                    # 别名匹配
                    for known, aliases in known_datasets.items():
                        if ds_lower in aliases or any(alias in ds_lower for alias in aliases):
                            if any(alias in scope_lower for alias in aliases):
                                matched_datasets.append(ds)
                                break

            if matched_datasets:
                score += self.weights["dataset_match"]
                reasons.append(f"数据集匹配: {', '.join(matched_datasets[:3])}")
            else:
                # 回退：通用重叠度
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

        # 创新类型匹配
        if paper_profile.novelty_type:
            novelty_keywords = {
                "new_method": ["novel", "new method", "新的方法", "创新方法"],
                "new_application": ["application", "应用", "场景"],
                "benchmark": ["benchmark", "基准", "dataset", "数据集"],
                "performance": ["performance", "improvement", "提升", "性能"],
                "efficiency": ["efficiency", "fast", "efficient", "高效", "加速"],
            }
            if paper_profile.novelty_type in novelty_keywords:
                for kw in novelty_keywords[paper_profile.novelty_type]:
                    if kw.lower() in journal.scope_text.lower():
                        score += self.weights["novelty_match"]
                        reasons.append(f"创新类型契合: {paper_profile.novelty_type}")
                        break

        # 分区加分（差分级）
        if journal.quartile == "Q1":
            score += self.weights["quartile_q1"]
            reasons.append(f"高分区期刊: Q1")
        elif journal.quartile == "Q2":
            score += self.weights["quartile_q2"]
            reasons.append(f"中分区期刊: Q2")
        elif journal.quartile == "Q3":
            score += self.weights["quartile_q3"]
            reasons.append(f"较低分区期刊: Q3")

        # 影响因子加分（归一化，值域 [0, 0.5]）
        if journal.impact_like_score and journal.impact_like_score > 0:
            # 假设影响因子范围 0-10，归一化到 0.5
            impact_bonus = min(journal.impact_like_score / 10.0 * 0.5, 0.5)
            score += impact_bonus

        # OA 偏好匹配
        if oa_preference != "any":
            if (oa_preference == "full_oa" and journal.oa_type == "full_oa") or \
               (oa_preference == "hybrid" and journal.oa_type in ["full_oa", "hybrid"]):
                score += self.weights["oa_preference_match"]
                reasons.append(f"OA类型匹配: {journal.oa_type}")

        # 论文质量与期刊分区匹配约束
        if paper_profile.quality_level:
            qlevel = self._get_quartile_weight(journal.quartile)
            plevel = self._get_quality_level_weight(paper_profile.quality_level)
            gap = qlevel - plevel
            if gap > 1:
                # 期刊分区高于论文质量太多，大幅降权
                score *= 0.5
                reasons.append(f"质量匹配: 期刊分区({journal.quartile})高于论文质量({paper_profile.quality_level})，匹配度下调")
            elif gap == 1:
                score *= 0.8
                reasons.append(f"质量匹配: 期刊分区({journal.quartile})略高于论文质量({paper_profile.quality_level})")

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
