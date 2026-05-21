"""推荐流程编排"""
import yaml
from typing import List, Optional, Dict, Any, Tuple

from ..utils.text import quality_adjustment_factor
from ..journals.journal_model import Journal, JournalMatch
from ..papers.paper_model import PaperInput, PaperProfile
from ..papers.quality_assessor import PaperQualityAssessor
from ..retriever.candidate_generator import CandidateGenerator
from ..ranker.rule_scorer import RuleScorer
from ..ranker.llm_ranker import LLMRanker, LLMRankerError


class RecommenderPipeline:
    """推荐流程编排器"""

    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        rule_scorer: RuleScorer,
        llm_ranker: Optional[LLMRanker] = None,
        quality_assessor: Optional[PaperQualityAssessor] = None,
    ):
        self.candidate_generator = candidate_generator
        self.rule_scorer = rule_scorer
        self.llm_ranker = llm_ranker
        self.quality_assessor = quality_assessor

    def recommend(
        self,
        paper_input: PaperInput,
        paper_profile: PaperProfile,
        top_k: int = 5,
        mode: str = "abstract",
        oa_preference: str = "any",
        quality_prompts: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """执行推荐流程"""
        # 0. 评估论文质量
        if self.quality_assessor and quality_prompts:
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

        # 1. 候选召回
        query_text = paper_input.title
        if paper_input.abstract:
            query_text += " " + paper_input.abstract

        candidates = self.candidate_generator.generate(
            query_text, paper_profile, top_k=50, mode=mode
        )

        if not candidates:
            return {"recommendations": [], "warning": "未找到合适的候选期刊"}

        # 2. 阶段一：规则打分（只做主题匹配，不含质量调整）
        rule_ranked = self.rule_scorer.rank(
            candidates, paper_profile, oa_preference=oa_preference, top_k=10
        )

        # 2.5 质量调整软权重（在 Pipeline 中统一应用，解耦）
        rule_ranked = self._apply_quality_adjustment(rule_ranked, paper_profile)

        # 3. 阶段二：LLM 精排（如失败则抛出明确错误，不再降级）
        rank_method = "rule"
        if self.llm_ranker:
            try:
                llm_ranked, rank_method = self.llm_ranker.rank(rule_ranked, paper_profile, top_k=top_k)
            except LLMRankerError as e:
                raise LLMRankerError(f"LLM精排失败: {e}")
        else:
            llm_ranked = [(j, s, r, 0.5) for j, s, r in rule_ranked[:top_k]]

        # 4. 构建推荐结果（直接使用 LLMRanker 输出的 reasons，不再单独调用 Explainer）
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
            "paper_profile": paper_profile,
            "mode_used": mode,
            "rank_method": rank_method,
        }

        # 标题模式加警告
        if mode == "title":
            result["warning"] = "置信度较低，建议补充摘要以获得更精确的推荐"

        return result

    def _apply_quality_adjustment(
        self,
        ranked: List[Tuple[Journal, float, List[str]]],
        paper_profile: PaperProfile,
    ) -> List[Tuple[Journal, float, List[str]]]:
        """应用质量软权重调整（解耦：质量评估结果不再直接流入 RuleScorer）"""
        if paper_profile.paper_strength is None:
            return ranked

        strength = paper_profile.paper_strength

        base_adjustment = quality_adjustment_factor(strength)

        adjusted = []
        for journal, score, reasons in ranked:
            ccf_multiplier = {"A": 1.05, "B": 1.02, "C": 1.0}.get(journal.ccf_rating, 1.0)
            adjustment = base_adjustment * ccf_multiplier

            adjusted_score = score * adjustment
            new_reasons = reasons.copy()
            if strength >= 0.75:
                new_reasons.append(f"强论文调整(+{(adjustment-1)*100:.0f}%)")
            elif strength < 0.35:
                new_reasons.append(f"弱论文调整({(adjustment-1)*100:.0f}%)")

            adjusted.append((journal, adjusted_score, new_reasons))

        # 重新排序
        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted

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