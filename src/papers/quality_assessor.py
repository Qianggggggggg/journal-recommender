"""论文质量评估"""
from typing import Optional, List
from pydantic import BaseModel
import re

from ..utils.llm import MiniMaxLLM
from .paper_model import PaperInput, PaperProfile


class PaperQuality(BaseModel):
    """论文质量评估结果"""
    level: str  # Q1/Q2/Q3/Q4
    confidence: float
    reasons: List[str]


class PaperQualityAssessor:
    """论文质量评估器"""

    def __init__(self, llm: Optional[MiniMaxLLM] = None):
        self.llm = llm

    def assess(
        self,
        paper_input: PaperInput,
        paper_profile: PaperProfile,
        system_prompt: str,
        user_prompt: str,
    ) -> PaperQuality:
        """评估论文质量"""
        if self.llm is None:
            return self._assess_by_rules(paper_input, paper_profile)

        try:
            return self._assess_by_llm(paper_input, paper_profile, system_prompt, user_prompt)
        except Exception:
            return self._assess_by_rules(paper_input, paper_profile)

    def _assess_by_llm(
        self,
        paper_input: PaperInput,
        paper_profile: PaperProfile,
        system_prompt: str,
        user_prompt: str,
    ) -> PaperQuality:
        """LLM 评估"""
        user_filled = user_prompt.format(
            title=paper_input.title,
            abstract=paper_input.abstract or "",
            full_text_summary=paper_input.full_text[:500] if paper_input.full_text else "",
            research_area=", ".join(paper_profile.research_area) if paper_profile.research_area else "未知",
            method_type=paper_profile.method_type,
            keywords=", ".join(paper_profile.keywords) if paper_profile.keywords else "无",
            techniques=", ".join(paper_profile.techniques) if paper_profile.techniques else "无",
            datasets=", ".join(paper_profile.datasets) if paper_profile.datasets else "无",
            evaluation_metrics=", ".join(paper_profile.evaluation_metrics) if paper_profile.evaluation_metrics else "无",
            novelty_type=paper_profile.novelty_type or "未知",
        )

        response = self.llm.chat(system_prompt, user_filled)

        import json
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return PaperQuality(
                level=data.get("quality_level", "Q3"),
                confidence=data.get("confidence", 0.5),
                reasons=data.get("reasons", []),
            )

        raise ValueError(f"Failed to parse LLM response: {response.content}")

    def _assess_by_rules(
        self,
        paper_input: PaperInput,
        paper_profile: PaperProfile,
    ) -> PaperQuality:
        """规则降级评估"""
        score = 0.0
        reasons = []

        # 创新性
        novelty_type = getattr(paper_profile, 'novelty_type', '') or ''
        if novelty_type == "new_method":
            score += 2.0
            reasons.append("新方法创新")
        elif novelty_type == "benchmark":
            score += 1.5
            reasons.append("基准贡献")
        elif novelty_type == "performance":
            score += 1.2
            reasons.append("性能提升")
        elif novelty_type == "new_application":
            score += 1.0
            reasons.append("新应用场景")
        else:
            score += 0.5
            reasons.append("方法类工作")

        # 数据集数量
        datasets = getattr(paper_profile, 'datasets', []) or []
        if len(datasets) >= 3:
            score += 1.5
            reasons.append(f"多数据集验证({len(datasets)}个)")
        elif len(datasets) >= 1:
            score += 0.8
            reasons.append(f"数据集验证({len(datasets)}个)")
        else:
            reasons.append("无数据集验证")

        # 评估指标
        metrics = getattr(paper_profile, 'evaluation_metrics', []) or []
        if len(metrics) >= 3:
            score += 1.0
            reasons.append(f"多指标评估({len(metrics)}项)")
        elif len(metrics) >= 1:
            score += 0.5

        # 技术数量（复杂度代理）
        techniques = getattr(paper_profile, 'techniques', []) or []
        if len(techniques) >= 3:
            score += 0.8
            reasons.append("多技术融合")

        # 摘要长度（论文完整度代理）
        abstract_len = len(paper_input.abstract) if paper_input.abstract else 0
        if abstract_len > 300:
            score += 0.5
            reasons.append("摘要完整")

        # 全文模式额外加分
        if paper_input.full_text and len(paper_input.full_text) > 2000:
            score += 1.0
            reasons.append("全文完整")

        # 映射到等级
        if score >= 6.0:
            level = "Q1"
        elif score >= 4.0:
            level = "Q2"
        elif score >= 2.0:
            level = "Q3"
        else:
            level = "Q4"

        confidence = min(score / 8.0, 1.0) if score > 0 else 0.3

        return PaperQuality(level=level, confidence=confidence, reasons=reasons)