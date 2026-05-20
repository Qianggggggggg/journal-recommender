"""论文质量评估"""
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
import re

from ..utils.llm import MiniMaxLLM
from .paper_model import PaperInput, PaperProfile


class PaperQuality(BaseModel):
    """论文质量评估结果（多维度）"""

    # 核心：论文本身强弱 (0~1)
    paper_strength: float = Field(description="论文本身强弱分数 0~1")

    # 投稿准备度
    readiness: str = Field(
        default="Preliminary",
        description="投稿准备度: Ready / Preliminary / Needs-Revision"
    )

    # 汇总标签（向后兼容）
    quality_level: str = Field(
        default="Q3",
        description="质量等级: Q1/Q2/Q3/Q4 (由 paper_strength 映射)"
    )

    # 置信度
    confidence: float = Field(default=0.5)

    # 评估理由
    reasons: List[str] = Field(default_factory=list)

    # 证据字段（可解释性）
    evidence: Dict[str, str] = Field(
        default_factory=dict,
        description="各维度证据: novelty/rigor/completeness/significance/clarity"
    )

    # 不确定原因
    uncertainty_reasons: List[str] = Field(default_factory=list)

    @staticmethod
    def _strength_to_level(strength: float) -> str:
        """将 strength 映射到 Q1-Q4"""
        if strength >= 0.75:
            return "Q1"
        elif strength >= 0.55:
            return "Q2"
        elif strength >= 0.35:
            return "Q3"
        else:
            return "Q4"

    @staticmethod
    def _strength_to_readiness(strength: float, novelty_score: int) -> str:
        """根据 strength 和 novelty 推断准备度"""
        if strength >= 0.6 and novelty_score >= 2:
            return "Ready"
        elif strength >= 0.35:
            return "Preliminary"
        else:
            return "Needs-Revision"


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
        """LLM 评估（新版多维度）"""
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

            # 提取各维度分数 (0~3)
            novelty_score = data.get("novelty_score", 2)
            rigor_score = data.get("rigor_score", 1)
            reproducibility_score = data.get("reproducibility_score", 1)
            significance_score = data.get("significance_score", 1)
            clarity_score = data.get("clarity_score", 1)

            # 计算 paper_strength (归一化到 0~1)
            # 各维度权重: novelty 35%, rigor 25%, reproducibility 15%, significance 15%, clarity 10%
            raw_strength = (
                novelty_score * 0.35 +
                rigor_score * 0.25 +
                reproducibility_score * 0.15 +
                significance_score * 0.15 +
                clarity_score * 0.10
            )
            paper_strength = min(raw_strength / 3.0, 1.0)  # 归一化到 0~1

            # 证据字段
            evidence = {
                "novelty": data.get("novelty_evidence", "未提供"),
                "rigor": data.get("rigor_evidence", "未提供"),
                "reproducibility": data.get("reproducibility_evidence", "未提供"),
                "significance": data.get("significance_evidence", "未提供"),
                "clarity": data.get("clarity_evidence", "未提供"),
            }

            # 不确定原因
            uncertainty_reasons = data.get("uncertainty_reasons", [])

            # 准备度
            readiness = self.PaperQuality__class__._strength_to_readiness(
                paper_strength, novelty_score
            ) if hasattr(self, 'PaperQuality__class__') else PaperQuality._strength_to_readiness(
                paper_strength, novelty_score
            )

            # 汇总等级
            quality_level = PaperQuality._strength_to_level(paper_strength)

            return PaperQuality(
                paper_strength=paper_strength,
                readiness=readiness,
                quality_level=quality_level,
                confidence=data.get("confidence", 0.5),
                reasons=data.get("reasons", []),
                evidence=evidence,
                uncertainty_reasons=uncertainty_reasons,
            )

        raise ValueError(f"Failed to parse LLM response: {response.content}")

    def _assess_by_rules(
        self,
        paper_input: PaperInput,
        paper_profile: PaperProfile,
    ) -> PaperQuality:
        """规则降级评估（新版多维度：证据驱动）"""

        # === novelty (0~3): 创新类型 + 数据集 ===
        novelty_type = getattr(paper_profile, 'novelty_type', '') or ''
        novelty_scores = {
            "new_method": 3, "benchmark": 2.5, "performance": 2,
            "new_application": 1.5, "efficiency": 1.5, "": 1
        }
        novelty_score = novelty_scores.get(novelty_type, 1)

        datasets = getattr(paper_profile, 'datasets', []) or []
        dataset_count = len(datasets)
        if dataset_count >= 3:
            dataset_score = 3
        elif dataset_count == 2:
            dataset_score = 2
        elif dataset_count == 1:
            dataset_score = 1
        else:
            dataset_score = 0

        # novelty 维度: 创新类型为主，数据集为辅
        novelty_dim = novelty_score * 0.6 + dataset_score * 0.4
        novelty_evidence = f"创新类型: {novelty_type}, 数据集: {dataset_count}个"
        if dataset_count == 0:
            novelty_evidence += " (insufficient_evidence: 未提供数据集)"

        # === rigor (0~3): 评估指标 + 技术复杂度 ===
        metrics = getattr(paper_profile, 'evaluation_metrics', []) or []
        metric_score = min(len(metrics), 3)
        techniques = getattr(paper_profile, 'techniques', []) or []
        tech_score = min(len(techniques) / 2, 2)
        rigor_dim = metric_score * 0.5 + tech_score * 0.3 + min(dataset_score, 2) * 0.2
        rigor_evidence = f"评估指标: {len(metrics)}项, 技术: {len(techniques)}个"
        if len(metrics) == 0:
            rigor_evidence += " (insufficient_evidence: 未提供评估指标)"

        # === reproducibility (0~3): 数据集 + 全文信息 ===
        # 规则：数据集数量是 reproducibility 的主要信号
        if dataset_count >= 3:
            repro_score = 3
        elif dataset_count == 2:
            repro_score = 2
        elif dataset_count == 1:
            repro_score = 1
        else:
            repro_score = 0

        full_text_len = len(paper_input.full_text) if paper_input.full_text else 0
        if full_text_len > 2000:
            repro_score = min(repro_score + 0.5, 3)

        repro_evidence = f"数据集: {dataset_count}个"
        if full_text_len > 2000:
            repro_evidence += ", 全文完整"
        if dataset_count == 0 and full_text_len < 500:
            repro_evidence += " (insufficient_evidence: 数据集和全文信息均不足)"

        # === significance (0~3): 通过 novelty_type 推断 ===
        if novelty_type in ("new_method", "benchmark"):
            significance_score = 2.5
        elif novelty_type in ("performance", "new_application"):
            significance_score = 2.0
        else:
            significance_score = 1.5
        significance_evidence = f"创新类型: {novelty_type}"
        if novelty_type == "":
            significance_evidence += " (insufficient_evidence: 未提供创新类型)"

        # === clarity (0~3): 摘要完整度 ===
        abstract_len = len(paper_input.abstract) if paper_input.abstract else 0
        if abstract_len > 300:
            clarity_dim = 2.5
            clarity_evidence = "摘要完整（>300字）"
        elif abstract_len > 100:
            clarity_dim = 2.0
            clarity_evidence = "摘要较完整（>100字）"
        elif abstract_len > 0:
            clarity_dim = 1.0
            clarity_evidence = "摘要较短"
        else:
            clarity_dim = 0
            clarity_evidence = "insufficient_evidence: 未提供摘要"

        # === 综合 paper_strength ===
        raw_strength = (
            novelty_dim * 0.35 +
            rigor_dim * 0.25 +
            repro_score * 0.15 +
            significance_score * 0.15 +
            clarity_dim * 0.10
        )
        paper_strength = min(raw_strength / 3.0, 1.0)

        # === 准备度 ===
        readiness = PaperQuality._strength_to_readiness(paper_strength, int(novelty_score))

        # === 汇总等级 ===
        quality_level = PaperQuality._strength_to_level(paper_strength)

        # === 理由 ===
        reasons = []
        if novelty_score >= 2.5:
            reasons.append("方法创新性强")
        if dataset_count >= 2:
            reasons.append(f"多数据集验证({dataset_count}个)")
        if len(metrics) >= 3:
            reasons.append(f"多指标评估({len(metrics)}项)")
        if paper_strength >= 0.7:
            reasons.append("论文整体强度较高")
        elif paper_strength < 0.35:
            reasons.append("建议补充实验后再投高分区")

        # === 不确定原因 ===
        uncertainty_reasons = []
        if abstract_len < 100:
            uncertainty_reasons.append("摘要信息不足")
        if dataset_count == 0:
            uncertainty_reasons.append("未提供数据集信息")
        if full_text_len < 500:
            uncertainty_reasons.append("全文信息有限")

        return PaperQuality(
            paper_strength=paper_strength,
            readiness=readiness,
            quality_level=quality_level,
            confidence=min(paper_strength + 0.1, 1.0) if paper_strength > 0 else 0.3,
            reasons=reasons,
            evidence={
                "novelty": novelty_evidence,
                "rigor": rigor_evidence,
                "reproducibility": repro_evidence,
                "significance": significance_evidence,
                "clarity": clarity_evidence,
            },
            uncertainty_reasons=uncertainty_reasons,
        )