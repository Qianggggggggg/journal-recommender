"""论文质量评估（纯LLM）"""
import logging
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
import tenacity

from ..utils.llm import MiniMaxLLM, parse_json_response
from .paper_model import PaperInput, PaperProfile

logger = logging.getLogger(__name__)


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
        default="C",
        description="质量等级: A/B/C/D (D表示未达发表水平)"
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

    # CCF 专业领域预测
    ccf_research_area: List[str] = Field(
        default_factory=list,
        description="CCF专业领域列表(1-3个): 计算机体系结构/并行与分布计算/存储系统, 计算机网络, 网络与信息安全, 软件工程/系统软件/程序设计语言, 数据库/数据挖掘/内容检索, 计算机科学理论, 计算机图形学与多媒体, 人工智能, 人机交互与普适计算, 交叉/综合/新兴"
    )

    @staticmethod
    def _strength_to_level(strength: float) -> str:
        """将 strength 映射到 A/B/C/D"""
        if strength >= 0.75:
            return "A"
        elif strength >= 0.55:
            return "B"
        elif strength >= 0.35:
            return "C"
        else:
            return "D"  # 未达发表水平

    @staticmethod
    def _strength_to_readiness(strength: float, novelty_score: int) -> str:
        """根据 strength 和 novelty 推断准备度"""
        if strength >= 0.6 and novelty_score >= 2:
            return "Ready"
        elif strength >= 0.35:
            return "Preliminary"
        else:
            return "Needs-Revision"


class PaperQualityError(Exception):
    """论文质量评估错误（明确的业务异常）"""
    pass


class PaperQualityAssessor:
    """论文质量评估器（仅LLM，无规则降级）"""

    def __init__(self, llm: MiniMaxLLM):
        if llm is None:
            raise PaperQualityError("LLM not configured, please set minimax API key")
        self.llm = llm

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=2, max=8),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(f"[PaperQualityAssessor] Retry {retry_state.attempt_number}/3 after error..."),
    )
    def assess(
        self,
        paper_input: PaperInput,
        paper_profile: PaperProfile,
        system_prompt: str,
        user_prompt: str,
    ) -> PaperQuality:
        """评估论文质量（LLM驱动，重试3次）"""
        user_filled = user_prompt.format(
            title=paper_input.title,
            abstract=paper_input.abstract or "",
            full_text_summary=paper_input.full_text if paper_input.full_text else "",
            research_area=", ".join(paper_profile.research_area) if paper_profile.research_area else "未知",
            method_type=paper_profile.method_type,
            keywords=", ".join(paper_profile.keywords) if paper_profile.keywords else "无",
            techniques=", ".join(paper_profile.techniques) if paper_profile.techniques else "无",
            datasets=", ".join(paper_profile.datasets) if paper_profile.datasets else "无",
            evaluation_metrics=", ".join(paper_profile.evaluation_metrics) if paper_profile.evaluation_metrics else "无",
            novelty_type=paper_profile.novelty_type or "未知",
        )

        try:
            response = self.llm.chat(system_prompt, user_filled)
        except Exception as e:
            raise PaperQualityError(f"LLM调用失败: {e}")

        # 解析 JSON 响应
        data = parse_json_response(response.content)
        if data:

            # 提取各维度分数 (0~3)
            novelty_score = data.get("novelty_score", 2)
            rigor_score = data.get("rigor_score", 1)
            reproducibility_score = data.get("reproducibility_score", 1)
            significance_score = data.get("significance_score", 1)
            clarity_score = data.get("clarity_score", 1)

            # 计算 paper_strength (归一化到 0~1)
            raw_strength = (
                novelty_score * 0.35 +
                rigor_score * 0.25 +
                reproducibility_score * 0.15 +
                significance_score * 0.15 +
                clarity_score * 0.10
            )
            paper_strength = min(raw_strength / 3.0, 1.0)

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
            readiness = PaperQuality._strength_to_readiness(paper_strength, novelty_score)

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
                ccf_research_area=data.get("ccf_research_area", []),
            )

        raise PaperQualityError(f"LLM响应格式错误，无法解析: {response.content}")