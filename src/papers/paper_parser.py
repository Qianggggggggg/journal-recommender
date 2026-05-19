"""论文解析（含降级策略）"""
import time
from typing import Optional

import tenacity

from ..utils.llm import MiniMaxLLM
from ..utils.text import clean_text, extract_keywords
from .paper_model import PaperProfile, PaperInput


class PaperParser:
    """论文解析器"""

    def __init__(self, llm: Optional[MiniMaxLLM] = None):
        self.llm = llm

    @tenacity.retry(
        wait=tenacity.wait_fixed(2),
        stop=tenacity.stop_after_attempt(2),
        reraise=True,
    )
    def parse_with_llm(self, paper_input: PaperInput, system_prompt: str, user_prompt: str) -> PaperProfile:
        """使用 LLM 解析论文"""
        if self.llm is None:
            raise ValueError("LLM not configured")

        user_filled = user_prompt.format(
            title=paper_input.title,
            abstract=paper_input.abstract or "",
            full_text_summary=paper_input.full_text[:500] if paper_input.full_text else "",
        )

        response = self.llm.chat(system_prompt, user_filled)
        # 解析 JSON 响应（简化处理）
        import json
        import re

        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return PaperProfile(
                title=paper_input.title,
                abstract=paper_input.abstract or "",
                **{k: v for k, v in data.items() if k != "title"}
            )

        raise ValueError(f"Failed to parse LLM response: {response.content}")

    def parse_with_fallback(self, paper_input: PaperInput, system_prompt: str, user_prompt: str) -> PaperProfile:
        """带降级策略的解析"""
        try:
            return self.parse_with_llm(paper_input, system_prompt, user_prompt)
        except Exception as e:
            # 降级策略 1: 使用规则 + 关键词提取
            return self._parse_by_rules(paper_input)

    def _parse_by_rules(self, paper_input: PaperInput) -> PaperProfile:
        """规则降级解析"""
        # 提取关键词
        combined_text = paper_input.title + " " + (paper_input.abstract or "")
        keywords = extract_keywords(combined_text, top_k=5)

        # 简单标签匹配
        research_area = self._match_research_area(combined_text)

        return PaperProfile(
            title=paper_input.title,
            abstract=paper_input.abstract or "",
            research_area=research_area,
            method_type=self._infer_method_type(combined_text),
            paper_type="application",
            keywords=keywords,
            novelty="",
            application_domain=[],
            difficulty_level="medium",
            style="unknown",
        )

    def _match_research_area(self, text: str) -> list:
        """匹配研究领域"""
        area_keywords = {
            "ai": ["artificial intelligence", "machine learning", "深度学习", "机器学习"],
            "cv": ["computer vision", "图像", "视频", "视觉", "cv"],
            "nlp": ["nlp", "natural language", "文本", "语言", "语言模型"],
            "se": ["software", "软件", "system"],
            "network": ["network", "网络", "通信"],
            "security": ["security", "安全", "隐私"],
            "db": ["database", "数据库", "数据存储"],
        }
        text_lower = text.lower()
        matched = []
        for area, kws in area_keywords.items():
            if any(kw.lower() in text_lower for kw in kws):
                matched.append(area)
        return matched if matched else ["other"]

    def _infer_method_type(self, text: str) -> str:
        """推断方法类型"""
        if any(kw in text.lower() for kw in ["survey", "综述", "review"]):
            return "survey"
        if any(kw in text.lower() for kw in ["system", "系统", "platform"]):
            return "system"
        if any(kw in text.lower() for kw in ["experiment", "实验", "evaluation"]):
            return "experiment"
        return "method"

    def parse(self, paper_input: PaperInput, system_prompt: str, user_prompt: str) -> PaperProfile:
        """解析论文（对外接口）"""
        if self.llm is None:
            return self._parse_by_rules(paper_input)

        try:
            return self.parse_with_llm(paper_input, system_prompt, user_prompt)
        except Exception:
            return self._parse_by_rules(paper_input)