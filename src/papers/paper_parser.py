"""论文解析（纯LLM）"""
import logging
import time
from typing import Optional

import tenacity

from ..utils.llm import MiniMaxLLM, parse_json_response
from .paper_model import PaperProfile, PaperInput

logger = logging.getLogger(__name__)


class PaperParserError(Exception):
    """论文解析错误（明确的业务异常）"""
    pass


class PaperParser:
    """论文解析器（仅LLM，无规则降级）"""

    def __init__(self, llm: MiniMaxLLM):
        if llm is None:
            raise PaperParserError("LLM not configured, please set minimax API key")
        self.llm = llm

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=2, max=8),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(f"[PaperParser] Retry {retry_state.attempt_number}/3 after error..."),
    )
    def parse(self, paper_input: PaperInput, system_prompt: str, user_prompt: str) -> PaperProfile:
        """解析论文（LLM驱动，重试3次）"""
        user_filled = user_prompt.format(
            title=paper_input.title,
            abstract=paper_input.abstract or "",
            full_text_summary=paper_input.full_text if paper_input.full_text else "",
        )

        try:
            response = self.llm.chat(system_prompt, user_filled)
        except Exception as e:
            raise PaperParserError(f"LLM调用失败: {e}")

        # 解析 JSON 响应
        data = parse_json_response(response.content)
        if data:

            # 确保列表字段是列表类型（防御 LLM 返回字符串的情况）
            list_fields = ["research_area", "application_domain", "keywords", "techniques", "datasets", "evaluation_metrics"]
            for field in list_fields:
                if field in data and not isinstance(data[field], list):
                    # 如果是字符串，尝试按逗号分割
                    if isinstance(data[field], str):
                        data[field] = [x.strip() for x in data[field].split(",") if x.strip()]
                    else:
                        data[field] = []

            return PaperProfile(
                title=paper_input.title,
                abstract=paper_input.abstract or "",
                **{k: v for k, v in data.items() if k != "title"}
            )

        raise PaperParserError(f"LLM响应格式错误，无法解析: {response.content}")