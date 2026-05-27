"""LLM 排序（阶段二）"""
import json
import logging
import tenacity
from typing import List, Tuple

from ..journals.journal_model import Journal
from ..papers.paper_model import PaperProfile
from ..utils.llm import MiniMaxLLM, parse_json_response

logger = logging.getLogger(__name__)


class LLMRankerError(Exception):
    """LLM 排序错误（明确的业务异常）"""
    pass


class LLMRanker:
    """LLM 排序器（仅LLM，无规则降级）"""

    def __init__(self, llm: MiniMaxLLM, system_prompt: str, user_prompt_template: str):
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=2, max=8),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(f"[LLMRanker] Retry {retry_state.attempt_number}/3 after error..."),
    )
    def rank(
        self,
        candidates: List[Tuple[Journal, float, List[str]]],
        paper_profile: PaperProfile,
        top_k: int = 5,
    ) -> Tuple[List[Tuple[Journal, float, List[str], float]], str]:
        """LLM 精排（LLM驱动，重试3次）"""
        # 构建期刊信息（精简字段，含 RuleScorer 参考信息）
        journals_info = []
        for idx, (journal, rule_score, reasons) in enumerate(candidates):
            journals_info.append({
                "journal_id": journal.journal_id,
                "journal_name": journal.journal_name,
                "scope": journal.scope_text or "",  # 完整 scope
                "oa_type": journal.oa_type,
                "subject_tags": journal.subject_tags[:5],  # 限制标签数量
                "keywords": journal.keywords[:5],  # 限制关键词数量
                "rule_rank": idx + 1,                         # 粗排排名（1-based）
                "rule_reasons": reasons if reasons else [],  # 粗排匹配理由（参考）
            })

        # 填充 prompt
        user_prompt = self.user_prompt_template.format(
            title=paper_profile.title,
            research_area=", ".join(paper_profile.research_area) or "未知",
            ccf_research_area=", ".join(paper_profile.ccf_research_area) if paper_profile.ccf_research_area else "未提供",
            method_type=paper_profile.method_type or "method",
            paper_type=paper_profile.paper_type or "application",
            keywords=", ".join(paper_profile.keywords) or "无",
            novelty=paper_profile.novelty or "未提供",
            application_domain=", ".join(paper_profile.application_domain) or "通用",
            techniques=", ".join(paper_profile.techniques) or "未提供",
            datasets=", ".join(paper_profile.datasets) or "未提供",
            evaluation_metrics=", ".join(paper_profile.evaluation_metrics) or "未提供",
            novelty_type=paper_profile.novelty_type or "method",
            journals_info=json.dumps(journals_info, ensure_ascii=False, indent=2),
            total_candidates=len(candidates),
        )

        # 调用 LLM（超时 180s，自动调整max_tokens）
        try:
            response = self.llm.chat_auto(self.system_prompt, user_prompt, timeout=180)
        except Exception as e:
            raise LLMRankerError(f"LLM精排调用失败: {e}")

        # 解析结果
        data = parse_json_response(response.content)
        if not data:
            raise LLMRankerError(f"LLM响应格式错误，无法解析: {response.content}")

        rankings = data.get("rankings", [])

        # 构建结果
        rank_map = {r["journal_id"]: r for r in rankings}
        results = []
        for journal, rule_score, reasons in candidates:
            if journal.journal_id in rank_map:
                r = rank_map[journal.journal_id]
                results.append((
                    journal,
                    r.get("score", rule_score),
                    r.get("reasons", reasons),
                    r.get("confidence", 0.5),
                ))

        # 按 LLM 分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k], "llm"
