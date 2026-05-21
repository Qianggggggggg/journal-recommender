"""推荐理由生成模块"""
import json
from typing import List, Optional

from ..journals.journal_model import Journal
from ..papers.paper_model import PaperProfile
from ..utils.llm import MiniMaxLLM, parse_json_response


class Explainer:
    """推荐理由生成器"""

    def __init__(
        self,
        llm: Optional[MiniMaxLLM] = None,
        system_prompt: str = "",
        user_prompt_template: str = "",
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    def explain(
        self,
        journal: Journal,
        paper_profile: PaperProfile,
        matched_fields: Optional[List[str]] = None,
    ) -> List[str]:
        """生成推荐理由"""
        if self.llm is None or not self.user_prompt_template:
            return self._generate_fallback_explanation(journal, paper_profile, matched_fields)

        user_prompt = self.user_prompt_template.format(
            title=paper_profile.title,
            research_area=", ".join(paper_profile.research_area) or "未知",
            method_type=paper_profile.method_type or "method",
            paper_type=paper_profile.paper_type or "application",
            keywords=", ".join(paper_profile.keywords) or "无",
            novelty=paper_profile.novelty or "未提供",
            application_domain=", ".join(paper_profile.application_domain) or "通用",
            techniques=", ".join(paper_profile.techniques) or "未提供",
            datasets=", ".join(paper_profile.datasets) or "未提供",
            evaluation_metrics=", ".join(paper_profile.evaluation_metrics) or "未提供",
            novelty_type=paper_profile.novelty_type or "method",
            journal_name=journal.journal_name,
            scope_text=journal.scope_text,
            oa_type=journal.oa_type,
            review_time=journal.review_time or "unknown",
            journal_keywords=", ".join(journal.keywords) or "无",
        )

        try:
            response = self.llm.chat(self.system_prompt, user_prompt)
            data = parse_json_response(response.content)
            if data:
                reasons = data.get("reasons", [])
                if reasons:
                    return reasons
        except Exception:
            pass

        return self._generate_fallback_explanation(journal, paper_profile, matched_fields)

    def _generate_fallback_explanation(
        self,
        journal: Journal,
        paper_profile: PaperProfile,
        matched_fields: Optional[List[str]] = None,
    ) -> List[str]:
        """生成简洁推荐理由"""
        reasons = []

        # 1. 研究领域匹配
        if paper_profile.research_area:
            matched = [a for a in paper_profile.research_area if a in journal.subject_tags]
            if matched:
                reasons.append(f"领域匹配：{', '.join(matched)}")
            else:
                reasons.append(f"领域：{', '.join(paper_profile.research_area[:2])}")

        # 2. 技术方法
        if paper_profile.techniques:
            reasons.append(f"技术：{', '.join(paper_profile.techniques[:2])}")

        # 3. 方法类型
        if paper_profile.method_type:
            labels = {"method": "方法论", "system": "系统设计", "experiment": "实验", "survey": "综述"}
            reasons.append(f"类型：{labels.get(paper_profile.method_type, paper_profile.method_type)}")

        # 4. OA模式
        if journal.oa_type:
            labels = {"full_oa": "完全OA", "hybrid": "混合OA", "subscription": "订阅"}
            reasons.append(f"出版：{labels.get(journal.oa_type, journal.oa_type)}")

        if not reasons:
            reasons.append("匹配度较高")

        return reasons[:4]  # 最多4条

    def explain_batch(
        self,
        journals: List[Journal],
        paper_profile: PaperProfile,
    ) -> List[List[str]]:
        """批量生成推荐理由"""
        return [self.explain(j, paper_profile) for j in journals]
