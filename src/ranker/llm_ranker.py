"""LLM 排序（阶段二）"""
import json
import re
from typing import List, Tuple

from ..journals.journal_model import Journal
from ..papers.paper_model import PaperProfile
from ..utils.llm import MiniMaxLLM


class LLMRanker:
    """LLM 排序器"""

    def __init__(self, llm: MiniMaxLLM, system_prompt: str, user_prompt_template: str):
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    def rank(
        self,
        candidates: List[Tuple[Journal, float, List[str]]],
        paper_profile: PaperProfile,
        top_k: int = 5,
    ) -> Tuple[List[Tuple[Journal, float, List[str], float]], str]:
        """LLM 精排，返回 (结果列表, 排序方法)"""
        # 构建期刊信息
        journals_info = []
        for journal, rule_score, reasons in candidates:
            journals_info.append({
                "journal_id": journal.journal_id,
                "journal_name": journal.journal_name,
                "scope": journal.scope_text,
                "quartile": journal.quartile or "unknown",
                "oa_type": journal.oa_type,
                "subject_tags": journal.subject_tags,
                "keywords": journal.keywords,
                "target_paper_type": journal.target_paper_type,
                "rule_score": rule_score,
            })

        # 填充 prompt（传入更多特征）
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
            journals_info=json.dumps(journals_info, ensure_ascii=False, indent=2),
        )

        # 调用 LLM
        try:
            response = self.llm.chat(self.system_prompt, user_prompt)
        except Exception:
            # LLM 调用失败，降级返回原始规则排序
            return [(j, s, r, 0.5) for j, s, r in candidates[:top_k]], "rule"

        # 解析结果
        try:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
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

        except Exception:
            pass

        # 降级：返回原始顺序
        return [(j, s, r, 0.5) for j, s, r in candidates[:top_k]], "rule"
