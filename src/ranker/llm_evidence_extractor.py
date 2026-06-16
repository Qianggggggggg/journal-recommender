"""Structured LLM evidence extraction for journal candidates."""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import tenacity

from ..journals.journal_model import Journal
from ..papers.paper_model import PaperProfile
from ..utils.llm import MiniMaxLLM, parse_json_response

logger = logging.getLogger(__name__)


class LLMEvidenceExtractorError(Exception):
    """LLM evidence extraction failed or returned an invalid response."""


class LLMEvidenceExtractor:
    """Extract per-journal fit evidence in one batched LLM request."""

    SCORE_FIELDS = (
        "scope_fit",
        "method_fit",
        "application_fit",
        "journal_position_fit",
        "too_broad_penalty",
        "too_narrow_penalty",
    )

    JSON_OUTPUT_CONTRACT = """

【输出格式硬约束】
只输出一个合法 JSON 对象，不要输出 Markdown，不要输出分析过程，不要使用代码块。
JSON 对象必须包含 evidence 数组，每个元素必须包含候选 journal_id、六个 0 到 1
之间的数值字段和非空 evidence 字符串数组。
"""

    def __init__(
        self,
        llm: MiniMaxLLM,
        system_prompt: str,
        user_prompt_template: str,
        timeout_seconds: float = 200,
        focused_user_prompt_template: Optional[str] = None,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        # Task 6.3 incremental repair: optional template for focused re-extraction
        # of only the missing candidates. Falls back to the full template (which
        # then has to be re-run in full) if not supplied.
        self.focused_user_prompt_template = focused_user_prompt_template
        self.timeout_seconds = timeout_seconds

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=2, max=8),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            "[LLMEvidenceExtractor] Retry %s/3 after error...",
            retry_state.attempt_number,
        ),
    )
    def extract(
        self,
        candidates: List[Tuple[Journal, float, List[str]]],
        paper_profile: PaperProfile,
        rule_ranks: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return validated evidence indexed by candidate journal ID.

        ``rule_ranks`` (Fix #2, real rule_rank): when supplied, the prompt
        tells the LLM the candidate's real RuleScorer rank, not its position
        in the (possibly LTR-sorted) input list. This prevents the LLM from
        receiving a mislabeled rank hint.
        """
        if not candidates:
            return {}

        rule_ranks = rule_ranks or {}
        journals_info = [
            {
                "journal_id": journal.journal_id,
                "journal_name": journal.journal_name,
                "scope": journal.scope_text or "",
                "subject_tags": journal.subject_tags[:5],
                "keywords": journal.keywords[:8],
                "ccf_rating": journal.ccf_rating or "未知",
                "rule_rank": rule_ranks.get(journal.journal_id, index + 1),
                "rule_reasons": reasons or [],
            }
            for index, (journal, _, reasons) in enumerate(candidates)
        ]
        user_prompt = self.user_prompt_template.format(
            title=paper_profile.title,
            abstract=paper_profile.abstract or "未提供",
            research_area=", ".join(paper_profile.research_area) or "未知",
            ccf_research_area=", ".join(paper_profile.ccf_research_area)
            if paper_profile.ccf_research_area
            else "未提供",
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

        try:
            response = self.llm.chat_auto(
                self._system_prompt(),
                user_prompt,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            raise LLMEvidenceExtractorError(f"LLM证据提取调用失败: {exc}") from exc

        if not response.content or not response.content.strip():
            usage = getattr(response, "usage", {}) or {}
            raise LLMEvidenceExtractorError(
                f"LLM返回空响应，无法解析证据 JSON。usage={usage}"
            )

        data = self._parse_response(response.content)
        if isinstance(data, dict):
            evidence_items = data.get("evidence", [])
        elif isinstance(data, list):
            evidence_items = data
        else:
            raise LLMEvidenceExtractorError(
                f"LLM证据响应格式错误，无法解析: {response.content}"
            )

        if not isinstance(evidence_items, list) or not evidence_items:
            raise LLMEvidenceExtractorError("LLM响应中的 evidence 为空")

        candidate_ids = {journal.journal_id for journal, _, _ in candidates}
        seen_ids = set()
        result: Dict[str, Dict[str, Any]] = {}
        for item in evidence_items:
            if not isinstance(item, dict) or not isinstance(item.get("journal_id"), str):
                raise LLMEvidenceExtractorError("LLM响应包含无效 evidence item")

            journal_id = item["journal_id"]
            if journal_id not in candidate_ids:
                continue
            if journal_id in seen_ids:
                raise LLMEvidenceExtractorError(
                    f"LLM evidence 包含重复 journal_id: {journal_id}"
                )
            seen_ids.add(journal_id)

            self._validate_item(item)
            result[journal_id] = dict(item)

        if not result:
            raise LLMEvidenceExtractorError("LLM evidence 没有匹配任何候选期刊")
        return result

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=2, max=8),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            "[LLMEvidenceExtractor.focused] Retry %s/3 after error...",
            retry_state.attempt_number,
        ),
    )
    def extract_focused(
        self,
        candidates: List[Tuple[Journal, float, List[str]]],
        paper_profile: PaperProfile,
        focus_journal_ids: List[str],
        already_covered_ids: Optional[List[str]] = None,
        rule_ranks: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Task 6.3 incremental repair: re-extract evidence for ONLY the missing
        candidates. Uses a narrower focused prompt that lists the missing
        journals and asks the LLM to return evidence just for those.

        Returns a dict like ``extract()`` but only the missing journal_ids
        should be present. Any extra items returned by the LLM (e.g., evidence
        for already-covered IDs) are silently filtered out to keep the merged
        snapshot clean.

        Falls back to the full ``extract()`` if no focused template was
        supplied at init time.
        """
        if not focus_journal_ids:
            return {}
        if self.focused_user_prompt_template is None:
            logger.warning(
                "extract_focused called without focused_user_prompt_template; "
                "falling back to full extract()"
            )
            return self.extract(
                candidates, paper_profile, rule_ranks=rule_ranks
            )

        # Build a sub-candidate list limited to focus_journal_ids, preserving
        # the original rule-rank order.
        rule_ranks = rule_ranks or {}
        candidates_by_id = {j.journal_id: (j, s, r) for j, s, r in candidates}
        missing_candidates: List[Tuple[Journal, float, List[str]]] = []
        for jid in focus_journal_ids:
            if jid in candidates_by_id:
                missing_candidates.append(candidates_by_id[jid])
        if not missing_candidates:
            return {}

        journals_info = [
            {
                "journal_id": j.journal_id,
                "journal_name": j.journal_name,
                "scope": j.scope_text or "",
                "subject_tags": j.subject_tags[:5],
                "keywords": j.keywords[:8],
                "ccf_rating": j.ccf_rating or "未知",
                "rule_rank": rule_ranks.get(j.journal_id, idx + 1),
                "rule_reasons": r or [],
            }
            for idx, (j, _s, r) in enumerate(missing_candidates)
        ]
        covered_list = already_covered_ids or []
        user_prompt = self.focused_user_prompt_template.format(
            title=paper_profile.title,
            abstract=paper_profile.abstract or "未提供",
            research_area=", ".join(paper_profile.research_area) or "未知",
            ccf_research_area=", ".join(paper_profile.ccf_research_area)
            if paper_profile.ccf_research_area
            else "未提供",
            method_type=paper_profile.method_type or "method",
            paper_type=paper_profile.paper_type or "application",
            keywords=", ".join(paper_profile.keywords) or "无",
            novelty=paper_profile.novelty or "未提供",
            application_domain=", ".join(paper_profile.application_domain) or "通用",
            techniques=", ".join(paper_profile.techniques) or "未提供",
            datasets=", ".join(paper_profile.datasets) or "未提供",
            evaluation_metrics=", ".join(paper_profile.evaluation_metrics) or "未提供",
            novelty_type=paper_profile.novelty_type or "method",
            already_covered_ids=", ".join(covered_list) or "(none)",
            covered_count=len(covered_list),
            missing_journal_ids=", ".join(focus_journal_ids),
            missing_count=len(focus_journal_ids),
            missing_journals_info=json.dumps(journals_info, ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_auto(
                self._system_prompt(),
                user_prompt,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            raise LLMEvidenceExtractorError(
                f"LLM focused evidence 调用失败: {exc}"
            ) from exc

        if not response.content or not response.content.strip():
            usage = getattr(response, "usage", {}) or {}
            raise LLMEvidenceExtractorError(
                f"LLM focused 返回空响应。usage={usage}"
            )

        data = self._parse_response(response.content)
        if isinstance(data, dict):
            evidence_items = data.get("evidence", [])
        elif isinstance(data, list):
            evidence_items = data
        else:
            raise LLMEvidenceExtractorError(
                f"LLM focused 响应格式错误: {response.content}"
            )

        if not isinstance(evidence_items, list) or not evidence_items:
            raise LLMEvidenceExtractorError("LLM focused 响应中的 evidence 为空")

        allowed = set(focus_journal_ids)
        seen = set()
        result: Dict[str, Dict[str, Any]] = {}
        for item in evidence_items:
            if not isinstance(item, dict) or not isinstance(item.get("journal_id"), str):
                raise LLMEvidenceExtractorError("LLM focused 包含无效 evidence item")
            jid = item["journal_id"]
            if jid not in allowed:
                # LLM returned evidence for an already-covered ID; ignore.
                continue
            if jid in seen:
                raise LLMEvidenceExtractorError(
                    f"LLM focused 包含重复 journal_id: {jid}"
                )
            seen.add(jid)
            self._validate_item(item)
            result[jid] = dict(item)

        if not result:
            raise LLMEvidenceExtractorError(
                "LLM focused evidence 没有匹配任何缺失候选期刊"
            )
        return result

    @staticmethod
    def _parse_response(content: str) -> Any:
        stripped = content.strip()
        fence_match = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```",
            stripped,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fence_match:
            payload = fence_match.group(1).strip()
        elif stripped.startswith(("{", "[")):
            payload = stripped
        else:
            raise LLMEvidenceExtractorError(
                "LLM证据响应只允许纯 JSON 或完整 JSON code fence"
            )

        try:
            strict_result = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise LLMEvidenceExtractorError(
                "LLM证据响应格式错误，无法解析严格 JSON"
            ) from exc
        if not isinstance(strict_result, (dict, list)):
            raise LLMEvidenceExtractorError("LLM证据响应必须是 JSON 对象或数组")
        return parse_json_response(payload)

    @classmethod
    def _validate_item(cls, item: Dict[str, Any]) -> None:
        for field in cls.SCORE_FIELDS:
            value = item.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise LLMEvidenceExtractorError(f"LLM evidence 的 {field} 必须是数值")
            if not 0 <= value <= 1:
                raise LLMEvidenceExtractorError(
                    f"LLM evidence 的 {field} 必须位于 0 到 1 之间"
                )

        evidence = item.get("evidence")
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(not isinstance(text, str) or not text.strip() for text in evidence)
        ):
            raise LLMEvidenceExtractorError("LLM evidence item 的 evidence 必须是非空字符串列表")

    def _system_prompt(self) -> str:
        if "【输出格式硬约束】" in self.system_prompt:
            return self.system_prompt
        return f"{self.system_prompt.rstrip()}{self.JSON_OUTPUT_CONTRACT}"


# P0 (2026-06-16 diagnostic): v2 evidence prompt adds CCF-tier calibration to
# counter the systematic under-scoring of C-tier application-oriented journals
# (C-tier hit@5 = 49-56% vs A-tier 76-85% on holdout240; C-tier gold
# evidence_composite median = 0.50 vs top1 median = 0.775). v1 is the original
# prompt; v2 is the calibration variant. Switch via
# configs/app.yaml::ranking.evidence_role.prompt_version = "v1" | "v2".
EVIDENCE_PROMPT_VERSIONS = ("v1", "v2")


def select_evidence_prompts(prompts: Dict[str, str], version: str) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt_template) for the requested version.

    Falls back to v1 keys for unknown versions so a typo in app.yaml does not
    crash the pipeline. The fallback is logged inside the caller's
    initialization (not here, to keep this function pure for testing).

    Locked by tests/test_prompt_templates.py::TestSelectEvidencePrompts.
    """
    if version == "v2":
        sys_key = "llm_evidence_extractor_system_v2"
        usr_key = "llm_evidence_extractor_user_v2"
    else:
        sys_key = "llm_evidence_extractor_system"
        usr_key = "llm_evidence_extractor_user"

    # If v2 keys are missing, fall back to v1 so legacy configs / test fixtures
    # that don't yet carry v2 still work.
    if version == "v2" and (sys_key not in prompts or usr_key not in prompts):
        sys_key = "llm_evidence_extractor_system"
        usr_key = "llm_evidence_extractor_user"

    return prompts[sys_key], prompts[usr_key]
