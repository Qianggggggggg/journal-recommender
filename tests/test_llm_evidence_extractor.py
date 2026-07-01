import json

import pytest
import tenacity

from src.journals.journal_model import Journal
from src.papers.paper_model import PaperProfile
from src.ranker.llm_evidence_extractor import (
    LLMEvidenceExtractor,
    LLMEvidenceExtractorError,
)
from src.utils.llm import LLMResponse


class FixedResponseLLM:
    def __init__(self, content: str):
        self.content = content
        self.call_count = 0
        self.last_system = ""
        self.last_user = ""

    def chat_auto(self, system: str, user: str, timeout: float):
        self.call_count += 1
        self.last_system = system
        self.last_user = user
        return LLMResponse(content=self.content, model="test", usage={})


class SequenceResponseLLM:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.call_count = 0

    def chat_auto(self, system: str, user: str, timeout: float):
        self.call_count += 1
        return LLMResponse(content=next(self.contents), model="test", usage={})


def _journal(journal_id: str, name: str) -> Journal:
    return Journal(
        journal_id=journal_id,
        journal_name=name,
        scope_text=f"{name} scope",
        subject_tags=["人工智能"],
        keywords=["machine learning"],
        ccf_rating="B",
    )


def _candidates():
    return [
        (_journal("j1", "Journal One"), 8.0, ["scope match"]),
        (_journal("j2", "Journal Two"), 7.0, ["method match"]),
    ]


def _profile() -> PaperProfile:
    return PaperProfile(
        title="Test Paper",
        abstract="We propose a transformer method for healthcare classification.",
        research_area=["人工智能"],
        ccf_research_area=["人工智能"],
        method_type="method",
        paper_type="application",
        keywords=["machine learning"],
        novelty="new method",
        application_domain=["healthcare"],
        techniques=["transformer"],
        datasets=["dataset-a"],
        evaluation_metrics=["accuracy"],
        novelty_type="new_method",
    )


def _valid_item(journal_id: str) -> dict:
    return {
        "journal_id": journal_id,
        "scope_fit": 0.9,
        "method_fit": 0.8,
        "application_fit": 0.7,
        "journal_position_fit": 0.6,
        "too_broad_penalty": 0.1,
        "too_narrow_penalty": 0.05,
        "evidence": ["scope explicitly covers the paper topic"],
    }


def _extractor(llm) -> LLMEvidenceExtractor:
    return LLMEvidenceExtractor(
        llm=llm,
        system_prompt="extract evidence",
        user_prompt_template=(
            "{title} {abstract} {research_area} {ccf_research_area} {method_type} "
            "{paper_type} {keywords} {novelty} {application_domain} "
            "{techniques} {datasets} {evaluation_metrics} {novelty_type} "
            "{journals_info} {total_candidates}"
        ),
    )


def _extract_once(content: str):
    extractor = _extractor(FixedResponseLLM(content))
    extract_once = LLMEvidenceExtractor.extract.retry_with(
        stop=tenacity.stop_after_attempt(1),
        wait=tenacity.wait_none(),
    )
    return extract_once(extractor, _candidates(), _profile())


def test_extract_returns_structured_evidence_for_multiple_candidates():
    llm = FixedResponseLLM(
        json.dumps({"evidence": [_valid_item("j1"), _valid_item("j2")]})
    )
    extractor = _extractor(llm)

    result = extractor.extract(_candidates(), _profile())

    assert list(result) == ["j1", "j2"]
    assert result["j1"]["scope_fit"] == 0.9
    assert result["j2"]["too_narrow_penalty"] == 0.05
    assert result["j1"]["evidence"] == [
        "scope explicitly covers the paper topic"
    ]
    assert llm.call_count == 1
    assert '"journal_id": "j1"' in llm.last_user
    assert '"rule_rank": 1' in llm.last_user
    assert "transformer method for healthcare classification" in llm.last_user


def test_extract_accepts_json_code_fence():
    content = "```json\n" + json.dumps({"evidence": [_valid_item("j1")]}) + "\n```"

    result = _extract_once(content)

    assert list(result) == ["j1"]


def test_extract_rejects_markdown_analysis_before_json():
    content = (
        "# Analysis\nThe first journal is a strong fit.\n"
        + json.dumps({"evidence": [_valid_item("j1")]})
    )

    with pytest.raises(LLMEvidenceExtractorError, match="只允许纯 JSON"):
        _extract_once(content)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"evidence": []}, "evidence 为空"),
        ({"evidence": [{**_valid_item("j1"), "scope_fit": "0.9"}]}, "scope_fit"),
        ({"evidence": [{**_valid_item("j1"), "method_fit": 1.1}]}, "method_fit"),
        ({"evidence": [{**_valid_item("j1"), "evidence": []}]}, "evidence"),
        ({"evidence": [_valid_item("unknown")]}, "没有匹配任何候选期刊"),
    ],
)
def test_extract_rejects_invalid_evidence(payload, message):
    with pytest.raises(LLMEvidenceExtractorError, match=message):
        _extract_once(json.dumps(payload))


def test_extract_rejects_duplicate_journal_ids():
    payload = {"evidence": [_valid_item("j1"), _valid_item("j1")]}

    with pytest.raises(LLMEvidenceExtractorError, match="重复 journal_id"):
        _extract_once(json.dumps(payload))


def test_extract_ignores_malformed_unknown_journal_when_candidate_is_valid():
    payload = {
        "evidence": [
            _valid_item("j1"),
            {"journal_id": "hallucinated", "scope_fit": "invalid"},
        ]
    }

    result = _extract_once(json.dumps(payload))

    assert list(result) == ["j1"]


def test_empty_evidence_is_retried_until_valid_response():
    llm = SequenceResponseLLM(
        [
            '{"evidence":[]}',
            '{"evidence":[]}',
            json.dumps({"evidence": [_valid_item("j1")]}),
        ]
    )
    extractor = _extractor(llm)
    extract_with_fast_retry = LLMEvidenceExtractor.extract.retry_with(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_none(),
    )

    result = extract_with_fast_retry(extractor, _candidates(), _profile())

    assert list(result) == ["j1"]
    assert llm.call_count == 3


def test_extract_empty_candidates_does_not_call_llm():
    llm = FixedResponseLLM(json.dumps({"evidence": [_valid_item("j1")]}))
    extractor = _extractor(llm)

    assert extractor.extract([], _profile()) == {}
    assert llm.call_count == 0


# ---------------------------------------------------------------------------
# Task 6.3 incremental repair: focused re-extraction
# ---------------------------------------------------------------------------


FOCUSED_USER_TEMPLATE = (
    "{title} {abstract} {research_area} {ccf_research_area} {method_type} "
    "{paper_type} {keywords} {novelty} {application_domain} "
    "{techniques} {datasets} {evaluation_metrics} {novelty_type} "
    "{already_covered_ids} {covered_count} {missing_journal_ids} "
    "{missing_count} {missing_journals_info}"
)


def _extractor_with_focused(llm) -> LLMEvidenceExtractor:
    return LLMEvidenceExtractor(
        llm=llm,
        system_prompt="extract evidence",
        user_prompt_template=FOCUSED_USER_TEMPLATE,
        focused_user_prompt_template=FOCUSED_USER_TEMPLATE,
    )


def test_extract_focused_returns_evidence_only_for_missing_journal_ids():
    llm = FixedResponseLLM(
        json.dumps({"evidence": [_valid_item("j2")]})
    )
    extractor = _extractor_with_focused(llm)
    cands = _candidates()

    result = extractor.extract_focused(
        candidates=cands,
        paper_profile=_profile(),
        focus_journal_ids=["j2"],
        already_covered_ids=["j1"],
        rule_ranks={"j1": 1, "j2": 2},
    )

    assert list(result) == ["j2"]
    assert result["j2"]["scope_fit"] == 0.9
    # The LLM was called exactly once (via the focused prompt).
    assert llm.call_count == 1
    # Behavior proof: the focused call returned only the missing journal
    # (the full extract would have returned both j1 and j2 if both candidates
    # were in scope). The filter-on-allow-list logic is exercised by the
    # other test, so this test focuses on the single-missing-journal path.


def test_extract_focused_filters_out_evidence_for_already_covered_ids():
    """If the LLM returns evidence for an already-covered journal, the focused
    extractor must silently drop it so the merged snapshot stays clean."""
    llm = FixedResponseLLM(
        json.dumps({
            "evidence": [_valid_item("j1"), _valid_item("j2")]
        })
    )
    extractor = _extractor_with_focused(llm)
    cands = _candidates()

    result = extractor.extract_focused(
        candidates=cands,
        paper_profile=_profile(),
        focus_journal_ids=["j2"],  # only j2 is missing
        already_covered_ids=["j1"],
        rule_ranks={"j1": 1, "j2": 2},
    )

    # j1 evidence is filtered out, only j2 is returned.
    assert list(result) == ["j2"]


def test_extract_focused_raises_when_llm_returns_no_valid_evidence():
    llm = FixedResponseLLM(json.dumps({"evidence": []}))
    extractor = _extractor_with_focused(llm)
    with pytest.raises(LLMEvidenceExtractorError, match="为空"):
        extractor.extract_focused(
            candidates=_candidates(),
            paper_profile=_profile(),
            focus_journal_ids=["j2"],
        )


def test_extract_focused_empty_focus_list_returns_empty_without_calling_llm():
    llm = FixedResponseLLM(json.dumps({"evidence": [_valid_item("j2")]}))
    extractor = _extractor_with_focused(llm)

    result = extractor.extract_focused(
        candidates=_candidates(),
        paper_profile=_profile(),
        focus_journal_ids=[],  # nothing missing
    )

    assert result == {}
    assert llm.call_count == 0


def test_extract_focused_falls_back_to_full_extract_when_no_focused_template():
    """If the extractor wasn't initialized with a focused template, it must
    fall back to the full extract() rather than silently returning empty."""
    llm = FixedResponseLLM(
        json.dumps({"evidence": [_valid_item("j1"), _valid_item("j2")]})
    )
    extractor = _extractor(llm)  # no focused template
    cands = _candidates()

    result = extractor.extract_focused(
        candidates=cands,
        paper_profile=_profile(),
        focus_journal_ids=["j2"],
    )

    # Falls back to full extract; both j1 and j2 returned.
    assert set(result) == {"j1", "j2"}
    assert llm.call_count == 1
