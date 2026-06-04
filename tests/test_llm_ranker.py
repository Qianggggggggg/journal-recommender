import pytest
import tenacity

from src.journals.journal_model import Journal
from src.papers.paper_model import PaperProfile
from src.ranker.llm_ranker import LLMRanker, LLMRankerError
from src.utils.llm import LLMResponse


class FixedResponseLLM:
    def __init__(self, content: str):
        self.content = content

    def chat_auto(self, system: str, user: str, timeout: float):
        return LLMResponse(content=self.content, model="test", usage={})


class SequenceResponseLLM:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.call_count = 0

    def chat_auto(self, system: str, user: str, timeout: float):
        self.call_count += 1
        return LLMResponse(content=next(self.contents), model="test", usage={})


def _ranker(llm):
    return LLMRanker(
        llm,
        system_prompt="rank",
        user_prompt_template=(
            "{title} {research_area} {ccf_research_area} {method_type} "
            "{paper_type} {keywords} {novelty} {application_domain} "
            "{techniques} {datasets} {evaluation_metrics} {novelty_type} "
            "{journals_info} {total_candidates}"
        ),
    )


def _rank_once(content: str):
    ranker = _ranker(FixedResponseLLM(content))
    journal = Journal(journal_id="target", journal_name="Target")
    rank_once = LLMRanker.rank.retry_with(
        stop=tenacity.stop_after_attempt(1),
        wait=tenacity.wait_none(),
    )
    return rank_once(
        ranker,
        [(journal, 1.0, ["rule"])],
        PaperProfile(title="Paper"),
        top_k=1,
    )


def test_llm_ranker_rejects_empty_rankings():
    with pytest.raises(LLMRankerError, match="rankings 为空"):
        _rank_once('{"rankings":[]}')


def test_llm_ranker_rejects_malformed_ranking_items():
    with pytest.raises(LLMRankerError, match="ranking item"):
        _rank_once('{"rankings":["target"]}')


def test_llm_ranker_rejects_unknown_only_journal_ids():
    with pytest.raises(LLMRankerError, match="没有匹配任何候选期刊"):
        _rank_once('{"rankings":[{"journal_id":"unknown","score":0.9}]}')


def test_llm_ranker_rejects_non_numeric_scores():
    with pytest.raises(LLMRankerError, match="score"):
        _rank_once('{"rankings":[{"journal_id":"target","score":["0.9"]}]}')


def test_empty_rankings_are_retried_until_valid_response():
    llm = SequenceResponseLLM(
        [
            '{"rankings":[]}',
            '{"rankings":[]}',
            '{"rankings":[{"journal_id":"target","score":0.9}]}',
        ]
    )
    ranker = _ranker(llm)
    journal = Journal(journal_id="target", journal_name="Target")
    rank_with_fast_retry = LLMRanker.rank.retry_with(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_none(),
    )

    ranked, method = rank_with_fast_retry(
        ranker,
        [(journal, 1.0, ["rule"])],
        PaperProfile(title="Paper"),
        top_k=1,
    )

    assert method == "llm"
    assert ranked[0][0].journal_id == "target"
    assert llm.call_count == 3


def test_llm_ranker_keeps_valid_ranking_behavior():
    ranked, method = _rank_once(
        '{"rankings":[{"journal_id":"target","score":0.9,'
        '"reasons":["fit"],"confidence":0.8}]}'
    )

    assert method == "llm"
    assert ranked[0][0].journal_id == "target"
    assert ranked[0][1:] == (0.9, ["fit"], 0.8)
