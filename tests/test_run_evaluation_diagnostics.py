"""Evaluation diagnostics regression tests."""
from scripts.run_evaluation import evaluate_single_paper
from src.journals.journal_model import Journal, JournalMatch
from src.journals.journal_store import JournalStore
from src.papers.paper_model import PaperProfile


class DummyParser:
    def parse(self, paper_input, system_prompt, user_prompt):
        return PaperProfile(
            title=paper_input.title,
            abstract=paper_input.abstract or "",
            ccf_research_area=["人工智能"],
        )


class DummyPipeline:
    def __init__(self, journal):
        store = JournalStore()
        store.add_journal(journal)
        self.parser = DummyParser()
        self.candidate_generator = type("Generator", (), {"store": store})()
        self.journal = journal

    def recommend(
        self,
        paper_input,
        profile,
        top_k=5,
        mode="abstract",
        quality_prompts=None,
        diagnostic_journal_ids=None,
    ):
        return {
            "recommendations": [
                JournalMatch(
                    journal=self.journal,
                    score=0.91,
                    confidence=0.8,
                    match_reasons=["范围文本覆盖论文主题"],
                )
            ],
            "candidates": [self.journal],
            "rule_ranked": [(self.journal, 0.77, ["期刊范围文本提供边界匹配证据"])],
            "llm_candidates": [(self.journal, 0.77, [])],
            "llm_candidate_ids": [self.journal.journal_id],
            "retrieval_trace": {
                self.journal.journal_id: {
                    "retrieval_rank": 1,
                    "total_score": 0.42,
                    "primary_routes": ["scope_bm25", "typical_bm25"],
                    "routes": {
                        "scope_bm25": {
                            "rank": 1,
                            "weighted_score": 0.3,
                            "normalized_score": 1.0,
                        },
                        "typical_bm25": {
                            "rank": 2,
                            "weighted_score": 0.12,
                            "normalized_score": 0.6,
                        },
                    },
                }
            },
        }


class WideMissPipeline(DummyPipeline):
    def __init__(self, target, recommended):
        super().__init__(target)
        self.recommended = recommended
        self.candidate_generator.store.add_journal(recommended)

    def recommend(
        self,
        paper_input,
        profile,
        top_k=5,
        mode="abstract",
        quality_prompts=None,
        diagnostic_journal_ids=None,
    ):
        return {
            "recommendations": [
                JournalMatch(
                    journal=self.recommended,
                    score=0.7,
                    confidence=0.6,
                    match_reasons=["邻近期刊"],
                )
            ],
            "candidates": [self.recommended],
            "rule_ranked": [(self.recommended, 0.5, [])],
            "llm_candidates": [(self.recommended, 0.5, [])],
            "llm_candidate_ids": [self.recommended.journal_id],
            "retrieval_trace": {
                self.recommended.journal_id: {
                    "retrieval_rank": 1,
                    "total_score": 0.5,
                    "primary_routes": ["scope_bm25"],
                    "routes": {"scope_bm25": {"rank": 1, "weighted_score": 0.5}},
                },
                self.journal.journal_id: {
                    "wide_retrieval_rank": 3,
                    "total_score": 0.2,
                    "primary_routes": ["scope_bm25"],
                    "routes": {"scope_bm25": {"rank": 3, "weighted_score": 0.2}},
                    "wide_routes": {"scope_bm25": {"rank": 3, "weighted_score": 0.2}},
                },
            },
        }


def test_evaluate_single_paper_writes_venue_diagnostic():
    journal = Journal(
        journal_id="target",
        journal_name="Target Journal",
        ccf_rating="B",
    )
    result = evaluate_single_paper(
        {
            "title": "Test Paper With A Long Enough Title That Must Not Be Truncated In Evaluation Diagnostics",
            "abstract": "A short abstract.",
            "venue": "Target Journal",
            "ccf_level": "B",
            "research_area": ["人工智能"],
            "external_ids": {"arXiv": "1234.5678"},
        },
        DummyPipeline(journal),
        {"paper_profile_system": "", "paper_profile_user": ""},
        mode="abstract",
        top_k=5,
    )

    diagnostic = result["venue_diagnostic"]
    assert result["title"] == "Test Paper With A Long Enough Title That Must Not Be Truncated In Evaluation Diagnostics"
    assert result["abstract_len"] == len("A short abstract.")
    assert result["gold_area"] == "人工智能"
    assert result["parsed_ccf_area"] == ["人工智能"]
    assert result["area_mismatch"] is False
    assert diagnostic["journal_id"] == "target"
    assert diagnostic["retrieval_rank"] == 1
    assert diagnostic["retrieval_score"] == 0.42
    assert diagnostic["rule_rank"] == 1
    assert diagnostic["rule_score"] == 0.77
    assert diagnostic["in_llm_pool"] is True
    assert diagnostic["retrieval_sources"] == ["scope_bm25", "typical_bm25"]
    assert diagnostic["target_journal_id"] == "target"
    assert diagnostic["gold_area"] == "人工智能"
    assert diagnostic["parsed_ccf_area"] == ["人工智能"]
    assert diagnostic["area_mismatch"] is False
    assert diagnostic["abstract_len"] == len("A short abstract.")
    assert diagnostic["miss_stage"] == "final_hit"
    assert result["paper_profile_snapshot"]["title"] == "Test Paper With A Long Enough Title That Must Not Be Truncated In Evaluation Diagnostics"
    assert result["paper_profile_snapshot"]["abstract_len"] == len("A short abstract.")
    assert result["paper_profile_snapshot"]["abstract_preview"] == "A short abstract."


def test_evaluate_single_paper_marks_wide_recalled_not_top50():
    target = Journal(journal_id="target", journal_name="Target Journal", ccf_rating="B")
    recommended = Journal(journal_id="other", journal_name="Other Journal", ccf_rating="B")

    result = evaluate_single_paper(
        {
            "title": "Test Paper",
            "abstract": "A short abstract.",
            "venue": "Target Journal",
            "ccf_level": "B",
            "research_area": ["人工智能"],
            "external_ids": {"arXiv": "1234.5678"},
        },
        WideMissPipeline(target, recommended),
        {"paper_profile_system": "", "paper_profile_user": ""},
        mode="abstract",
        top_k=5,
    )

    diagnostic = result["venue_diagnostic"]
    assert diagnostic["target_journal_id"] == "target"
    assert diagnostic["wide_retrieval_rank"] == 3
    assert diagnostic["retrieval_rank"] is None
    assert diagnostic["wide_retrieval_route_scores"]["scope_bm25"]["rank"] == 3
    assert diagnostic["miss_stage"] == "wide_recalled_but_not_top50"


def test_evaluate_single_paper_writes_auxiliary_acceptability_metrics():
    target = Journal(
        journal_id="target",
        journal_name="Target Journal",
        ccf_rating="B",
        subject_tags=["人工智能"],
    )
    recommended = Journal(
        journal_id="other",
        journal_name="Other Journal",
        ccf_rating="B",
        subject_tags=["人工智能"],
    )

    result = evaluate_single_paper(
        {
            "title": "Test Paper",
            "abstract": "A short abstract.",
            "venue": "Target Journal",
            "ccf_level": "B",
            "research_area": ["人工智能"],
            "external_ids": {"arXiv": "1234.5678"},
        },
        WideMissPipeline(target, recommended),
        {"paper_profile_system": "", "paper_profile_user": ""},
        mode="abstract",
        top_k=5,
    )

    assert result["hit_5"] is False
    assert result["same_area_hit_5"] is True
    assert result["same_ccf_level_hit_5"] is True
    assert result["acceptable_journal_hit_5"] is True
