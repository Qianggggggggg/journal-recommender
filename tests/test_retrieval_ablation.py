from src.journals.journal_model import Journal
from src.journals.journal_store import JournalStore
from src.papers.paper_model import PaperProfile
from src.ranker.rule_scorer import RuleScorer
from src.retriever.candidate_generator import CandidateGenerator

from scripts.run_retrieval_ablation import (
    _progress_snapshot,
    build_route_results_for_variant,
    evaluate_variant,
    load_comparable_eval_papers,
)
from scripts.run_retrieval_fusion_search import build_trials, sort_trials
from scripts.run_rule_scorer_search import (
    build_rule_trials,
    evaluate_rule_trial,
    format_rule_trial_status,
    sort_rule_trials,
)


class DummyRetriever:
    def __init__(self, results):
        self.results = results

    def retrieve(self, query, top_k=30):
        return self.results[:top_k]


def test_build_route_results_for_variant_keeps_ablation_routes_isolated():
    store = JournalStore()
    scope_journal = Journal(journal_id="scope", journal_name="Scope Journal", journal_profile="scope")
    typical_journal = Journal(journal_id="typical", journal_name="Typical Journal", journal_profile="typical")
    store.add_journals([scope_journal, typical_journal])

    generator = CandidateGenerator(
        store,
        bm25_retriever=DummyRetriever([(scope_journal, 1.0)]),
        retrieval_target="typical_abstracts",
        typical_bm25_retriever=DummyRetriever([(typical_journal, 1.0)]),
        typical_text_retriever=DummyRetriever([(typical_journal, 0.8)]),
    )
    profile = PaperProfile(title="typical")
    cfg = {"bm25": 5, "vector": 5, "text": 5}
    weights = {"bm25": 0.45, "vector": 0.35, "text": 0.20}

    scope_routes = build_route_results_for_variant(generator, "scope", "typical", profile, cfg, weights)
    typical_routes = build_route_results_for_variant(generator, "typical", "typical", profile, cfg, weights)
    hybrid_routes = build_route_results_for_variant(generator, "hybrid", "typical", profile, cfg, weights)

    assert set(scope_routes) == {"scope_bm25", "scope_text"}
    assert set(typical_routes) == {"typical_bm25", "typical_text"}
    assert "scope_bm25" in hybrid_routes
    assert "typical_bm25" in hybrid_routes
    assert "identity_anchor" in hybrid_routes


def test_hybrid_route_weights_are_configurable_for_experiments():
    store = JournalStore()
    scope_journal = Journal(journal_id="scope", journal_name="Scope Journal", journal_profile="scope")
    typical_journal = Journal(journal_id="typical", journal_name="Typical Journal", journal_profile="typical")
    store.add_journals([scope_journal, typical_journal])

    generator = CandidateGenerator(
        store,
        bm25_retriever=DummyRetriever([(scope_journal, 1.0)]),
        retrieval_target="typical_abstracts",
        typical_bm25_retriever=DummyRetriever([(typical_journal, 1.0)]),
        hybrid_scope_weight=0.60,
        hybrid_typical_weight=0.40,
        identity_anchor_weight=0.08,
    )
    routes = build_route_results_for_variant(
        generator,
        "hybrid",
        "semantic query",
        PaperProfile(title="semantic query"),
        {"bm25": 5, "vector": 5, "text": 5},
        {"bm25": 0.50, "vector": 0.30, "text": 0.20},
    )

    assert routes["scope_bm25"][1] == 0.30
    assert routes["typical_bm25"][1] == 0.20
    assert routes["identity_anchor"][1] == 0.08


def test_rrf_fusion_can_promote_candidates_seen_by_multiple_routes():
    store = JournalStore()
    first = Journal(journal_id="first", journal_name="First Journal")
    repeated = Journal(journal_id="repeated", journal_name="Repeated Journal")
    store.add_journals([first, repeated])
    route_results = {
        "route_a": ([(first, 10.0), (repeated, 9.0)], 1.0),
        "route_b": ([(repeated, 1.0)], 1.0),
    }

    minmax_generator = CandidateGenerator(
        store,
        bm25_retriever=DummyRetriever([]),
        fusion_strategy="weighted_minmax",
    )
    rrf_generator = CandidateGenerator(
        store,
        bm25_retriever=DummyRetriever([]),
        fusion_strategy="rrf",
    )

    minmax_candidates, _ = minmax_generator._merge_route_results(route_results, top_k=2)
    rrf_candidates, _ = rrf_generator._merge_route_results(route_results, top_k=2)

    assert [journal.journal_id for journal in minmax_candidates] == ["first", "repeated"]
    assert [journal.journal_id for journal in rrf_candidates] == ["repeated", "first"]


def test_evaluate_variant_reports_retrieval_and_rule_metrics():
    store = JournalStore()
    target = Journal(
        journal_id="target",
        journal_name="Target Journal",
        subject_tags=["ai"],
        journal_profile="semantic target ai",
    )
    distractor = Journal(
        journal_id="distractor",
        journal_name="Distractor Journal",
        journal_profile="other",
    )
    store.add_journals([target, distractor])

    generator = CandidateGenerator(
        store,
        bm25_retriever=DummyRetriever([(distractor, 1.0)]),
        retrieval_target="typical_abstracts",
        typical_bm25_retriever=DummyRetriever([(target, 1.0)]),
    )
    scorer = RuleScorer(journals=store.journals)
    papers = [
        {
            "title": "Semantic target",
            "abstract": "AI method",
            "venue": "Target Journal",
            "research_area": ["ai"],
        }
    ]
    journal_name_to_id = {"target journal": "target"}

    result = evaluate_variant(
        papers=papers,
        generator=generator,
        scorer=scorer,
        journal_name_to_id=journal_name_to_id,
        variant="typical",
        mode="abstract",
        candidate_top_k=10,
    )

    assert result["evaluated_count"] == 1
    assert result["retrieval"]["Hit@1"] == 1
    assert result["retrieval"]["MRR"] == 1.0
    assert result["rule"]["Hit@5"] == 1
    assert result["paper_results"][0]["retrieval_rank"] == 1


def test_evaluate_rule_trial_reuses_snapshots_without_llm():
    store = JournalStore()
    target = Journal(
        journal_id="target",
        journal_name="Target Journal",
        journal_profile="target scope",
    )
    distractor = Journal(
        journal_id="distractor",
        journal_name="Distractor Journal",
        journal_profile="distractor scope",
    )
    store.add_journals([target, distractor])
    generator = CandidateGenerator(
        store,
        bm25_retriever=DummyRetriever([(target, 1.0), (distractor, 0.5)]),
        retrieval_target="scope_text",
    )
    scorer = RuleScorer(
        journals=store.journals,
        weights={"retrieval_rank_prior": 1.0},
    )
    papers = [
        {
            "title": "Metadata title",
            "abstract": "Original abstract",
            "venue": "Target Journal",
            "paper_profile_snapshot": {
                "title": "Snapshot title",
                "research_area": ["ai"],
                "keywords": ["retrieval"],
            },
        }
    ]
    name_to_id = {"target journal": "target"}

    result = evaluate_rule_trial(
        papers=papers,
        generator=generator,
        scorer=scorer,
        journal_name_to_id=name_to_id,
        mode="abstract",
        candidate_top_k=10,
    )

    assert result["evaluated_count"] == 1
    assert result["coarse_hit_at_50"] == 1
    assert result["rule_hit_at_20"] == 1
    assert result["paper_results"][0]["profile_title"] == "Snapshot title"


def test_sort_rule_trials_prioritizes_rule20_then_rule_quality():
    trials = [
        {"rule_hit_at_20": 70, "rule_mrr": 0.30, "rule_hit_at_10": 55},
        {"rule_hit_at_20": 71, "rule_mrr": 0.20, "rule_hit_at_10": 56},
        {"rule_hit_at_20": 70, "rule_mrr": 0.35, "rule_hit_at_10": 54},
    ]

    ranked = sort_rule_trials(trials)

    assert ranked[0]["rule_hit_at_20"] == 71
    assert ranked[1]["rule_mrr"] == 0.35


def test_format_rule_trial_status_includes_live_metrics():
    summary = {
        "label": "baseline",
        "coarse_hit_at_50": 92,
        "rule_hit_at_10": 56,
        "rule_hit_at_20": 71,
        "rule_mrr": 0.2412,
    }

    status = format_rule_trial_status(2, 4, summary)

    assert "trial 2/4" in status
    assert "coarse@50=92" in status
    assert "rule@20=71" in status
    assert "rule_mrr=0.2412" in status
    assert status.endswith("baseline")


def test_build_rule_trials_includes_focused_stronger_candidates():
    labels = [trial["label"] for trial in build_rule_trials()]

    assert "rank1.5|scope1.0|confirm0.6|multi0.4|area0.5|typ0.08" in labels
    assert "rank1.2|scope0.8|confirm0.8|multi0.5|area0.5|typ0.08" in labels
    assert "rank1.8|scope1.0|typRank0.6|confirm0.5|multi0.3|area0.5|typ0.06" in labels


def test_load_comparable_eval_papers_reuses_profile_snapshots(tmp_path):
    papers_path = tmp_path / "papers.jsonl"
    papers_path.write_text(
        (
            '{"title":"A Full Paper Title","abstract":"Original abstract text",'
            '"venue":"Target Journal","external_ids":{"arXiv":"1234.5678"}}\n'
            '{"title":"Skipped Paper","abstract":"Skipped abstract","venue":"Other Journal"}\n'
        ),
        encoding="utf-8",
    )
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        """
{
  "paper_results": [
    {
      "arxiv": "1234.5678",
      "title": "A Full Paper Title",
      "venue": "Target Journal",
      "recommended_journals": ["Other Journal", "Target Journal"],
      "paper_profile_snapshot": {
        "title": "Snapshot Title",
        "research_area": ["ai"],
        "keywords": ["contrastive"],
        "techniques": ["retrieval"],
        "quality_level": "B",
        "paper_strength": 0.72
      }
    }
  ]
}
""",
        encoding="utf-8",
    )

    comparable = load_comparable_eval_papers(str(papers_path), str(eval_path))

    assert len(comparable) == 1
    assert comparable[0]["abstract"] == "Original abstract text"
    assert comparable[0]["final_hit_5"] is True
    assert comparable[0]["paper_profile_snapshot"]["title"] == "Snapshot Title"


def test_evaluate_variant_reports_comparable_stage_diagnostics():
    store = JournalStore()
    target = Journal(
        journal_id="target",
        journal_name="Target Journal",
        subject_tags=["ai"],
        journal_profile="semantic target ai",
    )
    distractor = Journal(
        journal_id="distractor",
        journal_name="Distractor Journal",
        journal_profile="other",
    )
    store.add_journals([target, distractor])

    generator = CandidateGenerator(
        store,
        bm25_retriever=DummyRetriever([(target, 1.0), (distractor, 0.5)]),
        retrieval_target="scope_text",
    )
    scorer = RuleScorer(journals=store.journals)
    papers = [
        {
            "title": "Metadata title",
            "abstract": "Metadata abstract",
            "venue": "Target Journal",
            "final_hit_5": False,
            "paper_profile_snapshot": {
                "title": "Snapshot title",
                "research_area": ["ai"],
                "keywords": ["semantic"],
            },
        }
    ]

    result = evaluate_variant(
        papers=papers,
        generator=generator,
        scorer=scorer,
        journal_name_to_id={"target journal": "target"},
        variant="scope",
        mode="abstract",
        candidate_top_k=10,
    )

    assert result["coarse_hit_at_50"] == 1
    assert result["rule_hit_at_20"] == 1
    assert result["baseline_final_hit_at_5"] == 0
    assert result["miss_stage_counts"]["after_rule_top20_lost"] == 1
    assert result["route_attribution"]["scope_bm25"]["coarse_hit"] == 1
    assert "scope_bm25" in result["paper_results"][0]["target_route_attribution"]["primary_routes"]


def test_progress_snapshot_reports_running_stage_metrics():
    snapshot = _progress_snapshot(
        evaluated_count=4,
        missing_target_count=1,
        retrieval={"Hit@50": 3, "MRR": 1.5, "NDCG@5": 1.25},
        rule={"Hit@20": 2, "MRR": 1.0},
        baseline_final_hit_at_5=1,
        miss_stage_counts={"not_in_top50": 1, "rule_suppressed": 2},
    )

    assert snapshot["coarse@50"] == "3/4 (75.0%)"
    assert snapshot["rule@20"] == "2/4 (50.0%)"
    assert snapshot["base@5"] == "1/4 (25.0%)"
    assert snapshot["ret_mrr"] == "0.375"
    assert snapshot["missing"] == "1"
    assert snapshot["miss_top50"] == "1"


def test_fusion_search_trials_cover_strategy_weights_and_route_config():
    trials = build_trials("abstract", max_trials=2)

    assert len(trials) == 2
    assert trials[0]["fusion_strategy"] == "weighted_minmax"
    assert trials[0]["hybrid_scope_weight"] == 0.55
    assert trials[0]["hybrid_typical_weight"] == 0.45
    assert trials[0]["route_config"]["bm25"] == 28
    assert trials[0]["route_config"]["vector"] == 28


def test_fusion_search_sorts_by_coarse_hit_then_retrieval_quality():
    ranked = sort_trials([
        {"coarse_hit_at_50": 90, "retrieval_mrr": 0.40, "retrieval_ndcg_at_5": 0.30, "rule_hit_at_20": 70},
        {"coarse_hit_at_50": 91, "retrieval_mrr": 0.20, "retrieval_ndcg_at_5": 0.10, "rule_hit_at_20": 65},
        {"coarse_hit_at_50": 90, "retrieval_mrr": 0.45, "retrieval_ndcg_at_5": 0.30, "rule_hit_at_20": 68},
    ])

    assert ranked[0]["coarse_hit_at_50"] == 91
    assert ranked[1]["retrieval_mrr"] == 0.45
