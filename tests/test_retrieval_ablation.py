from src.journals.journal_model import Journal
from src.journals.journal_store import JournalStore
from src.papers.paper_model import PaperProfile
from src.ranker.rule_scorer import RuleScorer
from src.retriever.candidate_generator import CandidateGenerator

import pytest

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


def test_build_route_results_for_accepted_paper_variants():
    """任务 3.3 新增 5 个变体:accepted / scope_typical / scope_accepted /
    typical_accepted / full_hybrid 必须各自只包含正确的路由集合。"""
    store = JournalStore()
    scope_journal = Journal(journal_id="scope", journal_name="Scope J", journal_profile="scope")
    typical_journal = Journal(journal_id="typ", journal_name="Typical J", journal_profile="typical")
    accepted_journal = Journal(journal_id="acc", journal_name="Accepted J", journal_profile="accepted")
    store.add_journals([scope_journal, typical_journal, accepted_journal])

    generator = CandidateGenerator(
        store,
        bm25_retriever=DummyRetriever([(scope_journal, 1.0)]),
        retrieval_target="typical_abstracts",
        typical_bm25_retriever=DummyRetriever([(typical_journal, 1.0)]),
        accepted_bm25_retriever=DummyRetriever([(accepted_journal, 1.0)]),
        accepted_embedding_retriever=DummyRetriever([(accepted_journal, 0.9)]),
    )
    profile = PaperProfile(title="x")
    cfg = {"bm25": 5, "vector": 5, "text": 5, "accepted_bm25": 5, "accepted_vector": 5}
    weights = {"bm25": 0.45, "vector": 0.35, "text": 0.20}

    accepted = build_route_results_for_variant(generator, "accepted", "x", profile, cfg, weights)
    assert set(accepted) == {"accepted_bm25", "accepted_vector"}

    scope_typical = build_route_results_for_variant(generator, "scope_typical", "x", profile, cfg, weights)
    assert "scope_bm25" in scope_typical
    assert "typical_bm25" in scope_typical
    assert "accepted_bm25" not in scope_typical

    scope_accepted = build_route_results_for_variant(generator, "scope_accepted", "x", profile, cfg, weights)
    assert "scope_bm25" in scope_accepted
    assert "accepted_bm25" in scope_accepted
    assert "typical_bm25" not in scope_accepted

    typical_accepted = build_route_results_for_variant(generator, "typical_accepted", "x", profile, cfg, weights)
    assert "typical_bm25" in typical_accepted
    assert "accepted_bm25" in typical_accepted
    assert "scope_bm25" not in typical_accepted

    full_hybrid = build_route_results_for_variant(generator, "full_hybrid", "x", profile, cfg, weights)
    assert {"scope_bm25", "typical_bm25", "accepted_bm25", "identity_anchor"} <= set(full_hybrid)


def test_ablation_runner_wires_accepted_retrievers_when_corpus_present(tmp_path):
    """回归测试:run_retrieval_ablation.build_candidate_generator 必须把 accepted
    retrievers 接进 CandidateGenerator。任务 3.3 commit 3515949 漏了这步,
    导致 light30 ablation 中 accepted / scope_accepted / full_hybrid 全部静默
    退化 (accepted variant coarse@50=0,scope_accepted == scope,
    full_hybrid == hybrid)。"""
    import json

    from scripts.run_retrieval_ablation import build_candidate_generator

    # 准备 minimal 但完整的 fixture
    accepted_dir = tmp_path / "accepted_papers"
    accepted_dir.mkdir()
    (accepted_dir / "ai.json").write_text(
        json.dumps({
            "journal_id": "ai",
            "journal_name": "Artificial Intelligence",
            "papers": [
                {"title": "P1", "abstract": "reasoning framework"},
                {"title": "P2", "abstract": "graph reasoning"},
            ],
        }),
        encoding="utf-8",
    )

    typical_dir = tmp_path / "typical_abstracts"
    typical_dir.mkdir()
    (typical_dir / "ai.json").write_text(
        json.dumps({
            "journal_id": "ai",
            "journal_name": "Artificial Intelligence",
            "abstracts": [{"abstract": "AI typical abstract", "method_type": "x", "novelty_level": "y"}],
        }),
        encoding="utf-8",
    )

    journal_store_path = tmp_path / "journals.jsonl"
    journal_store_path.write_text(
        json.dumps({"journal_id": "ai", "journal_name": "Artificial Intelligence", "scope_text": "AI"}) + "\n",
        encoding="utf-8",
    )

    app_config = {
        "data": {
            "journal_store_path": str(journal_store_path),
            "typical_abstracts_dir": str(typical_dir),
            "accepted_papers_dir": str(accepted_dir),
            # 索引文件不存在,embedding retriever 应自动 None,
            # 但 BM25 必须被构建并注入
            "accepted_papers_faiss_path": str(tmp_path / "no.faiss"),
            "accepted_papers_metadata_path": str(tmp_path / "no.parquet"),
        },
        "candidate_generator": {
            "retrieval_target": "typical_abstracts",
        },
        "ollama": {"base_url": "http://localhost:11434", "embedding_model": "qwen3-embedding:4b"},
    }

    # include_vector=False 避免依赖 Ollama 在线
    generator = build_candidate_generator(app_config, include_vector=False)

    assert generator.accepted_bm25_retriever is not None, (
        "ablation runner 必须把 accepted_bm25_retriever 接进 CandidateGenerator"
    )
    # embedding 索引不存在,vector retriever 应为 None (graceful)
    assert generator.accepted_embedding_retriever is None
    # 默认 accepted_paper_weight 应从 config 透传 (默认 0.20)
    assert generator.accepted_paper_weight == pytest.approx(0.20)


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


def test_evaluate_variant_persists_features_per_candidate():
    """evaluate_variant 输出 JSON 必须包含 feature_names + 每篇 paper 的 candidate_features。

    这是 4.1.e 的核心:features 必须能落到磁盘,4.1.f 才能拿 ablation JSON 转 LTR 训练数据。
    """
    from src.ranker.feature_builder import FEATURE_NAMES

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

    # 1. feature_names 出现在 variant 顶层
    assert result.get("feature_names") == FEATURE_NAMES
    # 2. 每篇 paper 的 candidate_features 是 {jid: [floats]} 字典
    pr = result["paper_results"][0]
    assert "candidate_features" in pr
    cf = pr["candidate_features"]
    assert isinstance(cf, dict)
    # 3. 至少 target 期刊有 features 列表
    assert "target" in cf
    assert isinstance(cf["target"], list)
    assert len(cf["target"]) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in cf["target"])


def test_evaluate_variant_persists_features_even_without_accepted_store():
    """accepted_paper_store 缺省时,features 也要落地(candidate_in_accepted_corpus=0.0)。"""
    from src.ranker.feature_builder import FEATURE_NAMES

    store = JournalStore()
    target = Journal(journal_id="target", journal_name="Target Journal", journal_profile="x")
    store.add_journal(target)

    generator = CandidateGenerator(
        store,
        bm25_retriever=DummyRetriever([(target, 1.0)]),
        retrieval_target="typical_abstracts",
        typical_bm25_retriever=DummyRetriever([(target, 1.0)]),
    )
    scorer = RuleScorer(journals=store.journals)
    papers = [{"title": "T", "abstract": "A", "venue": "Target Journal", "research_area": []}]
    journal_name_to_id = {"target journal": "target"}

    result = evaluate_variant(
        papers=papers,
        generator=generator,
        scorer=scorer,
        journal_name_to_id=journal_name_to_id,
        variant="typical",
        mode="abstract",
        candidate_top_k=5,
    )

    pr = result["paper_results"][0]
    corpus_idx = FEATURE_NAMES.index("candidate_in_accepted_corpus")
    # accepted store 缺省 → 全部候选 candidate_in_accepted_corpus=0.0
    for jid, feats in pr["candidate_features"].items():
        assert feats[corpus_idx] == 0.0


def test_evaluate_variant_persists_rule_top20_for_hard_negative_mining():
    """paper_result 必须含 rule_top20(完整 20 名),供硬负样本分类用(per plan 4.2)。"""
    store = JournalStore()
    target = Journal(journal_id="target", journal_name="Target Journal", journal_profile="x")
    distractor = Journal(journal_id="distractor", journal_name="Distractor", journal_profile="y")
    store.add_journals([target, distractor])

    generator = CandidateGenerator(
        store,
        bm25_retriever=DummyRetriever([(distractor, 1.0)]),
        retrieval_target="typical_abstracts",
        typical_bm25_retriever=DummyRetriever([(target, 1.0)]),
    )
    scorer = RuleScorer(journals=store.journals)
    papers = [{"title": "T", "abstract": "A", "venue": "Target Journal", "research_area": []}]
    journal_name_to_id = {"target journal": "target"}

    result = evaluate_variant(
        papers=papers,
        generator=generator,
        scorer=scorer,
        journal_name_to_id=journal_name_to_id,
        variant="typical",
        mode="abstract",
        candidate_top_k=5,
    )

    pr = result["paper_results"][0]
    assert "rule_top20" in pr
    assert isinstance(pr["rule_top20"], list)
    # rule_top20 至少要包含 rule_top5(向后兼容)
    assert "rule_top5" in pr
    assert set(pr["rule_top5"]).issubset(set(pr["rule_top20"]))


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
