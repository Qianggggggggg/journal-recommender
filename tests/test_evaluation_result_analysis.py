import hashlib
import json
from pathlib import Path

from scripts.run_evaluation import EvaluationResult, save_results


def test_save_results_records_reproducible_config_artifacts_and_diagnostics(tmp_path):
    benchmark = tmp_path / "holdout.jsonl"
    benchmark.write_text('{"title":"A"}\n', encoding="utf-8")
    model = tmp_path / "ltr.json"
    model.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_type": "logistic_regression",
                "feature_dim": 16,
                "feature_names": ["retrieval_rank", "rule_rank"],
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )

    paper_results = [
        {
            "title": "Paper B",
            "venue": "Journal B",
            "hit_1": False,
            "hit_3": True,
            "hit_5": True,
            "hit_10": True,
            "relevant_rank": 3,
            "relevant_rank_at_10": 3,
            "latency_seconds": 2.0,
            "coarse_hit": True,
            "coarse_hit_in_rule_top10": True,
            "coarse_hit_in_rule_top20": True,
            "evaluation_status": "ok",
            "rank_method": "llm_evidence_learned",
            "final_rank_source": "llm_evidence_learned",
            "llm_evidence_status": "precomputed",
            "llm_evidence_coverage": 1.0,
            "venue_diagnostic": {
                "target_journal_id": "jb",
                "wide_retrieval_rank": 2,
                "in_llm_pool": True,
                "miss_stage": "final_hit",
            },
        },
        {
            "title": "Paper A",
            "venue": "Journal A",
            "hit_1": False,
            "hit_3": False,
            "hit_5": False,
            "hit_10": False,
            "relevant_rank": 0,
            "relevant_rank_at_10": 0,
            "latency_seconds": 4.0,
            "coarse_hit": True,
            "coarse_hit_in_rule_top10": False,
            "coarse_hit_in_rule_top20": False,
            "evaluation_status": "fallback",
            "rank_method": "rule_fallback",
            "venue_diagnostic": {
                "target_journal_id": "ja",
                "wide_retrieval_rank": 8,
                "in_llm_pool": False,
                "miss_stage": "rule_suppressed",
            },
        },
    ]
    result = EvaluationResult(
        total_count=3,
        mode="abstract",
        top_k=5,
        hit_at_1=0,
        hit_at_3=1,
        hit_at_5=1,
        hit_at_10=1,
        area_match_count=1,
        area_subject_tag_match_count=1,
        level_match_count=2,
        mrr=1 / 3,
        ndcg_at_5=0.5,
        ndcg_at_10=0.5,
        elapsed_seconds=6.0,
        coarse_hit_count=2,
        coarse_hit_in_rule_top10_count=1,
        coarse_hit_in_rule_top20_count=1,
        fallback_count=1,
        llm_success_count=1,
        empty_recommendation_count=0,
        level_a_count=2,
        level_a_hit_at_5=1,
        by_area={"AI": {"total": 2, "hit": 1, "area_match": 1}},
        by_level={"A": {"total": 2, "hit": 1}},
        paper_results=paper_results,
    )
    app_config = {
        "minimax": {
            "api_key": "must-not-leak",
            "model": "MiniMax-M2.7",
            "temperature": 0.2,
        },
        "ollama": {
            "embedding_model": "qwen3-embedding:4b",
            "embedding_query_instruction": "query instruction",
        },
        "candidate_generator": {
            "hybrid_scope_weight": 0.65,
            "hybrid_typical_weight": 0.35,
        },
        "ranking": {
            "evidence_role": {
                "enabled": True,
                "evidence_weight": 0.55,
                "snapshot_path": str(tmp_path / "missing-evidence.json"),
            },
            "future_secret_token": "also-must-not-leak",
        },
        "data": {},
    }
    output = tmp_path / "result.json"

    save_results(
        result,
        filepath=str(output),
        benchmark_profile="holdout240",
        benchmark_path=str(benchmark),
        app_config=app_config,
        ltr_info={
            "enabled": True,
            "model_path": str(model),
            "model_converged": True,
        },
        benchmark_manifest={"app_config_hash": "app", "prompt_hash": "prompt"},
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["result_schema_version"] == 2
    assert payload["experiment_config"]["minimax"]["api_key"] == "<redacted>"
    assert (
        payload["experiment_config"]["ranking"]["future_secret_token"]
        == "<redacted>"
    )
    assert (
        payload["experiment_config"]["candidate_generator"][
            "hybrid_scope_weight"
        ]
        == 0.65
    )
    assert payload["metrics"]["level_a_hit_at_5_rate"] == 0.5
    assert payload["metrics"]["rule_retention_at_20_given_retrieved"] == 0.5
    assert payload["metrics"]["ndcg_at_10"] == 0.5 / 3
    assert payload["metrics"]["processed_count"] == 2
    assert payload["metrics"]["failed_or_skipped_count"] == 1
    assert payload["analysis"]["miss_stage_distribution"] == {
        "final_hit": 1,
        "rule_suppressed": 1,
    }
    assert payload["analysis"]["stage_funnel"]["retrieved_top50"] == {
        "count": 2,
        "rate": 0.6667,
    }
    assert payload["analysis"]["latency_seconds"]["median"] == 3.0
    assert payload["analysis"]["by_area"]["AI"]["hit_at_5_rate"] == 0.5
    assert [paper["title"] for paper in payload["paper_results"]] == [
        "Paper A",
        "Paper B",
    ]
    assert payload["paper_results"][1]["relevant_rank"] == 3

    benchmark_artifact = payload["artifacts"]["benchmark_input"]
    assert benchmark_artifact["sha256"] == hashlib.sha256(
        benchmark.read_bytes()
    ).hexdigest()
    assert payload["artifacts"]["ltr_model"]["model_metadata"]["feature_dim"] == 16
    assert payload["artifacts"]["evidence_snapshot"]["exists"] is False
