from scripts.run_llm_rerank_ablation import LLM_RERANK_ABLATION_VARIANTS


def test_llm_rerank_ablation_defines_required_variants():
    assert set(LLM_RERANK_ABLATION_VARIANTS) == {
        "old_prompt",
        "current_prompt",
        "current_prompt_rule_route_fusion",
        "current_prompt_strong_anchor_guard",
    }

    assert LLM_RERANK_ABLATION_VARIANTS["old_prompt"].prompt_path == "configs/prompts_llm_legacy.yaml"
    assert LLM_RERANK_ABLATION_VARIANTS["current_prompt"].llm_final_selection["enabled"] is False
    assert LLM_RERANK_ABLATION_VARIANTS["current_prompt_rule_route_fusion"].llm_final_selection["strategy"] == "rule_route_fusion"
    assert LLM_RERANK_ABLATION_VARIANTS["current_prompt_strong_anchor_guard"].llm_anchor_guard["protect_rule_rank"] == 15
