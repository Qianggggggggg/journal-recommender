"""Prompt template regression tests."""
import yaml


def _load_prompts() -> dict:
    with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_quality_assessor_user_prompt_formats_with_json_example():
    """Quality prompt JSON examples must not be treated as format placeholders."""
    prompts = _load_prompts()

    formatted = prompts["paper_quality_assessor_user"].format(
        title="Test Paper",
        abstract="A short abstract.",
        full_text_summary="",
        research_area="人工智能",
        method_type="method",
        keywords="machine learning",
        techniques="transformer",
        datasets="ImageNet",
        evaluation_metrics="accuracy",
        novelty_type="new_method",
    )

    assert '"novelty_score": 2' in formatted


def test_llm_ranker_user_prompt_formats_with_json_example():
    """Ranker prompt JSON examples must not be treated as format placeholders."""
    prompts = _load_prompts()

    formatted = prompts["llm_ranker_user"].format(
        title="Test Paper",
        research_area="人工智能",
        method_type="method",
        paper_type="application",
        keywords="machine learning",
        novelty="A new method.",
        application_domain="general AI",
        techniques="transformer",
        datasets="ImageNet",
        evaluation_metrics="accuracy",
        novelty_type="new_method",
        journals_info="[]",
        total_candidates=0,
    )

    assert '"rankings": [' in formatted


def test_llm_ranker_reasons_are_not_forced_to_fixed_labels():
    """Recommendation reasons should be natural advice, not rigid label templates."""
    prompts = _load_prompts()
    ranker_user = prompts["llm_ranker_user"]

    assert "不要机械地以\"Scope对齐：\"" in ranker_user
    assert "推荐理由（2-4条" in ranker_user
    assert "不要出现内部证据字段名" in ranker_user
    assert "强候选保护" in ranker_user
    assert "network" in ranker_user or "网络协议" in ranker_user
    assert "推荐理由必须以\"类型标签：\"开头" not in ranker_user


def test_llm_evidence_extractor_prompt_formats_with_json_example():
    """Evidence extractor prompt must format its JSON contract literally."""
    prompts = _load_prompts()

    formatted = prompts["llm_evidence_extractor_user"].format(
        title="Test Paper",
        abstract="A short abstract.",
        research_area="人工智能",
        ccf_research_area="人工智能",
        method_type="method",
        paper_type="application",
        keywords="machine learning",
        novelty="A new method.",
        application_domain="general AI",
        techniques="transformer",
        datasets="ImageNet",
        evaluation_metrics="accuracy",
        novelty_type="new_method",
        journals_info="[]",
        total_candidates=0,
    )

    assert "scope_fit" in prompts["llm_evidence_extractor_system"]
    assert "A short abstract." in formatted
    assert "journal_position_fit" in formatted
    assert "too_broad_penalty" in formatted
    assert '"evidence": [' in formatted
