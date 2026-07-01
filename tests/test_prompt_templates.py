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


def test_llm_ranker_reasons_use_public_facing_labels():
    """Reasons use stable user-facing labels while hiding internal jargon."""
    prompts = _load_prompts()
    ranker_user = prompts["llm_ranker_user"]

    assert "推荐理由必须以\"类型标签：\"开头" in ranker_user
    assert "Scope对齐：" in ranker_user
    assert "技术方法契合：" in ranker_user
    assert "禁止使用以下内部术语" in prompts["llm_ranker_system"]


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


def test_llm_evidence_extractor_v2_has_ccf_tier_calibration_clauses():
    """v2 evidence prompt must add CCF-tier calibration to counter C-tier under-scoring.

    Locked behaviors (P0 from 2026-06-16 diagnostic):
      * Keep v1 untouched (so A/B comparison stays possible)
      * v2 system prompt contains explicit CCF-C / 应用导向 guidance
      * v2 user prompt can format with the same template variables as v1
      * v2 still requires the original 6 evidence fields + JSON shape
    """
    prompts = _load_prompts()

    # v1 must still exist (A/B reference)
    assert "llm_evidence_extractor_system" in prompts
    assert "llm_evidence_extractor_user" in prompts

    # v2 keys must exist
    assert "llm_evidence_extractor_system_v2" in prompts, "v2 system prompt missing"
    assert "llm_evidence_extractor_user_v2" in prompts, "v2 user prompt missing"

    v2_system = prompts["llm_evidence_extractor_system_v2"]
    v2_user = prompts["llm_evidence_extractor_user_v2"]

    # CCF-C calibration clauses must be present and concrete
    # These are the specific clauses that broke C-tier composite into the 0.0-0.3 range
    # for application-oriented C-tier journals (HCI, security engineering, education).
    assert "CCF-C" in v2_system or "ccf_rating" in v2_system, (
        "v2 must reference CCF-C tier or ccf_rating to enable tier calibration"
    )
    assert "application_fit" in v2_system, (
        "v2 must mention application_fit since C-tier calibration targets this field"
    )
    # 强匹配阈值引导: 当 application_fit 0.7+ 时不要再被 method_fit 拖累
    assert "0.7" in v2_system or "强匹配" in v2_system, (
        "v2 must provide concrete threshold guidance for what counts as 'strong match' on C-tier"
    )

    # v2 still requires the 6 evidence fields
    for field in (
        "scope_fit",
        "method_fit",
        "application_fit",
        "journal_position_fit",
        "too_broad_penalty",
        "too_narrow_penalty",
    ):
        assert field in v2_system, f"v2 system prompt missing required field: {field}"

    # v2 user prompt must still format with the same variables as v1
    formatted = v2_user.format(
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
    assert "Test Paper" in formatted
    assert "scope_fit" in formatted or "evidence" in formatted


def test_llm_evidence_extractor_v2_user_prompt_passes_journal_ccf_rating():
    """v2 user prompt must inject each journal's ccf_rating so LLM can calibrate per-journal."""
    prompts = _load_prompts()
    v2_user = prompts["llm_evidence_extractor_user_v2"]

    formatted = v2_user.format(
        title="T",
        abstract="A",
        research_area="人工智能",
        ccf_research_area="人工智能",
        method_type="method",
        paper_type="application",
        keywords="k",
        novelty="n",
        application_domain="d",
        techniques="t",
        datasets="d",
        evaluation_metrics="m",
        novelty_type="new_method",
        # include a ccf_rating marker in one of the candidates
        journals_info='[{"journal_id":"j1","ccf_rating":"C","scope":"x"}]',
        total_candidates=1,
    )
    # v2 must preserve the ccf_rating info from journals_info
    assert "C" in formatted
    assert "j1" in formatted


class TestSelectEvidencePrompts:
    """P0 wiring: select_evidence_prompts picks the right v1/v2 key set.

    The function is the single source of truth for which evidence prompt
    version to use. api.py and run_evaluation.py both call it.
    """

    def test_v1_returns_v1_keys(self):
        from src.ranker.llm_evidence_extractor import select_evidence_prompts

        sys_p, usr_p = select_evidence_prompts(
            {
                "llm_evidence_extractor_system": "SYS_V1",
                "llm_evidence_extractor_user": "USR_V1",
            },
            "v1",
        )
        assert sys_p == "SYS_V1"
        assert usr_p == "USR_V1"

    def test_v2_returns_v2_keys(self):
        from src.ranker.llm_evidence_extractor import select_evidence_prompts

        prompts = _load_prompts()
        sys_p, usr_p = select_evidence_prompts(prompts, "v2")
        assert sys_p == prompts["llm_evidence_extractor_system_v2"]
        assert usr_p == prompts["llm_evidence_extractor_user_v2"]
        # v2 system must contain the CCF-C calibration clauses
        assert "CCF-C" in sys_p or "ccf_rating" in sys_p
        assert "application_fit" in sys_p

    def test_unknown_version_falls_back_to_v1(self):
        """A typo like 'v3' in app.yaml must not crash the pipeline."""
        from src.ranker.llm_evidence_extractor import select_evidence_prompts

        sys_p, usr_p = select_evidence_prompts(
            {
                "llm_evidence_extractor_system": "SYS_V1",
                "llm_evidence_extractor_user": "USR_V1",
            },
            "v3",
        )
        assert sys_p == "SYS_V1"
        assert usr_p == "USR_V1"

    def test_v2_missing_keys_falls_back_to_v1(self):
        """If a stale prompts.yaml lacks the v2 keys, fall back gracefully."""
        from src.ranker.llm_evidence_extractor import select_evidence_prompts

        sys_p, usr_p = select_evidence_prompts(
            {
                "llm_evidence_extractor_system": "SYS_V1",
                "llm_evidence_extractor_user": "USR_V1",
                # v2 keys deliberately absent
            },
            "v2",
        )
        assert sys_p == "SYS_V1"
        assert usr_p == "USR_V1"
