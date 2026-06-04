"""Tests for Task 6.3 LLM role ablation."""
from __future__ import annotations

import json
import re
from copy import deepcopy

import pytest

from scripts.run_llm_role_ablation import (
    LLM_ROLE_VARIANTS,
    compare_variant_fairness,
    load_evidence_snapshot,
)
from scripts.precompute_evidence import select_snapshot_candidates
from src.journals.journal_model import Journal
from src.papers.paper_model import PaperProfile
from src.ranker.feature_builder import (
    FEATURE_NAMES,
    FEATURE_NAMES_WITH_LLM_EVIDENCE,
)
from src.ranker.llm_evidence_role_ranker import (
    DirectLLMRoleRanker,
    LLMEvidenceRoleRanker,
)


class FixedEvidenceExtractor:
    def __init__(self, evidence):
        self.evidence = evidence
        self.calls = 0
        self.last_rule_ranks = None

    def extract(self, candidates, paper_profile, rule_ranks=None):
        self.calls += 1
        self.last_rule_ranks = rule_ranks
        return deepcopy(self.evidence)


class FailingEvidenceExtractor:
    def extract(self, candidates, paper_profile):
        raise RuntimeError("evidence service unavailable")


class StubJournalStore:
    def __init__(self, journals):
        self.by_id = {journal.journal_id: journal for journal in journals}

    def get_journal(self, journal_id):
        return self.by_id.get(journal_id)


class FixedDirectRanker:
    def rank(self, candidates, paper_profile, top_k=5, retrieval_trace=None):
        ranked = [
            (journal, 0.9 - index * 0.1, ["direct"], 0.8)
            for index, (journal, _score, _reasons) in enumerate(candidates)
        ]
        return ranked[:top_k], "llm"


def _journal(journal_id: str) -> Journal:
    return Journal(
        journal_id=journal_id,
        journal_name=journal_id.upper(),
        ccf_rating="B",
        subject_tags=["人工智能"],
    )


def _evidence(scope_fit: float) -> dict:
    return {
        "scope_fit": scope_fit,
        "method_fit": scope_fit,
        "application_fit": scope_fit,
        "journal_position_fit": scope_fit,
        "too_broad_penalty": 0.0,
        "too_narrow_penalty": 0.0,
        "evidence": [f"scope fit {scope_fit}"],
    }


def _trace(journals):
    return {
        journal.journal_id: {
            "retrieval_rank": index + 1,
            "routes": {},
            "primary_routes": [],
        }
        for index, journal in enumerate(journals)
    }


def test_evidence_role_ranker_combines_ltr_score_with_evidence_and_rank():
    """With ltr_score_weight=0.3, the final_score formula uses
    evidence (0.5*0.8) + rank_prior (0.2*1.0) + ltr_score (0.3*0.7) = 0.81
    for journal with evidence_composite=0.8, rank=1/1, ltr_score=0.7.

    Weights are renormalized so the 3 components sum to 1.0:
    0.5/(0.5+0.2+0.3) = 0.50
    0.2/(0.5+0.2+0.3) = 0.20
    0.3/(0.5+0.2+0.3) = 0.30
    """
    ranker = LLMEvidenceRoleRanker(
        evidence_extractor=FixedEvidenceExtractor({"j1": _evidence(0.8)}),
        journal_store=StubJournalStore([_journal("j1")]),
        prior_source="learned",
        evidence_weight=0.5,
        prior_weight=0.2,
        ltr_score_weight=0.3,
    )
    ranked, _m, diag = ranker.rank_with_diagnostics(
        candidates=[(_journal("j1"), 1.0, [])],
        paper_profile=PaperProfile(title="P1"),
        retrieval_trace=_trace([_journal("j1")]),
        rule_ranks={"j1": 1},
        rule_scores={"j1": 1.0},
        learned_ranks={"j1": 1},  # rank 1 of 1 -> prior 1.0
        learned_scores={"j1": 0.7},
    )
    # 0.50*0.8 + 0.20*1.0 + 0.30*0.7 = 0.40 + 0.20 + 0.21 = 0.81
    assert diag["candidates"]["j1"]["final_score"] == pytest.approx(0.81, abs=0.01)
    # Diagnostic fields should expose the LTR contribution
    detail = diag["candidates"]["j1"]
    assert detail["ltr_score"] == pytest.approx(0.7)
    assert detail["ltr_score_weight"] == pytest.approx(0.30, abs=0.01)


def test_evidence_role_ranker_zero_ltr_score_weight_matches_legacy_formula():
    """ltr_score_weight=0.0 must reproduce the legacy 2-component formula
    (evidence*W + rank_prior*(1-W)), and the diagnostic ltr_score must be 0.0
    when learned_scores is omitted or empty."""
    ranker = LLMEvidenceRoleRanker(
        evidence_extractor=FixedEvidenceExtractor({"j1": _evidence(0.6)}),
        journal_store=StubJournalStore([_journal("j1")]),
        prior_source="rule",
        evidence_weight=0.8,
        prior_weight=0.2,
        ltr_score_weight=0.0,
    )
    _ranked, _m, diag = ranker.rank_with_diagnostics(
        candidates=[(_journal("j1"), 1.0, [])],
        paper_profile=PaperProfile(title="P2"),
        retrieval_trace=_trace([_journal("j1")]),
        rule_ranks={"j1": 1},
        rule_scores={"j1": 1.0},
    )
    detail = diag["candidates"]["j1"]
    # Legacy: 0.8*0.6 + 0.2*1.0 = 0.68
    assert detail["final_score"] == pytest.approx(0.68, abs=0.01)
    assert detail["ltr_score"] == 0.0
    assert detail["ltr_score_weight"] == 0.0


def test_llm_role_variants_define_direct_rule_and_learned_roles():
    assert set(LLM_ROLE_VARIANTS) == {
        "llm_ranker_direct",
        "llm_evidence_plus_rule",
        "llm_evidence_plus_learned_reranker",
    }
    assert LLM_ROLE_VARIANTS["llm_ranker_direct"].ranker_role == "direct"
    assert LLM_ROLE_VARIANTS["llm_evidence_plus_rule"].prior_source == "rule"
    assert LLM_ROLE_VARIANTS["llm_evidence_plus_learned_reranker"].prior_source == "learned"
    assert LLM_ROLE_VARIANTS["llm_evidence_plus_learned_reranker"].ltr_enabled is True


def test_direct_role_wrapper_persists_complete_candidate_scores_without_shared_state():
    journals = [_journal("j1"), _journal("j2")]
    ranker = DirectLLMRoleRanker(FixedDirectRanker())

    ranked, method, diagnostics = ranker.rank_with_diagnostics(
        candidates=[(journals[0], 1.0, []), (journals[1], 0.8, [])],
        paper_profile=PaperProfile(title="T"),
        top_k=2,
        retrieval_trace=_trace(journals),
        rule_ranks={"j1": 1, "j2": 2},
        rule_scores={"j1": 1.0, "j2": 0.8},
    )

    assert method == "llm"
    assert len(ranked) == 2
    assert diagnostics["role"] == "direct"
    assert diagnostics["candidates"]["j1"]["final_rank"] == 1
    assert diagnostics["candidates"]["j2"]["llm_score"] == pytest.approx(0.8)
    assert not hasattr(ranker, "last_diagnostics")


def test_evidence_role_ranker_combines_evidence_and_linear_rank_prior():
    journals = [_journal("j1"), _journal("j2")]
    extractor = FixedEvidenceExtractor(
        {"j1": _evidence(0.5), "j2": _evidence(0.9)}
    )
    ranker = LLMEvidenceRoleRanker(
        evidence_extractor=extractor,
        journal_store=StubJournalStore(journals),
        prior_source="rule",
        evidence_weight=0.8,
        prior_weight=0.2,
    )

    ranked, method, diagnostics = ranker.rank_with_diagnostics(
        candidates=[(journals[0], 1.0, []), (journals[1], 0.8, [])],
        paper_profile=PaperProfile(title="T"),
        top_k=2,
        retrieval_trace=_trace(journals),
        rule_ranks={"j1": 1, "j2": 2},
        rule_scores={"j1": 1.0, "j2": 0.8},
    )

    assert extractor.calls == 1
    assert method == "llm_evidence_rule"
    assert [item[0].journal_id for item in ranked] == ["j2", "j1"]
    assert diagnostics["status"] == "ok"
    assert diagnostics["candidates"]["j1"]["rank_prior"] == 1.0
    assert diagnostics["candidates"]["j2"]["rank_prior"] == 0.0
    assert diagnostics["candidates"]["j1"]["final_score"] == pytest.approx(0.6)
    assert diagnostics["candidates"]["j2"]["final_score"] == pytest.approx(0.72)


def test_evidence_role_ranker_uses_neutral_evidence_when_extractor_fails():
    journals = [_journal("j1"), _journal("j2")]
    ranker = LLMEvidenceRoleRanker(
        evidence_extractor=FailingEvidenceExtractor(),
        journal_store=StubJournalStore(journals),
        prior_source="rule",
    )

    ranked, _method, diagnostics = ranker.rank_with_diagnostics(
        candidates=[(journals[0], 1.0, []), (journals[1], 0.8, [])],
        paper_profile=PaperProfile(title="T"),
        top_k=2,
        retrieval_trace=_trace(journals),
        rule_ranks={"j1": 1, "j2": 2},
        rule_scores={"j1": 1.0, "j2": 0.8},
    )

    assert diagnostics["status"] == "neutral_fallback"
    assert diagnostics["fallback_reason"]
    assert [item[0].journal_id for item in ranked] == ["j1", "j2"]
    assert diagnostics["candidates"]["j1"]["evidence_composite"] == 0.5
    assert diagnostics["candidates"]["j1"]["llm_scope_fit"] == 0.5
    assert diagnostics["candidates"]["j1"]["llm_too_broad_penalty"] == 0.0


def test_evidence_role_ranker_persists_20_and_26_dim_candidate_features():
    journals = [_journal("j1")]
    ranker = LLMEvidenceRoleRanker(
        evidence_extractor=FixedEvidenceExtractor({"j1": _evidence(0.8)}),
        journal_store=StubJournalStore(journals),
        prior_source="learned",
    )

    _ranked, _method, diagnostics = ranker.rank_with_diagnostics(
        candidates=[(journals[0], 1.0, [])],
        paper_profile=PaperProfile(title="T", paper_strength=0.7),
        top_k=1,
        retrieval_trace=_trace(journals),
        rule_ranks={"j1": 3},
        rule_scores={"j1": 0.9},
        learned_ranks={"j1": 2},
    )

    detail = diagnostics["candidates"]["j1"]
    assert detail["feature_names_base"] == FEATURE_NAMES
    assert detail["feature_names_with_llm_evidence"] == FEATURE_NAMES_WITH_LLM_EVIDENCE
    assert len(detail["features_base"]) == 20
    assert len(detail["features_with_llm_evidence"]) == 26
    assert detail["prior_source"] == "learned"
    assert not hasattr(ranker, "last_diagnostics")


def _paper_result(title, venue, coarse_hit, rule20):
    return {
        "title": title,
        "venue": venue,
        "coarse_hit": coarse_hit,
        "coarse_hit_in_rule_top20": rule20,
    }


def test_compare_variant_fairness_passes_for_identical_per_paper_stages():
    baseline = [
        _paper_result("A", "J1", True, True),
        _paper_result("B", "J2", False, False),
    ]
    comparison = list(reversed(deepcopy(baseline)))

    report = compare_variant_fairness(
        {"direct": baseline, "evidence": comparison}
    )

    assert report["fairness_pass"] is True
    assert report["denominator_match"] is True
    assert report["coarse_mismatches"] == []
    assert report["rule_top20_mismatches"] == []


def test_compare_variant_fairness_reports_per_paper_stage_mismatch():
    report = compare_variant_fairness(
        {
            "direct": [_paper_result("A", "J1", True, True)],
            "evidence": [_paper_result("A", "J1", False, True)],
        }
    )

    assert report["fairness_pass"] is False
    assert report["coarse_mismatches"] == ["A | J1"]


def _evidence_paper_result(title, venue, scope_fit, coverage=1.0):
    return {
        **_paper_result(title, venue, True, True),
        "llm_role": "evidence",
        "llm_evidence_coverage": coverage,
        "llm_candidates_detail": [
            {
                "journal_id": "j1",
                "llm_scope_fit": scope_fit,
                "llm_method_fit": 0.7,
                "llm_application_fit": 0.6,
                "llm_journal_position_fit": 0.8,
                "llm_too_broad_penalty": 0.1,
                "llm_too_narrow_penalty": 0.0,
            }
        ],
    }


def test_compare_variant_fairness_rejects_non_identical_evidence():
    report = compare_variant_fairness(
        {
            "llm_evidence_plus_rule": [
                _evidence_paper_result("A", "J1", scope_fit=0.8)
            ],
            "llm_evidence_plus_learned_reranker": [
                _evidence_paper_result("A", "J1", scope_fit=0.7)
            ],
        }
    )

    assert report["fairness_pass"] is False
    assert report["evidence_mismatches"] == ["A | J1"]


def test_compare_variant_fairness_rejects_incomplete_evidence_coverage():
    report = compare_variant_fairness(
        {
            "llm_evidence_plus_rule": [
                _evidence_paper_result("A", "J1", scope_fit=0.8, coverage=0.5)
            ],
            "llm_evidence_plus_learned_reranker": [
                _evidence_paper_result("A", "J1", scope_fit=0.8, coverage=1.0)
            ],
        }
    )

    assert report["fairness_pass"] is False
    assert report["evidence_coverage_failures"] == [
        "llm_evidence_plus_rule: A | J1 (0.500)"
    ]


def test_compare_variant_fairness_can_report_partial_coverage_without_failing_debug_run():
    report = compare_variant_fairness(
        {
            "llm_evidence_plus_rule": [
                _evidence_paper_result("A", "J1", scope_fit=0.8, coverage=0.5)
            ],
            "llm_evidence_plus_learned_reranker": [
                _evidence_paper_result("A", "J1", scope_fit=0.8, coverage=0.5)
            ],
        },
        require_full_evidence=False,
    )

    assert report["fairness_pass"] is True
    assert len(report["evidence_coverage_failures"]) == 2


# ---------------------------------------------------------------------------
# Fix #1 — evidence snapshot sharing
# ---------------------------------------------------------------------------


def test_evidence_role_ranker_uses_precomputed_evidence_without_calling_extractor():
    """Two evidence variants reading the same snapshot must produce byte-identical
    evidence scores (the entire point of the snapshot pre-pass)."""
    journals = [_journal("j1"), _journal("j2")]
    extractor = FixedEvidenceExtractor({"j1": _evidence(0.7), "j2": _evidence(0.3)})
    ranker_rule = LLMEvidenceRoleRanker(
        evidence_extractor=extractor,
        journal_store=StubJournalStore(journals),
        prior_source="rule",
    )
    ranker_learned = LLMEvidenceRoleRanker(
        evidence_extractor=extractor,
        journal_store=StubJournalStore(journals),
        prior_source="learned",
    )
    snapshot = {"j1": _evidence(0.7), "j2": _evidence(0.3)}
    profile = PaperProfile(title="T")
    candidates = [(journals[0], 1.0, []), (journals[1], 0.8, [])]
    trace = _trace(journals)

    _ranked_a, _m_a, diag_a = ranker_rule.rank_with_diagnostics(
        candidates=candidates,
        paper_profile=profile,
        retrieval_trace=trace,
        rule_ranks={"j1": 1, "j2": 2},
        rule_scores={"j1": 1.0, "j2": 0.8},
        precomputed_evidence=snapshot,
    )
    _ranked_b, _m_b, diag_b = ranker_learned.rank_with_diagnostics(
        candidates=candidates,
        paper_profile=profile,
        retrieval_trace=trace,
        rule_ranks={"j1": 1, "j2": 2},
        rule_scores={"j1": 1.0, "j2": 0.8},
        learned_ranks={"j1": 2, "j2": 1},
        precomputed_evidence=snapshot,
    )

    assert extractor.calls == 0, "precomputed snapshot must bypass the extractor"
    # Evidence scores bit-equal across variants (only the prior differs).
    for jid in ("j1", "j2"):
        for field in ("scope_fit", "method_fit", "application_fit", "journal_position_fit",
                      "too_broad_penalty", "too_narrow_penalty"):
            assert diag_a["candidates"][jid][f"llm_{field}"] == \
                   diag_b["candidates"][jid][f"llm_{field}"], \
                f"evidence diverged for {jid}.{field}"
        assert diag_a["candidates"][jid]["evidence_composite"] == \
               diag_b["candidates"][jid]["evidence_composite"]
    assert diag_a["status"] == "precomputed"
    assert diag_b["status"] == "precomputed"


# ---------------------------------------------------------------------------
# Fix #2 — real rule_rank for prior
# ---------------------------------------------------------------------------


def test_evidence_role_ranker_prior_uses_rule_ranks_not_input_position():
    """Fix #2: the prior must come from rule_ranks, not from where the candidate
    sat in the input list. Reverse the input list to expose the bug."""
    journals = [_journal("j1"), _journal("j2")]
    ranker = LLMEvidenceRoleRanker(
        evidence_extractor=FixedEvidenceExtractor(
            {"j1": _evidence(0.5), "j2": _evidence(0.5)}
        ),
        journal_store=StubJournalStore(journals),
        prior_source="rule",
    )
    # j1 has rule_rank=2, j2 has rule_rank=1. Input list reversed (j2 first).
    # Prior for j1 should be 0.0 (rule_rank 2 of 2), not 1.0 (input position 1).
    _ranked, _m, diagnostics = ranker.rank_with_diagnostics(
        candidates=[(journals[1], 0.8, []), (journals[0], 1.0, [])],  # reversed!
        paper_profile=PaperProfile(title="T"),
        retrieval_trace=_trace(journals),
        rule_ranks={"j1": 2, "j2": 1},
        rule_scores={"j1": 1.0, "j2": 0.8},
    )

    assert diagnostics["candidates"]["j1"]["prior_rank"] == 2  # rule rank, not input pos 1
    assert diagnostics["candidates"]["j2"]["prior_rank"] == 1  # rule rank, not input pos 2
    assert diagnostics["candidates"]["j1"]["rank_prior"] == pytest.approx(0.0)
    assert diagnostics["candidates"]["j2"]["rank_prior"] == pytest.approx(1.0)


def test_evidence_role_ranker_normalizes_rule_prior_against_full_rule_pool():
    journals = [_journal("j1"), _journal("j4")]
    ranker = LLMEvidenceRoleRanker(
        evidence_extractor=FixedEvidenceExtractor(
            {"j1": _evidence(0.5), "j4": _evidence(0.5)}
        ),
        journal_store=StubJournalStore(journals),
        prior_source="rule",
    )

    _ranked, _m, diagnostics = ranker.rank_with_diagnostics(
        candidates=[(journals[0], 1.0, []), (journals[1], 0.8, [])],
        paper_profile=PaperProfile(title="T"),
        retrieval_trace=_trace(journals),
        rule_ranks={"j1": 1, "j2": 2, "j3": 3, "j4": 4},
        rule_scores={"j1": 1.0, "j4": 0.8},
    )

    assert diagnostics["candidates"]["j1"]["rank_prior"] == 1.0
    assert diagnostics["candidates"]["j4"]["rank_prior"] == 0.0


def test_evidence_role_ranker_learned_prior_requires_learned_ranks():
    """prior_source='learned' must refuse to fall back to input position."""
    journals = [_journal("j1")]
    ranker = LLMEvidenceRoleRanker(
        evidence_extractor=FixedEvidenceExtractor({"j1": _evidence(0.5)}),
        journal_store=StubJournalStore(journals),
        prior_source="learned",
    )
    with pytest.raises(ValueError, match="requires learned_ranks"):
        ranker.rank_with_diagnostics(
            candidates=[(journals[0], 1.0, [])],
            paper_profile=PaperProfile(title="T"),
            retrieval_trace=_trace(journals),
            rule_ranks={"j1": 1},
            rule_scores={"j1": 1.0},
            # learned_ranks omitted on purpose
        )


# ---------------------------------------------------------------------------
# Fix #3 — evidence_coverage metric
# ---------------------------------------------------------------------------


def test_evidence_role_ranker_reports_evidence_coverage():
    journals = [_journal("j1"), _journal("j2"), _journal("j3")]
    ranker = LLMEvidenceRoleRanker(
        evidence_extractor=FailingEvidenceExtractor(),
        journal_store=StubJournalStore(journals),
        prior_source="rule",
    )
    _ranked, _m, diagnostics = ranker.rank_with_diagnostics(
        candidates=[(j, 1.0, []) for j in journals],
        paper_profile=PaperProfile(title="T"),
        retrieval_trace=_trace(journals),
        rule_ranks={j.journal_id: i + 1 for i, j in enumerate(journals)},
    )
    assert diagnostics["status"] == "neutral_fallback"
    assert diagnostics["evidence_coverage"] == 0.0  # all 3 fell back to neutral


def test_evidence_role_ranker_evidence_coverage_partial():
    journals = [_journal("j1"), _journal("j2")]
    ranker = LLMEvidenceRoleRanker(
        evidence_extractor=FixedEvidenceExtractor(
            {"j1": _evidence(0.5)}  # only j1 has evidence
        ),
        journal_store=StubJournalStore(journals),
        prior_source="rule",
    )
    _ranked, _m, diagnostics = ranker.rank_with_diagnostics(
        candidates=[(j, 1.0, []) for j in journals],
        paper_profile=PaperProfile(title="T"),
        retrieval_trace=_trace(journals),
        rule_ranks={j.journal_id: i + 1 for i, j in enumerate(journals)},
    )
    assert diagnostics["evidence_coverage"] == pytest.approx(0.5)  # 1 of 2 covered


def test_evidence_role_ranker_coverage_ignores_snapshot_candidates_outside_current_pool():
    journals = [_journal("j1"), _journal("j2")]
    ranker = LLMEvidenceRoleRanker(
        evidence_extractor=FailingEvidenceExtractor(),
        journal_store=StubJournalStore(journals),
        prior_source="rule",
    )
    snapshot = {
        "j1": _evidence(0.5),
        "j2": _evidence(0.5),
        "not-in-current-pool": _evidence(0.5),
    }

    _ranked, _m, diagnostics = ranker.rank_with_diagnostics(
        candidates=[(j, 1.0, []) for j in journals],
        paper_profile=PaperProfile(title="T"),
        retrieval_trace=_trace(journals),
        rule_ranks={"j1": 1, "j2": 2},
        precomputed_evidence=snapshot,
    )

    assert diagnostics["evidence_coverage"] == 1.0


def test_load_evidence_snapshot_normalizes_precompute_title_venue_keys(tmp_path):
    snapshot_path = tmp_path / "evidence.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "partial_coverage_count": 0,
                "papers": {
                    "paper title | venue": {
                        "title": "  Paper   Title ",
                        "venue": "Venue",
                        "evidence_coverage": 1.0,
                        "evidence": {"j1": _evidence(0.7)},
                        "learned_ranks": {"j1": 1},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_evidence_snapshot(str(snapshot_path), allow_partial=False)

    assert set(loaded) == {"paper title"}
    assert loaded["paper title"]["evidence"]["j1"]["scope_fit"] == 0.7


def test_load_evidence_snapshot_rejects_partial_coverage_by_default(tmp_path):
    snapshot_path = tmp_path / "partial.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "papers": {
                    "paper title": {
                        "title": "Paper Title",
                        "evidence_coverage": 0.5,
                        "evidence": {"j1": _evidence(0.7)},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="partial coverage"):
        load_evidence_snapshot(str(snapshot_path), allow_partial=False)


def test_select_snapshot_candidates_uses_actual_llm_candidate_pool():
    journals = [_journal("j1"), _journal("j2"), _journal("j3")]
    rec_result = {
        "rule_ranked": [(journals[0], 1.0, []), (journals[1], 0.9, []), (journals[2], 0.8, [])],
        "llm_candidates": [(journals[1], 0.9, []), (journals[0], 1.0, [])],
    }

    candidates = select_snapshot_candidates(rec_result)

    assert [journal.journal_id for journal, _score, _reasons in candidates] == [
        "j2",
        "j1",
    ]


# ---------------------------------------------------------------------------
# Fix #2 (extractor) — prompt uses real rule_ranks
# ---------------------------------------------------------------------------


def test_evidence_extractor_prompt_uses_supplied_rule_ranks():
    """When the caller passes rule_ranks, the prompt must report those ranks,
    not the input-list position. This prevents telling the LLM a mislabeled rank."""
    from src.ranker.llm_evidence_extractor import LLMEvidenceExtractor

    class CapturingLLM:
        def __init__(self):
            self.last_user_prompt = ""

        def chat_auto(self, system_prompt, user_prompt, timeout=None):
            self.last_user_prompt = user_prompt
            from types import SimpleNamespace

            return SimpleNamespace(
                content=json.dumps(
                    {
                        "evidence": [
                            {
                                "journal_id": "j2",
                                "scope_fit": 0.7,
                                "method_fit": 0.7,
                                "application_fit": 0.7,
                                "journal_position_fit": 0.7,
                                "too_broad_penalty": 0.0,
                                "too_narrow_penalty": 0.0,
                                "evidence": ["fits well"],
                            }
                        ]
                    }
                ),
                usage={},
            )

    llm = CapturingLLM()
    extractor = LLMEvidenceExtractor(
        llm=llm,
        system_prompt="sys",
        user_prompt_template="JOURNALS: {journals_info}",
    )
    journals = [_journal("j1"), _journal("j2")]
    # Input list is j1, j2 — but the *real* rule ranks are j1=5, j2=1.
    # The prompt must say j1.rule_rank=5 and j2.rule_rank=1.
    extractor.extract(
        candidates=[(journals[0], 1.0, []), (journals[1], 0.8, [])],
        paper_profile=PaperProfile(title="T"),
        rule_ranks={"j1": 5, "j2": 1},
    )
    # Re-parse the rendered journals_info to verify ranks.
    match = re.search(r"JOURNALS:\s*(\[.*\])", llm.last_user_prompt, re.DOTALL)
    assert match, "journals_info block not found in prompt"
    info = json.loads(match.group(1))
    ranks_by_id = {item["journal_id"]: item["rule_rank"] for item in info}
    assert ranks_by_id == {"j1": 5, "j2": 1}, \
        f"prompt rule_ranks should be {ranks_by_id}"
