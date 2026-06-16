"""Rank journal candidates using structured LLM evidence plus a weak rank prior."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.journals.accepted_paper_store import AcceptedPaperStore
from src.journals.journal_model import Journal
from src.journals.journal_store import JournalStore
from src.papers.paper_model import PaperProfile
from src.ranker.feature_builder import (
    FEATURE_NAMES,
    FEATURE_NAMES_WITH_LLM_EVIDENCE,
    build_features,
)
from src.ranker.llm_evidence_extractor import LLMEvidenceExtractor

logger = logging.getLogger(__name__)

EvidenceRankedCandidate = Tuple[Journal, float, List[str], float]
InputCandidate = Tuple[Journal, float, List[str]]


def load_evidence_snapshot(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load a precompute_evidence.py snapshot JSON into the lookup dict
    expected by :class:`LLMEvidenceRoleRanker`.

    Returns ``{normalized_title: paper_entry}`` where each ``paper_entry`` is
    the original dict from the snapshot (containing ``evidence``,
    ``rule_ranks``, ``learned_ranks``, etc.). Empty papers or papers with
    empty ``evidence`` dicts are skipped so the role ranker's
    ``_lookup_snapshot_entry`` returns ``None`` and the caller falls back
    to neutral defaults (or to the live extractor when no snapshot is
    available).
    """
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for paper_key, entry in (payload.get("papers") or {}).items():
        # Use the paper's own title field as the key (not the outer
        # ``paper_key`` which is ``title | venue``). The role ranker
        # and pipeline.py both look up by title only.
        title = entry.get("title") or paper_key
        key = " ".join(str(title or "").casefold().split())
        if not key:
            continue
        if key in out:
            raise ValueError(
                f"Duplicate normalized title in evidence snapshot: {key}"
            )
        evidence = entry.get("evidence") or {}
        if evidence:
            out[key] = entry
    return out


class DirectLLMRoleRanker:
    """Wrap the existing direct LLM ranker and return per-call diagnostics."""

    def __init__(self, direct_ranker: Any) -> None:
        self.direct_ranker = direct_ranker

    def rank(
        self,
        candidates: List[InputCandidate],
        paper_profile: PaperProfile,
        top_k: int = 5,
        retrieval_trace: Optional[Dict[str, dict]] = None,
        **kwargs: Any,
    ) -> Tuple[List[EvidenceRankedCandidate], str]:
        """Backward-compat shim for callers that use the legacy `.rank()`
        interface. Returns the ranked list and method name; discards
        diagnostics."""
        ranked, method, _diag = self.rank_with_diagnostics(
            candidates,
            paper_profile,
            top_k=top_k,
            retrieval_trace=retrieval_trace,
        )
        return ranked, method

    def rank_with_diagnostics(
        self,
        candidates: List[InputCandidate],
        paper_profile: PaperProfile,
        top_k: int = 5,
        retrieval_trace: Optional[Dict[str, dict]] = None,
        rule_ranks: Optional[Dict[str, int]] = None,
        rule_scores: Optional[Dict[str, float]] = None,
        learned_ranks: Optional[Dict[str, int]] = None,
        # 6.4: accepted for signature parity with LLMEvidenceRoleRanker so the
        # pipeline can pass learned_scores unconditionally. The direct role
        # does not use learned_scores in its formula.
        learned_scores: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[EvidenceRankedCandidate], str, Dict[str, Any]]:
        ranked, method = self.direct_ranker.rank(
            candidates,
            paper_profile,
            top_k=top_k,
            retrieval_trace=retrieval_trace,
        )
        ranked_by_id = {
            journal.journal_id: {
                "final_rank": index + 1,
                "final_score": float(score),
                "llm_score": float(score),
                "llm_reasons": list(reasons or []),
                "llm_confidence": float(confidence),
            }
            for index, (journal, score, reasons, confidence) in enumerate(ranked)
        }
        details = {}
        for input_index, (journal, rule_score, reasons) in enumerate(candidates):
            details[journal.journal_id] = {
                "journal_id": journal.journal_id,
                "journal_name": journal.journal_name,
                "input_rank": input_index + 1,
                "prior_source": None,
                "rule_rank": (rule_ranks or {}).get(journal.journal_id),
                "rule_score": float((rule_scores or {}).get(journal.journal_id, rule_score)),
                "candidate_reasons": list(reasons or []),
                **ranked_by_id.get(journal.journal_id, {}),
            }
        return ranked, method, {
            "status": "ok",
            "role": "direct",
            "prior_source": None,
            "fallback_reason": "",
            "candidates": details,
        }


class LLMEvidenceRoleRanker:
    """Use LLM as an evidence extractor instead of a direct ranking judge.

    The object is safe to share across evaluation worker threads because
    per-paper diagnostics are returned from ``rank_with_diagnostics`` and are
    never stored on the instance.
    """

    FIT_FIELDS = (
        "scope_fit",
        "method_fit",
        "application_fit",
        "journal_position_fit",
    )
    PENALTY_FIELDS = ("too_broad_penalty", "too_narrow_penalty")

    def __init__(
        self,
        evidence_extractor: LLMEvidenceExtractor,
        journal_store: JournalStore,
        accepted_paper_store: Optional[AcceptedPaperStore] = None,
        prior_source: str = "rule",
        evidence_weight: float = 0.8,
        prior_weight: float = 0.2,
        ltr_score_weight: float = 0.0,
        evidence_snapshot: Optional[Dict[str, Dict[str, Any]]] = None,
        evidence_field_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        if prior_source not in {"rule", "learned"}:
            raise ValueError(f"Unsupported prior_source: {prior_source}")
        if (
            evidence_weight < 0
            or prior_weight < 0
            or ltr_score_weight < 0
        ):
            raise ValueError(
                "evidence_weight, prior_weight, ltr_score_weight must be non-negative"
            )
        total_weight = evidence_weight + prior_weight + ltr_score_weight
        if total_weight <= 0:
            raise ValueError("At least one ranking weight must be positive")

        self.evidence_extractor = evidence_extractor
        self.journal_store = journal_store
        self.accepted_paper_store = accepted_paper_store
        self.prior_source = prior_source
        # Renormalize to sum=1.0 so the final_score formula is
        # evidence*W_e + rank_prior*W_p + ltr_score*W_l and never inflates
        # beyond 1.0 just because a caller passed unnormalized weights.
        self.evidence_weight = float(evidence_weight / total_weight)
        self.prior_weight = float(prior_weight / total_weight)
        self.ltr_score_weight = float(ltr_score_weight / total_weight)
        # Fix #1: per-paper evidence snapshot, keyed by normalized title.
        # When a paper's evidence is requested, the ranker looks it up here
        # and skips the LLM extractor call entirely.
        self.evidence_snapshot = evidence_snapshot
        # P0-PRIME 2026-06-16: optional per-fit-field weights for the
        # evidence_composite formula. None or invalid (sum != 1.0)
        # → fall back to equal-weight mean in _evidence_composite.
        self.evidence_field_weights = evidence_field_weights

    @staticmethod
    def _title_key(title: str) -> str:
        return " ".join(str(title or "").casefold().split())

    def _lookup_snapshot_entry(
        self, paper_profile: PaperProfile
    ) -> Optional[Dict[str, Any]]:
        if not self.evidence_snapshot:
            return None
        return self.evidence_snapshot.get(self._title_key(paper_profile.title))

    def rank(
        self,
        candidates: List[InputCandidate],
        paper_profile: PaperProfile,
        top_k: int = 5,
        retrieval_trace: Optional[Dict[str, dict]] = None,
        **kwargs: Any,
    ) -> Tuple[List[EvidenceRankedCandidate], str]:
        """Backward-compat shim for callers that use the legacy `.rank()`
        interface. Returns (ranked, method_name); discards diagnostics.

        For this role ranker, kwargs may include `precomputed_evidence`,
        `rule_ranks`, `rule_scores`, `learned_ranks`, `learned_scores`
        which are threaded through to `rank_with_diagnostics` when
        present in its signature.
        """
        import inspect as _inspect
        _rwd_params = _inspect.signature(self.rank_with_diagnostics).parameters
        _rwd_kwargs = {
            "candidates": candidates,
            "paper_profile": paper_profile,
            "top_k": top_k,
            "retrieval_trace": retrieval_trace,
        }
        for k in (
            "precomputed_evidence",
            "rule_ranks",
            "rule_scores",
            "learned_ranks",
            "learned_scores",
        ):
            if k in kwargs and k in _rwd_params:
                _rwd_kwargs[k] = kwargs[k]
        ranked, method, _diag = self.rank_with_diagnostics(**_rwd_kwargs)
        return ranked, method

    def rank_with_diagnostics(
        self,
        candidates: List[InputCandidate],
        paper_profile: PaperProfile,
        top_k: int = 5,
        retrieval_trace: Optional[Dict[str, dict]] = None,
        rule_ranks: Optional[Dict[str, int]] = None,
        rule_scores: Optional[Dict[str, float]] = None,
        precomputed_evidence: Optional[Dict[str, Dict[str, Any]]] = None,
        learned_ranks: Optional[Dict[str, int]] = None,
        learned_scores: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[EvidenceRankedCandidate], str, Dict[str, Any]]:
        """Rank candidates and return diagnostics local to this invocation.

        ``precomputed_evidence`` (Fix #1, evidence snapshot sharing): when supplied,
        no LLM call is made; the provided per-journal-id evidence is used directly.
        This guarantees two evidence variants see byte-identical evidence.

        ``learned_ranks`` (Fix #2, real rule_rank): when ``prior_source="learned"``,
        the prior is derived from this map rather than input-list position. The
        caller (typically the snapshot pre-pass) is responsible for computing
        learned ranks via the actual LTR model. For ``prior_source="rule"``, the
        prior is taken from ``rule_ranks``. Missing prior ranks are rejected
        instead of silently falling back to input-list position.

        ``learned_scores`` (Task 6.4, LTR score as formula component): when
        supplied AND the ranker was constructed with ``ltr_score_weight > 0``,
        the per-candidate LTR score is added as a third weighted component in
        the final formula. When omitted or empty, the LTR contribution is 0
        and the formula collapses to the legacy 2-component shape.
        """
        if not candidates:
            return [], self._rank_method(), {
                "status": "ok",
                "role": "evidence",
                "prior_source": self.prior_source,
                "evidence_weight": self.evidence_weight,
                "prior_weight": self.prior_weight,
                "ltr_score_weight": self.ltr_score_weight,
                "fallback_reason": "",
                "evidence_coverage": 1.0,
                "candidates": {},
            }

        retrieval_trace = retrieval_trace or {}
        rule_ranks = rule_ranks or {}
        rule_scores = rule_scores or {}
        learned_ranks = learned_ranks or {}
        learned_scores = learned_scores or {}

        # Fix #1: snapshot lookup. The pre-pass populated self.evidence_snapshot
        # with a per-paper dict of {journal_id: evidence_item}. Look up by title
        # and use the entry's evidence + learned_ranks. This bypasses the LLM
        # call AND guarantees byte-identical evidence across variants.
        snapshot_entry = self._lookup_snapshot_entry(paper_profile)

        fallback_reason = ""
        if precomputed_evidence is not None:
            evidence_by_id = dict(precomputed_evidence)
            status = "precomputed"
        elif snapshot_entry is not None:
            evidence_by_id = dict(snapshot_entry.get("evidence") or {})
            if not evidence_by_id and snapshot_entry.get("status") == "neutral_fallback":
                status = "neutral_fallback"
                fallback_reason = snapshot_entry.get("fallback_reason", "")
            else:
                status = "precomputed"
            # Lift learned_ranks from the snapshot when caller didn't supply them.
            if not learned_ranks and snapshot_entry.get("learned_ranks"):
                learned_ranks = dict(snapshot_entry["learned_ranks"])
        else:
            try:
                evidence_by_id = self.evidence_extractor.extract(
                    candidates, paper_profile, rule_ranks=rule_ranks
                )
                status = "ok"
            except Exception as exc:
                logger.warning(
                    "LLM evidence extraction failed; using neutral evidence: %s", exc
                )
                evidence_by_id = {}
                status = "neutral_fallback"
                fallback_reason = f"{type(exc).__name__}: {exc}"[:500]

        # Fix #3: evidence_coverage = fraction of candidates that received non-empty evidence.
        candidate_ids = {journal.journal_id for journal, _score, _reasons in candidates}
        covered_ids = {
            jid
            for jid, ev in evidence_by_id.items()
            if jid in candidate_ids
            and ev
            and all(field in ev for field in self.FIT_FIELDS + self.PENALTY_FIELDS)
        }
        evidence_coverage = len(covered_ids) / len(candidates) if candidates else 1.0

        candidate_count = len(candidates)
        prior_population_size = self._prior_population_size(
            rule_ranks=rule_ranks,
            learned_ranks=learned_ranks,
            fallback=candidate_count,
        )
        details: Dict[str, Dict[str, Any]] = {}
        ranked_rows: List[EvidenceRankedCandidate] = []

        for input_index, (journal, rule_score, original_reasons) in enumerate(candidates):
            # Fix #2: prior rank comes from the appropriate source, NOT input position.
            prior_rank = self._resolve_prior_rank(
                journal.journal_id,
                rule_ranks=rule_ranks,
                learned_ranks=learned_ranks,
            )
            rank_prior = self._linear_rank_prior(prior_rank, prior_population_size)
            evidence = evidence_by_id.get(journal.journal_id) or {}
            normalized = self._normalized_evidence(evidence)
            evidence_composite = self._evidence_composite(
                normalized, weights=self.evidence_field_weights
            )
            # Task 6.4: blend in the actual LTR score as a third component when
            # the ranker was constructed with ltr_score_weight > 0. When the
            # caller didn't supply learned_scores (e.g. legacy code paths) or
            # this candidate is missing from the map, the LTR contribution
            # collapses to 0 and the formula degrades gracefully to the
            # 2-component (evidence + rank_prior) shape.
            ltr_score = 0.0
            if learned_scores:
                ltr_score = float(learned_scores.get(journal.journal_id, 0.0))
            final_score = (
                evidence_composite * self.evidence_weight
                + rank_prior * self.prior_weight
                + ltr_score * self.ltr_score_weight
            )
            evidence_text = evidence.get("evidence")
            if not isinstance(evidence_text, list):
                evidence_text = []
            evidence_text_clean = [
                text for text in evidence_text if isinstance(text, str) and text.strip()
            ]
            reasons = list(evidence_text_clean)
            if not reasons:
                reasons = list(original_reasons or [])
            reasons.append(
                f"结构化证据分={evidence_composite:.3f};"
                f"{self.prior_source}先验={rank_prior:.3f}"
            )

            features = build_features(
                paper_profile=paper_profile,
                journal=journal,
                trace_entry=retrieval_trace.get(journal.journal_id) or {},
                rule_rank=rule_ranks.get(journal.journal_id),
                rule_score=rule_scores.get(journal.journal_id, rule_score),
                candidate_in_accepted_corpus=self._candidate_in_accepted_corpus(
                    journal.journal_id
                ),
                llm_evidence=normalized,
            )

            details[journal.journal_id] = {
                "journal_id": journal.journal_id,
                "journal_name": journal.journal_name,
                "input_rank": input_index + 1,
                "prior_rank": prior_rank,
                "prior_source": self.prior_source,
                "rank_prior": rank_prior,
                "rule_rank": rule_ranks.get(journal.journal_id),
                "rule_score": float(rule_scores.get(journal.journal_id, rule_score)),
                "learned_rank": learned_ranks.get(journal.journal_id),
                "learned_score": learned_scores.get(journal.journal_id),
                "ltr_score": ltr_score,
                "ltr_score_weight": self.ltr_score_weight,
                "has_evidence": journal.journal_id in covered_ids,
                **{
                    f"llm_{field}": normalized[field]
                    for field in self.FIT_FIELDS + self.PENALTY_FIELDS
                },
                "evidence": evidence_text_clean,
                "evidence_composite": evidence_composite,
                "final_score": final_score,
                "feature_names_base": list(FEATURE_NAMES),
                "features_base": features.to_vector(FEATURE_NAMES),
                "feature_names_with_llm_evidence": list(
                    FEATURE_NAMES_WITH_LLM_EVIDENCE
                ),
                "features_with_llm_evidence": features.to_vector(
                    FEATURE_NAMES_WITH_LLM_EVIDENCE
                ),
            }
            ranked_rows.append((journal, final_score, reasons, evidence_composite))

        # Python sort is stable, so equal scores retain the incoming Rule/LTR order.
        ranked_rows.sort(key=lambda item: item[1], reverse=True)
        for final_rank, (journal, _score, _reasons, _confidence) in enumerate(
            ranked_rows, start=1
        ):
            details[journal.journal_id]["final_rank"] = final_rank

        return ranked_rows[:top_k], self._rank_method(), {
            "status": status,
            "role": "evidence",
            "prior_source": self.prior_source,
            "evidence_weight": self.evidence_weight,
            "prior_weight": self.prior_weight,
            "ltr_score_weight": self.ltr_score_weight,
            "evidence_coverage": evidence_coverage,
            "fallback_reason": fallback_reason,
            "candidates": details,
        }

    def _rank_method(self) -> str:
        return f"llm_evidence_{self.prior_source}"

    def _prior_population_size(
        self,
        rule_ranks: Dict[str, int],
        learned_ranks: Dict[str, int],
        fallback: int,
    ) -> int:
        ranks = rule_ranks if self.prior_source == "rule" else learned_ranks
        valid_ranks = [
            int(rank)
            for rank in ranks.values()
            if isinstance(rank, int) and not isinstance(rank, bool) and rank >= 1
        ]
        return max(valid_ranks, default=max(fallback, 1))

    def _resolve_prior_rank(
        self,
        journal_id: str,
        rule_ranks: Dict[str, int],
        learned_ranks: Dict[str, int],
    ) -> int:
        """Return the rank (1-based) used to derive ``rank_prior``.

        ``prior_source="rule"``:    require ``rule_ranks``.
        ``prior_source="learned"``: require ``learned_ranks``; raise if missing,
                                    because using input position would silently
                                    re-introduce the very bug Fix #2 removes.
        """
        if self.prior_source == "rule":
            rank = rule_ranks.get(journal_id)
            if rank is None:
                raise ValueError(
                    f"prior_source='rule' requires rule_ranks; missing entry for "
                    f"{journal_id}. Refusing to fall back to input position."
                )
            return rank
        if self.prior_source == "learned":
            rank = learned_ranks.get(journal_id)
            if rank is None:
                raise ValueError(
                    f"prior_source='learned' requires learned_ranks; "
                    f"missing entry for {journal_id}. Refusing to fall back to "
                    "input position because that defeats Fix #2."
                )
            return rank
        raise ValueError(f"Unsupported prior_source: {self.prior_source}")

    @staticmethod
    def _linear_rank_prior(rank: int, candidate_count: int) -> float:
        if candidate_count <= 1:
            return 1.0
        return max(0.0, min(1.0, 1.0 - ((rank - 1) / (candidate_count - 1))))

    @classmethod
    def _normalized_evidence(cls, evidence: Dict[str, Any]) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for field in cls.FIT_FIELDS:
            result[field] = cls._score_or_default(evidence.get(field), 0.5)
        for field in cls.PENALTY_FIELDS:
            result[field] = cls._score_or_default(evidence.get(field), 0.0)
        return result

    @staticmethod
    def _score_or_default(value: Any, default: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        if not 0 <= value <= 1:
            return default
        return float(value)

    @classmethod
    def _evidence_composite(
        cls,
        evidence: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute evidence_composite score.

        Args:
            evidence: dict with 6 evidence fields (4 fit + 2 penalty).
            weights: optional per-field weights for fit_fields. If provided
                AND the 4 fit weights sum to 1.0 (within 1e-6 tolerance),
                weighted_fit = sum(w_f * f). Otherwise fall back to
                equal-weight mean of fit_fields. Penalty is always
                equal-weight mean (too_broad + too_narrow).

        Returns:
            composite score in [0.0, 1.0], clipped.

        P0-PRIME 2026-06-16: weights argument added to allow
        per-field weighting derived from discrimination analysis.
        Fallback behavior preserves backward compatibility with
        callers that don't pass weights.
        """
        penalty_mean = sum(evidence[field] for field in cls.PENALTY_FIELDS) / len(
            cls.PENALTY_FIELDS
        )

        # Decide between weighted and equal-weight fit aggregation.
        use_weighted = (
            weights is not None
            and len(weights) == len(cls.FIT_FIELDS)
            and all(f in weights for f in cls.FIT_FIELDS)
        )
        if use_weighted:
            weight_sum = sum(weights[f] for f in cls.FIT_FIELDS)
            # Tolerance for floating point comparison
            use_weighted = abs(weight_sum - 1.0) < 1e-6

        if use_weighted:
            fit_score = sum(weights[f] * evidence[f] for f in cls.FIT_FIELDS)
        else:
            fit_score = sum(evidence[field] for field in cls.FIT_FIELDS) / len(
                cls.FIT_FIELDS
            )

        return max(0.0, min(1.0, fit_score - penalty_mean))

    def _candidate_in_accepted_corpus(self, journal_id: str) -> bool:
        if self.accepted_paper_store is None:
            return False
        try:
            return bool(self.accepted_paper_store.get_papers(journal_id))
        except Exception:
            return False
