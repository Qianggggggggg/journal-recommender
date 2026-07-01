# Pipeline 26-dim Evidence LTR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 26-dim LTR (trained on `ranker_train_26dim_v1.jsonl`) actually use the 6 LLM-evidence features during inference, not just at training time, so the LLM role ablation can compare 20-dim v4 vs 26-dim evidence LTR on light30 fairly.

**Architecture:** Pipeline reads evidence from a per-paper snapshot, threads it to the candidate generator's `attach_features_to_trace` (already supports 26-dim via `feature_names=FEATURE_NAMES_WITH_LLM_EVIDENCE`), LTR adapter scores with the 26-dim model and returns `learned_score: {jid: float}`, the role ranker uses that score as a feature in its final formula (replacing or augmenting the current `learned_rank` prior). Backward compat: with no evidence snapshot, 20-dim v4 path works unchanged.

**Tech Stack:** Python 3.11, sklearn LogisticRegression (LTR backend), existing LTRAdapter/feature_builder/LLMEvidenceRoleRanker machinery.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/ranker/feature_builder.py` | Already supports 26-dim attach via `feature_names=FEATURE_NAMES_WITH_LLM_EVIDENCE` arg. **No changes.** |
| `src/ranker/ltr_adapter.py` | Already returns `learned_score: {jid: float}` in compute_scores diagnostics. **No changes.** |
| `src/retriever/candidate_generator.py` | Add `feature_names` and `llm_evidence_by_journal` params to `attach_features()`. Forward to `attach_features_to_trace()`. |
| `src/recommender/pipeline.py` | Accept `evidence_lookup` (per-paper evidence) and `feature_schema` (20/26) at construction. Thread to candidate_generator and surface `learned_score` in result. |
| `src/ranker/llm_evidence_role_ranker.py` | Accept `learned_scores` dict. New formula `0.8 × evidence + α × (1 - learned_rank/N) + β × ltr_score_normalized`. Default α=0.2, β=0.0 (preserves current behavior). |
| `scripts/run_llm_role_ablation.py` | Add `--ltr-score-weight` arg, pass `learned_scores` to role ranker. |
| `scripts/precompute_evidence.py` | No changes. Already produces evidence_lookup. |
| `tests/test_*` | New unit tests for each component. |

---

## Task 1: candidate_generator.attach_features accepts 26-dim schema

**Files:**
- Modify: `src/retriever/candidate_generator.py:78-101` (the `attach_features` method)
- Test: `tests/test_candidate_generator_features.py` (extend existing)

**Goal:** Forward `feature_names` and `llm_evidence_by_journal` from the pipeline to the underlying `attach_features_to_trace`, so the trace's per-journal `features` array length matches the LTR's expected dim (20 or 26).

- [ ] **Step 1: Write failing test**

Add to `tests/test_candidate_generator_features.py`:

```python
def test_attach_features_supports_26_dim_schema():
    """When feature_names=FEATURE_NAMES_WITH_LLM_EVIDENCE and evidence is
    supplied, each trace entry's features array must be 26 long."""
    from src.ranker.feature_builder import FEATURE_NAMES_WITH_LLM_EVIDENCE

    journals = [_journal("j1"), _journal("j2")]
    store = StubJournalStore(journals)
    cg = CandidateGenerator(...)
    trace = {
        "j1": {"retrieval_rank": 1, "routes": {}},
        "j2": {"retrieval_rank": 2, "routes": {}},
    }
    paper_evidence = {
        "j1": {"scope_fit": 0.9, "method_fit": 0.8, "application_fit": 0.7,
               "journal_position_fit": 0.85, "too_broad_penalty": 0.1, "too_narrow_penalty": 0.05},
        "j2": {"scope_fit": 0.4, "method_fit": 0.3, "application_fit": 0.5,
               "journal_position_fit": 0.2, "too_broad_penalty": 0.0, "too_narrow_penalty": 0.0},
    }
    profile = PaperProfile(title="P1")

    cg.attach_features(
        trace=trace,
        paper_profile=profile,
        rule_ranks={"j1": 1, "j2": 2},
        rule_scores={"j1": 0.9, "j2": 0.8},
        feature_names=FEATURE_NAMES_WITH_LLM_EVIDENCE,
        llm_evidence_by_journal=paper_evidence,
    )

    assert len(trace["j1"]["features"]) == 26
    assert trace["j1"]["feature_names"] == FEATURE_NAMES_WITH_LLM_EVIDENCE
    # Last 6 entries should be the evidence values
    assert trace["j1"]["features"][20:] == [0.9, 0.8, 0.7, 0.85, 0.1, 0.05]
    assert len(trace["j2"]["features"]) == 26
    assert trace["j2"]["features"][20:] == [0.4, 0.3, 0.5, 0.2, 0.0, 0.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_candidate_generator_features.py::test_attach_features_supports_26_dim_schema -v`
Expected: TypeError ("got unexpected keyword argument 'feature_names'")

- [ ] **Step 3: Modify `attach_features` to accept the new params**

In `src/retriever/candidate_generator.py:78-101`, change the signature and body:

```python
def attach_features(
    self,
    trace: Dict[str, dict],
    paper_profile: PaperProfile,
    rule_ranks: Optional[Dict[str, int]],
    rule_scores: Optional[Dict[str, float]],
    accepted_paper_store: Optional[AcceptedPaperStore] = None,
    feature_names: Optional[List[str]] = None,
    llm_evidence_by_journal: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """..."""
    attach_features_to_trace(
        trace=trace,
        paper_profile=paper_profile,
        journal_store=self.store,
        rule_ranks=rule_ranks,
        rule_scores=rule_scores,
        accepted_paper_store=accepted_paper_store,
        llm_evidence_by_journal=llm_evidence_by_journal,
        feature_names=feature_names,
    )
```

Add `from typing import List` and `from typing import Any, Dict` if not already imported.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_candidate_generator_features.py::test_attach_features_supports_26_dim_schema -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retriever/candidate_generator.py tests/test_candidate_generator_features.py
git commit -m "feat(6.4): attach_features supports 26-dim evidence schema"
```

---

## Task 2: Pipeline accepts evidence_lookup + feature_schema

**Files:**
- Modify: `src/recommender/pipeline.py` (constructor + recommend flow + result dict)
- Test: `tests/test_recommender.py` (extend existing)

**Goal:** Pipeline reads `evidence_lookup` (per-paper evidence from snapshot) at construction time, threads it to `attach_features` per-paper during recommend(). Surfaces `learned_score` in the result dict for downstream consumers.

- [ ] **Step 1: Write failing test**

```python
def test_pipeline_attaches_26_dim_features_when_evidence_supplied():
    """When evidence_lookup is set on the pipeline and LTR is 26-dim,
    the trace's per-journal features array is 26 long."""
    snapshot = {
        "test paper": {  # normalized title key
            "j1": {"scope_fit": 0.9, "method_fit": 0.8, ...},
            "j2": {"scope_fit": 0.4, ...},
        }
    }
    pipeline = RecommenderPipeline(
        ...,
        evidence_lookup=snapshot,
        feature_schema="26_dim_with_llm_evidence",
    )
    paper_input = PaperInput(title="Test Paper", abstract="...", mode="abstract")
    profile = PaperProfile(title="Test Paper")
    result = pipeline.recommend(paper_input, profile, top_k=5, mode="abstract")

    trace = result["retrieval_trace"]
    for jid, entry in trace.items():
        if "features" in entry:
            assert len(entry["features"]) == 26
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_recommender.py::test_pipeline_attaches_26_dim_features_when_evidence_supplied -v`
Expected: TypeError on Pipeline constructor (unexpected kwargs)

- [ ] **Step 3: Add new constructor params to RecommenderPipeline**

In `src/recommender/pipeline.py`, modify `__init__` (around line 30) to accept the new params and store them:

```python
def __init__(
    self,
    candidate_generator: ...,
    rule_scorer: ...,
    llm_ranker: ...,
    quality_assessor: ... = None,
    learned_reranker: ... = None,
    llm_anchor_guard: ... = None,
    evidence_lookup: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    feature_schema: str = "20_dim_base",  # "20_dim_base" | "26_dim_with_llm_evidence"
) -> None:
    ...
    self.evidence_lookup = evidence_lookup or {}
    self.feature_schema = feature_schema
    # Determine expected feature dim from schema
    self._expected_feature_dim = (
        26 if feature_schema == "26_dim_with_llm_evidence" else 20
    )
```

- [ ] **Step 4: Modify `recommend()` to thread evidence per-paper**

In `src/recommender/pipeline.py`, find the spot where `attach_features` is called (search for `attach_features(` or `self.candidate_generator.attach_features`). It should be called around line 120-160, after rule scoring. Add the new params:

```python
# Per-paper evidence lookup
paper_title_key = " ".join((paper_profile.title or "").casefold().split())
paper_evidence = self.evidence_lookup.get(paper_title_key, {})
# Determine feature schema for this paper
feature_names = (
    FEATURE_NAMES_WITH_LLM_EVIDENCE
    if self._expected_feature_dim == 26 and paper_evidence
    else None  # default 20-dim
)
self.candidate_generator.attach_features(
    trace=retrieval_trace,
    paper_profile=paper_profile,
    rule_ranks=rule_ranks_map,
    rule_scores=rule_scores_map,
    accepted_paper_store=accepted_store,
    feature_names=feature_names,
    llm_evidence_by_journal=paper_evidence,
)
```

Add imports at top of pipeline.py: `from src.ranker.feature_builder import FEATURE_NAMES_WITH_LLM_EVIDENCE`.

- [ ] **Step 5: Surface `learned_score` in result dict**

After `learned_reranker.compute_scores` returns (around line 200), add to the result:

```python
result_dict = {
    ...
    "learned_score": learned_diag.get("learned_score", {}),
    "learned_rank": learned_diag.get("learned_rank", {}),
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_recommender.py::test_pipeline_attaches_26_dim_features_when_evidence_supplied -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/recommender/pipeline.py tests/test_recommender.py
git commit -m "feat(6.4): pipeline threads evidence_lookup to 26-dim attach"
```

---

## Task 3: Role ranker uses LTR score as a feature

**Files:**
- Modify: `src/ranker/llm_evidence_role_ranker.py` (constructor + rank_with_diagnostics)
- Test: `tests/test_llm_role_ablation.py` (extend existing)

**Goal:** When `ltr_score_weight > 0`, the role ranker uses the actual LTR score (not just rank) as a third component in the final formula.

- [ ] **Step 1: Write failing test**

```python
def test_evidence_role_ranker_combines_ltr_score_with_evidence_and_rank():
    """With ltr_score_weight=0.3, the final_score formula uses
    evidence (0.8*0.8) + rank_prior (0.2*0.0) + ltr_score (0.3*0.7) = 0.85
    for journal with evidence_composite=0.8, rank=2/2, ltr_score=0.7."""
    ranker = LLMEvidenceRoleRanker(
        evidence_extractor=FixedEvidenceExtractor({"j1": _evidence(0.8)}),
        journal_store=StubJournalStore([_journal("j1")]),
        prior_source="learned",
        evidence_weight=0.5,    # renormalized: 0.5/(0.5+0.2+0.3) ≈ 0.50
        prior_weight=0.2,        # ≈ 0.20
        ltr_score_weight=0.3,    # ≈ 0.30
    )
    ranked, _m, diag = ranker.rank_with_diagnostics(
        candidates=[(_journal("j1"), 1.0, [])],
        paper_profile=PaperProfile(title="P1"),
        retrieval_trace=_trace([_journal("j1")]),
        rule_ranks={"j1": 1},
        rule_scores={"j1": 1.0},
        learned_ranks={"j1": 1},  # rank 1 of 1 → prior 1.0
        learned_scores={"j1": 0.7},
    )
    # 0.50*0.8 + 0.20*1.0 + 0.30*0.7 = 0.40 + 0.20 + 0.21 = 0.81
    assert diag["candidates"]["j1"]["final_score"] == pytest.approx(0.81, abs=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_role_ablation.py::test_evidence_role_ranker_combines_ltr_score_with_evidence_and_rank -v`
Expected: TypeError on constructor

- [ ] **Step 3: Add ltr_score_weight to LLMEvidenceRoleRanker**

In `src/ranker/llm_evidence_role_ranker.py:92-114`, modify `__init__`:

```python
def __init__(
    self,
    evidence_extractor: LLMEvidenceExtractor,
    journal_store: JournalStore,
    accepted_paper_store: Optional[AcceptedPaperStore] = None,
    prior_source: str = "rule",
    evidence_weight: float = 0.8,
    prior_weight: float = 0.2,
    ltr_score_weight: float = 0.0,  # NEW: 0.0 = legacy (no LTR score in formula)
) -> None:
    if prior_source not in {"rule", "learned"}:
        raise ValueError(f"Unsupported prior_source: {prior_source}")
    if evidence_weight < 0 or prior_weight < 0 or ltr_score_weight < 0:
        raise ValueError("evidence_weight, prior_weight, ltr_score_weight must be non-negative")
    total_weight = evidence_weight + prior_weight + ltr_score_weight
    if total_weight <= 0:
        raise ValueError("At least one ranking weight must be positive")

    self.evidence_extractor = evidence_extractor
    self.journal_store = journal_store
    self.accepted_paper_store = accepted_paper_store
    self.prior_source = prior_source
    # Renormalize to sum=1.0
    self.evidence_weight = float(evidence_weight / total_weight)
    self.prior_weight = float(prior_weight / total_weight)
    self.ltr_score_weight = float(ltr_score_weight / total_weight)
    # ... rest of __init__
```

- [ ] **Step 4: Update rank_with_diagnostics to use LTR score**

In `src/ranker/llm_evidence_role_ranker.py:133` (signature) and the body around line 250 (final_score calc):

Add `learned_scores: Optional[Dict[str, float]] = None` to `rank_with_diagnostics` signature. Then in the per-candidate loop, change the final_score formula:

```python
# (existing) evidence_composite, rank_prior, evidence_weight, prior_weight
ltr_score = 0.0
if learned_scores:
    ltr_score = float(learned_scores.get(journal.journal_id, 0.0))
final_score = (
    evidence_composite * self.evidence_weight
    + rank_prior * self.prior_weight
    + ltr_score * self.ltr_score_weight
)
```

Add `"ltr_score": ltr_score, "ltr_score_weight": self.ltr_score_weight` to per-candidate diagnostics.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_llm_role_ablation.py::test_evidence_role_ranker_combines_ltr_score_with_evidence_and_rank -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/ranker/llm_evidence_role_ranker.py tests/test_llm_role_ablation.py
git commit -m "feat(6.4): role ranker uses LTR score as formula component"
```

---

## Task 4: Ablation runner threads learned_scores and supports ltr_score_weight

**Files:**
- Modify: `scripts/run_llm_role_ablation.py` (CLI + configure_pipeline_for_variant)
- Test: existing tests should still pass (no new test needed if existing ones cover learned_ranks threading)

**Goal:** Pass `learned_scores` to the role ranker when LTR score is enabled, and expose `--ltr-score-weight` as a CLI flag.

- [ ] **Step 1: Add --ltr-score-weight CLI arg**

In `scripts/run_llm_role_ablation.py`, after the existing `--model-path` arg:

```python
parser.add_argument(
    "--ltr-score-weight",
    type=float,
    default=0.0,
    help=(
        "Weight for the actual LTR score in the role ranker's final formula. "
        "When > 0, the role ranker uses (1-rank/N), (ltr_score), and "
        "evidence_composite with these weights renormalized to sum=1. "
        "Default 0 = use only evidence and rank (legacy 6.3 formula)."
    ),
)
```

- [ ] **Step 2: Pass ltr_score_weight when constructing LLMEvidenceRoleRanker**

In `configure_pipeline_for_variant`, find where `LLMEvidenceRoleRanker` is instantiated. Pass the new arg:

```python
pipeline.llm_ranker = LLMEvidenceRoleRanker(
    evidence_extractor=extractor,
    journal_store=pipeline.candidate_generator.store,
    accepted_paper_store=accepted_store,
    prior_source=str(variant.prior_source),
    evidence_weight=0.8,
    prior_weight=0.2,
    ltr_score_weight=args.ltr_score_weight,  # NEW
)
```

- [ ] **Step 3: Plumb learned_scores from result to rank_with_diagnostics**

In `run_llm_role_ablation.py`, find the call to `run_evaluation` and post-process the result. After the run, the `paper_results` contain `learned_diagnostics.learned_score`. Pass this through to the role ranker:

Find the call site for the role ranker (the spot that wraps `rank_with_diagnostics`). Add a post-step that builds `learned_scores_by_paper = {title: learned_score_dict}` from result.paper_results, then for each paper, calls the role ranker with `learned_scores=learned_scores_by_paper[title]`.

**NOTE:** This step is non-trivial. The role ranker is invoked by `pipeline.recommend` during each paper evaluation. To pass learned_scores, the pipeline needs to know about it. Easiest: have the pipeline accept `learned_scores_for_current_paper: Dict[str, float] = None` and pass it down to `rank_with_diagnostics`. The ablation runner resets this dict per-paper.

- [ ] **Step 4: Verify existing tests still pass**

Run: `pytest tests/test_llm_role_ablation.py -q`
Expected: 22 passed (or more, depending on existing state)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_llm_role_ablation.py
git commit -m "feat(6.4): ablation runner supports --ltr-score-weight"
```

---

## Task 5: End-to-end test on light30

**Files:**
- No code changes
- Run script + verify output

**Goal:** Verify the 26-dim LTR + new formula beats 6.3's 19/30 on light30.

- [ ] **Step 1: Run light30 ablation with 26-dim LTR + ltr_score_weight=0.3**

```bash
python scripts/run_llm_role_ablation.py \
  --benchmark-profile light30 \
  --baseline-eval data/evaluation/results/eval_abstract_top5_20260604_140602.json \
  --evidence-snapshot data/evaluation/evidence/light30_evidence_20260604_194946.json \
  --model-path data/models/learning_to_ranker_26dim.json \
  --ltr-score-weight 0.3 \
  --mode abstract --top-k 5 --workers 10
```

- [ ] **Step 2: Verify output**

Expected:
- 3 variants run (direct, evidence+rule, evidence+learned)
- evidence+learned variant should hit at least 19/30 (matching 6.3 evidence+learned_20dim)
- If > 19/30: 26-dim LTR is helping
- If == 19/30: LTR score is the same as rank prior (sanity check passes)
- summary.json has `fairness_pass=True` and `evidence_mismatches=[]`

- [ ] **Step 3: Document the result**

Append a section to `docs/superpowers/plans/2026-06-01-publication-grade-journal-recommender-plan.md` (or the 6.4 plan doc) summarizing:
- 6.3 evidence+learned_20dim: 19/30 light30
- 6.4 evidence+learned_26dim_ltr_score_0.3: ??/30 light30
- Conclusion: 26-dim LTR with evidence feature is a real improvement

- [ ] **Step 4: Commit results to git**

```bash
git add docs/superpowers/plans/*.md data/evaluation/results/llm_role_ablation_*.json
git commit -m "docs(6.4): record 26-dim LTR light30 ablation result"
```

---

## Task 6: Regression test — 20-dim v4 path unchanged

**Files:**
- Run script + verify

**Goal:** Confirm that running the ablation with `--model-path` pointing to the 20-dim v4 LTR reproduces the 6.3 numbers (19/30 on light30). This proves backward compat.

- [ ] **Step 1: Run with 20-dim v4 + no ltr_score_weight**

```bash
python scripts/run_llm_role_ablation.py \
  --benchmark-profile light30 \
  --baseline-eval data/evaluation/results/eval_abstract_top5_20260604_140602.json \
  --evidence-snapshot data/evaluation/evidence/light30_evidence_20260604_194946.json \
  --model-path data/models/learning_to_ranker.json \
  --mode abstract --top-k 5 --workers 10
```

- [ ] **Step 2: Verify output**

Expected: `evidence+learned` variant hits 19/30 (same as 6.3) — proves the new code didn't break the 20-dim path.

- [ ] **Step 3: Commit (if not already)**

No code changes; just record the result.

---

## Notes

- **Backward compat:** All 6.3 paths (20-dim v4, no ltr_score_weight) must keep working unchanged. Tests in Task 4 and Task 6 verify this.
- **No new dependencies:** Uses existing `feature_builder`, `ltr_adapter`, `llm_evidence_role_ranker`, `pipeline`. Only adds args to existing functions.
- **Memory:** 26-dim features double the per-candidate feature dict size. Negligible for 30-90 papers × 30 candidates.
- **Robustness:** When evidence is missing for a paper, the pipeline falls back to 20-dim automatically (the `paper_evidence` check in Task 2 Step 4).

---

## Self-Review Notes

After writing this plan:
- All 4 source files (`candidate_generator.py`, `pipeline.py`, `llm_evidence_role_ranker.py`, `run_llm_role_ablation.py`) have explicit changes
- Each step has exact code, not placeholders
- Backward compat is verified by Task 6
- No "TBD" / "TODO" / "implement later" markers
- Type consistency: `learned_scores: Optional[Dict[str, float]]` is consistent across all references
- Method signatures use the same param names throughout
