# Evaluation Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make concurrent `run_evaluation.py --workers 10` comparisons reproducible by reusing fixed PaperProfile snapshots and preventing silent empty LLM recommendations.

**Architecture:** Validate LLM ranking output at the ranker boundary so existing retries cover malformed/empty results. Let the pipeline degrade to Rule TopK only after LLM retries are exhausted. Add an evaluation-only baseline snapshot loader and persist fallback/concurrency diagnostics in result JSON.

**Tech Stack:** Python 3.11, pytest, tenacity, dataclasses, concurrent futures.

---

### Task 1: Reject Empty LLM Ranking Results

**Files:**
- Modify: `src/ranker/llm_ranker.py`
- Create: `tests/test_llm_ranker.py`

- [ ] Write tests proving empty `rankings`, malformed items, and unknown-only journal IDs raise `LLMRankerError`.
- [ ] Run `pytest tests/test_llm_ranker.py -q` and confirm failures.
- [ ] Validate ranking structure and require at least one valid candidate result.
- [ ] Run `pytest tests/test_llm_ranker.py -q` and confirm pass.

### Task 2: Add Rule TopK Pipeline Fallback

**Files:**
- Modify: `src/recommender/pipeline.py`
- Modify: `tests/test_recommender.py`

- [ ] Write a failing test where `LLMRanker.rank()` raises and pipeline must return Rule TopK.
- [ ] Run the targeted test and confirm failure.
- [ ] Catch exhausted `LLMRankerError`, produce Rule TopK, and return fallback diagnostics.
- [ ] Verify normal LLM path and fallback path tests.

### Task 3: Reuse Fixed PaperProfile Snapshots

**Files:**
- Modify: `scripts/run_evaluation.py`
- Modify: `tests/test_run_evaluation_diagnostics.py`
- Modify: `tests/test_benchmark_manifest.py`

- [ ] Write failing tests proving a supplied snapshot bypasses parser and missing snapshots fail fast.
- [ ] Run targeted tests and confirm failures.
- [ ] Add baseline evaluation loader, snapshot-to-`PaperProfile` conversion, CLI `--baseline-eval`, and manifest fields.
- [ ] Verify snapshot reuse under concurrent evaluation.

### Task 4: Persist Stability Diagnostics

**Files:**
- Modify: `scripts/run_evaluation.py`
- Modify: `tests/test_run_evaluation_diagnostics.py`

- [ ] Write failing tests for per-paper fallback fields and aggregate fallback/empty counts.
- [ ] Run targeted tests and confirm failures.
- [ ] Persist `rank_method`, `evaluation_status`, fallback fields, and aggregate counters.
- [ ] Run targeted tests and confirm pass.

### Task 5: Regression Verification

**Files:**
- Verify only.

- [ ] Run:

```bash
pytest tests/test_llm_ranker.py tests/test_recommender.py tests/test_run_evaluation_diagnostics.py tests/test_benchmark_manifest.py -q
```

- [ ] Run:

```bash
python -m py_compile src/ranker/llm_ranker.py src/recommender/pipeline.py scripts/run_evaluation.py
```

- [ ] Run a local no-network snapshot-loader smoke check using the latest Light30 result.
- [ ] Confirm `git diff --check` passes for all changed files.
