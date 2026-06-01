# Clean Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a clean benchmark workflow that detects evaluation leakage in typical abstracts and can generate a sanitized typical-abstract snapshot for fair evaluation.

**Architecture:** Put leakage detection and snapshot generation in a small reusable module under `src/evaluation`. Add a CLI script for reports/snapshots, and wire `scripts/run_evaluation.py` to use a clean snapshot directory and clean typical-vector index paths when requested.

**Tech Stack:** Python stdlib JSON/pathlib, existing pytest suite, existing `TypicalAbstractStore` and evaluation CLI.

---

### Task 1: Leakage Detection Module

**Files:**
- Create: `src/evaluation/clean_benchmark.py`
- Test: `tests/test_clean_benchmark.py`

- [ ] Write tests for detecting exact title and abstract-snippet overlaps between evaluation papers and typical abstracts.
- [ ] Implement normalized text matching, leakage report records, and summary counts.
- [ ] Verify with `pytest tests/test_clean_benchmark.py -q`.

### Task 2: Clean Snapshot Generation

**Files:**
- Modify: `src/evaluation/clean_benchmark.py`
- Test: `tests/test_clean_benchmark.py`

- [ ] Write tests proving leaked typical abstract entries are removed while non-leaked entries and journal metadata are preserved.
- [ ] Implement snapshot writing into a separate directory without mutating `data/typical_abstracts`.
- [ ] Verify with `pytest tests/test_clean_benchmark.py -q`.

### Task 3: CLI And Evaluation Wiring

**Files:**
- Create: `scripts/clean_benchmark.py`
- Modify: `scripts/run_evaluation.py`
- Test: `tests/test_clean_benchmark.py`

- [ ] Add CLI flags for report-only and snapshot generation.
- [ ] Add `run_evaluation.py --clean-benchmark --clean-typical-dir ...` support, disabling dirty typical-vector paths unless clean FAISS paths are provided.
- [ ] Verify with targeted pytest and a report-only command on current data.
