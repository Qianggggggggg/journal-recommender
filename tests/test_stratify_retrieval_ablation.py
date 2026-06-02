"""Tests for scripts/stratify_retrieval_ablation.py."""
import json
from pathlib import Path

import pytest

from scripts.stratify_retrieval_ablation import (
    compute_coverage,
    compute_stratified_metrics,
    format_stratified_report,
)


def _write_journal_paper(dir_path: Path, journal_id: str, papers: list[dict]) -> None:
    """Helper: write a single accepted-paper JSON file for one journal."""
    payload = {"journal_id": journal_id, "journal_name": journal_id, "papers": papers}
    (dir_path / f"{journal_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_compute_coverage_marks_paper_covered_when_gold_venue_in_corpus(tmp_path: Path):
    """如果 paper 的 target_journal_id 在 accepted_papers 目录里存在,则 covered=True。"""
    _write_journal_paper(tmp_path, "tpds", [{"title": "x", "abstract": "y" * 50}])
    _write_journal_paper(tmp_path, "tog", [{"title": "a", "abstract": "b" * 50}])

    paper_results = [
        {"title": "P1", "target_journal_id": "tpds"},
        {"title": "P2", "target_journal_id": "tog"},
        {"title": "P3", "target_journal_id": "unknown_journal"},
    ]

    coverage = compute_coverage(paper_results, accepted_papers_dir=tmp_path)

    assert coverage == {0: True, 1: True, 2: False}


def test_compute_coverage_returns_empty_dict_for_empty_corpus(tmp_path: Path):
    """空 corpus 目录 → 所有 paper 都 uncovered。"""
    paper_results = [
        {"title": "P1", "target_journal_id": "tpds"},
        {"title": "P2", "target_journal_id": "tog"},
    ]

    coverage = compute_coverage(paper_results, accepted_papers_dir=tmp_path)

    assert coverage == {0: False, 1: False}


def test_compute_coverage_skips_journal_files_with_empty_papers_list(tmp_path: Path):
    """空 papers 列表的 journal 文件不计入 covered(避免把 '空壳' 当成有画像)。"""
    _write_journal_paper(tmp_path, "tpds", [])  # 空 papers
    _write_journal_paper(tmp_path, "tog", [{"title": "x", "abstract": "y" * 50}])

    paper_results = [
        {"title": "P1", "target_journal_id": "tpds"},
        {"title": "P2", "target_journal_id": "tog"},
    ]

    coverage = compute_coverage(paper_results, accepted_papers_dir=tmp_path)

    assert coverage == {0: False, 1: True}


def test_compute_stratified_metrics_reports_rule_at_5_for_subset():
    """对 subset=[0,1] 切片,只统计这两篇的 rule@5 等指标。"""
    paper_results = [
        # rule_rank=2 → rule@5 hit
        {"title": "P1", "target_journal_id": "a", "retrieval_rank": 1, "rule_rank": 2},
        # rule_rank=None → miss
        {"title": "P2", "target_journal_id": "b", "retrieval_rank": 5, "rule_rank": None},
        # rule_rank=8 → rule@5 miss, 但 rule@20 hit
        {"title": "P3", "target_journal_id": "c", "retrieval_rank": 12, "rule_rank": 8},
    ]

    metrics = compute_stratified_metrics(paper_results, subset=[0, 1])

    assert metrics["n"] == 2
    assert metrics["coarse_at_50"] == 2  # both have retrieval_rank <= 50
    assert metrics["rule_at_5"] == 1  # only P1
    assert metrics["rule_at_20"] == 1  # only P1 (P2 has rule_rank=None, P3 not in subset)


def test_format_stratified_report_emits_overall_covered_uncovered_sections():
    """报告必须包含 overall / covered / uncovered 三个分块标题。"""
    paper_results = [
        {"title": "P1", "target_journal_id": "a", "retrieval_rank": 1, "rule_rank": 1},
        {"title": "P2", "target_journal_id": "b", "retrieval_rank": 5, "rule_rank": 3},
    ]
    coverage = {0: True, 1: False}

    report = format_stratified_report(
        ablation_data={"variants": {"hybrid": {"paper_results": paper_results}}},
        variants=["hybrid"],
        coverage=coverage,
    )

    assert "## Overall (n=2)" in report
    assert "## Covered (n=1)" in report
    assert "## Uncovered (n=1)" in report
