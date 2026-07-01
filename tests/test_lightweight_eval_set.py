import json

from scripts.build_lightweight_eval_set import (
    CCF_AREAS,
    CCF_LEVELS,
    validate_lightweight_eval_set,
    select_lightweight_papers,
)
from scripts.run_evaluation import resolve_benchmark_input


def _paper(title, area, level, source="primary"):
    return {
        "title": title,
        "abstract": f"Abstract for {title}.",
        "venue": f"{title} Journal",
        "ccf_level": level,
        "research_area": [area],
        "_source": source,
    }


def test_select_lightweight_papers_prefers_primary_and_fills_missing_from_fallback():
    primary = [
        _paper(f"{area}-{level}-primary", area, level)
        for area in CCF_AREAS
        for level in CCF_LEVELS
        if not (area == "人机交互与普适计算" and level == "C")
    ]
    fallback = [
        _paper("hci-c-fallback", "人机交互与普适计算", "C", source="fallback")
    ]

    selected, report = select_lightweight_papers(primary, fallback)

    assert len(selected) == 30
    assert report["summary"]["selected_count"] == 30
    assert report["summary"]["missing_combo_count"] == 0
    assert {
        ((paper["research_area"] or [""])[0], paper["ccf_level"])
        for paper in selected
    } == {
        (area, level)
        for area in CCF_AREAS
        for level in CCF_LEVELS
    }
    hci_c = [
        paper
        for paper in selected
        if paper["research_area"] == ["人机交互与普适计算"] and paper["ccf_level"] == "C"
    ][0]
    assert hci_c["title"] == "hci-c-fallback"
    assert report["source_counts"]["fallback"] == 1


def test_select_lightweight_papers_reports_missing_combos():
    selected, report = select_lightweight_papers([], [])

    assert selected == []
    assert report["summary"]["selected_count"] == 0
    assert report["summary"]["missing_combo_count"] == 30
    assert report["missing_combos"][0] == {
        "research_area": CCF_AREAS[0],
        "ccf_level": "A",
    }


def test_validate_lightweight_eval_set_accepts_exact_30_combo_set(tmp_path):
    path = tmp_path / "light30.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for area in CCF_AREAS:
            for level in CCF_LEVELS:
                f.write(json.dumps(_paper(f"{area}-{level}", area, level), ensure_ascii=False) + "\n")

    report = validate_lightweight_eval_set(path)

    assert report["valid"] is True
    assert report["summary"]["selected_count"] == 30
    assert report["summary"]["missing_combo_count"] == 0
    assert report["summary"]["duplicate_combo_count"] == 0


def test_validate_lightweight_eval_set_rejects_duplicate_combo(tmp_path):
    path = tmp_path / "light30.jsonl"
    rows = [_paper("a", CCF_AREAS[0], "A"), _paper("b", CCF_AREAS[0], "A")]
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = validate_lightweight_eval_set(path)

    assert report["valid"] is False
    assert report["summary"]["selected_count"] == 2
    assert report["summary"]["duplicate_combo_count"] == 1


def test_resolve_benchmark_input_uses_profile_defaults():
    assert resolve_benchmark_input("light30", None) == "data/evaluation/papers_metadata_light_30.jsonl"
    assert resolve_benchmark_input("full-v2", None) == "data/evaluation/papers_metadata_v2.jsonl"
    assert resolve_benchmark_input("custom", "custom.jsonl") == "custom.jsonl"
