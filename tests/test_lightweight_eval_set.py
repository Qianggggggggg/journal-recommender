import json

from scripts.build_lightweight_eval_set import (
    CCF_AREAS,
    CCF_LEVELS,
    select_lightweight_papers,
)


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
