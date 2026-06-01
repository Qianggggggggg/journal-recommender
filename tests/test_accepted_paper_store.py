"""AcceptedPaperStore 加载与查询行为测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_journal(dir_path: Path, journal_id: str, payload: dict) -> Path:
    file_path = dir_path / f"{journal_id}.json"
    file_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return file_path


def test_load_single_journal_file(tmp_path):
    from src.journals.accepted_paper_store import AcceptedPaperStore

    _write_journal(
        tmp_path,
        "ton",
        {
            "journal_id": "ton",
            "journal_name": "IEEE/ACM Transactions on Networking",
            "papers": [
                {
                    "title": "A scalable congestion control mechanism",
                    "abstract": "We propose a scalable congestion control mechanism for high-speed networks.",
                    "year": 2025,
                    "source": "local_evaluation_metadata",
                    "doi": "10.1109/example",
                    "url": "https://example.org/paper",
                }
            ],
        },
    )

    store = AcceptedPaperStore(accepted_dir=str(tmp_path))
    store.load()

    assert store.journal_count == 1
    assert store.count == 1
    papers = store.get_papers("ton")
    assert len(papers) == 1
    assert papers[0]["title"] == "A scalable congestion control mechanism"
    assert papers[0]["year"] == 2025
    assert papers[0]["source"] == "local_evaluation_metadata"


def test_load_multiple_journal_files(tmp_path):
    from src.journals.accepted_paper_store import AcceptedPaperStore

    _write_journal(
        tmp_path,
        "ton",
        {
            "journal_id": "ton",
            "journal_name": "IEEE/ACM Transactions on Networking",
            "papers": [
                {"title": "Net paper 1", "abstract": "abs 1"},
                {"title": "Net paper 2", "abstract": "abs 2"},
            ],
        },
    )
    _write_journal(
        tmp_path,
        "ai",
        {
            "journal_id": "ai",
            "journal_name": "Artificial Intelligence",
            "papers": [
                {"title": "AI paper 1", "abstract": "abs A"},
            ],
        },
    )

    store = AcceptedPaperStore(accepted_dir=str(tmp_path))
    store.load()

    assert store.journal_count == 2
    assert store.count == 3
    assert len(store.get_papers("ton")) == 2
    assert len(store.get_papers("ai")) == 1


def test_get_papers_unknown_journal_returns_empty_list(tmp_path):
    from src.journals.accepted_paper_store import AcceptedPaperStore

    _write_journal(
        tmp_path,
        "ai",
        {"journal_id": "ai", "journal_name": "Artificial Intelligence", "papers": []},
    )
    store = AcceptedPaperStore(accepted_dir=str(tmp_path))
    store.load()

    assert store.get_papers("does-not-exist") == []


def test_iter_records_yields_paper_journal_pairs(tmp_path):
    from src.journals.accepted_paper_store import (
        AcceptedPaperRecord,
        AcceptedPaperStore,
    )

    _write_journal(
        tmp_path,
        "ton",
        {
            "journal_id": "ton",
            "journal_name": "IEEE/ACM Transactions on Networking",
            "papers": [
                {"title": "Paper A", "abstract": "abs A"},
                {"title": "Paper B", "abstract": "abs B"},
            ],
        },
    )
    _write_journal(
        tmp_path,
        "ai",
        {
            "journal_id": "ai",
            "journal_name": "Artificial Intelligence",
            "papers": [{"title": "Paper C", "abstract": "abs C"}],
        },
    )
    store = AcceptedPaperStore(accepted_dir=str(tmp_path))
    store.load()

    records = list(store.iter_records())
    assert len(records) == 3
    assert all(isinstance(r, AcceptedPaperRecord) for r in records)

    by_journal: dict[str, list[str]] = {}
    for record in records:
        by_journal.setdefault(record.journal_id, []).append(record.title)
    assert sorted(by_journal["ton"]) == ["Paper A", "Paper B"]
    assert by_journal["ai"] == ["Paper C"]
    # journal_name 也带过来
    ton_record = next(r for r in records if r.journal_id == "ton")
    assert ton_record.journal_name == "IEEE/ACM Transactions on Networking"


def test_records_missing_title_or_abstract_are_skipped(tmp_path):
    """title 或 abstract 缺失/空白的论文应跳过,但不应让整个加载失败。"""
    from src.journals.accepted_paper_store import AcceptedPaperStore

    _write_journal(
        tmp_path,
        "ton",
        {
            "journal_id": "ton",
            "journal_name": "IEEE/ACM Transactions on Networking",
            "papers": [
                {"title": "Valid paper", "abstract": "valid abstract"},
                {"title": "Missing abstract"},          # abstract 缺失
                {"abstract": "Missing title"},          # title 缺失
                {"title": "", "abstract": "empty title"},  # title 空
                {"title": "empty abstract", "abstract": "   "},  # abstract 仅空白
                {"title": "another valid", "abstract": "another"},
            ],
        },
    )
    store = AcceptedPaperStore(accepted_dir=str(tmp_path))
    store.load()

    # 6 篇里只有 2 篇有效
    papers = store.get_papers("ton")
    assert len(papers) == 2
    assert {p["title"] for p in papers} == {"Valid paper", "another valid"}
    assert store.count == 2


def test_load_uses_filename_stem_when_journal_id_missing(tmp_path):
    """journal_id 字段缺失时,使用文件名 stem 兜底,与 TypicalAbstractStore 行为对齐。"""
    from src.journals.accepted_paper_store import AcceptedPaperStore

    _write_journal(
        tmp_path,
        "ton",
        {
            # 故意省略 journal_id
            "journal_name": "IEEE/ACM Transactions on Networking",
            "papers": [{"title": "P", "abstract": "A"}],
        },
    )
    store = AcceptedPaperStore(accepted_dir=str(tmp_path))
    store.load()

    assert store.get_papers("ton")[0]["title"] == "P"


def test_optional_fields_default_to_empty_or_none(tmp_path):
    """year/source/doi/url 缺失时应使用稳定默认值,避免下游 KeyError。"""
    from src.journals.accepted_paper_store import AcceptedPaperStore

    _write_journal(
        tmp_path,
        "ai",
        {
            "journal_id": "ai",
            "journal_name": "Artificial Intelligence",
            "papers": [{"title": "Minimal paper", "abstract": "min abstract"}],
        },
    )
    store = AcceptedPaperStore(accepted_dir=str(tmp_path))
    store.load()

    paper = store.get_papers("ai")[0]
    assert paper["year"] is None
    assert paper["source"] == ""
    assert paper["doi"] == ""
    assert paper["url"] == ""


def test_load_missing_directory_does_not_raise(tmp_path):
    """目录不存在时,load() 应静默返回,不能抛异常。"""
    from src.journals.accepted_paper_store import AcceptedPaperStore

    missing = tmp_path / "does_not_exist"
    store = AcceptedPaperStore(accepted_dir=str(missing))
    store.load()

    assert store.count == 0
    assert store.journal_count == 0
    assert store.get_papers("anything") == []
    assert list(store.iter_records()) == []


def test_load_corrupt_json_skips_file_and_loads_others(tmp_path):
    """单个文件 JSON 解析失败时,应跳过该文件并继续加载其他文件,而不是整体失败。"""
    from src.journals.accepted_paper_store import AcceptedPaperStore

    # 一个坏的 JSON 文件
    (tmp_path / "broken.json").write_text("{ this is not json", encoding="utf-8")
    # 一个好的
    _write_journal(
        tmp_path,
        "ai",
        {
            "journal_id": "ai",
            "journal_name": "Artificial Intelligence",
            "papers": [{"title": "ok paper", "abstract": "ok"}],
        },
    )

    store = AcceptedPaperStore(accepted_dir=str(tmp_path))
    store.load()

    assert store.journal_count == 1
    assert store.get_papers("ai")[0]["title"] == "ok paper"


def test_papers_key_missing_does_not_raise(tmp_path):
    """papers 字段缺失时也不能崩,该期刊视为 0 论文。"""
    from src.journals.accepted_paper_store import AcceptedPaperStore

    _write_journal(
        tmp_path,
        "ai",
        {"journal_id": "ai", "journal_name": "Artificial Intelligence"},
    )
    store = AcceptedPaperStore(accepted_dir=str(tmp_path))
    store.load()

    # 没有 papers 字段也算成功加载该 journal,只是 papers 为空
    assert store.journal_count == 0  # 空 papers 不计入 journal_count
    assert store.get_papers("ai") == []
