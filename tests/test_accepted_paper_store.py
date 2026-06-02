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


# ---------------------------------------------------------------------------
# 以下为 collect_accepted_papers.py (任务 2.2) 的行为测试。
# 这一组测试覆盖:venue→journal_id 解析、--exclude-eval-input 排除、
# source 字段、缺字段跳过、未知 venue 跳过、跨文件去重,以及与
# AcceptedPaperStore.load() 的端到端兼容。
# ---------------------------------------------------------------------------


def _make_journal_store(records):
    """构造一个内存中的 JournalStore-like 对象,带 journals 属性。

    避免引入对 faiss / numpy 的依赖,只暴露 collect 脚本真正用到的接口。
    """
    class _FakeJournal:
        def __init__(self, journal_id, journal_name):
            self.journal_id = journal_id
            self.journal_name = journal_name

    class _FakeStore:
        def __init__(self, journals):
            self.journals = journals

    return _FakeStore([_FakeJournal(jid, jname) for jid, jname in records])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_collect_groups_papers_by_resolved_journal_id(tmp_path):
    from scripts.collect_accepted_papers import collect_accepted_papers

    journal_store = _make_journal_store([
        ("ton", "IEEE/ACM Transactions on Networking"),
        ("ai", "Artificial Intelligence"),
    ])
    meta_path = tmp_path / "metadata.jsonl"
    _write_jsonl(
        meta_path,
        [
            {
                "title": "Net paper 1",
                "abstract": "net abstract 1",
                "venue": "IEEE/ACM Transactions on Networking",
                "year": 2025,
                "external_ids": {"doi": "10.1109/net1"},
            },
            {
                "title": "Net paper 2",
                "abstract": "net abstract 2",
                "venue": "IEEE/ACM Transactions on Networking",
            },
            {
                "title": "AI paper 1",
                "abstract": "ai abstract 1",
                "venue": "Artificial Intelligence",
            },
        ],
    )
    out_dir = tmp_path / "out"

    summary = collect_accepted_papers(
        eval_inputs=[meta_path],
        exclude_inputs=[],
        journal_store=journal_store,
        output_dir=out_dir,
        source="local_evaluation_metadata",
    )

    assert (out_dir / "ton.json").exists()
    assert (out_dir / "ai.json").exists()
    ton = json.loads((out_dir / "ton.json").read_text(encoding="utf-8"))
    assert ton["journal_id"] == "ton"
    assert ton["journal_name"] == "IEEE/ACM Transactions on Networking"
    assert len(ton["papers"]) == 2
    assert summary["journal_count"] == 2
    assert summary["paper_count"] == 3


def test_collect_writes_source_field_for_every_paper(tmp_path):
    from scripts.collect_accepted_papers import collect_accepted_papers

    journal_store = _make_journal_store([("ai", "Artificial Intelligence")])
    meta_path = tmp_path / "metadata.jsonl"
    _write_jsonl(
        meta_path,
        [
            {"title": "P", "abstract": "A", "venue": "Artificial Intelligence"},
        ],
    )
    out_dir = tmp_path / "out"
    collect_accepted_papers(
        eval_inputs=[meta_path],
        exclude_inputs=[],
        journal_store=journal_store,
        output_dir=out_dir,
        source="local_evaluation_metadata",
    )

    payload = json.loads((out_dir / "ai.json").read_text(encoding="utf-8"))
    assert payload["papers"][0]["source"] == "local_evaluation_metadata"


def test_collect_preserves_year_and_doi_when_present(tmp_path):
    from scripts.collect_accepted_papers import collect_accepted_papers

    journal_store = _make_journal_store([("ai", "Artificial Intelligence")])
    meta_path = tmp_path / "metadata.jsonl"
    _write_jsonl(
        meta_path,
        [
            {
                "title": "P",
                "abstract": "A",
                "venue": "Artificial Intelligence",
                "year": 2024,
                "external_ids": {"doi": "10.1/x", "arXiv": "ignored"},
                "pdf_url": "https://example.org/p.pdf",
            },
        ],
    )
    out_dir = tmp_path / "out"
    collect_accepted_papers(
        eval_inputs=[meta_path],
        exclude_inputs=[],
        journal_store=journal_store,
        output_dir=out_dir,
        source="local_evaluation_metadata",
    )

    paper = json.loads((out_dir / "ai.json").read_text(encoding="utf-8"))["papers"][0]
    assert paper["year"] == 2024
    assert paper["doi"] == "10.1/x"
    assert paper["url"] == "https://example.org/p.pdf"


def test_collect_skips_unknown_venue(tmp_path):
    from scripts.collect_accepted_papers import collect_accepted_papers

    journal_store = _make_journal_store([("ai", "Artificial Intelligence")])
    meta_path = tmp_path / "metadata.jsonl"
    _write_jsonl(
        meta_path,
        [
            {"title": "Known", "abstract": "A", "venue": "Artificial Intelligence"},
            {"title": "Unknown", "abstract": "A", "venue": "Imaginary Journal of Nothing"},
        ],
    )
    out_dir = tmp_path / "out"
    summary = collect_accepted_papers(
        eval_inputs=[meta_path],
        exclude_inputs=[],
        journal_store=journal_store,
        output_dir=out_dir,
        source="local_evaluation_metadata",
    )

    payload = json.loads((out_dir / "ai.json").read_text(encoding="utf-8"))
    assert len(payload["papers"]) == 1
    assert payload["papers"][0]["title"] == "Known"
    assert summary["unresolved_venue_count"] == 1
    assert "Imaginary Journal of Nothing" in summary["unresolved_venues"]


def test_collect_skips_records_missing_required_fields(tmp_path):
    from scripts.collect_accepted_papers import collect_accepted_papers

    journal_store = _make_journal_store([("ai", "Artificial Intelligence")])
    meta_path = tmp_path / "metadata.jsonl"
    _write_jsonl(
        meta_path,
        [
            {"title": "Good", "abstract": "A", "venue": "Artificial Intelligence"},
            {"abstract": "A", "venue": "Artificial Intelligence"},          # title 缺
            {"title": "X", "venue": "Artificial Intelligence"},             # abstract 缺
            {"title": "Y", "abstract": "B"},                                # venue 缺
            {"title": "Z", "abstract": "   ", "venue": "Artificial Intelligence"},  # abstract 仅空白
        ],
    )
    out_dir = tmp_path / "out"
    summary = collect_accepted_papers(
        eval_inputs=[meta_path],
        exclude_inputs=[],
        journal_store=journal_store,
        output_dir=out_dir,
        source="local_evaluation_metadata",
    )

    payload = json.loads((out_dir / "ai.json").read_text(encoding="utf-8"))
    assert [p["title"] for p in payload["papers"]] == ["Good"]
    assert summary["skipped_missing_fields_count"] == 4


def test_collect_venue_match_is_case_and_whitespace_insensitive(tmp_path):
    from scripts.collect_accepted_papers import collect_accepted_papers

    journal_store = _make_journal_store([
        ("ton", "IEEE/ACM Transactions on Networking"),
    ])
    meta_path = tmp_path / "metadata.jsonl"
    _write_jsonl(
        meta_path,
        [
            {
                "title": "P",
                "abstract": "A",
                "venue": "  ieee/acm transactions on NETWORKING  ",
            },
        ],
    )
    out_dir = tmp_path / "out"
    collect_accepted_papers(
        eval_inputs=[meta_path],
        exclude_inputs=[],
        journal_store=journal_store,
        output_dir=out_dir,
        source="local_evaluation_metadata",
    )

    assert (out_dir / "ton.json").exists()


def test_collect_excludes_papers_from_exclude_eval_input(tmp_path):
    """同一篇 (title, venue) 若出现在 --exclude-eval-input 指定文件中,绝不进 corpus。"""
    from scripts.collect_accepted_papers import collect_accepted_papers

    journal_store = _make_journal_store([("ai", "Artificial Intelligence")])
    primary = tmp_path / "primary.jsonl"
    light30 = tmp_path / "light30.jsonl"
    _write_jsonl(
        primary,
        [
            {"title": "Held-out paper", "abstract": "secret", "venue": "Artificial Intelligence"},
            {"title": "Normal paper", "abstract": "normal", "venue": "Artificial Intelligence"},
        ],
    )
    _write_jsonl(
        light30,
        [
            {"title": "Held-out paper", "abstract": "secret", "venue": "Artificial Intelligence"},
        ],
    )
    out_dir = tmp_path / "out"
    summary = collect_accepted_papers(
        eval_inputs=[primary],
        exclude_inputs=[light30],
        journal_store=journal_store,
        output_dir=out_dir,
        source="local_evaluation_metadata",
    )

    payload = json.loads((out_dir / "ai.json").read_text(encoding="utf-8"))
    assert [p["title"] for p in payload["papers"]] == ["Normal paper"]
    assert summary["excluded_count"] == 1


def test_collect_dedupes_same_title_across_input_files(tmp_path):
    """同一篇论文(title 规范化后相同 + 同 venue)在多个 input 中只算一次。"""
    from scripts.collect_accepted_papers import collect_accepted_papers

    journal_store = _make_journal_store([("ai", "Artificial Intelligence")])
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_jsonl(
        a,
        [{"title": "Same Title", "abstract": "X", "venue": "Artificial Intelligence"}],
    )
    _write_jsonl(
        b,
        [
            {"title": "  same   title  ", "abstract": "X", "venue": "Artificial Intelligence"},
            {"title": "Different Title", "abstract": "Y", "venue": "Artificial Intelligence"},
        ],
    )
    out_dir = tmp_path / "out"
    summary = collect_accepted_papers(
        eval_inputs=[a, b],
        exclude_inputs=[],
        journal_store=journal_store,
        output_dir=out_dir,
        source="local_evaluation_metadata",
    )

    payload = json.loads((out_dir / "ai.json").read_text(encoding="utf-8"))
    titles = sorted(p["title"] for p in payload["papers"])
    assert titles == ["Different Title", "Same Title"]
    assert summary["paper_count"] == 2
    assert summary["duplicate_count"] == 1


def test_collect_output_is_loadable_by_accepted_paper_store(tmp_path):
    """端到端:collect 产物能被 AcceptedPaperStore.load() 完整读回。"""
    from scripts.collect_accepted_papers import collect_accepted_papers
    from src.journals.accepted_paper_store import AcceptedPaperStore

    journal_store = _make_journal_store([
        ("ton", "IEEE/ACM Transactions on Networking"),
        ("ai", "Artificial Intelligence"),
    ])
    meta = tmp_path / "metadata.jsonl"
    _write_jsonl(
        meta,
        [
            {"title": "Net A", "abstract": "x", "venue": "IEEE/ACM Transactions on Networking", "year": 2024},
            {"title": "AI A",  "abstract": "y", "venue": "Artificial Intelligence", "year": 2025},
        ],
    )
    out_dir = tmp_path / "out"
    collect_accepted_papers(
        eval_inputs=[meta],
        exclude_inputs=[],
        journal_store=journal_store,
        output_dir=out_dir,
        source="local_evaluation_metadata",
    )

    store = AcceptedPaperStore(accepted_dir=str(out_dir))
    store.load()
    assert store.journal_count == 2
    assert store.get_papers("ton")[0]["title"] == "Net A"
    assert store.get_papers("ai")[0]["year"] == 2025
    # 端到端检查 source 字段一致
    assert store.get_papers("ai")[0]["source"] == "local_evaluation_metadata"


def test_collect_excludes_pair_by_title_and_venue_not_just_title(tmp_path):
    """exclude 必须按 (normalized_title, journal_id) 精确匹配,不能因为别处一本期刊同名标题就误杀。"""
    from scripts.collect_accepted_papers import collect_accepted_papers

    journal_store = _make_journal_store([
        ("ton", "IEEE/ACM Transactions on Networking"),
        ("ai", "Artificial Intelligence"),
    ])
    primary = tmp_path / "primary.jsonl"
    excluded = tmp_path / "ex.jsonl"
    _write_jsonl(
        primary,
        [
            {"title": "Survey of X", "abstract": "x", "venue": "IEEE/ACM Transactions on Networking"},
            {"title": "Survey of X", "abstract": "x", "venue": "Artificial Intelligence"},
        ],
    )
    _write_jsonl(
        excluded,
        [
            {"title": "Survey of X", "abstract": "x", "venue": "Artificial Intelligence"},
        ],
    )
    out_dir = tmp_path / "out"
    collect_accepted_papers(
        eval_inputs=[primary],
        exclude_inputs=[excluded],
        journal_store=journal_store,
        output_dir=out_dir,
        source="local_evaluation_metadata",
    )

    ton = json.loads((out_dir / "ton.json").read_text(encoding="utf-8"))
    assert [p["title"] for p in ton["papers"]] == ["Survey of X"]
    assert not (out_dir / "ai.json").exists()  # AI 这条被 exclude 掉了

