"""AcceptedPaperBM25Retriever 行为测试。

任务 3.1 的核心契约:
- 从 ``AcceptedPaperStore`` 的真实论文集合建 BM25 索引
- 期刊聚合规则:max(paper_score) + 0.05 * matching_paper_count,封顶
- ``route_detail`` 记录每本期刊命中的 top matching paper title + score
- 与 ``Journal`` 对接,返回 ``List[Tuple[Journal, score]]``
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest


def _make_journal_store(records):
    """轻量 fake JournalStore (避免拉 faiss 依赖)。"""
    from src.journals.journal_model import Journal

    class _FakeStore:
        def __init__(self, journals):
            self._by_id = {j.journal_id: j for j in journals}
            self.journals = journals

        def get_journal(self, jid):
            return self._by_id.get(jid)

    journals = [
        Journal(journal_id=jid, journal_name=jname, scope_text="")
        for jid, jname in records
    ]
    return _FakeStore(journals)


def _write_corpus(tmp_path: Path, journal_id: str, journal_name: str, papers):
    file_path = tmp_path / f"{journal_id}.json"
    file_path.write_text(
        json.dumps(
            {"journal_id": journal_id, "journal_name": journal_name, "papers": papers},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return file_path


def _make_stores(tmp_path, payloads):
    """payloads: list of (journal_id, journal_name, [(title, abstract), ...])"""
    from src.journals.accepted_paper_store import AcceptedPaperStore

    for jid, jname, papers in payloads:
        _write_corpus(
            tmp_path,
            jid,
            jname,
            [{"title": t, "abstract": a} for t, a in papers],
        )

    accepted = AcceptedPaperStore(accepted_dir=str(tmp_path))
    accepted.load()
    journal_store = _make_journal_store([(jid, jname) for jid, jname, _ in payloads])
    return accepted, journal_store


def test_network_query_recalls_networking_journal(tmp_path):
    """plan 明文要求:网络时延相关 query 能从 accepted papers 中召回正确的网络期刊。

    BM25 在小语料下 IDF 不稳定,所以每个期刊给若干篇论文模拟真实 corpus 规模。
    """
    from src.retriever.accepted_paper_retriever import AcceptedPaperBM25Retriever

    accepted, journals = _make_stores(
        tmp_path,
        [
            (
                "ton",
                "IEEE/ACM Transactions on Networking",
                [
                    ("Reducing TCP latency in datacenter networks",
                     "We propose a congestion-aware scheduling algorithm "
                     "that reduces tail latency for TCP flows in datacenter networks."),
                    ("ECN-based flow scheduling for cloud datacenters",
                     "Explicit congestion notification combined with priority "
                     "scheduling improves datacenter network throughput under load."),
                    ("BBR over wide-area networks",
                     "Bottleneck bandwidth and round-trip propagation time (BBR) "
                     "congestion control deployed across wide-area Internet paths."),
                ],
            ),
            (
                "ai",
                "Artificial Intelligence",
                [
                    ("Symbolic reasoning over knowledge graphs",
                     "We present a neuro-symbolic framework for reasoning over "
                     "large-scale knowledge graphs with calibrated uncertainty."),
                    ("Probabilistic logic programming",
                     "A probabilistic logic programming language with efficient "
                     "weighted model counting for AI inference tasks."),
                    ("Commonsense reasoning benchmarks",
                     "Survey of commonsense reasoning benchmarks and the limits "
                     "of current large language models on multi-hop inference."),
                ],
            ),
        ],
    )

    retriever = AcceptedPaperBM25Retriever(accepted, journals)
    results = retriever.retrieve(
        "TCP congestion control datacenter scheduling latency",
        top_k=5,
    )

    assert len(results) >= 1
    top_journal, top_score = results[0]
    assert top_journal.journal_id == "ton"
    assert top_score > 0


def test_empty_store_returns_empty_list(tmp_path):
    from src.journals.accepted_paper_store import AcceptedPaperStore
    from src.retriever.accepted_paper_retriever import AcceptedPaperBM25Retriever

    accepted = AcceptedPaperStore(accepted_dir=str(tmp_path))
    accepted.load()
    journals = _make_journal_store([])

    retriever = AcceptedPaperBM25Retriever(accepted, journals)
    assert retriever.retrieve("any query", top_k=10) == []


def test_aggregate_uses_max_score_plus_count_bonus(tmp_path):
    """聚合规则:journal_score = max(paper_scores) + 0.05 * matching_paper_count,
    封顶 cap 防止 bonus 失控。"""
    from src.retriever.accepted_paper_retriever import AcceptedPaperBM25Retriever

    # ai 期刊有 3 篇都跟查询语义重合的论文 → 应有 count bonus
    # ton 期刊只有 1 篇匹配 → 没 count bonus
    accepted, journals = _make_stores(
        tmp_path,
        [
            (
                "ai",
                "Artificial Intelligence",
                [
                    ("Knowledge graph reasoning", "We propose a knowledge graph reasoning framework."),
                    ("Knowledge graph completion", "Methods for knowledge graph completion via embeddings."),
                    ("Knowledge graph embedding", "A new knowledge graph embedding scheme based on translation."),
                ],
            ),
            (
                "ton",
                "IEEE/ACM Transactions on Networking",
                [
                    ("Knowledge graph for network optimization",
                     "Using knowledge graph methods for network optimization."),
                ],
            ),
        ],
    )

    retriever = AcceptedPaperBM25Retriever(
        accepted, journals, paper_count_bonus=0.05, bonus_cap=0.5
    )
    results = retriever.retrieve("knowledge graph reasoning embedding", top_k=10)

    score_by_id = {j.journal_id: s for j, s in results}
    assert "ai" in score_by_id
    assert "ton" in score_by_id

    # 验证 ai 的分数 = max(per-paper) + 0.05 * 3,而非更小或更大
    details = retriever.last_route_details["ai"]
    top_paper_score = details["top_paper_score"]
    paper_count = details["matching_paper_count"]
    expected_unclamped = top_paper_score + 0.05 * paper_count
    bonus_cap = 0.5
    expected = min(expected_unclamped, top_paper_score + bonus_cap)
    assert score_by_id["ai"] == pytest.approx(expected, rel=1e-6)
    assert score_by_id["ai"] > score_by_id["ton"]


def test_bonus_is_capped_to_avoid_runaway(tmp_path):
    """期刊画像里堆 100 篇相似论文也不能让分数无限膨胀;bonus_cap 限制总加成。"""
    from src.retriever.accepted_paper_retriever import AcceptedPaperBM25Retriever

    many_papers = [
        (f"Knowledge graph paper {i}",
         f"Reasoning over knowledge graphs paper {i} with embedding methods.")
        for i in range(100)
    ]
    accepted, journals = _make_stores(
        tmp_path,
        [("ai", "Artificial Intelligence", many_papers)],
    )

    retriever = AcceptedPaperBM25Retriever(
        accepted, journals, paper_count_bonus=0.05, bonus_cap=0.3
    )
    results = retriever.retrieve("knowledge graph reasoning embedding", top_k=5)
    assert len(results) == 1
    journal, score = results[0]

    details = retriever.last_route_details["ai"]
    top_paper_score = details["top_paper_score"]
    # 加成必须 ≤ bonus_cap (0.3),即便 count_bonus*matching_count 远大于 cap
    assert score <= top_paper_score + 0.3 + 1e-6


def test_route_detail_contains_top_paper_title_and_score(tmp_path):
    """route detail 必须记录 top matching paper 的 title 和 score。

    给若干篇无关论文作为背景文档,让 BM25 IDF 正常工作。
    """
    from src.retriever.accepted_paper_retriever import AcceptedPaperBM25Retriever

    accepted, journals = _make_stores(
        tmp_path,
        [
            (
                "ton",
                "IEEE/ACM Transactions on Networking",
                [
                    ("Boring topic on cooking", "Something totally unrelated about cooking recipes and ingredients."),
                    ("TCP congestion control deep dive",
                     "We design a TCP congestion control algorithm that reduces "
                     "tail latency in modern datacenter networks under heavy load."),
                    ("Random survey of pottery",
                     "An overview of pottery techniques from ancient civilizations to modern artisans."),
                ],
            ),
            (
                "ai",
                "Artificial Intelligence",
                [
                    ("Neural networks for image classification",
                     "We train deep convolutional networks on ImageNet for image classification."),
                    ("Reinforcement learning in games",
                     "Reinforcement learning agents that learn to play Atari games from raw pixels."),
                ],
            ),
        ],
    )

    retriever = AcceptedPaperBM25Retriever(accepted, journals)
    retriever.retrieve("TCP congestion control datacenter latency", top_k=3)

    detail = retriever.last_route_details["ton"]
    assert detail["top_paper_title"] == "TCP congestion control deep dive"
    assert isinstance(detail["top_paper_score"], float)
    assert detail["matching_paper_count"] >= 1


def test_top_k_truncates_results(tmp_path):
    from src.retriever.accepted_paper_retriever import AcceptedPaperBM25Retriever

    accepted, journals = _make_stores(
        tmp_path,
        [
            ("a", "Journal A", [("X", "alpha beta gamma delta epsilon")]),
            ("b", "Journal B", [("Y", "alpha beta gamma delta")]),
            ("c", "Journal C", [("Z", "alpha beta gamma")]),
        ],
    )
    retriever = AcceptedPaperBM25Retriever(accepted, journals)
    results = retriever.retrieve("alpha beta", top_k=2)
    assert len(results) == 2


def test_journal_missing_from_store_is_skipped(tmp_path):
    """如果 corpus 里某期刊的 journal_id 在 JournalStore 中找不到 (孤儿 corpus 条目),
    BM25 不应崩溃,也不应该把该 id 列进结果。"""
    from src.retriever.accepted_paper_retriever import AcceptedPaperBM25Retriever

    accepted, journals = _make_stores(
        tmp_path,
        [
            ("ai", "Artificial Intelligence",
             [("Reasoning", "logical reasoning framework")]),
            ("orphan_unknown_id", "Phantom Journal",
             [("Reasoning twin", "logical reasoning twin")]),
        ],
    )

    # 故意从 journal_store 移除 orphan
    journals._by_id.pop("orphan_unknown_id", None)
    journals.journals = [j for j in journals.journals if j.journal_id != "orphan_unknown_id"]

    retriever = AcceptedPaperBM25Retriever(accepted, journals)
    results = retriever.retrieve("logical reasoning", top_k=10)
    journal_ids = {j.journal_id for j, _ in results}
    assert "orphan_unknown_id" not in journal_ids


def test_build_index_auto_called_by_retrieve(tmp_path):
    """与现有 retriever 风格对齐:第一次 retrieve 自动 build_index,
    无需调用方显式调用 build_index()。"""
    from src.retriever.accepted_paper_retriever import AcceptedPaperBM25Retriever

    accepted, journals = _make_stores(
        tmp_path,
        [("ai", "Artificial Intelligence", [("R", "reasoning framework")])],
    )

    retriever = AcceptedPaperBM25Retriever(accepted, journals)
    # 不调 build_index,直接 retrieve
    results = retriever.retrieve("reasoning", top_k=3)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# 任务 3.2:AcceptedPaperEmbeddingRetriever + builder 行为测试
# ---------------------------------------------------------------------------


def _stub_embedding_client(vectors_by_text, default_dim=4):
    """构造一个 stub embedding 客户端,按 text 查表返回固定向量,避免依赖 Ollama。

    若 ``vectors_by_text`` 为空,fall back 到 ``default_dim`` 的零向量。
    """

    class _StubClient:
        def __init__(self):
            self.calls = []

        def _dim(self):
            if vectors_by_text:
                return len(next(iter(vectors_by_text.values())))
            return default_dim

        def embed(self, text):
            self.calls.append(text)
            import numpy as np

            vec = vectors_by_text.get(text)
            if vec is None:
                return np.zeros(self._dim(), dtype=np.float32)
            return np.array(vec, dtype=np.float32)

        def embed_batch(self, texts, concurrency=1, timeout=None):
            return [self.embed(t) for t in texts]

    return _StubClient()


def _build_real_faiss_index(tmp_path, payloads, vectors_by_text):
    """直接调用 builder 脚本,生成真实 FAISS + parquet 文件,供 retriever 测试使用。"""
    from src.journals.accepted_paper_store import AcceptedPaperStore
    from scripts.build_accepted_paper_index import build_accepted_paper_index

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for jid, jname, papers in payloads:
        _write_corpus(
            corpus_dir,
            jid,
            jname,
            [{"title": t, "abstract": a, "year": 2025, "source": "local_evaluation_metadata"}
             for t, a in papers],
        )

    accepted = AcceptedPaperStore(accepted_dir=str(corpus_dir))
    accepted.load()
    client = _stub_embedding_client(vectors_by_text)

    faiss_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "metadata.parquet"

    summary = build_accepted_paper_index(
        accepted_store=accepted,
        embedding_client=client,
        faiss_path=str(faiss_path),
        metadata_path=str(metadata_path),
    )

    return accepted, faiss_path, metadata_path, summary, client


def test_builder_writes_faiss_and_metadata_with_required_fields(tmp_path):
    """plan 明文要求:metadata 必须包含 journal_id / journal_name / title / year / source。"""
    payloads = [
        ("ton", "IEEE/ACM Transactions on Networking",
         [("Net A", "abstract net a"), ("Net B", "abstract net b")]),
        ("ai", "Artificial Intelligence",
         [("AI A", "abstract ai a")]),
    ]
    # 每个 search_text (title + abstract) 给个唯一向量
    vectors = {
        f"{t} {a}": [1.0 if i == idx else 0.0 for i in range(4)]
        for idx, (t, a) in enumerate(
            [("Net A", "abstract net a"),
             ("Net B", "abstract net b"),
             ("AI A", "abstract ai a")]
        )
    }
    # search_text 在 AcceptedPaperRecord 是 "title abstract" 拼接
    accepted, faiss_path, metadata_path, summary, _ = _build_real_faiss_index(
        tmp_path, payloads, vectors
    )

    assert faiss_path.exists()
    assert metadata_path.exists()
    assert summary["vector_count"] == 3
    assert summary["journal_count"] == 2

    import pandas as pd

    df = pd.read_parquet(metadata_path)
    required = {"journal_id", "journal_name", "title", "year", "source"}
    assert required.issubset(set(df.columns))
    assert sorted(df["journal_id"].tolist()) == ["ai", "ton", "ton"]
    assert all(df["source"] == "local_evaluation_metadata")


def test_builder_respects_limit_parameter(tmp_path):
    """--limit 应该截取前 N 条 records,而非全量。"""
    from scripts.build_accepted_paper_index import build_accepted_paper_index
    from src.journals.accepted_paper_store import AcceptedPaperStore

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    _write_corpus(
        corpus_dir, "ai", "Artificial Intelligence",
        [{"title": f"P{i}", "abstract": f"abstract {i}"} for i in range(10)],
    )
    accepted = AcceptedPaperStore(accepted_dir=str(corpus_dir))
    accepted.load()
    client = _stub_embedding_client({})  # 全 0 向量,只测数量

    summary = build_accepted_paper_index(
        accepted_store=accepted,
        embedding_client=client,
        faiss_path=str(tmp_path / "index.faiss"),
        metadata_path=str(tmp_path / "metadata.parquet"),
        limit=3,
    )

    assert summary["vector_count"] == 3


def test_builder_resume_appends_only_new_records(tmp_path):
    """--resume:若已有 FAISS 索引,只 embed 剩余的 records 并追加,不重新算已嵌入的。"""
    from scripts.build_accepted_paper_index import build_accepted_paper_index
    from src.journals.accepted_paper_store import AcceptedPaperStore

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    _write_corpus(
        corpus_dir, "ai", "Artificial Intelligence",
        [{"title": f"P{i}", "abstract": f"abstract {i}"} for i in range(5)],
    )
    accepted = AcceptedPaperStore(accepted_dir=str(corpus_dir))
    accepted.load()
    client = _stub_embedding_client({})  # 零向量

    # 第一次:limit=3 只编码前 3 条
    faiss_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "metadata.parquet"
    build_accepted_paper_index(
        accepted_store=accepted, embedding_client=client,
        faiss_path=str(faiss_path), metadata_path=str(metadata_path),
        limit=3,
    )
    first_call_count = len(client.calls)
    assert first_call_count == 3

    # 第二次:resume=True,应该只编码剩余 2 条(P3、P4)
    summary = build_accepted_paper_index(
        accepted_store=accepted, embedding_client=client,
        faiss_path=str(faiss_path), metadata_path=str(metadata_path),
        resume=True,
    )

    second_call_count = len(client.calls) - first_call_count
    assert second_call_count == 2, (
        f"resume should only embed remaining 2, got {second_call_count}"
    )
    assert summary["vector_count"] == 5  # FAISS 总共 5 条


def test_embedding_retriever_recalls_correct_journal_via_index(tmp_path):
    """端到端:builder 写出来的 FAISS + metadata 必须能被
    AcceptedPaperEmbeddingRetriever 加载并产出正确召回。"""
    from src.retriever.accepted_paper_retriever import AcceptedPaperEmbeddingRetriever

    # 给三篇画像 (ton 1 / ai 1 / ai 1),用不同向量
    payloads = [
        ("ton", "IEEE/ACM Transactions on Networking",
         [("Net Paper", "network protocol research")]),
        ("ai", "Artificial Intelligence",
         [("AI Paper One", "machine learning research")]),
        ("ai_again", "Artificial Intelligence (twin)",
         [("AI Paper Two", "knowledge graph research")]),
    ]
    vectors = {
        "Net Paper network protocol research": [1.0, 0.0, 0.0, 0.0],
        "AI Paper One machine learning research": [0.0, 1.0, 0.0, 0.0],
        "AI Paper Two knowledge graph research": [0.0, 0.0, 1.0, 0.0],
    }
    accepted, faiss_path, metadata_path, _, _ = _build_real_faiss_index(
        tmp_path, payloads, vectors
    )

    journals = _make_journal_store(
        [(jid, jname) for jid, jname, _ in payloads]
    )
    # 查询向量与 ton 那篇相同 → ton 应该排第一
    query_client = _stub_embedding_client(
        {"network protocol query": [1.0, 0.0, 0.0, 0.0]}
    )

    retriever = AcceptedPaperEmbeddingRetriever(
        accepted_store=accepted,
        journal_store=journals,
        embedding_client=query_client,
        faiss_path=str(faiss_path),
        metadata_path=str(metadata_path),
    )
    assert retriever.is_available

    results = retriever.retrieve("network protocol query", top_k=3)
    assert len(results) >= 1
    top_journal, top_score = results[0]
    assert top_journal.journal_id == "ton"


def test_embedding_retriever_returns_empty_when_index_missing(tmp_path):
    """索引缺失时,retriever 应 graceful 处理,不抛异常。"""
    from src.journals.accepted_paper_store import AcceptedPaperStore
    from src.retriever.accepted_paper_retriever import AcceptedPaperEmbeddingRetriever

    accepted = AcceptedPaperStore(accepted_dir=str(tmp_path))
    accepted.load()
    journals = _make_journal_store([])
    client = _stub_embedding_client({})

    retriever = AcceptedPaperEmbeddingRetriever(
        accepted_store=accepted,
        journal_store=journals,
        embedding_client=client,
        faiss_path=str(tmp_path / "nonexistent.faiss"),
        metadata_path=str(tmp_path / "nonexistent.parquet"),
    )
    assert not retriever.is_available
    assert retriever.retrieve("anything", top_k=5) == []


def test_embedding_retriever_aggregates_with_max_plus_count_bonus(tmp_path):
    """与 BM25 retriever 一致的聚合规则:max + min(0.05*count, cap)。"""
    from src.retriever.accepted_paper_retriever import AcceptedPaperEmbeddingRetriever

    payloads = [
        ("ai", "Artificial Intelligence",
         [
             ("Paper One", "knowledge graph one"),
             ("Paper Two", "knowledge graph two"),
             ("Paper Three", "knowledge graph three"),
         ]),
        ("ton", "IEEE/ACM Transactions on Networking",
         [("Single Paper", "network research single")]),
    ]
    # 让 ai 三篇都与查询非常接近,ton 一篇与查询不那么接近
    vectors = {
        "Paper One knowledge graph one": [1.0, 0.0, 0.0],
        "Paper Two knowledge graph two": [0.95, 0.05, 0.0],
        "Paper Three knowledge graph three": [0.9, 0.1, 0.0],
        "Single Paper network research single": [0.6, 0.4, 0.0],
    }
    accepted, faiss_path, metadata_path, _, _ = _build_real_faiss_index(
        tmp_path, payloads, vectors
    )
    journals = _make_journal_store(
        [(jid, jname) for jid, jname, _ in payloads]
    )
    query_client = _stub_embedding_client({"query": [1.0, 0.0, 0.0]})

    retriever = AcceptedPaperEmbeddingRetriever(
        accepted_store=accepted,
        journal_store=journals,
        embedding_client=query_client,
        faiss_path=str(faiss_path),
        metadata_path=str(metadata_path),
        paper_count_bonus=0.05,
        bonus_cap=0.3,
    )
    retriever.retrieve("query", top_k=5)
    detail = retriever.last_route_details["ai"]
    assert detail["matching_paper_count"] >= 2  # 至少 2 篇 ai 论文被命中
    assert detail["bonus"] == pytest.approx(
        min(0.05 * detail["matching_paper_count"], 0.3), rel=1e-6
    )
