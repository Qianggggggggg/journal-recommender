#!/usr/bin/env python3
"""构建 accepted-paper FAISS 向量索引 (任务 3.2)。

把 ``data/accepted_papers/*.json`` 里每条真实发表论文的 ``title + abstract``
用 Ollama embedding 编码后写入 FAISS,配套生成 parquet metadata。

CLI 用法
========

第一次构建::

    python scripts/build_accepted_paper_index.py

中途中断后续传::

    python scripts/build_accepted_paper_index.py --resume

只编码前 N 条 (用于快速验证)::

    python scripts/build_accepted_paper_index.py --limit 20

设计要点
========

- ``--limit N``: 只取前 N 条 records,适合本地快速 sanity check。
- ``--resume``: 检测已有索引中的向量数 K,跳过 records[:K],只 embed 剩余
  records[K:] 并追加到 FAISS。配合断网/限流场景,不需要从头重跑。
- metadata 字段固定: ``journal_id`` / ``journal_name`` / ``title`` /
  ``year`` / ``source``。下游 retriever / feature_builder 都按这套字段读。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.journals.accepted_paper_store import (  # noqa: E402
    AcceptedPaperRecord,
    AcceptedPaperStore,
)
from src.utils.embedding import OllamaEmbedding  # noqa: E402


METADATA_FIELDS = ("journal_id", "journal_name", "title", "year", "source")


def _record_to_metadata_row(record: AcceptedPaperRecord) -> dict[str, Any]:
    return {
        "journal_id": record.journal_id,
        "journal_name": record.journal_name,
        "title": record.title,
        "year": record.year,
        "source": record.source,
    }


def _existing_vector_count(faiss_path: Path) -> int:
    if not faiss_path.exists():
        return 0
    import faiss as faiss_mod

    index = faiss_mod.read_index(str(faiss_path))
    return int(index.ntotal)


def build_accepted_paper_index(
    *,
    accepted_store: AcceptedPaperStore,
    embedding_client: Any,
    faiss_path: str,
    metadata_path: str,
    limit: Optional[int] = None,
    resume: bool = False,
) -> dict:
    """主流程。返回 summary。

    Args:
        accepted_store: 已 ``load()`` 过的 AcceptedPaperStore。
        embedding_client: 任何提供 ``embed_batch(texts)`` 接口的对象,
            通常是 ``OllamaEmbedding``。测试可注入 stub。
        faiss_path: FAISS 索引输出路径。
        metadata_path: parquet metadata 输出路径。
        limit: 可选,只编码前 N 条 records。
        resume: 若 True,跳过前 ``existing_index.ntotal`` 条,只编码剩余,
            然后追加到已有 FAISS 索引。要求已有索引和 metadata 都存在。
    """
    import faiss as faiss_mod

    faiss_p = Path(faiss_path)
    meta_p = Path(metadata_path)
    records = list(accepted_store.records)

    if limit is not None:
        records = records[:limit]

    start_offset = 0
    if resume:
        start_offset = _existing_vector_count(faiss_p)
        if start_offset >= len(records):
            # 已经全部嵌入完,无事可做
            existing_index = faiss_mod.read_index(str(faiss_p))
            existing_meta = pd.read_parquet(meta_p) if meta_p.exists() else pd.DataFrame()
            return {
                "vector_count": int(existing_index.ntotal),
                "journal_count": int(existing_meta["journal_id"].nunique())
                if not existing_meta.empty
                else 0,
                "newly_embedded": 0,
                "resumed_from": start_offset,
                "faiss_path": str(faiss_p),
                "metadata_path": str(meta_p),
            }

    pending = records[start_offset:]
    if not pending and start_offset == 0:
        # 完全没东西可嵌入
        raise RuntimeError("No accepted-paper records to embed")

    texts = [r.search_text for r in pending]
    embeddings = embedding_client.embed_batch(texts) if texts else []
    if embeddings:
        matrix = np.array(embeddings, dtype=np.float32)
    else:
        matrix = np.zeros((0, 0), dtype=np.float32)

    faiss_p.parent.mkdir(parents=True, exist_ok=True)

    if resume and start_offset > 0:
        index = faiss_mod.read_index(str(faiss_p))
        if matrix.shape[0] > 0:
            index.add(matrix)
    else:
        if matrix.shape[0] == 0:
            raise RuntimeError("Embedding client returned no vectors")
        index = faiss_mod.IndexFlatL2(matrix.shape[1])
        index.add(matrix)
    faiss_mod.write_index(index, str(faiss_p))

    # metadata
    new_meta_rows = [_record_to_metadata_row(r) for r in pending]
    if resume and start_offset > 0 and meta_p.exists():
        existing_meta = pd.read_parquet(meta_p)
        meta_df = pd.concat(
            [existing_meta, pd.DataFrame(new_meta_rows)], ignore_index=True
        )
    else:
        # 第一次或非 resume 时,记录与本批 vectors 一一对应的 metadata
        meta_df = pd.DataFrame(new_meta_rows if not resume else (
            [_record_to_metadata_row(r) for r in records[:start_offset]] + new_meta_rows
        ))
    meta_df.to_parquet(meta_p)

    return {
        "vector_count": int(index.ntotal),
        "journal_count": int(meta_df["journal_id"].nunique()) if not meta_df.empty else 0,
        "newly_embedded": len(pending),
        "resumed_from": start_offset,
        "faiss_path": str(faiss_p),
        "metadata_path": str(meta_p),
    }


def _load_config_paths(config_path: str = "configs/app.yaml") -> dict[str, str]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_section = config.get("data", {})
    ollama_section = config.get("ollama", {})
    return {
        "accepted_dir": data_section.get(
            "accepted_papers_dir", "data/accepted_papers"
        ),
        "faiss_path": data_section.get(
            "accepted_papers_faiss_path",
            "data/processed/accepted_papers_index.faiss",
        ),
        "metadata_path": data_section.get(
            "accepted_papers_metadata_path",
            "data/processed/accepted_papers_metadata.parquet",
        ),
        "ollama_base_url": ollama_section.get("base_url", "http://localhost:11434"),
        "ollama_model": ollama_section.get("embedding_model", "qwen3-embedding:4b"),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only embed the first N records (for fast sanity checks).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip records already present in existing FAISS index; append remaining.",
    )
    parser.add_argument(
        "--accepted-dir", default=None,
        help="Override data.accepted_papers_dir from configs/app.yaml.",
    )
    parser.add_argument(
        "--faiss-path", default=None,
        help="Override data.accepted_papers_faiss_path from configs/app.yaml.",
    )
    parser.add_argument(
        "--metadata-path", default=None,
        help="Override data.accepted_papers_metadata_path from configs/app.yaml.",
    )
    parser.add_argument(
        "--config", default="configs/app.yaml",
        help="Path to app config.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    paths = _load_config_paths(args.config)

    accepted_dir = args.accepted_dir or paths["accepted_dir"]
    faiss_path = args.faiss_path or paths["faiss_path"]
    metadata_path = args.metadata_path or paths["metadata_path"]

    store = AcceptedPaperStore(accepted_dir=accepted_dir)
    store.load()
    if store.count == 0:
        print(f"No accepted papers found in {accepted_dir}", file=sys.stderr)
        return 1

    client = OllamaEmbedding(
        base_url=paths["ollama_base_url"],
        model=paths["ollama_model"],
    )

    print(
        f"Building accepted-paper index: {store.count} papers across "
        f"{store.journal_count} journals → {faiss_path}"
    )
    summary = build_accepted_paper_index(
        accepted_store=store,
        embedding_client=client,
        faiss_path=faiss_path,
        metadata_path=metadata_path,
        limit=args.limit,
        resume=args.resume,
    )
    import json as _json

    print(_json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
