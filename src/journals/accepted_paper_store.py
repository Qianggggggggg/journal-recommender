"""真实已发表论文画像存储。

每本期刊一个 JSON 文件,文件格式::

    {
      "journal_id": "ton",
      "journal_name": "IEEE/ACM Transactions on Networking",
      "papers": [
        {
          "title": "Paper title",
          "abstract": "Paper abstract",
          "year": 2025,
          "source": "local_evaluation_metadata",
          "doi": "",
          "url": ""
        }
      ]
    }

设计纪律:

- title 或 abstract 为空白的论文整条跳过,不能让单条脏数据让整个加载失败。
- 单个 JSON 文件解析失败时跳过该文件,继续加载其他期刊。
- 缺失目录或缺 ``papers`` 字段时静默处理。

实现风格对齐 ``TypicalAbstractStore``,便于后续 retriever 复用相同心智模型。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class AcceptedPaperRecord:
    """一篇真实已发表论文与其归属期刊的对应记录。"""

    journal_id: str
    journal_name: str
    title: str
    abstract: str
    year: Optional[int] = None
    source: str = ""
    doi: str = ""
    url: str = ""

    @property
    def search_text(self) -> str:
        parts = [self.title, self.abstract]
        return " ".join(p for p in parts if p)


class AcceptedPaperStore:
    """加载 ``data/accepted_papers/*.json``,按期刊管理真实发表论文。"""

    def __init__(self, accepted_dir: str = "data/accepted_papers"):
        self.accepted_dir = accepted_dir
        self._records: List[AcceptedPaperRecord] = []
        self._by_journal: Dict[str, List[Dict[str, Any]]] = {}
        self._journal_names: Dict[str, str] = {}

    def load(self) -> None:
        """重新加载目录下所有期刊 JSON。

        目录不存在 / 文件 JSON 损坏 / papers 字段缺失 都不会抛异常,
        只是结果集中相应条目变空。
        """
        self._records = []
        self._by_journal = {}
        self._journal_names = {}

        base = Path(self.accepted_dir)
        if not base.exists():
            return

        for path in sorted(base.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                # 单个坏文件不应让整个加载失败
                continue

            if not isinstance(data, dict):
                continue

            journal_id = (data.get("journal_id") or path.stem).strip()
            if not journal_id:
                continue
            journal_name = data.get("journal_name") or journal_id

            raw_papers = data.get("papers")
            if not isinstance(raw_papers, list):
                continue

            cleaned_papers: List[Dict[str, Any]] = []
            for raw in raw_papers:
                paper = _normalize_paper(raw)
                if paper is None:
                    continue
                cleaned_papers.append(paper)
                self._records.append(
                    AcceptedPaperRecord(
                        journal_id=journal_id,
                        journal_name=journal_name,
                        title=paper["title"],
                        abstract=paper["abstract"],
                        year=paper["year"],
                        source=paper["source"],
                        doi=paper["doi"],
                        url=paper["url"],
                    )
                )

            if cleaned_papers:
                # 与 TypicalAbstractStore 一致:只有当真的有论文时才计入 journal_count
                self._by_journal[journal_id] = cleaned_papers
                self._journal_names[journal_id] = journal_name

    # ---- 查询接口 ----

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def journal_count(self) -> int:
        return len(self._by_journal)

    @property
    def journal_ids(self) -> List[str]:
        """所有至少有一篇 paper 的期刊 id 列表(顺序与 load 顺序一致)。"""
        return list(self._by_journal.keys())

    @property
    def records(self) -> List[AcceptedPaperRecord]:
        """所有真实论文记录,顺序与 iter_records() 一致。"""
        return self._records

    def get_papers(self, journal_id: str) -> List[Dict[str, Any]]:
        return self._by_journal.get(journal_id, [])

    def iter_records(self) -> Iterable[AcceptedPaperRecord]:
        return iter(self._records)

    def journal_name(self, journal_id: str) -> str:
        return self._journal_names.get(journal_id, journal_id)


def _normalize_paper(raw: Any) -> Optional[Dict[str, Any]]:
    """把原始 dict 规范成稳定字段集合;title/abstract 缺或全空白的返回 None。"""
    if not isinstance(raw, dict):
        return None

    title = (raw.get("title") or "").strip()
    abstract = (raw.get("abstract") or "").strip()
    if not title or not abstract:
        return None

    year_val = raw.get("year")
    year: Optional[int]
    if isinstance(year_val, int):
        year = year_val
    elif isinstance(year_val, str) and year_val.strip().isdigit():
        year = int(year_val.strip())
    else:
        year = None

    return {
        "title": title,
        "abstract": abstract,
        "year": year,
        "source": (raw.get("source") or "").strip(),
        "doi": (raw.get("doi") or "").strip(),
        "url": (raw.get("url") or "").strip(),
    }
