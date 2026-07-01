"""典型摘要库加载与轻量检索。"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class TypicalAbstract:
    """一本期刊的一篇生成式语义锚点摘要。"""

    anchor_id: str
    journal_id: str
    journal_name: str
    method_type: str
    novelty_level: str
    abstract: str
    ccf_rating: str = ""

    @property
    def search_text(self) -> str:
        parts = [
            self.journal_name,
            self.method_type,
            self.novelty_level,
            self.abstract,
        ]
        return " ".join(p for p in parts if p)


class TypicalAbstractStore:
    """从 data/typical_abstracts 加载 期刊 -> 多篇典型摘要。"""

    def __init__(self, abstracts_dir: str = "data/typical_abstracts"):
        self.abstracts_dir = abstracts_dir
        self._records: List[TypicalAbstract] = []
        self._by_journal: Dict[str, List[TypicalAbstract]] = {}

    def load(self) -> None:
        """加载目录下所有 JSON 摘要文件。"""
        base = Path(self.abstracts_dir)
        self._records = []
        self._by_journal = {}
        if not base.exists():
            return

        for path in sorted(base.glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            journal_id = data.get("journal_id") or path.stem
            journal_name = data.get("journal_name", journal_id)
            ccf_rating = data.get("ccf_rating", "")
            for idx, item in enumerate(data.get("abstracts", [])):
                abstract = (item.get("abstract") or "").strip()
                if not abstract:
                    continue
                record = TypicalAbstract(
                    anchor_id=f"{journal_id}:{idx}",
                    journal_id=journal_id,
                    journal_name=journal_name,
                    method_type=item.get("method_type", ""),
                    novelty_level=item.get("novelty_level", ""),
                    abstract=abstract,
                    ccf_rating=ccf_rating,
                )
                self._records.append(record)
                self._by_journal.setdefault(journal_id, []).append(record)

    @property
    def records(self) -> List[TypicalAbstract]:
        return self._records

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def journal_count(self) -> int:
        return len(self._by_journal)

    def get_by_journal_id(self, journal_id: str) -> List[TypicalAbstract]:
        return self._by_journal.get(journal_id, [])

    def search_by_text(self, query_text: str, top_k: int = 50) -> List[Tuple[TypicalAbstract, float]]:
        """简单关键词交集搜索，作为第三路 text 召回。"""
        query_terms = _token_set(query_text)
        if not query_terms:
            return []

        scored: List[Tuple[TypicalAbstract, float]] = []
        for record in self._records:
            terms = _token_set(record.search_text)
            if not terms:
                continue
            overlap = query_terms & terms
            if overlap:
                scored.append((record, len(overlap) / max(len(query_terms), 1)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def _token_set(text: str) -> set[str]:
    return {tok.lower() for tok in text.replace("-", " ").split() if len(tok) > 2}


def aggregate_anchor_scores(
    scored_records: Iterable[Tuple[TypicalAbstract, float]],
    mode: str = "max",
) -> Dict[str, float]:
    """把 anchor 级分数聚合到 journal_id。"""
    scores: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for record, score in scored_records:
        jid = record.journal_id
        if mode == "sum":
            scores[jid] = scores.get(jid, 0.0) + score
        elif mode == "mean":
            scores[jid] = scores.get(jid, 0.0) + score
            counts[jid] = counts.get(jid, 0) + 1
        else:
            scores[jid] = max(scores.get(jid, float("-inf")), score)

    if mode == "mean":
        scores = {jid: score / max(counts.get(jid, 1), 1) for jid, score in scores.items()}
    return scores
