#!/usr/bin/env python3
"""任务 0.3：整理标注论文-期刊对并构造负样本。"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.journals.journal_store import JournalStore
from src.retriever.bm25_retriever import BM25Retriever


def load_papers(path: str, limit: int | None = 100) -> List[dict]:
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            if paper.get("title") and paper.get("venue"):
                papers.append(paper)
            if limit and len(papers) >= limit:
                break
    return papers


def stratified_split(
    papers: List[dict],
    train_size: int = 60,
    val_size: int = 20,
    test_size: int = 20,
    seed: int = 42,
) -> Dict[str, List[dict]]:
    """按 research_area 分层切分，保证 60/20/20。"""
    rng = random.Random(seed)
    by_area: Dict[str, List[dict]] = defaultdict(list)
    for paper in papers:
        area = _first_area(paper)
        by_area[area].append(paper)

    for bucket in by_area.values():
        rng.shuffle(bucket)

    splits = {"train": [], "val": [], "test": []}
    quotas = {"train": train_size, "val": val_size, "test": test_size}
    order = ["train", "val", "test"]

    # 轮转分配，先保留领域分布，再用 quota 截断/补齐。
    for _, bucket in sorted(by_area.items()):
        for idx, paper in enumerate(bucket):
            split = order[idx % len(order)]
            splits[split].append(paper)

    all_papers = [p for bucket in by_area.values() for p in bucket]
    rng.shuffle(all_papers)
    assigned = set()
    final = {"train": [], "val": [], "test": []}

    for split in order:
        for paper in splits[split]:
            if len(final[split]) < quotas[split] and id(paper) not in assigned:
                final[split].append(paper)
                assigned.add(id(paper))

    for split in order:
        for paper in all_papers:
            if len(final[split]) >= quotas[split]:
                break
            if id(paper) not in assigned:
                final[split].append(paper)
                assigned.add(id(paper))

    return final


def attach_bm25_negatives(
    splits: Dict[str, List[dict]],
    store_path: str,
    negatives_per_paper: int = 50,
) -> Dict[str, List[dict]]:
    store = JournalStore(store_path)
    store.load()
    retriever = BM25Retriever(store)
    retriever.build_index()

    venue_to_id = {j.journal_name.strip().lower(): j.journal_id for j in store.journals}
    enriched = {}
    for split, papers in splits.items():
        enriched[split] = []
        for paper in papers:
            query = " ".join([paper.get("title", ""), paper.get("abstract", "")])
            positive_id = venue_to_id.get(paper.get("venue", "").strip().lower(), "")
            candidates = retriever.retrieve(query, top_k=max(negatives_per_paper + 5, negatives_per_paper))
            negatives = []
            for journal, score in candidates:
                if journal.journal_id == positive_id:
                    continue
                negatives.append({
                    "journal_id": journal.journal_id,
                    "journal_name": journal.journal_name,
                    "bm25_score": score,
                })
                if len(negatives) >= negatives_per_paper:
                    break

            enriched_paper = {
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "venue": paper.get("venue", ""),
                "positive_journal_id": positive_id,
                "ccf_level": paper.get("ccf_level", ""),
                "research_area": paper.get("research_area", []),
                "negative_journals": negatives,
            }
            enriched[split].append(enriched_paper)
    return enriched


def save_dataset(dataset: Dict[str, List[dict]], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


def _first_area(paper: dict) -> str:
    area = paper.get("research_area", "")
    if isinstance(area, list):
        return area[0] if area else ""
    return area or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 60/20/20 training data with BM25 negatives.")
    parser.add_argument("--input", default="data/evaluation/papers_metadata.jsonl")
    parser.add_argument("--store", default="data/processed/journals.jsonl")
    parser.add_argument("--output", default="data/training/training_pairs.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    papers = load_papers(args.input, limit=args.limit)
    if len(papers) < 100:
        print(f"Warning: only {len(papers)} papers loaded; expected 100.")
    splits = stratified_split(papers, seed=args.seed)
    dataset = attach_bm25_negatives(splits, args.store)
    save_dataset(dataset, args.output)
    print(f"Saved training data to {args.output}")
    print({split: len(items) for split, items in dataset.items()})


if __name__ == "__main__":
    main()
