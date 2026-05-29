#!/usr/bin/env python3
"""抽样对比 scope_text 召回与典型摘要召回。"""
import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.journals.journal_store import JournalStore
from src.journals.typical_abstract_store import TypicalAbstractStore
from src.papers.paper_model import PaperProfile
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.candidate_generator import CandidateGenerator
from src.retriever.typical_abstract_retriever import (
    TypicalAbstractBM25Retriever,
    TypicalAbstractTextRetriever,
)


def load_papers(path: str, n: int) -> list[dict]:
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
            if len(papers) >= n:
                break
    return papers


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare typical-abstract recall with scope recall.")
    parser.add_argument("--papers", default="data/evaluation/papers_metadata.jsonl")
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    store = JournalStore()
    store.load()
    scope_bm25 = BM25Retriever(store)
    scope_bm25.build_index()
    scope_generator = CandidateGenerator(store, scope_bm25, retrieval_target="scope_text")

    abstract_store = TypicalAbstractStore()
    abstract_store.load()
    typical_bm25 = TypicalAbstractBM25Retriever(abstract_store, store)
    typical_text = TypicalAbstractTextRetriever(abstract_store, store)
    typical_generator = CandidateGenerator(
        store,
        scope_bm25,
        retrieval_target="typical_abstracts",
        typical_bm25_retriever=typical_bm25,
        typical_text_retriever=typical_text,
    )

    papers = load_papers(args.papers, args.sample)
    for paper in papers:
        profile = PaperProfile(
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
            research_area=paper.get("research_area", []),
        )
        query = f"{profile.title} {profile.abstract}"
        scope = scope_generator.generate(query, profile, top_k=args.top_k)
        typical = typical_generator.generate(query, profile, top_k=args.top_k)
        venue = paper.get("venue", "")
        print("\n==", profile.title[:90])
        print("venue:", venue)
        print("scope hit:", venue.lower() in [j.journal_name.lower() for j in scope])
        print("typical hit:", venue.lower() in [j.journal_name.lower() for j in typical])
        print("scope top:", [j.journal_name for j in scope[:5]])
        print("typical top:", [j.journal_name for j in typical[:5]])


if __name__ == "__main__":
    main()
