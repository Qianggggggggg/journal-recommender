#!/usr/bin/env python3
"""任务 1.2: 为 100 篇论文生成伪标签权重（网格搜索最优三路组合）。"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.journals.journal_store import JournalStore
from src.papers.paper_model import PaperProfile
from src.retriever.candidate_generator import CandidateGenerator
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.embedding_retriever import EmbeddingRetriever
from src.retriever.typical_abstract_retriever import (
    TypicalAbstractBM25Retriever,
    TypicalAbstractEmbeddingRetriever,
    TypicalAbstractTextRetriever,
)
from src.journals.typical_abstract_store import TypicalAbstractStore
from src.retriever.gating_network import best_weight_label


def load_training_data(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_candidate_generator(store_path: str, use_typical: bool = True):
    store = JournalStore(store_path)
    store.load()

    scope_bm25 = BM25Retriever(store)
    scope_bm25.build_index()

    if not use_typical:
        from src.retriever.embedding_retriever import EmbeddingRetriever
        vector = EmbeddingRetriever(store)
        return CandidateGenerator(store, scope_bm25, embedding_retriever=vector)

    abstract_store = TypicalAbstractStore()
    abstract_store.load()

    typical_bm25 = TypicalAbstractBM25Retriever(abstract_store, store)
    typical_bm25.build_index()

    typical_text = TypicalAbstractTextRetriever(abstract_store, store)

    from src.retriever.embedding_retriever import EmbeddingRetriever
    vector = EmbeddingRetriever(store)

    return CandidateGenerator(
        store,
        scope_bm25,
        embedding_retriever=vector,
        retrieval_target="typical_abstracts",
        typical_bm25_retriever=typical_bm25,
        typical_text_retriever=typical_text,
    )


def generate_weight_labels(
    dataset: dict,
    store_path: str,
    output_path: str,
    use_typical: bool = True,
    grid_step: float = 0.1,
):
    print(f"Building candidate generator (typical={use_typical})...")
    generator = build_candidate_generator(store_path, use_typical=use_typical)

    all_labels = {}
    split_results = {}

    for split_name in ["train", "val", "test"]:
        papers = dataset.get(split_name, [])
        if not papers:
            print(f"  {split_name}: no papers, skipping")
            continue

        print(f"  {split_name}: processing {len(papers)} papers...")
        labels = []
        for i, paper in enumerate(papers):
            query = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            profile = PaperProfile(
                title=paper.get("title", ""),
                abstract=paper.get("abstract", ""),
                research_area=paper.get("research_area", []),
            )

            # 三路召回
            bm25_results = generator._active_bm25_retriever().retrieve(query, top_k=50)
            vector_results = generator._active_embedding_retriever().retrieve(query, top_k=50) if generator._active_embedding_retriever() else []
            text_results = generator._text_search(profile, rich_query=query, top_k=50)

            route_rankings = {
                "bm25": [j.journal_id for j, _ in bm25_results],
                "vector": [j.journal_id for j, _ in vector_results],
                "text": [j.journal_id for j, _ in text_results],
            }

            positive_id = paper.get("positive_journal_id", "")
            if not positive_id:
                print(f"    Paper {i}: no positive_journal_id, skipping")
                continue

            weights = best_weight_label(route_rankings, positive_id, grid_step=grid_step)
            labels.append(list(weights))

            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(papers)}...")

        split_results[split_name] = {
            "size": len(labels),
            "avg_weights": np.mean(labels, axis=0).tolist() if labels else [0, 0, 0],
        }
        all_labels[split_name] = labels
        print(f"  {split_name}: avg weights BM25={split_results[split_name]['avg_weights'][0]:.3f}, "
              f"vector={split_results[split_name]['avg_weights'][1]:.3f}, "
              f"text={split_results[split_name]['avg_weights'][2]:.3f}")

    # 保存结果
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result = {
        "labels": all_labels,
        "grid_step": grid_step,
        "split_stats": split_results,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nWeight labels saved to {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-label weights for gating network.")
    parser.add_argument("--training-data", default="data/training/training_pairs.json")
    parser.add_argument("--store", default="data/processed/journals.jsonl")
    parser.add_argument("--output", default="data/training/weight_labels.json")
    parser.add_argument("--use-typical", type=bool, default=True)
    parser.add_argument("--grid-step", type=float, default=0.1)
    args = parser.parse_args()

    dataset = load_training_data(args.training_data)
    generate_weight_labels(
        dataset,
        args.store,
        args.output,
        use_typical=args.use_typical,
        grid_step=args.grid_step,
    )


if __name__ == "__main__":
    main()