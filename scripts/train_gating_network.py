#!/usr/bin/env python3
"""任务 1.2：生成伪标签并训练动态门控网络。"""
import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.papers.paper_model import PaperProfile
from src.ranker.gating_network import best_weight_label, train_gating_network


def _profile_from_paper(paper: dict) -> PaperProfile:
    return PaperProfile(
        title=paper.get("title", ""),
        abstract=paper.get("abstract", ""),
        research_area=paper.get("research_area", []),
        keywords=paper.get("keywords", []),
        techniques=paper.get("techniques", []),
    )


def _route_rankings_from_paper(paper: dict) -> dict[str, list[str]]:
    """从训练数据的负样本构造可复现伪路由。

    完整实验可替换为真实 BM25/vector/text 三路召回结果；这里保留统一接口。
    """
    positive = paper.get("positive_journal_id", "")
    negatives = [n["journal_id"] for n in paper.get("negative_journals", [])]
    bm25 = [positive] + negatives if positive else negatives
    # 没有预计算 vector/text 时，用不同 hard-negative 顺序生成弱伪路由，训练脚本仍可跑通。
    return {
        "bm25": bm25,
        "vector": negatives[::2] + ([positive] if positive else []) + negatives[1::2],
        "text": negatives[-10:] + ([positive] if positive else []) + negatives[:-10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dynamic retrieval gating network.")
    parser.add_argument("--input", default="data/training/training_pairs.json")
    parser.add_argument("--output", default="data/models/gating_network.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    papers = dataset.get("train", [])
    profiles = []
    labels = []
    for paper in papers:
        positive = paper.get("positive_journal_id", "")
        if not positive:
            continue
        profiles.append(_profile_from_paper(paper))
        labels.append(best_weight_label(_route_rankings_from_paper(paper), positive))

    if not profiles:
        raise RuntimeError("No train profiles with positive_journal_id found.")

    gater = train_gating_network(profiles, labels, epochs=args.epochs, lr=args.lr)
    gater.save(args.output)
    print(f"Saved gating network to {args.output}")


if __name__ == "__main__":
    main()
