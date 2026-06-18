#!/usr/bin/env python3
"""Train a baseline learning-to-rank reranker (Task 5.2).

输入:``data/training/ranker_train.jsonl``(由 ``build_ranking_training_data.py`` 产出,
每行一个 paper-candidate pair,含 ``features`` / ``label`` / ``feature_names``)。
输出:``data/models/learning_to_ranker.json``,包含::

    {
      "schema_version": 1,
      "model_type": "logistic_regression",
      "feature_names": [...],
      "coef": [...],
      "intercept": float,
      "use_standardization": bool,
      "scaler_mean": [...] | null,
      "scaler_scale": [...] | null,
      "metrics": {
        "n_train": int, "n_positive": int, "n_negative": int,
        "pairwise_accuracy": float,
        "positive_mean_score": float,
        "hard_negative_mean_score": float,
      },
      "convergence_info": {
        "converged": bool, "n_iter": int | None,
        "max_iter": int, "warning_message": str | None,
      },
      "seed": int,
    }

加载方式: ``LearningToRanker.load(output_path)`` 即可获得可推理的 ranker。

plan 5.2 重点 — 真实 ranker_train_full_v2.jsonl 上 lbfgs 在默认 1000 iter 内
无法收敛(0/1 布尔特征 vs 0~999 rank 特征尺度差 3 个数量级),脚本必须:
1. 默认开 ``--use-standardization``(否则 convergence 必报 warning)。
2. 默认 ``--max-iter=5000``(给真实数据足够收敛空间)。
3. 收敛状态写到 ``convergence_info`` 而不是静默吞。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# 让脚本能 ``import src.*`` (per repo 惯例)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ranker.learning_to_rank import LearningToRanker  # noqa: E402
from src.ranker.feature_builder import LLM_EVIDENCE_FEATURE_NAMES  # noqa: E402


SCHEMA_VERSION = 1


def filter_llm_evidence_features(
    rows: list[dict], feature_names: list[str]
) -> tuple[list[dict], list[str]]:
    """Drop the 6 LLM evidence features from each row.

    Returns (filtered_rows, filtered_feature_names). The remaining 20 features
    are pure rule/retrieval signals — useful when we want the LTR to focus on
    structural priors and let ``LLMEvidenceRoleRanker`` own the LLM evidence.
    """
    keep_idx = [
        i for i, name in enumerate(feature_names) if name not in LLM_EVIDENCE_FEATURE_NAMES
    ]
    dropped = [name for name in feature_names if name in LLM_EVIDENCE_FEATURE_NAMES]
    if len(dropped) != len(LLM_EVIDENCE_FEATURE_NAMES):
        missing = set(LLM_EVIDENCE_FEATURE_NAMES) - set(dropped)
        raise ValueError(
            f"--exclude-llm-evidence: only found {len(dropped)}/6 LLM features in "
            f"feature_names. Missing: {sorted(missing)}"
        )
    print(
        f"[exclude-llm-evidence] dropping {len(dropped)} features: {dropped}",
        flush=True,
    )
    filtered_names = [feature_names[i] for i in keep_idx]
    filtered_rows = [
        {**r, "features": [r["features"][i] for i in keep_idx], "feature_names": filtered_names}
        for r in rows
    ]
    return filtered_rows, filtered_names


def compute_pairwise_accuracy(rows: list[dict], scores: list[float]) -> float:
    """同 paper_id 内,所有 (positive, hard_negative) 对中 positive 分数更高的比例。

    只用 ``negative_type == "hard_rule_top20"`` 的负样本做比较(easy 负样本
    太简单不算难例)。这是 LTR baseline 的标准 pairwise 指标。
    """
    groups: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for r, s in zip(rows, scores):
        groups[r["paper_id"]].append((r["label"], s, r.get("negative_type", "")))
    correct, total = 0, 0
    for group in groups.values():
        positives = [s for lab, s, _t in group if lab == 1]
        hard_negs = [s for lab, _s, t in group if lab == 0 and t == "hard_rule_top20"]
        if not positives or not hard_negs:
            continue
        for ps in positives:
            for ns in hard_negs:
                total += 1
                if ps > ns:
                    correct += 1
    return correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="训练 JSONL 路径 (build_ranking_training_data.py 产出)")
    parser.add_argument("--output", required=True, help="输出模型 JSON 路径,例如 data/models/learning_to_ranker.json")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (per plan 5.2 纪律:确定性)")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="LogisticRegression max_iter (per plan 5.2 重点:真实数据需要 ≥ 5000)",
    )
    parser.add_argument(
        "--use-standardization",
        dest="use_standardization",
        action="store_true",
        default=True,
        help="对训练特征做 z-score 标准化(plan 5.2 默认 True,真实数据必开)",
    )
    parser.add_argument(
        "--no-standardization",
        dest="use_standardization",
        action="store_false",
        help="关闭标准化,等价 plan 5.1 行为,仅供回归",
    )
    parser.add_argument(
        "--exclude-llm-evidence",
        dest="exclude_llm_evidence",
        action="store_true",
        default=False,
        help=(
            "训练时丢掉 6 个 LLM evidence 特征 (llm_scope_fit, llm_method_fit, "
            "llm_application_fit, llm_journal_position_fit, llm_too_broad_penalty, "
            "llm_too_narrow_penalty),只留 20 个 rule/retrieval 特征。"
            "配合 LLMEvidenceRoleRanker 的 evidence_weight 让 LTR 学结构先验、"
            "让 role ranker 独占 LLM 证据,避免两路重复消费 LLM 信号。"
        ),
    )
    parser.add_argument(
        "--model-type",
        dest="model_type",
        default="logistic",
        choices=["logistic", "lightgbm_lambdarank"],
        help=(
            "选择 backend (Task 5.5 引入):"
            "- 'logistic' (默认): sklearn LR 或 numpy fallback,行为与 5.1 一致"
            "- 'lightgbm_lambdarank': LightGBM LambdaRank,需要 lightgbm,"
            "rows 必须按 paper_id 连续(已由 build_ranking_training_data.py 保证)"
        ),
    )
    args = parser.parse_args()

    train_path = Path(args.train)
    if not train_path.exists():
        print(f"[error] training file not found: {train_path}", file=sys.stderr)
        sys.exit(2)

    rows: list[dict] = []
    with train_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        print(f"[error] training file is empty: {train_path}", file=sys.stderr)
        sys.exit(2)

    # 取 feature_names(所有行必须一致,这里做防御性检查)
    feature_names = rows[0].get("feature_names") or []
    feature_dim = len(rows[0]["features"])
    inconsistent = [r for r in rows[1:] if len(r["features"]) != feature_dim]
    if inconsistent:
        print(
            f"[error] {len(inconsistent)} rows have mismatched feature length (expected {feature_dim})",
            file=sys.stderr,
        )
        sys.exit(2)

    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = sum(1 for r in rows if r["label"] == 0)
    print(f"Loaded {len(rows)} rows from {train_path} (pos={n_pos}, neg={n_neg}, feature_dim={feature_dim})")

    # 可选:丢掉 6 个 LLM evidence 特征,只留 20 个 rule/retrieval 特征
    if args.exclude_llm_evidence:
        rows, feature_names = filter_llm_evidence_features(rows, feature_names)
        feature_dim = len(feature_names)
        # 二次防御性检查
        bad = [r for r in rows if len(r["features"]) != feature_dim]
        if bad:
            print(
                f"[error] {len(bad)} rows have mismatched feature length after filter (expected {feature_dim})",
                file=sys.stderr,
            )
            sys.exit(2)
        print(
            f"[exclude-llm-evidence] post-filter feature_dim={feature_dim}, "
            f"feature_names={feature_names}",
            flush=True,
        )

    ranker = LearningToRanker(
        seed=args.seed,
        max_iter=args.max_iter,
        use_standardization=args.use_standardization,
        model_type=args.model_type,
    )
    ranker.fit(rows)
    scores = ranker.predict_scores(rows)

    pos_scores = [s for r, s in zip(rows, scores) if r["label"] == 1]
    hard_neg_scores = [s for r, s, t in zip(rows, scores, [r.get("negative_type", "") for r in rows]) if r["label"] == 0 and t == "hard_rule_top20"]
    positive_mean = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0
    hard_neg_mean = sum(hard_neg_scores) / len(hard_neg_scores) if hard_neg_scores else 0.0
    pairwise_acc = compute_pairwise_accuracy(rows, scores)

    metrics = {
        "n_train": len(rows),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "pairwise_accuracy": pairwise_acc,
        "positive_mean_score": positive_mean,
        "hard_negative_mean_score": hard_neg_mean,
    }

    # 拿 coef / intercept (logistic only; lightgbm 走 booster_str 分支)
    if ranker._backend == "lightgbm":
        coef: list | None = None
        intercept: float | None = None
    elif ranker._backend == "sklearn":
        coef = ranker._model.coef_[0].tolist()
        intercept = float(ranker._model.intercept_[0])
    else:
        coef = ranker._model.weights.tolist()
        intercept = float(ranker._model.bias)

    scaler_mean, scaler_scale = None, None
    if ranker._scaler is not None:
        scaler_mean = ranker._scaler.mean.tolist()
        scaler_scale = ranker._scaler.scale.tolist()

    # 早声明 out_path,lightgbm 分支用 ranker.save() 写文件需要它
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 构造 payload: lightgbm 走 ranker.save() 的统一路径(它会写 booster_str)
    if ranker._backend == "lightgbm":
        # 让 ranker.save() 统一写 JSON(包括 booster_str),保持与 LTRAdapter.load 一致
        ranker.save(str(out_path))
        print(f"Wrote LightGBM model to {out_path}")
        print(f"  backend={ranker._backend}, use_standardization={args.use_standardization}, max_iter={args.max_iter}")
        print(f"  convergence_info: {ranker.convergence_info}")
        print(f"  metrics:")
        print(f"    pairwise_accuracy      = {pairwise_acc:.4f}")
        print(f"    positive_mean_score    = {positive_mean:.4f}")
        print(f"    hard_negative_mean     = {hard_neg_mean:.4f}")
        print(f"    margin (pos - hard_neg) = {positive_mean - hard_neg_mean:+.4f}")
        return

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_type": "logistic_regression",
        "backend": ranker._backend,
        "feature_names": feature_names,
        "feature_dim": feature_dim,
        "coef": coef,
        "intercept": intercept,
        "use_standardization": args.use_standardization,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "metrics": metrics,
        "convergence_info": ranker.convergence_info,
        "seed": args.seed,
        "max_iter": args.max_iter,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 报告
    print(f"Wrote model to {out_path}")
    print(f"  backend={ranker._backend}, use_standardization={args.use_standardization}, max_iter={args.max_iter}")
    print(f"  convergence_info: {ranker.convergence_info}")
    print(f"  metrics:")
    print(f"    pairwise_accuracy      = {pairwise_acc:.4f}")
    print(f"    positive_mean_score    = {positive_mean:.4f}")
    print(f"    hard_negative_mean     = {hard_neg_mean:.4f}")
    print(f"    margin (pos - hard_neg) = {positive_mean - hard_neg_mean:+.4f}")


if __name__ == "__main__":
    main()
