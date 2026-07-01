# ADR 0002: LTR v1 训练数据从 ablation JSON 入口(不是 evaluation JSON)

- **状态:** Accepted
- **日期:** 2026-06-02
- **适用范围:** 阶段 4.2 / 5.x(LTR v1 训练入口)
- **关联 plan:** `docs/superpowers/plans/2026-06-01-publication-grade-journal-recommender-plan.md` 4.2 段
- **关联 ADR:** 0001 (coverage-aware Gate A)

## 背景

plan 4.2 原文要求训练数据从 `run_evaluation.py` 输出的 evaluation JSON 导出:

> "输入:包含 `paper_results`、`llm_candidates_detail`、`recommendations_detail`、`venue_diagnostic` 的 evaluation JSON"
> CLI: `python scripts/build_ranking_training_data.py --eval-json <baseline>.json --output <jsonl>`

但 4.1 落地的实现只支持 `--ablation-json`(来自 `run_retrieval_ablation.py`)。
这与 plan 文字不一致,需澄清是设计选择还是遗漏。

## 决策

**LTR v1 训练数据从 `run_retrieval_ablation.py` 的 ablation JSON 入口,不走 `run_evaluation.py` 的 evaluation JSON。**

这是设计选择,不是遗漏。原因:

1. **LTR v1 不学 LLM 介入后的信号**。LTR reranker 的目的是学 per-query 路由权重,
   即给定 paper profile + 候选期刊列表,预测 gold venue。LLM 精排是这条 pipeline
   **之后**的环节,LLM 信心度是 LLM 自己的预测,不应回流到 LTR 的训练目标里。

2. **ablation 路径不含 LLM 噪声**。`run_evaluation.py` 的输出含有 LLM 选出来的
   Top-5 + confidence,这些是 LLM 模型的输出,不是 ground truth。把它当训练标签
   容易让 LTR 学到"跟随 LLM 偏好",失去独立 rerank 的价值。

3. **ablation 路径快、可重复**。89 篇 × 7 variants ≈ 15-30 分钟,
   evaluation 路径每次跑都要 89 次 LLM 调用 ≈ 1-2 小时,且受 LLM 随机性影响。
   训练数据需要"可重放出固定 features"以支持 ablation 与回归。

4. **evaluation JSON 路径延期到 LTR v2(LLM-evidence 版本)**。
   阶段 6 把 LLM 改造成"证据抽取器"后,LTR v2 需要把 LLM evidence
   (scope_fit / method_fit / journal_position_fit)当作额外特征,那时
   evaluation JSON 才有意义。

## 影响

| 阶段 | 入口 | 说明 |
|---|---|---|
| 4.1 feature_builder | n/a | 与入口无关,只关心 trace → features |
| **4.2 训练数据导出(本决策)** | `run_retrieval_ablation.py` | `--ablation-json` 入口(已实现) |
| 4.3 sidecar report | 同上 | 复用同一入口 |
| 5.x LTR 训练 | 同上 | `data/training/ranker_train_<set>.jsonl` |
| 6.x LLM evidence 训练(未来) | `run_evaluation.py` | **届时再实现 `--eval-json` 入口** |

## 拒绝的方案

### 方案 A:实现 `--eval-json` 入口,双轨支持

- 优点:严格按 plan 文字;未来 6.x 不再补代码。
- 拒绝理由:现在 evaluation 路径还没接 `attach_features`,JSON 里没有
  `candidate_features` 字段;要让 evaluation 路径"可用"得先改
  `run_evaluation.py`,而那不是 4.1-4.3 的范围。**双轨让 4.1 测试覆盖到一半的入口,
  调试效率反而降低**。

### 方案 B:不导训练数据,等阶段 6

- 优点:避免改 line 引起的 churn。
- 拒绝理由:LTR 阶段 5 不能在没数据的情况下做。

## 后续动作

1. **plan doc 4.2 段改写**:把"输入 evaluation JSON"替换为"输入 ablation JSON",
   标注延期 evaluation 路径到阶段 6。
2. **build_ranking_training_data.py 注释加 ADR 编号引用**。
3. **不删除** `data/training/` 目录(那是 ablation 路径产物,继续用)。

## 复现

```bash
# 1. 跑 ablation(15-30 分钟)
python scripts/run_retrieval_ablation.py \
  --papers data/evaluation/papers_metadata_v2.jsonl \
  --include-vector \
  --variants scope typical hybrid accepted scope_accepted typical_accepted full_hybrid \
  --output data/evaluation/results/retrieval_ablation_full_v2_$(date +%Y%m%d_%H%M%S).json

# 2. 转训练数据 + sidecar report(秒级)
python scripts/build_ranking_training_data.py \
  --ablation-json data/evaluation/results/retrieval_ablation_full_v2_<ts>.json \
  --journals-jsonl data/journals_ccf.jsonl \
  --output data/training/ranker_train_full_v2.jsonl \
  --variants full_hybrid \
  --max-negatives 10 \
  --report data/training/ranker_train_full_v2_report.json
```
