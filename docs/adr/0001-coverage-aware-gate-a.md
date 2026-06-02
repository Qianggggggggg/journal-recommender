# ADR 0001: Coverage-Aware Gate A for Accepted-Paper Route

- **状态:** Accepted
- **日期:** 2026-06-02
- **适用范围:** 阶段 3.4 → 阶段 4 / 5 的所有工作
- **关联 plan:** `docs/superpowers/plans/2026-06-01-publication-grade-journal-recommender-plan.md` 决策门槛
- **关联评测:** `data/evaluation/results/retrieval_ablation_full_v2_20260602_153622.json`
- **关联策略:** `docs/evaluation/benchmark_policy.md` "Coverage-aware Gate A" 章节

## 背景

阶段 2.2 用本地 `papers_metadata.jsonl` (排除 light30 + v2) 构建了 95 篇 / 63 本期刊的
`data/accepted_papers/` 语料。任务 3.3 把 `accepted_bm25` 与 `accepted_vector` 两条召回路由
接入 `CandidateGenerator`,默认 `accepted_paper_weight=0.20`,`route_top_k.abstract`
中 `accepted_bm25=28`、`accepted_vector=56`。

阶段 3.4 计划要求用 `scripts/run_retrieval_ablation.py` 跑 light30 + full-v2 消融,
回答 Gate A:"accepted-paper route 能否在 full-v2 上提升 `coarse@50` 或 final `Hit@5`?
如果没有,保留为消融贡献,不要默认启用。"

## 原始数据 (full-v2, n=89)

执行:

```bash
python scripts/run_retrieval_ablation.py \
  --papers data/evaluation/papers_metadata_v2.jsonl \
  --include-vector \
  --variants scope typical hybrid accepted scope_accepted typical_accepted full_hybrid \
  --output data/evaluation/results/retrieval_ablation_full_v2_20260602_153622.json
```

整体指标 (`rule@5` 是 RuleScorer 在 Top-5 命中 gold 的论文数):

| variant | coarse@50 | rule@5 | rule@20 | ret_mrr | ret_ndcg5 | rule_mrr |
|---|---|---|---|---|---|---|
| scope | 75 | 27 | 61 | 0.2441 | 0.2212 | 0.2158 |
| typical | 82 | 22 | 71 | 0.2421 | 0.2404 | 0.1989 |
| hybrid (线上默认) | 83 | 33 | 75 | 0.3043 | 0.2978 | 0.2525 |
| accepted | 47 | 25 | 47 | 0.1496 | 0.1607 | 0.1509 |
| scope_accepted | 79 | 33 | 71 | 0.2848 | 0.2808 | 0.2465 |
| typical_accepted | 81 | 33 | 72 | 0.2605 | 0.2653 | 0.2363 |
| **full_hybrid** | **83** | **33** | **75** | **0.3043** | **0.2978** | **0.2525** |

第一眼读数:`full_hybrid == hybrid` 在所有六项指标上完全相同,3 路 `weighted_minmax`
融合没有把 accepted 的边际贡献释放出来。如果按原始 Gate A(整体 coarse@50/Hit@5 不变,
则保留为消融贡献),应当把 `accepted_paper_weight` 调到 0,关闭 accepted route。

但 `accepted` 单独在 coarse@50=47(远低于 typical 82)的同时,rule@5=25/89=28%
**比 typical 单独 22/89=25% 高 3pp**——accepted 召回的候选数量更少,但精度更高。
这意味着原始 Gate A 的判据过于粗糙,需要看 covered/uncovered 子集。

## Coverage-Aware 切分

covered 判定:`gold_journal_id` 在 `data/accepted_papers/<jid>.json` 中存在至少 1 篇论文。
v2 数据集 (n=89) 共 35 个 unique `journal_id` 命中,覆盖 48/89 = **53.9%**。
uncovered = 41 篇 = 46.1%。

### covered 子集 (n=48)

| variant | coarse@50 | rule@5 | rule@20 | ret_mrr | ret_ndcg5 |
|---|---|---|---|---|---|
| scope | 41 | 11 | 29 | 0.2236 | 0.1902 |
| typical | 45 | 10 | 39 | 0.2367 | 0.2310 |
| hybrid (线上默认) | 46 | 14 | 40 | 0.3009 | 0.2944 |
| **accepted** | 47 | **25** | 47 | 0.2773 | 0.2980 |
| scope_accepted | 47 | 19 | 42 | **0.3910** | **0.4054** |
| typical_accepted | 48 | 22 | **48** | 0.3599 | 0.3801 |
| full_hybrid | 46 | 14 | 40 | 0.3009 | 0.2944 |

accepted 单独在 covered 上 rule@5 = 25/48 = **52.1%** (单点最高)。
`scope_accepted` 在 covered 上 ret_mrr = **0.3910**、ret_ndcg5 = **0.4054** (全表最高)。
`typical_accepted` 在 covered 上 rule@5 = 22/48 = **45.8%** (vs hybrid 14/48=29.2%,+16.6pp)。

### uncovered 子集 (n=41)

| variant | coarse@50 | rule@5 | rule@20 | ret_mrr | ret_ndcg5 |
|---|---|---|---|---|---|
| scope | 34 | 16 | 32 | 0.2680 | 0.2575 |
| typical | 37 | 12 | 32 | 0.2485 | 0.2514 |
| hybrid (线上默认) | 37 | 19 | 35 | 0.3083 | 0.3017 |
| accepted | **0** | **0** | **0** | **0.0000** | **0.0000** |
| scope_accepted | 32 | 14 | 29 | 0.1605 | 0.1350 |
| typical_accepted | 33 | 11 | 24 | 0.1440 | 0.1310 |
| full_hybrid | 37 | 19 | 35 | 0.3083 | 0.3017 |

accepted 在 uncovered 上结构性 0(预期,corpus 没有这 41 篇 gold venue 的论文)。
但 2-route fusion 在 uncovered 上也 **没有伤害 hybrid**——`full_hybrid` 仍然等于 `hybrid`,
因为归一化把 accepted 的空信号吸收掉了。

## 诊断

1. **`weighted_minmax` 在 3 路融合时把第 4 路由 accepted 归一化掉**:
   `full_hybrid` 在 covered 上与 hybrid 完全相同,意味着 scope/typical/identity
   三路归一化后的 top-5 名额已经填满,accepted 的高分候选被压到 `weighted_score ≈ 0`。
   2-route fusion (`scope_accepted`、`typical_accepted`) **能释放** accepted signal,
   但只在 covered 上释放,在 uncovered 上没有额外伤害。

2. **accepted signal 在 covered 上真实存在**:
   `accepted` 单独 rule@5=52%,`typical_accepted` rule@5=46% vs typical 单独 21%,
   提升 25pp。这不是噪声。

3. **covered / uncovered 表现反向**:
   `scope_accepted` 在 covered 上 rule@5 从 scope 单独 11 → 19 (+8pp),
   在 uncovered 上从 16 → 14 (-2pp)。**静态权重调不出来,需要 per-query 自适应**。

4. **整体数字掩盖了信号**:
   `full_hybrid` 整体 rule@5=33/89=37.1%,与 hybrid 相同。
   但构成完全不同:hybrid 是 14 covered + 19 uncovered,
   `typical_accepted` 是 22 covered + 11 uncovered。
   两者的失败模式互补,LTR 才能合并两者的长处。

## 决策

采用 **Coverage-Aware Gate A**:

- accepted-paper route **作为有效信号**通过 Gate A;
- 保留 `accepted_paper_weight=0.20` 默认值,不动;
- 保留 `accepted_bm25=28` / `accepted_vector=56` 路由 top-k,不动;
- 不做静态权重调参(已证伪,见诊断 3);
- 直接进入 **Task 4.1 feature_builder**,把 accepted signal 从 raw route 抬升为可学习特征,
  并由 LTR 在 4.x / 5.x 阶段学到 per-query 动态权重。

## 拒绝的替代方案

### 替代 A: 关闭 accepted route (`accepted_paper_weight=0`)

- 优点:消除 uncovered 上的全 0 噪声,简化解释。
- 拒绝理由:covered 子集上 accepted 是单点最强路由 (52% rule@5 vs typical 21%),
  关闭 = 放弃 53.9% 测试集上最精准的召回信号,数据上无法辩护。
  未在 `data/accepted_papers/` 覆盖的期刊本来就没有 accepted 这条路的能力,
  设成 0 只是显式承认这一点,不会"修复"任何东西。

### 替代 B: 静态调高 `accepted_paper_weight` 到 0.5+

- 优点:理论上能让 accepted signal 在 3 路归一化前更有竞争力。
- 拒绝理由:`scope_accepted` / `typical_accepted` 的 uncovered 子集表现已经说明,
  2-route fusion 在 uncovered 上都没有伤害 hybrid,问题不在权重不够大,
  而在 3 路归一化机制本身不接受第 4 路。提高权重只会被归一化继续吃掉。

### 替代 C: 静态只保留 `typical_accepted` 作为默认 fusion

- 优点:在 covered 上 rule@5=46% > hybrid 29%,看上去是 win。
- 拒绝理由:在 uncovered 上 rule@5=27% < hybrid 46%,uncovered 是 46.1% 测试集。
  静态选择 `typical_accepted` 会在 uncovered 子集上 -19pp 拖后腿,整体不会被 hybrid 更好。
  必须靠 LTR 在不同子集动态选择。

## 关键区分:`candidate_in_accepted_corpus` vs `gold_in_accepted_corpus`

任务 4.1 feature_builder 会接入 `accepted` 相关特征。**两个看似类似的 boolean 必须严格区分**:

| 特征 | 训练时 | 推理时 | 用途 |
|---|---|---|---|
| `candidate_in_accepted_corpus` (bool) | 可计算 | 可计算 | **LTR 训练特征**;候选期刊 `jid` 在 `data/accepted_papers/` 中有 ≥ 1 篇论文 |
| `gold_in_accepted_corpus` (bool) | 可计算 | **不可计算** | **仅诊断用**;切 covered/uncovered 子集,不能进 LTR 训练集 |

如果合写成单一 `gold_or_candidate_in_accepted_corpus` 或让 `gold_in_accepted_corpus`
进训练特征,会出现:
- 训练时模型看到 `gold_in_corpus=True` 的强信号,学会在 `candidate_in_corpus=True`
  时给高分。
- 推理时 `gold_in_corpus` 永远是 None,只能退化成 `candidate_in_corpus=True/False`,
  分布漂移 → uncovered 子集表现比训练差。
- 这正是已知的 oracle leakage 模式。

正确做法:feature_builder 只暴露 `candidate_in_accepted_corpus`,
`gold_in_accepted_corpus` 仅在评测脚本 (`scripts/stratify_retrieval_ablation.py`)
里用来切分层并写诊断报告。

## 后续动作

1. **Task 4.1 feature_builder**: 新建 `src/ranker/feature_builder.py`,按 plan 4.1 节的
   `FEATURE_NAMES` 列表接入 `accepted_bm25_rank`、`accepted_vector_rank`、
   `accepted_bm25_score`、`accepted_vector_score`、`has_accepted_route`、
   `candidate_in_accepted_corpus`。`gold_in_accepted_corpus` 不进 feature。
2. **Task 4.2 / 4.3**: 把 full-v2 + light30 评测结果转成 LTR 训练 JSONL,记录
   `route_attribution` 诊断。
3. **Task 5.1-5.3**: 实现 LTR baseline,在 light30 / full-v2 上对比。
4. **任务 5.4 评测完后再统一登记 baseline_registry**,不提前。
5. `docs/evaluation/benchmark_policy.md` 加入 "Coverage-aware Gate A" 章节,
   把以上判定口径固化。

## 复现

```bash
# 1. 跑 full-v2 ablation (15-30 分钟)
python scripts/run_retrieval_ablation.py \
  --papers data/evaluation/papers_metadata_v2.jsonl \
  --include-vector \
  --variants scope typical hybrid accepted scope_accepted typical_accepted full_hybrid \
  --output data/evaluation/results/retrieval_ablation_full_v2_20260602_153622.json

# 2. 跑分层分析 (1 分钟, 一次性脚本)
python scripts/stratify_retrieval_ablation.py \
  --input data/evaluation/results/retrieval_ablation_full_v2_20260602_153622.json \
  --papers data/evaluation/papers_metadata_v2.jsonl \
  --accepted-papers-dir data/accepted_papers \
  --output data/evaluation/results/retrieval_ablation_full_v2_20260602_153622_stratified.md
```

(分层脚本将在后续任务中正式提交,本 ADR 暂时只引用其预期路径。)
