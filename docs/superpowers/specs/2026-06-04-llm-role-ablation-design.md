# 阶段 6.3：LLM Role Ablation 设计

## 目标

公平比较 LLM 在期刊推荐中的三种角色：

1. `llm_ranker_direct`：LLM 直接决定最终排名。
2. `llm_evidence_plus_rule`：LLM 只提取结构化 evidence，Rule 顺序提供弱先验。
3. `llm_evidence_plus_learned_reranker`：LLM 只提取结构化 evidence，现有
   20 维 LTR 顺序提供弱先验。

本阶段不训练 26 维 evidence-aware LTR。第三个变体必须明确记录为“现有
20 维 LTR prior + LLM evidence”，不能声称 LTR 已消费六维 evidence 特征。

## 排名公式

Evidence 变体对每本候选计算：

```text
fit_mean = mean(scope_fit, method_fit, application_fit, journal_position_fit)
penalty_mean = mean(too_broad_penalty, too_narrow_penalty)
evidence_composite = clamp(fit_mean - penalty_mean, 0, 1)
rank_prior = 1 - (source_rank - 1) / max(source_population_size - 1, 1)
final_score = evidence_composite * 0.8 + rank_prior * 0.2
```

`source_rank` 必须是真实 Rule rank 或真实 learned rank，禁止回退到输入列表
位置。Rule prior 使用完整 Rule 候选池大小归一化；learned prior 使用完整 learned
rank 映射归一化。

若 Evidence Extractor 最终失败，所有候选使用中性 evidence：

- fit=`0.5`
- penalty=`0.0`
- `evidence_composite=0.5`

该论文仍按 prior 产生完整推荐，不返回空结果。

## 组件

### `LLMEvidenceRoleRanker`

新增 `src/ranker/llm_evidence_role_ranker.py`。

- 兼容 pipeline 的候选排序职责。
- 使用 `LLMEvidenceExtractor` 单次批量提取 evidence。
- 返回排名和本次调用独立的 diagnostics。
- diagnostics 包含每个候选的六维 evidence、composite、prior、final score、
  20/26 维特征向量与 schema。
- 不在实例字段保存“最近一次调用结果”，避免 `workers=10` 并发串数据。

### Pipeline 诊断透传

若 ranker 提供 `rank_with_diagnostics(...)`，pipeline 使用它并把本次返回的
diagnostics 写入 `result["llm_role_diagnostics"]`。6.3 runner 会用
`DirectLLMRoleRanker` 包装直接 LLM baseline；普通前端和默认评测继续使用现有
`LLMRanker.rank()` 路径，返回结构保持不变。

### 实验 Runner

新增 `scripts/run_llm_role_ablation.py`：

- 强制要求 `--baseline-eval`，复用固定 denominator 与
  `paper_profile_snapshot`。
- 支持 `--benchmark-profile light30|full-v2-90|custom`。
- 支持重复 `--variant`、`--workers`、`--papers`。
- 三个变体都关闭 anchor guard，隔离直接排序与 evidence+prior 的作用。
- 使用 `scripts/precompute_evidence.py` 为每篇论文的真实 LLM 候选池只提取一次
  evidence；两个 evidence 变体通过 `--evidence-snapshot` 复用相同 snapshot。
- 自动检查 denominator、逐论文 coarse@50、逐论文 rule@20、完整 evidence
  coverage，以及两个 evidence 变体的候选 evidence 是否逐字段完全一致。
- 每个变体保存独立 JSON，汇总保存一个 summary JSON。

## 完整候选诊断

Evidence 变体的每篇结果保存 `llm_candidates_detail`：

- journal ID/name
- input rank / prior source / rank prior
- 六维 evidence
- evidence 文本
- evidence composite
- final score / final rank
- Rule rank / score
- learned rank / score（LTR 变体）
- 20 维基础 features
- 26 维 evidence features
- feature schema

Gold venue 的 `venue_diagnostic` 同时保存对应 evidence 排名和分数。

## 公平性

公平性检查不只比较汇总计数，还比较每篇论文：

- 标题与 venue denominator 一致。
- `coarse_hit` 一致。
- `coarse_hit_in_rule_top20` 一致。
- 两个 evidence 变体的候选集合和六维 evidence/evidence 文本完全一致。
- 正式实验中每篇 `evidence_coverage == 1.0`。

任一不一致时 summary 标记 `fairness_pass=false` 并以非零退出。
`--allow-partial-snapshot` 仅用于调试：仍报告 coverage 问题，但不因 coverage
不足单独阻止调试运行。

## 非目标

- 不改变前端或正式 pipeline 默认配置。
- 不训练或替换当前 LTR 模型。
- 不运行真实 Light30/full-v2-90 评测。
- 不把 LLM evidence 直接写进期刊 corpus。
