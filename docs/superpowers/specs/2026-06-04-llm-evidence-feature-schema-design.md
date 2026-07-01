# 阶段 6.2：LLM Evidence Feature Schema 设计

## 目标

把阶段 6.1 产生的六维结构化 LLM evidence 加入 FeatureBuilder，同时保持现有
20 维 LTR 模型、训练数据和线上推理路径完全兼容。

## 兼容策略

FeatureBuilder 提供两套显式 schema：

- `FEATURE_NAMES`：现有 20 维基础 schema。默认使用，现有 LTR 模型继续消费。
- `FEATURE_NAMES_WITH_LLM_EVIDENCE`：基础 20 维加六个 evidence 特征，共 26 维。

新增的六个特征是：

```text
llm_scope_fit
llm_method_fit
llm_application_fit
llm_journal_position_fit
llm_too_broad_penalty
llm_too_narrow_penalty
```

`PaperCandidateFeatures.to_vector()` 默认仍输出 20 维。只有调用者显式传入
`FEATURE_NAMES_WITH_LLM_EVIDENCE` 时才输出 26 维。

## Evidence 输入与默认值

`build_features()` 接收可选的单本期刊 evidence 字典。
`attach_features_to_trace()` 接收可选的、按 `journal_id` 索引的 evidence 字典。

缺失或非法 evidence 使用中性默认值：

- 四个 fit 分数：`0.5`
- 两个 penalty 分数：`0.0`

非法包括：布尔值、非数值、超出 `[0, 1]` 的数值。非法值不截断，以中性值替代，
避免坏响应静默制造强正向或负向信号。

## 默认行为

- `build_features()` 默认构建包含中性 evidence 字段的 dataclass，但
  `to_vector()` 默认只序列化基础 20 维。
- `attach_features_to_trace()` 默认写入基础 20 维，保持所有现有调用者行为不变。
- evidence 实验必须同时显式传入 evidence 字典和
  `FEATURE_NAMES_WITH_LLM_EVIDENCE`。

## 非目标

- 不在任务 6.2 中调用 LLM。
- 不接入推荐 pipeline。
- 不重训或替换现有 20 维 LTR 模型。
- 不实现任务 6.3 的 LLM 角色消融。

## 验证标准

- 默认 `FEATURE_NAMES` 仍为 20 维，旧路径输出不变。
- evidence schema 为 26 维且后六维顺序固定。
- 合法 evidence 正确进入 26 维向量。
- 缺失或非法 evidence 使用中性默认值。
- `attach_features_to_trace()` 可显式写入 26 维 evidence 特征。
