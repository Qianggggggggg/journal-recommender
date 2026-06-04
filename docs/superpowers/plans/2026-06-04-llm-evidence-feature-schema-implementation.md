# LLM Evidence Feature Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 FeatureBuilder 增加可选的六维 LLM evidence 特征，同时保持现有 20 维 LTR 模型兼容。

**Architecture:** 保留 `FEATURE_NAMES` 作为默认 20 维 schema，新增由基础 schema 和六维 evidence schema 组成的 26 维 `FEATURE_NAMES_WITH_LLM_EVIDENCE`。FeatureBuilder 总是持有 evidence 字段，但默认序列化 20 维；调用者只有显式选择 26 维 schema 时才消费 evidence。

**Tech Stack:** Python 3.11、dataclasses、pytest。

---

### Task 1: 锁定版本化 schema 与中性默认值

**Files:**
- Modify: `tests/test_feature_builder.py`
- Modify: `src/ranker/feature_builder.py`

- [x] **Step 1: 写失败测试**

测试要求：

- `FEATURE_NAMES` 仍为 20 维。
- `LLM_EVIDENCE_FEATURE_NAMES` 固定为六个字段。
- `FEATURE_NAMES_WITH_LLM_EVIDENCE` 为 26 维且基础 20 维位于前部。
- `PaperCandidateFeatures.to_vector()` 默认输出 20 维，显式 evidence schema 输出 26 维。
- 缺失 evidence 使用 fit=`0.5`、penalty=`0.0`。

- [x] **Step 2: 运行测试确认 RED**

Run:

```bash
pytest tests/test_feature_builder.py -q
```

Expected: FAIL，因为版本化 evidence schema 尚不存在。

- [x] **Step 3: 写最小实现**

在 `feature_builder.py` 中新增 schema 常量、dataclass 字段、可选 schema 的
`to_vector()`，并让 `build_features()` 接受单本期刊 evidence。

- [x] **Step 4: 运行测试确认 GREEN**

Run:

```bash
pytest tests/test_feature_builder.py -q
```

Expected: PASS。

### Task 2: 支持 trace 注入 26 维 evidence 特征

**Files:**
- Modify: `tests/test_feature_builder.py`
- Modify: `src/ranker/feature_builder.py`

- [x] **Step 1: 写失败测试**

测试 `attach_features_to_trace()`：

- 默认仍注入 20 维。
- 显式传入 evidence 字典与 26 维 schema 时写入合法 evidence 分数。
- 某候选缺失或 evidence 非法时写入中性默认值。

- [x] **Step 2: 运行测试确认 RED**

Run:

```bash
pytest tests/test_feature_builder.py -q
```

Expected: 新增 trace evidence 测试 FAIL。

- [x] **Step 3: 写最小实现**

给 `attach_features_to_trace()` 增加两个可选参数：

```python
llm_evidence_by_journal: Optional[Dict[str, Dict[str, Any]]] = None
feature_names: Optional[List[str]] = None
```

默认使用 `FEATURE_NAMES`；显式 26 维 schema 时按 journal ID 注入 evidence。

- [x] **Step 4: 运行测试确认 GREEN**

Run:

```bash
pytest tests/test_feature_builder.py -q
```

Expected: PASS。

### Task 3: 回归验证与计划收口

**Files:**
- Modify: `docs/superpowers/plans/2026-06-01-publication-grade-journal-recommender-plan.md`

- [x] **Step 1: 运行 FeatureBuilder 定向测试**

Run:

```bash
pytest tests/test_feature_builder.py -q
```

- [x] **Step 2: 运行相关 LTR 与训练数据回归测试**

Run:

```bash
pytest tests/test_candidate_generator_features.py tests/test_ltr_adapter.py tests/test_build_ranking_training_data.py tests/test_learning_to_rank.py tests/test_train_learning_to_rank.py -q
```

- [x] **Step 3: 静态检查**

Run:

```bash
python -m py_compile src/ranker/feature_builder.py
git diff --check
```

- [x] **Step 4: 更新主计划**

勾选任务 6.2 的实现、默认值与测试项，记录默认 20 维、显式 26 维的兼容策略。

**实际验证结果（2026-06-04）：**

- TDD RED：`tests/test_feature_builder.py` 因缺少
  `FEATURE_NAMES_WITH_LLM_EVIDENCE` 导入失败。
- FeatureBuilder 定向测试：`35 passed`。
- 候选生成、检索消融、LTRAdapter、训练数据、LTR 训练与推荐 pipeline
  相关回归：`110 passed`。
- 当前 `data/models/learning_to_ranker.json` 的 20 维 `feature_names` 与
  默认 `FEATURE_NAMES` 逐项一致。
- `py_compile` 与相关文件 `git diff --check` 均通过。
