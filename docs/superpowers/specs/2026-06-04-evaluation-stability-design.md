# 评测稳定性设计

## 背景

Light30 的 LTR ON/OFF 对比存在两类非算法波动：

1. `run_evaluation.py` 每轮重新调用 `PaperParser`，同一论文的领域、质量等级、关键词和强度会变化，进而改变召回、RuleScorer 和 LTR 输入。
2. `LLMRanker` 接受 `{"rankings":[]}`，pipeline 最终返回空推荐，评测会把 API/格式失败静默计为模型排序失败。

正式评测仍需支持 `--workers 10`，不能依赖串行运行来规避问题。

## 目标

- LTR ON/OFF 等排序消融可以复用完全相同的 `paper_profile_snapshot`。
- LLM 空响应、空 `rankings` 或无有效候选匹配时触发已有重试。
- LLM 三次重试后仍失败时，使用 Rule Top5 生成非空结果。
- 结果文件明确区分正常 LLM 推荐和 fallback 推荐。
- 并发评测使用 `--workers 10` 时，稳定性机制仍然生效。

## 非目标

- 不改变正常成功时的 LLM 排序算法。
- 不通过降低并发数解决稳定性问题。
- 不在本任务实现阶段 6 的 LLM Evidence Extractor。
- 不把 fallback 结果伪装成纯 LLM 结果。

## 设计

### 1. 固定 PaperProfile 快照

`scripts/run_evaluation.py` 增加：

```text
--baseline-eval <completed-evaluation.json>
```

传入后，程序按规范化标题复用 baseline 结果中的 `paper_profile_snapshot`，跳过 `PaperParser`。输入论文集合和顺序以当前 benchmark 文件为准；如果某篇论文缺少匹配快照，立即报错，不允许静默重新解析。

manifest 写入：

- `profile_snapshot_reused: true`
- `baseline_eval_path`
- `workers`

这样 LTR ON/OFF 可以在 `--workers 10` 下共享同一套解析输入。

### 2. LLMRanker 空结果视为失败

`LLMRanker.rank()` 在以下情况抛出 `LLMRankerError`，由已有 tenacity 机制重试三次：

- `rankings` 不是非空列表；
- rankings 中没有任何当前候选的合法 `journal_id`；
- ranking item 不是对象或缺少合法 `journal_id`。

正常 LLM 返回路径保持不变。

### 3. Pipeline 最终兜底

LLM 三次重试后仍失败时，pipeline 不再向调用方抛出并产生空结果，而是使用当前 `llm_candidates` 的 RuleScorer 顺序前 TopK：

```text
LLM success -> existing final selection
LLM exhausted -> Rule Top5 fallback
```

fallback 返回项保留 RuleScorer 的分数与理由，并使用中性置信度。该路径不改变召回、RuleScorer 或 LTR 的候选集合。

### 4. 诊断与指标

每篇结果新增：

- `evaluation_status`: `ok` 或 `fallback`
- `fallback_used`: boolean
- `fallback_stage`: `llm_ranking` 或空
- `fallback_reason`: 精简错误类别
- `rank_method`: `llm` 或 `rule_fallback`

总结果新增：

- `fallback_count`
- `llm_success_count`
- `empty_recommendation_count`

正式报告必须同时说明 fallback 数量。端到端 Hit@5 包含 fallback；纯 LLM 表现应在 `fallback_used=false` 子集上另行统计。

### 5. 并发要求

正式 Light30 评测允许并推荐：

```bash
--workers 10
```

固定快照由只读输入构建，不在线程间修改。每篇论文的 LLM 重试和 fallback 独立执行，不共享可变状态。

## 公平评测流程

先生成一次固定解析基准：

```bash
python3 -u scripts/run_evaluation.py \
  --benchmark-profile light30 \
  --mode abstract \
  --top-k 5 \
  --workers 10
```

随后所有 LTR ON/OFF 对比复用该结果：

```bash
python3 -u scripts/run_evaluation.py \
  --benchmark-profile light30 \
  --baseline-eval data/evaluation/results/<baseline>.json \
  --mode abstract \
  --top-k 5 \
  --workers 10
```

公平性检查：

- `profile_snapshot_reused=true`
- ON/OFF 的解析字段完全一致
- ON/OFF 的 coarse@50 在候选算法配置相同时一致
- `empty_recommendation_count=0`
- 单独报告 `fallback_count`

## 测试

- `LLMRanker` 空 `rankings` 会重试并最终抛错。
- `LLMRanker` 无有效候选 ID 会重试并最终抛错。
- pipeline 在 LLM 最终失败时返回 Rule Top5，且标记 fallback。
- 正常 LLM 成功路径不受影响。
- `run_evaluation.py --baseline-eval` 复用快照且不调用 Parser。
- baseline 缺少某篇快照时立即失败。
- `workers=10` 下每篇结果完整，且固定快照不被修改。

## 成功标准

- 同一固定快照下重复运行不会再出现 Parser 字段漂移。
- LLM 空排名不再被静默接受。
- LLM 排名失败时最终推荐列表非空。
- Light30 LTR ON/OFF 可使用 `--workers 10` 做公平对比。
- 结果 JSON 能明确识别所有 fallback 样本。
