# 阶段 6.1：结构化 LLM Evidence Extractor 设计

## 目标

把 LLM 从直接决定最终期刊顺序的黑盒排序器，逐步改造成结构化证据提取器。任务 6.1 只实现独立的 Evidence Extractor，不接入推荐 pipeline，不改变当前线上或评测结果。

## 核心设计

Evidence Extractor 对一篇论文和一批候选期刊执行一次 LLM 请求，返回每本候选期刊的结构化匹配证据：

```json
{
  "evidence": [
    {
      "journal_id": "ton",
      "scope_fit": 0.87,
      "method_fit": 0.73,
      "application_fit": 0.62,
      "journal_position_fit": 0.81,
      "too_broad_penalty": 0.12,
      "too_narrow_penalty": 0.05,
      "evidence": ["期刊 scope 明确覆盖网络协议与性能分析"]
    }
  ]
}
```

采用单次批量提取，而不是逐候选调用：

- 保留全部 `llm_candidates`，避免丢失 LTR 从低 Rule 排名救回的候选。
- 一篇论文只调用一次 LLM，成本与现有 `LLMRanker` 接近。
- 每个候选独立输出证据，不要求 LLM 直接给出最终排序。

## 组件边界

新增 `src/ranker/llm_evidence_extractor.py`：

- `LLMEvidenceExtractorError`：明确的业务异常。
- `LLMEvidenceExtractor.extract(...)`：接收候选期刊与 `PaperProfile`，返回按 `journal_id` 索引的证据字典。
- 复用 `MiniMaxLLM.chat_auto()` 和 `parse_json_response()`。
- 复用现有三次指数退避重试策略。

新增 Prompt：

- `llm_evidence_extractor_system`
- `llm_evidence_extractor_user`

## 输入

论文侧输入：

- title
- abstract
- research_area / ccf_research_area
- method_type / paper_type
- keywords / novelty / application_domain
- techniques / datasets / evaluation_metrics / novelty_type

期刊侧输入：

- journal_id / journal_name
- scope / subject_tags / keywords / ccf_rating
- rule_rank / rule_reasons

Rule 信息仅作为弱先验和解释上下文，不作为强制排序约束。

## 校验与错误处理

- 空响应、无法解析 JSON、空 evidence：抛出 `LLMEvidenceExtractorError` 并重试。
- evidence item 必须是对象，且 `journal_id` 必须是字符串。
- 六个分数字段必须存在、必须是数值、必须位于 `[0, 1]`。
- `evidence` 必须是非空字符串列表。
- 未知候选期刊 ID 忽略。
- 重复 `journal_id` 拒绝，避免静默覆盖。
- 若合法结果没有匹配任何候选期刊，则抛错并重试。
- 允许 LLM 只返回候选池的一部分；6.2 对缺失候选使用中性默认值。

## 非目标

- 不接入 `RecommenderPipeline`。
- 不修改 `FeatureBuilder`。
- 不替换现有 `LLMRanker`。
- 不改变当前 LTR、RuleScorer 或最终排序。
- 不在 6.1 中实现 LLM role ablation。

## 验证标准

- 单次请求可返回多个候选的结构化证据。
- JSON code fence 可由现有解析器修复。
- 空结果、非法分数、重复 ID、未知-only ID 会触发明确异常。
- 空结果会按现有策略重试，第三次合法响应可以成功。
- Prompt 模板可正常格式化。
- `tests/test_llm_evidence_extractor.py` 与 `tests/test_llm.py` 通过。
