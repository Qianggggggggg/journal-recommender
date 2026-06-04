# LLM Evidence Extractor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现阶段 6.1 的单次批量结构化 LLM Evidence Extractor，且不改变当前推荐 pipeline。

**Architecture:** 新建独立 `LLMEvidenceExtractor`，复用 `MiniMaxLLM.chat_auto()`、`parse_json_response()` 和 tenacity 重试。Extractor 将论文画像和完整候选池格式化为一次请求，严格校验返回的六维证据分数与文本证据，返回按 `journal_id` 索引的字典。

**Tech Stack:** Python 3.11、tenacity、pytest、YAML Prompt 模板。

---

### Task 1: 锁定 Evidence Extractor API 与合法响应行为

**Files:**
- Create: `tests/test_llm_evidence_extractor.py`
- Create: `src/ranker/llm_evidence_extractor.py`

- [x] **Step 1: 写合法批量响应的失败测试**

测试构造两个候选期刊，要求 `extract()` 只调用一次 LLM，并返回包含两个 `journal_id` 的证据字典；同时断言六个分数和文本证据被保留。

- [x] **Step 2: 运行测试确认 RED**

Run: `pytest tests/test_llm_evidence_extractor.py::test_extract_returns_structured_evidence_for_multiple_candidates -q`

Expected: FAIL，因为 `src.ranker.llm_evidence_extractor` 尚不存在。

- [x] **Step 3: 写最小实现**

实现：

```python
class LLMEvidenceExtractor:
    def extract(
        self,
        candidates: List[Tuple[Journal, float, List[str]]],
        paper_profile: PaperProfile,
    ) -> Dict[str, Dict[str, Any]]:
        ...
```

构建候选 JSON、调用 `chat_auto()`、解析 `{"evidence": [...]}` 或直接列表、严格校验后返回字典。

- [x] **Step 4: 运行测试确认 GREEN**

Run: `pytest tests/test_llm_evidence_extractor.py::test_extract_returns_structured_evidence_for_multiple_candidates -q`

Expected: PASS。

### Task 2: 增加严格校验和重试

**Files:**
- Modify: `tests/test_llm_evidence_extractor.py`
- Modify: `src/ranker/llm_evidence_extractor.py`

- [x] **Step 1: 写失败测试**

覆盖：

- 空 `evidence`
- 分数不是数值或超出 `[0, 1]`
- `evidence` 文本为空或类型错误
- 重复 `journal_id`
- 返回的 ID 全部不在候选池
- 前两次空结果、第三次合法结果时成功

- [x] **Step 2: 运行测试确认 RED**

Run: `pytest tests/test_llm_evidence_extractor.py -q`

Expected: 新增校验测试 FAIL。

- [x] **Step 3: 写最小校验实现**

新增 `LLMEvidenceExtractorError`、六个固定分数字段、候选 ID 过滤、重复检测，以及 tenacity 三次重试。

- [x] **Step 4: 运行测试确认 GREEN**

Run: `pytest tests/test_llm_evidence_extractor.py -q`

Expected: PASS。

### Task 3: 增加 Evidence Prompt 模板

**Files:**
- Modify: `configs/prompts.yaml`
- Modify: `tests/test_prompt_templates.py`

- [x] **Step 1: 写 Prompt 格式化失败测试**

读取 `llm_evidence_extractor_system` 和 `llm_evidence_extractor_user`，格式化所有输入字段，并断言输出合同包含 `scope_fit`、`journal_position_fit`、`too_broad_penalty` 和 `evidence`。

- [x] **Step 2: 运行测试确认 RED**

Run: `pytest tests/test_prompt_templates.py::test_llm_evidence_extractor_prompt_formats_with_json_example -q`

Expected: FAIL，因为 Prompt key 尚不存在。

- [x] **Step 3: 添加 Prompt**

Prompt 要求：

- 只输出合法 JSON。
- 对每个候选独立判断，不直接排序。
- 六个分数字段位于 `[0, 1]`。
- evidence 引用论文与期刊 scope 的具体匹配点。
- 不因为 CCF 等级或 rule_rank 单独提高 fit。

- [x] **Step 4: 运行测试确认 GREEN**

Run: `pytest tests/test_prompt_templates.py::test_llm_evidence_extractor_prompt_formats_with_json_example -q`

Expected: PASS。

### Task 4: 回归验证与阶段 6.1 收口

**Files:**
- Verify only.

- [x] **Step 1: 运行定向测试**

Run: `pytest tests/test_llm_evidence_extractor.py tests/test_prompt_templates.py tests/test_llm.py tests/test_llm_ranker.py -q`

Expected: 全部 PASS。

- [x] **Step 2: 运行相关回归测试**

Run: `pytest tests/test_feature_builder.py tests/test_ltr_adapter.py tests/test_recommender.py tests/test_api.py -q`

Expected: 全部 PASS，证明 6.1 没有改变现有 pipeline。

- [x] **Step 3: 静态检查**

Run: `python -m py_compile src/ranker/llm_evidence_extractor.py`

Expected: exit code 0。

Run: `git diff --check`

Expected: exit code 0。

**实际验证结果（2026-06-04）：**

- 6.1 原计划命令：`15 passed`。
- 6.1 + LLMRanker + Prompt + LTR/pipeline 回归：`75 passed, 2 deselected`。
- 两个 deselected 是已有配置/测试不一致：
  - 当前 `configs/app.yaml` 启用了 LTR，但 API 测试仍断言默认关闭。
  - 当前 LLMRanker Prompt 使用固定类型标签，但旧测试仍期待自然理由版 Prompt。
- `py_compile`、Prompt YAML 加载与相关文件 `git diff --check` 均通过。
