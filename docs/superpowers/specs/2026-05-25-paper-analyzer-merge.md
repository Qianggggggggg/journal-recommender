# PaperParser + PaperQualityAssessor 合并设计

> **目标：** 将两次 LLM 调用（PaperParser.parse + PaperQualityAssessor.assess）合并为单次调用，消除 research_area / ccf_research_area 不一致，降低延迟和成本。

> **范围：** Call #1 + Call #2 合并；LLMRanker 保持不变。

---

## 一、Prompt 组合方式

### System Prompt（唯一）

```
你是一个学术论文分析与质量评估专家。请严格按照下面用户消息中的两个阶段完成任务：
第一阶段：从论文中提取结构化特征。
第二阶段：基于特征和 CCF 审稿标准进行多维度质量评估。
最终输出一个完整的 flat JSON 对象，包含两部分的所有字段。
```

### User Prompt：两个 Section 嵌入原 system prompt 内容

```
Section A - 结构化特征提取：

[原 paper_profile_system 内容]

[原 paper_profile_user 内容]

---

Section B - 质量评审：

[原 paper_quality_assessor_system 内容]

[原 paper_quality_assessor_user 内容]
```

**原理：** 原两份 system prompt 的指令语义原封不动保留在 user message 中，单次 LLM 调用完成两类任务，LLM 不会感到角色混乱。

---

## 二、输出字段（Flat JSON）

```json
{
  // Section A - PaperParser 等效输出
  "research_area": ["AI", "NLP"],
  "method_type": "method",
  "paper_type": "application",
  "keywords": ["transformer", "..."],
  "novelty": "...",
  "application_domain": ["..."],
  "techniques": ["..."],
  "datasets": ["..."],
  "evaluation_metrics": ["..."],
  "novelty_type": "...",

  // Section B - PaperQualityAssessor 等效输出
  "ccf_research_area": ["人工智能", "交叉/综合/新兴"],
  "quality_level": "B",
  "paper_strength": 0.72,
  "novelty_score": 2,
  "rigor_score": 2,
  "reproducibility_score": 1,
  "significance_score": 2,
  "clarity_score": 2,
  "confidence": 0.75,
  "reasons": ["..."],
  "uncertainty_reasons": ["..."],
  "major_weaknesses": ["..."],
  "fatal_issues": null,
  "method_innovation_analysis": "..."
}
```

### 字段类型防御

同原有逻辑，对列表字段（research_area, ccf_research_area, keywords, techniques, datasets, evaluation_metrics, application_domain）做类型检查：LLM 可能返回字符串而非列表，发现后按逗号分割或置空。

---

## 三、新增模块：PaperAnalyzer

### 文件

- 新建：`src/papers/paper_analyzer.py`
- 修改：`src/papers/paper_model.py`（新增 preferred_area 字段）
- 修改：`src/recommender/pipeline.py`（调用 PaperAnalyzer 替换原有两次调用）

### PaperAnalyzer 接口

```python
class PaperAnalyzer:
    """合并后的论文分析与质量评估器（单次LLM调用）"""

    def __init__(self, llm: MiniMaxLLM):
        ...

    @tenacity.retry(...)
    def analyze(self, paper_input: PaperInput, prompts: dict) -> PaperProfile:
        """执行合并后的分析，返回完整 PaperProfile"""
        # 填 user prompt（含两个 Section）
        # 调用 llm.chat_auto()
        # 解析 JSON，防御类型
        # 赋值 ccf_research_area 到 paper_profile.ccf_research_area
        # 赋值 preferred_area = ccf_research_area
        ...

    def analyze_with_fallback(self, paper_input: PaperInput, prompts: dict) -> PaperProfile:
        """优先合并调用，字段缺失时自动回退到两次独立调用"""
        ...
```

### 错误处理

沿用原有 tenacity 重试机制（exponential backoff，min=2s, max=8s, 最多3次），解析失败时抛出 `PaperAnalyzerError`。

---

## 四、下游切换 ccf_research_area（关键设计）

合并后 `ccf_research_area` 稳定产出，**强制下游使用它**。

### 统一访问入口

在 `analyze()` 返回前：
```python
paper_profile.ccf_research_area = ccf_research_area_list
paper_profile.preferred_area = ccf_research_area_list  # 统一访问入口
```

### CandidateGenerator 修改

- Tag 检索/领域预过滤 → 只读 `preferred_area`（即 ccf_research_area）
- 原来的 `research_area` 不参与任何匹配逻辑，保留为辅助展示字段

### RuleScorer 修改

- `ccf_area_match` 特征 → 使用 `preferred_area`
- 改为**二值命中**：命中至少一个 CCF 标签即得满分，避免广域期刊过度加分

---

## 五、Pipeline 变更

原有流程：
```
PaperParser.parse() [LLM#1] → PaperQualityAssessor.assess() [LLM#2] → Candidates → Rule → LLM Ranker
```

新流程：
```
PaperAnalyzer.analyze() [LLM#1] → Candidates → Rule → LLM Ranker
```

Pipeline 中调用方式：
```python
# 原来
paper_profile = self.parser.parse(paper_input, system_prompt, user_prompt)
quality = self.assessor.assess(paper_input, paper_profile, ...)

# 改为
paper_profile = self.analyzer.analyze(paper_input, prompts)  # 一次搞定
```

---

## 六、向后兼容

- `PaperParser` 类保留，不删除（其他模块可能直接调用）
- `PaperQualityAssessor` 类保留，不删除
- `analyze_with_fallback()` 提供配置开关 `enable_fallback: true/false`，默认开启

---

## 七、测试计划

1. **paper_strength 相关性**：新版的 paper_strength 与旧版两次调用结果的 Pearson 相关性
2. **Top5 推荐列表重叠率**：旧版 vs 新版的 Top5 期刊重叠比例，要求 > 90%
3. **ccf_research_area 命中率**：验证新驱动后候选召回是否更精准
4. **Fallback 触发测试**：模拟解析失败，验证 fallback 是否正常回退到两次调用
5. **字段完整性检查**：所有 profile 字段 + quality 字段均非空时的百分比

---

## 八、文件变更清单

| 操作 | 文件 |
|---|---|
| 新建 | `src/papers/paper_analyzer.py` |
| 修改 | `src/papers/paper_model.py`（新增 `preferred_area` 字段） |
| 修改 | `src/recommender/pipeline.py`（替换调用方） |
| 修改 | `configs/prompts.yaml`（新增 unified_analyzer_system / unified_analyzer_user） |
| 新建 | `tests/test_paper_analyzer.py` |
| 修改 | `tests/` 下原有相关测试（更新调用方式） |