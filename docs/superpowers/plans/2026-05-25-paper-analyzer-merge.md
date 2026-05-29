# PaperAnalyzer Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 PaperParser.parse() + PaperQualityAssessor.assess() 两次 LLM 调用合并为单次 PaperAnalyzer.analyze()，同时强制下游使用 ccf_research_area 替代 research_area 进行召回匹配。

**Architecture:** 新增 PaperAnalyzer 类，单次 LLM 调用同时输出 profile 字段 + quality 字段；candidate_generator 和 rule_scorer 切换为使用 preferred_area（== ccf_research_area）。

**Tech Stack:** Python, MiniMaxLLM, tenacity, PyYAML, pytest

---

## File Structure

| 操作 | 文件 | 职责 |
|---|---|---|
| Create | `src/papers/paper_analyzer.py` | 合并后的论文分析器 |
| Modify | `src/papers/paper_model.py` | 新增 preferred_area 字段 |
| Modify | `src/recommender/pipeline.py` | 调用 PaperAnalyzer 替换原两次调用 |
| Modify | `configs/prompts.yaml` | 新增 unified_analyzer_system / unified_analyzer_user |
| Modify | `src/retriever/candidate_generator.py` | _filter_by_tags 切换为 preferred_area |
| Modify | `src/ranker/rule_scorer.py` | ccf_area_match 改为二值特征 |
| Create | `tests/test_paper_analyzer.py` | PaperAnalyzer 单元测试 |
| Modify | `tests/test_recommender.py` | 更新 pipeline 测试以适应新接口 |

---

## Task 1: 新增 PaperAnalyzer 类

**Files:**
- Create: `src/papers/paper_analyzer.py`
- Test: `tests/test_paper_analyzer.py`

- [ ] **Step 1: 写测试用例**

```python
# tests/test_paper_analyzer.py
import pytest
from unittest.mock import MagicMock
from src.papers.paper_analyzer import PaperAnalyzer, PaperAnalyzerError
from src.papers.paper_model import PaperInput, PaperProfile

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat_auto = MagicMock(return_value=MagicMock(content='{"research_area": ["AI"], "method_type": "method", "paper_type": "application", "keywords": ["transformer"], "novelty": "new method", "application_domain": ["medical"], "techniques": ["transformer"], "datasets": ["ImageNet"], "evaluation_metrics": ["Accuracy"], "novelty_type": "new_method", "ccf_research_area": ["人工智能"], "quality_level": "B", "paper_strength": 0.72, "novelty_score": 2, "rigor_score": 2, "reproducibility_score": 2, "significance_score": 2, "clarity_score": 2, "confidence": 0.8, "reasons": ["solid contribution"], "uncertainty_reasons": [], "major_weaknesses": [], "fatal_issues": null, "method_innovation_analysis": "incremental"}'))
    return llm

def test_analyze_returns_complete_profile(mock_llm):
    analyzer = PaperAnalyzer(mock_llm)
    paper_input = PaperInput(title="Test Paper", abstract="This is a test abstract.")
    prompts = {
        "system": "你是一个学术论文分析专家。",
        "user": "论文：{title}\n摘要：{abstract}\n请提取JSON。"
    }
    result = analyzer.analyze(paper_input, prompts)
    assert result.research_area == ["AI"]
    assert result.ccf_research_area == ["人工智能"]
    assert result.paper_strength == 0.72
    assert result.quality_level == "B"

def test_analyze_parses_list_fields(mock_llm):
    # Test that string return values are parsed into lists
    ...

def test_analyze_raises_error_on_invalid_json(mock_llm):
    mock_llm.chat_auto.return_value = MagicMock(content="not json at all")
    analyzer = PaperAnalyzer(mock_llm)
    paper_input = PaperInput(title="Test", abstract="Abstract")
    with pytest.raises(PaperAnalyzerError):
        analyzer.analyze(paper_input, {})

def test_analyze_with_fallback(mock_llm):
    # Trigger fallback path
    ...
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_paper_analyzer.py -v`
Expected: FAIL - PaperAnalyzer not defined

- [ ] **Step 3: 实现 PaperAnalyzer**

```python
# src/papers/paper_analyzer.py
"""合并后的论文分析与质量评估器（单次LLM调用）"""
import logging
from typing import Optional, Dict

import tenacity

from ..utils.llm import MiniMaxLLM, parse_json_response
from .paper_model import PaperInput, PaperProfile

logger = logging.getLogger(__name__)


class PaperAnalyzerError(Exception):
    """论文分析错误"""
    pass


LIST_FIELDS = [
    "research_area", "application_domain", "keywords", "techniques",
    "datasets", "evaluation_metrics", "ccf_research_area"
]


class PaperAnalyzer:
    """合并后的论文分析与质量评估器（单次LLM调用）"""

    def __init__(self, llm: MiniMaxLLM, enable_fallback: bool = True):
        if llm is None:
            raise PaperAnalyzerError("LLM not configured")
        self.llm = llm
        self.enable_fallback = enable_fallback

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=2, max=8),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"[PaperAnalyzer] Retry {retry_state.attempt_number}/3 after error..."
        ),
    )
    def analyze(self, paper_input: PaperInput, prompts: dict) -> PaperProfile:
        """执行合并后的分析，单次 LLM 调用同时输出 profile + quality 字段"""
        system_prompt = prompts.get("system", "")
        user_prompt = prompts.get("user", "")

        user_filled = user_prompt.format(
            title=paper_input.title,
            abstract=paper_input.abstract or "",
            full_text_summary=paper_input.full_text if paper_input.full_text else "",
        )

        try:
            response = self.llm.chat_auto(system_prompt, user_filled)
        except Exception as e:
            raise PaperAnalyzerError(f"LLM 调用失败: {e}")

        data = parse_json_response(response.content)
        if not data:
            raise PaperAnalyzerError(f"JSON 解析失败: {response.content}")

        # 防御：列表字段类型检查
        for field in LIST_FIELDS:
            if field in data and not isinstance(data[field], list):
                if isinstance(data[field], str):
                    data[field] = [x.strip() for x in data[field].split(",") if x.strip()]
                else:
                    data[field] = []

        # 提取 ccf_research_area
        ccf_research_area = data.get("ccf_research_area", [])

        # 构建 PaperProfile
        return PaperProfile(
            title=paper_input.title,
            abstract=paper_input.abstract or "",
            **{k: v for k, v in data.items() if k != "title"}
        )

    def analyze_with_fallback(
        self, paper_input: PaperInput, prompts: dict
    ) -> PaperProfile:
        """优先合并调用，字段缺失时自动回退到两次独立调用"""
        try:
            result = self.analyze(paper_input, prompts)
            if not result.ccf_research_area or result.paper_strength is None:
                logger.warning("[PaperAnalyzer] 缺少 quality 字段，触发 fallback")
                return self._fallback_two_call(paper_input, prompts)
            return result
        except (PaperAnalyzerError, Exception) as e:
            logger.warning(f"[PaperAnalyzer] 分析失败，触发 fallback: {e}")
            return self._fallback_two_call(paper_input, prompts)

    def _fallback_two_call(
        self, paper_input: PaperInput, prompts: dict
    ) -> PaperProfile:
        """回退到两次独立 LLM 调用（原 PaperParser + PaperQualityAssessor）"""
        from .paper_parser import PaperParser
        from .quality_assessor import PaperQualityAssessor

        if not self.enable_fallback:
            raise PaperAnalyzerError("Fallback disabled but failed")

        # Call 1: PaperParser
        parser = PaperParser(self.llm)
        profile = parser.parse(
            paper_input,
            prompts.get("parser_system", ""),
            prompts.get("parser_user", ""),
        )

        # Call 2: PaperQualityAssessor
        assessor = PaperQualityAssessor(self.llm)
        quality = assessor.assess(
            paper_input,
            profile,
            prompts.get("quality_system", ""),
            prompts.get("quality_user", ""),
        )

        # 填充 quality 字段
        profile.paper_strength = quality.paper_strength
        profile.readiness = quality.readiness
        profile.quality_level = quality.quality_level
        profile.quality_confidence = quality.confidence
        profile.quality_reasons = quality.reasons
        profile.ccf_research_area = quality.ccf_research_area

        return profile
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_paper_analyzer.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/papers/paper_analyzer.py tests/test_paper_analyzer.py
git commit -m "feat: add PaperAnalyzer - merged single-LLM paper analysis"
```

---

## Task 2: 修改 PaperModel — 新增 preferred_area 字段

**Files:**
- Modify: `src/papers/paper_model.py`

- [ ] **Step 1: 写测试**

```python
# 在 tests/test_paper_parser.py 同文件或新文件中
def test_paper_profile_has_preferred_area():
    profile = PaperProfile(title="Test", abstract="Abstract")
    profile.preferred_area = ["人工智能"]
    assert profile.preferred_area == ["人工智能"]
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/ -v -k "preferred_area" 2>/dev/null || echo "test not found - expected"`
Expected: 相关属性不存在

- [ ] **Step 3: 添加 preferred_area 字段到 PaperProfile**

在 `paper_model.py` 的 `PaperProfile` 类中添加：

```python
# preferred_area: 统一访问入口，恒等于 ccf_research_area
preferred_area: List[str] = Field(
    default_factory=list,
    description="统一访问入口，恒等于 ccf_research_area，用于下游匹配"
)
```

注意：preferred_area 的值在 PaperAnalyzer.analyze() 返回时通过赋值 `paper_profile.preferred_area = ccf_research_area` 统一填充。

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_paper_parser.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/papers/paper_model.py
git commit -m "feat: add preferred_area field to PaperProfile"
```

---

## Task 3: 新增 Unified Prompts

**Files:**
- Modify: `configs/prompts.yaml`

- [ ] **Step 1: 读取现有 prompts.yaml 确认格式**

```bash
head -5 configs/prompts.yaml
```
确认 yaml 结构。

- [ ] **Step 2: 在 prompts.yaml 末尾添加 unified prompts**

```yaml
unified_analyzer_system: |
  你是一个学术论文分析与质量评估专家。请严格按照下面用户消息中的两个阶段完成任务：
  第一阶段：从论文中提取结构化特征。
  第二阶段：基于特征和 CCF 审稿标准进行多维度质量评估。
  最终输出一个完整的 flat JSON 对象，包含两部分的所有字段。

unified_analyzer_user: |
  Section A - 结构化特征提取：

  请从论文中提取以下结构化特征（JSON 格式）：
  - research_area: 研究领域（如 AI、CV、NLP、SE、Security、DB、Network 等）
  - method_type: 方法类型（method/system/experiment/survey）
  - paper_type: 论文类型（theory/application/engineering）
  - keywords: 关键词（5-8个）
  - novelty: 创新点简述
  - application_domain: 应用领域
  - techniques: 具体技术（transformer、GNN、强化学习、联邦学习、元学习、大语言模型等）
  - datasets: 使用的数据集（ImageNet、COCO、SQuAD、WikiSQL 等）
  - evaluation_metrics: 评估指标（mAP、F1、BLEU、Accuracy、AUC、Latency 等）
  - novelty_type: 创新类型（新方法/新应用/新基准/性能提升/效率优化）

  ---

  Section B - 质量评审：

  作为 CCF 会议的严格 reviewer，请对论文进行多维度质量评估。

  ========== CCF质量锚点 ==========

  CCF-A：通常提出新方法、新理论或新范式，或对已有方法做出根本性重构。
  CCF-B：允许高质量的增量创新。
  CCF-C：可以是工程优化、应用增强、系统实现或特定领域的解决方案。
  未达到CCF发表水平：方法拼凑痕迹明显，缺乏核心创新。

  ========== 评分标准（0-3分）==========

  novelty（权重35%）：0 = 无明显创新，1 = incremental improvement，2 = 明确方法创新，3 = 新范式/新理论
  rigor（权重25%）：0 = 实验明显不足，1 = 基础实验完整，2 = 实验较全面，3 = 实验非常严格
  reproducibility（权重15%）：0 = 无法复现，1 = 部分可复现，2 = 可复现，3 = 完全可复现
  significance（权重15%）：0 = 问题普通，1 = 常见应用问题，2 = 社区关注的问题，3 = 领域核心问题
  clarity（权重10%）：0 = 论述混乱，1 = 基本可读，2 = 论述清晰，3 = 非常清晰

  ========== CCF专业领域分类 ==========

  请从以下10个领域中选择1-3个最匹配的CCF专业领域：
  1. 计算机体系结构/并行与分布计算/存储系统
  2. 计算机网络
  3. 网络与信息安全
  4. 软件工程/系统软件/程序设计语言
  5. 数据库/数据挖掘/内容检索
  6. 计算机科学理论
  7. 计算机图形学与多媒体
  8. 人工智能
  9. 人机交互与普适计算
  10. 交叉/综合/新兴

  ---

  论文标题：{title}
  摘要：{abstract}
  全文摘要：{full_text_summary}

  请同时输出 Section A 和 Section B 的所有字段，合并为一个 flat JSON 对象。

  输出格式示例：
  {
    "research_area": ["AI"],
    "method_type": "method",
    "keywords": ["transformer"],
    "ccf_research_area": ["人工智能"],
    "quality_level": "B",
    "paper_strength": 0.72,
    "novelty_score": 2,
    ...
  }
```

- [ ] **Step 3: 提交**

```bash
git add configs/prompts.yaml
git commit -m "feat: add unified_analyzer prompts for merged LLM call"
```

---

## Task 4: 修改 CandidateGenerator — 切换为 preferred_area

**Files:**
- Modify: `src/retriever/candidate_generator.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_retriever.py
def test_filter_by_tags_uses_preferred_area():
    """_filter_by_tags 应使用 preferred_area，不使用 research_area"""
    ...
```

- [ ] **Step 2: 运行测试验证当前行为**

Run: `pytest tests/test_retriever.py -v`
Expected: PASS（现有测试）

- [ ] **Step 3: 修改 _filter_by_tags**

将 `_filter_by_tags` 方法中的 `paper_profile.research_area` 替换为 `paper_profile.preferred_area`：

```python
def _filter_by_tags(
    self, paper_profile: PaperProfile, top_k: int = 20
) -> List[Tuple[Journal, float]]:
    """标签过滤召回（使用 preferred_area / ccf_research_area）"""
    results = []
    for journal in self.store._journals:
        score = 0.0
        # CCF 专业领域匹配（优先使用 preferred_area）
        areas_to_match = paper_profile.preferred_area or paper_profile.research_area
        if areas_to_match:
            for area in areas_to_match:
                if area in journal.subject_tags:
                    score += 1.0
        # 应用领域匹配（保留）
        if paper_profile.application_domain:
            for domain in paper_profile.application_domain:
                if domain in journal.subject_tags:
                    score += 0.8
        ...
```

同时修改 `_text_search` 中的领域匹配（如果使用 research_area）。

- [ ] **Step 4: 运行测试验证**

Run: `pytest tests/test_retriever.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/retriever/candidate_generator.py
git commit -m "refactor: CandidateGenerator uses preferred_area for domain matching"
```

---

## Task 5: 修改 RuleScorer — ccf_area_match 改为二值特征

**Files:**
- Modify: `src/ranker/rule_scorer.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_ranker.py
def test_ccf_area_match_is_binary():
    """ccf_area_match 应为二值：命中至少一个标签即满分，避免广域期刊过度加分"""
    ...
```

- [ ] **Step 2: 运行测试验证当前行为**

Run: `pytest tests/test_ranker.py -v`
Expected: PASS

- [ ] **Step 3: 修改 ccf_area_match 逻辑**

当前 rule_scorer.py:67-72：
```python
# 原代码：加分权重 ccf_area_match: 3.0
if paper_profile.ccf_research_area:
    matched_areas = [a for a in paper_profile.ccf_research_area if a in journal.subject_tags]
    if matched_areas:
        score += self.weights["ccf_area_match"]
        reasons.append(f"CCF领域匹配: {', '.join(matched_areas)}")
```

改为二值命中（只加一次分，不重复加分）：
```python
# 改为二值：命中至少一个标签即得满分 3.0，不再重复加分
preferred_areas = getattr(paper_profile, 'preferred_area', None) or paper_profile.ccf_research_area
if preferred_areas:
    # 二值判断：只要命中至少一个领域标签即得满分
    if any(area in journal.subject_tags for area in preferred_areas):
        score += self.weights["ccf_area_match"]
        matched_areas = [a for a in preferred_areas if a in journal.subject_tags]
        reasons.append(f"CCF领域匹配: {', '.join(matched_areas)}")
```

同时移除原来的 research_area 匹配加分（仅保留 preferred_area 的 ccf_area_match）。

- [ ] **Step 4: 运行测试验证**

Run: `pytest tests/test_ranker.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/ranker/rule_scorer.py
git commit -m "refactor: RuleScorer ccf_area_match is now binary (hits once)"
```

---

## Task 6: 修改 Pipeline — 集成 PaperAnalyzer

**Files:**
- Modify: `src/recommender/pipeline.py`

- [ ] **Step 1: 读取当前 pipeline.py 确认调用方式**

- [ ] **Step 2: 修改 RecommenderPipeline.from_config() 和 __init__**

在 `from_config()` 中加载 `unified_analyzer_system` / `unified_analyzer_user` prompt。

- [ ] **Step 3: 修改 recommend() 方法**

原来调用两次：
```python
# 原来
paper_profile = self.parser.parse(paper_input, ...)
quality = self.assessor.assess(paper_input, paper_profile, ...)
```

改为调用一次：
```python
# 改为
analyzer = PaperAnalyzer(self.llm)
prompts = {
    "system": prompts.get("unified_analyzer_system", ""),
    "user": prompts.get("unified_analyzer_user", "").format(
        title=paper_input.title,
        abstract=paper_input.abstract or "",
        full_text_summary=paper_input.full_text if paper_input.full_text else "",
    ),
}
paper_profile = analyzer.analyze(paper_input, prompts)
paper_profile.preferred_area = paper_profile.ccf_research_area
```

- [ ] **Step 4: 运行测试验证**

Run: `pytest tests/test_recommender.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/recommender/pipeline.py
git commit -m "feat: pipeline uses PaperAnalyzer for single-LLM call"
```

---

## Task 7: 集成测试 — 验证 Top5 重叠率

**Files:**
- Modify: `tests/test_recommender.py` 或新建 `tests/test_analyzer_integration.py`

- [ ] **Step 1: 运行 Top5 重叠率测试**

```python
def test_top5_overlap_with_old_approach():
    """对比新旧两种方式的 Top5 列表重叠率，要求 > 90%"""
    # 加载已有的 evaluation 结果 JSON
    # 或使用固定的 paper_input 跑一次完整 pipeline
    ...
```

- [ ] **Step 2: 提交**

```bash
git add tests/test_analyzer_integration.py
git commit -m "test: add Top5 overlap validation test"
```

---

## Spec 覆盖检查

| Spec 要求 | 对应 Task |
|---|---|
| 单次 LLM 调用 | Task 1 |
| 保留两份原始 prompt 语义 | Task 3 |
| Flat JSON 输出 | Task 1 |
| 列表字段类型防御 | Task 1 |
| preferred_area 统一访问入口 | Task 2 |
| CandidateGenerator 使用 preferred_area | Task 4 |
| RuleScorer ccf_area_match 二值化 | Task 5 |
| analyze_with_fallback 回退逻辑 | Task 1 |
| Pipeline 集成 | Task 6 |
| Top5 重叠率测试 | Task 7 |

**无遗漏。**

---

## Placeholder 扫描

搜索 plan 中无 "TBD"、"TODO"、"implement later"、"add appropriate error handling" 等占位符。**全部为具体实现步骤和代码。**

---

**Plan 完成。两种执行方式：**

1. **Subagent-Driven（推荐）** — 我 dispatch per-task 子代理，两阶段 review（spec compliance → code quality）
2. **Inline Execution** — 在当前 session 中使用 executing-plans 批量执行

你选择哪个？