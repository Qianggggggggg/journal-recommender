# 质量评估反思机制实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 paper_quality_assessor_system 的"输出格式"之前插入双重反思机制，提升 CCF 等级判断准确性

**Architecture:** 修改 prompts.yaml 中的 paper_quality_assessor_system，在输出 JSON 格式前增加预判步骤和自检步骤

**Tech Stack:** YAML 配置文件修改

---

## 任务清单

### Task 1: 在输出格式前插入预判与自检机制

**Files:**
- Modify: `configs/prompts.yaml:340-342`

- [ ] **Step 1: 读取当前 prompts.yaml 确认插入位置**

确认"输出格式"前一段的位置（当前在第340-342行附近）

- [ ] **Step 2: 在"在 reasons 中必须包含..."之后、"输出格式"之前插入预判步骤**

在 `configs/prompts.yaml` 中找到以下内容：

```yaml
  在 reasons 中必须包含对整体等级倾向的说明，格式为"根据综合评分(raw=X.X, strength=Y.YY)，本文达到CCF-X水平"。如果你认为论文属于CCF-C，请确保各维度分数在1.5-2.0范围内，不要无意识地给出2.5以上的分数。

  输出格式（必须为有效JSON）:
```

替换为：

```yaml
  在 reasons 中必须包含对整体等级倾向的说明，格式为"根据综合评分(raw=X.X, strength=Y.YY)，本文达到CCF-X水平"。如果你认为论文属于CCF-C，请确保各维度分数在1.5-2.0范围内，不要无意识地给出2.5以上的分数。

  ========== 反思机制（必须执行）==========

  在你开始打分之前，请先完成以下预判步骤：

  1. 等级预判：根据论文摘要和标题，用2-3句话说明你判断这篇论文属于CCF-A/B/C中的哪个等级，以及最主要的依据是什么。不要在这里给具体分数，只需要给出等级和简要理由。

  2. 分数推导：基于你的等级预判，推断各维度应该在什么范围。例如："如果我认为这是CCF-C论文，那么 novelty 应在1.5-2.0之间..."

  3. 正式开始打分：只有在完成上述两步之后，才进入输出格式中的各项分数。

  4. 自检：在输出最终分数后，重新审视"这些分数是否与我在步骤1中的等级预判一致"。如果出现矛盾（例如预判CCF-C但 novelty=2.5），需要调整分数使其与预判等级匹配，并说明调整原因。

  输出格式（必须为有效JSON）:
```

- [ ] **Step 3: 验证修改结果**

读取 `configs/prompts.yaml` 第330-370行，确认插入内容正确

---

## 实施确认

计划完成并保存到 `docs/superpowers/plans/2026-05-28-quality-reflection-plan.md`。

**执行选项：**

1. **Subagent-Driven (推荐)** - 我调度子 agent 逐任务执行，任务间审查，快速迭代

2. **Inline Execution** - 在当前 session 中批量执行，带检查点

选择哪个？