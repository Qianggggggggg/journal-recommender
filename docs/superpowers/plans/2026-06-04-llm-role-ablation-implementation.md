# LLM Role Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现任务 6.3 的三角色 LLM 消融 runner，并保存公平性与完整候选诊断。

**Architecture:** 新增无共享调用状态的 `LLMEvidenceRoleRanker`，通过 pipeline 的 `rank_with_diagnostics` 窄接口返回 evidence 排名和候选诊断。独立 runner 复用固定 profile snapshots，按变体配置 direct LLM、evidence+Rule、evidence+LTR，并执行逐论文公平性检查。

**Tech Stack:** Python 3.11、pytest、现有 evaluation pipeline、LLMEvidenceExtractor。

---

### Task 1: Evidence Role Ranker

**Files:**
- Create: `src/ranker/llm_evidence_role_ranker.py`
- Create: `tests/test_llm_role_ablation.py`

- [x] 写测试锁定 composite、线性 rank prior、最终分数、稳定排序。
- [x] 写测试锁定 extractor 失败时使用中性 evidence。
- [x] 写测试锁定 diagnostics 包含 20/26 维 features 且不保存共享 last-result 状态。
- [x] 实现 `LLMEvidenceRoleRanker.rank_with_diagnostics(...)`。

### Task 2: Pipeline 与 Evaluation 诊断透传

**Files:**
- Modify: `src/recommender/pipeline.py`
- Modify: `scripts/run_evaluation.py`
- Modify: `tests/test_recommender.py`
- Modify: `tests/test_run_evaluation_diagnostics.py`

- [x] 写测试锁定现有 `LLMRanker.rank()` 路径不变。
- [x] 写测试锁定 `rank_with_diagnostics()` 返回值进入 pipeline result。
- [x] 写测试锁定 evidence 变体保存完整 `llm_candidates_detail` 与 gold evidence 诊断。
- [x] 实现本次调用独立的 diagnostics 透传。

### Task 3: 三角色实验 Runner

**Files:**
- Create: `scripts/run_llm_role_ablation.py`
- Modify: `tests/test_llm_role_ablation.py`

- [x] 定义三个固定变体及其 LTR/prior 配置。
- [x] 强制要求 `--baseline-eval`。
- [x] 实现逐论文 denominator/coarse@50/rule@20 公平性检查。
- [x] 保存每变体结果、effective config 与 summary JSON。
- [x] 支持 `--workers 10`、重复 `--variant`、`--papers`。
- [x] 新增 evidence 预计算脚本，只针对真实 LLM 候选池提取一次 evidence。
- [x] Runner 加载 `--evidence-snapshot` 并让两个 evidence 变体复用同一份 evidence。
- [x] Rule/LTR prior 使用真实 rank，禁止静默回退到输入列表位置。
- [x] 公平性检查增加完整 coverage 与 evidence bit-equal gate。

### Task 4: 文档与静态检查

**Files:**
- Modify: `docs/superpowers/plans/2026-06-01-publication-grade-journal-recommender-plan.md`

- [x] 勾选任务 6.3 实现项并记录真实测试待用户运行。
- [x] 运行 `python -m py_compile`。
- [x] 运行 `git diff --check`。
- [x] 向用户提供单元测试、Light30 和 full-v2-90 命令。

**实现状态（2026-06-04）：**

- 代码、测试契约、设计文档、evidence pre-pass 和 runner 已实现。
- 定向回归测试已通过；真实 Light30/full-v2-90 角色消融仍由用户运行。
- `py_compile`、CLI `--help` 与相关文件 `git diff --check` 已通过。
- 推荐 Light30 固定快照：
  `data/evaluation/results/eval_abstract_top5_20260604_140602.json`。
- 推荐 full-v2-90 固定快照：
  `data/evaluation/results/eval_abstract_top5_20260603_204839.json`。
