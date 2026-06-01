# 评测基准治理策略 (Benchmark Governance Policy)

本策略文件定义推荐系统评测基准 (`benchmark`) 的三层分层、泄漏规则、profile snapshot 规则，以及违反时的处理流程。它是 `docs/superpowers/plans/2026-06-01-publication-grade-journal-recommender-plan.md` 阶段 1.3 的正式落档，被 `run_evaluation.py` 的 `--benchmark-profile`、`build_lightweight_eval_set.py`、`clean_benchmark.py` 等脚本共同遵守。

## 背景 / 为什么需要分层

推荐系统对 prompt、reranker、parser 随机性都敏感，单一 benchmark 难以同时支持"快速迭代"与"可信论文数字"。本策略把 benchmark 显式分成 `light30-dev` / `full-v2-dev` / `heldout-final` 三层，分别承担"快"、"准"、"冻结报告"三种职责，避免在 holdout 上调参或拿 dev 数字冒充最终结果。

## 三层 Benchmark 定义

| 层级 | 数据文件 | 规模 | 用途 | 迭代频率 |
|------|----------|------|------|----------|
| `light30-dev` | `data/evaluation/papers_metadata_light_30.jsonl` | 30 篇 (按 `(research_area, ccf_level)` 各取 1 篇) | 快速迭代：prompt 调优、reranker 草稿、parser 改动 | 高频；LLM 可自由重跑；不进入论文表格 |
| `full-v2-dev` | `data/evaluation/papers_metadata_v2.jsonl` | 全量 v2 评测集 | 参数选择：特征消融、reranker 训练数据生成、阈值确定 | 中频；可多次 LLM 调用，但应复用 `paper_profile_snapshot` |
| `heldout-final` | `data/evaluation/papers_metadata_heldout*.jsonl` (TBD；后续阶段生成) | TBD | 仅用于最终论文表格 | **冻结**；仅允许 1 次正式 run，任何重跑必须公开记录 |

`heldout-final` 的具体文件路径在后续 plan 阶段确定之前，本文件保持 forward-looking；`run_evaluation.py` 的 `--benchmark-profile` 接入以 `heldout-final` 命名即可。

## 各层使用规则

### `light30-dev` —— 快速迭代

允许：prompt 调优、reranker 草稿、parser 改动、embedding 切换、多次 LLM 调用、删除/补全样本。
禁止：写入正式论文表格、作为最终数字被引用。

### `full-v2-dev` —— 参数选择

允许：特征消融、reranker 权重选择、阈值扫描、不同 `paper_profile_snapshot` 的对照实验。
要求：同一组配置下应复用同一个 `paper_profile_snapshot`（详见下文），以便数字可复现。
禁止：挑选"最有利的 run"作为最终结果；不得在调参后追加"过拟合"嫌疑实验。

### `heldout-final` —— 冻结

允许：单次正式 run、产出最终论文表格数字。
禁止：任何形式的调参、特征筛选、reranker 选择、候选重排筛选；不得基于 `heldout-final` 反馈修改算法并再次运行。
例外：若发现 run 异常（如 pipeline crash、配置错误），允许 1 次 rerun，但必须在 paper 中公开声明 rerun 原因与日期。

## 泄漏规则 (Leakage Rule)

正式实验开始前，必须对 `data/typical_abstracts/` 与 `data/accepted_papers/` 跑一次 `scripts/clean_benchmark.py`，且报告 `summary` 中 `typical_abstract` 与 `accepted_paper` 命中数均为 0。具体规则：

- 测试论文 `title` 不得出现在 `data/typical_abstracts/*.json` 的 `abstracts[*].title` / `paper_title` / `source_title` 字段。
- 测试论文 `abstract` 不得作为整体或 **≥ 160 字符片段** 出现在 typical abstracts 中。
- 测试论文 `title` 不得出现在 `data/accepted_papers/*.json` 的 `papers[*].title` 字段。
- 测试论文 `abstract` 不得作为整体或 **≥ 160 字符片段** 出现在 accepted papers 中。

160 字符阈值与 1.2 阶段的实现保持一致，由 `src/evaluation/clean_benchmark.py` 中 `DEFAULT_MIN_ABSTRACT_CHARS = 160` 定义。

实现由 `scripts/clean_benchmark.py` 完成，报告 JSON 中每条命中带 `source_type` 字段，取值为 `"typical_abstract"` 或 `"accepted_paper"`，用于定位泄漏源。报告路径默认在 `data/evaluation/results/clean_benchmark_leakage_<timestamp>.json`。

```bash
python scripts/clean_benchmark.py \
  --papers-jsonl data/evaluation/papers_metadata_light_30.jsonl \
  --typical-dir data/typical_abstracts \
  --accepted-paper-dir data/accepted_papers \
  --report data/evaluation/results/light30_leakage_report.json
```

报告 `matches` 数组必须为空（`summary.total_matches == 0`）才能进入正式实验；否则应先清理 typical/accepted-paper 库再重新生成干净库。

## Profile Snapshot 规则

`paper_profile_snapshot` 是 LLM 把 raw paper 解析成 `PaperProfile` 时使用的固定输入切片，由 parser 版本、prompt 版本、模型名共同决定。

- 正式论文结果必须使用固定的 `paper_profile_snapshot`，并在 `benchmark_manifest` 中以 `profile_snapshot_reused: true` 标识。
- 当且仅当 `light30-dev` 调试允许使用 non-fixed snapshot 时，必须在报告里单独报告 parser 随机性：
  - 同一篇 paper 多次 LLM 调用的 score 方差、rank 变化、推荐 Top-1 翻转率。
  - 报告以附录形式呈现，不与主表数字混合。
- `full-v2-dev` 与 `heldout-final` 一律要求 fixed snapshot；非 fixed 视为违反本策略。

snapshot 内容由 `src/evaluation/benchmark_manifest.py` 在保存结果时记录；任何 `manifest.profile_snapshot_reused == false` 的结果不能进入论文表格。

## 违反与处理

- 在 `heldout-final` 上发现调参、重新选择候选、基于反馈修改算法：视为泄漏 holdout，对应结果作废，`heldout-final` 集合应保持冻结；新实验必须更换 holdout 集。
- 在 `light30-dev` / `full-v2-dev` 上发现泄漏：立即用 `scripts/clean_benchmark.py` 清理 `data/typical_abstracts/` 与 `data/accepted_papers/`，重新生成干净库，并重跑相应 benchmark。
- 任何例外（如 `heldout-final` 必要的 1 次 rerun）必须以 ADR (`docs/adr/`) 或 issue 形式记录原因、日期、操作人，不允许在 commit message 之外单独留痕。

## 相关文件

- 计划：`docs/superpowers/plans/2026-06-01-publication-grade-journal-recommender-plan.md` 阶段 1.3
- 扫描脚本：`scripts/clean_benchmark.py`
- 实现：`src/evaluation/clean_benchmark.py`、`src/evaluation/benchmark_manifest.py`
- 数据：`data/typical_abstracts/*.json`、`data/accepted_papers/*.json`
- 评测入口：`scripts/run_evaluation.py`（参数 `--benchmark-profile light30|full-v2|custom`）
