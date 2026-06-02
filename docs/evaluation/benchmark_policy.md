# 评测基准治理策略 (Benchmark Governance Policy)

本策略文件定义推荐系统评测基准 (`benchmark`) 的三层分层、泄漏规则、profile snapshot 规则，以及违反时的处理流程。它是 `docs/superpowers/plans/2026-06-01-publication-grade-journal-recommender-plan.md` 阶段 1.3 的正式落档，被 `run_evaluation.py` 的 `--benchmark-profile`、`build_lightweight_eval_set.py`、`clean_benchmark.py` 等脚本共同遵守。

## 背景 / 为什么需要分层

推荐系统对 prompt、reranker、parser 随机性都敏感，单一 benchmark 难以同时支持"快速迭代"与"可信论文数字"。本策略把 benchmark 显式分成 `light30-dev` / `full-v2-dev` / `heldout-final` 三层，分别承担"快"、"准"、"冻结报告"三种职责，避免在 holdout 上调参或拿 dev 数字冒充最终结果。

## 三层 Benchmark 定义

| 层级 | 数据文件 | 规模 | 用途 | 迭代频率 |
|------|----------|------|------|----------|
| `light30-dev` | `data/evaluation/papers_metadata_light_30.jsonl` | 30 篇 (按 `(research_area, ccf_level)` 各取 1 篇) | 快速迭代：prompt 调优、reranker 草稿、parser 改动 | 高频；LLM 可自由重跑；不进入论文表格 |
| `full-v2-dev` | `data/evaluation/papers_metadata_v2.jsonl` | 全量 v2 评测集 | 参数选择：特征消融、reranker 训练数据生成、阈值确定 | 中频；可多次 LLM 调用，但应复用 `paper_profile_snapshot` |
| `heldout-final` | `data/evaluation/papers_metadata_heldout*.jsonl` (TBD) | TBD | 仅用于最终论文表格 | **冻结**；仅允许 1 次正式 run，任何重跑必须公开记录 |

`heldout-final` 的具体文件路径在后续 plan 阶段确定之前，本文件保持 forward-looking；`run_evaluation.py` 的 `--benchmark-profile` 接入以 `heldout-final` 命名即可。

## 各层使用规则

### `light30-dev` —— 快速迭代

允许：prompt 调优、reranker 草稿、parser 改动、embedding 切换、多次 LLM 调用、删除/补全样本。
禁止：写入正式论文表格、作为最终数字被引用。

进入 light30 benchmark 前必须先通过 `python scripts/build_lightweight_eval_set.py --validate-only` 校验（30 篇数量与 `(research_area, ccf_level)` 各 1 篇的分布）。

### `full-v2-dev` —— 参数选择

允许：特征消融、reranker 权重选择、阈值扫描、不同 `paper_profile_snapshot` 的对照实验。
要求：同一组配置下应复用同一个 `paper_profile_snapshot`（详见下文），以便数字可复现。
禁止：挑选"最有利的 run"作为最终结果；不得在调参后追加"过拟合"嫌疑实验。

### `heldout-final` —— 冻结

允许：单次正式 run、产出最终论文表格数字。
禁止：任何形式的调参、特征筛选、reranker 选择、候选重排筛选；不得基于 `heldout-final` 反馈修改算法并再次运行。
例外：若发现 run 异常（如 pipeline crash、配置错误），允许 1 次 rerun，但必须在论文中公开声明 rerun 原因与日期。例外仅限于以下客观条件之一：
  - (a) 流水线 crash 且未产出完整结果；
  - (b) 已保存的 `benchmark_manifest` 与运行配置不一致（由 manifest 自检发现）；
  - (c) LLM / embedding 服务 5xx 错误率超过 5% 且无法补救。
  数字偏低、Top-1 翻转率突增、acceptable@5 数值不理想等不构成 rerun 理由。

## 泄漏规则 (Leakage Rule)

正式实验开始前，必须对 `data/typical_abstracts/` 与 `data/accepted_papers/` 跑一次 `scripts/clean_benchmark.py`，且报告 `summary.leaked_typical_entry_count == 0` 与 `summary.leaked_accepted_paper_entry_count == 0`（`summary.match_count == 0` 为其别名）。具体规则：

- 测试论文 `title` 不得出现在 `data/typical_abstracts/*.json` 的扫描字段：`title` / `paper_title` / `source_title` / `abstract` / `text`（五字段均参与拼接，缺失字段视为空串）。
- 测试论文 `abstract` 不得作为整体或 **≥ 160 字符片段** 出现在 typical abstracts 中。
- 测试论文 `title` 不得出现在 `data/accepted_papers/*.json` 的 `papers[*].title` 字段。
- 测试论文 `abstract` 不得作为整体或 **≥ 160 字符片段** 出现在 accepted papers 中。

160 字符阈值与 1.2 阶段的实现保持一致，由 `src/evaluation/clean_benchmark.py` 中 `DEFAULT_MIN_ABSTRACT_CHARS = 160` 定义。

实现由 `scripts/clean_benchmark.py` 完成，报告 JSON 中每条命中带 `source_type` 字段，取值为 `"typical_abstract"` 或 `"accepted_paper"`，用于定位泄漏源。报告路径默认在 `data/evaluation/results/clean_benchmark_leakage_<timestamp>.json`。

```bash
python scripts/clean_benchmark.py \
  --input data/evaluation/papers_metadata_light_30.jsonl \
  --typical-dir data/typical_abstracts \
  --accepted-paper-dir data/accepted_papers \
  --report data/evaluation/results/light30_leakage_report.json
```

注：`data/accepted_papers/` 目录在阶段 2 才正式构建；本阶段的检测在目录缺失时静默跳过，文档中其它示例均按 forward-looking 写法。

报告 `matches` 数组必须为空（`summary.leaked_entry_count == 0`）才能进入正式实验；否则应先清理 typical/accepted-paper 库再重新生成干净库。

## Profile Snapshot 规则

`paper_profile_snapshot` 是 LLM 把 raw paper 解析成 `PaperProfile` 时使用的固定输入切片，由 parser 版本、prompt 版本、模型名共同决定。

- 正式论文结果必须使用固定的 `paper_profile_snapshot`，并在 `benchmark_manifest` 中以 `profile_snapshot_reused: true` 标识。
- 当且仅当 `light30-dev` 调试允许使用 non-fixed snapshot 时，必须在报告里单独报告 parser 随机性：
  - 同一篇 paper 多次 LLM 调用的 score 方差、rank 变化、推荐 Top-1 翻转率。
  - 报告以附录形式呈现，不与主表数字混合。
- `full-v2-dev` 与 `heldout-final` 一律要求 fixed snapshot；非 fixed 视为违反本策略。

snapshot 内容由 `src/evaluation/benchmark_manifest.py` 在保存结果时记录；任何 `manifest.profile_snapshot_reused == false` 的结果不能进入论文表格。

语料侧 snapshot（`data/typical_abstracts/` 清理后的目录路径与 `data/accepted_papers/` 选定的版本）同样属于可复现契约：typical 侧的 clean 目录在 `clean_benchmark.py` 报告中以 `summary.clean_typical_dir` 字段记录；accepted-paper 侧的 snapshot 路径在阶段 2 完成后由 `benchmark_manifest` 记录（当前 forward-looking）。

## Accepted-Paper 外部数据源契约 (External Source Contract)

任务 2.2 完成后,`data/accepted_papers/` 仅由本地 evaluation metadata 灌出 63 本期刊画像;剩下 232 本期刊 (`typical_abstracts/` 覆盖 295 本,差额 232) 无 accepted-paper 信号。任务 2.3 把这个数据缺口正式定义成一份契约,使任何后续接入外部数据源的工作都以该契约为准。

### 启用前置 (Gate A)

外部数据源 (`semantic-scholar`、`openalex`) 当前 **预留接口、暂未启用**。`scripts/collect_accepted_papers.py --source semantic-scholar|openalex` 会立刻退出并返回 `external collection source is not enabled in this plan phase` 文案。启用必须同时满足:

1. 阶段 3.4 的 retrieval ablation 已经在 light30 / full-v2 上证明 accepted-paper route 对 `coarse@50` 或 `exact Hit@5` 有正贡献 (Gate A)。
2. 接入方案有显式 ADR (`docs/adr/`) 记录,说明数据源选择、字段映射、限速策略。
3. 第一次抓取必须用 `--exclude-eval-input` 同时排除 `papers_metadata_light_30.jsonl` 和 `papers_metadata_v2.jsonl`,以及未来 `heldout-final` 的输入文件。

在 Gate A 未达成之前,即便有同事提议"先抓数据备着",也应当拒绝——避免数据准备早于算法验证,导致 sunk-cost 偏置。

### 必填字段 (与本地源对齐)

每条外部源抓回来的论文,在写入 `data/accepted_papers/<journal_id>.json` 时必须满足与 `AcceptedPaperStore` 的契约一致 (见 `src/journals/accepted_paper_store.py`):

| 字段 | 必填 | 说明 |
|------|------|------|
| `title` | ✅ | 论文标题,空白/None 直接弃用 |
| `abstract` | ✅ | 论文摘要,长度过短 (< 50 字符) 视作脏数据弃用 |
| `year` | 可选 | 发表年份,整数或可转 int 的字符串,否则 `null` |
| `source` | ✅ | 必须是 `semantic_scholar` 或 `openalex`,**禁止与本地源混用同一 `journal_id` 文件下的 source 标记** |
| `doi` | 可选 | 字符串;若为空字符串则视为缺失 |
| `url` | 可选 | 论文原文 / pdf 链接 |

### Venue → journal_id 解析

外部源返回的 venue 名通常更脏 (缩写、变体、大小写不一致)。模糊匹配规则:

1. 先做与本地源一致的规范化 (`_normalize_venue`: lowercase + 折叠空白)。
2. 若一击命中 `JournalStore.journal_name`,接收。
3. 否则进入二次匹配:
   - NFKD + 去非字母数字后比对 (`_normalize_text` 风格)。
   - ISSN 字段交叉确认 (Semantic Scholar / OpenAlex 都返回 ISSN,可与 `journals_ccf.jsonl` 或 `journals_v1.jsonl` 中 ISSN 字段对照)。
4. 仍不命中,记录到 `summary.unresolved_venues`,**禁止猜测匹配**;后续可人工补 alias 后重跑。

### 限速与缓存

- **Semantic Scholar API**: 公共端点 100 req/5min;批量端点 5000 paper/req。每个抓取脚本必须显式 sleep,把单 host 的并发降到 1。
- **OpenAlex API**: 100k req/day,推荐使用 `mailto` 参数提升优先级。
- 抓取必须落到 `data/raw/external_papers/<source>/<journal_id>.json` 作为缓存层 (raw 响应原样存档),再由 collect 脚本转换成 `data/accepted_papers/<journal_id>.json`。raw 缓存允许 commit 进仓库以加速复现。

### 泄漏检测 (强制)

外部源接入后,**每次抓取完成必须重跑 leakage 检测**:

```bash
python scripts/clean_benchmark.py \
  --input data/evaluation/papers_metadata_light_30.jsonl \
  --typical-dir data/typical_abstracts \
  --accepted-paper-dir data/accepted_papers \
  --report data/evaluation/results/leakage_after_external_<timestamp>.json

python scripts/clean_benchmark.py \
  --input data/evaluation/papers_metadata_v2.jsonl \
  --typical-dir data/typical_abstracts \
  --accepted-paper-dir data/accepted_papers \
  --report data/evaluation/results/leakage_after_external_v2_<timestamp>.json
```

两份报告必须同时满足 `summary.leaked_accepted_paper_entry_count == 0`。若任一非零,该批抓取作废,不允许进入正式实验。

### source 字段命名表

| source 值 | 含义 | 引入阶段 |
|-----------|------|---------|
| `local_evaluation_metadata` | 来自 `papers_metadata*.jsonl` | 阶段 2.2 |
| `semantic_scholar` | 来自 Semantic Scholar API | 阶段 2.3 启用后 |
| `openalex` | 来自 OpenAlex API | 阶段 2.3 启用后 |

下游 (`feature_builder`, retriever, LTR) **不得对不同 source 标记做差别加权**——所有真实发表论文画像在算法层面一视同仁;source 只用于追溯、调试与按需回退。

## 违反与处理

- 在 `heldout-final` 上发现调参、重新选择候选、基于反馈修改算法：视为泄漏 holdout，对应结果作废，`heldout-final` 集合应保持冻结；新实验必须更换 holdout 集。
- 在 `light30-dev` / `full-v2-dev` 上发现泄漏：对 `data/typical_abstracts/`，使用 `scripts/clean_benchmark.py` 的快照模式重写干净库；对 `data/accepted_papers/`，当前仅产出报告（阶段 2 才完成 accepted-paper 清理能力），需人工从源头重新构建。两边修复后重跑相应 benchmark。
- 任何例外（如 `heldout-final` 必要的 1 次 rerun）必须以 ADR (`docs/adr/`) 或 issue 形式记录原因、日期、操作人，不允许在 commit message 之外单独留痕。

## 相关文件

- 计划：`docs/superpowers/plans/2026-06-01-publication-grade-journal-recommender-plan.md` 阶段 1.3
- 扫描脚本：`scripts/clean_benchmark.py`
- 实现：`src/evaluation/clean_benchmark.py`、`src/evaluation/benchmark_manifest.py`
- 数据：`data/typical_abstracts/*.json`、`data/accepted_papers/*.json`
- 评测入口：`scripts/run_evaluation.py`（参数 `--benchmark-profile light30|full-v2|custom`；`custom` 需显式给 `--input`，可指向任意 jsonl）
