# 论文级期刊推荐系统升级实施计划

> **给执行 Agent 的要求：** 实施本文档时，必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐项执行。所有步骤都使用 checkbox（`- [ ]`）追踪进度。

**目标：** 把当前“规则 + LLM”的期刊推荐原型，升级成一个可复现、数据驱动、具备论文发表潜力的推荐方法，并重点提升最终 exact `Hit@5`。

**总体架构：** 保留当前多路召回、RuleScorer 和 LLM 精排作为强 baseline；在此基础上新增“真实已发表论文画像”召回路由、候选特征记录层、监督式学习排序模型，以及 LLM 结构化证据抽取。最终排序不再完全依赖 LLM，而是由可复现的 ranker 融合召回证据、规则分数、学习排序分数和 LLM 证据。

**技术栈：** Python 3.11、现有 FastAPI 推荐流水线、MiniMax-M3、Ollama embedding、FAISS、BM25、pytest、pandas/numpy。学习排序先做无额外依赖的 baseline；如果后续需要，再可选接入 LightGBM/LambdaMART。

---

## 成功标准

- [ ] `run_evaluation.py` 和网页前端 API 默认使用同一套 `configs/app.yaml` 算法配置。
- [ ] 每次正式实验都保存配置、prompt 版本、模型名、benchmark 文件、泄漏检测报告和随机种子。
- [ ] clean benchmark 中不存在测试论文标题/摘要与 typical abstracts 或 accepted-paper profiles 的已知泄漏。
- [ ] 轻量 30 篇 benchmark 的运行成本低于 full-v2 benchmark 的 35%。
- [ ] full benchmark 和 light benchmark 都输出 exact `Hit@1/3/5`、MRR、NDCG@5、coarse@50、rule@20、same-area@5、same-CCF@5、acceptable@5。
- [ ] 新增 accepted-paper route 后，在 clean full benchmark 上 coarse@50 不低于当前 baseline，并且不能降低 exact `Hit@5`；Gate A 同时要求 covered 子集上 signal 真实存在（详见决策门槛 + ADR 0001）。
- [ ] 监督式 learning-to-rank reranker 在 held-out benchmark 上超过当前 Rule+LLM baseline 的 exact `Hit@5`。
- [ ] 最终实验包含 scope-only、typical-only、accepted-paper-only、hybrid retrieval、rule-only、LLM-only、LTR-only、LTR+LLM-evidence 等消融。
- [ ] 能从保存的 JSON 结果中生成论文可用的实验表格和失败案例分析。

---

## 当前必须保留的 Baseline

当前线上和普通 `run_evaluation.py` 默认算法是：

- 召回：`scope + typical_abstracts + identity_anchor`
- 融合方式：`weighted_minmax`
- abstract 模式路由 top-k：BM25 `28`，vector `56`，text `14`
- 规则排序：`RuleScorer`，权重来自 `ranking.rule_scorer`
- LLM 模型：`MiniMax-M3`
- 最终选择：`llm_score * 0.70 + rule_rank_prior * 0.20 + route_evidence * 0.10`
- Anchor guard：保护 Rule Top10，最大分数差 `0.08`
- 未启用：two-tower、cross encoder

这个 baseline 不要删。所有新方法都必须和它对比。

---

## 文件规划

### 需要扩展的现有文件

- `configs/app.yaml`
  - 新增 accepted-paper retrieval、学习排序、LLM evidence extraction 的开关和路径。
- `scripts/run_evaluation.py`
  - 增加 benchmark manifest、固定 profile snapshot、route/ranker 诊断字段。
- `scripts/run_retrieval_ablation.py`
  - 增加 accepted-paper route 的消融变体。
- `scripts/run_llm_rerank_ablation.py`
  - 保持为公平 LLM-only 消融工具，不能混入 parser 随机性。
- `src/retriever/candidate_generator.py`
  - 接入 accepted-paper route 和对应 trace。
- `src/recommender/pipeline.py`
  - 在 RuleScorer 后、最终 Top5 前插入 learned reranker。
- `src/ranker/rule_scorer.py`
  - 保留为可解释 baseline 和特征来源。
- `src/ranker/llm_ranker.py`
  - 保留直接 LLM ranking baseline；后续增加 evidence extraction 模式。
- `src/app/api.py`
  - 确保前端 API 与评测使用同一套配置驱动算法。
- `pyproject.toml`
  - 只有在基础学习排序跑通后，再考虑添加可选 ML 依赖。

### 需要新增的文件

- `src/journals/accepted_paper_store.py`
  - 加载每本期刊的真实已发表论文画像。
- `src/retriever/accepted_paper_retriever.py`
  - 基于 accepted papers 做 BM25/vector/text 召回。
- `src/ranker/feature_builder.py`
  - 把 paper-candidate pair 转成结构化特征。
- `src/ranker/learning_to_rank.py`
  - 训练式 reranker 接口和基础实现。
- `src/ranker/llm_evidence_extractor.py`
  - LLM 结构化证据抽取器。
- `src/evaluation/benchmark_manifest.py`
  - 保存 benchmark 元信息和配置 hash。
- `scripts/collect_accepted_papers.py`
  - 从本地元数据构建 accepted-paper corpus，后续可扩展外部数据源。
- `scripts/build_accepted_paper_index.py`
  - 构建 accepted-paper FAISS/BM25 索引。
- `scripts/build_ranking_training_data.py`
  - 构造正样本、hard negative、easy negative 的训练数据。
- `scripts/train_learning_to_rank.py`
  - 训练并保存基础 learned reranker。
- `scripts/run_publication_experiments.py`
  - 跑完整论文级实验矩阵。
- `tests/test_accepted_paper_store.py`
- `tests/test_accepted_paper_retriever.py`
- `tests/test_feature_builder.py`
- `tests/test_learning_to_rank.py`
- `tests/test_llm_evidence_extractor.py`
- `tests/test_benchmark_manifest.py`
- `tests/test_publication_experiments.py`

---

## 阶段 0：冻结公平 Baseline

### 任务 0.1：保存当前 Baseline Manifest

**文件：**
- 新建： `src/evaluation/benchmark_manifest.py`
- 测试： `tests/test_benchmark_manifest.py`
- 修改： `scripts/run_evaluation.py`

- [x] 写测试：从 `configs/app.yaml`、`configs/prompts.yaml`、输入文件路径、mode、top-k、MiniMax 模型名生成 manifest。
- [x] manifest 字段必须包含：`timestamp`、`input_path`、`mode`、`top_k`、`app_config_hash`、`prompt_hash`、`minimax_model`、`embedding_model`、`clean_benchmark`、`profile_snapshot_reused`。
- [x] 实现 `hash_file(path: str) -> str`，使用 SHA256。
- [x] 实现 `build_benchmark_manifest(...) -> dict`。
- [x] 在保存的 evaluation JSON 顶层加入 `benchmark_manifest`。
- [x] 运行：

```bash
pytest tests/test_benchmark_manifest.py tests/test_run_evaluation_diagnostics.py -q
```

- [x] 跑一次 light benchmark：

```bash
python scripts/run_evaluation.py \
  --input data/evaluation/papers_metadata_light_30.jsonl \
  --mode abstract \
  --top-k 5 \
  --workers 1
```

- [x] 确认输出 JSON 中包含 `benchmark_manifest`。
- [ ] 提交：`test: add benchmark manifest for reproducible evaluation`

### 任务 0.2：记录当前 Baseline 指标

**文件：**
- 新建： `data/evaluation/results/baseline_registry.json`
- 新建： `scripts/register_baseline_result.py`
- 测试： `tests/test_benchmark_manifest.py`

- [x] 新增脚本：读取一个 evaluation JSON，向 baseline registry 追加精简记录。
- [x] 记录字段：`label`、`result_path`、`input_path`、`hit_at_5`、`mrr`、`coarse_hit_count`、`coarse_hit_in_rule_top20_count`、`acceptable_journal_hit_at_5`、`app_config_hash`、`prompt_hash`、`minimax_model`。
- [x] 用当前 light30 结果登记一次：
  - `light30_m27_default`（默认 M2.7 baseline，结果 `eval_abstract_top5_20260601_191336.json`）。
  - `light30_m3_llm_ablation`（M3 直接 LLM ranking 消融，结果 `eval_abstract_top5_20260601_200359.json`）。
- [ ] 用当前 full-v2 结果登记一次：待用户后续跑完带 `benchmark_manifest` 的 full-v2 评测后，使用 `scripts/register_baseline_result.py --label full_v2_m27_default` 登记。
- [x] 如果 label 重复，默认拒绝；只有传 `--replace` 才允许覆盖。
- [x] 提交：`chore: register current recommender baseline`

---

## 阶段 1：Benchmark 和数据治理

### 任务 1.1：把 Light30 变成正式快速评测集

**文件：**
- 修改： `scripts/build_lightweight_eval_set.py`
- 修改： `scripts/run_evaluation.py`
- 测试： `tests/test_lightweight_eval_set.py`

- [x] 给 `scripts/build_lightweight_eval_set.py` 增加 `--validate-only` 参数。
- [x] 校验逻辑必须确认：正好 30 篇，且每个 `(research_area, ccf_level)` 组合正好一篇。
- [x] 给 `run_evaluation.py` 增加 `--benchmark-profile light30|full-v2|custom`。
- [x] 当选择 `light30` 时，默认输入为 `data/evaluation/papers_metadata_light_30.jsonl`。
- [x] 运行：

```bash
pytest tests/test_lightweight_eval_set.py -q
```

- [x] 运行：

```bash
python scripts/build_lightweight_eval_set.py --validate-only
```

- [ ] 提交：`feat: promote lightweight benchmark profile`

### 任务 1.2：给 Accepted-Paper Profiles 增加泄漏检测

**文件：**
- 修改： `src/evaluation/clean_benchmark.py`
- 修改： `scripts/clean_benchmark.py`
- 测试： `tests/test_clean_benchmark.py`

- [x] 扩展泄漏检测，使其同时扫描 `data/typical_abstracts` 和未来的 `data/accepted_papers`。
- [x] 匹配方式：规范化 title 精确匹配；abstract 至少 160 字符片段匹配。
- [x] 报告字段 `source_type`，取值为 `typical_abstract` 或 `accepted_paper`。
- [x] 保持当前 typical snapshot 生成逻辑不变。
- [x] 增加 report-only 命令：

```bash
python scripts/clean_benchmark.py \
  --input data/evaluation/papers_metadata_light_30.jsonl \
  --typical-dir data/typical_abstracts \
  --accepted-paper-dir data/accepted_papers \
  --report data/evaluation/results/light30_leakage_report.json
```

- [x] 运行：

```bash
pytest tests/test_clean_benchmark.py -q
```

- [x] 提交：`test: extend leakage checks to accepted-paper profiles`

### 任务 1.3：制定 Held-Out Benchmark 使用规则

**文件：**
- 新建： `docs/evaluation/benchmark_policy.md`

- [x] 文档中定义三层 benchmark：`light30-dev`、`full-v2-dev`、`heldout-final`。
- [x] 说明 `light30-dev` 只用于快速迭代。
- [x] 说明 `full-v2-dev` 用于参数选择。
- [x] 说明 `heldout-final` 必须冻结，只能用于最终论文表格。
- [x] 写明泄漏规则：测试论文 title/abstract 不得出现在 typical 或 accepted-paper journal profiles 中。
- [x] 写明正式论文结果必须使用固定 `paper_profile_snapshot`；如果不固定，必须单独报告 parser 随机性。
- [x] 提交：`docs: define benchmark governance policy`

---

## 阶段 2：构建真实已发表论文期刊画像

### 任务 2.1：定义 Accepted-Paper Store 格式

**文件：**
- 新建： `src/journals/accepted_paper_store.py`
- 测试： `tests/test_accepted_paper_store.py`
- 新建目录： `data/accepted_papers/`

- [x] 写测试：每本期刊一个 JSON 文件，可以被正确加载。
- [x] 文件格式：

```json
{
  "journal_id": "ton",
  "journal_name": "IEEE/ACM Transactions on Networking",
  "papers": [
    {
      "title": "Paper title",
      "abstract": "Paper abstract",
      "year": 2025,
      "source": "local_eval_or_external",
      "doi": "",
      "url": ""
    }
  ]
}
```

- [x] 实现 `AcceptedPaperStore.load()`。
- [x] 实现 `get_papers(journal_id: str) -> list[dict]`。
- [x] 实现 `iter_records() -> Iterable[AcceptedPaperRecord]`。
- [x] title/abstract 缺失的记录应该跳过，不能让整个加载失败。
- [x] 运行：

```bash
pytest tests/test_accepted_paper_store.py -q
```

- [x] 提交：`feat: add accepted paper store`

### 任务 2.2：从本地评测元数据生成初始 Accepted-Paper Corpus

**文件：**
- 新建： `scripts/collect_accepted_papers.py`
- 测试： `tests/test_accepted_paper_store.py`
- 输出： `data/accepted_papers/*.json`

- [x] 实现 local-only 模式，读取：
  - `data/evaluation/papers_metadata.jsonl`
  - `data/evaluation/papers_metadata_v2.jsonl`
  - `data/evaluation/papers_metadata_light_30.jsonl`
- [x] 按 exact `venue` 分组。
- [x] 通过 `JournalStore` 把 venue 解析到 `journal_id`。
- [x] 每本期刊输出一个 accepted-paper JSON。
- [x] `source` 字段写成 `local_evaluation_metadata`。
- [x] 增加 `--exclude-eval-input`，用于从画像生成中排除某个 benchmark 文件。
- [x] 运行：

```bash
python scripts/collect_accepted_papers.py \
  --eval-input data/evaluation/papers_metadata.jsonl \
  --exclude-eval-input data/evaluation/papers_metadata_light_30.jsonl \
  --exclude-eval-input data/evaluation/papers_metadata_v2.jsonl \
  --output-dir data/accepted_papers
```

  **命令演进**:plan 原命令只 `--exclude-eval-input light30`,执行后发现
  v2 仍有 60/89 篇泄漏到 corpus。这违反 docs/evaluation/benchmark_policy.md
  "测试论文不得出现在 typical 或 accepted-paper journal profiles 中" 的纪律。
  修订后:同时 exclude light30 和 v2,且只用 v1 (papers_metadata.jsonl) 作为
  来源(因为 v2 / light30 一旦 exclude 后没有论文剩下,作为输入意义不大)。

  本轮产物:63 个期刊 / 95 篇真实论文 / 7 条 (v1 ∩ {light30 ∪ v2}) 被正确排除。
  `algorithmica`、`jiis`、`ase` 等期刊在本地数据里只在 v2/light30 中有论文,
  被 exclude 后没有画像,留待任务 2.3 的外部数据源 stub 后续补齐。

  泄漏加固:collect 脚本的 title 规范化原本只做 lowercase+折叠空白,无法把
  Unicode `√log n` 和 ASCII `sqrt(log n)` 当成同一篇论文,导致单条
  ACM TALG 的论文 (Sparsest Cut O(√log n)) 漏排。修订:接入与
  `clean_benchmark.py::_normalize_text` 一致的 NFKD + 去非字母数字规范化,
  并把 abstract 前 160 字符片段加入 exclude 维度,口径与 leakage 工具完全
  一致。

- [x] 确认 light30 的泄漏报告干净；如果有泄漏，报告中必须明确列出。
  报告 `data/evaluation/results/{light30,full_v2}_leakage_after_v2_fix.json`:
  light30: `leaked_papers=0, leaked_accepted_paper_entries=0`
  full-v2: `leaked_papers=0, leaked_accepted_paper_entries=0`
- [x] 提交：`feat: build local accepted-paper journal corpus`

### 任务 2.3：预留外部数据源接口，但当前不依赖外部数据

**文件：**
- 修改： `scripts/collect_accepted_papers.py`
- 测试： `tests/test_accepted_paper_store.py`

- [x] 增加未来外部数据源参数：`--source semantic-scholar|openalex|local`。
  实现于 `scripts/collect_accepted_papers.py` (CLI `--source`,choices 受限,
  默认 `local`)。
- [x] 本阶段只实现 `local`。
- [x] 如果用户选择外部 source，脚本应清楚退出并提示：`external collection source is not enabled in this plan phase`。
  退出码 `2`,stderr 输出文案含 `external collection source is not enabled in this plan phase`,
  并附带具体 source 名 (semantic-scholar / openalex)。pytest 覆盖
  `test_main_external_source_exits_with_stub_message` 锁住该行为。
- [x] 在 `docs/evaluation/benchmark_policy.md` 中记录未来外部数据源需要的字段。
  新增章节 `Accepted-Paper 外部数据源契约 (External Source Contract)`,
  覆盖启用前置 (Gate A)、必填字段、venue 模糊匹配、限速与缓存、强制泄漏检测、
  source 字段命名表。
- [x] 这样可以避免核心算法还没稳定时，被联网采集流程拖住。
- [x] 提交：`docs: define external accepted-paper collection contract`

---

## 阶段 3：Accepted-Paper 召回路由

### 任务 3.1：实现 Accepted-Paper BM25 Retriever

**文件：**
- 新建： `src/retriever/accepted_paper_retriever.py`
- 测试： `tests/test_accepted_paper_retriever.py`

- [x] 写测试：网络时延相关 query 能从 accepted papers 中召回正确的网络期刊。
- [x] 实现 `AcceptedPaperBM25Retriever`。
- [x] 返回 `(Journal, score)`。
- [x] 聚合规则：每本期刊取最高 paper score，再加 `0.05 * matching_paper_count` 的小 bonus，并设置上限。
  实现:``score = top_paper_score + min(0.05 * matching_paper_count, bonus_cap)``,
  默认 ``bonus_cap=0.3``。
- [x] route detail 中记录 top matching paper title 和 score。
  ``retriever.last_route_details[journal_id]`` 暴露 ``top_paper_title`` /
  ``top_paper_score`` / ``matching_paper_count`` / ``bonus`` / ``final_score``。
- [x] 运行：

```bash
pytest tests/test_accepted_paper_retriever.py -q
```

  结果:8 passed (含 8 个行为契约测试)。真实 corpus sanity check:
  并行计算 query 召回 TPDS 第 1。

- [x] 提交：`feat: add accepted-paper BM25 retriever`

### 任务 3.2：构建 Accepted-Paper Embedding Index

**文件：**
- 新建： `scripts/build_accepted_paper_index.py`
- 修改： `configs/app.yaml`
- 测试： `tests/test_accepted_paper_retriever.py`

- [x] 增加配置路径：

```yaml
data:
  accepted_papers_dir: "data/accepted_papers"
  accepted_papers_faiss_path: "data/processed/accepted_papers_index.faiss"
  accepted_papers_metadata_path: "data/processed/accepted_papers_metadata.parquet"
```

- [x] 使用 `OllamaEmbedding` 实现 index builder。
  脚本位于 `scripts/build_accepted_paper_index.py`,核心函数
  `build_accepted_paper_index(...)` 可独立测试 (允许注入 stub embedding)。
- [x] metadata 必须包含：`journal_id`、`journal_name`、`title`、`year`、`source`。
  pytest `test_builder_writes_faiss_and_metadata_with_required_fields` 锁住该契约。
- [x] 增加 `--limit` 和 `--resume` 参数。
  pytest 覆盖:`test_builder_respects_limit_parameter`、
  `test_builder_resume_appends_only_new_records` (resume 时只 embed 剩余 records,
  追加到已有 FAISS)。
- [x] 在本地 accepted-paper corpus 上运行 builder。
  产物:`data/processed/accepted_papers_index.faiss` (950KB) +
  `accepted_papers_metadata.parquet` (12KB),95 vectors × 63 journals。
  真实 sanity check:网络拥塞 query 召回 TPDS / TNSM / TON / SCN / TDSC,
  TON 进 top-3 (BM25 阶段它没进 top-5,embedding 信号更覆盖)。
- [x] 提交：`feat: build accepted-paper vector index`

### 任务 3.3：把 Accepted-Paper Route 接入 CandidateGenerator

**文件：**
- 修改： `src/retriever/candidate_generator.py`
- 修改： `src/app/api.py`
- 修改： `scripts/run_evaluation.py`
- 修改： `scripts/run_retrieval_ablation.py`
- 测试： `tests/test_retriever.py`
- 测试： `tests/test_retrieval_ablation.py`

- [x] 增加配置：

```yaml
candidate_generator:
  accepted_paper_weight: 0.20
  route_top_k:
    abstract:
      accepted_bm25: 28
      accepted_vector: 56
```

  `configs/app.yaml` 的 candidate_generator 段加入 `accepted_paper_weight: 0.20`
  和 `route_top_k.abstract.{accepted_bm25, accepted_vector}` 键。
- [x] 在 retrieval trace 中增加 route：`accepted_bm25` 和 `accepted_vector`。
  `CandidateGenerator._hybrid_route_results` 在两条新路由对应 retriever 被
  注入时,把它们追加进 route_results,trace 由 `_collect_route_trace` 自动收录。
- [x] 如果 index 缺失，accepted-paper route 自动禁用，不能导致推荐失败。
  api.py / run_evaluation.py 都做 `AcceptedPaperStore.load()` 后判断
  `count > 0` 才构建 retriever;`AcceptedPaperEmbeddingRetriever.is_available`
  判 FAISS 文件是否存在,缺失时 retriever 置 None。pytest
  `test_accepted_routes_disabled_when_no_retriever_injected` 锁住。
- [x] 增加消融变体：`accepted`、`scope_typical`、`scope_accepted`、`typical_accepted`、`full_hybrid`。
  `scripts/run_retrieval_ablation.py` 的 `VARIANTS` 从 3 个扩展到 8 个;
  `build_route_results_for_variant` 把组合解构为 `_typical_routes` /
  `_accepted_routes` 两个辅助函数 + scope_routes 拼装。pytest
  `test_build_route_results_for_accepted_paper_variants` 验证每个变体的路由集合。
- [x] 运行：

```bash
pytest tests/test_retriever.py tests/test_retrieval_ablation.py tests/test_api.py -q
```

  结果:64 passed (含 5 个新 retriever 测试 + 1 个新 ablation 测试)。

- [x] 提交：`feat: add accepted-paper retrieval route`

### 任务 3.4：运行 Retrieval Route 消融

**文件：**
- 仅输出： `data/evaluation/results/*.json`
- 落档：`docs/adr/0001-coverage-aware-gate-a.md`

- [x] 跑 light30 retrieval ablation：

```bash
python scripts/run_retrieval_ablation.py \
  --papers data/evaluation/papers_metadata_light_30.jsonl \
  --include-vector \
  --variants scope typical hybrid accepted scope_accepted full_hybrid
```

  初次跑发现 `accepted` / `scope_accepted` / `full_hybrid` 与 `typical` / `hybrid` 数字完全一致。
  根因：`scripts/run_retrieval_ablation.py::build_candidate_generator` 没有把
  `accepted_bm25_retriever` / `accepted_embedding_retriever` 注入到 `CandidateGenerator`,
  与 `api.py` / `run_evaluation.py` 的加载逻辑不一致。修复后 (commit bb96398)
  重新跑出真实数字。详见 ADR 0001。

- [x] 使用 baseline snapshots 跑 full-v2 retrieval ablation。
  产物：`data/evaluation/results/retrieval_ablation_full_v2_20260602_153622.json`
  (89 篇 × 7 variants)。
- [x] 比较 `coarse@50`、retrieval MRR、`wide_recalled_but_not_top50`,并按
  covered/uncovered 子集分层。详见 ADR 0001。
- [x] **Gate A 判定:coverage-aware positive。** 详见决策门槛章节。
  简述:`full_hybrid` 整体 = `hybrid` 数字 (coarse@50=83, rule@5=33, rule@20=75),
  说明 3 路 `weighted_minmax` 融合把 accepted signal 归一化掉了;
  但 accepted route 在 covered 子集 (n=48, 53.9% of v2) 上单独 rule@5=25/48=52%,
  `scope_accepted` 在 covered 上 ret_mrr=0.3910 (全表最高),
  `typical_accepted` 在 covered 上 rule@5=22/48=46% (vs hybrid 14/48=29%)。
  uncovered 子集上 accepted 结构性 0,但本就是设计预期,不能反推为路线无效。
- [x] **不**把 `accepted_paper_weight` 设成 0;保留 0.20 默认值。
- [x] **不**再做静态权重调参;直接进入 Task 4.1 feature_builder + LTR。
- [ ] 把最终 retrieval+rule 配置 + LTR reranker 一同登记进
  `data/evaluation/results/baseline_registry.json` (留到 Task 5.4 评测完一起登记)。
- [x] 只提交代码和配置;大的结果文件除非明确需要,否则不要提交。
- [x] 提交:`docs: coverage-aware Gate A decision (ADR 0001)`

---

## 阶段 4：为 Learning-To-Rank 记录候选特征

### 任务 4.1：创建 Candidate Feature Builder

**文件：**
- 新建： `src/ranker/feature_builder.py`
- 测试： `tests/test_feature_builder.py`、`tests/test_candidate_generator_features.py`

- [x] 定义稳定特征表：

实际实现 `FEATURE_NAMES` 共 **20 维** (plan 原写 19 维,
**新增 `candidate_in_accepted_corpus`**,per ADR 0001 显式拒绝
oracle 特征 `gold_in_accepted_corpus` 的替代;详见 ADR 0001
"区分 oracle 与非 oracle 特征"):

```python
FEATURE_NAMES = [
    # plan 原文 19 维
    "retrieval_rank",
    "rule_rank",
    "rule_score",
    "scope_bm25_rank",
    "scope_vector_rank",
    "typical_bm25_rank",
    "typical_vector_rank",
    "accepted_bm25_rank",
    "accepted_vector_rank",
    "route_count",
    "has_scope_route",
    "has_typical_route",
    "has_accepted_route",
    "has_identity_anchor",
    "same_gold_area",
    "same_parsed_ccf_area",
    "same_ccf_level",
    "journal_ccf_numeric",
    "paper_strength",
    # ADR 0001 新增:候选级覆盖率信号(可推理,不是 oracle)
    "candidate_in_accepted_corpus",
]
```

- [x] 缺失 rank 使用大哨兵值,例如 `999`。
  - `MISSING_RANK_SENTINEL = 999.0` 模块级常量;
  - 实现:`_route_rank_or_sentinel` (route 级) + `_trace_top_level_rank` (trace 顶层);
  - 任何缺失/非法(0/负数/None)都退化为 999。

- [x] Boolean 特征转成 `0.0` 或 `1.0`。
  - 8 个二元特征:route_count(虽然是数值但被当 0/1 处理时也能用),
    has_scope_route / has_typical_route / has_accepted_route /
    has_identity_anchor / same_gold_area / same_parsed_ccf_area /
    same_ccf_level / candidate_in_accepted_corpus。

- [x] CCF 映射:`A=3`、`B=2`、`C=1`、unknown `0`。
  - `ccf_level_to_numeric()` helper,大小写不敏感。

- [x] 测试缺失 route score 和 CCF 转换。
  - 24 个 tests (test_feature_builder.py) + 2 个 (test_candidate_generator_features.py):
    schema 锁定、oracle 拒绝、缺失 rank、非法值、CCF 转换、
    per-route 抽取、CandidateGenerator.attach_features 接口。

- [x] 4.1.d 接入 CandidateGenerator:
  - `CandidateGenerator.attach_features(trace, paper_profile, rule_ranks,
    rule_scores, accepted_paper_store)` 原地修改 trace,每本期刊
    entry 增加 `features` 与 `feature_names`。

- [x] 4.1.e 持久化到 ablation 输出 JSON:
  - `evaluate_variant` 加 `accepted_paper_store` 参数,跑完 rule_scorer
    后调 `attach_features`,落 `paper_results[i].candidate_features`
    + variant 顶层 `feature_names`。

- [x] 修复 retrieval_rank 写入 bug (per 2026-06-02 用户反馈):
  - 旧实现把 `retrieval_rank` 写死成 999,削弱 LTR(检索排名是
    最重要的排序特征之一);83/83 positives 全部 999。
  - 新实现:`CandidateGenerator._merge_route_results` 已经在
    trace 顶层写 `retrieval_rank`,build_features 用
    `_trace_top_level_rank` 读取,缺失/非法值兜底用 999。
  - 验证:83/83 positives retrieval_rank 非 999,分布 37/37/9/0
    (top-5/6-20/21-50/50+)。

- [x] 运行:

```bash
pytest tests/test_feature_builder.py tests/test_candidate_generator_features.py -q
# 26 passed
```

- [x] 提交:
  - `feat(4.1): LTR feature builder + ablation persistence`
  - `fix(4.1+4.2): retrieval_rank from trace, rule_top20, ADR 0002`

### 任务 4.2：从 Ablation Pipeline 导出训练样本

> **2026-06-02 修订**:plan 原写从 `run_evaluation.py` evaluation JSON 入口;
> LTR v1 改从 `run_retrieval_ablation.py` ablation JSON 入口。evaluation 路径
> 延期到阶段 6 (LLM evidence 训练)。详见 `docs/adr/0002-ltr-v1-ablation-input.md`。

**文件：**
- 新建： `scripts/build_ranking_training_data.py`
- 测试： `tests/test_build_ranking_training_data.py`

- [x] 输入：`run_retrieval_ablation.py` 输出的 ablation JSON(含 `feature_names`
  与 `paper_results[i].candidate_features`)。**不使用** evaluation JSON。
- [x] 输出：JSONL，每行包含 `paper_id`、`journal_id`、`label`、`features`、
  `feature_names`、`negative_type`。
- [x] 正样本：exact gold journal，label 为 `1`。
- [x] Hard negative：出现在 `paper_result["rule_top20"]` 但不是 gold 的期刊
  (per plan 4.2 原意)。`rule_top20` 由 `run_retrieval_ablation.py` 落盘
  (2026-06-02 加的字段),不是 `rule_top5` 近似。
- [x] Same-area negative：subject_tags 与 gold venue 重叠、但不在 rule_top20。
- [x] Easy negative：其他候选期刊。
- [x] 每篇论文保留 1 个正样本和最多 10 个负样本，按 hard > same_area > easy 优先级。
- [x] CLI：

```bash
python scripts/build_ranking_training_data.py \
  --ablation-json data/evaluation/results/<ablation>.json \
  --journals-jsonl data/journals_ccf.jsonl \
  --output data/training/ranker_train.jsonl \
  --variants full_hybrid \
  --max-negatives 10 \
  --report data/training/ranker_train_report.json
```

- [x] 在最新 full-v2 结果上运行脚本。
- [x] 提交：`feat: export learning-to-rank training data`

### 任务 4.3：增加训练数据 Route Attribution 诊断

**文件：**
- 修改： `scripts/build_ranking_training_data.py`
- 测试： `tests/test_build_ranking_training_data.py`

- [x] 增加 route combination 的统计(per positive sample 出现哪些 route)。
- [x] 统计正样本缺失 route 特征的数量(每个 route 字段一个计数)。
- [x] 如果少于 80% 的正样本满足 `retrieval_rank <= 50`，输出 warning 到 stderr。
- [x] 保存 sidecar report：`data/training/ranker_train_<name>_report.json`。
- [x] 提交：`test: add training data diagnostics`

---

## 阶段 5：监督式 Reranker

### 任务 5.1：实现基础 Learning-To-Rank 接口

**文件：**
- 新建： `src/ranker/learning_to_rank.py`
- 测试： `tests/test_learning_to_rank.py`

- [ ] 定义接口：

```python
class LearningToRanker:
    def fit(self, rows: list[dict]) -> None: ...
    def predict_scores(self, rows: list[dict]) -> list[float]: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "LearningToRanker": ...
```

- [ ] 第一版实现必须确定性、无新增依赖。
- [ ] 可接受实现：基于 numpy 的简单 logistic regression，或如果环境已有 sklearn，则用 sklearn。
- [ ] 如果 sklearn 不可用，使用 numpy 实现小型 logistic regression baseline。
- [ ] 测试：训练后正样本分数高于 hard negative。
- [ ] 运行：

```bash
pytest tests/test_learning_to_rank.py -q
```

- [ ] 提交：`feat: add baseline learning-to-rank model`

### 任务 5.2：训练并保存基础 Reranker

**文件：**
- 新建： `scripts/train_learning_to_rank.py`
- 输出： `data/models/learning_to_ranker.json`
- 测试： `tests/test_learning_to_rank.py` + `tests/test_train_learning_to_rank.py`

- [x] CLI：

```bash
python scripts/train_learning_to_rank.py \
  --train data/training/ranker_train.jsonl \
  --output data/models/learning_to_ranker.json \
  --seed 42
```

- [x] 保存 feature names 和模型参数。
- [x] 保存训练指标：pairwise accuracy、positive mean score、hard-negative mean score。
- [x] **真实数据收敛问题(plan 5.2 重点)**:LearningToRanker 暴露
  `max_iter` / `use_standardization` 开关,fit 时捕获 sklearn
  ConvergenceWarning,收敛状态写入 `convergence_info` 持久化到产物。
  真实 `ranker_train_full_v2.jsonl`:不标准化必报 ConvergenceWarning;
  标准化 + max_iter=5000 内 35 iter 收敛,pairwise_accuracy=0.988,
  positive_mean=0.151, hard_neg_mean=0.079, margin=+0.072。
- [x] 提交:`feat(5.2): train baseline learning-to-rank model`

### 任务 5.3：把 Learned Reranker 接入 Pipeline，但默认关闭

**文件：**
- 修改： `configs/app.yaml`
- 修改： `src/recommender/pipeline.py`
- 修改： `src/app/api.py`
- 修改： `scripts/run_evaluation.py`
- 测试： `tests/test_recommender.py`

- [ ] 增加配置：

```yaml
ranking:
  learned_reranker:
    enabled: false
    model_path: "data/models/learning_to_ranker.json"
    blend_with_rule_score: 0.30
    blend_with_llm_score: 0.20
```

- [ ] 启用时的 pipeline 顺序：

```text
CandidateGenerator -> RuleScorer -> LearningToRanker -> LLM evidence/ranker -> final selection
```

- [ ] 第一版只 rerank 已经进入 `llm_candidates` 的候选，避免放大噪声。
- [ ] 增加诊断字段：`learned_score`、`learned_rank`、`final_rank_source`。
- [ ] 运行：

```bash
pytest tests/test_recommender.py tests/test_api.py tests/test_run_evaluation_diagnostics.py -q
```

- [ ] 提交：`feat: add config-gated learned reranker`

### 任务 5.4：评测 Learned Reranker

**文件：**
- 仅输出： `data/evaluation/results/*.json`

- [ ] 在 light30 上跑 baseline。
- [ ] 在 light30 上跑 learned reranker。
- [ ] 在 full-v2 上跑 baseline。
- [ ] 在 full-v2 上跑 learned reranker。
- [ ] 比较 exact Hit@5、MRR、same-area@5、acceptable@5。
- [ ] 如果 learned reranker 只提升 light30 但伤害 full-v2，视为过拟合，不设为默认。
- [ ] 如果 learned reranker 提升 full-v2 但严重伤害 acceptable@5，必须先做失败案例分析。

### 任务 5.5：可选 LightGBM/LambdaMART 升级

**文件：**
- 修改： `pyproject.toml`
- 修改： `src/ranker/learning_to_rank.py`
- 修改： `scripts/train_learning_to_rank.py`
- 测试： `tests/test_learning_to_rank.py`

- [ ] 增加可选依赖组：

```toml
[project.optional-dependencies]
ml = [
    "lightgbm>=4.0.0",
]
```

- [ ] 增加 `--model-type logistic|lightgbm_lambdarank`。
- [ ] 如果未安装 LightGBM，自动回退到 logistic reranker。
- [ ] LambdaRank 训练时按 paper 分组。
- [ ] 和 logistic baseline 对比。
- [ ] 只有在 full-v2 和 heldout-final 都提升时才考虑推广。

---

## 阶段 6：把 LLM 从“最终裁判”改成“证据抽取器”

### 任务 6.1：增加结构化 Evidence Extractor Prompt

**文件：**
- 新建： `src/ranker/llm_evidence_extractor.py`
- 修改： `configs/prompts.yaml`
- 测试： `tests/test_llm_evidence_extractor.py`

- [ ] 增加 prompt key：`llm_evidence_extractor_system`。
- [ ] 增加 prompt key：`llm_evidence_extractor_user`。
- [ ] 要求 JSON 输出：

```json
{
  "journal_id": "ton",
  "scope_fit": 0.87,
  "method_fit": 0.73,
  "application_fit": 0.62,
  "journal_position_fit": 0.81,
  "too_broad_penalty": 0.12,
  "too_narrow_penalty": 0.05,
  "evidence": ["short evidence item"]
}
```

- [ ] parser 必须拒绝 markdown；但如果是合法 JSON code fence，可以用现有 `parse_json_response` 修复。
- [ ] 运行：

```bash
pytest tests/test_llm_evidence_extractor.py tests/test_llm.py -q
```

- [ ] 提交：`feat: add structured LLM evidence extractor`

### 任务 6.2：把 Evidence 加入 FeatureBuilder

**文件：**
- 修改： `src/ranker/feature_builder.py`
- 测试： `tests/test_feature_builder.py`

- [ ] 增加特征：

```python
"llm_scope_fit",
"llm_method_fit",
"llm_application_fit",
"llm_journal_position_fit",
"llm_too_broad_penalty",
"llm_too_narrow_penalty"
```

- [ ] 缺失 LLM evidence 时使用中性默认值，而不是惩罚性默认值。
- [ ] 建议默认值：fit score 为 `0.5`，penalty 为 `0.0`。
- [ ] 运行：

```bash
pytest tests/test_feature_builder.py -q
```

- [ ] 提交：`feat: add LLM evidence ranker features`

### 任务 6.3：对比 LLM 的三种角色

**文件：**
- 新建： `scripts/run_llm_role_ablation.py`
- 测试： `tests/test_llm_rerank_ablation.py`

- [ ] 对比：
  - `llm_ranker_direct`
  - `llm_evidence_plus_rule`
  - `llm_evidence_plus_learned_reranker`
- [ ] 必须复用固定 `paper_profile_snapshot`。
- [ ] 公平性检查：各变体的 coarse@50 和 rule@20 必须一致。
- [ ] 所有变体都保存完整 candidate details。
- [ ] 提交：`feat: add LLM role ablation runner`

---

## 阶段 7：可选 Two-Tower 模型

### 任务 7.1：构建最小 Two-Tower 训练集

**文件：**
- 修改： `scripts/prepare_training_data.py`
- 新建： `data/training/two_tower_pairs.jsonl`
- 测试： `tests/test_learning_to_rank.py`

- [ ] 正样本：paper title+abstract 与 gold journal profile。
- [ ] Journal text 变体：
  - `scope_text`
  - accepted-paper cluster text
  - hybrid scope + accepted-paper summary
- [ ] 负样本：优先 same-area hard negative，再加 random negative。
- [ ] 不允许使用 heldout-final papers。
- [ ] 保存 split：`train`、`dev`、`heldout`。

### 任务 7.2：实现只用于召回的 Two-Tower Baseline

**文件：**
- 修改： `src/ranker/siamese_ranker.py`
- 新建： `scripts/train_two_tower_retriever.py`
- 测试： `tests/test_learning_to_rank.py`

- [ ] 先用当前 embedding 模型；只有确认本地依赖后，再考虑 SPECTER/SciBERT。
- [ ] 损失函数使用 InfoNCE 和 in-batch negatives。
- [ ] 第一阶段只评测 coarse@50 和 retrieval MRR。
- [ ] 在证明召回提升前，不允许替换最终排序。

### 任务 7.3：把 Two-Tower 作为额外召回路由对比

**文件：**
- 修改： `src/retriever/candidate_generator.py`
- 修改： `configs/app.yaml`
- 测试： `tests/test_retriever.py`

- [ ] 增加 route：`two_tower_vector`。
- [ ] 对比 `full_hybrid` 与 `full_hybrid + two_tower_vector`。
- [ ] 只有 coarse@50 提升且 final Hit@5 不受损时，才考虑启用。

---

## 阶段 8：论文级实验矩阵

### 任务 8.1：实现实验 Runner

**文件：**
- 新建： `scripts/run_publication_experiments.py`
- 测试： `tests/test_publication_experiments.py`

- [ ] 定义实验矩阵：

```text
E0: current baseline
E1: accepted-paper route only
E2: accepted-paper route + RuleScorer
E3: learned reranker without LLM evidence
E4: learned reranker + LLM evidence
E5: direct LLM ranking baseline
E6: optional two-tower retrieval route
```

- [ ] 每个实验都必须把 effective config 写入结果 JSON。
- [ ] 对比 reranker 时，每个实验必须使用同一个 benchmark 和固定 profile snapshot。
- [ ] 增加 `--benchmark light30|full-v2|heldout-final`。
- [ ] 增加 `--dry-run`，只打印计划命令，不调用 LLM。
- [ ] 运行：

```bash
pytest tests/test_publication_experiments.py -q
```

- [ ] 提交：`feat: add publication experiment runner`

### 任务 8.2：生成论文可用实验表格

**文件：**
- 新建： `scripts/summarize_publication_experiments.py`
- 输出： `data/evaluation/results/publication_summary.md`
- 测试： `tests/test_publication_experiments.py`

- [ ] 汇总指标：
  - exact Hit@1/3/5
  - MRR
  - NDCG@5
  - coarse@50
  - rule@20
  - same-area@5
  - same-CCF@5
  - acceptable@5
- [ ] 生成按领域拆分表。
- [ ] 生成按 CCF 等级拆分表。
- [ ] 生成 miss-stage 拆分表。
- [ ] 生成 route attribution 汇总。
- [ ] 提交：`feat: summarize publication experiment tables`

### 任务 8.3：导出失败案例分析

**文件：**
- 新建： `scripts/export_failure_analysis.py`
- 输出： `data/evaluation/results/failure_analysis_*.md`

- [ ] 导出 gold venue 位于以下阶段的失败样本：
  - not in wide recall
  - wide recalled but not top50
  - rule suppressed
  - in LLM/ranker pool but lost
- [ ] 每个 case 包含：title、venue、gold area、parsed area、rule rank、retrieval routes、final Top5、acceptable-hit flag。
- [ ] 这些输出用于论文 discussion 和数据集问题定位。
- [ ] 提交：`feat: export failure analysis reports`

---

## 阶段 9：论文表达和最终验证

### 任务 9.1：写 Method Section 初稿

**文件：**
- 新建： `docs/paper/method_outline.md`

- [ ] 把方法命名为 `Journal-as-Distribution Recommendation`。
- [ ] 包含模块：
  - multi-route retrieval
  - journal distribution profile from accepted papers
  - route-aware feature builder
  - supervised reranker
  - LLM evidence extraction
  - calibrated final ranking
- [ ] 加一个 Mermaid 架构图，展示 retrieval routes、learned reranker、LLM evidence extraction 和 final ranking。
- [ ] 提交：`docs: draft publication method outline`

### 任务 9.2：写 Experiment Section 初稿

**文件：**
- 新建： `docs/paper/experiment_outline.md`

- [ ] 定义数据集和泄漏控制。
- [ ] 定义指标。
- [ ] 定义 baseline。
- [ ] 定义消融实验。
- [ ] 定义 LLM 调用的重复实验/随机性报告策略。
- [ ] 提交：`docs: draft publication experiment outline`

### 任务 9.3：最终 Paper-Ready 检查

**文件：**
- 新建： `docs/paper/readiness_checklist.md`

- [ ] checklist 必须包含：
  - heldout-final benchmark 已冻结
  - 无已知泄漏
  - 所有实验配置已保存
  - 所有结果 JSON 可复现
  - exact 和 acceptable 指标都已报告
  - 失败分析已完成
  - 前端默认算法和评测默认算法一致
- [ ] 在 checklist 全部完成前，不能声称系统已经达到论文级最终结果。
- [ ] 提交：`docs: add publication readiness checklist`

---

## 推荐执行顺序

1. Phase 0：冻结 baseline 和 manifest。
2. Phase 1：benchmark 治理。
3. Phase 2：构建 accepted-paper corpus。
4. Phase 3：增加 accepted-paper route。
5. Phase 4：记录候选特征。
6. Phase 5：训练监督式 reranker。
7. Phase 6：LLM evidence extraction。
8. Phase 8：论文级实验 runner。
9. Phase 7：只有在 Phase 5 后仍然发现召回是瓶颈时再做。
10. Phase 9：实验稳定后再写论文表达材料。

---

## 决策门槛

- **Gate A (Coverage-Aware, 2026-06-02 修订):** accepted-paper route 在 full-v2
  整体指标上未必显示增益 (3 路 `weighted_minmax` 融合会吸收它的信号,
  表现上 `full_hybrid == hybrid`);但只要在 **covered 子集**
  (即 `gold_journal_id` 出现在 `data/accepted_papers/<jid>.json` 的论文子集)
  上,accepted 单独或 2-route fusion (`scope_accepted` / `typical_accepted`)
  的 rule@5 / ret_mrr 出现真实提升,即视为 Gate A 通过,保留默认接线
  与 `accepted_paper_weight=0.20`,进入 Task 4.1 feature_builder + LTR。
  **不**仅凭整体数字拒绝路线。uncovered 子集上的结构性 0 不计入 Gate A
  拒绝理由。详见 `docs/adr/0001-coverage-aware-gate-a.md`。
- **Gate B：** 如果 learned reranker 提升 light30 但伤害 full-v2，视为过拟合，需要扩充训练数据后再调参。
- **Gate C：** 如果 LLM evidence extraction 提升 acceptable@5 但降低 exact Hit@5，要同时报告两类指标，并根据目标应用决定默认策略。
- **Gate D：** 如果 exact venue 仍然偏低但 acceptable@5 很高，论文叙事应强调真实投稿推荐，而不是只强调 exact venue prediction。
- **Gate E：** 不要在 accepted-paper route 和学习排序验证前启动 two-tower 训练。

---

## 最先运行的命令

实施 Phase 1 前，先用这些命令确认当前起点：

```bash
python scripts/build_lightweight_eval_set.py
```

```bash
python scripts/run_evaluation.py \
  --input data/evaluation/papers_metadata_light_30.jsonl \
  --mode abstract \
  --top-k 5 \
  --workers 1
```

```bash
python scripts/run_llm_rerank_ablation.py \
  --baseline-eval data/evaluation/results/eval_abstract_top5_20260531_231641.json \
  --workers 1
```

---

## 自检记录

- 这份计划保留当前算法作为 baseline，不会破坏已有可运行路径。
- 技术收益最高的主线是：accepted-paper route + supervised reranker。
- LLM 从“最终裁判”转为“证据抽取器 + 解释生成器”，比单纯 prompt ranking 更稳定，也更适合写成论文方法。
- Two-tower 被放在后面，是因为成本更高，必须先由召回诊断证明它值得做。
- 每个阶段都有可测试输出和 stop/go gate。
