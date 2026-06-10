# Claude Code 指南：论文投稿期刊推荐系统

## 项目目标

本项目是一个面向计算机类论文投稿的期刊推荐系统。系统根据论文标题、摘要或全文，推荐适合投稿的期刊 Top N，并输出推荐理由、匹配标签、期刊基本信息和置信度。

当前长期目标不是继续堆 prompt，而是把现有“多路召回 + RuleScorer + LLM 精排”原型升级为可复现、数据驱动、可写成论文实验的方法。核心方向见：

- `docs/superpowers/plans/2026-06-01-publication-grade-journal-recommender-plan.md`

后续 Agent 必须优先维护评测口径和实验可复现性，再做模型或算法优化。

## 当前架构

推荐主流程：

```text
Paper Input
-> PaperParser 生成 PaperProfile
-> CandidateGenerator 多路候选召回
-> RuleScorer 可解释规则排序
-> LLMRanker 当前 baseline 精排
-> Recommendation 输出
```

主要模块：

- `src/app/api.py`：FastAPI 接口和前端 SSE 流式推荐。
- `src/recommender/pipeline.py`：推荐流程编排。
- `src/retriever/candidate_generator.py`：scope、typical abstracts、identity anchor 等候选召回融合。
- `src/ranker/rule_scorer.py`：规则排序和可解释特征。
- `src/ranker/llm_ranker.py`：当前直接 LLM ranking baseline。
- `scripts/run_evaluation.py`：正式评测入口。
- `src/evaluation/benchmark_manifest.py`：评测 manifest/hash。
- `src/evaluation/clean_benchmark.py`：泄漏检测和 clean typical snapshot。

## 当前默认配置原则

所有线上前端和评测默认都应从 `configs/app.yaml` 读取同一套算法配置。不要在 API、脚本或测试里私自硬编码另一套模型/权重。

当前推荐使用：

```yaml
minimax:
  model: "MiniMax-M2.7"
  temperature: 0.2

ollama:
  embedding_model: "qwen3-embedding:4b"
  timeout_seconds: 180

candidate_generator:
  retrieval_target: "typical_abstracts"
  fusion_strategy: "weighted_minmax"
  hybrid_scope_weight: 0.65
  hybrid_typical_weight: 0.35
  identity_anchor_weight: 0.10
  route_top_k:
    abstract:
      bm25: 28
      vector: 56
      text: 14

ranking:
  llm_ranker_timeout_seconds: 420
```

说明：

- `retrieval_target: "typical_abstracts"` 在当前代码中不是纯 typical，而是 hybrid：`scope + typical + identity_anchor`。
- M2.7 是当前默认开发/前端/baseline 模型。M3 只能作为 LLM ablation，不要默认切换。
- `temperature` 必须通过 `src/utils/llm_config.py` 生效。改配置后需要重启后端，前端缓存的 pipeline 才会更新。

## 实验纪律

每次正式实验必须保存并检查：

- `benchmark_manifest`
- `app_config_hash`
- `prompt_hash`
- `minimax_model`
- `embedding_model`
- benchmark 输入文件
- 是否 clean benchmark
- 是否复用 `paper_profile_snapshot`

不要把没有 manifest 的历史结果作为正式 baseline。

### 快速评测

Light30 是正式快速评测集，包含 10 个 CCF 领域 × A/B/C 三档各 1 篇，共 30 篇。

校验 light30：

```bash
python scripts/build_lightweight_eval_set.py --validate-only
```

运行 light30：

```bash
python scripts/run_evaluation.py \
  --benchmark-profile light30 \
  --mode abstract \
  --top-k 5 \
  --workers 1
```

运行 full-v2：

```bash
python scripts/run_evaluation.py \
  --benchmark-profile full-v2 \
  --mode abstract \
  --top-k 5 \
  --workers 1
```

`--workers 1` 更适合 baseline，因为并发 LLM 请求容易造成限流、超时和格式错误。调参探索可以提高 workers，但结果不宜直接作为正式 baseline。

### Baseline Registry

登记 baseline：

```bash
python scripts/register_baseline_result.py \
  --result data/evaluation/results/<eval_result>.json \
  --label <unique_label>
```

登记文件：

- `data/evaluation/results/baseline_registry.json`

当前已登记：

- `light30_m27_default`：默认 M2.7 baseline。
- `light30_m3_llm_ablation`：M3 直接 LLM ranking 消融。

重复 label 默认拒绝；只有确认覆盖时使用 `--replace`。

## Benchmark 和泄漏检测

正式实验前必须检查测试集泄漏，尤其是 typical abstracts 和未来 accepted-paper profiles。

运行泄漏检测：

```bash
python scripts/clean_benchmark.py \
  --input data/evaluation/papers_metadata_light_30.jsonl \
  --typical-dir data/typical_abstracts \
  --accepted-paper-dir data/accepted_papers \
  --report data/evaluation/results/light30_leakage_report.json
```

规则：

- title 使用规范化精确匹配。
- abstract 使用至少 160 字符片段匹配。
- 报告中的 `source_type` 必须区分 `typical_abstract` 和 `accepted_paper`。
- clean snapshot 只清理 typical abstracts；accepted-paper profiles 当前只 report-only。

## 当前已完成的计划阶段

已完成：

- 任务 0.1：评测结果保存 `benchmark_manifest`。
- 任务 0.2：`baseline_registry.json` 和登记脚本。
- 任务 1.1：Light30 正式快速评测集和 `--benchmark-profile`。
- 任务 1.2：typical/accepted-paper 泄漏检测。

下一步：

- 任务 1.3：制定 held-out benchmark 使用规则。
- 阶段 2：构建真实已发表论文期刊画像 accepted-paper corpus。
- 阶段 3：接入 accepted-paper retrieval route。
- 阶段 4/5：记录候选特征并训练 supervised learning-to-rank reranker。
- 阶段 6：把 LLM 从“最终裁判”改为“结构化证据抽取器”。

## 未来算法方向

不要继续把主要优化押在 LLM prompt ranking 上。当前结果显示大量失败是 `in_llm_but_lost`，说明直接让 LLM 决定 Top5 不稳定。

计划中的论文级方法是：

```text
多路召回
-> RuleScorer
-> accepted-paper route
-> feature_builder 记录 paper-candidate 特征
-> supervised learning-to-rank
-> LLM evidence extraction
-> 可复现 final ranker 融合排序
```

LLM 后续应主要输出结构化证据，例如：

- scope 是否匹配
- 论文类型是否匹配
- 期刊定位宽窄是否匹配
- 是否存在过度泛化推荐
- 是否存在明显 mismatch

这些证据进入 LTR/ranker，而不是让 LLM 单独决定最终 Top5。

## 代码修改规则

- 修改算法前先看 `docs/superpowers/plans/2026-06-01-publication-grade-journal-recommender-plan.md`。
- 新功能或 bugfix 使用测试先行；至少补一个能锁住行为的 pytest。
- 不要改动无关数据文件和历史结果文件。
- 不要删除已有 baseline、历史 evaluation JSON 或 registry 记录。
- 不要把测试集论文复制进 typical abstracts 或 accepted-paper profiles。
- 不要在没有 clean/leakage 报告的情况下声称结果公平。
- 不要把 M3 结果默认替代 M2.7；M3 是 ablation。

## 常用验证命令

```bash
pytest tests/test_benchmark_manifest.py tests/test_run_evaluation_diagnostics.py -q
pytest tests/test_lightweight_eval_set.py -q
pytest tests/test_clean_benchmark.py -q
pytest tests/test_embedding.py tests/test_ranker.py tests/test_api.py tests/test_llm_config.py -q
```

在完成一个计划任务后，优先跑该任务涉及的最小测试集合，再跑相关评测或校验脚本。

## Agent Skills 和项目管理

Issue tracker：

- GitHub Issues：`Qianggggggggg/journal-recommender`
- 说明：`docs/agents/issue-tracker.md`

Triage labels：

- `needs-triage`
- `needs-info`
- `ready-for-agent`
- `ready-for-human`
- `wontfix`

Domain docs：

- 如果存在 `CONTEXT.md` 或 `docs/adr/`，改架构前先读。
- 如果不存在，静默继续，不要为了没有文档而停工。

执行计划文档时：

- 使用 `superpowers:executing-plans` 或 `superpowers:subagent-driven-development`。
- 每个任务按 checkbox 逐项完成。
- 完成后更新计划状态，说明测试和产物。
