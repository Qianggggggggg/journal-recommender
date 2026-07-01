# Accepted Corpus LTR 训练数据修复设计

> **日期**: 2026-06-26
> **目标**: 把 4 个 dead 特征 (same_gold_area / same_parsed_ccf_area / same_ccf_level / candidate_in_accepted_corpus) 接通训练数据；删 paper_strength；用扩大后的 accepted_paper corpus 重训
> **范围**: `feature_builder.py` + `build_ranking_training_data.py` + `train_learning_to_rank.py` + `configs/app.yaml`
> **非目标**: 改 ablation 跑法、改 pipeline 端到端、保留旧模型

---

## 一、问题定义

### 1.1 训练数据里 5 个 dead 特征全是 0

`learning_to_ranker_balanced_v4_lr.json` 训练时（`ranker_train_v4_26dim.jsonl` 6/15 22:26），5 个特征在 **全部 7236 行** 都是 0.0：

| 特征 idx | 名称 | v4_26dim 训练数据 | 实际意义 |
|---------|------|------------------|---------|
| f14 | `same_gold_area` | 0.0 (0/7236) | paper.research_area ∩ gold.subject_tags |
| f15 | `same_parsed_ccf_area` | 0.0 (0/7236) | paper.ccf_research_area ∩ gold.subject_tags |
| f16 | `same_ccf_level` | 0.0 (0/7236) | paper gold CCF 等级 == journal CCF 等级 |
| f18 | `paper_strength` | 0.0 (0/7236) | quality_assessor LLM 评分 |
| f19 | `candidate_in_accepted_corpus` | 0.0 (0/7236) | 候选期刊在 accepted_paper corpus 里有真实发表论文 |

`feature_builder.py` 已经有 `gold_journal` / `paper_ccf_target_level` kwargs 接通 dead 特征（line 258-305, 388-426），但 `build_ranking_training_data.py:218/249` 直接从 ablation JSON 读预存的 `candidate_features` dict，**不**调 `attach_features_to_trace` —— 所以即使下游修好了，上游 ablation 跑的时候没传 gold_journal / accepted_paper_store，5 个特征就是 0。

### 1.2 accepted_paper corpus 6/24 大幅扩充

`data/accepted_papers/*.json` 大部分文件时间戳 6/24 17:12（`acta.json` / `aamas.json` / ...），与训练数据生成时间（6/15）相差 9 天。`scripts/build_ranking_training_data.py` 全文 grep 不到 `AcceptedPaperStore` 或 `accepted_paper_store` —— 训练数据生成**完全没**用到扩大后的 corpus。

### 1.3 决策

- **删 `paper_strength`**：需要调 LLM，540 篇 × 3 retry ≈ 1-2h 成本且需要新的 quality cache sidecar。dead 特征里它最弱（paper_strength 只代表 paper 本身质量，跟 candidate relevance 关系弱）。从 schema 删除而不是 0 占位，避免未来误用。
- **保留其他 4 个 dead 特征**：直接连通，不需要 LLM。

---

## 二、目标 Schema

### 2.1 新 dim

| Schema | 旧 dim | 新 dim | 删的特征 |
|--------|--------|--------|---------|
| `FEATURE_NAMES` (base) | 20 | 19 | paper_strength |
| `FEATURE_NAMES_WITH_LLM_EVIDENCE` | 26 | 25 | paper_strength |
| `FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY` | 28 | 27 | paper_strength |

### 2.2 `paper_strength` 删除影响

- `feature_builder.py:65-66, 150, 333, 401, 422, 425, 430`：删 `paper_strength` 字段、参数、赋值
- `ltr_adapter.py:_FEATURE_SCHEMA_BY_DIM`：自动适配（按 `expected_dim` 查表，不用手动改）
- `learning_to_rank.py`：feature_dim 自适应，不用改
- **所有现有模型**（`v4_lr.json` / 3 个 lightgbm）都是 26/28-dim，retrain 完成后**全部移到 `_archive_20260626_pre_accepted_corpus/`**

---

## 三、实施步骤

### 3.1 删 paper_strength（feature_builder.py）

文件：`src/ranker/feature_builder.py`

- 删 `FEATURE_NAMES` / `FEATURE_NAMES_WITH_LLM_EVIDENCE` / `FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY` 三个常量里的 `"paper_strength"` 字符串
- 删 `PaperCandidateFeatures` dataclass 的 `paper_strength: float = 0.0` 字段
- 删 `build_features()` 里的 `paper_strength=float(paper_profile.paper_strength) if paper_profile.paper_strength is not None else 0.0,` 赋值
- 删 `attach_features_to_trace()` 里无关代码（如果 `paper_profile.paper_strength` 没用到的就保留）
- 验证 `len(FEATURE_NAMES) == 19` / `len(FEATURE_NAMES_WITH_LLM_EVIDENCE) == 25` / `len(FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY) == 27`

### 3.2 改 build_ranking_training_data.py

文件：`scripts/build_ranking_training_data.py`

**main() 新增 CLI 参数**：
- `--accepted-corpus-dir` (default: `data/accepted_papers`)
- 加载 `AcceptedPaperStore`，构造 `accepted_jid_set = set(store._by_journal.keys())` 或新加 `get_journal_ids()` 方法
- 读 `journals.jsonl` 时构造 `journals_by_id_full: Dict[str, dict]`（带 `subject_tags` 和 `ccf_rating`）
- 从 `papers_by_title[paper_id].research_area` 读 paper.research_area（已有逻辑 line 230-236）

**`build_training_rows()` 签名扩展**：
- 加 `accepted_jid_set: Set[str]`
- 加 `journals_by_id_full: Dict[str, dict]`（与已有 `journals_by_id` 区分：前者带 `subject_tags` / `ccf_rating`，后者可能不完整）

**`_row()` 闭包覆写 4 个 dead 特征**：
- 找 `feature_names` 里 4 个 dead 特征的 idx（用 `feature_names.index("same_gold_area")` 动态查）
- 计算：
  ```python
  gold_journal_meta = journals_by_id_full.get(target_jid) or {}
  gold_subject_tags = set(gold_journal_meta.get("subject_tags") or [])
  paper_research_area = set(paper_meta.get("research_area") or [])
  paper_ccf_research_area = set(paper_meta.get("ccf_research_area") or [])
  paper_ccf_target_level = (gold_journal_meta.get("ccf_rating") or "").upper()
  candidate_meta = journals_by_id_full.get(jid) or {}

  same_gold_area = 1.0 if (paper_research_area & gold_subject_tags) else 0.0
  same_parsed_ccf_area = 1.0 if (paper_ccf_research_area & gold_subject_tags) else 0.0
  same_ccf_level = 1.0 if (
      paper_ccf_target_level
      and (candidate_meta.get("ccf_rating") or "").upper() == paper_ccf_target_level
  ) else 0.0
  candidate_in_accepted_corpus = 1.0 if jid in accepted_jid_set else 0.0
  ```
- 覆写 `feats[14] / feats[15] / feats[16] / feats[19]`（旧 schema 的 idx；新 schema 用动态 idx 查表）
- `feats[18]` 删掉（paper_strength 位置 trim）

**`_row()` 长度对齐**：
- 旧 base 20-dim → 新 base 19-dim：trim 掉 idx 18 即可
- evidence 仍然是 6-dim，拼接后总 25-dim
- 28-dim schema (FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY) = 19 base + 6 evidence + 2 tier/area = 27-dim

**Sidecar report 加 4 个 dead 特征 nonzero 计数**：
- `n_rows_same_gold_area_1`
- `n_rows_same_parsed_ccf_area_1`
- `n_rows_same_ccf_level_1`
- `n_rows_candidate_in_accepted_corpus_1`

### 3.3 验证 train_learning_to_rank.py 自适应

文件：`scripts/train_learning_to_rank.py`

- 读 `feature_names` 长度自适应（应该已经支持，确认即可）

### 3.4 更新 configs/app.yaml

```yaml
learned_reranker:
  model_path: "data/models/learning_to_ranker_balanced_v5_25dim_lr.json"
```

旧模型移到 `data/models/_archive_20260626_pre_accepted_corpus/learning_to_ranker_balanced_v4_lr.json`。

### 3.5 数据生成 + 重训

**Step 1: 生成 25-dim 训练数据**：
```bash
python scripts/build_ranking_training_data.py \
  --ablation-json data/evaluation/results/retrieval_ablation_540_balanced_v4.json \
  --journals-jsonl data/journals/journals_normalized.jsonl \
  --papers-jsonl data/papers/papers_metadata_540.jsonl \
  --evidence-snapshot data/evaluation/evidence/balanced_540_90v3_evidence.json \
  --accepted-corpus-dir data/accepted_papers \
  --max-negatives 10 \
  --output data/training/ranker_train_balanced_540_v5_25dim.jsonl \
  --report data/training/ranker_train_balanced_540_v5_25dim_report.json
```

**Step 2: 检查 sidecar 报告**：
- 4 个 dead 特征 nonzero count > 0
- `same_gold_area=1.0` count > 200
- `candidate_in_accepted_corpus=1.0` count > 50

**Step 3: 训练新 LR 模型**：
```bash
python scripts/train_learning_to_rank.py \
  --train data/training/ranker_train_balanced_540_v5_25dim.jsonl \
  --output data/models/learning_to_ranker_balanced_v5_25dim_lr.json \
  --model-type logistic_regression
```

**Step 4: Pipeline smoke test**（用 `--baseline-eval` 固定 profile snapshot）：
```bash
python scripts/run_evaluation.py \
  --benchmark-profile holdout240 \
  --mode abstract --top-k 5 --workers 1 \
  --baseline-eval data/evaluation/results/eval_holdout240_ltr_weight_020_20260622.json \
  --output data/evaluation/results/eval_holdout240_v5_25dim_lr.json
```

### 3.6 测试 (TDD)

**`tests/test_feature_builder.py`**：
- 验证 3 个 schema 都不含 `paper_strength`（确保删除干净）
- 验证 `len(FEATURE_NAMES) == 19` / `len(FEATURE_NAMES_WITH_LLM_EVIDENCE) == 25` / `len(FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY) == 27`

**`tests/test_build_ranking_training_data.py`**（新文件）：
- 构造 mock ablation data + mock AcceptedPaperStore + mock journals_by_id_full + mock papers_by_title
- 验证 `_row` 覆写 same_gold_area=1.0 when paper_research_area ∩ gold.subject_tags ≠ ∅
- 验证 `_row` 覆写 same_parsed_ccf_area=1.0 同理
- 验证 `_row` 覆写 same_ccf_level=1.0 when paper_ccf_target_level matches
- 验证 `_row` 覆写 candidate_in_accepted_corpus=1.0 when jid in accepted_jid_set
- 端到端 fixture：1 paper + 3 candidates，验证 4 个 dead 特征 nonzero，paper_strength 已从 schema 删除
- 验证 base features 长度 = 19（不是 20）

**`tests/test_ltr_adapter.py`**：
- 验证 `_FEATURE_SCHEMA_BY_DIM` 包含 19/25/27 的新 lookup

---

## 四、关键文件

| 文件 | 改动 |
|------|------|
| `src/ranker/feature_builder.py` | 3.1 — 删 paper_strength |
| `scripts/build_ranking_training_data.py` | 3.2 — 加 accepted corpus + 4 dead 特征覆写 |
| `scripts/train_learning_to_rank.py` | 3.3 — 验证自适应 |
| `configs/app.yaml` | 3.4 — model_path 换新 |
| `tests/test_feature_builder.py` | 3.6 — schema 长度测试 |
| `tests/test_build_ranking_training_data.py` | 3.6 — 新建测试文件 |
| `tests/test_ltr_adapter.py` | 3.6 — schema lookup 测试 |

**保留不动**：
- `src/recommender/pipeline.py`（已通过 ltr_adapter 自动适配新 dim）
- `src/ranker/ltr_adapter.py`（除 `_FEATURE_SCHEMA_BY_DIM` 已被自动 lookup 外无变化）
- `src/ranker/learning_to_rank.py`（已自适应）
- `scripts/eval_ltr_on_holdout.py`（已自适应）
- `data/accepted_papers/*.json`（只读不写）
- `data/evaluation/results/retrieval_ablation_540_balanced_v4.json`（不重跑）

---

## 五、接受条件

### 5.1 数据生成阶段
- ✅ Sidecar 报告 4 个 dead 特征 nonzero count > 0
- ✅ `same_gold_area=1.0` 数量 > 200
- ✅ `candidate_in_accepted_corpus=1.0` 数量 > 50
- ✅ 输出 JSONL 每行 features 长度 = 25（25-dim with evidence）

### 5.2 模型训练阶段
- ✅ `learning_to_ranker_balanced_v5_25dim_lr.json` `feature_dim=25`, `model_type=logistic_regression`
- ✅ 4 个 dead 特征的 coef ≠ 0（说明模型真的学到了）
- ✅ `pairwise_accuracy ≥ 0.96`

### 5.3 Pipeline 阶段
- ✅ `eval_holdout240_v5_25dim_lr.json` hit@5 ≥ 158/240（不 regression）
- ✅ `area_hit_5` ≥ 220（baseline 220 +2）
- ✅ MRR 持平或上升

### 5.4 测试
- ✅ `pytest tests/test_feature_builder.py tests/test_build_ranking_training_data.py tests/test_ltr_adapter.py -v` 全 pass
- ✅ `pytest tests/ -v` 无 regression

---

## 六、风险与缓解

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| accepted corpus 扩大后基线变了，pipeline hit@5 regression | 中 | 接受条件未达 → 旧模型（v4_lr）已在 archive，可快速回滚 yaml |
| 25-dim 跟现有 lgb Booster 训练路径不兼容 | 低 | lgb backend 训练时 feature_dim 自适应；只 train LR 不再训 lightgbm |
| ablation JSON 是 6/15 跑，accepted_papers 是 6/24 扩 | 低 | 4 个 dead 特征（特别是 `candidate_in_accepted_corpus`）用新 corpus 计算；retrieval signals (accepted_bm25/vector_rank) 来自旧 ablation，两套不矛盾 |
| `papers_by_title` 没有 `ccf_research_area` 字段 | 中 | fallback 到 `research_area`（与 feature_builder 内部逻辑一致） |
| `journals_by_id_full` 缺 `subject_tags` | 低 | fallback 到空集，`same_gold_area=0.0`（与现有 dead 行为一致） |
| 训练时 gold_journal lookup 失败（gold jid 不在 journals_by_id_full） | 低 | log warning，`same_gold_area=0.0`（与现有 dead 行为一致） |

---

## 七、回滚预案

1. `configs/app.yaml` 改回 `model_path: "data/models/_archive_20260626_pre_accepted_corpus/learning_to_ranker_balanced_v4_lr.json"`
2. 重启后端
3. 跑 `--baseline-eval` smoke test 验证 hit@5 = 158 恢复
4. 已生成的 v5 训练数据 + 模型保留为 ablation artifact

---

## 八、工作量估算

| 步骤 | 时间 |
|------|------|
| 3.1 删 paper_strength + 测试 | 1h |
| 3.2 改 build script + 测试 | 3h |
| 3.5 数据生成 + 重训 + smoke | 30min |
| 调试 + 验证接受条件 | 1.5h |
| **总计** | **~6h** |
