# Accepted Corpus LTR 训练数据修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 4 个 dead 特征（same_gold_area / same_parsed_ccf_area / same_ccf_level / candidate_in_accepted_corpus）接通训练数据，删 `paper_strength`，用扩大后的 `accepted_paper` corpus 重训出能利用 accepted-corpus 信号的 LTR 模型。

**Architecture:** 训练数据生成时**不**再读 ablation JSON 里预存的 0.0 死特征，而是在 `build_training_rows._row()` 闭包里**重算** 4 个 dead 特征（用 AcceptedPaperStore + Journal metadata + papers metadata）。`feature_builder.py` schema 从 20/26/28 维降到 19/25/27 维。`configs/app.yaml` `model_path` 切到新 25-dim LR 模型。旧模型移到 archive。

**Tech Stack:** Python 3.11, sklearn LogisticRegression, AcceptedPaperStore, JSONL training data, sidecar report JSON.

**Spec:** `docs/superpowers/specs/2026-06-26-accepted-corpus-ltr-design.md`

---

## File Structure

| 文件 | 角色 | 改动 |
|------|------|------|
| `src/ranker/feature_builder.py` | 锁定 schema 定义 + `PaperCandidateFeatures` dataclass + `build_features()` | 删 `paper_strength` |
| `scripts/build_ranking_training_data.py` | LTR 训练数据生成入口 | 加 `--accepted-corpus-dir`，改 `build_training_rows` + `_row` + sidecar report |
| `scripts/train_learning_to_rank.py` | LTR 模型训练（已有，自适应） | 不改（仅验证） |
| `configs/app.yaml` | LTR 部署配置 | `model_path` 切到 v5 25-dim 模型 |
| `tests/test_feature_builder.py` | schema 测试 | 加 3 个新 dim 长度测试 |
| `tests/test_build_ranking_training_data.py` | 新建 | 4 dead 特征 + base dim 测试 |
| `tests/test_ltr_adapter.py` | schema lookup 测试 | 加 19/25/27 lookup 测试 |
| `data/training/ranker_train_balanced_540_v5_25dim.jsonl` | 新建 | 25-dim 训练数据 |
| `data/training/ranker_train_balanced_540_v5_25dim_report.json` | 新建 | sidecar report |
| `data/models/learning_to_ranker_balanced_v5_25dim_lr.json` | 新建 | 新 LR 模型 |
| `data/models/_archive_20260626_pre_accepted_corpus/learning_to_ranker_balanced_v4_lr.json` | 移动 | 旧模型归档 |
| `data/evaluation/results/eval_holdout240_v5_25dim_lr.json` | 新建 | smoke test 结果 |

---

## Task 1: 测试 feature_builder.py schema 变化 (TDD)

**Files:**
- Modify: `tests/test_feature_builder.py`

- [ ] **Step 1: 读现有 test_feature_builder.py 看测试风格**

Run: `ls tests/test_feature_builder.py && wc -l tests/test_feature_builder.py`
Expected: 文件存在，能看到现有测试 pattern。

- [ ] **Step 2: 在 tests/test_feature_builder.py 末尾加 3 个新 dim 长度测试**

```python
def test_feature_names_excludes_paper_strength():
    """2026-06-26: paper_strength removed from schema (5 dead features → 4)."""
    from src.ranker.feature_builder import FEATURE_NAMES
    assert "paper_strength" not in FEATURE_NAMES


def test_feature_names_19_dim():
    """2026-06-26: base schema is 19-dim (was 20)."""
    from src.ranker.feature_builder import FEATURE_NAMES
    assert len(FEATURE_NAMES) == 19
    assert len(set(FEATURE_NAMES)) == 19  # 无重复


def test_feature_names_with_llm_evidence_25_dim():
    """2026-06-26: 19 base + 6 evidence = 25-dim (was 26)."""
    from src.ranker.feature_builder import FEATURE_NAMES_WITH_LLM_EVIDENCE
    assert len(FEATURE_NAMES_WITH_LLM_EVIDENCE) == 25
    assert "paper_strength" not in FEATURE_NAMES_WITH_LLM_EVIDENCE


def test_feature_names_with_tier_and_exclusivity_27_dim():
    """2026-06-26: 25 + 2 tier/area = 27-dim (was 28)."""
    from src.ranker.feature_builder import FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY
    assert len(FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY) == 27
    assert "paper_strength" not in FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY


def test_paper_candidate_features_no_paper_strength():
    """2026-06-26: PaperCandidateFeatures dataclass no longer has paper_strength field."""
    from src.ranker.feature_builder import PaperCandidateFeatures
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(PaperCandidateFeatures)}
    assert "paper_strength" not in field_names
```

- [ ] **Step 3: 运行新测试，验证 FAIL**

Run: `cd /Users/qian/PycharmProjects/paper && pytest tests/test_feature_builder.py -v -k "paper_strength or 19_dim or 25_dim or 27_dim or excludes_paper"`
Expected: 5 个 FAIL（"paper_strength" 还在 schema 里，dim 是 20/26/28 不是 19/25/27）

- [ ] **Step 4: 提交测试（TDD 顺序：先 failing test）**

```bash
cd /Users/qian/PycharmProjects/paper
git add tests/test_feature_builder.py
git commit -m "test(7.1): add failing tests for paper_strength removal + new dim (19/25/27)"
```

---

## Task 2: 从 feature_builder.py 删 paper_strength

**Files:**
- Modify: `src/ranker/feature_builder.py:46-69, 122-161, 333`

- [ ] **Step 1: 从 `FEATURE_NAMES` 删 `"paper_strength"`**

Edit `src/ranker/feature_builder.py:46-69`. 把：

```python
FEATURE_NAMES: List[str] = [
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
    "paper_strength",       # ← 删这一行
    "candidate_in_accepted_corpus",
]
```

改为：

```python
FEATURE_NAMES: List[str] = [
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
    "candidate_in_accepted_corpus",
]
```

- [ ] **Step 2: 更新 schema 头注释**

Edit `src/ranker/feature_builder.py:6-12`. 把：

```python
- ``FEATURE_NAMES`` 是**锁定**的 20 维基础 schema,顺序与名字都不能改,
  改了会破坏已保存的训练向量。
- ``FEATURE_NAMES_WITH_LLM_EVIDENCE`` 是阶段 6.2 的显式 26 维 schema;
  只有 evidence 实验和对应新模型可以消费。
```

改为：

```python
- ``FEATURE_NAMES`` 是**锁定**的 19 维基础 schema,顺序与名字都不能改,
  改了会破坏已保存的训练向量。
  2026-06-26: 从 20 维降到 19 维,删 ``paper_strength`` (dead feature,
  训练时全部 0.0,无 oracle-quality 信号)。所有现有 20/26/28 维模型
  不再兼容,需要 retrain 19/25/27 维新模型。
- ``FEATURE_NAMES_WITH_LLM_EVIDENCE`` 是阶段 6.2 的显式 25 维 schema
  (2026-06-26 调整: 19 base + 6 evidence,旧 26 维模型需要 retrain)。
```

- [ ] **Step 3: 从 `PaperCandidateFeatures` dataclass 删 `paper_strength` 字段**

Edit `src/ranker/feature_builder.py:150`。删这一行：

```python
    paper_strength: float = 0.0
```

- [ ] **Step 4: 从 `build_features()` 返回值删 `paper_strength=...` 赋值**

Edit `src/ranker/feature_builder.py:333`。删这一行：

```python
        paper_strength=float(paper_profile.paper_strength) if paper_profile.paper_strength is not None else 0.0,
```

- [ ] **Step 5: 运行测试验证 PASS**

Run: `cd /Users/qian/PycharmProjects/paper && pytest tests/test_feature_builder.py -v`
Expected: Task 1 加的 5 个新测试 PASS；旧测试不 regression。

- [ ] **Step 6: 跑完整 feature_builder 测试集**

Run: `cd /Users/qian/PycharmProjects/paper && pytest tests/test_feature_builder.py -v`
Expected: All pass.

- [ ] **Step 7: 提交**

```bash
cd /Users/qian/PycharmProjects/paper
git add src/ranker/feature_builder.py
git commit -m "feat(7.2): remove paper_strength from feature schema (20/26/28 → 19/25/27 dim)"
```

---

## Task 3: 测试 build_training_rows 重算 4 个 dead 特征 (TDD)

**Files:**
- Create: `tests/test_build_ranking_training_data.py`

- [ ] **Step 1: 创建测试文件**

```python
"""Tests for scripts/build_ranking_training_data.py dead-feature augmentation.

2026-06-26: 4 dead features (same_gold_area, same_parsed_ccf_area,
same_ccf_level, candidate_in_accepted_corpus) are now computed in
build_training_rows._row() instead of being read as 0.0 from the
ablation JSON's pre-stored candidate_features.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Set

import pytest

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from scripts.build_ranking_training_data import build_training_rows


# ---------- fixtures ----------

def _make_journal_meta(jid: str, subject_tags: list, ccf_rating: str) -> dict:
    return {
        "journal_id": jid,
        "journal_name": jid,
        "subject_tags": subject_tags,
        "ccf_rating": ccf_rating,
    }


def _make_paper_meta(title: str, research_area: list, ccf_research_area: list = None) -> dict:
    return {
        "title": title,
        "research_area": research_area,
        "ccf_research_area": ccf_research_area or research_area,
    }


def _make_ablation_entry(
    title: str, target_jid: str, candidate_jids: list, candidate_features: dict
) -> dict:
    return {
        "title": title,
        "venue": "TestVenue",
        "target_journal_id": target_jid,
        "retrieval_rank": 1,
        "rule_top20": candidate_jids,
        "candidate_features": candidate_features,
    }


# 19-dim base features (no paper_strength).
# 0:retrieval_rank, 1:rule_rank, 2:rule_score,
# 3-8: scope/typical/accepted bm25/vector rank
# 9-13: route_count + has_*_route + has_identity_anchor
# 14:same_gold_area, 15:same_parsed_ccf_area, 16:same_ccf_level,
# 17:journal_ccf_numeric, 18:candidate_in_accepted_corpus
BASE_19_DIM_DEFAULTS = [999.0] * 19  # 哨兵填充
BASE_19_DIM_DEFAULTS[0] = 1.0  # retrieval_rank=1
BASE_19_DIM_DEFAULTS[9] = 0.0  # route_count


def _make_candidate_features(journal_ids: List[str]) -> Dict[str, List[float]]:
    """Build a 19-dim base feature vector for each journal id (all zeros for dead)."""
    return {jid: list(BASE_19_DIM_DEFAULTS) for jid in journal_ids}


# ---------- tests ----------

def test_same_gold_area_computed_when_research_area_overlaps():
    """same_gold_area=1.0 when paper.research_area ∩ gold.subject_tags ≠ ∅."""
    # Paper: research_area=["AI"]; Gold: subject_tags=["AI", "ML"]
    paper_title = "Test Paper A"
    target_jid = "gold_j"
    neg_jid = "neg_j"
    candidate_jids = [target_jid, neg_jid]
    candidate_features = _make_candidate_features(candidate_jids)

    papers_by_title = {
        paper_title: _make_paper_meta(paper_title, research_area=["AI"]),
    }
    journals_by_id = {
        target_jid: _make_journal_meta(target_jid, ["AI", "ML"], ccf_rating="A"),
        neg_jid: _make_journal_meta(neg_jid, ["Databases"], ccf_rating="B"),
    }
    ablation_data = {
        "variants": {
            "full_hybrid": {
                "feature_names": None,
                "paper_results": [
                    _make_ablation_entry(paper_title, target_jid, candidate_jids, candidate_features)
                ],
            }
        }
    }

    rows = list(build_training_rows(
        ablation_data=ablation_data,
        journals_by_id=journals_by_id,
        max_negatives=1,
        accepted_jid_set=set(),
        papers_by_title=papers_by_title,
    ))

    # gold row (pos)
    pos_rows = [r for r in rows if r["label"] == 1]
    assert len(pos_rows) == 1
    pos = pos_rows[0]
    feature_names = pos["feature_names"]
    same_gold_area_idx = feature_names.index("same_gold_area")
    assert pos["features"][same_gold_area_idx] == 1.0

    # neg row: neg_j has no overlap with "AI"
    neg_rows = [r for r in rows if r["label"] == 0]
    assert len(neg_rows) == 1
    neg = neg_rows[0]
    assert neg["features"][same_gold_area_idx] == 0.0


def test_same_parsed_ccf_area_computed_when_ccf_area_overlaps():
    """same_parsed_ccf_area=1.0 when paper.ccf_research_area ∩ gold.subject_tags ≠ ∅."""
    paper_title = "Test Paper B"
    target_jid = "gold_j"
    candidate_features = _make_candidate_features([target_jid])

    papers_by_title = {
        paper_title: _make_paper_meta(
            paper_title,
            research_area=["machine learning"],
            ccf_research_area=["人工智能"],  # CCF 风格 area
        ),
    }
    journals_by_id = {
        target_jid: _make_journal_meta(target_jid, ["人工智能", "机器学习"], ccf_rating="A"),
    }
    ablation_data = {
        "variants": {
            "full_hybrid": {
                "feature_names": None,
                "paper_results": [
                    _make_ablation_entry(paper_title, target_jid, [target_jid], candidate_features)
                ],
            }
        }
    }

    rows = list(build_training_rows(
        ablation_data=ablation_data,
        journals_by_id=journals_by_id,
        max_negatives=0,
        accepted_jid_set=set(),
        papers_by_title=papers_by_title,
    ))
    pos = [r for r in rows if r["label"] == 1][0]
    feature_names = pos["feature_names"]
    idx = feature_names.index("same_parsed_ccf_area")
    assert pos["features"][idx] == 1.0


def test_same_ccf_level_computed_when_levels_match():
    """same_ccf_level=1.0 when paper.gold.ccf_rating == candidate.ccf_rating."""
    paper_title = "Test Paper C"
    target_jid = "gold_j"   # ccf A
    other_a_jid = "other_a"  # ccf A
    other_b_jid = "other_b"  # ccf B
    candidate_jids = [target_jid, other_a_jid, other_b_jid]
    candidate_features = _make_candidate_features(candidate_jids)

    papers_by_title = {paper_title: _make_paper_meta(paper_title, research_area=["AI"])}
    journals_by_id = {
        target_jid: _make_journal_meta(target_jid, ["AI"], "A"),
        other_a_jid: _make_journal_meta(other_a_jid, ["AI"], "A"),
        other_b_jid: _make_journal_meta(other_b_jid, ["AI"], "B"),
    }
    ablation_data = {
        "variants": {
            "full_hybrid": {
                "feature_names": None,
                "paper_results": [
                    _make_ablation_entry(paper_title, target_jid, candidate_jids, candidate_features)
                ],
            }
        }
    }

    rows = list(build_training_rows(
        ablation_data=ablation_data,
        journals_by_id=journals_by_id,
        max_negatives=2,
        accepted_jid_set=set(),
        papers_by_title=papers_by_title,
    ))
    feature_names = rows[0]["feature_names"]
    idx = feature_names.index("same_ccf_level")

    by_jid = {r["journal_id"]: r for r in rows}
    assert by_jid[target_jid]["features"][idx] == 1.0  # gold ccf=A
    assert by_jid[other_a_jid]["features"][idx] == 1.0  # other A
    assert by_jid[other_b_jid]["features"][idx] == 0.0  # other B


def test_candidate_in_accepted_corpus_set_when_jid_in_corpus():
    """candidate_in_accepted_corpus=1.0 when jid is in AcceptedPaperStore."""
    paper_title = "Test Paper D"
    target_jid = "in_corpus_j"
    other_jid = "not_in_corpus_j"
    candidate_jids = [target_jid, other_jid]
    candidate_features = _make_candidate_features(candidate_jids)

    papers_by_title = {paper_title: _make_paper_meta(paper_title, research_area=["AI"])}
    journals_by_id = {
        target_jid: _make_journal_meta(target_jid, ["AI"], "A"),
        other_jid: _make_journal_meta(other_jid, ["AI"], "A"),
    }
    accepted_jid_set = {target_jid}  # 只 target 在 corpus 里

    ablation_data = {
        "variants": {
            "full_hybrid": {
                "feature_names": None,
                "paper_results": [
                    _make_ablation_entry(paper_title, target_jid, candidate_jids, candidate_features)
                ],
            }
        }
    }

    rows = list(build_training_rows(
        ablation_data=ablation_data,
        journals_by_id=journals_by_id,
        max_negatives=1,
        accepted_jid_set=accepted_jid_set,
        papers_by_title=papers_by_title,
    ))
    feature_names = rows[0]["feature_names"]
    idx = feature_names.index("candidate_in_accepted_corpus")
    by_jid = {r["journal_id"]: r for r in rows}
    assert by_jid[target_jid]["features"][idx] == 1.0
    assert by_jid[other_jid]["features"][idx] == 0.0


def test_base_features_19_dim_not_20():
    """2026-06-26: paper_strength removed → base features are 19-dim, not 20."""
    paper_title = "Test Paper E"
    target_jid = "gold_j"
    candidate_features = _make_candidate_features([target_jid])
    papers_by_title = {paper_title: _make_paper_meta(paper_title, research_area=["AI"])}
    journals_by_id = {target_jid: _make_journal_meta(target_jid, ["AI"], "A")}
    ablation_data = {
        "variants": {
            "full_hybrid": {
                "feature_names": None,
                "paper_results": [
                    _make_ablation_entry(paper_title, target_jid, [target_jid], candidate_features)
                ],
            }
        }
    }

    rows = list(build_training_rows(
        ablation_data=ablation_data,
        journals_by_id=journals_by_id,
        max_negatives=0,
        accepted_jid_set=set(),
        papers_by_title=papers_by_title,
    ))
    pos = [r for r in rows if r["label"] == 1][0]
    # 19 base + 6 evidence = 25
    assert len(pos["features"]) == 25
    assert len(pos["feature_names"]) == 25


def test_no_paper_strength_in_feature_names():
    """2026-06-26: paper_strength is gone from feature_names."""
    paper_title = "Test Paper F"
    target_jid = "gold_j"
    candidate_features = _make_candidate_features([target_jid])
    papers_by_title = {paper_title: _make_paper_meta(paper_title, research_area=["AI"])}
    journals_by_id = {target_jid: _make_journal_meta(target_jid, ["AI"], "A")}
    ablation_data = {
        "variants": {
            "full_hybrid": {
                "feature_names": None,
                "paper_results": [
                    _make_ablation_entry(paper_title, target_jid, [target_jid], candidate_features)
                ],
            }
        }
    }

    rows = list(build_training_rows(
        ablation_data=ablation_data,
        journals_by_id=journals_by_id,
        max_negatives=0,
        accepted_jid_set=set(),
        papers_by_title=papers_by_title,
    ))
    pos = [r for r in rows if r["label"] == 1][0]
    assert "paper_strength" not in pos["feature_names"]
```

- [ ] **Step 2: 运行测试，验证 FAIL（因为新签名 `accepted_jid_set` 还没加）**

Run: `cd /Users/qian/PycharmProjects/paper && pytest tests/test_build_ranking_training_data.py -v 2>&1 | head -40`
Expected: 测试运行时 ImportError / TypeError（`build_training_rows` 不接受 `accepted_jid_set` 参数）。这是预期失败。

- [ ] **Step 3: 提交 failing tests**

```bash
cd /Users/qian/PycharmProjects/paper
git add tests/test_build_ranking_training_data.py
git commit -m "test(7.3): add failing tests for 4 dead-feature recompute + 19-dim base"
```

---

## Task 4: 改 build_training_rows 签名 + 加 accepted_jid_set 透传

**Files:**
- Modify: `scripts/build_ranking_training_data.py:176-280` (build_training_rows signature + body)

- [ ] **Step 1: 改 `build_training_rows` 签名加 `accepted_jid_set` 和验证 `papers_by_title` 非 None**

Edit `scripts/build_ranking_training_data.py:176-183`。把：

```python
def build_training_rows(
    ablation_data: dict,
    journals_by_id: Dict[str, dict],
    max_negatives: int = 10,
    only_variants: Optional[Iterable[str]] = None,
    evidence_lookup: Optional[Dict[str, Dict[str, dict]]] = None,
    papers_by_title: Optional[Dict[str, dict]] = None,
) -> Iterable[dict]:
```

改为：

```python
def build_training_rows(
    ablation_data: dict,
    journals_by_id: Dict[str, dict],
    max_negatives: int = 10,
    only_variants: Optional[Iterable[str]] = None,
    evidence_lookup: Optional[Dict[str, Dict[str, dict]]] = None,
    papers_by_title: Optional[Dict[str, dict]] = None,
    accepted_jid_set: Optional[Set[str]] = None,
) -> Iterable[dict]:
    """从 ablation JSON 产出训练样本。

    仅在 paper 的 ``candidate_features[target_jid]`` 存在时产正样本;
    负样本来自同 paper 的其他候选期刊,按 NEGATIVE_PRIORITY 分类。

    ``evidence_lookup`` (Task 6.4, 26-dim schema → 2026-06-26 25-dim): when supplied,
    each row's 19-dim base features are extended with the 6 LLM-evidence fields
    looked up by (paper title, journal_id) and the row's ``feature_names`` is
    set to ``FEATURE_NAMES_WITH_LLM_EVIDENCE`` (25-dim). When omitted, output
    is the legacy 19-dim schema.

    ``papers_by_title`` (阶段 6.5, 27-dim schema, 2026-06-26: was 28): when
    supplied with ``evidence_lookup``, each row's 25-dim features are further
    extended with 2 tier/area features (journal_tier_weight + area_exclusivity),
    and feature_names is set to FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY
    (27-dim). Used for P2-mini 27-dim LTR retrain.

    ``accepted_jid_set`` (2026-06-26): set of journal_ids that are present
    in the AcceptedPaperStore. Used to compute ``candidate_in_accepted_corpus``
    and to compute ``same_*`` features via the gold venue's subject_tags.

    2026-06-26 schema changes: paper_strength removed (was 0.0 in all v4
    training rows; dead feature). Base dim 20→19, evidence schema 26→25,
    tier+exclusivity schema 28→27. Old models incompatible; retrain required.
    """
```

- [ ] **Step 2: 在 `build_training_rows` 顶部加 `Set` import**

Edit `scripts/build_ranking_training_data.py:36`。把：

```python
from typing import Dict, Iterable, List, Optional
```

改为：

```python
from typing import Dict, Iterable, List, Optional, Set
```

- [ ] **Step 3: 在 `_row()` 闭包上面加 gold_journal 提取（重算 4 dead 特征）**

Edit `scripts/build_ranking_training_data.py:200-280`。完整替换 `for paper_idx, paper_result in enumerate(variant_data.get("paper_results") or []):` 这一段（约 line 214-272），改成：

```python
        for paper_idx, paper_result in enumerate(variant_data.get("paper_results") or []):
            target_jid = paper_result.get("target_journal_id")
            if not target_jid:
                continue
            candidate_features = paper_result.get("candidate_features") or {}
            if not candidate_features:
                continue
            paper_id = paper_result.get("title") or f"paper_{paper_idx}"

            # Per-paper venue (for snapshot key lookup). Default to empty
            # so the snapshot key becomes just the title, which is what
            # the role ranker does when paper_profile has no venue.
            paper_venue = paper_result.get("venue", "") or ""

            # 阶段 6.5 (P2-mini): paper 锚 area + 同领域候选数。
            # 从 papers_by_title (joined from papers metadata jsonl) 读。
            paper_anchor_area: Optional[str] = None
            paper_meta = papers_by_title.get(paper_id, {}) if papers_by_title else {}
            pra = paper_meta.get("research_area") or []
            if isinstance(pra, list) and pra:
                paper_anchor_area = pra[0]
            elif isinstance(pra, str) and pra:
                paper_anchor_area = pra

            # 算 n_matching_in_pool (同领域候选数)。
            n_matching_in_pool: Optional[int] = None
            if paper_anchor_area:
                n_matching_in_pool = sum(
                    1
                    for jid in candidate_features.keys()
                    if paper_anchor_area
                    in (journals_by_id.get(jid, {}).get("subject_tags") or [])
                )

            # 2026-06-26: 提取 gold venue 上下文,重算 4 个 dead 特征。
            # 旧的 candidate_features (从 ablation JSON 来) 里 4 个 dead
            # 特征全是 0.0,我们现在用真实元数据覆写。
            gold_journal_meta = journals_by_id.get(target_jid) or {}
            gold_subject_tags = set(gold_journal_meta.get("subject_tags") or [])
            paper_research_area = set(paper_meta.get("research_area") or [])
            paper_ccf_research_area = set(paper_meta.get("ccf_research_area") or []) or paper_research_area
            paper_ccf_target_level = (gold_journal_meta.get("ccf_rating") or "").upper()
            accepted_jid_set = accepted_jid_set or set()

            def _row(label: int, jid: str, neg_type: str) -> dict:
                feats = candidate_features.get(jid) or []
                # 2026-06-26: ablation JSON 里 candidate_features 是 20 维
                # (含 paper_strength 占位),新 schema 是 19 维。trim 掉 idx 18
                # (paper_strength 位置) 保持 base 19 维。
                if len(feats) > 19:
                    feats = list(feats[:18]) + list(feats[19:])
                elif len(feats) < 19:
                    feats = list(feats) + [0.0] * (19 - len(feats))
                else:
                    feats = list(feats)
                # 2026-06-26: 覆写 4 个 dead 特征 (位置在 trim 后的 schema 里)
                # 用 feature_names 动态查 idx(因为 19/25/27 维 schema 顺序可能不同)
                if use_evidence_schema:
                    base_feature_names = list(FEATURE_NAMES_WITH_LLM_EVIDENCE)[: len(FEATURE_NAMES)]
                else:
                    base_feature_names = list(FEATURE_NAMES)
                # 计算 4 个 dead 特征
                _same_gold_area = 1.0 if (paper_research_area & gold_subject_tags) else 0.0
                _same_parsed_ccf_area = 1.0 if (paper_ccf_research_area & gold_subject_tags) else 0.0
                cand_meta = journals_by_id.get(jid) or {}
                cand_ccf = (cand_meta.get("ccf_rating") or "").upper()
                _same_ccf_level = 1.0 if (
                    paper_ccf_target_level and cand_ccf and paper_ccf_target_level == cand_ccf
                ) else 0.0
                _candidate_in_accepted_corpus = 1.0 if jid in accepted_jid_set else 0.0
                # 写入 feats (新 schema 19 维,idx 同 base_feature_names)
                if "same_gold_area" in base_feature_names:
                    feats[base_feature_names.index("same_gold_area")] = _same_gold_area
                if "same_parsed_ccf_area" in base_feature_names:
                    feats[base_feature_names.index("same_parsed_ccf_area")] = _same_parsed_ccf_area
                if "same_ccf_level" in base_feature_names:
                    feats[base_feature_names.index("same_ccf_level")] = _same_ccf_level
                if "candidate_in_accepted_corpus" in base_feature_names:
                    feats[base_feature_names.index("candidate_in_accepted_corpus")] = _candidate_in_accepted_corpus

                if use_evidence_schema:
                    feats = list(feats) + _evidence_vector_for_row(
                        paper_id, paper_venue, jid, evidence_lookup
                    )
                # 阶段 6.5 (P2-mini): 27-dim schema 时附加 2 维 tier/area。
                if use_28_dim_schema:
                    journal_meta = journals_by_id.get(jid) or {}
                    feats = list(feats) + [
                        _tier_weight_value(journal_meta.get("ccf_rating")),
                        _area_exclusivity_value(
                            paper_anchor_area=paper_anchor_area,
                            candidate_subject_tags=journal_meta.get("subject_tags"),
                            n_matching_in_pool=n_matching_in_pool,
                        ),
                    ]
                row = {
                    "paper_id": paper_id,
                    "journal_id": jid,
                    "label": label,
                    "features": feats,
                    "feature_names": feature_names,
                    "negative_type": "gold" if label == 1 else neg_type,
                    "variant": variant_name,
                }
                return row
```

- [ ] **Step 4: 改 `_row` 后面的正样本/负样本生成代码调用 `_row` 时不再传 features 列表**

Edit `scripts/build_ranking_training_data.py:280-310`。找到旧代码（约 line 280-300，生成 pos/neg 调用的地方），保留现有结构（用 `_row` 而不是手动构造 dict）。原来的代码是用 `feats = candidate_features.get(jid) or []` + 构造 dict；新代码是直接调 `_row(label, jid, neg_type)`。具体来说，找到原代码里正样本生成块：

```python
            # 1. 正样本(若 gold 期刊在 candidate_features)
            if target_jid in candidate_features:
                ...
                yield {
                    "paper_id": paper_id,
                    "journal_id": target_jid,
                    "label": 1,
                    "features": ...,
                    ...
                }
```

把它改成：

```python
            # 1. 正样本(若 gold 期刊在 candidate_features)
            if target_jid in candidate_features:
                yield _row(1, target_jid, "gold")
            # 2. 负样本(从 candidate_features 池里挑,排除 gold)
            if max_negatives > 0:
                neg_candidates = [
                    jid for jid in candidate_features.keys() if jid != target_jid
                ]
                negatives = _build_negatives(
                    candidate_jids=neg_candidates,
                    target_jid=target_jid,
                    rule_top20=paper_result.get("rule_top20") or [],
                    journals_by_id=journals_by_id,
                    max_negatives=max_negatives,
                )
                for jid, neg_type in negatives:
                    yield _row(0, jid, neg_type)
```

- [ ] **Step 5: 改 imports 加 `Set`**

Done in step 2 above.

- [ ] **Step 6: 跑新测试，验证 PASS**

Run: `cd /Users/qian/PycharmProjects/paper && pytest tests/test_build_ranking_training_data.py -v`
Expected: 6 个新测试 PASS。

- [ ] **Step 7: 跑完整 feature_builder 测试 + build_training_data 测试，无 regression**

Run: `cd /Users/qian/PycharmProjects/paper && pytest tests/test_feature_builder.py tests/test_build_ranking_training_data.py tests/test_ltr_adapter.py -v`
Expected: All pass.

- [ ] **Step 8: 提交**

```bash
cd /Users/qian/PycharmProjects/paper
git add scripts/build_ranking_training_data.py tests/test_build_ranking_training_data.py
git commit -m "feat(7.4): recompute 4 dead features in build_training_rows (accepted corpus + gold venue)"
```

---

## Task 5: 加 AcceptedPaperStore 加载 + journals_by_id_full 构造到 main()

**Files:**
- Modify: `scripts/build_ranking_training_data.py:437-525` (main)

- [ ] **Step 1: 加 `--accepted-corpus-dir` CLI 参数**

Edit `scripts/build_ranking_training_data.py:466-475`（在 `--papers-jsonl` 之后）。加：

```python
    parser.add_argument(
        "--accepted-corpus-dir",
        default="data/accepted_papers",
        help=(
            "2026-06-26: path to AcceptedPaperStore directory. When supplied, "
            "the build script loads the corpus and uses the set of journal_ids "
            "with papers to compute candidate_in_accepted_corpus. Default: "
            "data/accepted_papers (the project's standard corpus location)."
        ),
    )
```

- [ ] **Step 2: 在 main() 加载 AcceptedPaperStore 构造 accepted_jid_set**

Edit `scripts/build_ranking_training_data.py:478-498`（加载 ablation_data 之后）。在 `papers_by_title` 加载后加：

```python
    # 2026-06-26: 加载 AcceptedPaperStore 构造 accepted_jid_set,用于计算
    # candidate_in_accepted_corpus。set 取 _by_journal.keys() (只含至少
    # 有一篇 paper 的 jid;空 journal 不计入)。
    accepted_jid_set: set = set()
    try:
        from src.journals.accepted_paper_store import AcceptedPaperStore
        accepted_store = AcceptedPaperStore(accepted_dir=args.accepted_corpus_dir)
        accepted_store.load()
        accepted_jid_set = set(accepted_store._by_journal.keys())
        print(
            f"Loaded {accepted_store.journal_count} journals "
            f"({accepted_store.count} papers) from {args.accepted_corpus_dir}"
        )
    except Exception as e:
        print(
            f"[warn] failed to load AcceptedPaperStore from "
            f"{args.accepted_corpus_dir}: {e}; accepted_jid_set will be empty",
            file=sys.stderr,
        )
```

- [ ] **Step 3: 把 `accepted_jid_set` 透传给 `build_training_rows`**

Edit `scripts/build_ranking_training_data.py:515-524`。把：

```python
    rows = list(
        build_training_rows(
            ablation_data=ablation_data,
            journals_by_id=journals_by_id,
            max_negatives=args.max_negatives,
            only_variants=args.variants,
            evidence_lookup=evidence_lookup,
            papers_by_title=papers_by_title,
        )
    )
```

改为：

```python
    rows = list(
        build_training_rows(
            ablation_data=ablation_data,
            journals_by_id=journals_by_id,
            max_negatives=args.max_negatives,
            only_variants=args.variants,
            evidence_lookup=evidence_lookup,
            papers_by_title=papers_by_title,
            accepted_jid_set=accepted_jid_set,
        )
    )
```

- [ ] **Step 4: 跑已有 build_training_data 测试 + 集成验证（小数据集）**

Run:
```bash
cd /Users/qian/PycharmProjects/paper
pytest tests/test_build_ranking_training_data.py -v
```

Expected: 6 个新测试仍 PASS。

- [ ] **Step 5: 提交**

```bash
cd /Users/qian/PycharmProjects/paper
git add scripts/build_ranking_training_data.py
git commit -m "feat(7.5): load AcceptedPaperStore in build script, pass accepted_jid_set to build_training_rows"
```

---

## Task 6: 加 sidecar report 4 dead 特征 nonzero 计数

**Files:**
- Modify: `scripts/build_ranking_training_data.py:374-450` (build_training_report)

- [ ] **Step 1: 在 `build_training_report` 里加 dead-feature nonzero 计数**

Edit `scripts/build_ranking_training_data.py:374-450`（`build_training_report` 函数体）。在 `positives_total` 累加后加：

```python
    # 2026-06-26: 4 dead features nonzero 计数(证明 augmentation 真生效)
    dead_feature_names = [
        "same_gold_area",
        "same_parsed_ccf_area",
        "same_ccf_level",
        "candidate_in_accepted_corpus",
    ]
    dead_feature_nonzero: Dict[str, int] = {f: 0 for f in dead_feature_names}
    for row in positive_rows:
        feats = row.get("features") or []
        fns = row.get("feature_names") or []
        for fname in dead_feature_names:
            if fname in fns:
                idx = fns.index(fname)
                if idx < len(feats) and feats[idx] > 0.0:
                    dead_feature_nonzero[fname] += 1
```

并在函数返回的 dict 里加：

```python
    return {
        "positives_total": positives_total,
        "positives_by_variant": positives_by_variant,
        "positives_with_target_in_top50_count": positives_in_top50,
        "positives_with_target_in_top50_ratio": (
            positives_in_top50 / positives_total if positives_total else 0.0
        ),
        "retrieval_topk_80_warning": positives_in_top50 < (
            RETRIEVAL_TOPK_80_THRESHOLD * positives_total
        ) if positives_total else False,
        "retrieval_topk_80_threshold": RETRIEVAL_TOPK_80_THRESHOLD,
        "positives_missing_route_features": missing_per_feature,
        "route_combination_counts": combination_counts,
        "dead_feature_nonzero": dead_feature_nonzero,  # 2026-06-26 新加
    }
```

- [ ] **Step 2: 跑测试 + 提交**

Run: `cd /Users/qian/PycharmProjects/paper && pytest tests/test_build_ranking_training_data.py -v && git add scripts/build_ranking_training_data.py && git commit -m "feat(7.6): add dead_feature_nonzero to sidecar report"`

Expected: 测试 PASS；commit 成功。

---

## Task 7: 跑数据生成，验证 sidecar 报告

**Files:**
- Run: 重新生成 25-dim 训练数据

- [ ] **Step 1: 跑数据生成**

```bash
cd /Users/qian/PycharmProjects/paper
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

Expected: 输出 25-dim JSONL + sidecar report。stdout 应包含:
- `Loaded N journals (M papers) from data/accepted_papers`
- `Loaded research_area for ... papers`
- `Wrote X rows to ...`
- `Wrote sidecar report to ...`

- [ ] **Step 2: 验证 sidecar 报告 4 dead 特征 nonzero count**

```bash
cd /Users/qian/PycharmProjects/paper
python3 -c "
import json
r = json.load(open('data/training/ranker_train_balanced_540_v5_25dim_report.json'))
print('positives_total:', r['positives_total'])
print('dead_feature_nonzero:', r['dead_feature_nonzero'])
assert r['dead_feature_nonzero']['same_gold_area'] > 200, f'same_gold_area too low: {r[\"dead_feature_nonzero\"][\"same_gold_area\"]}'
assert r['dead_feature_nonzero']['candidate_in_accepted_corpus'] > 50, f'candidate_in_accepted_corpus too low: {r[\"dead_feature_nonzero\"][\"candidate_in_accepted_corpus\"]}'
print('OK')
"
```

Expected: 4 个 dead 特征 nonzero > 0；`same_gold_area > 200`, `candidate_in_accepted_corpus > 50`。脚本打印 "OK"。

- [ ] **Step 3: 验证 JSONL schema 是 25-dim**

```bash
cd /Users/qian/PycharmProjects/paper
head -1 data/training/ranker_train_balanced_540_v5_25dim.jsonl | python3 -c "
import json, sys
r = json.loads(sys.stdin.read())
print('len(features):', len(r['features']))
print('len(feature_names):', len(r['feature_names']))
print('feature_names:', r['feature_names'])
assert len(r['features']) == 25, f'expected 25, got {len(r[\"features\"])}'
assert 'paper_strength' not in r['feature_names']
print('OK')
"
```

Expected: `len(features) == 25`, `paper_strength not in feature_names`. 脚本打印 "OK"。

- [ ] **Step 4: 提交新训练数据 + sidecar report**

```bash
cd /Users/qian/PycharmProjects/paper
git add data/training/ranker_train_balanced_540_v5_25dim.jsonl data/training/ranker_train_balanced_540_v5_25dim_report.json
git commit -m "data(7.7): regenerate 25-dim training data with 4 dead features alive"
```

---

## Task 8: 训练新 LR 模型

**Files:**
- Run: train new LR model

- [ ] **Step 1: 检查 train_learning_to_rank.py 是否自适应 feature_dim**

```bash
cd /Users/qian/PycharmProjects/paper
grep -n "feature_dim\|len(feature_names)" scripts/train_learning_to_rank.py | head -10
```

Expected: 看到 `len(feature_names[0])` 或类似自适应代码（不需要 hardcode dim）。

- [ ] **Step 2: 跑训练**

```bash
cd /Users/qian/PycharmProjects/paper
python scripts/train_learning_to_rank.py \
  --train data/training/ranker_train_balanced_540_v5_25dim.jsonl \
  --output data/models/learning_to_ranker_balanced_v5_25dim_lr.json \
  --model-type logistic_regression
```

Expected: stdout 应包含:
- "Loaded N rows from ..."
- "feature_dim: 25"
- "model_type: logistic_regression"
- "pairwise_accuracy: > 0.96"
- "Convergence: converged"
- "Saved model to ..."

- [ ] **Step 3: 验证模型 feature_dim + 4 dead 特征 coef != 0**

```bash
cd /Users/qian/PycharmProjects/paper
python3 -c "
import json
m = json.load(open('data/models/learning_to_ranker_balanced_v5_25dim_lr.json'))
print('feature_dim:', m['feature_dim'])
print('backend:', m['backend'])
print('model_type:', m['model_type'])
assert m['feature_dim'] == 25
assert m['model_type'] == 'logistic_regression'
fnames = m['feature_names']
coefs = m['coef']
dead_features = ['same_gold_area', 'same_parsed_ccf_area', 'same_ccf_level', 'candidate_in_accepted_corpus']
for f in dead_features:
    if f in fnames:
        i = fnames.index(f)
        c = coefs[i]
        print(f'{f}: coef={c}')
        assert c != 0.0, f'{f} coef is still 0!'
    else:
        print(f'[warn] {f} not in feature_names')
print('OK')
"
```

Expected: 4 dead 特征 coef != 0；脚本打印 "OK"。

- [ ] **Step 4: 提交新模型**

```bash
cd /Users/qian/PycharmProjects/paper
git add data/models/learning_to_ranker_balanced_v5_25dim_lr.json
git commit -m "model(7.8): train v5 25-dim LR on accepted-corpus training data"
```

---

## Task 9: 切 configs/app.yaml model_path

**Files:**
- Modify: `configs/app.yaml:105`

- [ ] **Step 1: 改 `learned_reranker.model_path`**

Edit `configs/app.yaml:105`。把：

```yaml
    model_path: "data/models/learning_to_ranker_balanced_v4_lr.json"
```

改为：

```yaml
    model_path: "data/models/learning_to_ranker_balanced_v5_25dim_lr.json"
```

- [ ] **Step 2: 移动旧模型到 archive**

```bash
cd /Users/qian/PycharmProjects/paper
mkdir -p data/models/_archive_20260626_pre_accepted_corpus
git mv data/models/learning_to_ranker_balanced_v4_lr.json \
       data/models/_archive_20260626_pre_accepted_corpus/
git mv data/models/learning_to_ranker_balanced_v3_lr.json \
       data/models/_archive_20260626_pre_accepted_corpus/ 2>/dev/null || true
git mv data/models/learning_to_ranker_balanced_v3p_lr.json \
       data/models/_archive_20260626_pre_accepted_corpus/ 2>/dev/null || true
git mv data/models/learning_to_ranker_balanced_v4_lr_20dim.json \
       data/models/_archive_20260626_pre_accepted_corpus/ 2>/dev/null || true
git mv data/models/learning_to_ranker_balanced_v4_lr_28dim.json \
       data/models/_archive_20260626_pre_accepted_corpus/ 2>/dev/null || true
git mv data/models/learning_to_ranker_balanced_v4_lightgbm28.json \
       data/models/_archive_20260626_pre_accepted_corpus/ 2>/dev/null || true
git mv data/models/learning_to_ranker_balanced_v4_lightgbm28_pool50.json \
       data/models/_archive_20260626_pre_accepted_corpus/ 2>/dev/null || true
```

- [ ] **Step 3: 提交 yaml + archive 移动**

```bash
cd /Users/qian/PycharmProjects/paper
git add configs/app.yaml
git status --short data/models/
git commit -m "chore(7.9): switch to v5 25-dim model + archive 26/28-dim models"
```

---

## Task 10: Pipeline smoke test

**Files:**
- Run: holdout240 smoke test

- [ ] **Step 1: 跑 holdout240 pipeline eval**

```bash
cd /Users/qian/PycharmProjects/paper
python scripts/run_evaluation.py \
  --benchmark-profile holdout240 \
  --mode abstract --top-k 5 --workers 1 \
  --baseline-eval data/evaluation/results/eval_holdout240_ltr_weight_020_20260622.json \
  --output data/evaluation/results/eval_holdout240_v5_25dim_lr.json
```

Expected: 跑完，stdout 应包含:
- `Hit@5: 158/240 (65.8%)` 或更高（**不 regression**）
- `同领域命中@5: ≥ 220`
- `可接受期刊命中@5: ≥ ...`
- `结果已保存到: data/evaluation/results/eval_holdout240_v5_25dim_lr.json`

- [ ] **Step 2: 验证 hit@5 ≥ 158/240 (baseline 不 regression)**

```bash
cd /Users/qian/PycharmProjects/paper
python3 -c "
import json
r = json.load(open('data/evaluation/results/eval_holdout240_v5_25dim_lr.json'))
hit5 = r['metrics']['hit_at_5']
print('hit@5:', hit5, '/240 =', f'{hit5/240:.1%}')
assert hit5 >= 158, f'REGRESSION: hit@5 = {hit5} < 158'
print('OK (no regression)')
"
```

Expected: hit@5 ≥ 158；脚本打印 "OK (no regression)"。

- [ ] **Step 3: 提交新 eval result**

```bash
cd /Users/qian/PycharmProjects/paper
git add data/evaluation/results/eval_holdout240_v5_25dim_lr.json
git commit -m "eval(7.10): holdout240 smoke test on v5 25-dim LR (hit@5 ≥ 158)"
```

---

## Task 11: 跑完整测试 + 全量回归

**Files:**
- Run: pytest

- [ ] **Step 1: 跑相关测试集**

```bash
cd /Users/qian/PycharmProjects/paper
pytest tests/test_feature_builder.py tests/test_build_ranking_training_data.py tests/test_ltr_adapter.py tests/test_learning_to_rank.py tests/test_run_evaluation_diagnostics.py tests/test_benchmark_manifest.py -v
```

Expected: All pass.

- [ ] **Step 2: 跑全量测试**

```bash
cd /Users/qian/PycharmProjects/paper
pytest tests/ -v
```

Expected: All pass (or known pre-existing failures only, no new ones).

- [ ] **Step 3: 如果有 regression 报告**

如果有新 failure, 报告给用户: 包括失败测试名 + traceback 头 5 行。

- [ ] **Step 4: 最终提交（如果还有未提交的改动）**

```bash
cd /Users/qian/PycharmProjects/paper
git status --short
# 如果有未提交改动:
# git add <new_files>
# git commit -m "test(7.11): final regression check"
```

---

## Self-Review

**1. Spec coverage** — 各 spec 章节都有对应 task：
- §3.1 删 paper_strength → Task 1+2 ✅
- §3.2 改 build script → Task 3+4+5+6 ✅
- §3.3 验证 train_learning_to_rank.py 自适应 → Task 8 step 1 ✅
- §3.4 更新 yaml → Task 9 ✅
- §3.5 数据生成 + 重训 → Task 7+8 ✅
- §3.6 测试 → Task 1+3 ✅
- §5 接受条件 → Task 7 step 2-3, Task 8 step 3, Task 10 step 2 ✅

**2. Placeholder scan** — 全文搜 "TBD" / "TODO" / "fill in" / "implement later"：
- 无。所有 step 都有具体代码/命令/预期输出。

**3. Type consistency** — 关键签名匹配：
- `build_training_rows(ablation_data, journals_by_id, max_negatives=10, only_variants, evidence_lookup, papers_by_title, accepted_jid_set)` — Task 3 测试 + Task 4 实现 + Task 5 调用一致
- `AcceptedPaperStore._by_journal.keys()` — 用 set 装 `accepted_jid_set` ✅
- `FEATURE_NAMES` / `FEATURE_NAMES_WITH_LLM_EVIDENCE` / `FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY` 长度: 19/25/27 — Task 1 测试 + Task 2 实现一致
- 4 dead 特征名: `same_gold_area` / `same_parsed_ccf_area` / `same_ccf_level` / `candidate_in_accepted_corpus` — 跨 Task 1/3/6/7 一致

**4. No spec gaps**.
