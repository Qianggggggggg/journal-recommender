# ABCD 期刊评级系统迁移计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将系统中 Q1/Q2/Q3/Q4 分区策略全面替换为 ABCD 评级体系，其中 D 代表论文质量未达发表水平

**Architecture:** 移除 Journal.quartile 字段，PaperQuality.quality_level 从 Q1-Q4 改为 A-B-C-D，排序逻辑不再依赖 quartile 权重

**Tech Stack:** Python (FastAPI), Pydantic, JavaScript (Vanilla), CSS

---

## 一、变更范围梳理

### 1.1 核心数据结构变更

| 文件 | 变更内容 |
|------|----------|
| `src/journals/journal_model.py` | `Journal.quartile` 字段移除 |
| `src/papers/quality_assessor.py` | `PaperQuality._strength_to_level()` 从 Q1-Q4 改为 A-B-C-D |
| `src/papers/paper_model.py` | `PaperProfile.quality_level` 类型说明从 Q1-Q4 改为 A-B-C |
| `src/app/schemas.py` | `RecommendItem.quartile` 字段移除 |

### 1.2 业务逻辑变更

| 文件 | 变更内容 |
|------|----------|
| `src/ranker/rule_scorer.py` | 移除 `_get_quartile_weight()` 和 `quartile_q*` 加分 |
| `src/ranker/llm_ranker.py` | 移除 `quartile` 字段返回 |
| `src/recommender/explainer.py` | 移除 `quartile` 展示逻辑 |
| `src/utils/pdf_exporter.py` | 移除 `quartile` 字段 |

### 1.3 前端变更

| 文件 | 变更内容 |
|------|----------|
| `frontend/js/app.js` | 移除 `quartileHtml` 渲染逻辑 |
| `frontend/css/style.css` | 移除 `.quartile-q*` 样式和 `--q*-bg/--q*-color` CSS 变量 |

### 1.4 配置变更

| 文件 | 变更内容 |
|------|----------|
| `configs/prompts.yaml` | 修改 prompt 中的 `{quartile}` 为 `{ccf_rating}` |

### 1.5 测试更新

| 文件 | 变更内容 |
|------|----------|
| `tests/test_quality_assessor.py` | 更新 `_strength_to_level` 的断言从 Q1-Q4 改为 A-B-C-D |
| `tests/test_ranker.py` | 移除 `quartile` 参数 |
| `tests/test_journal_model.py` | 移除 `quartile` 参数 |
| `tests/test_api.py` | 更新 `quality_level` 断言 |

---

## 二、新等级映射规则

### 论文质量等级 (PaperQuality.quality_level)

| strength 阈值 | 旧值 | 新值 | 含义 |
|---------------|------|------|------|
| >= 0.75 | Q1 | **A** | 顶级论文，可投CCF-A/B |
| >= 0.55 | Q2 | **B** | 良好论文，可投CCF-B/C |
| >= 0.35 | Q3 | **C** | 一般论文，可投CCF-C |
| < 0.35 | Q4 | **D** | 未达发表水平 |

### 期刊等级 (Journal.ccf_rating)

保持不变：A/B/C 三级

---

## 三、任务详情

### Task 1: 更新 PaperQuality 质量等级映射

**Files:**
- Modify: `src/papers/quality_assessor.py:53-62`

- [ ] **Step 1: 修改 `_strength_to_level` 方法**

```python
@staticmethod
def _strength_to_level(strength: float) -> str:
    """将 strength 映射到 A/B/C/D"""
    if strength >= 0.75:
        return "A"
    elif strength >= 0.55:
        return "B"
    elif strength >= 0.35:
        return "C"
    else:
        return "D"  # 未达发表水平
```

- [ ] **Step 2: 更新 docstring**

将 `quality_level` 字段的 description 从 `"质量等级: Q1/Q2/Q3/Q4"` 改为 `"质量等级: A/B/C/D (D表示未达发表水平)"`

- [ ] **Step 3: 更新 `readiness` 映射的边界**

`readiness` 阈值可能需要调整，当前逻辑对 A/B 论文使用 0.6，对 C/D 使用 0.35，保持不变

- [ ] **Step 4: 验证文件修改正确**

---

### Task 2: 更新 Journal 数据模型

**Files:**
- Modify: `src/journals/journal_model.py:18`

- [ ] **Step 1: 移除 `quartile` 字段或标记为废弃**

将 `quartile` 字段的 description 从 `"分区: Q1/Q2/Q3/Q4"` 改为 `"已废弃，请使用 ccf_rating"`，并设置 `deprecated=True` 或直接删除（需检查是否有调用方依赖）

**注意：** 需要确认 `Journal.quartile` 是否还有调用方，如果直接删除会影响 API 响应结构。建议保留字段但置为 None，或在 API 层过滤掉。

---

### Task 3: 更新 RuleScorer 排序逻辑

**Files:**
- Modify: `src/ranker/rule_scorer.py:53-60`

- [ ] **Step 1: 移除 `_get_quartile_weight` 方法**

删除整个 `_get_quartile_weight` 方法

- [ ] **Step 2: 移除 `quartile_q*` 权重配置**

删除 `weights` 字典中的：
```python
"quartile_q1": 1.0,    # Q1 加分
"quartile_q2": 0.6,    # Q2 加分
"quartile_q3": 0.2,    # Q3 加分
```

- [ ] **Step 3: 移除 `rule_score` 中的 quartile 加分**

当前 `rule_score` 方法中包含 `score += self.weights.get(f"quartile_{journal.quartile}", 0)` 的逻辑，需要移除

- [ ] **Step 4: 更新 CCF 加分逻辑（如果需要）**

当前 CCF 加分使用 A=4, B=3, C=2 映射到 quartile，但这个映射关系会消失。需要确认是否保留 CCF 直接加分逻辑（原 227-228 行注释已说明这个逻辑）

---

### Task 4: 更新 LLM Ranker

**Files:**
- Modify: `src/ranker/llm_ranker.py:47`

- [ ] **Step 1: 移除 `quartile` 字段**

从返回的 journal_info 字典中移除 `"quartile": journal.quartile or "unknown"`

---

### Task 5: 更新 Explainer

**Files:**
- Modify: `src/recommender/explainer.py:47, 92-93`

- [ ] **Step 1: 移除 `quartile` 字段传递**

删除 `quartile=journal.quartile or "unknown"`

- [ ] **Step 2: 移除 quartile 展示理由**

删除 `if journal.quartile: reasons.append(f"期刊：{journal.quartile}区")`

---

### Task 6: 更新 API Schemas

**Files:**
- Modify: `src/app/schemas.py:35, 55`

- [ ] **Step 1: 移除 `RecommendItem.quartile` 字段**

从 `RecommendItem` class 中删除 `quartile: Optional[str]`

- [ ] **Step 2: 确认 API 响应不包含 quartile**

检查 `src/app/api.py` 中所有返回 `RecommendItem` 的位置，确保没有手动传递 `quartile`

---

### Task 7: 更新 PDF Exporter

**Files:**
- Modify: `src/utils/pdf_exporter.py:111, 192, 214`

- [ ] **Step 1: 移除 PDF 中的 quartile 显示**

检查 PDF 模板中是否有 quartile 显示区域，如有则移除或替换为 ccf_rating

---

### Task 8: 更新前端 JS 渲染

**Files:**
- Modify: `frontend/js/app.js:264-276`

- [ ] **Step 1: 移除 `quartileHtml` 相关逻辑**

删除：
```javascript
const quartileClass = rec.quartile ? `quartile-${rec.quartile.toLowerCase()}` : '';
const quartileHtml = rec.quartile ? `<span class="journal-quartile ${quartileClass}">${rec.quartile}</span>` : '';
```

删除模板中的 `${quartileHtml}`

---

### Task 9: 更新前端 CSS 样式

**Files:**
- Modify: `frontend/css/style.css:451-461`

- [ ] **Step 1: 移除 `.journal-quartile` 样式**

删除整个 `.journal-quartile` class

- [ ] **Step 2: 移除 `.quartile-q*` 样式**

删除 `.quartile-q1`, `.quartile-q2`, `.quartile-q3`, `.quartile-q4`

- [ ] **Step 3: 检查并移除 CSS 变量**

检查文件顶部是否有 `--q1-bg`, `--q1-color` 等变量，如有则移除

---

### Task 10: 更新测试文件

**Files:**
- Modify: `tests/test_quality_assessor.py:101-112`
- Modify: `tests/test_ranker.py:16, 35-37`
- Modify: `tests/test_journal_model.py:15`
- Modify: `tests/test_api.py:63, 72`

- [ ] **Step 1: 更新 `test_quality_assessor.py` 中的断言**

```python
# A: >= 0.75
assert PaperQuality._strength_to_level(0.8) == "A"
assert PaperQuality._strength_to_level(0.75) == "A"
# B: >= 0.55
assert PaperQuality._strength_to_level(0.6) == "B"
assert PaperQuality._strength_to_level(0.55) == "B"
# C: >= 0.35
assert PaperQuality._strength_to_level(0.4) == "C"
assert PaperQuality._strength_to_level(0.35) == "C"
# D: < 0.35
assert PaperQuality._strength_to_level(0.3) == "D"
assert PaperQuality._strength_to_level(0.0) == "D"
```

- [ ] **Step 2: 更新其他测试中的 `quartile` 参数**

从所有 Journal 构造中移除 `quartile=` 参数

- [ ] **Step 3: 更新 API 测试中的 `quality_level` 断言**

将 `"Q1"` 断言改为 `"A"`，以此类推

---

### Task 11: 更新配置文件 prompts.yaml

**Files:**
- Modify: `configs/prompts.yaml:177`

- [ ] **Step 1: 修改 prompt 中的 `{quartile}` 引用**

将 `{quartile}` 替换为 `{ccf_rating}` 或直接删除该行

---

## 四、验证步骤

完成所有任务后，运行以下验证：

```bash
# 1. 测试质量评估映射
python -c "from src.papers.quality_assessor import PaperQuality; print(PaperQuality._strength_to_level(0.8))"  # 应输出: A

# 2. 运行所有测试
pytest tests/ -v

# 3. 启动 API 验证
cd /Users/qian/PycharmProjects/paper && python -m uvicorn src.app.api:app --reload &
curl -s "http://localhost:8000/api/journals" | python -m json.tool | head -50

# 4. 前端验证
# 打开 http://localhost:8000 输入标题测试推荐功能
```

---

## 五、回滚计划

如需回滚，执行：
```bash
git stash
# 或
git checkout <commit-hash>
```