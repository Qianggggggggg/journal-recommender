# 论文投稿期刊推荐系统 - 设计规格书

**日期**: 2026-05-18
**版本**: v1.0
**状态**: 已批准

---

## 一、项目概述

构建一个「论文投稿期刊推荐系统」，能够根据论文内容（标题 / 摘要 / 全文）推荐适合投稿的计算机类期刊，并提供推荐理由、领域标签、影响力分区等辅助信息。

**核心特性**:
- 三种输入模式：标题模式 / 摘要模式 / 全文模式
- 两阶段推荐：混合召回 → 两阶段排序
- 独立解释模块：推荐理由与排序逻辑分离
- 降级策略：LLM 失败时回退到规则分类

---

## 二、系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Web UI)                       │
│   输入论文信息 → 显示推荐结果 + 匹配理由 + 期刊信息           │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP REST
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     API Layer (FastAPI)                      │
│   POST /recommend {title, abstract?, full_text?, mode}      │
└────────────────────────────┬────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Paper Parser   │ │    Retriever    │ │     Ranker      │
│  论文理解层      │ │    召回层       │ │     排序层       │
│  - 结构化特征    │ │  混合召回20-50 │ │  两阶段排序Top5  │
│  - 降级策略      │ │  BM25+向量+标签 │ │  规则→LLM精排   │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     Explainer                                │
│              独立推荐理由生成模块                              │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                 Journal Store (JSONL+FAISS)                  │
│               计算机类期刊库 + 向量索引                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、数据层（Journal Store）

### 3.1 数据来源
- **DOAJ**: 开放获取期刊元数据（学科分类、关键词、出版信息、OA 状态）
- **SCImago**: 期刊分区/排名信息（SCImago Journal Rank、CiteScore）

### 3.2 标准化字段

| 字段 | 说明 |
|------|------|
| journal_id | 唯一标识 |
| journal_name | 期刊名称 |
| publisher | 出版社 |
| subject_tags | 学科标签（AI/CV/NLP/SE/网络/安全/数据库/理论计算等）|
| keywords | 期刊范围关键词 |
| scope_text | 期刊 scope 说明 |
| oa_type | OA 类型（Full OA / Hybrid / Subscription）|
| submission_url | 投稿链接 |
| homepage_url | 期刊主页 |
| sqr_rank | SCImago 排名 |
| quartile | 分区 (Q1/Q2/Q3/Q4) |
| impact_like_score | 影响因子指标 |
| review_time | 审稿周期（估算）|
| apc | APC 费用（若有）|
| target_paper_type | 适合的论文类型（method/system/experiment/survey）|
| journal_profile | **预拼接文本**: `journal_name + scope_text + keywords + subject_tags` |

### 3.3 存储方案
- 元数据: `data/processed/journals.jsonl`
- 向量索引: `data/processed/journals_index.faiss`
- 向量元数据: `data/processed/journals_metadata.parquet`

---

## 四、论文理解层（Paper Parser）

### 4.1 输入模式

| 模式 | 输入 | 处理 |
|------|------|------|
| 标题模式 | title | 轻量语义分析，只做粗筛 |
| 摘要模式 | title + abstract | 标准解析，提取研究方向/方法/贡献 |
| 全文模式 | title + abstract + full_text | 章节摘要后解析，提取完整特征 |

### 4.2 输出：PaperProfile

```json
{
  "title": "论文标题",
  "abstract": "摘要",
  "research_area": ["AI", "NLP"],
  "method_type": "method",
  "paper_type": "experiment",
  "keywords": ["transformer", "预训练"],
  "novelty": "创新点描述",
  "application_domain": ["对话系统"],
  "difficulty_level": "high",
  "style": "conference_like",
  "sections_summary": {
    "introduction": "...",
    "method": "...",
    "experiment": "...",
    "conclusion": "..."
  }
}
```

### 4.3 降级策略

当 LLM 调用失败时，按序尝试：
1. **重试**: 等待 2 秒后重试一次
2. **降级**: 使用规则 + 关键词提取生成基本 PaperProfile
3. **保底**: 返回 BM25-only 召回结果，不做排序

### 4.4 全文模式：章节摘要

不直接喂整篇论文。先用规则分章（Introduction / Method / Experiment / Conclusion），对每章做摘要要点提取，再送入 Parser。降低 token 消耗。

---

## 五、召回层（Retriever）

### 5.1 混合召回策略

三种召回信号并行执行，结果合并去重：

1. **BM25 召回**
   - 对 `journal_profile` 建 BM25 索引
   - 输入: 论文的 title + abstract（或全文摘要）
   - 输出: 候选期刊 + BM25 分数

2. **向量检索召回**
   - 对 `journal_profile` 做 embedding，存 FAISS
   - 输入: 论文的 embedding 向量
   - 输出: 候选期刊 + cosine 相似度

3. **标签过滤**
   - 根据 PaperProfile.research_area 过滤
   - 根据 PaperProfile.paper_type 匹配 target_paper_type
   - 根据 oa_type 偏好过滤

### 5.2 合并去重

```python
# 伪代码
candidates = merge_and_deduplicate(
    bm25_results,    # Top 30
    vector_results,   # Top 30
    tag_filtered,    # Top 20
    top_k=50
)
```

### 5.3 候选数量

| 模式 | 候选数量 |
|------|----------|
| 标题模式 | 30 |
| 摘要模式 | 40 |
| 全文模式 | 50 |

---

## 六、排序层（Ranker）

### 6.1 两阶段排序

**第一阶段：规则打分（Top 30 → Top 10）**

打分维度：
- 主题契合度（基于 research_area 匹配）
- 方法契合度（基于 method_type / paper_type 匹配）
- 影响力与门槛（参考 quartile 过滤）
- OA 偏好匹配

输出：Top 10 + 初步理由

**第二阶段：LLM 精排（Top 10 → Top 5）**

Prompt 模板：
```
给定论文信息和期刊信息，请从以下维度打分并排序：

论文：《title》
- 研究领域：{research_area}
- 方法类型：{method_type}
- 论文类型：{paper_type}

期刊：《journal_name》
- Scope：{scope_text}
- 分区：{quartile}
- OA类型：{oa_type}

请输出：
1. 排序后的期刊列表（从最合适到最不合适）
2. 每个期刊的推荐理由
3. 置信度（0-1）
```

---

## 七、解释模块（Explainer）

### 7.1 职责

独立模块，专门负责生成推荐理由：

```json
{
  "journal_id": "tpami",
  "journal_name": "IEEE TPAMI",
  "match_reasons": [
    "论文研究领域（CV/NN）与期刊 scope 高度契合",
    "期刊偏好 method/system 类型论文，与本文匹配",
    "Q1 分区，与论文影响力相符"
  ],
  "matched_fields": ["research_area", "paper_type", "quartile"],
  "confidence": 0.85
}
```

### 7.2 解释维度

- 主题契合：研究领域 / 关键词匹配
- 方法契合：方法类型 / 论文结构
- 影响力匹配：分区 / 门槛
- 偏好契合：OA 类型 / 审稿周期
- 风险提示：命中率 / 拒稿风险

---

## 八、API 设计

### 8.1 端点

**POST /recommend**

Request:
```json
{
  "title": "论文标题",
  "abstract": "论文摘要（可选）",
  "full_text": "论文全文（可选）",
  "mode": "title | abstract | full",
  "top_k": 5,
  "oa_preference": "any | full_oa | hybrid"
}
```

Response:
```json
{
  "recommendations": [
    {
      "journal_id": "tpami",
      "journal_name": "IEEE TPAMI",
      "score": 0.92,
      "confidence": 0.88,
      "match_reasons": ["..."],
      "matched_fields": ["..."],
      "tags": ["CV", "AI"],
      "oa_type": "subscription",
      "quartile": "Q1",
      "submission_url": "https://..."
    }
  ],
  "paper_profile": {...},
  "mode_used": "abstract",
  "warning": "置信度较低，建议补充摘要" // 仅标题模式
}
```

### 8.2 其他端点

- **GET /journals**: 列出期刊库中的期刊（支持分页和过滤）
- **GET /journals/{id}**: 获取单本期刊详情
- **GET /health**: 健康检查

---

## 九、前端设计

### 9.1 页面布局

单页应用，分为三个区域：

```
┌─────────────────────────────────────────────────────────────┐
│  论文投稿期刊推荐系统                            [模式切换]  │
├────────────────────────────┬────────────────────────────────┤
│                            │                                │
│     输入区                   │         结果区                 │
│     - 论文标题               │     - 推荐期刊列表             │
│     - 摘要（可选）           │     - 匹配理由                 │
│     - 全文（可选）           │     - 期刊信息                 │
│     - [推荐] 按钮            │     - 置信度                   │
│                            │                                │
├────────────────────────────┴────────────────────────────────┤
│  底部: 模式说明 + 置信度提示                                │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 模式切换

- 标题模式：轻量，适合快速筛选
- 摘要模式：主力推荐，平衡精度和成本
- 全文模式：高精度，适合正式投稿前

### 9.3 结果展示

每本推荐期刊显示：
- 期刊名称 + 分区
- 推荐理由（bullet points）
- OA 类型标识
- 投稿链接
- 置信度进度条

---

## 十、技术栈

| 组件 | 选型 |
|------|------|
| API | FastAPI + Pydantic |
| 前端 | HTML + Vanilla JS |
| LLM | MiniMax M2.7 (API 调用) |
| Embedding | Ollama qwen3-embedding:4b |
| 向量索引 | FAISS |
| 元数据存储 | JSONL + Parquet |
| 配置管理 | YAML (PyYAML) |

---

## 十一、目录结构

```
journal-recommender/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── configs/
│   ├── app.yaml          # 应用配置
│   ├── journal_taxonomy.yaml  # 期刊分类体系
│   ├── prompts.yaml      # Prompt 模板
│   └── eval.yaml         # 评测配置
├── data/
│   ├── raw/
│   │   ├── doaj/         # DOAJ 原始数据
│   │   └── scimago/      # SCImago 原始数据
│   ├── processed/
│   │   ├── journals.jsonl
│   │   ├── journals_index.faiss
│   │   ├── journals_metadata.parquet
│   │   └── eval_set.jsonl
│   └── sample/
├── docs/
│   ├── project_overview.md
│   ├── label_taxonomy.md
│   ├── data_sources.md
│   └── evaluation_protocol.md
├── scripts/
│   ├── crawl_doaj.py         # DOAJ 数据采集
│   ├── crawl_scimago.py      # SCImago 数据采集
│   ├── normalize_journals.py # 数据清洗与标准化
│   ├── build_journal_index.py # 构建向量索引
│   ├── parse_paper.py        # 论文解析入口
│   └── evaluate.py            # 评测脚本
├── src/
│   ├── __init__.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI 应用入口
│   │   ├── api.py            # API 路由
│   │   └── schemas.py        # Pydantic 模型
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetchers/         # 数据采集
│   │   ├── cleaners/         # 数据清洗
│   │   └── loaders.py       # 数据加载
│   ├── journals/
│   │   ├── __init__.py
│   │   ├── taxonomy.py      # 期刊分类体系
│   │   ├── journal_model.py  # 期刊数据模型
│   │   └── journal_store.py  # 期刊存储与检索
│   ├── papers/
│   │   ├── __init__.py
│   │   ├── paper_model.py    # 论文数据模型
│   │   ├── paper_parser.py   # 论文解析
│   │   ├── section_splitter.py # 章节切分（全文模式）
│   │   └── paper_profiler.py # 论文特征提取
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── bm25_retriever.py # BM25 召回
│   │   ├── embedding_retriever.py # 向量召回
│   │   └── candidate_generator.py # 混合召回
│   ├── ranker/
│   │   ├── __init__.py
│   │   ├── feature_builder.py # 排序特征构建
│   │   ├── rule_scorer.py     # 规则打分（阶段一）
│   │   ├── llm_ranker.py      # LLM 排序（阶段二）
│   │   └── scoring.py         # 分数计算工具
│   ├── recommender/
│   │   ├── __init__.py
│   │   ├── pipeline.py       # 推荐流程编排
│   │   └── Explainer.py      # 独立解释模块
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py        # 评测指标
│   │   ├── benchmark.py      # 基准测试
│   │   └── error_analysis.py # 错误分析
│   └── utils/
│       ├── __init__.py
│       ├── llm.py            # LLM 调用封装
│       ├── embedding.py      # Embedding 调用
│       ├── text.py           # 文本处理工具
│       └── logging.py        # 日志配置
├── tests/
│   ├── __init__.py
│   ├── test_journal_store.py
│   ├── test_paper_parser.py
│   ├── test_retriever.py
│   ├── test_ranker.py
│   └── test_api.py
└── frontend/
    ├── index.html
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
```

---

## 十二、实现优先级

### Phase 1: MVP（最小可运行版本）

1. [ ] 项目脚手架搭建（目录结构 + 依赖）
2. [ ] 期刊数据采集脚本（DOAJ + SCImago）
3. [ ] 期刊数据标准化与存储
4. [ ] 向量索引构建
5. [ ] Paper Parser 基础版（降级策略）
6. [ ] 混合召回模块
7. [ ] 两阶段排序（规则打分 + LLM 精排）
8. [ ] Explainer 模块
9. [ ] FastAPI 接口
10. [ ] 简单 Web 前端

### Phase 2: 增强

- [ ] 全文模式的章节摘要
- [ ] 评测模块
- [ ] 错误分析工具
- [ ] 前端交互优化

---

## 十三、关键设计决策

1. **混合召回 > 单召回**: BM25 + 向量 + 标签三重信号，结果更稳定
2. **两阶段排序**: 规则先过滤 → LLM 精排，控制成本并提升精度
3. **独立 Explainer**: 推荐理由与排序逻辑分离，便于维护和迭代
4. **降级策略**: LLM 失败时有保底，系统不断流
5. **journal_profile 预拼接**: 检索和解释都用同一份拼接文本，保证一致性