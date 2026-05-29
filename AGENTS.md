# 🧠 Codex 指南：论文投稿期刊推荐系统

## 一、项目概述

本项目目标是构建一个“论文投稿期刊推荐系统”，能够根据论文内容推荐适合投稿的计算机类期刊。

系统支持三种输入模式：
1. 标题（Title）
2. 摘要（Abstract）
3. 全文（Full Text）

系统输出：
- 推荐期刊 Top N（默认 5）
- 推荐理由（可解释）
- 匹配标签
- 期刊基本信息（OA、分区、投稿链接等）
- 推荐置信度

---

## 二、系统整体架构

系统采用模块化设计，包含以下核心模块：

1. 期刊数据模块（Journal Corpus）
2. 论文解析模块（Paper Parser）
3. 候选召回模块（Retriever）
4. 排序模块（Ranker）
5. 推荐解释模块（Explainer）
6. API 服务模块（API）
7. 评测模块（Evaluation）

推荐流程：

Paper Input → Paper Profile → Candidate Retrieval → Ranking → Explanation → Output

---

## 三、核心数据结构

### 1. Journal 数据结构

```json
{
  "journal_id": "",
  "journal_name": "",
  "publisher": "",
  "scope_text": "",
  "subject_tags": [],
  "keywords": [],
  "oa_type": "",
  "submission_url": "",
  "homepage_url": "",
  "quartile": "",
  "impact_like_score": "",
  "review_time": "",
  "apc": ""
}
```
### 2. PaperProfile 数据结构

```json
{
  "title": "",
  "abstract": "",
  "full_text_summary": "",
  "research_area": [],
  "method_type": [],
  "paper_type": "",
  "keywords": [],
  "novelty": "",
  "application_domain": [],
  "difficulty_level": "",
  "style": ""
}
```
### 3. Recommendation 数据结构

```json
{
  "journal_id": "",
  "journal_name": "",
  "score": 0.0,
  "confidence": 0.0,
  "match_reasons": [],
  "matched_fields": [],
  "tags": []
}
```

## Agent skills

### Issue tracker

GitHub Issues (Qianggggggggg/journal-recommender). See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` at repo root + `docs/adr/` for architectural decisions. See `docs/agents/domain.md`.