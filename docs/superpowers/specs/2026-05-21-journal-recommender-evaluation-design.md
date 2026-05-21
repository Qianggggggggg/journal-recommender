# 期刊推荐评估系统设计

## 1. 目标

构建离线评估流程，验证系统推荐的准确率。

核心思路：爬取 DBLP 论文-期刊对应关系作为 ground truth，用系统对论文进行推荐，检查实际发表期刊是否在推荐列表中，以及论文质量与期刊等级是否匹配。

## 2. 评估数据集构建

### 数据来源
- **DBLP**：公开的论文-期刊对应关系
- **规模**：100-200 篇论文，覆盖不同领域（AI/NLP/CV/Security 等）
- **字段**：标题、摘要、发表期刊名称、期刊 CCF 等级

### 数据格式
```json
{
  "title": "论文标题",
  "abstract": "论文摘要（可选）",
  "published_journal": "IEEE TPAMI",
  "ccf_rating": "A",
  "research_area": ["计算机视觉", "深度学习"]
}
```

### 存储位置
`data/evaluation/ground_truth.jsonl`

## 3. 评估流程

### 3.1 对每篇论文运行推荐
- 输入：论文标题/摘要
- 输出：推荐期刊 Top-5（含 CCF 等级和置信度）

### 3.2 检查指标

| 指标 | 定义 |
|------|------|
| **Hit@5** | 实际发表期刊在推荐 Top-5 中 |
| **Level Match Rate** | 论文质量等级与推荐期刊 CCF 等级匹配 |
| **Quality Hit Rate** | 分质量等级（强/中/弱）的 Hit@5 |

## 4. 评估指标详解

### 4.1 Hit@5
```
Hit@5 = (实际期刊在 Top-5 中的论文数) / (总论文数)
```

### 4.2 论文质量等级划分
基于 PaperQualityAssessor 输出的 paper_strength：
- **强论文**：paper_strength >= 0.7
- **中论文**：0.4 <= paper_strength < 0.7
- **弱论文**：paper_strength < 0.4

### 4.3 Level Match
| 论文质量 | 推荐期刊 CCF 等级 |
|----------|-------------------|
| 强 | A 或 B |
| 中 | B 或 C |
| 弱 | C 或非 CCF |

### 4.4 分质量等级的 Hit@5
分别计算强/中/弱论文的 Hit@5，观察系统是否偏向推荐高/低级别期刊。

## 5. 评估脚本

### 5.1 数据采集脚本
`scripts/crawl_dblp_evaluation.py`
- 从 DBLP 爬取论文数据
- 人工标注或自动匹配期刊 CCF 等级
- 输出 ground truth 数据集

### 5.2 评估脚本
`scripts/evaluate_recommender.py`
```
输入：ground_truth.jsonl
输出：评估报告（Hit@5、Level Match Rate 等）
```

### 5.3 输出格式
```
=== 评估报告 ===
总论文数：150

--- Hit@5 ---
Hit@5: 65.3% (98/150)
Top-1: 32.0%
Top-3: 51.3%
Top-5: 65.3%

--- Level Match Rate ---
强论文 (n=50): 72.0%
中论文 (n=70): 64.3%
弱论文 (n=30): 53.3%
Overall: 65.3%

--- 按领域分布 ---
计算机视觉: 68.0%
自然语言处理: 62.5%
人工智能: 70.0%
```

## 6. 数据采集方案

### DBLP 爬取策略
1. 按领域关键词搜索（如 "computer vision", "neural network"）
2. 过滤有摘要的论文（确保可推荐）
3. 期刊名称匹配 CCF 分类

### 期刊 CCF 等级匹配
- 使用 `data/journals_ccf.jsonl` 中的 CCF 分类
- 未收录期刊标记为"非 CCF"

## 7. 实现步骤

1. **构建评估数据集**
   - [ ] 编写 DBLP 爬取脚本
   - [ ] 采集 100-200 篇论文
   - [ ] 标注 CCF 等级
   - [ ] 保存到 ground_truth.jsonl

2. **实现评估脚本**
   - [ ] 加载 ground truth 数据
   - [ ] 对每篇论文运行推荐
   - [ ] 计算 Hit@5、Level Match Rate
   - [ ] 输出评估报告

3. **运行评估**
   - [ ] 执行评估脚本
   - [ ] 分析结果
   - [ ] 根据结果优化系统

## 8. 预期结果

预期 Hit@5 在 60-70% 左右（领域匹配场景）。如果低于 50%，需要检查：
- 论文特征提取是否准确
- CCF 分类是否完整
- 召回策略是否涵盖足够候选期刊