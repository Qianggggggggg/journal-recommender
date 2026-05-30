# 生成语义锚点 + 自适应检索与排序框架

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为学术期刊推荐系统构建一个生成式语义增强 + 自适应检索与排序的联合学习框架，提升推荐效果并支撑 CCF-A 级别论文的系统性方法创新贡献。

**Architecture:** 三阶段流程：（1）生成式语义锚点检索——用 LLM 为每本期刊生成多面典型摘要，结合 DBLP 真实标题作为 Few-shot 上下文；（2）动态门控网络——根据论文 profile 自适应融合 BM25/向量/关键词三路召回权重；（3）两阶段排序——伪孪生双塔粗排 + 轻量交叉编码器精排。三个阶段各自对应一个明确的理论贡献点，构成完整的方法论体系。

**Tech Stack:** Python 3.10+, PyTorch, Sentence-Transformers/SciBERT, DeBERTa-tiny, Elasticsearch/FAISS, BM25Plus, GPT-4o-mini (摘要生成), BeautifulSoup (DBLP 爬虫)

---

## 背景与动机

当前系统的核心瓶颈：期刊的语义表征仅依赖 `scope_text` 和少量关键词，粒度停留在"领域级"而非"子方向/方法级"。这导致三路召回（BM25、向量、关键词）对新兴方法论论文的匹配性能显著下降。

本框架的核心洞察：期刊的语义空间是**多峰的**——一本期刊可能同时接收系统设计型、理论分析型、实验评估型论文，但每种类型的发表风格和关注点不同。用 LLM 为每本期刊生成一组"多面典型摘要"，将检索目标从单一的领域描述转化为多个语义锚点，从根本上解决分布方差问题。

---

## 核心贡献点

### 贡献一：生成式语义锚点（Generated Semantic Anchors）

为每本期刊生成 4 篇典型摘要，覆盖两个正交维度：
- **方法主导类型**：理论分析型、系统设计型、实验评估型、算法改进型、应用驱动型（取 1 种）
- **创新层次**：范式创新、显著改进、增量完善（取 1 种）

生成输入包括：期刊 scope_text、CCF 等级、DBLP 2020-2024 真实标题 5-10 条作为 Few-shot 示例。生成后清洗噪声，得到结构化的典型摘要库（295 刊 × 4 篇 = 1,180 篇）。

**信息论解释**：典型摘要子空间比单一 scope 文本的语义分布方差更小，最小化论文-期刊匹配的 KL 散度上界。

### 贡献二：内容感知动态证据融合（Content-Aware Dynamic Evidence Fusion）

设计门控网络 `G(·)`，输入论文的多维度 profile 向量（方法类型 one-hot、创新等级 one-hot、关键词 embedding 拼接），输出 BM25/向量/关键词三路信号的动态融合权重：

```
[w_bm25, w_vec, w_text] = softmax(G(paper_profile))
```

训练信号：从 100 篇标注数据出发，对每篇论文离线搜索最优权重组合作为伪标签，训练门控网络拟合。推理时根据论文内容自适应决定检索策略。

### 贡献三：伪孪生交互排序（Pseudo-Siamese Interactive Ranking）

**两阶段架构**：

**阶段 2a——双塔粗排**：
- 论文塔：SciBERT/SPECTER 编码论文标题+摘要
- 期刊塔：4 篇典型摘要分别编码后，经多维注意力池化得到期刊表示
- 损失：InfoNCE 对比损失，拉近论文与正例期刊，推远负例期刊
- 输出：Top-20 候选

**阶段 2b——交叉编码器精排**：
- 输入：`[CLS] paper_title [SEP] paper_abstract [SEP] journal_typical_abstracts + scope_text [SEP]`
- 额外特征嵌入：CCF 等级差、方法类型匹配度
- 模型：DeBERTa-tiny / BERT-mini（<15M 参数）
- 损失：LambdaRank pairwise 损失
- 训练数据构造：从双塔粗排结果中提取 hard negatives，扩大有效训练规模

---

## 实施计划

### 当前实现状态（截至 2026-05-30）

| 模块 | 状态 | 证据 | 下一步 |
|-----|------|------|--------|
| DBLP 标题爬取 | 部分完成 | `data/dblp_titles/` 已有 183 个期刊标题文件；`src/generation/dblp_crawler.py` 已实现基础爬取与非研究条目过滤 | 不作为短期主线，后续仅在需要重生成典型摘要或做 Few-shot 消融时补齐覆盖率 |
| 典型摘要库 | 已完成基础版本 | `data/typical_abstracts/` 覆盖 295 刊，共 1,180 篇，每刊 4 篇 | 先做质量抽检与召回消融；不要立即重生成 |
| 训练数据 | 已完成基础版本 | `data/training/training_pairs.json` 已按 60/20/20 划分，且每篇含 50 个 BM25 负样本 | 后续补充真实 vector/text hard negatives，用于双塔和门控训练 |
| 典型摘要索引 | 已完成基础版本 | `data/processed/typical_abstracts_index.faiss` 与 metadata 已存在；API 已能加载典型摘要 BM25/text/vector 召回 | 优先做 scope-only、typical-only、hybrid 三组消融 |
| 混合召回策略 | 已接入，需重新表述 | 当前 `CandidateGenerator` 采用 scope 作为身份边界、typical 作为语义扩展，而不是完全替换 scope | 文档改为“scope 边界 + typical 扩展”的混合设计 |
| 门控网络 | 部分完成，暂不作为核心主线 | `src/ranker/gating_network.py`、`scripts/train_gating_network.py` 和 `data/models/gating_network.pt` 已存在；但 `configs/app.yaml` 仍为 fixed，API 未加载 gater | 修正伪标签脚本与 API 接入后再做消融；若无显著提升则降级为附加实验 |
| 双塔粗排 | 骨架完成，未训练接入 | `src/ranker/siamese_ranker.py` 已有 SciBERT/SPECTER fallback、注意力池化和 InfoNCE；尚无训练脚本和在线接入 | 作为下一阶段高优先级：先离线训练和消融，再决定是否接入推荐链路 |
| 交叉编码器精排 | 骨架完成，建议暂缓 | `src/ranker/cross_ranker.py` 已有 pair encoder 与 LambdaRank loss；但训练数据规模较小，且当前已有 LLM 精排 | 暂缓，等双塔粗排证明收益后再启动 |
| 评测体系 | 已有基础指标，需补消融组织 | `scripts/run_evaluation.py` 已支持 Hit@K、MRR、NDCG、粗召回命中和 Rule topK 命中 | 新增可复现实验配置，统一输出 scope/typical/hybrid/双塔对照 |

### 阶段 0：数据准备（第 1-2 个月）

#### 任务 0.1：DBLP 标题爬取

- [x] 编写 DBLP 爬虫，按期刊名称搜索 2020-2024 年发表论文
- [x] 每刊最多取 50 篇英文标题，去除 special issue、book review 等非研究性条目
- [x] 存储结构：`journal_id → [title_1, title_2, ...]`
- [ ] 覆盖率目标：295 刊中至少 90% 成功爬取（当前约 183/295；短期不阻塞主线）

#### 任务 0.2：典型摘要生成

- [x] 设计生成 Prompt（包含维度定义与结构化 JSON 输出）
- [x] 批量生成：295 刊 × 4 摘要 = 1,180 篇
- [x] 清洗：去除格式噪声、截断超长摘要
- [x] 存储结构：`journal_id → abstracts[]`
- [ ] 可选增强：将 DBLP 标题作为 Few-shot 上下文重生成或补生成，并做“有/无 DBLP Few-shot”消融

#### 任务 0.3：训练数据整理

- [x] 整理 100 篇标注论文-期刊对
- [x] 划分：训练集 60 篇、验证集 20 篇、测试集 20 篇（按期刊领域分层抽样）
- [x] 为训练集构造负样本：BM25 召回 Top-50 中排除正例的期刊
- [ ] 补充 vector/text hard negatives，避免训练信号只反映 BM25 排序

### 阶段 1：检索增强（第 2-3 个月）

#### 任务 1.1：典型摘要索引构建

- [x] 将 1,180 篇典型摘要接入现有检索系统（BM25 索引 + FAISS 向量索引）
- [x] 修改 CandidateGenerator：采用 `scope_text` 身份边界 + 典型摘要语义扩展的混合召回
- [ ] 验证：系统性对比 scope-only、typical-only、hybrid 三种召回策略的 Hit@K、MRR、NDCG 与粗召回命中

#### 任务 1.2：门控网络实现

- [x] 实现 `G(·)` 网络（全连接 + softmax，输入 paper_profile embedding）
- [ ] 修正并统一伪标签生成脚本，使其使用真实 BM25/vector/text 三路召回结果
- [x] 用伪标签训练门控网络（已有基础 checkpoint）
- [ ] 将 `weighting_strategy: gating` 接入 API/pipeline 初始化
- [ ] 消融：固定权重 vs. 动态门控在召回阶段的 Hit@K 对比；若提升不稳定则降级为附加实验

### 阶段 2：两阶段排序（第 3-5 个月）

#### 任务 2.1：双塔粗排模型

- [x] 论文塔：加载 SciBERT/SPECTER 预训练模型（本地无权重时 fallback 到 hashing encoder）
- [x] 期刊塔：4 篇典型摘要各自编码 → 多维注意力池化
- [x] 实现 InfoNCE 对比损失
- [ ] 在训练集上训练，验证集调参
- [ ] 消融：典型摘要池化 vs. 单一 scope 文本的双塔效果对比
- [ ] 先离线评估双塔粗排，不默认接入线上推荐链路

#### 任务 2.2：交叉编码器精排模型

- [x] 实现 DeBERTa-tiny 精排模型骨架（本地无权重时 fallback 到 hashing pair encoder）
- [x] 实现 LambdaRank pairwise 损失
- [ ] 从双塔粗排结果构造训练对（正例 + hard negatives）
- [ ] 训练并调参
- [ ] 消融：去交叉编码器（只用双塔）vs. 完整两阶段的效果对比
- [ ] 当前建议暂缓：待双塔粗排证明稳定收益后再启动

### 阶段 3：实验与评估（第 5-7 个月）

#### 任务 3.1：基线对比实验

| 基线 | 描述 |
|-----|------|
| **现有系统** | 三路固定权重 + LLM 精排 |
| **固定权重 BM25+Vector** | 移除关键词路，固定 0.45/0.35/0.20 权重 |
| **SciBERT 向量检索** | 纯向量检索，无混合召回 |
| **JournalFinder** | Elsevier 官方期刊推荐工具（API 或逻辑复现）|
| **Springer Journal Suggester** | Springer 官方工具（API 或逻辑复现）|
| **TF-IDF 关键词匹配** | 经典文本匹配基线 |

指标：MRR、NDCG@5/10、Hit@3/5/10、CCF 等级匹配精度（Q 值加权）

#### 任务 3.2：消融实验

| 消融项 | 预期影响 |
|-------|---------|
| 去除典型摘要（回退到 scope_text）| 召回质量下降，尤其是跨类型论文 |
| 去除门控网络（固定权重）| 动态权重优势，对新兴方法论文影响更大 |
| 去除双塔粗排（直接全文向量精排）| 计算成本变化 + 可能降低召回多样性 |
| 去除交叉编码器精排（只用双塔）| 精排质量下降，Top-5/10 指标回落 |
| 去除 DBLP 标题 Few-shot | 生成摘要质量下降（需人工评估子集）|

#### 任务 3.3：泛化与鲁棒性实验

- **跨子领域泛化**：按 CCF 子领域（人工智能、网络安全、软件工程等）划分训练/测试集，验证框架对未见领域的适应性
- **时间泛化**：用 2015-2019 论文训练，2020-2024 论文测试，验证时间稳定性
- **方法类型鲁棒性**：按方法类型（理论/系统/应用）分层统计各子群体的 Hit@K

#### 任务 3.4：可解释性量化评估

**人工评估**：
- 招募 3 位研究人员
- 对 30 篇论文的推荐结果随机抽取 5 条推荐（Top-5）
- 每条推荐理由按"可信度"和"帮助度"打 1-5 分
- 报告评分者间一致性（Cohen's κ）

**自动评估**：
- 计算推荐理由与论文实际内容的关键词重叠度（ROUGE-L）
- 对比本框架理由 vs. 固定权重基线理由的 ROUGE-L 分布

### 阶段 4：论文撰写（第 7-9 个月）

#### 目标格式

TKDE / TPAMI 格式（长文，12-16 页正文 + 参考文献）

#### 建议结构

1. **Introduction**：问题定义、现有系统不足、本文贡献（三个贡献点明确列出）
2. **Related Work**：期刊推荐系统、知识增强检索、学习排序、学术文本匹配
3. **Preliminary**：现有系统流程、问题形式化
4. **生成式语义锚点**：方法、维度设计、生成流程、信息论分析
5. **内容感知动态融合**：门控网络设计、权重学习策略
6. **伪孪生交互排序**：双塔架构、注意力池化、两阶段训练
7. **实验**：数据集、基线对比、消融分析、泛化实验、可解释性评估
8. **Discussion & Future Work**
9. **Conclusion**

#### 理论包装要点

- 典型摘要的信息论解释（KL 散度上界、最小化语义方差）
- 门控网络的元学习视角
- 对比学习约束与排序表示学习的联系

---

## 文件结构

```
src/
├── retriever/
│   ├── candidate_generator.py      # 修改：切换为典型摘要检索
│   ├── bm25_retriever.py          # 修改：支持典型摘要索引
│   └── embedding_retriever.py     # 修改：支持典型摘要向量索引
├── ranker/
│   ├── gating_network.py           # 新增：动态门控网络
│   ├── siamese_ranker.py          # 新增：双塔粗排模型
│   └── cross_ranker.py            # 新增：交叉编码器精排模型
├── generation/
│   ├── typical_abstract_generator.py  # 新增：典型摘要生成
│   └── dblp_crawler.py             # 新增：DBLP 标题爬虫
├── data/
│   ├── typical_abstracts/          # 新增：生成的典型摘要库
│   └── dblp_titles/               # 新增：DBLP 标题数据
tests/
├── test_gating_network.py         # 新增
├── test_siamese_ranker.py         # 新增
├── test_cross_ranker.py           # 新增
└── test_typical_abstracts.py      # 新增
```

---

## 风险与缓解

| 风险 | 概率 | 影响 | 缓解策略 |
|-----|-----|-----|---------|
| DBLP 爬虫被禁/数据不足 | 中 | 高 | 准备备用数据源（Semantic Scholar API）；减少每刊爬取量至 30 篇 |
| 典型摘要质量不佳 | 中 | 中 | 人工抽检 10% 生成结果；Few-shot 示例中加入质量好的真实摘要示范 |
| 100 篇数据训练交叉编码器过拟合 | 中 | 中 | hard negative mining + 早停；或回退到纯双塔路线 |
| 门控网络伪标签质量不足 | 低 | 中 | 改用强化学习或直接优化最终排序指标 |
| 外部工具基线无法调用 | 低 | 低 | 复现其核心 TF-IDF/关键词匹配逻辑作为替代 |

---

## 成功标准

### 系统效果

当前基线以 `data/evaluation/results/eval_abstract_top5_20260530_010215.json` 为准，评测集共 102 条；其中 `paper_results` 实际返回 100 条有效结果。后续报告必须同时给出样本数、配置、随机种子、召回目标和是否启用 LLM 精排，避免与早期 33 条抽样结果混用。

| 指标 | 当前基线（现有系统） | 下一阶段目标 | 说明 |
|-----|---------------------|--------------|------|
| Hit@5 | 64/102（62.7%） | hybrid 召回与排序优化后达到 68%-72%；若接入双塔，目标达到 72%+ | 不再使用早期 24/33 作为主基线 |
| Hit@1 | 27/102（26.5%） | 提升到 30%+ | 衡量最终排序头部质量 |
| 粗召回命中@50 | 90/102（88.2%） | 稳定保持 88%+，优先提升进入 Rule/双塔 Top20 的比例 | 当前主要瓶颈在召回后排序，不是完全召不回 |
| Rule Top20 命中 | 68/102（66.7%） | 提升到 75%+ | 双塔粗排的主要优化目标 |
| MRR | 0.3956 | 提升到 0.43+ | 按评测脚本已累计后的均值口径报告 |
| NDCG@5 | 0.4531 | 提升到 0.49+ | 与 Hit@5 同时报告，避免只看命中率 |
| 同领域命中@5 | 91/102（89.2%） | 不低于当前基线 | 防止为追求真实 venue 命中牺牲领域相关性 |

### 阶段性验收

| 阶段 | 必须达成 | 否则处理 |
|-----|----------|----------|
| 典型摘要召回消融 | scope-only、typical-only、hybrid 三组实验可复现，并输出 Hit@K、MRR、NDCG、粗召回命中 | 若 typical-only 漂移明显，保留 hybrid，不推进完全替换 scope |
| 门控网络 | fixed vs. gating 的召回指标提升稳定，且不会降低 scope 边界候选覆盖 | 若提升不稳定，门控降级为附加实验，不进入主流程 |
| 双塔粗排 | 在验证集上提升 Rule/双塔 Top20 命中，并在测试集上不低于现有 hybrid baseline | 若只提升训练集，保留离线实验，不接入线上推荐链路 |
| 交叉编码器精排 | 双塔已证明收益后再启动；必须优于 LLM 精排候选前置策略或显著降低推理成本 | 若数据量不足或过拟合，暂缓到论文后期实验 |

### 论文贡献

- 至少一个核心贡献点有完整的**方法描述 + 消融实验 + 误差分析**支撑；当前优先级为“生成式语义锚点 + hybrid 召回”。
- 双塔粗排若验证有效，可作为第二贡献点；若提升有限，则作为附加学习排序实验呈现。
- 门控网络和交叉编码器不再默认作为必须贡献点，只有在消融稳定提升时进入主论文叙事。
- 外部系统基线（JournalFinder、Springer Journal Suggester）放到后期可选实验；短期主基线应优先保证本地可复现、公平可控。
- 可解释性评估先使用自动证据（召回来源、匹配摘要、关键词/领域对齐），人工评估作为论文后期增强项。

---

## 时间线（9 个月）

```
Month 1-2:   数据准备（DBLP 爬虫 + 典型摘要生成 + 训练数据整理）
Month 2-3:   检索增强（典型摘要索引 + 门控网络）
Month 3-5:   两阶段排序（双塔 + 交叉编码器）
Month 5-7:   实验与消融（基线 + 消融 + 泛化 + 可解释性）
Month 7-9:   论文撰写 + 投稿
```

---

## 待解答问题（设计阶段保留）

1. [ ] 门控网络训练的超参搜索空间（论文 profile 哪些维度进入 embedding）
2. [ ] 双塔模型的负样本策略（in-batch negatives vs. 外部负样本）
3. [ ] 典型摘要的段落长度限制（是否需要截断到固定长度）
4. [ ] 论文审稿人可能的"非端到端"质疑如何回应（当前设计保留了 LLM 生成环节）

以上问题在实施阶段根据初期实验结果动态调整，不阻塞整体框架推进。
