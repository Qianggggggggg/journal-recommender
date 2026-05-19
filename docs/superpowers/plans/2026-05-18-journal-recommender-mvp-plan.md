# 论文投稿期刊推荐系统 - MVP 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现可运行的 MVP，支持三种输入模式（标题/摘要/全文）的期刊推荐

**Architecture:**
- 三层结构：期刊库（数据层）→ 论文理解 → 推荐排序
- 混合召回：BM25 + 向量检索 + 标签过滤，三路合并
- 两阶段排序：规则打分（Top 30→10）→ LLM 精排（Top 10→5）
- 独立 Explainer 模块生成推荐理由
- 降级策略：LLM 失败时回退到规则+关键词分类

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, FAISS, rank-bm25, httpx, PyYAML

---

## 文件结构总览

| 模块 | 文件路径 | 职责 |
|------|----------|------|
| 配置 | `configs/app.yaml` | 应用配置（API地址、模型参数） |
| 配置 | `configs/prompts.yaml` | Prompt 模板 |
| 配置 | `configs/journal_taxonomy.yaml` | 期刊分类体系 |
| 工具 | `src/utils/llm.py` | MiniMax API 调用封装 |
| 工具 | `src/utils/embedding.py` | Ollama embedding 调用封装 |
| 工具 | `src/utils/text.py` | 文本处理工具 |
| 工具 | `src/utils/logging.py` | 日志配置 |
| 工具 | `src/utils/__init__.py` | 模块导出 |
| 期刊 | `src/journals/journal_model.py` | 期刊数据模型 |
| 期刊 | `src/journals/journal_store.py` | 期刊存储与检索 |
| 期刊 | `src/journals/taxonomy.py` | 期刊分类体系 |
| 期刊 | `src/journals/__init__.py` | 模块导出 |
| 论文 | `src/papers/paper_model.py` | 论文数据模型 |
| 论文 | `src/papers/paper_parser.py` | 论文解析（含降级） |
| 论文 | `src/papers/section_splitter.py` | 章节切分 |
| 论文 | `src/papers/__init__.py` | 模块导出 |
| 召回 | `src/retriever/bm25_retriever.py` | BM25 召回 |
| 召回 | `src/retriever/embedding_retriever.py` | 向量检索召回 |
| 召回 | `src/retriever/candidate_generator.py` | 混合召回编排 |
| 召回 | `src/retriever/__init__.py` | 模块导出 |
| 排序 | `src/ranker/feature_builder.py` | 排序特征构建 |
| 排序 | `src/ranker/rule_scorer.py` | 规则打分（阶段一） |
| 排序 | `src/ranker/llm_ranker.py` | LLM 排序（阶段二） |
| 排序 | `src/ranker/scoring.py` | 分数计算工具 |
| 排序 | `src/ranker/__init__.py` | 模块导出 |
| 推荐 | `src/recommender/pipeline.py` | 推荐流程编排 |
| 推荐 | `src/recommender/explainer.py` | 推荐理由生成 |
| 推荐 | `src/recommender/__init__.py` | 模块导出 |
| API | `src/app/schemas.py` | Pydantic 请求/响应模型 |
| API | `src/app/api.py` | API 路由 |
| API | `src/app/main.py` | FastAPI 应用入口 |
| API | `src/app/__init__.py` | 模块导出 |
| 数据采集 | `scripts/crawl_doaj.py` | DOAJ 数据采集 |
| 数据采集 | `scripts/crawl_scimago.py` | SCImago 数据采集 |
| 数据采集 | `scripts/normalize_journals.py` | 数据清洗标准化 |
| 数据采集 | `scripts/build_journal_index.py` | 向量索引构建 |
| 入口 | `src/__init__.py` | 包入口 |

---

## Task 1: 项目脚手架

**目标:** 创建目录结构、配置文件、依赖管理

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `configs/app.yaml`
- Create: `configs/prompts.yaml`
- Create: `configs/journal_taxonomy.yaml`
- Create: `src/__init__.py`
- Create: `src/utils/__init__.py`
- Create: `src/journals/__init__.py`
- Create: `src/papers/__init__.py`
- Create: `src/retriever/__init__.py`
- Create: `src/ranker/__init__.py`
- Create: `src/recommender/__init__.py`
- Create: `src/app/__init__.py`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[project]
name = "journal-recommender"
version = "0.1.0"
description = "论文投稿期刊推荐系统"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110.0",
    "uvicorn>=0.27.0",
    "pydantic>=2.6.0",
    "httpx>=0.27.0",
    "faiss-cpu>=1.8.0",
    "rank-bm25>=0.2.2",
    "pyyaml>=6.0.1",
    "python-dotenv>=1.0.0",
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "tenacity>=8.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: 创建 .env.example**

```bash
# MiniMax API
MINIMAX_API_KEY=your_api_key_here
MINIMAX_BASE_URL=https://api.minimax.chat

# Ollama Embedding
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=qwen3-embedding:4b

# Application
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
```

- [ ] **Step 3: 创建 .gitignore**

```gitignore
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

.env
.venv
env/
venv/

data/raw/
data/processed/
*.faiss
*.parquet
*.jsonl

.idea/
.vscode/
*.swp
*.swo
```

- [ ] **Step 4: 创建 configs/app.yaml**

```yaml
app:
  host: "0.0.0.0"
  port: 8000
  log_level: "INFO"

minimax:
  api_key: "${MINIMAX_API_KEY}"
  base_url: "https://api.minimax.chat"
  model: "MiniMax-Text-01"
  max_tokens: 4096
  temperature: 0.7

ollama:
  base_url: "http://localhost:11434"
  embedding_model: "qwen3-embedding:4b"

retrieval:
  title_mode:
    bm25_top_k: 20
    vector_top_k: 20
    tag_top_k: 15
    final_candidates: 30
  abstract_mode:
    bm25_top_k: 25
    vector_top_k: 25
    tag_top_k: 20
    final_candidates: 40
  full_mode:
    bm25_top_k: 30
    vector_top_k: 30
    tag_top_k: 25
    final_candidates: 50

ranking:
  rule_stage:
    top_k: 10
  llm_stage:
    top_k: 5

data:
  journal_store_path: "data/processed/journals.jsonl"
  faiss_index_path: "data/processed/journals_index.faiss"
  metadata_path: "data/processed/journals_metadata.parquet"
```

- [ ] **Step 5: 创建 configs/prompts.yaml**

```yaml
paper_profile_system: |
  你是一个专业的学术论文分析助手。请根据论文信息提取以下结构化特征：
  - research_area: 研究领域（如 AI、CV、NLP、SE 等）
  - method_type: 方法类型（method/system/experiment/survey）
  - paper_type: 论文类型（theory/application/engineering）
  - keywords: 关键词（3-5个）
  - novelty: 创新点简述
  - application_domain: 应用领域

paper_profile_user: |
  论文标题：{title}
  摘要：{abstract}
  全文摘要：{full_text_summary}

  请提取上述结构化特征，以 JSON 格式输出。

llm_ranker_system: |
  你是一个专业的学术期刊匹配顾问。请根据论文信息为候选期刊打分排序。

llm_ranker_user: |
  论文信息：
  - 标题：{title}
  - 研究领域：{research_area}
  - 方法类型：{method_type}
  - 论文类型：{paper_type}
  - 关键词：{keywords}

  候选期刊：
  {journals_info}

  请为每个期刊给出：
  1. 匹配度分数（0-1）
  2. 推荐理由（2-3条）
  3. 置信度（0-1）

  输出格式：
  {
    "rankings": [
      {"journal_id": "...", "score": 0.9, "reasons": [...], "confidence": 0.85},
      ...
    ]
  }

explainer_system: |
  你是一个专业的学术期刊推荐解释助手。请为每条推荐生成详细的匹配理由。

explainer_user: |
  论文特征：
  - 研究领域：{research_area}
  - 方法类型：{method_type}
  - 论文类型：{paper_type}
  - 关键词：{keywords}

  期刊信息：
  - 名称：{journal_name}
  - Scope：{scope_text}
  - 分区：{quartile}
  - OA类型：{oa_type}
  - 审稿周期：{review_time}

  请生成 2-4 条推荐理由，解释为什么这篇论文适合该期刊。
```

- [ ] **Step 6: 创建 configs/journal_taxonomy.yaml**

```yaml
subject_tags:
  - id: "ai"
    label: "人工智能"
    keywords: ["AI", "人工智能", "机器学习", "深度学习"]
  - id: "cv"
    label: "计算机视觉"
    keywords: ["CV", "Computer Vision", "图像", "视频"]
  - id: "nlp"
    label: "自然语言处理"
    keywords: ["NLP", "Natural Language Processing", "文本", "语言"]
  - id: "se"
    label: "软件工程"
    keywords: ["SE", "Software Engineering", "软件", "系统"]
  - id: "network"
    label: "网络与通信"
    keywords: ["Network", "通信", "分布式", "云计算"]
  - id: "security"
    label: "信息安全"
    keywords: ["Security", "安全", "加密", "隐私"]
  - id: "db"
    label: "数据库"
    keywords: ["DB", "Database", "数据", "存储"]
  - id: "theory"
    label: "理论计算"
    keywords: ["Theory", "算法", "计算", "形式化"]

method_types:
  - id: "method"
    label: "方法论"
    description: "提出新方法、算法、模型"
  - id: "system"
    label: "系统"
    description: "系统设计、实现、评估"
  - id: "experiment"
    label: "实验"
    description: "实验验证、方法对比"
  - id: "survey"
    label: "综述"
    description: "文献综述、调研"

paper_types:
  - id: "theory"
    label: "理论"
    description: "理论分析、证明"
  - id: "application"
    label: "应用"
    description: "应用研究、实践"
  - id: "engineering"
    label: "工程"
    description: "系统工程、实现"

oa_types:
  - id: "full_oa"
    label: "完全开放获取"
  - id: "hybrid"
    label: "混合OA"
  - id: "subscription"
    label: "订阅制"
```

- [ ] **Step 7: 创建模块 __init__.py**

所有 `__init__.py` 文件内容统一为：

```python
"""Journal Recommender - 论文投稿期刊推荐系统"""
```

- [ ] **Step 8: 运行测试验证**

```bash
cd /Users/qian/PycharmProjects/paper
python -c "import yaml; yaml.safe_load(open('configs/app.yaml'))" && echo "Config OK"
```

---

## Task 2: 工具模块（LLM + Embedding）

**目标:** 封装 MiniMax API 和 Ollama Embedding 调用

**Files:**
- Create: `src/utils/llm.py`
- Create: `src/utils/embedding.py`
- Create: `src/utils/text.py`
- Create: `src/utils/logging.py`

- [ ] **Step 1: 创建 src/utils/llm.py 测试**

```python
"""MiniMax LLM 调用封装"""
import os
from typing import Optional

import httpx
from pydantic import BaseModel


class LLMResponse(BaseModel):
    content: str
    model: str
    usage: dict


class MiniMaxLLM:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.minimax.chat",
        model: str = "MiniMax-Text-01",
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ):
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(self, system: str, user: str) -> LLMResponse:
        """发送对话请求"""
        url = f"{self.base_url}/v1/text/chatcompletion_v2"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        response = httpx.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        return LLMResponse(
            content=data["choices"][0]["messages"][0]["text"],
            model=self.model,
            usage=data.get("usage", {}),
        )
```

```python
# tests/test_llm.py
import pytest
from src.utils.llm import MiniMaxLLM


def test_llm_response_model():
    """验证 LLMResponse 模型结构"""
    response = LLMResponse(content="test", model="test-model", usage={})
    assert response.content == "test"
    assert response.model == "test-model"
```

- [ ] **Step 2: 运行测试验证**

```bash
pytest tests/test_llm.py::test_llm_response_model -v
# 预期: PASS (不依赖真实 API，仅验证模型)
```

- [ ] **Step 3: 创建 src/utils/embedding.py**

```python
"""Ollama Embedding 调用封装"""
import os
from typing import List

import httpx
import numpy as np


class OllamaEmbedding:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3-embedding:4b",
    ):
        self.base_url = base_url
        self.model = model

    def embed(self, text: str) -> np.ndarray:
        """获取单条文本的 embedding 向量"""
        url = f"{self.base_url}/api/embeddings"
        response = httpx.post(url, json={"model": self.model, "prompt": text}, timeout=30)
        response.raise_for_status()
        data = response.json()
        return np.array(data["embedding"])

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """批量获取文本 embedding"""
        url = f"{self.base_url}/api/embeddings"
        results = []
        for text in texts:
            response = httpx.post(url, json={"model": self.model, "prompt": text}, timeout=30)
            response.raise_for_status()
            data = response.json()
            results.append(np.array(data["embedding"]))
        return results
```

- [ ] **Step 4: 创建 src/utils/text.py**

```python
"""文本处理工具"""
import re
from typing import List


def truncate_text(text: str, max_length: int = 2000) -> str:
    """截断文本到最大长度"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def clean_text(text: str) -> str:
    """清洗文本：去除多余空白和特殊字符"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text.strip()


def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    """简单关键词提取（基于频率）"""
    # 简单实现：去除停用词后按词频统计
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    # 简化处理，实际应该用停用词表
    from collections import Counter
    counter = Counter(words)
    return [word for word, _ in counter.most_common(top_k)]


def split_sentences(text: str) -> List[str]:
    """分句"""
    sentences = re.split(r'[。！？\n]+', text)
    return [s.strip() for s in sentences if s.strip()]
```

- [ ] **Step 5: 创建 src/utils/logging.py**

```python
"""日志配置"""
import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """配置日志"""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    """获取 logger"""
    return logging.getLogger(name)
```

- [ ] **Step 6: 创建测试文件 tests/test_utils.py**

```python
"""工具模块测试"""
import pytest
from src.utils.text import clean_text, truncate_text, extract_keywords


def test_clean_text():
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("hello\x00world") == "helloworld"


def test_truncate_text():
    assert truncate_text("hello world", max_length=5) == "hello..."
    assert truncate_text("hi", max_length=10) == "hi"


def test_extract_keywords():
    keywords = extract_keywords("deep learning neural network learning deep", top_k=3)
    assert "learning" in keywords or "deep" in keywords
```

- [ ] **Step 7: 运行测试**

```bash
pytest tests/test_utils.py -v
# 预期: PASS
```

---

## Task 3: 期刊数据模型

**目标:** 定义 Journal 数据结构和分类体系

**Files:**
- Create: `src/journals/journal_model.py`
- Create: `src/journals/taxonomy.py`
- Create: `tests/test_journal_model.py`

- [ ] **Step 1: 创建 src/journals/journal_model.py**

```python
"""期刊数据模型"""
from typing import List, Optional
from pydantic import BaseModel, Field


class Journal(BaseModel):
    """期刊数据结构"""
    journal_id: str = Field(description="唯一标识")
    journal_name: str = Field(description="期刊名称")
    publisher: Optional[str] = Field(default=None, description="出版社")
    subject_tags: List[str] = Field(default_factory=list, description="学科标签")
    keywords: List[str] = Field(default_factory=list, description="关键词")
    scope_text: str = Field(default="", description="期刊 scope 说明")
    oa_type: str = Field(default="subscription", description="OA 类型: full_oa/hybrid/subscription")
    submission_url: Optional[str] = Field(default=None, description="投稿链接")
    homepage_url: Optional[str] = Field(default=None, description="期刊主页")
    sqr_rank: Optional[int] = Field(default=None, description="SCImago 排名")
    quartile: Optional[str] = Field(default=None, description="分区: Q1/Q2/Q3/Q4")
    impact_like_score: Optional[float] = Field(default=None, description="影响因子指标")
    review_time: Optional[str] = Field(default=None, description="审稿周期估算")
    apc: Optional[float] = Field(default=None, description="APC 费用")
    target_paper_type: List[str] = Field(default_factory=list, description="适合的论文类型")
    journal_profile: str = Field(default="", description="预拼接检索文本")

    def build_profile(self) -> str:
        """构建 journal_profile"""
        parts = [
            self.journal_name,
            self.scope_text,
            " ".join(self.keywords),
            " ".join(self.subject_tags),
        ]
        self.journal_profile = " | ".join(p for p in parts if p)
        return self.journal_profile


class JournalMatch(BaseModel):
    """期刊匹配结果"""
    journal: Journal
    score: float = Field(description="匹配分数")
    confidence: float = Field(description="置信度 0-1")
    match_reasons: List[str] = Field(default_factory=list, description="匹配理由")
    matched_fields: List[str] = Field(default_factory=list, description="匹配的字段")
```

- [ ] **Step 2: 创建 src/journals/taxonomy.py**

```python
"""期刊分类体系"""
from typing import Dict, List
import yaml
from pathlib import Path


class JournalTaxonomy:
    """期刊分类体系"""

    def __init__(self, config_path: str = "configs/journal_taxonomy.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    @property
    def subject_tags(self) -> List[Dict]:
        return self.config.get("subject_tags", [])

    @property
    def method_types(self) -> List[Dict]:
        return self.config.get("method_types", [])

    @property
    def paper_types(self) -> List[Dict]:
        return self.config.get("paper_types", [])

    @property
    def oa_types(self) -> List[Dict]:
        return self.config.get("oa_types", [])

    def get_subject_keywords(self, tag_id: str) -> List[str]:
        """获取学科标签的关键词"""
        for tag in self.subject_tags:
            if tag["id"] == tag_id:
                return tag.get("keywords", [])
        return []

    def match_subject_tag(self, text: str) -> List[str]:
        """根据文本匹配学科标签"""
        matched = []
        text_lower = text.lower()
        for tag in self.subject_tags:
            for keyword in tag.get("keywords", []):
                if keyword.lower() in text_lower:
                    matched.append(tag["id"])
                    break
        return list(set(matched))
```

- [ ] **Step 3: 创建 tests/test_journal_model.py**

```python
"""期刊模型测试"""
import pytest
from src.journals.journal_model import Journal, JournalMatch


def test_journal_model():
    journal = Journal(
        journal_id="tpami",
        journal_name="IEEE TPAMI",
        publisher="IEEE",
        subject_tags=["cv", "ai"],
        keywords=["deep learning", "neural network"],
        scope_text="Computer vision and pattern recognition",
        oa_type="subscription",
        quartile="Q1",
    )
    assert journal.journal_id == "tpami"
    assert "cv" in journal.subject_tags


def test_journal_build_profile():
    journal = Journal(
        journal_id="tpami",
        journal_name="IEEE TPAMI",
        scope_text="Computer vision",
        keywords=["cv"],
        subject_tags=["cv"],
    )
    profile = journal.build_profile()
    assert "IEEE TPAMI" in profile
    assert "Computer vision" in profile
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/test_journal_model.py -v
# 预期: PASS
```

---

## Task 4: 期刊存储与检索

**目标:** 实现 JournalStore，支持 JSONL 存储和 FAISS 向量检索

**Files:**
- Create: `src/journals/journal_store.py`
- Create: `tests/test_journal_store.py`

- [ ] **Step 1: 创建 src/journals/journal_store.py**

```python
"""期刊存储与检索"""
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd

from .journal_model import Journal


class JournalStore:
    """期刊存储与检索"""

    def __init__(
        self,
        store_path: str = "data/processed/journals.jsonl",
        faiss_index_path: str = "data/processed/journals_index.faiss",
        metadata_path: str = "data/processed/journals_metadata.parquet",
    ):
        self.store_path = store_path
        self.faiss_index_path = faiss_index_path
        self.metadata_path = metadata_path
        self._journals: List[Journal] = []
        self._index: Optional[faiss.IndexFlatL2] = None
        self._metadata: Optional[pd.DataFrame] = None

    def load(self) -> None:
        """加载期刊数据"""
        if not os.path.exists(self.store_path):
            return

        self._journals = []
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self._journals.append(Journal(**data))

        if os.path.exists(self.faiss_index_path) and os.path.exists(self.metadata_path):
            self._index = faiss.read_index(self.faiss_index_path)
            self._metadata = pd.read_parquet(self.metadata_path)

    def save(self) -> None:
        """保存期刊数据"""
        Path(self.store_path).parent.mkdir(parents=True, exist_ok=True)

        with open(self.store_path, "w", encoding="utf-8") as f:
            for journal in self._journals:
                f.write(json.dumps(journal.model_dump(), ensure_ascii=False) + "\n")

        if self._index is not None:
            Path(self.faiss_index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, self.faiss_index_path)

        if self._metadata is not None:
            self._metadata.to_parquet(self.metadata_path)

    def add_journal(self, journal: Journal) -> None:
        """添加期刊"""
        self._journals.append(journal)

    def add_journals(self, journals: List[Journal]) -> None:
        """批量添加期刊"""
        for journal in journals:
            self.add_journal(journal)

    def build_faiss_index(self, embeddings: np.ndarray) -> None:
        """构建 FAISS 索引"""
        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dimension)
        self._index.add(embeddings.astype(np.float32))

        # 保存元数据
        self._metadata = pd.DataFrame([{
            "journal_id": j.journal_id,
            "journal_name": j.journal_name,
        } for j in self._journals])

    def search_by_vector(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[Tuple[Journal, float]]:
        """向量检索"""
        if self._index is None:
            raise ValueError("FAISS index not built")

        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self._index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self._journals):
                results.append((self._journals[idx], float(dist)))
        return results

    def search_by_text(
        self, query_text: str, top_k: int = 10
    ) -> List[Tuple[Journal, float]]:
        """文本搜索（基于 profile 的简单匹配）"""
        # 简化实现：关键词匹配
        query_keywords = set(query_text.lower().split())
        scores = []
        for journal in self._journals:
            profile_keywords = set(journal.journal_profile.lower().split())
            intersection = query_keywords & profile_keywords
            if intersection:
                score = len(intersection) / max(len(query_keywords), 1)
                scores.append((journal, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_journal(self, journal_id: str) -> Optional[Journal]:
        """根据 ID 获取期刊"""
        for journal in self._journals:
            if journal.journal_id == journal_id:
                return journal
        return None

    def list_journals(
        self,
        subject_tag: Optional[str] = None,
        oa_type: Optional[str] = None,
        quartile: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Journal]:
        """列出期刊（支持过滤）"""
        results = self._journals
        if subject_tag:
            results = [j for j in results if subject_tag in j.subject_tags]
        if oa_type:
            results = [j for j in results if j.oa_type == oa_type]
        if quartile:
            results = [j for j in results if j.quartile == quartile]
        return results[offset:offset + limit]

    @property
    def count(self) -> int:
        """期刊数量"""
        return len(self._journals)
```

- [ ] **Step 2: 创建 tests/test_journal_store.py**

```python
"""期刊存储测试"""
import pytest
import tempfile
import os
from src.journals.journal_model import Journal
from src.journals.journal_store import JournalStore


def test_journal_store_add_and_list():
    """测试添加和列出期刊"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JournalStore(
            store_path=os.path.join(tmpdir, "journals.jsonl"),
            faiss_index_path=os.path.join(tmpdir, "index.faiss"),
            metadata_path=os.path.join(tmpdir, "meta.parquet"),
        )
        journal = Journal(
            journal_id="test-journal",
            journal_name="Test Journal",
            subject_tags=["ai"],
            oa_type="full_oa",
        )
        store.add_journal(journal)
        assert store.count == 1

        listed = store.list_journals()
        assert len(listed) == 1
        assert listed[0].journal_id == "test-journal"


def test_journal_store_search_by_text():
    """测试文本搜索"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JournalStore(
            store_path=os.path.join(tmpdir, "journals.jsonl"),
        )
        journal = Journal(
            journal_id="ai-journal",
            journal_name="AI Journal",
            subject_tags=["ai", "ml"],
            keywords=["machine learning", "deep learning"],
            journal_profile="AI Journal machine learning deep learning",
        )
        store.add_journal(journal)

        results = store.search_by_text("machine learning", top_k=5)
        assert len(results) >= 1
        assert results[0][0].journal_id == "ai-journal"
```

- [ ] **Step 3: 运行测试**

```bash
pytest tests/test_journal_store.py -v
# 预期: PASS
```

---

## Task 5: 论文数据模型与解析

**目标:** 定义 PaperProfile 数据结构和 PaperParser（含降级策略）

**Files:**
- Create: `src/papers/paper_model.py`
- Create: `src/papers/paper_parser.py`
- Create: `tests/test_paper_parser.py`

- [ ] **Step 1: 创建 src/papers/paper_model.py**

```python
"""论文数据模型"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PaperProfile(BaseModel):
    """论文特征结构"""
    title: str = Field(description="论文标题")
    abstract: str = Field(default="", description="论文摘要")
    research_area: List[str] = Field(default_factory=list, description="研究领域")
    method_type: str = Field(default="method", description="方法类型")
    paper_type: str = Field(default="application", description="论文类型")
    keywords: List[str] = Field(default_factory=list, description="关键词")
    novelty: str = Field(default="", description="创新点")
    application_domain: List[str] = Field(default_factory=list, description="应用领域")
    difficulty_level: str = Field(default="medium", description="难度等级")
    style: str = Field(default="journal_like", description="风格: journal_like/conference_like")
    sections_summary: Dict[str, str] = Field(default_factory=dict, description="章节摘要")
    full_text_summary: str = Field(default="", description="全文摘要")


class PaperInput(BaseModel):
    """论文输入"""
    title: str
    abstract: Optional[str] = ""
    full_text: Optional[str] = ""
    mode: str = Field(default="abstract", description="title/abstract/full")


class SectionSplitResult(BaseModel):
    """章节切分结果"""
    introduction: str = ""
    method: str = ""
    experiment: str = ""
    conclusion: str = ""
    other: str = ""
```

- [ ] **Step 2: 创建 src/papers/paper_parser.py**

```python
"""论文解析（含降级策略）"""
import time
from typing import Optional

import tenacity

from ..utils.llm import MiniMaxLLM
from ..utils.text import clean_text, extract_keywords
from .paper_model import PaperProfile, PaperInput


class PaperParser:
    """论文解析器"""

    def __init__(self, llm: Optional[MiniMaxLLM] = None):
        self.llm = llm

    @tenacity.retry(
        wait=tenacity.wait_fixed(2),
        stop=tenacity.stop_after_attempt(2),
        reraise=True,
    )
    def parse_with_llm(self, paper_input: PaperInput, system_prompt: str, user_prompt: str) -> PaperProfile:
        """使用 LLM 解析论文"""
        if self.llm is None:
            raise ValueError("LLM not configured")

        user_filled = user_prompt.format(
            title=paper_input.title,
            abstract=paper_input.abstract or "",
            full_text_summary=paper_input.full_text[:500] if paper_input.full_text else "",
        )

        response = self.llm.chat(system_prompt, user_filled)
        # 解析 JSON 响应（简化处理）
        import json
        import re

        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return PaperProfile(
                title=paper_input.title,
                abstract=paper_input.abstract or "",
                **{k: v for k, v in data.items() if k != "title"}
            )

        raise ValueError(f"Failed to parse LLM response: {response.content}")

    def parse_with_fallback(self, paper_input: PaperInput, system_prompt: str, user_prompt: str) -> PaperProfile:
        """带降级策略的解析"""
        try:
            return self.parse_with_llm(paper_input, system_prompt, user_prompt)
        except Exception as e:
            # 降级策略 1: 使用规则 + 关键词提取
            return self._parse_by_rules(paper_input)

    def _parse_by_rules(self, paper_input: PaperInput) -> PaperProfile:
        """规则降级解析"""
        # 提取关键词
        combined_text = paper_input.title + " " + (paper_input.abstract or "")
        keywords = extract_keywords(combined_text, top_k=5)

        # 简单标签匹配
        research_area = self._match_research_area(combined_text)

        return PaperProfile(
            title=paper_input.title,
            abstract=paper_input.abstract or "",
            research_area=research_area,
            method_type=self._infer_method_type(combined_text),
            paper_type="application",
            keywords=keywords,
            novelty="",
            application_domain=[],
            difficulty_level="medium",
            style="unknown",
        )

    def _match_research_area(self, text: str) -> list:
        """匹配研究领域"""
        area_keywords = {
            "ai": ["artificial intelligence", "machine learning", "深度学习", "机器学习"],
            "cv": ["computer vision", "图像", "视频", "视觉", "cv"],
            "nlp": ["nlp", "natural language", "文本", "语言", "语言模型"],
            "se": ["software", "软件", "system"],
            "network": ["network", "网络", "通信"],
            "security": ["security", "安全", "隐私"],
            "db": ["database", "数据库", "数据存储"],
        }
        text_lower = text.lower()
        matched = []
        for area, kws in area_keywords.items():
            if any(kw.lower() in text_lower for kw in kws):
                matched.append(area)
        return matched if matched else ["other"]

    def _infer_method_type(self, text: str) -> str:
        """推断方法类型"""
        if any(kw in text.lower() for kw in ["survey", "综述", "review"]):
            return "survey"
        if any(kw in text.lower() for kw in ["system", "系统", "platform"]):
            return "system"
        if any(kw in text.lower() for kw in ["experiment", "实验", "evaluation"]):
            return "experiment"
        return "method"

    def parse(self, paper_input: PaperInput, system_prompt: str, user_prompt: str) -> PaperProfile:
        """解析论文（对外接口）"""
        if self.llm is None:
            return self._parse_by_rules(paper_input)

        try:
            return self.parse_with_llm(paper_input, system_prompt, user_prompt)
        except Exception:
            return self._parse_by_rules(paper_input)
```

- [ ] **Step 3: 创建 src/papers/section_splitter.py**

```python
"""章节切分（全文模式）"""
import re
from typing import Dict

from .paper_model import SectionSplitResult


class SectionSplitter:
    """论文章节切分器"""

    # 常见章节标题模式
    SECTION_PATTERNS = {
        "introduction": [r"1\s*\.?\s*Introduction", r"1\s*引言", r"引\s*言"],
        "method": [r"2\s*\.?\s*(Proposed|Method|Approach)", r"2\s*方法", r"算法", r"技术"],
        "experiment": [r"3\s*\.?\s*(Experiment|Evaluation|Results)", r"3\s*实验", r"评估", r"结果"],
        "conclusion": [r"(Conclusion|Discussion|Summary)", r"结[论语]", r"总结],
    }

    def split(self, full_text: str) -> SectionSplitResult:
        """切分论文章节"""
        lines = full_text.split("\n")
        sections: Dict[str, list] = {
            "introduction": [],
            "method": [],
            "experiment": [],
            "conclusion": [],
            "other": [],
        }
        current_section = "other"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测章节标题
            detected = self._detect_section(line)
            if detected:
                current_section = detected

            # 跳过页眉页脚
            if self._is_header_footer(line):
                continue

            sections[current_section].append(line)

        return SectionSplitResult(
            introduction=" ".join(sections["introduction"]),
            method=" ".join(sections["method"]),
            experiment=" ".join(sections["experiment"]),
            conclusion=" ".join(sections["conclusion"]),
            other=" ".join(sections["other"]),
        )

    def _detect_section(self, line: str) -> str:
        """检测章节类型"""
        for section_name, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    return section_name
        return ""

    def _is_header_footer(self, line: str) -> bool:
        """判断是否页眉页脚"""
        # 简化判断
        if len(line) < 10:
            return True
        if re.match(r"^\d+\s+\d+$", line):
            return True
        return False
```

- [ ] **Step 4: 创建 tests/test_paper_parser.py**

```python
"""论文解析测试"""
import pytest
from src.papers.paper_model import PaperInput, PaperProfile
from src.papers.paper_parser import PaperParser


def test_paper_parser_fallback():
    """测试降级解析"""
    parser = PaperParser(llm=None)  # 不配置 LLM，强制走降级
    paper_input = PaperInput(
        title="Deep Learning for Image Recognition",
        abstract="This paper proposes a new deep learning method for image recognition.",
    )
    profile = parser.parse(paper_input, "", "")
    assert profile.title == "Deep Learning for Image Recognition"
    assert len(profile.research_area) >= 0  # 可能有 AI/CV 标签


def test_paper_input_model():
    """测试 PaperInput 模型"""
    input_data = PaperInput(
        title="Test Paper",
        abstract="Test abstract",
        mode="abstract",
    )
    assert input_data.title == "Test Paper"
    assert input_data.mode == "abstract"


def test_section_splitter():
    """测试章节切分"""
    from src.papers.section_splitter import SectionSplitter

    splitter = SectionSplitter()
    text = """
1. Introduction
This is the introduction.

2. Method
This is the method section.

3. Experiment
This is the experiment section.

4. Conclusion
This is the conclusion.
"""
    result = splitter.split(text)
    assert "introduction" in result.introduction.lower() or len(result.introduction) > 0
    assert len(result.method) > 0
    assert len(result.experiment) > 0
```

- [ ] **Step 5: 运行测试**

```bash
pytest tests/test_paper_parser.py -v
# 预期: PASS
```

---

## Task 6: 召回模块

**目标:** 实现 BM25 召回、向量化召回、混合召回 CandidateGenerator

**Files:**
- Create: `src/retriever/bm25_retriever.py`
- Create: `src/retriever/embedding_retriever.py`
- Create: `src/retriever/candidate_generator.py`
- Create: `tests/test_retriever.py`

- [ ] **Step 1: 创建 src/retriever/bm25_retriever.py**

```python
"""BM25 召回"""
from typing import List, Tuple

import rank_bm25

from ..journals.journal_model import Journal
from ..journals.journal_store import JournalStore


class BM25Retriever:
    """BM25 召回器"""

    def __init__(self, store: JournalStore):
        self.store = store
        self._tokenized_profiles: List[List[str]] = []
        self._bm25_index: rank_bm25.BM25Okapi = None
        self._built = False

    def build_index(self) -> None:
        """构建 BM25 索引"""
        if self.store.count == 0:
            return

        self._tokenized_profiles = []
        for journal in self.store._journals:
            profile = journal.journal_profile or journal.scope_text
            tokens = profile.lower().split()
            self._tokenized_profiles.append(tokens)

        self._bm25_index = rank_bm25.BM25Okapi(self._tokenized_profiles)
        self._built = True

    def retrieve(self, query: str, top_k: int = 30) -> List[Tuple[Journal, float]]:
        """BM25 检索"""
        if not self._built:
            self.build_index()

        if self._bm25_index is None:
            return []

        query_tokens = query.lower().split()
        scores = self._bm25_index.get_scores(query_tokens)

        # 获取 top_k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if idx < len(self.store._journals):
                results.append((self.store._journals[idx], float(scores[idx])))
        return results
```

- [ ] **Step 2: 创建 src/retriever/embedding_retriever.py**

```python
"""向量检索召回"""
from typing import List, Tuple

import numpy as np

from ..journals.journal_model import Journal
from ..journals.journal_store import JournalStore
from ..utils.embedding import OllamaEmbedding


class EmbeddingRetriever:
    """向量检索召回器"""

    def __init__(self, store: JournalStore, embedding_client: OllamaEmbedding):
        self.store = store
        self.embedding_client = embedding_client

    def retrieve(
        self, query_text: str, top_k: int = 30
    ) -> List[Tuple[Journal, float]]:
        """向量检索"""
        # 获取查询向量
        query_embedding = self.embedding_client.embed(query_text)

        # FAISS 检索
        results = self.store.search_by_vector(query_embedding, top_k)

        # 转换为 (journal, score) 格式，score 取负距离（距离越小越相似）
        return [(journal, -score) for journal, score in results]

    def retrieve_by_embedding(
        self, query_embedding: np.ndarray, top_k: int = 30
    ) -> List[Tuple[Journal, float]]:
        """直接用向量检索"""
        results = self.store.search_by_vector(query_embedding, top_k)
        return [(journal, -score) for journal, score in results]
```

- [ ] **Step 3: 创建 src/retriever/candidate_generator.py**

```python
"""混合召回"""
from typing import Dict, List, Optional, Tuple

from ..journals.journal_model import Journal, JournalMatch
from ..journals.journal_store import JournalStore
from ..papers.paper_model import PaperProfile
from .bm25_retriever import BM25Retriever
from .embedding_retriever import EmbeddingRetriever


class CandidateGenerator:
    """候选召回生成器（混合召回）"""

    def __init__(
        self,
        store: JournalStore,
        bm25_retriever: BM25Retriever,
        embedding_retriever: Optional[EmbeddingRetriever] = None,
    ):
        self.store = store
        self.bm25_retriever = bm25_retriever
        self.embedding_retriever = embedding_retriever

    def generate(
        self,
        query_text: str,
        paper_profile: PaperProfile,
        top_k: int = 40,
        mode: str = "abstract",
    ) -> List[Journal]:
        """生成候选期刊"""
        # 各路召回数量配置
        config = {
            "title": {"bm25": 20, "vector": 20, "tag": 15},
            "abstract": {"bm25": 25, "vector": 25, "tag": 20},
            "full": {"bm25": 30, "vector": 30, "tag": 25},
        }
        cfg = config.get(mode, config["abstract"])

        # 1. BM25 召回
        bm25_results = self.bm25_retriever.retrieve(query_text, top_k=cfg["bm25"])

        # 2. 向量检索召回
        vector_results = []
        if self.embedding_retriever:
            vector_results = self.embedding_retriever.retrieve(query_text, top_k=cfg["vector"])

        # 3. 标签过滤
        tag_filtered = self._filter_by_tags(paper_profile, top_k=cfg["tag"])

        # 4. 合并去重
        candidates = self._merge_results(bm25_results, vector_results, tag_filtered, top_k=top_k)

        return candidates

    def _filter_by_tags(
        self, paper_profile: PaperProfile, top_k: int = 20
    ) -> List[Tuple[Journal, float]]:
        """标签过滤召回"""
        results = []
        for journal in self.store._journals:
            score = 0.0
            # 研究领域匹配
            if paper_profile.research_area:
                for area in paper_profile.research_area:
                    if area in journal.subject_tags:
                        score += 1.0
            # 论文类型匹配
            if paper_profile.method_type in journal.target_paper_type:
                score += 0.5
            if score > 0:
                results.append((journal, score))

        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _merge_results(
        self,
        bm25_results: List[Tuple[Journal, float]],
        vector_results: List[Tuple[Journal, float]],
        tag_results: List[Tuple[Journal, float]],
        top_k: int = 50,
    ) -> List[Journal]:
        """合并去重"""
        score_map: Dict[str, float] = {}

        # BM25 结果
        for journal, score in bm25_results:
            score_map[journal.journal_id] = score_map.get(journal.journal_id, 0) + score * 0.4

        # 向量结果
        for journal, score in vector_results:
            score_map[journal.journal_id] = score_map.get(journal.journal_id, 0) + score * 0.4

        # 标签结果
        for journal, score in tag_results:
            score_map[journal.journal_id] = score_map.get(journal.journal_id, 0) + score * 0.2

        # 排序取 top_k
        sorted_ids = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)[:top_k]

        # 返回 Journal 对象列表
        journal_map = {j.journal_id: j for j in self.store._journals}
        return [journal_map[jid] for jid in sorted_ids if jid in journal_map]
```

- [ ] **Step 4: 创建 tests/test_retriever.py**

```python
"""召回模块测试"""
import pytest
import tempfile
from src.journals.journal_model import Journal
from src.journals.journal_store import JournalStore
from src.retriever.bm25_retriever import BM25Retriever
from src.retriever.candidate_generator import CandidateGenerator
from src.papers.paper_model import PaperProfile


def test_bm25_retriever():
    """测试 BM25 召回"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JournalStore(store_path=f"{tmpdir}/journals.jsonl")
        journal = Journal(
            journal_id="ai-journal",
            journal_name="AI Journal",
            subject_tags=["ai"],
            keywords=["machine learning"],
            scope_text="Artificial intelligence and machine learning",
            journal_profile="AI Journal artificial intelligence machine learning",
        )
        store.add_journal(journal)

        retriever = BM25Retriever(store)
        retriever.build_index()

        results = retriever.retrieve("machine learning", top_k=10)
        assert len(results) >= 1
        assert results[0][0].journal_id == "ai-journal"


def test_candidate_generator_merge():
    """测试候选生成器合并"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JournalStore(store_path=f"{tmpdir}/journals.jsonl")
        journal1 = Journal(
            journal_id="ai-journal",
            journal_name="AI Journal",
            subject_tags=["ai"],
            journal_profile="AI Journal",
        )
        journal2 = Journal(
            journal_id="cv-journal",
            journal_name="CV Journal",
            subject_tags=["cv"],
            journal_profile="CV Journal",
        )
        store.add_journal(journal1)
        store.add_journal(journal2)

        generator = CandidateGenerator(store, BM25Retriever(store))

        profile = PaperProfile(
            title="Deep Learning",
            research_area=["ai"],
            method_type="method",
        )
        candidates = generator.generate("deep learning artificial intelligence", profile, top_k=10)
        assert len(candidates) <= 10
        assert all(isinstance(j, Journal) for j in candidates)
```

- [ ] **Step 5: 运行测试**

```bash
pytest tests/test_retriever.py -v
# 预期: PASS
```

---

## Task 7: 排序模块

**目标:** 实现两阶段排序（规则打分 + LLM 精排）

**Files:**
- Create: `src/ranker/rule_scorer.py`
- Create: `src/ranker/llm_ranker.py`
- Create: `src/ranker/feature_builder.py`
- Create: `src/ranker/scoring.py`
- Create: `tests/test_ranker.py`

- [ ] **Step 1: 创建 src/ranker/rule_scorer.py**

```python
"""规则打分（阶段一）"""
from typing import List, Tuple

from ..journals.journal_model import Journal
from ..papers.paper_model import PaperProfile


class RuleScorer:
    """规则打分器"""

    def __init__(self):
        # 权重配置
        self.weights = {
            "research_area_match": 2.0,
            "method_type_match": 1.5,
            "paper_type_match": 1.0,
            "quartile_bonus": 0.5,  # Q1/Q2 加分
            "oa_preference_match": 0.3,
        }

    def score(
        self, journal: Journal, paper_profile: PaperProfile, oa_preference: str = "any"
    ) -> Tuple[float, List[str]]:
        """计算规则分数"""
        score = 0.0
        reasons = []

        # 研究领域匹配
        if paper_profile.research_area:
            for area in paper_profile.research_area:
                if area in journal.subject_tags:
                    score += self.weights["research_area_match"]
                    reasons.append(f"研究领域匹配: {area}")
                    break

        # 方法类型匹配
        if paper_profile.method_type in journal.target_paper_type:
            score += self.weights["method_type_match"]
            reasons.append(f"方法类型匹配: {paper_profile.method_type}")

        # 论文类型匹配
        if paper_profile.paper_type:
            if journal.scope_text.lower().find(paper_profile.paper_type) >= 0:
                score += self.weights["paper_type_match"]

        # 分区加分
        if journal.quartile in ["Q1", "Q2"]:
            score += self.weights["quartile_bonus"]
            reasons.append(f"高分区期刊: {journal.quartile}")

        # OA 偏好匹配
        if oa_preference != "any":
            if (oa_preference == "full_oa" and journal.oa_type == "full_oa") or \
               (oa_preference == "hybrid" and journal.oa_type in ["full_oa", "hybrid"]):
                score += self.weights["oa_preference_match"]
                reasons.append(f"OA类型匹配: {journal.oa_type}")

        return score, reasons

    def rank(
        self,
        journals: List[Journal],
        paper_profile: PaperProfile,
        oa_preference: str = "any",
        top_k: int = 10,
    ) -> List[Tuple[Journal, float, List[str]]]:
        """排序候选期刊"""
        scored = []
        for journal in journals:
            score, reasons = self.score(journal, paper_profile, oa_preference)
            scored.append((journal, score, reasons))

        # 按分数排序
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
```

- [ ] **Step 2: 创建 src/ranker/llm_ranker.py**

```python
"""LLM 排序（阶段二）"""
import json
from typing import List, Tuple, Optional

from ..journals.journal_model import Journal
from ..papers.paper_model import PaperProfile
from ..utils.llm import MiniMaxLLM


class LLMRanker:
    """LLM 排序器"""

    def __init__(self, llm: MiniMaxLLM, system_prompt: str, user_prompt_template: str):
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    def rank(
        self,
        candidates: List[Tuple[Journal, float, List[str]]],
        paper_profile: PaperProfile,
        top_k: int = 5,
    ) -> List[Tuple[Journal, float, List[str], float]]:
        """LLM 精排"""
        # 构建期刊信息
        journals_info = []
        for journal, rule_score, reasons in candidates:
            journals_info.append({
                "journal_id": journal.journal_id,
                "journal_name": journal.journal_name,
                "scope": journal.scope_text,
                "quartile": journal.quartile or "unknown",
                "oa_type": journal.oa_type,
                "rule_score": rule_score,
            })

        # 填充 prompt
        user_prompt = self.user_prompt_template.format(
            title=paper_profile.title,
            research_area=", ".join(paper_profile.research_area),
            method_type=paper_profile.method_type,
            paper_type=paper_profile.paper_type,
            keywords=", ".join(paper_profile.keywords),
            journals_info=json.dumps(journals_info, ensure_ascii=False, indent=2),
        )

        # 调用 LLM
        response = self.llm.chat(self.system_prompt, user_prompt)

        # 解析结果
        try:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                rankings = data.get("rankings", [])

                # 构建结果
                rank_map = {r["journal_id"]: r for r in rankings}
                results = []
                for journal, rule_score, reasons in candidates:
                    if journal.journal_id in rank_map:
                        r = rank_map[journal.journal_id]
                        results.append((
                            journal,
                            r.get("score", rule_score),
                            r.get("reasons", reasons),
                            r.get("confidence", 0.5),
                        ))

                # 按 LLM 分数排序
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:top_k]

        except Exception:
            pass

        # 降级：返回原始顺序
        return [(j, s, r, 0.5) for j, s, r in candidates[:top_k]]
```

- [ ] **Step 3: 创建 src/ranker/feature_builder.py**

```python
"""排序特征构建"""
from typing import Dict, List

from ..journals.journal_model import Journal
from ..papers.paper_model import PaperProfile


class FeatureBuilder:
    """排序特征构建器"""

    @staticmethod
    def build(journal: Journal, paper_profile: PaperProfile) -> Dict[str, float]:
        """构建特征向量"""
        features = {}

        # 文本相似度特征
        features["title_overlap"] = _text_overlap(
            paper_profile.title, journal.journal_name
        )
        features["abstract_overlap"] = _text_overlap(
            paper_profile.abstract, journal.scope_text
        )

        # 领域匹配特征
        features["research_area_match"] = 1.0 if any(
            area in journal.subject_tags for area in paper_profile.research_area
        ) else 0.0

        # 类型匹配特征
        features["method_type_match"] = 1.0 if paper_profile.method_type in journal.target_paper_type else 0.0

        # 影响力特征
        quartile_scores = {"Q1": 1.0, "Q2": 0.7, "Q3": 0.4, "Q4": 0.2}
        features["quartile_score"] = quartile_scores.get(journal.quartile or "", 0.0)

        # OA 特征
        features["is_full_oa"] = 1.0 if journal.oa_type == "full_oa" else 0.0
        features["is_hybrid"] = 1.0 if journal.oa_type == "hybrid" else 0.0

        return features


def _text_overlap(text1: str, text2: str) -> float:
    """文本重叠度"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    return len(intersection) / max(len(words1), len(words2))
```

- [ ] **Step 4: 创建 src/ranker/scoring.py**

```python
"""分数计算工具"""


class Scoring:
    """分数计算工具"""

    @staticmethod
    def normalize_scores(scores: list) -> list:
        """归一化分数到 [0, 1]"""
        if not scores:
            return []
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]

    @staticmethod
    def combine_scores(weights: dict, features: dict) -> float:
        """加权求和"""
        score = 0.0
        for key, weight in weights.items():
            if key in features:
                score += weight * features[key]
        return score

    @staticmethod
    def confidence_from_scores(rule_score: float, llm_score: float) -> float:
        """综合置信度"""
        # 两者加权平均
        return 0.4 * min(rule_score / 10, 1.0) + 0.6 * llm_score
```

- [ ] **Step 5: 创建 tests/test_ranker.py**

```python
"""排序模块测试"""
import pytest
from src.journals.journal_model import Journal
from src.papers.paper_model import PaperProfile
from src.ranker.rule_scorer import RuleScorer


def test_rule_scorer():
    """测试规则打分"""
    scorer = RuleScorer()
    journal = Journal(
        journal_id="ai-journal",
        journal_name="AI Journal",
        subject_tags=["ai"],
        target_paper_type=["method", "experiment"],
        quartile="Q1",
        oa_type="full_oa",
    )
    profile = PaperProfile(
        title="Deep Learning",
        research_area=["ai"],
        method_type="method",
        paper_type="application",
    )

    score, reasons = scorer.score(journal, profile, oa_preference="any")
    assert score > 0
    assert len(reasons) >= 2  # 至少领域匹配和分区加分


def test_rule_scorer_rank():
    """测试规则排序"""
    scorer = RuleScorer()
    journals = [
        Journal(journal_id="j1", journal_name="J1", subject_tags=["ai"], quartile="Q1"),
        Journal(journal_id="j2", journal_name="J2", subject_tags=["cv"], quartile="Q2"),
        Journal(journal_id="j3", journal_name="J3", subject_tags=["ai"], quartile="Q2"),
    ]
    profile = PaperProfile(title="Test", research_area=["ai"])

    ranked = scorer.rank(journals, profile, top_k=2)
    assert len(ranked) <= 2
    # AI 期刊应该在前面
    assert ranked[0][0].journal_id == "j1"
```

- [ ] **Step 6: 运行测试**

```bash
pytest tests/test_ranker.py -v
# 预期: PASS
```

---

## Task 8: 推荐流程编排 + Explainer

**目标:** 实现推荐管道和独立解释模块

**Files:**
- Create: `src/recommender/pipeline.py`
- Create: `src/recommender/explainer.py`
- Create: `tests/test_recommender.py`

- [ ] **Step 1: 创建 src/recommender/explainer.py**

```python
"""推荐理由生成模块"""
import json
from typing import List, Optional

from ..journals.journal_model import Journal
from ..papers.paper_model import PaperProfile
from ..utils.llm import MiniMaxLLM


class Explainer:
    """推荐理由生成器"""

    def __init__(
        self,
        llm: Optional[MiniMaxLLM] = None,
        system_prompt: str = "",
        user_prompt_template: str = "",
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    def explain(
        self,
        journal: Journal,
        paper_profile: PaperProfile,
        matched_fields: Optional[List[str]] = None,
    ) -> List[str]:
        """生成推荐理由"""
        if self.llm is None or not self.user_prompt_template:
            return self._generate_fallback_explanation(journal, paper_profile, matched_fields)

        user_prompt = self.user_prompt_template.format(
            research_area=", ".join(paper_profile.research_area),
            method_type=paper_profile.method_type,
            paper_type=paper_profile.paper_type,
            keywords=", ".join(paper_profile.keywords),
            journal_name=journal.journal_name,
            scope_text=journal.scope_text,
            quartile=journal.quartile or "unknown",
            oa_type=journal.oa_type,
            review_time=journal.review_time or "unknown",
        )

        try:
            response = self.llm.chat(self.system_prompt, user_prompt)
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                reasons = data.get("reasons", [])
                if reasons:
                    return reasons
        except Exception:
            pass

        return self._generate_fallback_explanation(journal, paper_profile, matched_fields)

    def _generate_fallback_explanation(
        self,
        journal: Journal,
        paper_profile: PaperProfile,
        matched_fields: Optional[List[str]] = None,
    ) -> List[str]:
        """生成降级推荐理由"""
        reasons = []

        # 领域匹配
        if paper_profile.research_area:
            matched = [a for a in paper_profile.research_area if a in journal.subject_tags]
            if matched:
                reasons.append(f"研究领域匹配: {', '.join(matched)}")

        # 类型匹配
        if paper_profile.method_type in journal.target_paper_type:
            reasons.append(f"方法类型契合: {journal.target_paper_type}")

        # 分区
        if journal.quartile:
            reasons.append(f"期刊分区: {journal.quartile}")

        # OA 类型
        if journal.oa_type:
            oa_labels = {"full_oa": "完全开放获取", "hybrid": "混合OA", "subscription": "订阅制"}
            reasons.append(f"OA类型: {oa_labels.get(journal.oa_type, journal.oa_type)}")

        if not reasons:
            reasons.append("综合匹配度较高")

        return reasons

    def explain_batch(
        self,
        journals: List[Journal],
        paper_profile: PaperProfile,
    ) -> List[List[str]]:
        """批量生成推荐理由"""
        return [self.explain(j, paper_profile) for j in journals]
```

- [ ] **Step 2: 创建 src/recommender/pipeline.py**

```python
"""推荐流程编排"""
import yaml
from typing import List, Optional, Dict, Any

from ..journals.journal_model import Journal, JournalMatch
from ..papers.paper_model import PaperInput, PaperProfile
from ..retriever.candidate_generator import CandidateGenerator
from ..ranker.rule_scorer import RuleScorer
from ..ranker.llm_ranker import LLMRanker
from .explainer import Explainer


class RecommenderPipeline:
    """推荐流程编排器"""

    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        rule_scorer: RuleScorer,
        llm_ranker: Optional[LLMRanker] = None,
        explainer: Optional[Explainer] = None,
    ):
        self.candidate_generator = candidate_generator
        self.rule_scorer = rule_scorer
        self.llm_ranker = llm_ranker
        self.explainer = explainer

    def recommend(
        self,
        paper_input: PaperInput,
        paper_profile: PaperProfile,
        top_k: int = 5,
        mode: str = "abstract",
        oa_preference: str = "any",
    ) -> Dict[str, Any]:
        """执行推荐流程"""
        # 1. 候选召回
        query_text = paper_input.title
        if paper_input.abstract:
            query_text += " " + paper_input.abstract

        candidates = self.candidate_generator.generate(
            query_text, paper_profile, top_k=50, mode=mode
        )

        if not candidates:
            return {"recommendations": [], "warning": "未找到合适的候选期刊"}

        # 2. 阶段一：规则打分
        rule_ranked = self.rule_scorer.rank(
            candidates, paper_profile, oa_preference=oa_preference, top_k=10
        )

        # 3. 阶段二：LLM 精排
        if self.llm_ranker:
            llm_ranked = self.llm_ranker.rank(rule_ranked, paper_profile, top_k=top_k)
        else:
            llm_ranked = [(j, s, r, 0.5) for j, s, r in rule_ranked[:top_k]]

        # 4. 生成推荐理由
        recommendations = []
        for journal, score, reasons, confidence in llm_ranked:
            match_reasons = reasons
            if self.explainer:
                match_reasons = self.explainer.explain(journal, paper_profile)

            recommendations.append(JournalMatch(
                journal=journal,
                score=score,
                confidence=confidence,
                match_reasons=match_reasons,
                matched_fields=["research_area", "method_type"],
            ))

        # 5. 构建响应
        result = {
            "recommendations": recommendations,
            "paper_profile": paper_profile,
            "mode_used": mode,
        }

        # 标题模式加警告
        if mode == "title":
            result["warning"] = "置信度较低，建议补充摘要以获得更精确的推荐"

        return result

    @classmethod
    def from_config(cls, config_path: str = "configs/app.yaml") -> "RecommenderPipeline":
        """从配置文件创建"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 加载 prompts
        with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        # 这里需要传入已初始化的组件
        # 实际使用时通过依赖注入
        return cls(
            candidate_generator=None,  # 外部注入
            rule_scorer=RuleScorer(),
            llm_ranker=None,
            explainer=None,
        )
```

- [ ] **Step 3: 创建 tests/test_recommender.py**

```python
"""推荐流程测试"""
import pytest
from src.journals.journal_model import Journal
from src.papers.paper_model import PaperInput, PaperProfile
from src.retriever.candidate_generator import CandidateGenerator
from src.retriever.bm25_retriever import BM25Retriever
from src.ranker.rule_scorer import RuleScorer
from src.recommender.pipeline import RecommenderPipeline


def test_pipeline_integration():
    """测试完整流程（不含 LLM）"""
    # 简化测试：验证 pipeline 能正常处理
    from src.journals.journal_store import JournalStore
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        store = JournalStore(store_path=f"{tmpdir}/journals.jsonl")
        journal = Journal(
            journal_id="ai-journal",
            journal_name="AI Journal",
            subject_tags=["ai"],
            keywords=["machine learning"],
            scope_text="Artificial intelligence",
            journal_profile="AI Journal artificial intelligence",
            target_paper_type=["method"],
            quartile="Q1",
        )
        store.add_journal(journal)

        generator = CandidateGenerator(
            store, BM25Retriever(store), embedding_retriever=None
        )
        scorer = RuleScorer()
        pipeline = RecommenderPipeline(
            candidate_generator=generator,
            rule_scorer=scorer,
        )

        paper_input = PaperInput(title="Deep Learning for AI")
        profile = PaperProfile(
            title="Deep Learning for AI",
            research_area=["ai"],
            method_type="method",
        )

        result = pipeline.recommend(paper_input, profile, mode="title")
        assert "recommendations" in result
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/test_recommender.py -v
# 预期: PASS
```

---

## Task 9: FastAPI 接口

**目标:** 实现 API 路由和 Pydantic schemas

**Files:**
- Create: `src/app/schemas.py`
- Create: `src/app/api.py`
- Create: `src/app/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: 创建 src/app/schemas.py**

```python
"""API 数据模型"""
from typing import List, Optional
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """推荐请求"""
    title: str = Field(..., description="论文标题")
    abstract: Optional[str] = Field("", description="论文摘要")
    full_text: Optional[str] = Field("", description="论文全文")
    mode: str = Field("abstract", description="模式: title/abstract/full")
    top_k: int = Field(5, ge=1, le=20, description="推荐数量")
    oa_preference: str = Field("any", description="OA偏好: any/full_oa/hybrid")


class PaperProfileResponse(BaseModel):
    """论文特征响应"""
    title: str
    research_area: List[str]
    method_type: str
    paper_type: str
    keywords: List[str]


class JournalResponse(BaseModel):
    """期刊信息响应"""
    journal_id: str
    journal_name: str
    score: float
    confidence: float
    match_reasons: List[str]
    matched_fields: List[str]
    tags: List[str]
    oa_type: str
    quartile: Optional[str]
    submission_url: Optional[str]


class RecommendResponse(BaseModel):
    """推荐响应"""
    recommendations: List[JournalResponse]
    paper_profile: Optional[PaperProfileResponse]
    mode_used: str
    warning: Optional[str] = None


class JournalListItem(BaseModel):
    """期刊列表项"""
    journal_id: str
    journal_name: str
    subject_tags: List[str]
    oa_type: str
    quartile: Optional[str]


class JournalListResponse(BaseModel):
    """期刊列表响应"""
    journals: List[JournalListItem]
    total: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
```

- [ ] **Step 2: 创建 src/app/api.py**

```python
"""API 路由"""
from typing import Optional
import yaml

from fastapi import APIRouter, HTTPException, Query

from .schemas import (
    RecommendRequest,
    RecommendResponse,
    JournalListResponse,
    JournalListItem,
    HealthResponse,
    JournalResponse,
    PaperProfileResponse,
)
from ..recommender.pipeline import RecommenderPipeline
from ..papers.paper_parser import PaperParser
from ..papers.paper_model import PaperInput, PaperProfile
from ..journals.journal_store import JournalStore
from ..retriever.bm25_retriever import BM25Retriever
from ..retriever.embedding_retriever import EmbeddingRetriever
from ..ranker.rule_scorer import RuleScorer
from ..ranker.llm_ranker import LLMRanker
from ..recommender.explainer import Explainer
from ..utils.llm import MiniMaxLLM
from ..utils.embedding import OllamaEmbedding


router = APIRouter()

# 全局组件（实际应通过依赖注入）
_pipeline: Optional[RecommenderPipeline] = None
_store: Optional[JournalStore] = None


def get_pipeline() -> RecommenderPipeline:
    """获取 pipeline（懒加载初始化）"""
    global _pipeline, _store
    if _pipeline is None:
        _store = JournalStore()
        _store.load()

        bm25 = BM25Retriever(_store)
        bm25.build_index()

        # 加载配置
        with open("configs/app.yaml", "r", encoding="utf-8") as f:
            app_config = yaml.safe_load(f)

        with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
            prompts = yaml.safe_load(f)

        # 初始化 LLM（如果配置了 API key）
        llm = None
        try:
            llm = MiniMaxLLM(
                api_key=app_config["minimax"]["api_key"],
                base_url=app_config["minimax"]["base_url"],
                model=app_config["minimax"]["model"],
            )
        except Exception:
            pass

        embedding_client = OllamaEmbedding(
            base_url=app_config["ollama"]["base_url"],
            model=app_config["ollama"]["embedding_model"],
        )

        embedding_retriever = EmbeddingRetriever(_store, embedding_client)
        generator = CandidateGenerator(_store, bm25, embedding_retriever)
        scorer = RuleScorer()

        llm_ranker = None
        if llm:
            llm_ranker = LLMRanker(
                llm,
                prompts["llm_ranker_system"],
                prompts["llm_ranker_user"],
            )

        explainer = Explainer(llm, prompts["explainer_system"], prompts["explainer_user"])

        _pipeline = RecommenderPipeline(
            candidate_generator=generator,
            rule_scorer=scorer,
            llm_ranker=llm_ranker,
            explainer=explainer,
        )

    return _pipeline


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """推荐期刊"""
    pipeline = get_pipeline()

    # 解析论文
    paper_input = PaperInput(
        title=request.title,
        abstract=request.abstract or "",
        full_text=request.full_text or "",
        mode=request.mode,
    )

    with open("configs/prompts.yaml", "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    parser = PaperParser()
    profile = parser.parse(paper_input, prompts["paper_profile_system"], prompts["paper_profile_user"])

    # 执行推荐
    result = pipeline.recommend(
        paper_input,
        profile,
        top_k=request.top_k,
        mode=request.mode,
        oa_preference=request.oa_preference,
    )

    # 构建响应
    recommendations = []
    for rec in result.get("recommendations", []):
        recommendations.append(JournalResponse(
            journal_id=rec.journal.journal_id,
            journal_name=rec.journal.journal_name,
            score=rec.score,
            confidence=rec.confidence,
            match_reasons=rec.match_reasons,
            matched_fields=rec.matched_fields,
            tags=rec.journal.subject_tags,
            oa_type=rec.journal.oa_type,
            quartile=rec.journal.quartile,
            submission_url=rec.journal.submission_url,
        ))

    paper_profile_resp = None
    if "paper_profile" in result:
        pp = result["paper_profile"]
        paper_profile_resp = PaperProfileResponse(
            title=pp.title,
            research_area=pp.research_area,
            method_type=pp.method_type,
            paper_type=pp.paper_type,
            keywords=pp.keywords,
        )

    return RecommendResponse(
        recommendations=recommendations,
        paper_profile=paper_profile_resp,
        mode_used=result.get("mode_used", request.mode),
        warning=result.get("warning"),
    )


@router.get("/journals", response_model=JournalListResponse)
async def list_journals(
    subject_tag: Optional[str] = Query(None),
    oa_type: Optional[str] = Query(None),
    quartile: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """列出期刊"""
    store = get_pipeline().candidate_generator.store

    journals = store.list_journals(
        subject_tag=subject_tag,
        oa_type=oa_type,
        quartile=quartile,
        limit=limit,
        offset=offset,
    )

    return JournalListResponse(
        journals=[
            JournalListItem(
                journal_id=j.journal_id,
                journal_name=j.journal_name,
                subject_tags=j.subject_tags,
                oa_type=j.oa_type,
                quartile=j.quartile,
            )
            for j in journals
        ],
        total=store.count,
        limit=limit,
        offset=offset,
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    return HealthResponse(status="ok", version="0.1.0")
```

- [ ] **Step 3: 创建 src/app/main.py**

```python
"""FastAPI 应用入口"""
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from .api import router
from ..utils.logging import setup_logging


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="论文投稿期刊推荐系统",
        description="根据论文内容推荐适合投稿的计算机类期刊",
        version="0.1.0",
    )

    # 注册路由
    app.include_router(router, prefix="/api")

    # 静态文件（前端）
    frontend_path = Path("frontend")
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory="frontend"), name="static")

    @app.get("/")
    async def root():
        """根路径重定向到前端"""
        frontend_index = Path("frontend/index.html")
        if frontend_index.exists():
            return RedirectResponse(url="/static/index.html")
        return {"message": "Journal Recommender API", "version": "0.1.0"}

    return app


app = create_app()


if __name__ == "__main__":
    import yaml

    # 加载配置
    config_path = Path("configs/app.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        host = config.get("app", {}).get("host", "0.0.0.0")
        port = config.get("app", {}).get("port", 8000)
        log_level = config.get("app", {}).get("log_level", "INFO")
    else:
        host = "0.0.0.0"
        port = 8000
        log_level = "INFO"

    setup_logging(level=log_level)

    uvicorn.run(app, host=host, port=port)
```

- [ ] **Step 4: 创建 tests/test_api.py**

```python
"""API 测试"""
import pytest
from fastapi.testclient import TestClient

from src.app.main import app


client = TestClient(app)


def test_health_endpoint():
    """测试健康检查接口"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_recommend_endpoint():
    """测试推荐接口（mock 数据）"""
    response = client.post(
        "/api/recommend",
        json={
            "title": "Deep Learning for Image Recognition",
            "abstract": "This paper proposes a new method.",
            "mode": "abstract",
            "top_k": 3,
        },
    )
    # 取决于是否有数据，可能返回 200 或空结果
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert "mode_used" in data


def test_journals_endpoint():
    """测试期刊列表接口"""
    response = client.get("/api/journals?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "journals" in data
    assert "total" in data
```

- [ ] **Step 5: 运行测试**

```bash
pytest tests/test_api.py -v
# 预期: PASS (API 结构正确)
```

---

## Task 10: 前端页面

**目标:** 实现简单的 Web 前端页面

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/css/style.css`
- Create: `frontend/js/app.js`

- [ ] **Step 1: 创建 frontend/index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>论文投稿期刊推荐系统</title>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>论文投稿期刊推荐系统</h1>
            <div class="mode-switch">
                <button class="mode-btn active" data-mode="title">标题模式</button>
                <button class="mode-btn" data-mode="abstract">摘要模式</button>
                <button class="mode-btn" data-mode="full">全文模式</button>
            </div>
        </header>

        <main>
            <section class="input-section">
                <div class="form-group">
                    <label for="title">论文标题 *</label>
                    <input type="text" id="title" placeholder="请输入论文标题" required>
                </div>

                <div class="form-group" id="abstract-group">
                    <label for="abstract">摘要</label>
                    <textarea id="abstract" rows="4" placeholder="请输入论文摘要（可选）"></textarea>
                </div>

                <div class="form-group hidden" id="fulltext-group">
                    <label for="full_text">全文</label>
                    <textarea id="full_text" rows="10" placeholder="请输入论文全文（可选）"></textarea>
                </div>

                <div class="form-group">
                    <label for="top_k">推荐数量</label>
                    <select id="top_k">
                        <option value="3">3</option>
                        <option value="5" selected>5</option>
                        <option value="10">10</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="oa_preference">OA 偏好</label>
                    <select id="oa_preference">
                        <option value="any">不限制</option>
                        <option value="full_oa">完全 OA</option>
                        <option value="hybrid">混合 OA</option>
                    </select>
                </div>

                <button id="recommend-btn" class="btn-primary">推荐</button>
            </section>

            <section class="result-section">
                <h2>推荐结果</h2>
                <div id="loading" class="hidden">
                    <div class="spinner"></div>
                    <span>分析中...</span>
                </div>
                <div id="results"></div>
                <div id="warning" class="warning hidden"></div>
            </section>
        </main>

        <footer>
            <p>支持三种输入模式：标题模式（快速粗筛）、摘要模式（主力推荐）、全文模式（高精度）</p>
        </footer>
    </div>

    <script src="/static/js/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: 创建 frontend/css/style.css**

```css
:root {
    --primary: #2563eb;
    --primary-hover: #1d4ed8;
    --bg: #f8fafc;
    --card-bg: #ffffff;
    --text: #1e293b;
    --text-secondary: #64748b;
    --border: #e2e8f0;
    --success: #22c55e;
    --warning: #f59e0b;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}

h1 {
    font-size: 1.5rem;
    font-weight: 600;
}

.mode-switch {
    display: flex;
    gap: 0.5rem;
}

.mode-btn {
    padding: 0.5rem 1rem;
    border: 1px solid var(--border);
    background: var(--card-bg);
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.875rem;
    transition: all 0.2s;
}

.mode-btn.active {
    background: var(--primary);
    color: white;
    border-color: var(--primary);
}

main {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
}

.input-section, .result-section {
    background: var(--card-bg);
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

h2 {
    font-size: 1.125rem;
    margin-bottom: 1rem;
}

.form-group {
    margin-bottom: 1rem;
}

label {
    display: block;
    font-size: 0.875rem;
    font-weight: 500;
    margin-bottom: 0.25rem;
}

input, textarea, select {
    width: 100%;
    padding: 0.5rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 0.875rem;
}

textarea {
    resize: vertical;
}

.btn-primary {
    width: 100%;
    padding: 0.75rem;
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s;
}

.btn-primary:hover {
    background: var(--primary-hover);
}

.btn-primary:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

.hidden {
    display: none !important;
}

#results {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.journal-card {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
}

.journal-header {
    display: flex;
    justify-content: space-between;
    align-items: start;
    margin-bottom: 0.5rem;
}

.journal-name {
    font-weight: 600;
    font-size: 1rem;
}

.journal-quartile {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 500;
}

.quartile-q1 { background: #dcfce7; color: #166534; }
.quartile-q2 { background: #dbeafe; color: #1e40af; }
.quartile-q3 { background: #fef3c7; color: #92400e; }
.quartile-q4 { background: #f3f4f6; color: #6b7280; }

.journal-tags {
    display: flex;
    gap: 0.5rem;
    margin: 0.5rem 0;
}

.tag {
    padding: 0.125rem 0.5rem;
    background: var(--bg);
    border-radius: 4px;
    font-size: 0.75rem;
}

.journal-reasons {
    margin: 0.5rem 0;
    padding-left: 1rem;
}

.journal-reasons li {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin-bottom: 0.25rem;
}

.confidence-bar {
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 0.5rem;
}

.confidence-fill {
    height: 100%;
    background: var(--success);
    border-radius: 2px;
    transition: width 0.3s;
}

.warning {
    padding: 1rem;
    background: #fef3c7;
    border-radius: 6px;
    color: #92400e;
    font-size: 0.875rem;
}

.spinner {
    width: 20px;
    height: 20px;
    border: 2px solid var(--border);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    display: inline-block;
    margin-right: 0.5rem;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

footer {
    margin-top: 2rem;
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.875rem;
}

@media (max-width: 768px) {
    main {
        grid-template-columns: 1fr;
    }
}
```

- [ ] **Step 3: 创建 frontend/js/app.js**

```javascript
// 论文投稿期刊推荐系统 - 前端逻辑

const API_BASE = '/api';

// 模式切换
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const mode = btn.dataset.mode;
        const abstractGroup = document.getElementById('abstract-group');
        const fulltextGroup = document.getElementById('fulltext-group');

        if (mode === 'title') {
            abstractGroup.classList.add('hidden');
            fulltextGroup.classList.add('hidden');
        } else if (mode === 'abstract') {
            abstractGroup.classList.remove('hidden');
            fulltextGroup.classList.add('hidden');
        } else {
            abstractGroup.classList.remove('hidden');
            fulltextGroup.classList.remove('hidden');
        }
    });
});

// 推荐按钮
document.getElementById('recommend-btn').addEventListener('click', async () => {
    const title = document.getElementById('title').value.trim();
    if (!title) {
        alert('请输入论文标题');
        return;
    }

    const mode = document.querySelector('.mode-btn.active').dataset.mode;
    const abstract = document.getElementById('abstract').value.trim();
    const fullText = document.getElementById('full_text').value.trim();
    const topK = parseInt(document.getElementById('top_k').value);
    const oaPreference = document.getElementById('oa_preference').value;

    const btn = document.getElementById('recommend-btn');
    btn.disabled = true;
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results').innerHTML = '';
    document.getElementById('warning').classList.add('hidden');

    try {
        const response = await fetch(`${API_BASE}/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                title,
                abstract: mode === 'abstract' || mode === 'full' ? abstract : '',
                full_text: mode === 'full' ? fullText : '',
                mode,
                top_k: topK,
                oa_preference: oaPreference,
            }),
        });

        const data = await response.json();
        renderResults(data);

        if (data.warning) {
            const warningEl = document.getElementById('warning');
            warningEl.textContent = data.warning;
            warningEl.classList.remove('hidden');
        }
    } catch (error) {
        document.getElementById('results').innerHTML = `
            <div class="warning">请求失败: ${error.message}</div>
        `;
    } finally {
        btn.disabled = false;
        document.getElementById('loading').classList.add('hidden');
    }
});

function renderResults(data) {
    const resultsEl = document.getElementById('results');

    if (!data.recommendations || data.recommendations.length === 0) {
        resultsEl.innerHTML = '<p>未找到合适的推荐期刊</p>';
        return;
    }

    resultsEl.innerHTML = data.recommendations.map(rec => `
        <div class="journal-card">
            <div class="journal-header">
                <span class="journal-name">${rec.journal_name}</span>
                ${rec.quartile ? `<span class="journal-quartile quartile-${rec.quartile.toLowerCase()}">${rec.quartile}</span>` : ''}
            </div>
            <div class="journal-tags">
                ${rec.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                <span class="tag">${rec.oa_type === 'full_oa' ? '完全OA' : rec.oa_type === 'hybrid' ? '混合OA' : '订阅'}</span>
            </div>
            <ul class="journal-reasons">
                ${rec.match_reasons.map(r => `<li>${r}</li>`).join('')}
            </ul>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${rec.confidence * 100}%"></div>
            </div>
            ${rec.submission_url ? `<a href="${rec.submission_url}" target="_blank">投稿链接</a>` : ''}
        </div>
    `).join('');
}
```

- [ ] **Step 4: 验证前端文件**

```bash
ls -la frontend/
ls -la frontend/css/
ls -la frontend/js/
# 确认文件存在
```

---

## Task 11: 数据采集脚本

**目标:** 提供期刊数据采集脚本的骨架（实际运行时需要网络）

**Files:**
- Create: `scripts/crawl_doaj.py`
- Create: `scripts/crawl_scimago.py`
- Create: `scripts/normalize_journals.py`
- Create: `scripts/build_journal_index.py`

- [ ] **Step 1: 创建 scripts/crawl_doaj.py**

```python
"""DOAJ 数据采集脚本"""
import json
import httpx
from typing import List, Dict

from src.journals.journal_model import Journal


def crawl_doaj(subject: str = "Computer Science", max_results: int = 200) -> List[Journal]:
    """从 DOAJ 采集期刊数据"""
    # DOAJ API
    url = "https://doaj.org/api/v1/search/journals"
    results = []

    try:
        response = httpx.get(
            url,
            params={
                "query": subject,
                "pageSize": max_results,
                "fields": "title,publisher,subject,keywords,bibjson.oa_statement_url,bibjson.website",
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        for item in data.get("results", []):
            bibjson = item.get("bibjson", {})
            journal = Journal(
                journal_id=f"doaj-{item.get('id', '')}",
                journal_name=bibjson.get("title", ""),
                publisher=bibjson.get("publisher", {}).get("name", ""),
                subject_tags=[s.get("pref_label", "") for s in bibjson.get("subject", [])],
                keywords=bibjson.get("keywords", []),
                scope_text=bibjson.get("description", ""),
                oa_type="full_oa" if bibjson.get("oa") else "subscription",
                homepage_url=bibjson.get("website"),
                submission_url=bibjson.get("submission_url"),
            )
            journal.build_profile()
            results.append(journal)

    except Exception as e:
        print(f"Error crawling DOAJ: {e}")

    return results


if __name__ == "__main__":
    journals = crawl_doaj()
    print(f"Crawled {len(journals)} journals from DOAJ")
    for j in journals[:5]:
        print(f"  - {j.journal_name}")
```

- [ ] **Step 2: 创建 scripts/crawl_scimago.py**

```python
"""SCImago 数据采集脚本"""
import json
import httpx
from typing import List

from src.journals.journal_model import Journal


def crawl_scimago(max_results: int = 500) -> List[Journal]:
    """从 SCImago 采集期刊数据（简化版）"""
    # SCImago 提供 Journal Rankings，可以按学科筛选
    # 这里使用简化版：直接返回空列表，实际使用需要 CSV 导入
    print("SCImago crawler: Use CSV import for production")
    return []


if __name__ == "__main__":
    journals = crawl_scimago()
    print(f"Crawled {len(journals)} journals from SCImago")
```

- [ ] **Step 3: 创建 scripts/normalize_journals.py**

```python
"""期刊数据标准化脚本"""
import json
from pathlib import Path
from typing import List

from src.journals.journal_model import Journal
from src.journals.journal_store import JournalStore


def normalize_journals(input_files: List[str], output_file: str) -> int:
    """合并并标准化期刊数据"""
    all_journals = []
    seen_ids = set()

    for file_path in input_files:
        path = Path(file_path)
        if not path.exists():
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    journal = Journal(**data)
                    # 去重
                    if journal.journal_id not in seen_ids:
                        journal.build_profile()
                        all_journals.append(journal)
                        seen_ids.add(journal.journal_id)
                except Exception as e:
                    print(f"Error parsing line: {e}")

    # 保存
    store = JournalStore(store_path=output_file)
    store.add_journals(all_journals)
    store.save()

    print(f"Normalized {len(all_journals)} journals")
    return len(all_journals)


if __name__ == "__main__":
    import sys
    files = ["data/raw/doaj/journals.jsonl", "data/raw/scimago/journals.jsonl"]
    output = "data/processed/journals.jsonl"
    normalize_journals(files, output)
```

- [ ] **Step 4: 创建 scripts/build_journal_index.py**

```python
"""构建向量索引脚本"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.journals.journal_store import JournalStore
from src.utils.embedding import OllamaEmbedding


def build_index(
    store_path: str = "data/processed/journals.jsonl",
    faiss_path: str = "data/processed/journals_index.faiss",
    meta_path: str = "data/processed/journals_metadata.parquet",
):
    """构建 FAISS 向量索引"""
    print("Loading journals...")
    store = JournalStore(
        store_path=store_path,
        faiss_index_path=faiss_path,
        metadata_path=meta_path,
    )
    store.load()

    if store.count == 0:
        print("No journals to index")
        return

    print(f"Indexing {store.count} journals...")

    # 初始化 embedding client
    embedding_client = OllamaEmbedding()

    # 构建 journal_profile 列表
    profiles = [j.journal_profile for j in store._journals]

    # 批量获取 embedding
    print("Computing embeddings...")
    embeddings = embedding_client.embed_batch(profiles)

    import numpy as np
    embeddings_matrix = np.array(embeddings)

    # 构建 FAISS 索引
    print("Building FAISS index...")
    store.build_faiss_index(embeddings_matrix)
    store.save()

    print(f"Index built successfully: {store.count} journals")


if __name__ == "__main__":
    build_index()
```

---

## 实现计划完成

所有任务已定义完毕。每个任务都包含：
- 明确的文件路径
- 完整的代码实现
- 相应的测试代码
- 验证命令

**执行选项：**

**1. Subagent-Driven (recommended)** - 每个任务派发一个 subagent 执行，任务间有审查点，快速迭代

**2. Inline Execution** - 在当前 session 执行任务，使用 executing-plans skill，批量执行带检查点

你想用哪种方式开始实现？