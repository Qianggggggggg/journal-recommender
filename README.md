# 论文投稿期刊推荐系统

根据论文内容（标题/摘要/全文）推荐适合投稿的计算机类期刊。

## 功能特性

- **三种输入模式**：标题模式 / 摘要模式 / 全文模式
- **混合召回**：BM25 + 向量检索 + 标签过滤
- **两阶段排序**：规则打分 → LLM 精排
- **独立解释**：每条推荐附带匹配理由
- **降级策略**：LLM 失败时回退到规则分类

## 环境要求

- Python 3.11+
- Ollama 服务（用于 Embedding，本地部署 qwen3-embedding:4b）
- MiniMax API Key（用于 LLM 调用）

## 安装

### 1. 创建虚拟环境

```bash
# 创建名为 paper 的虚拟环境
python -m venv paper

# 激活虚拟环境
source paper/bin/activate
```

### 2. 安装依赖

```bash
# 安装项目及所有依赖
pip install -e .
```

如果 `pip install -e .` 失败（hatchling 问题），可以手动安装依赖：

```bash
pip install fastapi uvicorn pydantic httpx faiss-cpu rank-bm25 pyyaml python-dotenv numpy pandas tenacity pyarrow
```

### 3. 配置环境变量

创建 `.env` 文件：

```bash
# MiniMax API
MINIMAX_API_KEY=your_api_key_here
MINIMAX_BASE_URL=https://api.minimax.chat

# Ollama Embedding（本地服务）
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=qwen3-embedding:4b

# 应用配置
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
```

### 4. 准备数据（首次运行）

```bash
# 创建示例期刊数据
python scripts/create_sample_journals.py

# 构建向量索引
python scripts/build_journal_index.py
```

或者手动运行：

```bash
# DOAJ 数据采集（如 DOAJ API 可用）
python scripts/crawl_doaj.py

# SCImago 数据采集
python scripts/crawl_scimago.py

# 标准化并构建索引
python scripts/normalize_journals.py
python scripts/build_journal_index.py
```

## 启动项目

### 方式一：直接运行

```bash
# 激活虚拟环境
source paper/bin/activate

# 启动服务
python -m src.app.main
```

### 方式二：使用 uvicorn

```bash
# 激活虚拟环境
source paper/bin/activate

# 启动服务（支持热重载）
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 访问

启动后访问：http://localhost:8000

## 重启项目

### 正常重启

```bash
# 1. 停止当前服务（Ctrl+C）

# 2. 重新激活虚拟环境
source paper/bin/activate

# 3. 重新启动
python -m src.app.main
```

### 代码修改后重启

如果使用 `uvicorn --reload`，代码修改后会自动重载。

手动重启：

```bash
# 1. 停止当前服务
# 在运行 uvicorn 的终端按 Ctrl+C

# 2. 重新启动
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 清理后重启

如果需要重新构建数据：

```bash
# 1. 停止当前服务

# 2. 删除旧数据（可选）
rm -rf data/processed/journals.jsonl
rm -rf data/processed/journals_index.faiss
rm -rf data/processed/journals_metadata.parquet

# 3. 重新采集数据
python scripts/create_sample_journals.py
python scripts/build_journal_index.py

# 4. 重启服务
python -m src.app.main
```

## 项目结构

```
journal-recommender/
├── configs/                  # 配置文件
│   ├── app.yaml              # 应用配置
│   ├── prompts.yaml          # Prompt 模板
│   └── journal_taxonomy.yaml # 期刊分类体系
├── data/
│   ├── processed/            # 处理后的数据
│   │   ├── journals.jsonl    # 期刊数据
│   │   ├── journals_index.faiss  # FAISS 向量索引
│   │   └── journals_metadata.parquet
│   └── raw/                  # 原始数据
├── scripts/                  # 数据采集脚本
│   ├── crawl_doaj.py
│   ├── crawl_scimago.py
│   ├── normalize_journals.py
│   ├── build_journal_index.py
│   └── create_sample_journals.py
├── src/
│   ├── app/                  # FastAPI 接口
│   ├── journals/             # 期刊数据模型
│   ├── papers/               # 论文解析
│   ├── retriever/            # 召回模块
│   ├── ranker/               # 排序模块
│   ├── recommender/          # 推荐流程
│   └── utils/                # 工具模块
├── frontend/                 # Web 前端
├── tests/                    # 测试
└── pyproject.toml
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `GET /` | GET | 首页（重定向到前端） |
| `POST /api/recommend` | POST | 推荐期刊 |
| `GET /api/journals` | GET | 列出期刊 |
| `GET /api/health` | GET | 健康检查 |

### 推荐接口示例

```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Deep Learning for Image Recognition",
    "abstract": "This paper proposes a new method for image recognition using deep learning.",
    "mode": "abstract",
    "top_k": 5
  }'
```

## 常见问题

### 1. Ollama 服务未运行

```bash
# 启动 Ollama 服务
ollama serve

# 确认模型已加载
ollama list
```

### 2. FAISS 索引未构建

```bash
python scripts/build_journal_index.py
```

### 3. 端口被占用

```bash
# 查找占用端口的进程
lsof -i :8000

# 或使用其他端口
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8080
```

## 停止服务

```bash
# 在运行服务的终端按 Ctrl+C

# 或强制停止
pkill -f "src.app.main"
```

---

**版本**: 0.1.0
**作者**: Journal Recommender Team