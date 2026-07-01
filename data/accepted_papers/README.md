# 真实已发表论文画像 (Accepted-Paper Profiles)

这个目录存放每本期刊真实发表过的论文,作为期刊画像的另一个维度,
与 `data/typical_abstracts/`(LLM 生成的"典型"摘要)互补。

## 一个期刊一个 JSON 文件

文件名约定:`<journal_id>.json`。`journal_id` 与 `data/journals_*.jsonl` 中
的 ID 保持一致。

## 文件格式

```json
{
  "journal_id": "ton",
  "journal_name": "IEEE/ACM Transactions on Networking",
  "papers": [
    {
      "title": "Paper title",
      "abstract": "Paper abstract",
      "year": 2025,
      "source": "local_evaluation_metadata",
      "doi": "",
      "url": ""
    }
  ]
}
```

字段说明:

- `journal_id` (必填):期刊唯一 ID,与 `JournalStore` 中的 ID 对齐。
- `journal_name` (必填):期刊全名。
- `papers` (必填):该期刊的论文数组。每条至少含 `title` + `abstract`,
  其他字段可缺;`AcceptedPaperStore` 会用稳定默认值填充。

每条 paper 内字段:

- `title` (必填):论文标题。
- `abstract` (必填):论文摘要。
- `year` (可选):发表年份,整数。
- `source` (可选):数据来源标记,如 `local_evaluation_metadata`、
  `semantic_scholar`、`openalex` 等。便于追溯。
- `doi` (可选):DOI 字符串。
- `url` (可选):原文 URL。

## 加载约定

`src/journals/accepted_paper_store.py::AcceptedPaperStore` 提供以下能力:

- `load()`:加载整个目录;目录缺失/单文件 JSON 损坏都不会让加载失败。
- `get_papers(journal_id)`:取某期刊的论文(已规范化为统一字段集)。
- `iter_records()`:迭代所有 `(paper, journal)` 配对。
- `title` 或 `abstract` 缺失/纯空白的论文条目自动跳过。

## 纪律

- **泄漏控制**:任何 held-out / heldout-final benchmark 中的论文都不得
  出现在这里。`scripts/clean_benchmark.py` 已支持 `--accepted-paper-dir`
  做泄漏报告。
- **来源标注**:必须正确填 `source`,以便后续可以按来源做消融或回退。
- **每条记录最低门槛**:`title` + `abstract` 必填。其他字段允许后续补全。
