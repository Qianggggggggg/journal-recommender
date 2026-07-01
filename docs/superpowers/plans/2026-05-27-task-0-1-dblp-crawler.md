# Task 0.1: DBLP 标题爬取

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 295 本期刊各爬取 2020-2024 年发表的论文标题（每刊最多 50 篇），去除 special issue、book review 等非研究性条目，存储为 `journal_id → [title_1, title_2, ...]`。

**Architecture:**
1. 从 `journals.jsonl` 加载 295 本期刊
2. 与 `journals_backup.jsonl` 交叉匹配获取已知 DBLP URL（覆盖 ~175 本）
3. 对剩余期刊，通过 journal_id 构造或 DBLP API 按期刊名搜索获取 URL
4. 并行爬取各期刊 DBLP 页面，提取 2020-2024 年论文标题
5. 清洗过滤，去除非研究性条目
6. 输出到 `data/dblp_titles/{journal_id}.json`

**Tech Stack:** Python 3.10+, requests, BeautifulSoup4, lxml, concurrent.futures, logging

---

## 已知数据情况

| 数据源 | 数量 | 说明 |
|-------|-----|------|
| journals.jsonl | 295 | 目标期刊列表，含 journal_id |
| journals_backup.jsonl | 556 | 含 263 个 DBLP URL，与当前 175 个 journal_id 匹配 |

---

## 任务步骤

### Step 1: 创建目录结构

```bash
mkdir -p data/dblp_titles
```

### Step 2: 创建 DBLP 爬虫基础模块

**File:** `src/generation/dblp_crawler.py`

```python
"""DBLP 标题爬虫"""
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class DBLPJournalPapers:
    """一本期刊的论文数据"""
    journal_id: str
    titles: list[str]
    dblp_url: str
    year_range: tuple[int, int] = (2020, 2024)


class DBLPCrawler:
    """DBLP 期刊论文标题爬虫"""

    # DBLP base URL patterns
    DBLP_JOURNAL_URL_TEMPLATE = "http://dblp.uni-trier.de/db/journals/{abbr}/"
    DBLP_SEARCH_API = "https://dblp.org/search/publ/api"

    # Filter patterns for non-research entries
    FILTER_PATTERNS = [
        re.compile(r"special\s*issue", re.I),
        re.compile(r"book\s*review", re.I),
        re.compile(r"editorial", re.I),
        re.compile(r"Erratum", re.I),
        re.compile(r"Correction", re.I),
        re.compile(r"Obituary", re.I),
        re.compile(r"Retraction", re.I),
    ]

    # Stop words indicating non-English titles
    NON_ENGLISH_PATTERN = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]")

    def __init__(self, timeout: int = 30, max_retries: int = 3, delay: float = 1.0):
        self.timeout = timeout
        self.max_retries = max_retries
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; JournalRecommenderBot/1.0; research project)",
            "Accept": "text/html,application/xhtml+xml",
        })

    def is_research_title(self, title: str) -> bool:
        """判断是否为有效研究论文标题"""
        title_lower = title.lower()
        # Filter by pattern
        for pattern in self.FILTER_PATTERNS:
            if pattern.search(title):
                return False
        # Must be English (ASCII letters present)
        if self.NON_ENGLISH_PATTERN.search(title):
            return False
        return True

    def fetch_journal_page(self, url: str, year_range: tuple[int, int]) -> Optional[str]:
        """爬取期刊页面 HTML"""
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt+1}/{self.max_retries} failed for {url}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))
        return None

    def parse_titles_from_html(self, html: str, year_range: tuple[int, int]) -> list[str]:
        """从 HTML 解析论文标题"""
        soup = BeautifulSoup(html, "lxml")
        titles = []

        # Find all <li> items with class "article" or "inproceedings"
        for item in soup.find_all("li", class_=["article", "inproceedings"]):
            # Extract year from the item
            year_elem = item.find("span", class_="year")
            if not year_elem:
                continue
            try:
                year = int(year_elem.get_text().strip())
            except ValueError:
                continue

            # Filter by year range
            if not (year_range[0] <= year <= year_range[1]):
                continue

            # Extract title from <span class="title">
            title_elem = item.find("span", class_="title")
            if title_elem:
                title = title_elem.get_text().strip()
                # Remove leading/trailing punctuation
                title = title.strip(".,;:")
                if title and self.is_research_title(title):
                    titles.append(title)

        return titles

    def crawl_journal(self, journal_id: str, url: str, year_range: tuple[int, int] = (2020, 2024)) -> DBLPJournalPapers:
        """爬取单本期刊的论文标题"""
        logger.info(f"Crawling {journal_id} from {url}")
        html = self.fetch_journal_page(url, year_range)
        if not html:
            logger.warning(f"Failed to fetch {journal_id}")
            return DBLPJournalPapers(journal_id=journal_id, titles=[], dblp_url=url)

        titles = self.parse_titles_from_html(html, year_range)
        logger.info(f"Extracted {len(titles)} titles for {journal_id}")
        return DBLPJournalPapers(journal_id=journal_id, titles=titles, dblp_url=url)
```

### Step 3: 创建 URL 匹配器模块

**File:** `src/generation/dblp_url_resolver.py`

```python
"""DBLP URL 解析器：将 journal_id 映射到 DBLP URL"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DBLPURLResolver:
    """解析期刊到 DBLP URL 的映射"""

    # ACM/IEEE 缩写到 DBLP 的映射规则
    ACM_PREFIX_MAP = {
        "ACM Transactions on": "t",
        "Journal of the ACM": "jacm",
    }

    IEEE_PREFIX_MAP = {
        "IEEE Transactions on": "t",
        "IEEE Transactions": "tc",
        "IEEE Transactions on": "to",
    }

    def __init__(self, journals_path: str, backup_path: str):
        self.journals_path = journals_path
        self.backup_path = backup_path
        self._load_data()

    def _load_data(self):
        """加载并匹配期刊数据"""
        # Load current journals
        self.current_journals = {}
        with open(self.journals_path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                self.current_journals[j["journal_id"]] = j

        # Load backup journals with DBLP URLs
        self.backup_dblp = {}
        with open(self.backup_path, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                url = j.get("submission_url", "")
                if url and "dblp" in url.lower():
                    self.backup_dblp[j["journal_id"]] = url

        # Match by journal_id
        self.matched_urls = {}
        for jid, j in self.current_journals.items():
            if jid in self.backup_dblp:
                self.matched_urls[jid] = self.backup_dblp[jid]

        logger.info(f"Matched DBLP URLs: {len(self.matched_urls)}/{len(self.current_journals)}")

    def construct_url_from_id(self, journal_id: str, journal_name: str) -> Optional[str]:
        """从 journal_id 和 journal_name 构造 DBLP URL"""
        # Try common patterns
        # Pattern 1: journal_id is already the DBLP abbrev
        if journal_id:
            # Common DBLP abbrevs
            constructed = f"http://dblp.uni-trier.de/db/journals/{journal_id}/"
            return constructed
        return None

    def search_dblp_by_name(self, journal_name: str) -> Optional[str]:
        """通过 DBLP 搜索 API 按期刊名查找 URL（备用）"""
        import requests
        # DBLP search API
        params = {
            "q": journal_name,
            "format": "json",
            "h": 5,
        }
        try:
            resp = requests.get("https://dblp.org/search/publ/api", params=params, timeout=10)
            data = resp.json()
            # Parse hits and find journal URL
            for hit in data.get("result", {}).get("hits", {}).get("hit", []):
                info = hit.get("info", {})
                if "journal" in info.get("type", "").lower():
                    return info.get("url", "")
        except Exception:
            pass
        return None

    def get_url(self, journal_id: str, journal_name: str) -> Optional[str]:
        """获取期刊的 DBLP URL"""
        # 1. 优先从已匹配的 backup 中查找
        if journal_id in self.matched_urls:
            return self.matched_urls[journal_id]

        # 2. 尝试从 journal_id 构造
        url = self.construct_url_from_id(journal_id, journal_name)
        if url:
            return url

        # 3. 备用：搜索
        return self.search_dblp_by_name(journal_name)
```

### Step 4: 创建批量爬取脚本

**File:** `src/generation/crawl_all_journals.py`

```python
#!/usr/bin/env python3
"""批量爬取所有期刊的 DBLP 论文标题"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dblp_crawler import DBLPCrawler
from dblp_url_resolver import DBLPURLResolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    base_dir = Path("data/dblp_titles")
    base_dir.mkdir(exist_ok=True)

    # 初始化 URL 解析器
    resolver = DBLPURLResolver(
        journals_path="data/processed/journals.jsonl",
        backup_path="data/processed/journals_backup.jsonl",
    )

    # 初始化爬虫
    crawler = DBLPCrawler(timeout=30, max_retries=3, delay=1.0)

    # 统计
    success_count = 0
    fail_count = 0
    no_url_count = 0

    # 并行爬取（限制并发数）
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for jid, j in resolver.current_journals.items():
            journal_name = j.get("journal_name", "")
            url = resolver.get_url(jid, journal_name)

            if not url:
                logger.warning(f"No DBLP URL for {jid}: {journal_name}")
                no_url_count += 1
                continue

            future = executor.submit(crawler.crawl_journal, jid, url)
            futures[future] = jid

        for future in as_completed(futures):
            jid = futures[future]
            try:
                result = future.result()
                # 保存结果
                output_path = base_dir / f"{jid}.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "journal_id": result.journal_id,
                        "dblp_url": result.dblp_url,
                        "titles": result.titles,
                    }, f, ensure_ascii=False, indent=2)

                if result.titles:
                    success_count += 1
                    logger.info(f"✓ {jid}: {len(result.titles)} titles")
                else:
                    fail_count += 1
                    logger.warning(f"✗ {jid}: no titles extracted")

            except Exception as e:
                fail_count += 1
                logger.error(f"✗ {jid}: {e}")

    # 总结
    logger.info(f"\n{'='*50}")
    logger.info(f"Summary: {success_count} succeeded, {fail_count} failed, {no_url_count} no URL")
    logger.info(f"Data saved to: {base_dir}")

    # 生成汇总文件
    summary = []
    for f in base_dir.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
            summary.append({
                "journal_id": data["journal_id"],
                "title_count": len(data["titles"]),
                "dblp_url": data["dblp_url"],
            })
    summary.sort(key=lambda x: x["title_count"], reverse=True)

    with open(base_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 统计覆盖率
    journals_with_50 = sum(1 for s in summary if s["title_count"] >= 30)  # 放宽到30作为"成功"阈值
    coverage = journals_with_50 / len(summary) * 100 if summary else 0
    logger.info(f"Coverage (>30 titles): {journals_with_50}/{len(summary)} ({coverage:.1f}%)")


if __name__ == "__main__":
    main()
```

### Step 5: 创建清洗脚本

**File:** `src/generation/clean_dblp_titles.py`

```python
#!/usr/bin/env python3
"""清洗 DBLP 标题：去除噪声、验证英文、统计"""
import json
import re
from pathlib import Path
from collections import defaultdict

NON_ENGLISH = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")
FILTER_KEYWORDS = ["special issue", "book review", "editorial", "erratum", "correction", "retraction", "obituary"]


def is_valid_title(title: str) -> bool:
    """验证标题是否有效"""
    t = title.lower()
    # 过滤关键词
    for kw in FILTER_KEYWORDS:
        if kw in t:
            return False
    # 必须有英文字母
    if not re.search(r"[a-zA-Z]", title):
        return False
    # 过滤纯中文
    if NON_ENGLISH.search(title):
        return False
    return True


def clean_titles(titles: list[str]) -> list[str]:
    """清洗标题列表"""
    cleaned = []
    for t in titles:
        t = t.strip()
        # 去除首尾标点
        t = re.sub(r"^[.,;:]+|[.,;:]+$", "", t)
        if t and is_valid_title(t):
            cleaned.append(t)
    return cleaned


def main():
    src_dir = Path("data/dblp_titles")
    stats = []

    for fp in sorted(src_dir.glob("*.json")):
        if fp.name == "summary.json":
            continue
        with open(fp) as f:
            data = json.load(f)

        original_count = len(data["titles"])
        cleaned = clean_titles(data["titles"])
        cleaned_count = len(cleaned)

        # 更新文件
        data["titles"] = cleaned
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        stats.append({
            "journal_id": data["journal_id"],
            "original": original_count,
            "cleaned": cleaned_count,
            "removed": original_count - cleaned_count,
        })

    # 汇总统计
    total_original = sum(s["original"] for s in stats)
    total_cleaned = sum(s["cleaned"] for s in stats)
    logger.info(f"Total: {total_original} -> {total_cleaned} (removed {total_original - total_cleaned})")

    # 按清洗比例排序，报告异常
    for s in sorted(stats, key=lambda x: x["removed"]/max(x["original"],1), reverse=True)[:10]:
        if s["removed"] > 5:
            logger.warning(f"{s['journal_id']}: removed {s['removed']}/{s['original']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

---

## 验证步骤

- [ ] **验证1：运行爬虫脚本**

```bash
cd /Users/qian/PycharmProjects/paper
python -m src.generation.crawl_all_journals
```

Expected output: 爬取日志，显示成功/失败数量，data/dblp_titles/ 下生成 ~295 个 JSON 文件

- [ ] **验证2：检查输出**

```bash
ls data/dblp_titles/ | wc -l  # 预期 > 260
python3 -c "
import json
from pathlib import Path
titles_dir = Path('data/dblp_titles')
stats = []
for fp in titles_dir.glob('*.json'):
    if fp.name == 'summary.json': continue
    with open(fp) as f:
        d = json.load(f)
    stats.append((d['journal_id'], len(d['titles'])))
stats.sort(key=lambda x: x[1], reverse=True)
print(f'Top 5 journals by title count:')
for jid, cnt in stats[:5]:
    print(f'  {jid}: {cnt} titles')
print(f'Total journals: {len(stats)}')
print(f'Journals with >30 titles: {sum(1 for _, c in stats if c >= 30)}')
"
```

Expected: >90% 期刊成功爬取，每刊 >= 30 篇标题

- [ ] **验证3：清洗效果**

```bash
python -m src.generation.clean_dblp_titles
```

Expected: 日志显示清洗前后标题数量差异，人工抽检 data/dblp_titles/{journal_id}.json 确认无噪声标题

---

## 文件清单

| File | Action |
|-----|--------|
| `src/generation/__init__.py` | Create (empty init) |
| `src/generation/dblp_crawler.py` | Create |
| `src/generation/dblp_url_resolver.py` | Create |
| `src/generation/crawl_all_journals.py` | Create |
| `src/generation/clean_dblp_titles.py` | Create |
| `data/dblp_titles/*.json` | Generated output |

---

## 风险与缓解

| 风险 | 缓解 |
|-----|------|
| DBLP 反爬虫限制 | 添加 delay=1s，最大重试3次，User-Agent 标识为研究项目 |
| 部分期刊无 DBLP URL | 备用：通过 journal_id 构造 + DBLP search API 搜索 |
| 爬取内容不足（<30篇/刊）| 放宽阈值至20篇；统计时标记为"部分覆盖" |
