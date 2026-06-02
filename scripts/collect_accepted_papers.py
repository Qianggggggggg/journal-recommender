#!/usr/bin/env python3
"""从本地评测元数据生成 ``data/accepted_papers/*.json``。

任务 2.2:本地模式 corpus 构建。读取多份 ``papers_metadata*.jsonl``,按 venue
分组,经 ``JournalStore`` 解析到 ``journal_id`` 后,逐期刊输出符合
``AcceptedPaperStore`` 加载契约的 JSON 文件。

设计要点
========

- ``--exclude-eval-input``: 任何在排除文件中出现的 ``(normalized_title,
  journal_id)`` 对绝不进入 corpus,用来在生成时彻底隔离 heldout / light30
  测试集,防止 retrieval/LTR 阶段把测试论文自己当训练画像 (数据泄漏)。
- ``--source``: 该批数据的来源标记 (默认 ``local_evaluation_metadata``),
  写入每条 paper,后续如果接入外部源可以按 source 区分/回退。
- ``--source semantic-scholar|openalex|local``: 外部来源在阶段 2.3 stub,
  本脚本只实现 local 路径 (任务 2.3 才接外部 stub)。

CLI
===

::

    python scripts/collect_accepted_papers.py \\
        --exclude-eval-input data/evaluation/papers_metadata_light_30.jsonl \\
        --output-dir data/accepted_papers
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

# 让 scripts/ 直接 python 也能找到项目模块
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


logger = logging.getLogger("collect_accepted_papers")


DEFAULT_EVAL_INPUTS = [
    Path("data/evaluation/papers_metadata.jsonl"),
    Path("data/evaluation/papers_metadata_v2.jsonl"),
    Path("data/evaluation/papers_metadata_light_30.jsonl"),
]
DEFAULT_OUTPUT_DIR = Path("data/accepted_papers")
DEFAULT_SOURCE = "local_evaluation_metadata"
# 与 src/evaluation/clean_benchmark.py 的 abstract_snippet 匹配口径保持一致
ABSTRACT_SNIPPET_LENGTH = 160

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# 规范化辅助
# ---------------------------------------------------------------------------


def _normalize_venue(venue: str) -> str:
    """venue 名规范化:lowercase + 去首尾/折叠空白。对齐 run_evaluation.py。"""
    if not venue:
        return ""
    return _WHITESPACE_RE.sub(" ", venue.strip()).lower()


def _normalize_text(text: str) -> str:
    """与 ``src/evaluation/clean_benchmark.py::_normalize_text`` 严格一致:
    NFKD + lowercase + 去掉所有非字母数字字符 + 折叠空白。

    这是 title 比对与 abstract 片段比对的统一口径。任何对其中一处的修改
    必须同步另一处,以保证 ``--exclude-eval-input`` 与 leakage 报告匹配
    的论文集合完全相同。
    """
    text = unicodedata.normalize("NFKD", str(text)).lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _normalize_title(title: str) -> str:
    """title 规范化 (与 _normalize_text 同口径)。用于跨文件去重和 exclude 匹配。"""
    return _normalize_text(title)


def _abstract_snippet_key(abstract: str) -> str:
    """abstract 前 ``ABSTRACT_SNIPPET_LENGTH`` 字符的规范化片段。"""
    norm = _normalize_text(abstract)
    return norm[:ABSTRACT_SNIPPET_LENGTH]


def _build_venue_index(journal_store: Any) -> tuple[dict[str, str], dict[str, str]]:
    """期刊全名 (规范化) → journal_id,以及 journal_id → 原始 journal_name。"""
    name_to_id: dict[str, str] = {}
    id_to_name: dict[str, str] = {}
    for journal in getattr(journal_store, "journals", []):
        jid = getattr(journal, "journal_id", "")
        jname = getattr(journal, "journal_name", "")
        norm = _normalize_venue(jname)
        if jid and norm:
            name_to_id[norm] = jid
            id_to_name[jid] = jname
    return name_to_id, id_to_name


# ---------------------------------------------------------------------------
# 输入读取
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path) -> Iterable[dict]:
    """逐行解析 jsonl;单行解析失败时 log warning 并跳过。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("skip malformed line %s:%d (%s)", path, line_no, exc)
    except FileNotFoundError:
        logger.warning("input file not found: %s", path)


def _build_exclude_keys(
    exclude_inputs: Sequence[Path], venue_index: dict[str, str]
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """收集两类 exclude key 集合,任一命中即算 exclude:

    - title_keys: ``(normalized_title, journal_id)``
    - abstract_keys: ``(abstract_snippet, journal_id)``,abstract 前 160 字符

    一篇 paper 同时会塞进两个集合,collect 主流程只要 title 或 abstract 任一
    匹配就视为已在 exclude 文件中出现。abstract 片段维度兜底了 title 字符串
    差异 (例如 unicode ``√`` vs ASCII ``sqrt``) 这类无法通过 title 规范化
    解决的情况。
    """
    title_keys: set[tuple[str, str]] = set()
    abstract_keys: set[tuple[str, str]] = set()
    for path in exclude_inputs:
        for row in _iter_jsonl(path):
            venue_norm = _normalize_venue(str(row.get("venue", "")))
            if not venue_norm:
                continue
            journal_id = venue_index.get(venue_norm)
            if not journal_id:
                continue

            title_norm = _normalize_title(str(row.get("title", "")))
            if title_norm:
                title_keys.add((title_norm, journal_id))

            abstract_snippet = _abstract_snippet_key(str(row.get("abstract", "")))
            if len(abstract_snippet) >= ABSTRACT_SNIPPET_LENGTH:
                abstract_keys.add((abstract_snippet, journal_id))
    return title_keys, abstract_keys


# ---------------------------------------------------------------------------
# 核心 collect
# ---------------------------------------------------------------------------


def _extract_doi(row: dict) -> str:
    ext = row.get("external_ids") or {}
    if isinstance(ext, dict):
        doi = ext.get("doi") or ext.get("DOI") or ""
        if doi:
            return str(doi).strip()
    return ""


def _extract_url(row: dict) -> str:
    for key in ("pdf_url", "url"):
        val = row.get(key)
        if val:
            return str(val).strip()
    return ""


def _coerce_year(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def collect_accepted_papers(
    *,
    eval_inputs: Sequence[Path],
    exclude_inputs: Sequence[Path],
    journal_store: Any,
    output_dir: Path,
    source: str = DEFAULT_SOURCE,
) -> dict:
    """主流程。返回统计 summary。

    Args:
        eval_inputs: 评测 jsonl 列表 (来源数据)。
        exclude_inputs: 需要从 corpus 排除的 jsonl 列表 (防泄漏)。
        journal_store: 提供 ``.journals`` 序列,每项有 ``journal_id`` /
            ``journal_name`` 属性。
        output_dir: 输出目录,会按 journal_id 写入 ``<id>.json``。
        source: 写入每条 paper 的 ``source`` 字段。
    """
    output_dir = Path(output_dir)
    venue_index, id_to_name = _build_venue_index(journal_store)
    title_exclude_keys, abstract_exclude_keys = _build_exclude_keys(
        [Path(p) for p in exclude_inputs], venue_index
    )

    grouped: dict[str, dict] = {}
    seen_keys: set[tuple[str, str]] = set()
    unresolved_venues: set[str] = set()
    skipped_missing_fields = 0
    excluded = 0
    duplicates = 0
    unresolved_venue_records = 0

    for input_path in eval_inputs:
        for row in _iter_jsonl(Path(input_path)):
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            venue = (row.get("venue") or "").strip()
            if not title or not abstract or not venue:
                skipped_missing_fields += 1
                continue

            venue_norm = _normalize_venue(venue)
            journal_id = venue_index.get(venue_norm)
            if not journal_id:
                unresolved_venues.add(venue)
                unresolved_venue_records += 1
                continue

            title_norm = _normalize_title(title)
            abstract_snippet = _abstract_snippet_key(abstract)
            key = (title_norm, journal_id)

            title_hit = key in title_exclude_keys
            abstract_hit = (
                len(abstract_snippet) >= ABSTRACT_SNIPPET_LENGTH
                and (abstract_snippet, journal_id) in abstract_exclude_keys
            )
            if title_hit or abstract_hit:
                excluded += 1
                continue
            if key in seen_keys:
                duplicates += 1
                continue
            seen_keys.add(key)

            paper = {
                "title": title,
                "abstract": abstract,
                "year": _coerce_year(row.get("year")),
                "source": source,
                "doi": _extract_doi(row),
                "url": _extract_url(row),
            }

            bucket = grouped.setdefault(
                journal_id,
                {
                    "journal_id": journal_id,
                    "journal_name": id_to_name.get(journal_id, ""),
                    "papers": [],
                },
            )
            bucket["papers"].append(paper)

    output_dir.mkdir(parents=True, exist_ok=True)
    for journal_id, payload in grouped.items():
        out_path = output_dir / f"{journal_id}.json"
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    summary = {
        "journal_count": len(grouped),
        "paper_count": sum(len(p["papers"]) for p in grouped.values()),
        "skipped_missing_fields_count": skipped_missing_fields,
        "unresolved_venue_count": unresolved_venue_records,
        "unresolved_venues": sorted(unresolved_venues),
        "excluded_count": excluded,
        "duplicate_count": duplicates,
        "source": source,
        "output_dir": str(output_dir),
        "eval_inputs": [str(p) for p in eval_inputs],
        "exclude_inputs": [str(p) for p in exclude_inputs],
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--eval-input",
        action="append",
        type=Path,
        help="评测 metadata jsonl 路径。可多次指定。若不指定,使用三份默认输入。",
    )
    parser.add_argument(
        "--exclude-eval-input",
        action="append",
        type=Path,
        default=[],
        help="生成 corpus 时需要排除的评测 jsonl,用于隔离测试集 (防泄漏)。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="accepted-paper JSON 输出目录,默认 data/accepted_papers。",
    )
    parser.add_argument(
        "--source",
        choices=["local", "semantic-scholar", "openalex"],
        default="local",
        help="数据源。当前阶段只实现 local;其他选项预留给任务 2.3。",
    )
    parser.add_argument(
        "--journal-store-path",
        default="data/processed/journals.jsonl",
        help="JournalStore 的 jsonl 路径,用于 venue→journal_id 解析。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印 INFO 级别日志。",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.source != "local":
        # 任务 2.3 才会真的实现外部源;现在按计划直接清晰退出。
        print(
            f"external collection source is not enabled in this plan phase (got: {args.source})",
            file=sys.stderr,
        )
        return 2

    eval_inputs = args.eval_input or DEFAULT_EVAL_INPUTS

    from src.journals.journal_store import JournalStore

    journal_store = JournalStore(store_path=args.journal_store_path)
    journal_store.load()
    if journal_store.count == 0:
        print(
            f"JournalStore is empty at {args.journal_store_path}; cannot resolve venues",
            file=sys.stderr,
        )
        return 1

    summary = collect_accepted_papers(
        eval_inputs=eval_inputs,
        exclude_inputs=args.exclude_eval_input,
        journal_store=journal_store,
        output_dir=args.output_dir,
        source=DEFAULT_SOURCE,
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
