"""典型摘要生成器"""
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..utils.llm import MiniMaxLLM, parse_json_response

logger = logging.getLogger(__name__)


# 方法主导类型
METHOD_TYPES = [
    "理论分析型",    # 以理论分析、形式化证明为主的论文
    "系统设计型",    # 以系统设计、实现、架构创新为主的论文
    "实验评估型",    # 以实验评估、基准测试、性能分析为主的论文
    "算法改进型",    # 以算法改进、优化、方法创新为主的论文
    "应用驱动型",    # 以应用场景、实际问题驱动的研究型论文
]

# 创新层次
NOVELTY_LEVELS = [
    "范式创新",      # 提出全新的研究范式或框架
    "显著改进",      # 在现有方法上有显著的性能或效果提升
    "增量完善",      # 对现有方法进行增量式优化或扩展
]


@dataclass
class TypicalAbstract:
    """一篇典型摘要"""
    method_type: str
    novelty_level: str
    abstract: str
    journal_id: str
    journal_name: str


class TypicalAbstractGenerator:
    """典型摘要生成器"""

    SYSTEM_PROMPT = """你是一个学术期刊专家，负责为学术期刊生成"典型论文摘要"。

生成规则：
1. 每篇摘要应该代表该期刊在指定维度和创新层次下的典型发表风格
2. 摘要长度：200-300词（英文），结构完整（背景→方法→结果→贡献）
3. 使用学术写作风格，避免泛泛而谈
4. 摘要中应自然融入该期刊的scope_text关键词和研究方向
5. 不要使用原文的scope_text短语，保持原创性
6. 4篇摘要要有明显差异，覆盖不同角度

输出格式（JSON数组）：
[
    {
        "method_type": "理论分析型",
        "novelty_level": "范式创新",
        "abstract": "生成的英文摘要内容",
        "keywords": ["关键词1", "关键词2", "关键词3"],
        "contribution": "一句话贡献描述"
    },
    ...
]"""

    USER_PROMPT_TEMPLATE_BATCH = """请为以下学术期刊一次性生成4篇典型论文摘要（2种方法类型 × 2种创新层次）。

## 期刊信息
- 期刊名称：{journal_name}
- CCF等级：{ccf_rating}
- 期刊Scope：{scope_text}
- 关键词：{keywords}

## 必须生成的4篇摘要（严格按此顺序）
1. 方法类型：{method_type_1}，创新层次：{novelty_level_1}
2. 方法类型：{method_type_1}，创新层次：{novelty_level_2}
3. 方法类型：{method_type_2}，创新层次：{novelty_level_1}
4. 方法类型：{method_type_2}，创新层次：{novelty_level_2}

每篇200-300词，使用学术英文。
重要：必须返回包含4个对象的JSON数组！"""

    VALIDATION_PROMPT = """请验证以下摘要是否符合要求。

## 要求
- 方法主导类型：{method_type}
- 创新层次：{novelty_level}

## 摘要内容
{abstract}

请返回JSON格式：
{{"valid": true/false, "reason": "验证原因"}}"""

    def __init__(
        self,
        llm: Optional[MiniMaxLLM] = None,
        model: str = "MiniMax-M2.7",
        temperature: float = 0.7,
    ):
        self.llm = llm or MiniMaxLLM(model=model, temperature=temperature)

    def generate_one(
        self,
        journal_id: str,
        journal_name: str,
        scope_text: str,
        keywords: list[str],
        ccf_rating: str,
        method_type: str,
        novelty_level: str,
    ) -> TypicalAbstract:
        """为单本期刊生成一篇典型摘要"""
        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            journal_name=journal_name,
            ccf_rating=ccf_rating or "未知",
            scope_text=scope_text or "未提供",
            keywords=", ".join(keywords) if keywords else "无",
            method_type=method_type,
            novelty_level=novelty_level,
        )

        # 调用 LLM
        try:
            response = self.llm.chat_auto(self.SYSTEM_PROMPT, user_prompt, timeout=120)
        except Exception as e:
            logger.error(f"LLM调用失败 for {journal_id} ({method_type}/{novelty_level}): {e}")
            return TypicalAbstract(
                method_type=method_type,
                novelty_level=novelty_level,
                abstract="",
                journal_id=journal_id,
                journal_name=journal_name,
            )

        # 解析 JSON
        data = parse_json_response(response.content)
        if not data:
            logger.warning(f"无法解析LLM响应 for {journal_id} ({method_type}/{novelty_level})")
            return TypicalAbstract(
                method_type=method_type,
                novelty_level=novelty_level,
                abstract="",
                journal_id=journal_id,
                journal_name=journal_name,
            )

        abstract = data.get("abstract", "").strip()
        return TypicalAbstract(
            method_type=method_type,
            novelty_level=novelty_level,
            abstract=abstract,
            journal_id=journal_id,
            journal_name=journal_name,
        )

    def generate_batch_for_journal(
        self,
        journal_id: str,
        journal_name: str,
        scope_text: str,
        keywords: list[str],
        ccf_rating: str,
        method_type_1: str,
        method_type_2: str,
        novelty_level_1: str,
        novelty_level_2: str,
    ) -> list[TypicalAbstract]:
        """一次API调用生成4篇典型摘要（2方法类型 × 2创新层次）"""
        user_prompt = self.USER_PROMPT_TEMPLATE_BATCH.format(
            journal_name=journal_name,
            ccf_rating=ccf_rating or "未知",
            scope_text=scope_text or "未提供",
            keywords=", ".join(keywords) if keywords else "无",
            method_type_1=method_type_1,
            method_type_2=method_type_2,
            novelty_level_1=novelty_level_1,
            novelty_level_2=novelty_level_2,
        )

        try:
            response = self.llm.chat_auto(self.SYSTEM_PROMPT, user_prompt, timeout=180)
        except Exception as e:
            logger.error(f"LLM调用失败 for {journal_id}: {e}")
            return [
                TypicalAbstract(method_type=method_type_1, novelty_level=novelty_level_1, abstract="", journal_id=journal_id, journal_name=journal_name),
                TypicalAbstract(method_type=method_type_1, novelty_level=novelty_level_2, abstract="", journal_id=journal_id, journal_name=journal_name),
                TypicalAbstract(method_type=method_type_2, novelty_level=novelty_level_1, abstract="", journal_id=journal_id, journal_name=journal_name),
                TypicalAbstract(method_type=method_type_2, novelty_level=novelty_level_2, abstract="", journal_id=journal_id, journal_name=journal_name),
            ]

        data = parse_json_response(response.content)
        if not data or not isinstance(data, list):
            logger.warning(f"无法解析LLM响应 for {journal_id}, 返回了: {response.content[:200]}")
            return [
                TypicalAbstract(method_type=method_type_1, novelty_level=novelty_level_1, abstract="", journal_id=journal_id, journal_name=journal_name),
                TypicalAbstract(method_type=method_type_1, novelty_level=novelty_level_2, abstract="", journal_id=journal_id, journal_name=journal_name),
                TypicalAbstract(method_type=method_type_2, novelty_level=novelty_level_1, abstract="", journal_id=journal_id, journal_name=journal_name),
                TypicalAbstract(method_type=method_type_2, novelty_level=novelty_level_2, abstract="", journal_id=journal_id, journal_name=journal_name),
            ]

        # 构建结果（按顺序对应4个组合）
        combos = [
            (method_type_1, novelty_level_1),
            (method_type_1, novelty_level_2),
            (method_type_2, novelty_level_1),
            (method_type_2, novelty_level_2),
        ]

        abstracts = []
        for i, (mt, nl) in enumerate(combos):
            item = data[i] if i < len(data) else {}
            abstracts.append(TypicalAbstract(
                method_type=mt,
                novelty_level=nl,
                abstract=item.get("abstract", "").strip(),
                journal_id=journal_id,
                journal_name=journal_name,
            ))

        return abstracts

    def generate_for_journal(
        self,
        journal_id: str,
        journal_name: str,
        scope_text: str,
        keywords: list[str],
        ccf_rating: str,
        method_types: list[str] = None,
        novelty_levels: list[str] = None,
    ) -> list[TypicalAbstract]:
        """为单本期刊生成多篇典型摘要（覆盖多维度），使用批量API调用"""
        method_types = method_types or METHOD_TYPES[:2]  # 默认取前2种
        novelty_levels = novelty_levels or NOVELTY_LEVELS[:2]  # 默认取前2种

        return self.generate_batch_for_journal(
            journal_id=journal_id,
            journal_name=journal_name,
            scope_text=scope_text,
            keywords=keywords,
            ccf_rating=ccf_rating,
            method_type_1=method_types[0],
            method_type_2=method_types[1],
            novelty_level_1=novelty_levels[0],
            novelty_level_2=novelty_levels[1],
        )


def generate_all_abstracts(
    journals_path: str = "data/processed/journals.jsonl",
    output_dir: str = "data/typical_abstracts",
    max_workers: int = 3,
) -> dict:
    """批量生成所有期刊的典型摘要"""
    # 加载期刊数据
    journals = []
    with open(journals_path, "r", encoding="utf-8") as f:
        for line in f:
            journals.append(json.loads(line))

    logger.info(f"Loaded {len(journals)} journals")

    # 初始化生成器
    generator = TypicalAbstractGenerator()

    # 输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 统计
    stats = {
        "total": len(journals),
        "succeeded": 0,
        "failed": 0,
        "empty": 0,
    }

    # 批量生成
    for i, journal in enumerate(journals):
        jid = journal["journal_id"]
        logger.info(f"[{i+1}/{len(journals)}] Generating for {jid}...")

        abstracts = generator.generate_for_journal(
            journal_id=jid,
            journal_name=journal.get("journal_name", ""),
            scope_text=journal.get("scope_text", ""),
            keywords=journal.get("keywords", []),
            ccf_rating=journal.get("ccf_rating", ""),
        )

        # 保存结果
        output_file = output_path / f"{jid}.json"
        data = {
            "journal_id": jid,
            "journal_name": journal.get("journal_name", ""),
            "ccf_rating": journal.get("ccf_rating", ""),
            "abstracts": [
                {
                    "method_type": a.method_type,
                    "novelty_level": a.novelty_level,
                    "abstract": a.abstract,
                }
                for a in abstracts
            ],
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 统计
        non_empty = sum(1 for a in abstracts if a.abstract)
        if non_empty == len(abstracts):
            stats["succeeded"] += 1
        elif non_empty == 0:
            stats["empty"] += 1
        else:
            stats["failed"] += 1

        logger.info(f"  -> {non_empty}/{len(abstracts)} non-empty abstracts")

    logger.info(f"\n{'='*50}")
    logger.info(f"Summary: {stats}")
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    generate_all_abstracts()
