"""Feature builder for LTR reranker (Task 4.1).

本模块定义 paper-candidate pair 的稳定特征 schema,
供 LTR 训练与推理使用。

纪律(per ADR 0001):

- ``FEATURE_NAMES`` 是**锁定**的 schema,顺序与名字都不能改,改了会破坏
  已保存的训练向量。
- 缺失 rank 用 ``MISSING_RANK_SENTINEL = 999.0``,不能默认成 0
  (0 会被 LTR 误读成"排名第一")。
- 布尔/二元特征以 ``0.0`` / ``1.0`` 存储。
- ``gold_in_accepted_corpus`` 一类 oracle 特征**禁止**加入 ``FEATURE_NAMES``,
  它会让 covered 训练样本的分布与 uncovered 推理样本的分布漂移。
- 推理时可用的覆盖率信号只有 ``candidate_in_accepted_corpus``(候选期刊
  在 corpus 中是否有论文);该信号在训练和推理时都可计算,可用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.journals.accepted_paper_store import AcceptedPaperStore
from src.journals.journal_model import Journal
from src.journals.journal_store import JournalStore
from src.papers.paper_model import PaperProfile

# 哨兵值:缺失的检索排名用 999,不能默认成 0。
MISSING_RANK_SENTINEL: float = 999.0

# CCF 等级 → 数值映射(unknown=0)
CCF_LEVEL_TO_NUMERIC: dict = {"A": 3, "B": 2, "C": 1}


def ccf_level_to_numeric(level: Optional[str]) -> float:
    """把 CCF 等级字符串转成 0/1/2/3 数值。"""
    if not level:
        return 0.0
    return float(CCF_LEVEL_TO_NUMERIC.get(str(level).upper(), 0))


# 锁定 schema:20 个特征。顺序就是 feature vector 的列顺序,
# 训练与推理必须严格一致。修改此列表前请读 ADR 0001。
FEATURE_NAMES: List[str] = [
    # 4.1 段原计划的 19 个特征
    "retrieval_rank",
    "rule_rank",
    "rule_score",
    "scope_bm25_rank",
    "scope_vector_rank",
    "typical_bm25_rank",
    "typical_vector_rank",
    "accepted_bm25_rank",
    "accepted_vector_rank",
    "route_count",
    "has_scope_route",
    "has_typical_route",
    "has_accepted_route",
    "has_identity_anchor",
    "same_gold_area",
    "same_parsed_ccf_area",
    "same_ccf_level",
    "journal_ccf_numeric",
    "paper_strength",
    # ADR 0001 新增:候选级别覆盖率信号(可推理,不算 oracle)
    "candidate_in_accepted_corpus",
]
assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), "FEATURE_NAMES 出现重复"
assert "gold_in_accepted_corpus" not in FEATURE_NAMES, "oracle 特征被错误加入"


# trace["routes"] 中的子集,会映射到独立 rank 特征。
# 列表顺序就是 build_features 抽取的字段顺序。
ROUTE_RANK_FIELDS: List[str] = [
    "scope_bm25",
    "scope_vector",
    "scope_text",
    "typical_bm25",
    "typical_vector",
    "typical_text",
    "accepted_bm25",
    "accepted_vector",
    "identity_anchor",
]


@dataclass
class PaperCandidateFeatures:
    """一篇 paper × 一本候选期刊 的结构化特征向量。

    字段顺序与 ``FEATURE_NAMES`` 完全一致,改动请同步修改两边。
    缺失字段用 ``MISSING_RANK_SENTINEL`` 或 ``0.0`` 填充。
    """

    retrieval_rank: float = MISSING_RANK_SENTINEL
    rule_rank: float = MISSING_RANK_SENTINEL
    rule_score: float = 0.0
    scope_bm25_rank: float = MISSING_RANK_SENTINEL
    scope_vector_rank: float = MISSING_RANK_SENTINEL
    typical_bm25_rank: float = MISSING_RANK_SENTINEL
    typical_vector_rank: float = MISSING_RANK_SENTINEL
    accepted_bm25_rank: float = MISSING_RANK_SENTINEL
    accepted_vector_rank: float = MISSING_RANK_SENTINEL
    route_count: float = 0.0
    has_scope_route: float = 0.0
    has_typical_route: float = 0.0
    has_accepted_route: float = 0.0
    has_identity_anchor: float = 0.0
    same_gold_area: float = 0.0
    same_parsed_ccf_area: float = 0.0
    same_ccf_level: float = 0.0
    journal_ccf_numeric: float = 0.0
    paper_strength: float = 0.0
    candidate_in_accepted_corpus: float = 0.0

    def to_vector(self) -> List[float]:
        """按 ``FEATURE_NAMES`` 顺序返回数值向量。"""
        return [float(getattr(self, name)) for name in FEATURE_NAMES]


def _route_rank_or_sentinel(routes: Dict[str, Any], route_name: str) -> float:
    """从 trace["routes"][route_name]["rank"] 抽值;缺失用哨兵 999.0。"""
    entry = routes.get(route_name)
    if not isinstance(entry, dict):
        return MISSING_RANK_SENTINEL
    rank = entry.get("rank")
    if not isinstance(rank, (int, float)) or rank <= 0:
        return MISSING_RANK_SENTINEL
    return float(rank)


def _has_route(routes: Dict[str, Any], prefix: str) -> float:
    """trace 中是否存在以 prefix 开头的 route(0.0/1.0)。"""
    return 1.0 if any(name.startswith(prefix) for name in routes) else 0.0


def build_features(
    paper_profile: PaperProfile,
    journal: Journal,
    trace_entry: Dict[str, Any],
    rule_rank: Optional[int],
    rule_score: float,
    candidate_in_accepted_corpus: bool,
) -> PaperCandidateFeatures:
    """从候选生成器 trace + RuleScorer 结果 + 候选期刊元数据,构建特征向量。

    关键不变量(per ADR 0001):
    - 缺失的 route rank 不会传 0,统一用 ``MISSING_RANK_SENTINEL``。
    - ``candidate_in_accepted_corpus`` 由调用者预计算后传入(避免本函数
      再次访问磁盘);不接受 ``gold_in_accepted_corpus`` 一类 oracle 参数。
    - ``rule_rank=None``(候选未进 RuleScorer Top20)用哨兵 999。
    """
    routes = trace_entry.get("routes", {}) if isinstance(trace_entry, dict) else {}
    routes = routes if isinstance(routes, dict) else {}

    # 逐 route 抽 rank
    rank_by_route = {name: _route_rank_or_sentinel(routes, name) for name in ROUTE_RANK_FIELDS}

    return PaperCandidateFeatures(
        # retrieval_rank 暂用哨兵(由 caller 注入;trace 没有总排名)
        retrieval_rank=MISSING_RANK_SENTINEL,
        # rule_rank:None → 哨兵
        rule_rank=float(rule_rank) if isinstance(rule_rank, (int, float)) and rule_rank > 0 else MISSING_RANK_SENTINEL,
        rule_score=float(rule_score or 0.0),
        scope_bm25_rank=rank_by_route["scope_bm25"],
        scope_vector_rank=rank_by_route["scope_vector"],
        typical_bm25_rank=rank_by_route["typical_bm25"],
        typical_vector_rank=rank_by_route["typical_vector"],
        accepted_bm25_rank=rank_by_route["accepted_bm25"],
        accepted_vector_rank=rank_by_route["accepted_vector"],
        # route_count:有 rank 不为哨兵的 route 个数
        route_count=float(sum(1 for v in rank_by_route.values() if v != MISSING_RANK_SENTINEL)),
        has_scope_route=_has_route(routes, "scope_"),
        has_typical_route=_has_route(routes, "typical_"),
        has_accepted_route=_has_route(routes, "accepted_"),
        # identity_anchor 单独成 route(无前缀匹配)
        has_identity_anchor=1.0 if "identity_anchor" in routes else 0.0,
        # same_gold_area / same_parsed_ccf_area / same_ccf_level
        # 在 4.1.b 阶段先用 0.0 占位,等 4.1.d 接入 trace 详细数据后再实现
        same_gold_area=0.0,
        same_parsed_ccf_area=0.0,
        same_ccf_level=0.0,
        journal_ccf_numeric=ccf_level_to_numeric(getattr(journal, "ccf_rating", None)),
        paper_strength=float(paper_profile.paper_strength) if paper_profile.paper_strength is not None else 0.0,
        candidate_in_accepted_corpus=1.0 if candidate_in_accepted_corpus else 0.0,
    )


def _compute_in_corpus_set(
    accepted_paper_store: Optional[AcceptedPaperStore],
    jids_in_trace: List[str],
) -> set:
    """对 trace 中出现的 jid,计算哪些在 corpus 中(优化:只检查涉及的 jid)。

    接受 store=None(返回空集)与 store 异常(同样返回空集)。
    """
    if accepted_paper_store is None:
        return set()
    in_corpus: set = set()
    try:
        for jid in jids_in_trace:
            if accepted_paper_store.get_papers(jid):
                in_corpus.add(jid)
    except Exception:
        return set()
    return in_corpus


def attach_features_to_trace(
    trace: Dict[str, dict],
    paper_profile: PaperProfile,
    journal_store: JournalStore,
    rule_ranks: Optional[Dict[str, int]],
    rule_scores: Optional[Dict[str, float]],
    accepted_paper_store: Optional[AcceptedPaperStore],
) -> None:
    """把 features dict 注入到 trace 中每本期刊的 entry(原地修改)。

    - trace[jid]["features"]:list[float],长度 == len(FEATURE_NAMES)
    - trace[jid]["feature_names"]:list[str],== FEATURE_NAMES(冗余存,
      方便 LTR 推理时不依赖外部 schema)

    缺失的 journal_id(在 journal_store 中找不到)会被**静默跳过**,
    不抛异常,也不污染该 entry。这是防御性策略,理论不会发生。

    rule_ranks/rule_scores 允许为 None;按 jid 查询时缺失则视为未参与 RuleScorer。
    """
    rule_ranks = rule_ranks or {}
    rule_scores = rule_scores or {}

    jids_in_trace = list(trace.keys())
    in_corpus = _compute_in_corpus_set(accepted_paper_store, jids_in_trace)

    for jid, entry in trace.items():
        journal = journal_store.get_journal(jid)
        if journal is None:
            # 防御:trace 里的 jid 应当都来自 store,见 caller 的 invariants
            continue
        feats = build_features(
            paper_profile=paper_profile,
            journal=journal,
            trace_entry=entry,
            rule_rank=rule_ranks.get(jid),
            rule_score=rule_scores.get(jid, 0.0),
            candidate_in_accepted_corpus=(jid in in_corpus),
        )
        entry["features"] = feats.to_vector()
        entry["feature_names"] = list(FEATURE_NAMES)
