"""Feature builder for LTR reranker (Task 4.1).

本模块定义 paper-candidate pair 的版本化特征 schema,
供 LTR 训练与推理使用。

纪律(per ADR 0001):

- ``FEATURE_NAMES`` 是**锁定**的 20 维基础 schema,顺序与名字都不能改,
  改了会破坏已保存的训练向量。
- ``FEATURE_NAMES_WITH_LLM_EVIDENCE`` 是阶段 6.2 的显式 26 维 schema;
  只有 evidence 实验和对应新模型可以消费。
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

# 阶段 6.2:LLM Evidence 可选特征。默认路径仍使用上面的 20 维 FEATURE_NAMES。
LLM_EVIDENCE_FEATURE_NAMES: List[str] = [
    "llm_scope_fit",
    "llm_method_fit",
    "llm_application_fit",
    "llm_journal_position_fit",
    "llm_too_broad_penalty",
    "llm_too_narrow_penalty",
]
FEATURE_NAMES_WITH_LLM_EVIDENCE: List[str] = (
    list(FEATURE_NAMES) + list(LLM_EVIDENCE_FEATURE_NAMES)
)
assert len(FEATURE_NAMES_WITH_LLM_EVIDENCE) == len(
    set(FEATURE_NAMES_WITH_LLM_EVIDENCE)
), "FEATURE_NAMES_WITH_LLM_EVIDENCE 出现重复"


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

    基础字段顺序与 ``FEATURE_NAMES`` 完全一致,evidence 字段顺序与
    ``LLM_EVIDENCE_FEATURE_NAMES`` 完全一致。
    缺失字段用 ``MISSING_RANK_SENTINEL`` 或 ``0.0`` 填充。
    缺失 LLM fit evidence 使用中性值 0.5,penalty 使用 0.0。
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
    llm_scope_fit: float = 0.5
    llm_method_fit: float = 0.5
    llm_application_fit: float = 0.5
    llm_journal_position_fit: float = 0.5
    llm_too_broad_penalty: float = 0.0
    llm_too_narrow_penalty: float = 0.0

    def to_vector(self, feature_names: Optional[List[str]] = None) -> List[float]:
        """按显式 schema 返回向量;默认保持现有 20 维 ``FEATURE_NAMES``。"""
        selected_names = FEATURE_NAMES if feature_names is None else feature_names
        return [float(getattr(self, name)) for name in selected_names]


def _route_rank_or_sentinel(routes: Dict[str, Any], route_name: str) -> float:
    """从 trace["routes"][route_name]["rank"] 抽值;缺失用哨兵 999.0。"""
    entry = routes.get(route_name)
    if not isinstance(entry, dict):
        return MISSING_RANK_SENTINEL
    rank = entry.get("rank")
    if not isinstance(rank, (int, float)) or rank <= 0:
        return MISSING_RANK_SENTINEL
    return float(rank)


def _trace_top_level_rank(trace_entry: Dict[str, Any]) -> float:
    """从 trace 顶层抽 retrieval_rank;缺失用哨兵 999.0。

    retrieval_rank 是 CandidateGenerator._merge_route_results 写入的"在
    top_k 候选列表中的位置",是 LTR 最重要的排序特征之一(per plan 4.1)。
    """
    if not isinstance(trace_entry, dict):
        return MISSING_RANK_SENTINEL
    rank = trace_entry.get("retrieval_rank")
    if not isinstance(rank, (int, float)) or rank <= 0:
        return MISSING_RANK_SENTINEL
    return float(rank)


def _has_route(routes: Dict[str, Any], prefix: str) -> float:
    """trace 中是否存在以 prefix 开头的 route(0.0/1.0)。"""
    return 1.0 if any(name.startswith(prefix) for name in routes) else 0.0


def _llm_evidence_score(
    llm_evidence: Optional[Dict[str, Any]],
    field: str,
    default: float,
) -> float:
    """抽取合法 [0,1] evidence 分数;缺失或非法时返回中性默认值。"""
    if not isinstance(llm_evidence, dict):
        return default
    value = llm_evidence.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    if not 0 <= value <= 1:
        return default
    return float(value)


def build_features(
    paper_profile: PaperProfile,
    journal: Journal,
    trace_entry: Dict[str, Any],
    rule_rank: Optional[int],
    rule_score: float,
    candidate_in_accepted_corpus: bool,
    llm_evidence: Optional[Dict[str, Any]] = None,
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
        # retrieval_rank 从 trace 顶层读(CandidateGenerator._merge_route_results 写入)
        # trace 缺该字段时退化为哨兵 999(防御性,理论上不会发生)
        retrieval_rank=_trace_top_level_rank(trace_entry),
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
        #
        # Note (2026-06-18, audit finding #6): this binary feature is the
        # THIRD of three places identity_anchor is consumed. See the
        # cross-reference comment in src/ranker/rule_scorer.py:161 for the
        # full mapping (candidate_generator / rule_scorer / feature_builder).
        has_identity_anchor=1.0 if "identity_anchor" in routes else 0.0,
        # same_gold_area / same_parsed_ccf_area / same_ccf_level
        # 在 4.1.b 阶段先用 0.0 占位,等 4.1.d 接入 trace 详细数据后再实现
        same_gold_area=0.0,
        same_parsed_ccf_area=0.0,
        same_ccf_level=0.0,
        journal_ccf_numeric=ccf_level_to_numeric(getattr(journal, "ccf_rating", None)),
        paper_strength=float(paper_profile.paper_strength) if paper_profile.paper_strength is not None else 0.0,
        candidate_in_accepted_corpus=1.0 if candidate_in_accepted_corpus else 0.0,
        llm_scope_fit=_llm_evidence_score(llm_evidence, "scope_fit", 0.5),
        llm_method_fit=_llm_evidence_score(llm_evidence, "method_fit", 0.5),
        llm_application_fit=_llm_evidence_score(llm_evidence, "application_fit", 0.5),
        llm_journal_position_fit=_llm_evidence_score(
            llm_evidence, "journal_position_fit", 0.5
        ),
        llm_too_broad_penalty=_llm_evidence_score(
            llm_evidence, "too_broad_penalty", 0.0
        ),
        llm_too_narrow_penalty=_llm_evidence_score(
            llm_evidence, "too_narrow_penalty", 0.0
        ),
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
    llm_evidence_by_journal: Optional[Dict[str, Dict[str, Any]]] = None,
    feature_names: Optional[List[str]] = None,
) -> None:
    """把 features dict 注入到 trace 中每本期刊的 entry(原地修改)。

    - 默认 trace[jid]["features"] 长度 == len(FEATURE_NAMES)。
    - 显式传入 ``FEATURE_NAMES_WITH_LLM_EVIDENCE`` 时输出 26 维 evidence schema。
    - trace[jid]["feature_names"] 冗余保存实际 schema,方便 LTR 推理时校验。

    缺失的 journal_id(在 journal_store 中找不到)会被**静默跳过**,
    不抛异常,也不污染该 entry。这是防御性策略,理论不会发生。

    rule_ranks/rule_scores 允许为 None;按 jid 查询时缺失则视为未参与 RuleScorer。
    """
    rule_ranks = rule_ranks or {}
    rule_scores = rule_scores or {}
    llm_evidence_by_journal = llm_evidence_by_journal or {}
    selected_feature_names = FEATURE_NAMES if feature_names is None else feature_names

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
            llm_evidence=llm_evidence_by_journal.get(jid),
        )
        entry["features"] = feats.to_vector(selected_feature_names)
        entry["feature_names"] = list(selected_feature_names)
