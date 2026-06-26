"""Feature builder for LTR reranker (Task 4.1).

本模块定义 paper-candidate pair 的版本化特征 schema,
供 LTR 训练与推理使用。

纪律(per ADR 0001):

- ``FEATURE_NAMES`` 是**锁定**的 16 维基础 schema,顺序与名字都不能改,
  改了会破坏已保存的训练向量。
  2026-06-26: 从 19 维降到 16 维,删 4 个 noise/harmful features:
  - ``same_gold_area``: discrim 0.03,noise (paper×gold 重复信号)
  - ``same_parsed_ccf_area``: discrim 0.03,noise (与 same_gold_area 对称)
  - ``candidate_in_accepted_corpus``: 99.8% = 1.0,无信号 (corpus 覆盖率近 100%)
  - ``journal_tier_weight``: coef +0.80,harmful (bias toward A/B journals)
  历史 19/25/27-dim 模型需要 retrain 16/22/23-dim 新模型。
- ``FEATURE_NAMES_WITH_LLM_EVIDENCE`` 是阶段 6.2 的显式 22 维 schema
  (16 base + 6 evidence)。
- ``FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY`` 是 23 维 opt-in schema
  (22 + 1 area_exclusivity),仅在 ``--enable-tier-exclusivity`` 启用。
- 缺失 rank 用 ``MISSING_RANK_SENTINEL = 999.0``,不能默认成 0
  (0 会被 LTR 误读成"排名第一")。
- 布尔/二元特征以 ``0.0`` / ``1.0`` 存储。
- ``gold_in_accepted_corpus`` 一类 oracle 特征**禁止**加入 ``FEATURE_NAMES``,
  它会让 covered 训练样本的分布与 uncovered 推理样本的分布漂移。
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


# 锁定 schema:16 个特征。顺序就是 feature vector 的列顺序,
# 训练与推理必须严格一致。修改此列表前请读 ADR 0001。
# 2026-06-26: 从 19 维降到 16 维 (删 4 noise/harmful features)。
FEATURE_NAMES: List[str] = [
    # 4.1 段原计划的 19 个特征 - 删 4 noise/harmful 后剩 16
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
    "same_ccf_level",
    "journal_ccf_numeric",
    # 删 4 noise/harmful (2026-06-26 23-dim plan):
    # - same_gold_area (discrim 0.03, noise)
    # - same_parsed_ccf_area (discrim 0.03, noise — symmetric)
    # - candidate_in_accepted_corpus (99.8% = 1.0, no signal)
    # - journal_tier_weight (coef +0.80, harmful — biases toward A/B)
]
assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)), "FEATURE_NAMES 出现重复"
assert "gold_in_accepted_corpus" not in FEATURE_NAMES, "oracle 特征被错误加入"

# 阶段 6.2:LLM Evidence 可选特征。默认路径仍使用上面的 16 维 FEATURE_NAMES。
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

# 2026-06-26: v4 26-dim 旧 schema (含 paper_strength),仅供
# learning_to_ranker_balanced_v4_lr.json (archive 26-dim 模型) 的回滚路径。
# 不在生产路径使用;新模型都用 16/22/23-dim。
FEATURE_NAMES_V4_LEGACY: List[str] = [
    "retrieval_rank", "rule_rank", "rule_score",
    "scope_bm25_rank", "scope_vector_rank",
    "typical_bm25_rank", "typical_vector_rank",
    "accepted_bm25_rank", "accepted_vector_rank",
    "route_count", "has_scope_route", "has_typical_route",
    "has_accepted_route", "has_identity_anchor",
    "same_gold_area", "same_parsed_ccf_area", "same_ccf_level",
    "journal_ccf_numeric", "paper_strength", "candidate_in_accepted_corpus",
]
assert len(FEATURE_NAMES_V4_LEGACY) == 20
FEATURE_NAMES_WITH_LLM_EVIDENCE_V4_LEGACY: List[str] = (
    list(FEATURE_NAMES_V4_LEGACY) + list(LLM_EVIDENCE_FEATURE_NAMES)
)
assert len(FEATURE_NAMES_WITH_LLM_EVIDENCE_V4_LEGACY) == 26

# 阶段 6.5 (P2-mini): area 互斥度(2026-06-26 删 journal_tier_weight)。
# 注意:不进 FEATURE_NAMES (locked 16-dim),只用于 opt-in 23 维扩展。
TIER_WEIGHT_BY_CCF: Dict[str, float] = {"A": 0.7, "B": 1.0, "C": 1.5}
TIER_EXCLUSIVITY_FEATURE_NAMES: List[str] = [
    "area_exclusivity",
]
FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY: List[str] = (
    list(FEATURE_NAMES_WITH_LLM_EVIDENCE) + list(TIER_EXCLUSIVITY_FEATURE_NAMES)
)
assert len(FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY) == 23, (
    f"expected 23-dim, got {len(FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY)}"
)
assert len(set(FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY)) == len(
    FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY
), "FEATURE_NAMES_WITH_TIER_AND_EXCLUSIVITY 出现重复"


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

    2026-06-26: 删 4 noise/harmful fields: same_gold_area,
    same_parsed_ccf_area, candidate_in_accepted_corpus, journal_tier_weight。
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
    same_ccf_level: float = 0.0
    journal_ccf_numeric: float = 0.0
    llm_scope_fit: float = 0.5
    llm_method_fit: float = 0.5
    llm_application_fit: float = 0.5
    llm_journal_position_fit: float = 0.5
    llm_too_broad_penalty: float = 0.0
    llm_too_narrow_penalty: float = 0.0
    # 阶段 6.5:23 维扩展 (2026-06-26: 从 27 维降到 23 维,删 journal_tier_weight)
    area_exclusivity: float = 0.0

    def to_vector(self, feature_names: Optional[List[str]] = None) -> List[float]:
        """按显式 schema 返回向量;默认保持现有 16 维 ``FEATURE_NAMES``。"""
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


def _tier_weight_value(ccf_rating: Optional[str]) -> float:
    """CCF 等级 → 反向提权系数 (A=0.7, B=1.0, C=1.5)。

    缺失或未知 → 1.0 (中性)。
    与 ``journal_ccf_numeric`` 单调相反但独立:LTR 可分别拟合
    "绝对偏好" (ccf_numeric) 与 "提权强度" (tier_weight)。
    """
    if not ccf_rating:
        return 1.0
    return TIER_WEIGHT_BY_CCF.get(str(ccf_rating).upper(), 1.0)


def _area_exclusivity_value(
    candidate_subject_tags: Optional[List[str]],
    paper_anchor_area: Optional[str],
    n_matching_in_pool: Optional[int],
) -> float:
    """1 / 同领域候选数;paper_anchor_area 不在 candidate 时 → 0.0。

    设计:
    - paper_anchor_area 取 ``paper_profile.research_area[0]``
      (与 ``journal.subject_tags`` 同命名空间,10/10 已审计 overlap)。
    - 训练/推理都用 paper_profile 字段,无分布漂移。
    - n_matching=None 或 <=0 → 防御性当作 n=1, 返回 1.0
      (paper_anchor 不匹配时仍返回 0.0)。
    """
    if not paper_anchor_area:
        return 0.0
    if paper_anchor_area not in (candidate_subject_tags or []):
        return 0.0
    n = n_matching_in_pool if (n_matching_in_pool and n_matching_in_pool > 0) else 1
    return 1.0 / float(n)


def build_features(
    paper_profile: PaperProfile,
    journal: Journal,
    trace_entry: Dict[str, Any],
    rule_rank: Optional[int],
    rule_score: float,
    candidate_in_accepted_corpus: bool,
    llm_evidence: Optional[Dict[str, Any]] = None,
    paper_anchor_area: Optional[str] = None,
    n_matching_in_pool: Optional[int] = None,
    gold_journal: Optional[Journal] = None,
    paper_ccf_target_level: Optional[str] = None,
) -> PaperCandidateFeatures:
    """从候选生成器 trace + RuleScorer 结果 + 候选期刊元数据,构建特征向量。

    关键不变量(per ADR 0001):
    - 缺失的 route rank 不会传 0,统一用 ``MISSING_RANK_SENTINEL``。
    - ``candidate_in_accepted_corpus`` 由调用者预计算后传入(2026-06-26:
      此参数仍接受但内部不再写入 features — corpus 覆盖率近 100%,
      99.8% = 1.0,无信号,feature 被删除)。
    - ``rule_rank=None``(候选未进 RuleScorer Top20)用哨兵 999。

    2026-06-26: 删 4 noise/harmful features。``same_ccf_level`` 仍用
    ``paper_ccf_target_level`` vs ``journal.ccf_rating``。
    ``gold_journal`` 参数保留向后兼容(已不再写入 features)。
    """
    routes = trace_entry.get("routes", {}) if isinstance(trace_entry, dict) else {}
    routes = routes if isinstance(routes, dict) else {}

    # 逐 route 抽 rank
    rank_by_route = {name: _route_rank_or_sentinel(routes, name) for name in ROUTE_RANK_FIELDS}

    # 2026-06-26: same_ccf_level 保留;same_gold_area / same_parsed_ccf_area 已删除。
    _same_ccf_level = 1.0 if (
        paper_ccf_target_level is not None
        and getattr(journal, "ccf_rating", None) is not None
        and str(paper_ccf_target_level).upper() == str(journal.ccf_rating).upper()
    ) else 0.0

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
        has_identity_anchor=1.0 if "identity_anchor" in routes else 0.0,
        # same_ccf_level (2026-06-26: same_gold_area / same_parsed_ccf_area 已删除)
        same_ccf_level=_same_ccf_level,
        journal_ccf_numeric=ccf_level_to_numeric(getattr(journal, "ccf_rating", None)),
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
        # 阶段 6.5:area 互斥度 (2026-06-26: 删 journal_tier_weight)
        area_exclusivity=_area_exclusivity_value(
            candidate_subject_tags=getattr(journal, "subject_tags", None),
            paper_anchor_area=paper_anchor_area,
            n_matching_in_pool=n_matching_in_pool,
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
    paper_anchor_area: Optional[str] = None,
    n_matching_in_pool: Optional[int] = None,
    gold_journal: Optional[Journal] = None,
    paper_ccf_target_level: Optional[str] = None,
) -> None:
    """把 features dict 注入到 trace 中每本期刊的 entry(原地修改)。

    - 默认 trace[jid]["features"] 长度 == len(FEATURE_NAMES)。
    - 显式传入 ``FEATURE_NAMES_WITH_LLM_EVIDENCE`` 时输出 25 维 evidence schema。
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
            paper_anchor_area=paper_anchor_area,
            n_matching_in_pool=n_matching_in_pool,
            gold_journal=gold_journal,
            paper_ccf_target_level=paper_ccf_target_level,
        )
        entry["features"] = feats.to_vector(selected_feature_names)
        entry["feature_names"] = list(selected_feature_names)
