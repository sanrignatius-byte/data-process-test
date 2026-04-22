"""检索评测指标三件套 —— Phase 4 从 src/retrieval/__init__.py 拆出。

- :func:`reciprocal_rank_binary`  → MRR
- :func:`coverage_at_k`            → Recall@k
- :func:`ndcg_at_k`                → 带排名折扣的二值 NDCG

实验结论（2026-03-16 Phase0 eval v3，保留作历史参考）：neighbor_prop 是最有效
的图信号（1-hop 标签传播），graph_full 最优配置 nw=1.00, hw=0.15, cw=0
（citation walk 负贡献关掉了）。
"""

from __future__ import annotations

import math
from typing import List, Set


def reciprocal_rank_binary(ranked_ids: List[str], relevant: Set[str]) -> float:
    """MRR：第一个相关结果排在第几名？排名越靠前分越高（1/rank），没命中就是 0。"""
    for i, rid in enumerate(ranked_ids, 1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def coverage_at_k(ranked_ids: List[str], relevant: Set[str], k: int = 10) -> float:
    """Recall@k：top-k 里找到了多少比例的相关文档。"""
    if not relevant:
        return 0.0
    found = sum(1 for rid in ranked_ids[:k] if rid in relevant)
    return found / len(relevant)


def ndcg_at_k(ranked_ids: List[str], relevant: Set[str], k: int = 10) -> float:
    """NDCG@k（二值相关性版）：考虑排名位置折扣的评测指标。"""
    dcg = 0.0
    for i, rid in enumerate(ranked_ids[:k], 1):
        if rid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0


__all__ = ["reciprocal_rank_binary", "coverage_at_k", "ndcg_at_k"]
