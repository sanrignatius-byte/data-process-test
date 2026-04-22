"""检索层入口 —— 实现拆到 :mod:`src.retrieval.bm25` 和 :mod:`src.retrieval.metrics`。

这个包的 __init__ 只做 re-export，让 ``from src.retrieval import BM25Lite`` 这种
既有写法继续工作，同时支持新的精确导入。
"""

from src.retrieval.bm25 import BM25Lite
from src.retrieval.metrics import (
    coverage_at_k,
    ndcg_at_k,
    reciprocal_rank_binary,
)

__all__ = [
    "BM25Lite",
    "coverage_at_k",
    "ndcg_at_k",
    "reciprocal_rank_binary",
]
