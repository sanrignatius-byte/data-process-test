"""负样本采样 —— 给对比学习挑"干扰项"。

采样器是可插拔的（Protocol 接口），目前有三个实现：
  - HeuristicNegativeSampler：规则采样（random / in_doc_swap）
  - GraphAwareNegativeSampler：图感知采样（目前是 stub，fallback 到 random）
  - build_sampler() 工厂函数：按 config 字典自动选实现

in_doc_swap 是核心策略：优先从同文档里挑负样本（最难的干扰项），
不够再从其他文档补。注意 pos_doc_ids 必须在过滤 pool 之前提取 ——
不然 positive chunks 已被移除就取不到 doc_id 了（这个 bug 坑过我们一次）。
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Protocol, Sequence

from src.models import Chunk


# ── Protocol ──────────────────────────────────────────────────────────────────

class NegativeSampler(Protocol):
    """负样本采样器接口 —— 只要实现 sample() 就行，不用继承 ABC。"""

    def sample(
        self,
        query_text: str,
        positive_ids: Sequence[str],
        candidates: Sequence[Chunk],
        n: int,
    ) -> List[Chunk]:
        """Return up to *n* hard-negative chunks.

        Parameters
        ----------
        query_text : str
            The query text (used by embedding samplers for similarity).
        positive_ids : Sequence[str]
            ``chunk_id`` values of positive evidence — must be excluded.
        candidates : Sequence[Chunk]
            The full candidate pool.
        n : int
            Maximum negatives to return.
        """
        ...


# ── Heuristic sampler ────────────────────────────────────────────────────────

class HeuristicNegativeSampler:
    """基于规则的负样本采样，不需要 embedding，零成本。

    两种策略：
      - random：排除正样本后随便抽
      - in_doc_swap：优先从同文档抽（hard negative），不够从其他文档补

    这里有个历史 bug 教训：doc_id 不能从 element_id 字符串 rsplit 猜，
    arXiv ID 有多个下划线会猜错。必须直接用 Chunk.doc_id 字段！
    """

    def __init__(
        self,
        strategy: str = "random",
        distribution: Dict[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        self.strategy = strategy
        self.distribution = distribution or {}
        self._rng = random.Random(seed)

    # ── public API ────────────────────────────────────────────────────────

    def sample(
        self,
        query_text: str,
        positive_ids: Sequence[str],
        candidates: Sequence[Chunk],
        n: int,
    ) -> List[Chunk]:
        """Sample *n* hard negatives from *candidates*."""
        pos_set = set(positive_ids)
        # Extract doc_ids BEFORE filtering out positives from pool
        pos_doc_ids = {c.doc_id for c in candidates if c.chunk_id in pos_set and c.doc_id}
        pool = [c for c in candidates if c.chunk_id not in pos_set]
        if not pool:
            return []

        if self.strategy == "random":
            return self._random(pool, n)
        if self.strategy == "in_doc_swap":
            return self._in_doc_swap(pool, pos_doc_ids, n)
        # default / modal_mixed → random fallback for now
        return self._random(pool, n)

    # ── private helpers ───────────────────────────────────────────────────

    def _random(self, pool: List[Chunk], n: int) -> List[Chunk]:
        return self._rng.sample(pool, min(n, len(pool)))

    def _in_doc_swap(
        self,
        pool: List[Chunk],
        doc_ids: set[str],
        n: int,
    ) -> List[Chunk]:
        """同文档换元素 —— 最狠的 hard negative 策略。

        优先挑同文档的非正样本元素，不够再从其他文档补。
        补的时候只从 other_doc pool 里抽，避免重复。
        """
        if not doc_ids:
            return self._random(pool, n)

        same_doc = [c for c in pool if c.doc_id in doc_ids]
        if len(same_doc) >= n:
            return self._rng.sample(same_doc, n)
        # pad with chunks from OTHER documents to avoid duplicates
        other_doc = [c for c in pool if c.doc_id not in doc_ids]
        rest = self._rng.sample(other_doc, min(n - len(same_doc), len(other_doc)))
        return same_doc + rest


# ── Graph-aware sampler (stub) ────────────────────────────────────────────────

class GraphAwareNegativeSampler:
    """图感知负样本 —— 理想中用图的 1-hop 邻居做 hard negative。

    目前还是个 stub（TODO），fallback 到 random。
    等 src/graph/ 的 PageRank / label propagation 做好了再接上。
    """

    def __init__(self, seed: int = 42) -> None:
        self._fallback = HeuristicNegativeSampler(strategy="random", seed=seed)

    def sample(
        self,
        query_text: str,
        positive_ids: Sequence[str],
        candidates: Sequence[Chunk],
        n: int,
    ) -> List[Chunk]:
        # TODO: integrate with src.graph once it is populated
        return self._fallback.sample(query_text, positive_ids, candidates, n)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_sampler(config: Dict[str, Any]) -> NegativeSampler:
    """工厂函数 —— 按 config 字典自动选采样器实现。

    config 里写 strategy="in_doc_swap" 就用同文档换元素，
    strategy="graph_aware" 就用图感知（目前 fallback 到 random），
    其他都 fallback 到 random。
    """
    strategy = config.get("strategy", "random")
    seed = config.get("seed", 42)

    if strategy == "graph_aware":
        return GraphAwareNegativeSampler(seed=seed)

    return HeuristicNegativeSampler(
        strategy=strategy,
        distribution=config.get("distribution"),
        seed=seed,
    )
