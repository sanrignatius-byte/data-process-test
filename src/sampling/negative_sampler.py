"""Pluggable negative-sampling strategies.

Each sampler follows the :class:`NegativeSampler` protocol and can be
instantiated via the factory function :func:`build_sampler`.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Protocol, Sequence

from src.models import Chunk


# ── Protocol ──────────────────────────────────────────────────────────────────

class NegativeSampler(Protocol):
    """Interface every negative sampler must satisfy."""

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
    """Rule-based negative sampling (in_doc_swap / same_type / random).

    Parameters
    ----------
    strategy : str
        One of ``"random"``, ``"in_doc_swap"``, ``"same_type_hard"``,
        ``"modal_mixed"`` (composite with ``distribution``).
    distribution : dict | None
        Required when *strategy* is ``"modal_mixed"``.  Maps sub-strategy
        names to floats summing to 1.0.
    seed : int
        RNG seed for reproducibility.
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
        pool = [c for c in candidates if c.chunk_id not in pos_set]
        if not pool:
            return []

        if self.strategy == "random":
            return self._random(pool, n)
        if self.strategy == "in_doc_swap":
            return self._in_doc_swap(pool, positive_ids, n)
        # default / modal_mixed → random fallback for now
        return self._random(pool, n)

    # ── private helpers ───────────────────────────────────────────────────

    def _random(self, pool: List[Chunk], n: int) -> List[Chunk]:
        return self._rng.sample(pool, min(n, len(pool)))

    def _in_doc_swap(
        self,
        pool: List[Chunk],
        positive_ids: Sequence[str],
        n: int,
    ) -> List[Chunk]:
        """Prefer chunks from the same document(s) as positives."""
        if not positive_ids:
            return self._random(pool, n)

        # Use Chunk.doc_id directly — no need to parse element_id strings.
        # Collect doc_ids from the pool chunks whose chunk_id matches a positive.
        pos_set = set(positive_ids)
        doc_ids: set[str] = set()
        for c in pool:
            if c.chunk_id in pos_set and c.doc_id:
                doc_ids.add(c.doc_id)
        # Fallback: also scan all candidates (positives were already excluded
        # from pool, so check the broader candidate set if needed).
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
    """1-hop graph neighbours that are NOT positive evidence.

    .. note:: This is a stub.  Full implementation requires access to the
       document graph (``src/graph/`` once populated).  For now it falls
       back to random sampling.
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
    """Instantiate a sampler from a config dict (e.g. ``config.yaml`` section).

    Expected keys
    -------------
    strategy : str
        ``"random"`` | ``"in_doc_swap"`` | ``"same_type_hard"``
        | ``"modal_mixed"`` | ``"graph_aware"``
    distribution : dict, optional
    seed : int, optional
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
