"""Pluggable negative-sampling strategies for contrastive learning.

Strategies
----------
- ``heuristic``   – rule-based (in_doc_swap, same_type_hard, random)
- ``embedding``   – FAISS / cosine nearest-neighbour hard negatives
- ``graph_aware`` – 1-hop graph neighbours that are *not* positive evidence

All samplers implement the :class:`NegativeSampler` protocol so they can be
composed and configured from ``configs/config.yaml``.
"""

from src.sampling.negative_sampler import (
    GraphAwareNegativeSampler,
    HeuristicNegativeSampler,
    NegativeSampler,
    build_sampler,
)

__all__ = [
    "NegativeSampler",
    "HeuristicNegativeSampler",
    "GraphAwareNegativeSampler",
    "build_sampler",
]
