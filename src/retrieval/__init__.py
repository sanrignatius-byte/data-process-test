"""Lightweight BM25 implementation + retrieval metrics.

Extracted from run_phase0_eval_ab.py to allow reuse across all eval scripts.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Set

from src.utils.text_utils import tokenize_for_retrieval


# ── BM25 ──────────────────────────────────────────────────────────────────────

class BM25Lite:
    """Lightweight BM25 scorer — zero external dependencies."""

    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.N = len(docs)
        self.doc_lens = [len(d) for d in docs]
        self.avgdl = sum(self.doc_lens) / max(1, self.N)
        self.df: Dict[str, int] = defaultdict(int)
        self.tf_docs: List[Counter] = []
        for d in docs:
            tf = Counter(d)
            self.tf_docs.append(tf)
            for t in tf.keys():
                self.df[t] += 1

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        tf = self.tf_docs[doc_idx]
        dl = self.doc_lens[doc_idx]
        s = 0.0
        for t in set(query_tokens):
            f = tf.get(t, 0)
            if f <= 0:
                continue
            _idf = self.idf(t)
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-6))
            s += _idf * (f * (self.k1 + 1)) / max(denom, 1e-6)
        return s


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def reciprocal_rank_binary(ranked_ids: List[str], relevant: Set[str]) -> float:
    """Return reciprocal rank (1/rank of first relevant hit, or 0)."""
    for i, rid in enumerate(ranked_ids, 1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def coverage_at_k(ranked_ids: List[str], relevant: Set[str], k: int = 10) -> float:
    """Fraction of *relevant* items found in top-k."""
    if not relevant:
        return 0.0
    found = sum(1 for rid in ranked_ids[:k] if rid in relevant)
    return found / len(relevant)


def ndcg_at_k(ranked_ids: List[str], relevant: Set[str], k: int = 10) -> float:
    """Binary-relevance NDCG@k."""
    dcg = 0.0
    for i, rid in enumerate(ranked_ids[:k], 1):
        if rid in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0
