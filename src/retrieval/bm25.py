"""BM25 打分器 —— Phase 4 从 src/retrieval/__init__.py 拆出。

BM25Lite 是经典 Okapi BM25 实现（k1=1.5, b=0.75），零外部依赖，纯标准库。
主要消费方是评测脚本（eval_bm25_retrieval.py / run_phase0_eval_ab.py 等），
历史上这些脚本各自实现过 BM25，后被统一到这一份。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List


class BM25Lite:
    """轻量 BM25 打分器 —— 不装 rank_bm25，自己写的才放心。

    建索引时预计算 df 和 tf，query 时 O(|query_terms|) 打分。
    用的标准 Okapi BM25 公式，k1=1.5, b=0.75。
    """

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


__all__ = ["BM25Lite"]
