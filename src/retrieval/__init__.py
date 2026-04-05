"""自己手搓的 BM25 + 检索评测指标 —— 零外部依赖，纯标准库。

BM25Lite 是经典 Okapi BM25 实现（k1=1.5, b=0.75），
从 run_phase0_eval_ab.py 抽出来复用的。

评测指标三件套：
  - reciprocal_rank_binary (MRR)：第一个命中在第几名
  - coverage_at_k (Recall@k)：top-k 里命中了几个
  - ndcg_at_k：带排名折扣的标准评测

实验结论：neighbor_prop 是最有效的图信号（1-hop 标签传播），
graph_full 最优配置 nw=1.00, hw=0.15, cw=0（citation walk 负贡献关掉了）。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Set

from src.utils.text_utils import tokenize_for_retrieval


# ── BM25 ──────────────────────────────────────────────────────────────────────

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


# ── Retrieval metrics ─────────────────────────────────────────────────────────

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
