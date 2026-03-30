#!/usr/bin/env python3
"""
Phase-0 A/B retrieval experiment runner.

Implements the locked protocol:
- Fixed query pool from v4.4 pass + v3 pass (deduped union)
- Ground-truth from required_evidence_spans
- Localization hit if char overlap >= threshold (default 0.5)
- Methods:
    bm25                : BM25Lite baseline
    dense               : TF-IDF cosine baseline
    graph_hub_rerank    : BM25 + static hub prior boost (alpha-blending)
    graph_neighbor_prop : BM25 + dynamic 1-hop intra-element neighbor propagation
    graph_citation_walk : BM25 + cross-doc citation relevance propagation
    graph_full          : BM25 + hub_boost + neighbor_prop + citation_walk (combined)

Outputs a JSON report with Recall@10 / MRR and go/no-go decision gates.

Graph signal design (搜广推 perspective):
  hub_boost       : static prior — element appears in bridge-hub candidate pair
  neighbor_prop   : dynamic label propagation — top-scoring elements propagate score
                    to intra-doc neighbors (co-referenced / hub-pair adjacency)
  citation_walk   : cross-doc authority propagation — high-scoring doc boosts
                    elements in docs it cites / is cited by (1-hop citation walk)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


# 轻量 BM25：不依赖外部库，直接 tf+idf 打分，够用且零依赖
class BM25Lite:
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
            idf = self.idf(t)
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-6))
            s += idf * (f * (self.k1 + 1)) / max(denom, 1e-6)
        return s


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    caption: str = ""
    content: str = ""
    context: str = ""
    enriched_title: str = ""
    enriched_content: str = ""


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# 按 query_id 去重，合并多文件来源的 query pool
def dedupe_queries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        qid = r.get("query_id")
        q = (r.get("query") or "").strip()
        key = qid if qid else f"q::{hash(q)}"
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float, bool)):
        return str(v)
    return ""


# 每个元素 = 一个 chunk：拼接 caption+content+context+enriched，限 1800 字符
def build_chunks(elements_json: Path, max_chars: int = 1800) -> List[Chunk]:
    data = json.loads(elements_json.read_text(encoding="utf-8"))
    docs = data.get("documents", {}) or {}
    chunks: List[Chunk] = []
    for doc_id, d in docs.items():
        elements = d.get("elements", {}) or {}
        for element_id, e in elements.items():
            caption = _to_text(e.get("caption"))
            content = _to_text(e.get("content"))
            context_before = _to_text(e.get("context_before"))
            context_after = _to_text(e.get("context_after"))
            context = " ".join([x for x in [context_before, context_after] if x]).strip()
            enriched_title = _to_text(e.get("enriched_title"))
            enriched_content = _to_text(e.get("enriched_content"))
            fields = [caption, content, context, enriched_title, enriched_content]
            text = "\n".join([x for x in fields if x]).strip()
            if not text:
                continue
            if len(text) > max_chars:
                text = text[:max_chars]
            chunks.append(
                Chunk(
                    chunk_id=element_id,
                    doc_id=str(doc_id),
                    text=text,
                    caption=caption,
                    content=content,
                    context=context,
                    enriched_title=enriched_title,
                    enriched_content=enriched_content,
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# Prior / adjacency loaders
# ---------------------------------------------------------------------------

# 文档级 hub prior：取该文档下最高 hub_score，归一化到 [0,1]
def load_doc_hub_prior(hubs_json: Path) -> Dict[str, float]:
    obj = json.loads(hubs_json.read_text(encoding="utf-8"))
    hubs = obj.get("hubs", []) or []
    by_doc: Dict[str, float] = defaultdict(float)
    for h in hubs:
        doc = str(h.get("doc_id", ""))
        hs = float(h.get("hub_score", 0.0) or 0.0)
        if doc:
            by_doc[doc] = max(by_doc[doc], hs)

    if not by_doc:
        return {}
    mx = max(by_doc.values())
    if mx <= 0:
        return {k: 0.0 for k in by_doc}
    return {k: v / mx for k, v in by_doc.items()}


# 元素级 hub prior：除 candidate pairs 外还纳入 adjacent_bridge_elements
# 扩大覆盖面，否则大部分 query 的 evidence 落不到 hub 邻域里
def load_element_hub_prior(
    hub_candidates_json: Path,
    hubs_json: Path | None = None,
) -> Dict[str, float]:
    """Element-level hub prior from enriched candidate pairs + adjacent bridges.

    Each element that appears in any hub candidate pair gets a score proportional
    to its max quality_score across all pairs it participates in.

    Additionally, elements from adjacent_bridge_elements (MinerU-mapped IDs
    produced by enrich_hub_candidates.py) are included to expand hub coverage.
    """
    obj = json.loads(hub_candidates_json.read_text(encoding="utf-8"))
    pairs = obj.get("pairs", []) or []
    by_element: Dict[str, float] = {}
    for p in pairs:
        for key in ("element_a_id", "element_b_id"):
            eid = str(p.get(key, "") or "").strip()
            if eid:
                qs = float(p.get("quality_score", 1.0) or 1.0)
                by_element[eid] = max(by_element.get(eid, 0.0), qs)

    # Expand with adjacent_bridge_elements (MinerU IDs, from enrich_hub_candidates.py)
    adj_elements = obj.get("adjacent_bridge_elements", {})
    for eid, score in adj_elements.items():
        eid = str(eid).strip()
        if eid and eid not in by_element:
            by_element[eid] = float(score)

    if not by_element:
        return {}
    mx = max(by_element.values())
    if mx <= 0:
        return {k: 0.0 for k in by_element}
    return {k: v / mx for k, v in by_element.items()}


# 元素邻接图 + 边权重：hub candidate pairs + adjacent_bridge 双源合并
def load_element_adjacency(
    hub_candidates_json: Path,
    hubs_json: Path | None = None,
) -> tuple[Dict[str, Set[str]], Dict[str, Dict[str, float]]]:
    """Build element→neighbor element adjacency from hub candidate pairs + adjacent bridges.

    Uses adjacent_bridge_adjacency (MinerU IDs) from enriched candidates for
    expanded neighbor propagation coverage.

    Returns:
        adj: element_id → set of neighbor element_ids
        adj_weights: element_id → {neighbor_id: weight} (higher = stronger connection)
    """
    obj = json.loads(hub_candidates_json.read_text(encoding="utf-8"))
    pairs = obj.get("pairs", []) or []
    adj: Dict[str, Set[str]] = defaultdict(set)
    adj_weights: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for p in pairs:
        a = str(p.get("element_a_id", "") or "").strip()
        b = str(p.get("element_b_id", "") or "").strip()
        if a and b:
            adj[a].add(b)
            adj[b].add(a)
            # Weight by quality_score: higher quality pairs = stronger edges
            qs = float(p.get("quality_score", 1.0) or 1.0)
            adj_weights[a][b] = max(adj_weights[a][b], qs)
            adj_weights[b][a] = max(adj_weights[b][a], qs)

    # Expand adjacency with adjacent_bridge_adjacency (MinerU IDs)
    for ab in (obj.get("adjacent_bridge_adjacency") or []):
        eids_i = [str(e).strip() for e in (ab.get("elements_i") or []) if str(e).strip()]
        eids_j = [str(e).strip() for e in (ab.get("elements_j") or []) if str(e).strip()]
        for ei in eids_i:
            for ej in eids_j:
                adj[ei].add(ej)
                adj[ej].add(ei)
                # Adjacent bridges get a moderate default weight
                adj_weights[ei][ej] = max(adj_weights[ei][ej], 0.6)
                adj_weights[ej][ei] = max(adj_weights[ej][ei], 0.6)

    # Normalize weights to [0, 1] per element
    for eid in adj_weights:
        inner = adj_weights[eid]
        mx = max(inner.values()) if inner else 1.0
        if mx > 0:
            adj_weights[eid] = {k: v / mx for k, v in inner.items()}

    return dict(adj), {k: dict(v) for k, v in adj_weights.items()}


# 文档间引用邻接：doc→{cites, cited_by}，供 citation_walk 1-hop 传播
def load_citation_adjacency(citation_graph_json: Path) -> Dict[str, Dict[str, Set[str]]]:
    """Build doc→{cites, cited_by} adjacency from citation_graph.json.

    Used for cross-doc citation walk: a high-scoring document propagates its
    relevance to documents it cites and documents that cite it (1-hop walk).

    Returns: doc_id → {"cites": set, "cited_by": set}
    """
    obj = json.loads(citation_graph_json.read_text(encoding="utf-8"))
    raw_adj = obj.get("adjacency", {}) or {}
    result: Dict[str, Dict[str, Set[str]]] = {}
    for doc_id, info in raw_adj.items():
        result[str(doc_id)] = {
            "cites": set(str(x) for x in (info.get("cites") or [])),
            "cited_by": set(str(x) for x in (info.get("cited_by") or [])),
        }
    return result


# ---------------------------------------------------------------------------
# Hit evaluation helpers
# ---------------------------------------------------------------------------

# 证据定位命中判定：精确子串匹配得 1.0，否则用最长公共子串比例
def span_overlap(span: str, text: str) -> float:
    span = (span or "").strip()
    text = (text or "").strip()
    if not span or not text:
        return 0.0
    if span in text:
        return 1.0
    m = SequenceMatcher(None, span, text).find_longest_match(0, len(span), 0, len(text))
    return m.size / max(1, len(span))


def query_spans(q: Dict[str, Any]) -> List[str]:
    spans = []
    for s in (q.get("required_evidence_spans") or []):
        st = (s.get("span") if isinstance(s, dict) else None) or ""
        st = st.strip()
        if st:
            spans.append(st)
    return spans


def query_element_ids(q: Dict[str, Any]) -> List[str]:
    """Extract ground-truth element_ids from required_evidence_spans."""
    ids = []
    for s in (q.get("required_evidence_spans") or []):
        eid = (s.get("element_id") if isinstance(s, dict) else None) or ""
        eid = eid.strip()
        if eid:
            ids.append(eid)
    return ids


def query_element_gains(q: Dict[str, Any]) -> Dict[str, float]:
    gains: Dict[str, float] = {}
    for s in (q.get("required_evidence_spans") or []):
        if not isinstance(s, dict):
            continue
        eid = str(s.get("element_id") or "").strip()
        if not eid:
            continue
        et = str(s.get("evidence_type") or "").strip().lower()
        g = 2.0 if et in {"observation", "result"} else 1.0
        gains[eid] = max(gains.get(eid, 0.0), g)
    return gains


def reciprocal_rank_binary(hit_ranks: List[int]) -> float:
    if not hit_ranks:
        return 0.0
    return 1.0 / min(hit_ranks)


def coverage_at_k(retrieved_ids: List[str], gt_ids: Set[str], k: int) -> float:
    if not gt_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & gt_ids) / len(gt_ids)


def ndcg_at_k(retrieved_ids: List[str], gt_gains: Dict[str, float], k: int) -> float:
    def _dcg(ids: List[str]) -> float:
        s = 0.0
        for rank, eid in enumerate(ids[:k], start=1):
            g = gt_gains.get(eid, 0.0)
            if g > 0:
                s += g / math.log2(rank + 1)
        return s

    if not gt_gains:
        return 0.0
    ideal = [eid for eid, _ in sorted(gt_gains.items(), key=lambda kv: kv[1], reverse=True)]
    idcg = _dcg(ideal)
    if idcg <= 1e-9:
        return 0.0
    return _dcg(retrieved_ids) / idcg


# ---------------------------------------------------------------------------
# Core scoring: BM25 → normalized scores
# ---------------------------------------------------------------------------

# min-max 归一化到 [0,1]，graph 信号与 BM25 在同一尺度上混合
def _bm25_norm_scores(
    bm25: BM25Lite,
    q_toks: List[str],
    n_chunks: int,
) -> List[float]:
    """Return BM25 scores min-max normalized to [0, 1]."""
    raw = [bm25.score(q_toks, i) for i in range(n_chunks)]
    lo = min(raw)
    hi = max(raw)
    rng = hi - lo
    if rng < 1e-9:
        return [0.0] * n_chunks
    return [(x - lo) / rng for x in raw]


# ---------------------------------------------------------------------------
# evaluate_method — all six methods in one function
# ---------------------------------------------------------------------------

# 六种检索方法统一评估：BM25 / dense / hub_rerank /
# neighbor_prop / citation_walk / graph_full
# graph 信号设计：静态 prior → 标签传播 → authority walk
def evaluate_method(
    method: str,
    queries: List[Dict[str, Any]],
    chunks: List[Chunk],
    bm25: BM25Lite,
    doc_hub_prior: Dict[str, float],
    element_hub_prior: Dict[str, float],
    element_adjacency: Dict[str, Set[str]],
    citation_adjacency: Dict[str, Dict[str, Set[str]]],
    # index: chunk_id → index in chunks list
    chunk_id_to_idx: Dict[str, int],
    # index: doc_id → list of chunk indices
    doc_to_chunk_idxs: Dict[str, List[int]],
    dense_matrix,
    vectorizer,
    chunk_tokens: List[List[str]],
    field_tokens: Dict[str, List[List[str]]],
    collection_tf: Dict[str, int],
    total_collection_terms: int,
    top_k: int,
    overlap_threshold: float,
    graph_alpha: float,
    graph_rerank_topn: int,
    neighbor_decay: float,
    citation_decay: float,
    # Decoupled component weights for graph_full
    hub_weight: float | None = None,
    nprop_weight: float | None = None,
    cite_weight: float | None = None,
    neighbor_hops: int = 1,
    # Edge weights for weighted propagation
    element_adj_weights: Dict[str, Dict[str, float]] | None = None,
) -> Dict[str, Any]:
    recall_sums = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0, 20: 0.0}
    mrr = 0.0
    coverage10 = 0.0
    ndcg10 = 0.0
    per_query = []
    hub_bm25_overlap_rates: List[float] = []

    n_chunks = len(chunks)

    for q in queries:
        qtxt = (q.get("query") or "").strip()
        spans = query_spans(q)
        if not qtxt or not spans:
            continue

        # ------------------------------------------------------------------
        # 1. BM25 baseline
        # ------------------------------------------------------------------
        if method == "bm25":
            q_toks = tokenize(qtxt)
            scored = [(i, bm25.score(q_toks, i)) for i in range(n_chunks)]

        # ------------------------------------------------------------------
        # 2. Dense (TF-IDF cosine) baseline
        # ------------------------------------------------------------------
        elif method == "dense":
            q_toks = tokenize(qtxt)
            q_tf = Counter(q_toks)
            q_weights: Dict[str, float] = {}
            for t, tf in q_tf.items():
                if t in bm25.df:
                    q_weights[t] = (1.0 + math.log(tf)) * bm25.idf(t)
            q_norm = math.sqrt(sum(v * v for v in q_weights.values())) or 1.0

            scored = []
            for i in range(n_chunks):
                tf = Counter(chunk_tokens[i])
                d_weights: Dict[str, float] = {}
                for t, f in tf.items():
                    d_weights[t] = (1.0 + math.log(f)) * bm25.idf(t)
                d_norm = math.sqrt(sum(v * v for v in d_weights.values())) or 1.0
                dot = 0.0
                for t, qv in q_weights.items():
                    dot += qv * d_weights.get(t, 0.0)
                scored.append((i, dot / (q_norm * d_norm)))

        # ------------------------------------------------------------------
        # 2b. BM25F: field-weighted BM25
        # ------------------------------------------------------------------
        elif method == "bm25f":
            q_toks = tokenize(qtxt)
            qset = set(q_toks)
            fweights = {
                "caption": 3.0,
                "enriched_content": 2.0,
                "context": 0.5,
                "content": 1.0,
                "enriched_title": 2.0,
            }
            field_avgdl = {
                k: sum(len(v) for v in vals) / max(1, len(vals))
                for k, vals in field_tokens.items()
            }
            scored = []
            for i in range(n_chunks):
                s = 0.0
                for fname, toks_by_chunk in field_tokens.items():
                    tf = Counter(toks_by_chunk[i])
                    dl = len(toks_by_chunk[i])
                    avgdl = field_avgdl.get(fname, 1.0)
                    fs = 0.0
                    for t in qset:
                        f = tf.get(t, 0)
                        if f <= 0:
                            continue
                        idf = bm25.idf(t)
                        denom = f + 1.2 * (1 - 0.75 + 0.75 * dl / max(avgdl, 1e-6))
                        fs += idf * (f * 2.2) / max(denom, 1e-6)
                    s += fweights.get(fname, 1.0) * fs
                scored.append((i, s))

        # ------------------------------------------------------------------
        # 2c. LM Dirichlet smoothing
        # ------------------------------------------------------------------
        elif method == "lm_dirichlet":
            q_toks = tokenize(qtxt)
            mu = 2000.0
            scored = []
            for i in range(n_chunks):
                tf = Counter(chunk_tokens[i])
                dl = len(chunk_tokens[i])
                s = 0.0
                for t in q_toks:
                    p_bg = collection_tf.get(t, 0) / max(1, total_collection_terms)
                    p_bg = max(p_bg, 1e-12)
                    s += math.log((tf.get(t, 0) + mu * p_bg) / (dl + mu))
                scored.append((i, s))

        # ------------------------------------------------------------------
        # 2d. PRF Rocchio-like expansion
        # ------------------------------------------------------------------
        elif method == "prf":
            q_toks = tokenize(qtxt)
            base = [(i, bm25.score(q_toks, i)) for i in range(n_chunks)]
            top_fb = [i for i, _ in sorted(base, key=lambda x: x[1], reverse=True)[:5]]
            term_w: Dict[str, float] = defaultdict(float)
            qset = set(q_toks)
            for i in top_fb:
                tf = Counter(chunk_tokens[i])
                for t, f in tf.items():
                    if t in qset:
                        continue
                    term_w[t] += f * bm25.idf(t)
            expand = [t for t, _ in sorted(term_w.items(), key=lambda kv: kv[1], reverse=True)[:8]]
            expanded_q = list(q_toks) + expand
            scored = [(i, bm25.score(expanded_q, i)) for i in range(n_chunks)]

        # ------------------------------------------------------------------
        # 2e. BM25 + title exact-match boost
        # ------------------------------------------------------------------
        elif method == "bm25_title_boost":
            q_toks = tokenize(qtxt)
            qnorm = " ".join(q_toks)
            scored = []
            for i, c in enumerate(chunks):
                s = bm25.score(q_toks, i)
                cap = " ".join(tokenize(c.caption))
                title = " ".join(tokenize(c.enriched_title))
                if qnorm and (qnorm in cap or qnorm in title):
                    s += 1.5
                scored.append((i, s))

        # ------------------------------------------------------------------
        # 2f. BM25 + same-doc proximity boost (GT doc oracle analysis)
        # ------------------------------------------------------------------
        elif method == "proximity":
            q_toks = tokenize(qtxt)
            gt_eids = set(query_element_ids(q))
            gt_docs = {
                chunks[chunk_id_to_idx[eid]].doc_id
                for eid in gt_eids
                if eid in chunk_id_to_idx
            }
            scored = []
            for i, c in enumerate(chunks):
                s = bm25.score(q_toks, i)
                if c.doc_id in gt_docs:
                    s += 0.35
                scored.append((i, s))

        # ------------------------------------------------------------------
        # 3. graph_hub_rerank: BM25 + static hub prior
        #    Only boosts elements within BM25 top-N; alpha-blends normalized
        #    BM25 with element/doc-level hub prior score.
        # ------------------------------------------------------------------
        elif method == "graph_hub_rerank":
            q_toks = tokenize(qtxt)
            norm_bm25 = _bm25_norm_scores(bm25, q_toks, n_chunks)
            ranked_bm25 = sorted(range(n_chunks), key=lambda i: norm_bm25[i], reverse=True)
            rerank_n = max(top_k, min(graph_rerank_topn, n_chunks))
            candidate_set = set(ranked_bm25[:rerank_n])

            bm25_top10_set = set(ranked_bm25[:top_k])
            hub_boosted_ids = {
                i for i in candidate_set
                if element_hub_prior.get(chunks[i].chunk_id,
                   doc_hub_prior.get(chunks[i].doc_id, 0.0)) > 0
            }
            if hub_boosted_ids:
                hub_bm25_overlap_rates.append(
                    len(bm25_top10_set & hub_boosted_ids) / len(hub_boosted_ids)
                )

            scored = []
            for i, c in enumerate(chunks):
                base = norm_bm25[i]
                if i in candidate_set:
                    prior = element_hub_prior.get(c.chunk_id,
                            doc_hub_prior.get(c.doc_id, 0.0))
                    s = base + graph_alpha * prior
                else:
                    s = base
                scored.append((i, s))

        # ------------------------------------------------------------------
        # 4. graph_neighbor_prop: BM25 + 1-hop intra-element neighbor propagation
        #
        #    Signal design (label propagation variant):
        #      For each element e in BM25 top-N:
        #        For each neighbor n of e (from hub candidate pairs):
        #          neighbor_boost[n] += norm_bm25[e] * neighbor_decay
        #    Final score = norm_bm25[i] + neighbor_boost[i]
        #
        #    This captures: "if a figure is relevant, co-referenced elements
        #    (tables/formulas bridged via the same hub path) should also rank up."
        # ------------------------------------------------------------------
        elif method == "graph_neighbor_prop":
            q_toks = tokenize(qtxt)
            norm_bm25 = _bm25_norm_scores(bm25, q_toks, n_chunks)
            ranked_bm25 = sorted(range(n_chunks), key=lambda i: norm_bm25[i], reverse=True)
            rerank_n = max(top_k, min(graph_rerank_topn, n_chunks))

            # Propagate from top-N seeds to their neighbors (1-hop or 2-hop)
            # Edge weights modulate propagation strength: strong refs propagate more
            neighbor_boost = [0.0] * n_chunks
            for seed_idx in ranked_bm25[:rerank_n]:
                seed_id = chunks[seed_idx].chunk_id
                seed_score = norm_bm25[seed_idx]
                nbrs = element_adjacency.get(seed_id, set())
                seed_weights = (element_adj_weights or {}).get(seed_id, {})
                for nbr_id in nbrs:
                    edge_w = seed_weights.get(nbr_id, 1.0)
                    nbr_idx = chunk_id_to_idx.get(nbr_id)
                    if nbr_idx is not None:
                        neighbor_boost[nbr_idx] = min(
                            neighbor_boost[nbr_idx] + seed_score * neighbor_decay * edge_w,
                            neighbor_decay,
                        )
                    # 2-hop: propagate from neighbors to their neighbors (decay²)
                    if neighbor_hops >= 2:
                        nbrs2 = element_adjacency.get(nbr_id, set())
                        nbr_weights = (element_adj_weights or {}).get(nbr_id, {})
                        for nbr2_id in nbrs2:
                            if nbr2_id == seed_id:
                                continue  # skip backtrack
                            edge_w2 = nbr_weights.get(nbr2_id, 1.0)
                            nbr2_idx = chunk_id_to_idx.get(nbr2_id)
                            if nbr2_idx is not None:
                                neighbor_boost[nbr2_idx] = min(
                                    neighbor_boost[nbr2_idx] + seed_score * neighbor_decay * neighbor_decay * edge_w * edge_w2,
                                    neighbor_decay * neighbor_decay,
                                )

            scored = [(i, norm_bm25[i] + neighbor_boost[i]) for i in range(n_chunks)]

        # ------------------------------------------------------------------
        # 4b. graph_ppr: BM25 + Personalized PageRank on element graph
        #
        #    Signal design:
        #      Teleport set = BM25 top-N elements with their BM25 scores as
        #      personalization weights. Run PPR iterations on element_adjacency
        #      graph. Result: smooth relevance scores that propagate through
        #      the graph structure rather than just 1-hop.
        #
        #    Advantages over simple neighbor_prop:
        #      - Multi-hop propagation with damping (no fixed hop cutoff)
        #      - Edge-type agnostic smoothing
        #      - Naturally handles cycles and high-degree nodes
        # ------------------------------------------------------------------
        elif method == "graph_ppr":
            q_toks = tokenize(qtxt)
            norm_bm25 = _bm25_norm_scores(bm25, q_toks, n_chunks)
            ranked_bm25 = sorted(range(n_chunks), key=lambda i: norm_bm25[i], reverse=True)
            rerank_n = max(top_k, min(graph_rerank_topn, n_chunks))

            # Build teleport distribution from BM25 top-N
            teleport: Dict[str, float] = {}
            for seed_idx in ranked_bm25[:rerank_n]:
                seed_id = chunks[seed_idx].chunk_id
                if seed_id in element_adjacency:
                    teleport[seed_id] = norm_bm25[seed_idx]

            # Normalize teleport
            t_sum = sum(teleport.values())
            if t_sum > 1e-9:
                teleport = {k: v / t_sum for k, v in teleport.items()}

            # Run PPR on element adjacency graph
            # Collect all nodes reachable within 3 hops of teleport set
            active_nodes: Set[str] = set(teleport.keys())
            frontier = set(teleport.keys())
            for _ in range(3):
                next_frontier: Set[str] = set()
                for nid in frontier:
                    next_frontier.update(element_adjacency.get(nid, set()))
                active_nodes.update(next_frontier)
                frontier = next_frontier

            if active_nodes and teleport:
                damping = 0.85
                ppr: Dict[str, float] = {nid: teleport.get(nid, 0.0) for nid in active_nodes}
                for _ in range(20):  # 20 iterations sufficient for convergence
                    new_ppr: Dict[str, float] = {}
                    for nid in active_nodes:
                        # Teleport component
                        r = (1.0 - damping) * teleport.get(nid, 0.0)
                        # Propagation from neighbors
                        for nbr in element_adjacency.get(nid, set()):
                            if nbr in active_nodes:
                                out_deg = len(element_adjacency.get(nbr, set()))
                                if out_deg > 0:
                                    edge_w = (element_adj_weights or {}).get(nbr, {}).get(nid, 1.0)
                                    r += damping * ppr.get(nbr, 0.0) * edge_w / out_deg
                        new_ppr[nid] = r
                    ppr = new_ppr

                # Normalize PPR scores
                max_ppr = max(ppr.values()) if ppr else 1.0
                if max_ppr > 1e-9:
                    ppr = {k: v / max_ppr for k, v in ppr.items()}

                # Apply PPR boost
                ppr_boost = [0.0] * n_chunks
                for nid, score in ppr.items():
                    idx = chunk_id_to_idx.get(nid)
                    if idx is not None:
                        ppr_boost[idx] = score * neighbor_decay

                scored = [(i, norm_bm25[i] + ppr_boost[i]) for i in range(n_chunks)]
            else:
                scored = [(i, norm_bm25[i]) for i in range(n_chunks)]

        # ------------------------------------------------------------------
        # 4c. HITS: authority score on active element graph
        # ------------------------------------------------------------------
        elif method == "hits":
            q_toks = tokenize(qtxt)
            norm_bm25 = _bm25_norm_scores(bm25, q_toks, n_chunks)
            seeds = sorted(range(n_chunks), key=lambda i: norm_bm25[i], reverse=True)[:100]
            active = set(chunks[i].chunk_id for i in seeds)
            for nid in list(active):
                active.update(element_adjacency.get(nid, set()))

            if not active:
                scored = [(i, norm_bm25[i]) for i in range(n_chunks)]
            else:
                hubs = {nid: 1.0 for nid in active}
                auth = {nid: 1.0 for nid in active}
                for _ in range(20):
                    new_auth = {}
                    for nid in active:
                        new_auth[nid] = sum(
                            hubs.get(nb, 0.0) for nb in element_adjacency.get(nid, set()) if nb in active
                        )
                    na = math.sqrt(sum(v * v for v in new_auth.values())) or 1.0
                    for nid in new_auth:
                        new_auth[nid] /= na

                    new_hubs = {}
                    for nid in active:
                        new_hubs[nid] = sum(
                            new_auth.get(nb, 0.0) for nb in element_adjacency.get(nid, set()) if nb in active
                        )
                    nh = math.sqrt(sum(v * v for v in new_hubs.values())) or 1.0
                    for nid in new_hubs:
                        new_hubs[nid] /= nh
                    auth, hubs = new_auth, new_hubs

                scored = []
                for i, c in enumerate(chunks):
                    scored.append((i, norm_bm25[i] + 0.4 * auth.get(c.chunk_id, 0.0)))

        # ------------------------------------------------------------------
        # 5. graph_citation_walk: BM25 + cross-doc citation relevance propagation
        #
        #    Signal design (authority propagation):
        #      doc_score[d] = max(norm_bm25[i] for i in chunks of d)
        #      For each doc d:
        #        citation_boost[d] = sum(doc_score[d2] * citation_decay
        #                               for d2 in cites[d] ∪ cited_by[d])
        #      All chunks of doc d get += citation_boost[d]
        #
        #    This captures: "if paper A is relevant, papers A cites and papers
        #    that cite A are more likely to contain supporting evidence."
        # ------------------------------------------------------------------
        elif method == "graph_citation_walk":
            q_toks = tokenize(qtxt)
            norm_bm25 = _bm25_norm_scores(bm25, q_toks, n_chunks)

            # Aggregate doc-level relevance score
            doc_score: Dict[str, float] = defaultdict(float)
            for i, c in enumerate(chunks):
                doc_score[c.doc_id] = max(doc_score[c.doc_id], norm_bm25[i])

            # Bidirectional citation walk with corrected asymmetry:
            #   - "A cites B" → B is a reference that A builds upon → B likely has
            #     foundational evidence. Weight = 1.0 (full propagation TO cited papers).
            #   - "A cited_by C" → C builds on A → C may have follow-up evidence.
            #     Weight = 0.5 (weaker, exploratory signal).
            # Previous bug: weights were inverted ("cites" got full, "cited_by" got 0.5
            # but "cites" means A→B where B is the cited paper, so we should boost B
            # when A is relevant, which is correct direction. The real fix is to also
            # propagate IN REVERSE: when B is relevant, boost A (papers that cite B).
            citation_boost: Dict[str, float] = defaultdict(float)
            score_gate = 0.3  # lowered from 0.5 to catch more propagation sources
            for doc_id, dscore in doc_score.items():
                if dscore < score_gate:
                    continue
                adj = citation_adjacency.get(doc_id, {})
                # Forward: if A is relevant, boost papers A cites (foundational refs)
                for neighbor_doc in adj.get("cites", set()):
                    citation_boost[neighbor_doc] += dscore * citation_decay
                # Forward: if A is relevant, boost papers that cite A (follow-up work)
                for neighbor_doc in adj.get("cited_by", set()):
                    citation_boost[neighbor_doc] += dscore * citation_decay * 0.5

            # Reverse propagation: for each doc D that has high BM25 score,
            # also boost docs that are cited BY D (i.e., D's references).
            # This captures: "if evidence is in paper B, papers that cite B
            # might also contain related evidence."
            for doc_id, dscore in doc_score.items():
                if dscore < score_gate:
                    continue
                # Find docs that cite this doc (reverse direction)
                adj = citation_adjacency.get(doc_id, {})
                for citing_doc in adj.get("cited_by", set()):
                    # The citing doc likely discusses similar topics
                    citing_adj = citation_adjacency.get(citing_doc, {})
                    # Boost other papers that the citing doc also cites (2-hop)
                    for co_cited in citing_adj.get("cites", set()):
                        if co_cited != doc_id:
                            citation_boost[co_cited] += dscore * citation_decay * 0.3

            # Normalize to [0, 1]
            if citation_boost:
                max_cb = max(citation_boost.values())
                if max_cb > 1e-9:
                    citation_boost = {k: v / max_cb for k, v in citation_boost.items()}

            scored = [
                (i, norm_bm25[i] + citation_decay * citation_boost.get(chunks[i].doc_id, 0.0))
                for i in range(n_chunks)
            ]

        # ------------------------------------------------------------------
        # 6. graph_full: BM25 + hub_boost + neighbor_prop + citation_walk
        #
        #    Combined graph signal with decoupled component weights:
        #      s = norm_bm25
        #          + hub_w   * hub_prior          (static bridge signal)
        #          + nprop_w * neighbor_boost      (dynamic intra propagation)
        #          + cite_w  * citation_boost      (cross-doc authority)
        #
        #    hub_w, nprop_w, cite_w default to graph_alpha, 1.0, citation_decay
        #    but can be independently tuned via --hub-weight/--nprop-weight/--cite-weight.
        # ------------------------------------------------------------------
        elif method == "graph_full":
            q_toks = tokenize(qtxt)
            norm_bm25 = _bm25_norm_scores(bm25, q_toks, n_chunks)
            ranked_bm25 = sorted(range(n_chunks), key=lambda i: norm_bm25[i], reverse=True)
            rerank_n = max(top_k, min(graph_rerank_topn, n_chunks))
            candidate_set = set(ranked_bm25[:rerank_n])

            # Resolve component weights (use explicit overrides or defaults)
            _hw = hub_weight if hub_weight is not None else graph_alpha
            _nw = nprop_weight if nprop_weight is not None else 1.0
            _cw = cite_weight if cite_weight is not None else citation_decay

            # --- hub boost (static prior) ---
            hub_boost_arr = [0.0] * n_chunks
            if _hw > 1e-9:
                for i in candidate_set:
                    c = chunks[i]
                    hub_boost_arr[i] = element_hub_prior.get(
                        c.chunk_id, doc_hub_prior.get(c.doc_id, 0.0)
                    )

            # --- neighbor propagation (dynamic intra-element, 1 or 2-hop, weighted) ---
            neighbor_boost = [0.0] * n_chunks
            if _nw > 1e-9:
                for seed_idx in ranked_bm25[:rerank_n]:
                    seed_id = chunks[seed_idx].chunk_id
                    seed_score = norm_bm25[seed_idx]
                    seed_weights = (element_adj_weights or {}).get(seed_id, {})
                    for nbr_id in element_adjacency.get(seed_id, set()):
                        edge_w = seed_weights.get(nbr_id, 1.0)
                        nbr_idx = chunk_id_to_idx.get(nbr_id)
                        if nbr_idx is not None:
                            neighbor_boost[nbr_idx] = min(
                                neighbor_boost[nbr_idx] + seed_score * neighbor_decay * edge_w,
                                neighbor_decay,
                            )
                        # 2-hop propagation
                        if neighbor_hops >= 2:
                            nbr_weights = (element_adj_weights or {}).get(nbr_id, {})
                            for nbr2_id in element_adjacency.get(nbr_id, set()):
                                if nbr2_id == seed_id:
                                    continue
                                edge_w2 = nbr_weights.get(nbr2_id, 1.0)
                                nbr2_idx = chunk_id_to_idx.get(nbr2_id)
                                if nbr2_idx is not None:
                                    neighbor_boost[nbr2_idx] = min(
                                        neighbor_boost[nbr2_idx] + seed_score * neighbor_decay * neighbor_decay * edge_w * edge_w2,
                                        neighbor_decay * neighbor_decay,
                                    )

            # --- citation walk (cross-doc, bidirectional + 2-hop co-citation) ---
            citation_boost_map: Dict[str, float] = defaultdict(float)
            if _cw > 1e-9:
                doc_score: Dict[str, float] = defaultdict(float)
                for i, c in enumerate(chunks):
                    doc_score[c.doc_id] = max(doc_score[c.doc_id], norm_bm25[i])

                score_gate = 0.3
                for doc_id, dscore in doc_score.items():
                    if dscore < score_gate:
                        continue
                    adj = citation_adjacency.get(doc_id, {})
                    for nbr_doc in adj.get("cites", set()):
                        citation_boost_map[nbr_doc] += dscore * citation_decay
                    for nbr_doc in adj.get("cited_by", set()):
                        citation_boost_map[nbr_doc] += dscore * citation_decay * 0.5

                for doc_id, dscore in doc_score.items():
                    if dscore < score_gate:
                        continue
                    adj = citation_adjacency.get(doc_id, {})
                    for citing_doc in adj.get("cited_by", set()):
                        citing_adj = citation_adjacency.get(citing_doc, {})
                        for co_cited in citing_adj.get("cites", set()):
                            if co_cited != doc_id:
                                citation_boost_map[co_cited] += dscore * citation_decay * 0.3

                if citation_boost_map:
                    max_cb = max(citation_boost_map.values())
                    if max_cb > 1e-9:
                        citation_boost_map = {k: v / max_cb for k, v in citation_boost_map.items()}

            # --- combine with decoupled weights ---
            scored = []
            for i, c in enumerate(chunks):
                s = (
                    norm_bm25[i]
                    + _hw * hub_boost_arr[i]
                    + _nw * neighbor_boost[i]
                    + _cw * citation_boost_map.get(c.doc_id, 0.0)
                )
                scored.append((i, s))

        # ------------------------------------------------------------------
        # 7. rrf: Reciprocal Rank Fusion of bm25 and graph_full
        # ------------------------------------------------------------------
        elif method == "rrf":
            q_toks = tokenize(qtxt)
            bm25_raw = [bm25.score(q_toks, i) for i in range(n_chunks)]
            bm_rank = sorted(range(n_chunks), key=lambda i: bm25_raw[i], reverse=True)

            norm_bm25 = _bm25_norm_scores(bm25, q_toks, n_chunks)
            ranked_bm25 = sorted(range(n_chunks), key=lambda i: norm_bm25[i], reverse=True)
            rerank_n = max(top_k, min(graph_rerank_topn, n_chunks))
            candidate_set = set(ranked_bm25[:rerank_n])
            _hw = hub_weight if hub_weight is not None else graph_alpha
            _nw = nprop_weight if nprop_weight is not None else 1.0
            _cw = cite_weight if cite_weight is not None else citation_decay

            hub_boost_arr = [0.0] * n_chunks
            for i in candidate_set:
                c = chunks[i]
                hub_boost_arr[i] = element_hub_prior.get(c.chunk_id, doc_hub_prior.get(c.doc_id, 0.0))

            neighbor_boost = [0.0] * n_chunks
            for seed_idx in ranked_bm25[:rerank_n]:
                seed_id = chunks[seed_idx].chunk_id
                seed_score = norm_bm25[seed_idx]
                seed_weights = (element_adj_weights or {}).get(seed_id, {})
                for nbr_id in element_adjacency.get(seed_id, set()):
                    edge_w = seed_weights.get(nbr_id, 1.0)
                    nbr_idx = chunk_id_to_idx.get(nbr_id)
                    if nbr_idx is not None:
                        neighbor_boost[nbr_idx] += seed_score * neighbor_decay * edge_w

            citation_boost_map: Dict[str, float] = defaultdict(float)
            doc_score: Dict[str, float] = defaultdict(float)
            for i, c in enumerate(chunks):
                doc_score[c.doc_id] = max(doc_score[c.doc_id], norm_bm25[i])
            for doc_id, dscore in doc_score.items():
                if dscore < 0.3:
                    continue
                adj = citation_adjacency.get(doc_id, {})
                for nbr_doc in adj.get("cites", set()):
                    citation_boost_map[nbr_doc] += dscore * citation_decay
                for nbr_doc in adj.get("cited_by", set()):
                    citation_boost_map[nbr_doc] += dscore * citation_decay * 0.5
            if citation_boost_map:
                max_cb = max(citation_boost_map.values())
                if max_cb > 1e-9:
                    citation_boost_map = {k: v / max_cb for k, v in citation_boost_map.items()}

            g_scores = []
            for i, c in enumerate(chunks):
                g_scores.append(
                    norm_bm25[i]
                    + _hw * hub_boost_arr[i]
                    + _nw * neighbor_boost[i]
                    + _cw * citation_boost_map.get(c.doc_id, 0.0)
                )
            g_rank = sorted(range(n_chunks), key=lambda i: g_scores[i], reverse=True)
            bm_pos = {idx: r for r, idx in enumerate(bm_rank, start=1)}
            g_pos = {idx: r for r, idx in enumerate(g_rank, start=1)}
            scored = []
            for i in range(n_chunks):
                s = 1.0 / (60 + bm_pos[i]) + 1.0 / (60 + g_pos[i])
                scored.append((i, s))

        else:
            raise ValueError(f"Unknown method: {method!r}")

        # ------------------------------------------------------------------
        # Rank and compute metrics
        # ------------------------------------------------------------------
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        top = ranked[:top_k]

        gt_eids = set(query_element_ids(q))
        eid_hit_ranks: List[int] = []
        for rank_idx, (ci, _s) in enumerate(top, start=1):
            if chunks[ci].chunk_id in gt_eids:
                eid_hit_ranks.append(rank_idx)

        hit_ranks_overlap: List[int] = []
        best_overlap = 0.0
        for rank_idx, (ci, _s) in enumerate(top, start=1):
            ctext = chunks[ci].text
            ov = max((span_overlap(sp, ctext) for sp in spans), default=0.0)
            best_overlap = max(best_overlap, ov)
            if ov >= overlap_threshold:
                hit_ranks_overlap.append(rank_idx)

        hit_ranks = eid_hit_ranks if gt_eids else hit_ranks_overlap

        retrieved_ids = [chunks[ci].chunk_id for ci, _ in ranked]
        hit10 = 1.0 if hit_ranks else 0.0
        rr = reciprocal_rank_binary(hit_ranks)
        for k in (1, 3, 5, 10, 20):
            recall_sums[k] += coverage_at_k(retrieved_ids, gt_eids, k)
        mrr += rr
        coverage10 += coverage_at_k(retrieved_ids, gt_eids, 10)
        ndcg10 += ndcg_at_k(retrieved_ids, query_element_gains(q), 10)
        per_query.append(
            {
                "query_id": q.get("query_id"),
                "hit_at_10": hit10,
                "rr": rr,
                "best_overlap_in_topk": round(best_overlap, 4),
                "gt_element_ids": list(gt_eids),
            }
        )

    n = max(1, len(per_query))
    result: Dict[str, Any] = {
        "n": len(per_query),
        "recall_at_1": round(recall_sums[1] / n, 4),
        "recall_at_3": round(recall_sums[3] / n, 4),
        "recall_at_5": round(recall_sums[5] / n, 4),
        "recall_at_10": round(recall_sums[10] / n, 4),
        "recall_at_20": round(recall_sums[20] / n, 4),
        "mrr": round(mrr / n, 4),
        "coverage_at_10": round(coverage10 / n, 4),
        "ndcg_at_10": round(ndcg10 / n, 4),
        "per_query": per_query,
    }
    if hub_bm25_overlap_rates:
        result["hub_bm25_overlap_mean"] = round(
            sum(hub_bm25_overlap_rates) / len(hub_bm25_overlap_rates), 4
        )
        result["hub_bm25_overlap_n_queries"] = len(hub_bm25_overlap_rates)
    return result


# go/no-go 门禁：Recall@10≥BM25+0.05 OR MRR≥BM25+0.03 则 continue_expand
def decision(graph_full_metrics: Dict[str, Any], bm25_metrics: Dict[str, Any]) -> Dict[str, Any]:
    g_r10 = graph_full_metrics["recall_at_10"]
    g_mrr = graph_full_metrics["mrr"]
    b_r10 = bm25_metrics["recall_at_10"]
    b_mrr = bm25_metrics["mrr"]

    delta_r10 = g_r10 - b_r10
    delta_mrr = g_mrr - b_mrr
    continue_expand = (delta_r10 >= 0.05) or (delta_mrr >= 0.03)
    return {
        "primary_comparison": "graph_full vs bm25",
        "delta_recall_at_10_vs_bm25": round(delta_r10, 4),
        "delta_mrr_vs_bm25": round(delta_mrr, 4),
        "continue_expand": continue_expand,
        "rule": "continue if Recall@10 >= BM25+0.05 OR MRR >= BM25+0.03",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Phase-0 locked A/B retrieval evaluation")
    ap.add_argument("--q1", type=Path, default=Path("data/l1_dual_evidence_queries_v4_4_run1_pass.jsonl"))
    ap.add_argument("--q2", type=Path, default=Path("data/l1_dual_evidence_queries_v3_pass.jsonl"))
    ap.add_argument("--q3", type=Path, default=None, help="Optional extra query file")
    ap.add_argument("--elements", type=Path, default=Path("data111/multimodal_elements_enriched.json"))
    ap.add_argument("--hubs", type=Path, default=Path("data111/latex_graph_hubs (1).json"))
    ap.add_argument("--hub-candidates", type=Path, default=Path("data111/hub_candidates_enriched_v2.json"),
                    help="Enriched hub candidate pairs for element-level prior + adjacency")
    ap.add_argument("--citation-graph", type=Path, default=Path("data/citation_graph.json"),
                    help="Citation graph JSON for cross-doc citation walk")
    ap.add_argument("--output", type=Path, default=Path("data/phase0_eval_report.json"))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--overlap-threshold", type=float, default=0.5)
    # Graph signal weights
    ap.add_argument("--graph-alpha", type=float, default=0.2,
                    help="Hub prior boost weight (graph_hub_rerank + graph_full)")
    ap.add_argument("--graph-rerank-topn", type=int, default=100,
                    help="Apply graph prior only within BM25 top-N candidates")
    ap.add_argument("--neighbor-decay", type=float, default=0.5,
                    help="Neighbor propagation decay factor (graph_neighbor_prop + graph_full)")
    ap.add_argument("--citation-decay", type=float, default=0.3,
                    help="Citation walk decay factor (graph_citation_walk + graph_full)")
    # Independent component weights for graph_full (decoupled from individual method params)
    ap.add_argument("--hub-weight", type=float, default=None,
                    help="Hub prior weight in graph_full (default: same as --graph-alpha)")
    ap.add_argument("--nprop-weight", type=float, default=None,
                    help="Neighbor propagation weight in graph_full (default: 1.0, uses neighbor_decay internally)")
    ap.add_argument("--cite-weight", type=float, default=None,
                    help="Citation walk weight in graph_full (default: same as --citation-decay). Set to 0 to disable.")
    ap.add_argument("--neighbor-hops", type=int, default=1,
                    help="Number of neighbor propagation hops (1 or 2)")
    ap.add_argument("--max-chars", type=int, default=1800)
    ap.add_argument("--embedding-edges", type=Path, default=None,
                    help="Embedding-based edges JSON (from build_embedding_edges.py). "
                         "Merged into element adjacency for neighbor propagation.")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    rows = load_jsonl(args.q1) + load_jsonl(args.q2)
    if args.q3 is not None and args.q3.exists():
        rows += load_jsonl(args.q3)
    queries = dedupe_queries(rows)
    chunks = build_chunks(args.elements, max_chars=args.max_chars)

    doc_hub_prior = load_doc_hub_prior(args.hubs)
    element_hub_prior = (
        load_element_hub_prior(args.hub_candidates)
        if args.hub_candidates.exists() else {}
    )
    element_adjacency, element_adj_weights = (
        load_element_adjacency(args.hub_candidates)
        if args.hub_candidates.exists() else ({}, {})
    )
    # Merge embedding-based edges if provided
    if args.embedding_edges and args.embedding_edges.exists():
        embed_data = json.loads(args.embedding_edges.read_text(encoding="utf-8"))
        embed_edge_list = embed_data.get("edges", [])
        n_new = 0
        for e in embed_edge_list:
            a = str(e.get("element_a_id", "")).strip()
            b = str(e.get("element_b_id", "")).strip()
            if a and b:
                if a not in element_adjacency:
                    element_adjacency[a] = set()
                elif not isinstance(element_adjacency[a], set):
                    element_adjacency[a] = set(element_adjacency[a])
                if b not in element_adjacency:
                    element_adjacency[b] = set()
                elif not isinstance(element_adjacency[b], set):
                    element_adjacency[b] = set(element_adjacency[b])
                if b not in element_adjacency[a]:
                    n_new += 1
                element_adjacency[a].add(b)
                element_adjacency[b].add(a)
        print(f"  Merged {n_new} new embedding edges from {args.embedding_edges.name}")
    citation_adjacency = (
        load_citation_adjacency(args.citation_graph)
        if args.citation_graph.exists() else {}
    )

    if not queries:
        raise SystemExit("No queries loaded")
    if not chunks:
        raise SystemExit("No chunks loaded")

    # Build reverse lookup indices
    chunk_id_to_idx: Dict[str, int] = {c.chunk_id: i for i, c in enumerate(chunks)}
    doc_to_chunk_idxs: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        doc_to_chunk_idxs[c.doc_id].append(i)

    chunk_tokens = [tokenize(c.text) for c in chunks]
    bm25 = BM25Lite(chunk_tokens)
    field_tokens = {
        "caption": [tokenize(c.caption) for c in chunks],
        "content": [tokenize(c.content) for c in chunks],
        "context": [tokenize(c.context) for c in chunks],
        "enriched_title": [tokenize(c.enriched_title) for c in chunks],
        "enriched_content": [tokenize(c.enriched_content) for c in chunks],
    }
    collection_tf: Dict[str, int] = defaultdict(int)
    for toks in chunk_tokens:
        for t in toks:
            collection_tf[t] += 1
    total_collection_terms = sum(collection_tf.values())

    # Diagnostic summary
    n_elem_adj = sum(len(v) for v in element_adjacency.values())
    n_cite_docs = len(citation_adjacency)
    print(f"[phase0] queries={len(queries)}  chunks={len(chunks)}")
    print(f"[phase0] element_hub_prior={len(element_hub_prior)}  "
          f"element_adjacency_edges={n_elem_adj // 2}  "
          f"citation_docs={n_cite_docs}")

    eval_kwargs = dict(
        chunks=chunks,
        bm25=bm25,
        doc_hub_prior=doc_hub_prior,
        element_hub_prior=element_hub_prior,
        element_adjacency=element_adjacency,
        citation_adjacency=citation_adjacency,
        chunk_id_to_idx=chunk_id_to_idx,
        doc_to_chunk_idxs=doc_to_chunk_idxs,
        dense_matrix=None,
        vectorizer=None,
        chunk_tokens=chunk_tokens,
        field_tokens=field_tokens,
        collection_tf=collection_tf,
        total_collection_terms=total_collection_terms,
        top_k=args.top_k,
        overlap_threshold=args.overlap_threshold,
        graph_alpha=args.graph_alpha,
        graph_rerank_topn=args.graph_rerank_topn,
        neighbor_decay=args.neighbor_decay,
        citation_decay=args.citation_decay,
        hub_weight=args.hub_weight,
        nprop_weight=args.nprop_weight,
        cite_weight=args.cite_weight,
        neighbor_hops=args.neighbor_hops,
        element_adj_weights=element_adj_weights,
    )

    all_methods = [
        "bm25",
        "bm25f",
        "lm_dirichlet",
        "prf",
        "bm25_title_boost",
        "proximity",
        "dense",
        "graph_hub_rerank",
        "graph_neighbor_prop",
        "graph_ppr",
        "hits",
        "graph_citation_walk",
        "graph_full",
        "rrf",
    ]

    metrics: Dict[str, Any] = {}
    details: Dict[str, Any] = {}
    for m_name in all_methods:
        print(f"[phase0] running {m_name} ...")
        res = evaluate_method(m_name, queries, **eval_kwargs)
        metrics[m_name] = {k: v for k, v in res.items() if k != "per_query"}
        details[m_name] = res["per_query"]

    # ---- Layered evaluation: split queries by hub_overlap ----
    # For each query, check if any GT element_id is in element_hub_prior
    hub_overlap_queries: List[Dict[str, Any]] = []
    non_hub_queries: List[Dict[str, Any]] = []
    for q in queries:
        gt_eids = set(query_element_ids(q))
        has_hub = any(eid in element_hub_prior for eid in gt_eids)
        if has_hub:
            hub_overlap_queries.append(q)
        else:
            non_hub_queries.append(q)

    layered: Dict[str, Any] = {
        "hub_overlap_count": len(hub_overlap_queries),
        "non_hub_count": len(non_hub_queries),
        "total": len(queries),
        "hub_overlap_pct": round(len(hub_overlap_queries) / max(1, len(queries)) * 100, 2),
    }

    # Run layered eval for key methods on hub_overlap subset
    if hub_overlap_queries:
        for m_name in ["bm25", "graph_hub_rerank", "graph_neighbor_prop", "graph_full"]:
            res = evaluate_method(m_name, hub_overlap_queries, **eval_kwargs)
            layered[f"{m_name}_hub_subset"] = {
                "n": res["n"],
                "recall_at_10": res["recall_at_10"],
                "mrr": res["mrr"],
            }

    report = {
        "config": {
            "q1": str(args.q1),
            "q2": str(args.q2),
            "elements": str(args.elements),
            "hubs": str(args.hubs),
            "hub_candidates": str(args.hub_candidates),
            "citation_graph": str(args.citation_graph),
            "n_element_hub_prior": len(element_hub_prior),
            "n_element_adjacency_edges": n_elem_adj // 2,
            "n_citation_docs": n_cite_docs,
            "top_k": args.top_k,
            "overlap_threshold": args.overlap_threshold,
            "graph_alpha": args.graph_alpha,
            "graph_rerank_topn": args.graph_rerank_topn,
            "neighbor_decay": args.neighbor_decay,
            "citation_decay": args.citation_decay,
            "max_chars": args.max_chars,
            "n_queries": len(queries),
            "n_chunks": len(chunks),
        },
        "metrics": metrics,
        "layered": layered,
        "decision": decision(metrics["graph_full"], metrics["bm25"]),
        "details": details,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[phase0] report written:", args.output)
    print(f"{'method':<22} {'n':>5}  {'Recall@10':>10}  {'MRR':>8}")
    print("-" * 52)
    for m_name in all_methods:
        m = metrics[m_name]
        extra = ""
        if "hub_bm25_overlap_mean" in m:
            extra = f"  hub_overlap={m['hub_bm25_overlap_mean']:.4f}"
        print(f"  {m_name:<20} {m['n']:>5}  {m['recall_at_10']:>10.4f}  {m['mrr']:>8.4f}{extra}")

    # Print layered results
    print(f"\n[phase0] Layered evaluation:")
    print(f"  Hub-overlap queries: {layered['hub_overlap_count']} ({layered['hub_overlap_pct']}%)")
    print(f"  Non-hub queries: {layered['non_hub_count']}")
    if hub_overlap_queries:
        print(f"\n  {'method':<22} {'n':>5}  {'Recall@10':>10}  {'MRR':>8}  (hub-overlap subset)")
        print("  " + "-" * 55)
        for m_name in ["bm25", "graph_hub_rerank", "graph_neighbor_prop", "graph_full"]:
            key = f"{m_name}_hub_subset"
            if key in layered:
                m = layered[key]
                print(f"    {m_name:<20} {m['n']:>5}  {m['recall_at_10']:>10.4f}  {m['mrr']:>8.4f}")

    print()
    print("[phase0] decision:", report["decision"])


if __name__ == "__main__":
    main()
