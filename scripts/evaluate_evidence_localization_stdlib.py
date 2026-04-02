#!/usr/bin/env python3
"""Pure-stdlib retrieval evaluation with classic IR/recsys algorithms.

Added methods (all from scratch, no third-party deps):
  - bm25
  - bm25f
  - lm_dirichlet
  - prf (Rocchio-style pseudo relevance feedback)
  - bm25_title_boost
  - proximity
  - hits
  - graph_full
  - rrf (BM25 + graph_full)

Metrics:
  - Recall@1/3/5/10/20
  - MRR
  - Coverage@10
  - NDCG@10 (graded by evidence_type)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.text_utils import tokenize_for_retrieval as tokenize

# TOKEN_RE and tokenize() moved to src.utils.text_utils.tokenize_for_retrieval

# TOKEN_RE moved to src.utils.text_utils
DEFAULT_METHODS = [
    "bm25",
    "bm25f",
    "lm_dirichlet",
    "prf",
    "bm25_title_boost",
    "proximity",
    "hits",
    "graph_full",
    "rrf",
]

EVIDENCE_GAIN = {
    "observation": 2.0,
    "result": 2.0,
    "mechanism": 1.0,
}


class Element:
    def __init__(self, element_id: str, doc_id: str, fields: Dict[str, str]):
        self.element_id = element_id
        self.doc_id = doc_id
        self.fields = fields

    @property
    def full_text(self) -> str:
        return "\n".join(v for v in self.fields.values() if v).strip()


# tokenize() moved to src.utils.text_utils.tokenize_for_retrieval


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dedupe_queries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out = []
    for r in rows:
        qid = str(r.get("query_id") or "")
        q = (r.get("query") or "").strip()
        key = qid or f"txt::{q}"
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


def load_elements(elements_path: Path, max_chars: int = 2000) -> List[Element]:
    raw = json.loads(elements_path.read_text(encoding="utf-8"))
    docs = raw.get("documents", {}) or {}
    out: List[Element] = []
    for doc_id, d in docs.items():
        for eid, e in (d.get("elements", {}) or {}).items():
            context = " ".join([
                _to_text(e.get("context_before")),
                _to_text(e.get("context_after")),
            ]).strip()
            fields = {
                "caption": _to_text(e.get("caption")),
                "content": _to_text(e.get("content")),
                "enriched_title": _to_text(e.get("enriched_title")),
                "enriched_content": _to_text(e.get("enriched_content")),
                "context": context,
            }
            fields = {k: (v[:max_chars] if len(v) > max_chars else v) for k, v in fields.items()}
            if not any(fields.values()):
                continue
            out.append(Element(str(eid), str(doc_id), fields))
    return out


class CorpusIndex:
    def __init__(self, elements: List[Element]):
        self.elements = elements
        self.N = len(elements)
        self.field_names = ["caption", "content", "enriched_title", "enriched_content", "context"]

        self.tokens_all: List[List[str]] = []
        self.tf_all: List[Counter] = []
        self.doc_lens: List[int] = []
        self.field_tokens: Dict[str, List[List[str]]] = {f: [] for f in self.field_names}
        self.field_tf: Dict[str, List[Counter]] = {f: [] for f in self.field_names}
        self.field_lens: Dict[str, List[int]] = {f: [] for f in self.field_names}

        self.df: Dict[str, int] = defaultdict(int)
        self.cf: Dict[str, int] = defaultdict(int)

        for e in elements:
            all_tok = tokenize(e.full_text)
            self.tokens_all.append(all_tok)
            tf = Counter(all_tok)
            self.tf_all.append(tf)
            self.doc_lens.append(len(all_tok))

            for t, c in tf.items():
                self.cf[t] += c
            for t in tf.keys():
                self.df[t] += 1

            for f in self.field_names:
                toks = tokenize(e.fields.get(f, ""))
                self.field_tokens[f].append(toks)
                self.field_tf[f].append(Counter(toks))
                self.field_lens[f].append(len(toks))

        self.avgdl = sum(self.doc_lens) / max(self.N, 1)
        self.avg_field_len = {
            f: sum(self.field_lens[f]) / max(self.N, 1)
            for f in self.field_names
        }
        self.total_tokens = sum(self.doc_lens)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def p_collection(self, term: str) -> float:
        return self.cf.get(term, 0) / max(1, self.total_tokens)


def load_hub_prior_and_graph(hub_candidates_path: Optional[Path]) -> Tuple[Dict[str, float], Dict[str, Set[str]]]:
    if not hub_candidates_path or not hub_candidates_path.exists():
        return {}, {}
    obj = json.loads(hub_candidates_path.read_text(encoding="utf-8"))

    prior: Dict[str, float] = {}
    graph: Dict[str, Set[str]] = defaultdict(set)

    for p in (obj.get("pairs") or []):
        a = str(p.get("element_a_id") or "").strip()
        b = str(p.get("element_b_id") or "").strip()
        if not a or not b:
            continue
        q = float(p.get("quality_score", 1.0) or 1.0)
        prior[a] = max(prior.get(a, 0.0), q)
        prior[b] = max(prior.get(b, 0.0), q)
        graph[a].add(b)
        graph[b].add(a)

    for ab in (obj.get("adjacent_bridge_adjacency") or []):
        left = [str(x).strip() for x in (ab.get("elements_i") or []) if str(x).strip()]
        right = [str(x).strip() for x in (ab.get("elements_j") or []) if str(x).strip()]
        for a in left:
            for b in right:
                graph[a].add(b)
                graph[b].add(a)

    if prior:
        mx = max(prior.values())
        if mx > 0:
            prior = {k: v / mx for k, v in prior.items()}
    return prior, dict(graph)


def load_citation_adjacency(path: Optional[Path]) -> Dict[str, Set[str]]:
    if not path or not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    adj: Dict[str, Set[str]] = defaultdict(set)
    for doc, info in (obj.get("adjacency") or {}).items():
        d = str(doc)
        for c in (info.get("cites") or []):
            adj[d].add(str(c))
        for c in (info.get("cited_by") or []):
            adj[d].add(str(c))
    return dict(adj)


def query_gt(q: Dict[str, Any]) -> Tuple[Set[str], Dict[str, float]]:
    ids: Set[str] = set()
    gains: Dict[str, float] = {}
    for s in (q.get("required_evidence_spans") or []):
        if not isinstance(s, dict):
            continue
        eid = str(s.get("element_id") or "").strip()
        if not eid:
            continue
        ids.add(eid)
        et = str(s.get("evidence_type") or "").strip().lower()
        g = EVIDENCE_GAIN.get(et, 1.0)
        gains[eid] = max(gains.get(eid, 0.0), g)
    for eid in q.get("element_ids", []) or []:
        eid = str(eid).strip()
        if eid:
            ids.add(eid)
            gains[eid] = max(gains.get(eid, 0.0), 1.0)
    return ids, gains


def score_bm25(index: CorpusIndex, q_toks: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    out = [0.0] * index.N
    qset = set(q_toks)
    for i in range(index.N):
        tf = index.tf_all[i]
        dl = index.doc_lens[i]
        s = 0.0
        for t in qset:
            f = tf.get(t, 0)
            if f <= 0:
                continue
            idf = index.idf(t)
            denom = f + k1 * (1 - b + b * dl / max(index.avgdl, 1e-9))
            s += idf * (f * (k1 + 1)) / max(denom, 1e-9)
        out[i] = s
    return out


def score_bm25f(index: CorpusIndex, q_toks: List[str]) -> List[float]:
    weights = {
        "caption": 3.0,
        "enriched_content": 2.0,
        "context": 0.5,
        "enriched_title": 2.0,
        "content": 1.0,
    }
    k1 = 1.2
    b = 0.75
    out = [0.0] * index.N
    qset = set(q_toks)
    for i in range(index.N):
        s = 0.0
        for f in index.field_names:
            tf = index.field_tf[f][i]
            dl = index.field_lens[f][i]
            avgfl = index.avg_field_len[f]
            fw = weights.get(f, 1.0)
            field_s = 0.0
            for t in qset:
                freq = tf.get(t, 0)
                if freq <= 0:
                    continue
                idf = index.idf(t)
                denom = freq + k1 * (1 - b + b * dl / max(avgfl, 1e-9))
                field_s += idf * (freq * (k1 + 1)) / max(denom, 1e-9)
            s += fw * field_s
        out[i] = s
    return out


def score_lm_dirichlet(index: CorpusIndex, q_toks: List[str], mu: float = 2000.0) -> List[float]:
    out = [0.0] * index.N
    for i in range(index.N):
        tf = index.tf_all[i]
        dl = index.doc_lens[i]
        s = 0.0
        for t in q_toks:
            p_bg = max(index.p_collection(t), 1e-12)
            num = tf.get(t, 0) + mu * p_bg
            den = dl + mu
            s += math.log(num / den)
        out[i] = s
    return out


def minmax_norm(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-12:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def ranked_ids(elements: List[Element], scores: Sequence[float], top_k: int) -> List[str]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [elements[i].element_id for i in order[:top_k]]


def score_prf(index: CorpusIndex, q_toks: List[str], fb_docs: int = 5, fb_terms: int = 8, beta: float = 0.5) -> List[float]:
    base = score_bm25(index, q_toks)
    seeds = sorted(range(index.N), key=lambda i: base[i], reverse=True)[:fb_docs]
    if not seeds:
        return base

    agg: Dict[str, float] = defaultdict(float)
    qset = set(q_toks)
    for i in seeds:
        tf = index.tf_all[i]
        for t, f in tf.items():
            if t in qset:
                continue
            agg[t] += f * index.idf(t)

    exp_terms = [t for t, _ in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:fb_terms]]
    if not exp_terms:
        return base

    qweights = defaultdict(float)
    for t in q_toks:
        qweights[t] += 1.0
    max_w = max((agg.get(t, 0.0) for t in exp_terms), default=1.0)
    for t in exp_terms:
        qweights[t] += beta * (agg.get(t, 0.0) / max(max_w, 1e-9))

    out = [0.0] * index.N
    for i in range(index.N):
        tf = index.tf_all[i]
        dl = index.doc_lens[i]
        s = 0.0
        for t, qw in qweights.items():
            f = tf.get(t, 0)
            if f <= 0:
                continue
            idf = index.idf(t)
            denom = f + 1.5 * (1 - 0.75 + 0.75 * dl / max(index.avgdl, 1e-9))
            s += qw * idf * (f * 2.5) / max(denom, 1e-9)
        out[i] = s
    return out


def score_title_boost(index: CorpusIndex, q_text: str, q_toks: List[str], bonus: float = 1.5) -> List[float]:
    base = score_bm25(index, q_toks)
    qn = " ".join(q_toks)
    for i, e in enumerate(index.elements):
        cap = " ".join(tokenize(e.fields.get("caption", "")))
        tit = " ".join(tokenize(e.fields.get("enriched_title", "")))
        if qn and (qn in cap or qn in tit):
            base[i] += bonus
    return base


def score_proximity(index: CorpusIndex, q_toks: List[str], gt_ids: Set[str], eid2idx: Dict[str, int], doc_bonus: float = 0.35) -> List[float]:
    base = score_bm25(index, q_toks)
    gt_docs = {index.elements[eid2idx[eid]].doc_id for eid in gt_ids if eid in eid2idx}
    for i, e in enumerate(index.elements):
        if e.doc_id in gt_docs:
            base[i] += doc_bonus
    return base


def score_graph_full(index: CorpusIndex, q_toks: List[str], prior: Dict[str, float], elem_graph: Dict[str, Set[str]], doc_citation: Dict[str, Set[str]], eid2idx: Dict[str, int]) -> List[float]:
    bm = minmax_norm(score_bm25(index, q_toks))
    out = bm[:]

    # hub prior
    for i, e in enumerate(index.elements):
        out[i] += 0.2 * prior.get(e.element_id, 0.0)

    # neighbor propagation from BM25 top seeds
    seeds = sorted(range(index.N), key=lambda i: bm[i], reverse=True)[:100]
    nboost = [0.0] * index.N
    for si in seeds:
        sid = index.elements[si].element_id
        for nid in elem_graph.get(sid, set()):
            ni = eid2idx.get(nid)
            if ni is not None:
                nboost[ni] += bm[si] * 0.5
    nboost = minmax_norm(nboost)
    out = [out[i] + 0.8 * nboost[i] for i in range(index.N)]

    # citation walk doc-level
    doc_score: Dict[str, float] = defaultdict(float)
    for i, e in enumerate(index.elements):
        doc_score[e.doc_id] = max(doc_score[e.doc_id], bm[i])
    cite_boost: Dict[str, float] = defaultdict(float)
    for d, s in doc_score.items():
        if s < 0.2:
            continue
        for nb in doc_citation.get(d, set()):
            cite_boost[nb] += s * 0.3
    if cite_boost:
        mx = max(cite_boost.values())
        if mx > 0:
            cite_boost = {k: v / mx for k, v in cite_boost.items()}
    for i, e in enumerate(index.elements):
        out[i] += 0.2 * cite_boost.get(e.doc_id, 0.0)

    return out


def score_hits(index: CorpusIndex, q_toks: List[str], elem_graph: Dict[str, Set[str]], eid2idx: Dict[str, int], max_iter: int = 20) -> List[float]:
    bm = minmax_norm(score_bm25(index, q_toks))
    seeds = sorted(range(index.N), key=lambda i: bm[i], reverse=True)[:100]

    active: Set[str] = set(index.elements[i].element_id for i in seeds)
    for eid in list(active):
        active.update(elem_graph.get(eid, set()))
    if not active:
        return bm

    hub = {eid: 1.0 for eid in active}
    auth = {eid: 1.0 for eid in active}

    for _ in range(max_iter):
        new_auth = {}
        for v in active:
            new_auth[v] = sum(hub.get(u, 0.0) for u in elem_graph.get(v, set()) if u in active)
        norm_a = math.sqrt(sum(x * x for x in new_auth.values())) or 1.0
        for v in new_auth:
            new_auth[v] /= norm_a

        new_hub = {}
        for v in active:
            new_hub[v] = sum(new_auth.get(u, 0.0) for u in elem_graph.get(v, set()) if u in active)
        norm_h = math.sqrt(sum(x * x for x in new_hub.values())) or 1.0
        for v in new_hub:
            new_hub[v] /= norm_h

        auth, hub = new_auth, new_hub

    out = bm[:]
    for eid, a in auth.items():
        i = eid2idx.get(eid)
        if i is not None:
            out[i] += 0.4 * a
    return out


def score_rrf(elements: List[Element], bm25_scores: Sequence[float], graph_scores: Sequence[float], c: int = 60) -> List[float]:
    b_rank = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    g_rank = sorted(range(len(graph_scores)), key=lambda i: graph_scores[i], reverse=True)
    b_pos = {idx: r for r, idx in enumerate(b_rank, start=1)}
    g_pos = {idx: r for r, idx in enumerate(g_rank, start=1)}
    out = [0.0] * len(elements)
    for i in range(len(elements)):
        out[i] = 1.0 / (c + b_pos[i]) + 1.0 / (c + g_pos[i])
    return out


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & relevant)
    return hits / len(relevant)


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    for r, eid in enumerate(retrieved, start=1):
        if eid in relevant:
            return 1.0 / r
    return 0.0


def coverage_at_10(retrieved: List[str], relevant: Set[str]) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:10]) & relevant) / len(relevant)


def ndcg_at_k(retrieved: List[str], gains: Dict[str, float], k: int = 10) -> float:
    def dcg(ids: Sequence[str]) -> float:
        s = 0.0
        for i, eid in enumerate(ids[:k], start=1):
            g = gains.get(eid, 0.0)
            if g > 0:
                s += g / math.log2(i + 1)
        return s

    ideal_ids = [eid for eid, _ in sorted(gains.items(), key=lambda kv: kv[1], reverse=True)]
    idcg = dcg(ideal_ids)
    if idcg <= 1e-12:
        return 0.0
    return dcg(retrieved) / idcg


def evaluate_method(
    method: str,
    queries: List[Dict[str, Any]],
    index: CorpusIndex,
    elem_graph: Dict[str, Set[str]],
    prior: Dict[str, float],
    doc_citation: Dict[str, Set[str]],
    top_k: int,
) -> Dict[str, Any]:
    eid2idx = {e.element_id: i for i, e in enumerate(index.elements)}
    recall_keys = [1, 3, 5, 10, 20]
    sums = {k: 0.0 for k in recall_keys}
    sum_mrr = 0.0
    sum_cov10 = 0.0
    sum_ndcg10 = 0.0
    n = 0
    per_query = []

    for q in queries:
        q_text = (q.get("query") or "").strip()
        gt, gains = query_gt(q)
        if not q_text or not gt:
            continue
        q_toks = tokenize(q_text)

        bm = score_bm25(index, q_toks)
        if method == "bm25":
            scores = bm
        elif method == "bm25f":
            scores = score_bm25f(index, q_toks)
        elif method == "lm_dirichlet":
            scores = score_lm_dirichlet(index, q_toks)
        elif method == "prf":
            scores = score_prf(index, q_toks)
        elif method == "bm25_title_boost":
            scores = score_title_boost(index, q_text, q_toks)
        elif method == "proximity":
            scores = score_proximity(index, q_toks, gt, eid2idx)
        elif method == "hits":
            scores = score_hits(index, q_toks, elem_graph, eid2idx)
        elif method == "graph_full":
            scores = score_graph_full(index, q_toks, prior, elem_graph, doc_citation, eid2idx)
        elif method == "rrf":
            gf = score_graph_full(index, q_toks, prior, elem_graph, doc_citation, eid2idx)
            scores = score_rrf(index.elements, bm, gf)
        else:
            raise ValueError(f"unknown method {method}")

        retrieved = ranked_ids(index.elements, scores, top_k)
        row = {
            "query_id": q.get("query_id"),
            "top10": retrieved[:10],
        }
        for k in recall_keys:
            rk = recall_at_k(retrieved, gt, k)
            sums[k] += rk
            row[f"recall@{k}"] = round(rk, 4)
        rr = mrr(retrieved, gt)
        cov = coverage_at_10(retrieved, gt)
        ndcg = ndcg_at_k(retrieved, gains, 10)
        sum_mrr += rr
        sum_cov10 += cov
        sum_ndcg10 += ndcg
        row["rr"] = round(rr, 4)
        row["coverage@10"] = round(cov, 4)
        row["ndcg@10"] = round(ndcg, 4)
        per_query.append(row)
        n += 1

    if n == 0:
        return {"n": 0, "error": "no valid queries", "per_query": []}

    metrics = {
        "n": n,
        "mrr": round(sum_mrr / n, 4),
        "coverage@10": round(sum_cov10 / n, 4),
        "ndcg@10": round(sum_ndcg10 / n, 4),
        "per_query": per_query,
    }
    for k in recall_keys:
        metrics[f"recall@{k}"] = round(sums[k] / n, 4)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Classic retrieval eval (pure stdlib)")
    ap.add_argument("--q1", type=Path, default=Path("data/l1_dual_evidence_queries_v4_4_run1_pass.jsonl"))
    ap.add_argument("--q2", type=Path, default=Path("data/l1_dual_evidence_queries_v3_pass.jsonl"))
    ap.add_argument("--q3", type=Path, default=None)
    ap.add_argument("--elements", type=Path, default=Path("data111/multimodal_elements_enriched.json"))
    ap.add_argument("--hub-candidates", type=Path, default=Path("data111/hub_candidates_enriched_v2.json"))
    ap.add_argument("--citation-graph", type=Path, default=Path("data/citation_graph.json"))
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--output", type=Path, default=Path("data/m2/eval_classic_stdlib.json"))
    args = ap.parse_args()

    rows = load_jsonl(args.q1) + load_jsonl(args.q2)
    if args.q3 and args.q3.exists():
        rows += load_jsonl(args.q3)
    queries = dedupe_queries(rows)

    elements = load_elements(args.elements, max_chars=args.max_chars)
    index = CorpusIndex(elements)
    prior, elem_graph = load_hub_prior_and_graph(args.hub_candidates)
    doc_citation = load_citation_adjacency(args.citation_graph)

    all_metrics: Dict[str, Any] = {}
    details: Dict[str, Any] = {}
    for m in args.methods:
        print(f"[eval] running {m} ...")
        res = evaluate_method(m, queries, index, elem_graph, prior, doc_citation, args.top_k)
        all_metrics[m] = {k: v for k, v in res.items() if k != "per_query"}
        details[m] = res.get("per_query", [])

    report = {
        "config": {
            "q1": str(args.q1),
            "q2": str(args.q2),
            "q3": str(args.q3) if args.q3 else None,
            "elements": str(args.elements),
            "hub_candidates": str(args.hub_candidates),
            "citation_graph": str(args.citation_graph),
            "methods": args.methods,
            "top_k": args.top_k,
            "n_queries": len(queries),
            "n_elements": len(elements),
        },
        "metrics": all_metrics,
        "details": details,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[eval] wrote report:", args.output)
    header = f"{'method':<18} {'n':>5} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'R@10':>7} {'R@20':>7} {'MRR':>7} {'Cov@10':>8} {'NDCG@10':>9}"
    print(header)
    print("-" * len(header))
    for m in args.methods:
        r = all_metrics.get(m, {})
        print(
            f"{m:<18} {r.get('n', 0):>5} {r.get('recall@1', 0):>7.4f} {r.get('recall@3', 0):>7.4f} "
            f"{r.get('recall@5', 0):>7.4f} {r.get('recall@10', 0):>7.4f} {r.get('recall@20', 0):>7.4f} "
            f"{r.get('mrr', 0):>7.4f} {r.get('coverage@10', 0):>8.4f} {r.get('ndcg@10', 0):>9.4f}"
        )


if __name__ == "__main__":
    main()
