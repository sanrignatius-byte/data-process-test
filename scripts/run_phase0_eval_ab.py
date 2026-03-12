#!/usr/bin/env python3
"""
Phase-0 A/B retrieval experiment runner.

Implements the locked protocol:
- Fixed query pool from v4.4 pass + v3 pass (deduped union)
- Ground-truth from required_evidence_spans
- Localization hit if char overlap >= threshold (default 0.5)
- Methods: bm25, dense(tfidf cosine), graph_hub_rerank(bm25 + hub prior)

Outputs a JSON report with Recall@10 / MRR and go/no-go decision gates.
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
from typing import Any, Dict, List, Tuple


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


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


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def build_chunks(elements_json: Path, max_chars: int = 1800) -> List[Chunk]:
    data = json.loads(elements_json.read_text(encoding="utf-8"))
    docs = data.get("documents", {}) or {}
    chunks: List[Chunk] = []
    for doc_id, d in docs.items():
        elements = d.get("elements", {}) or {}
        for element_id, e in elements.items():
            fields = [
                _to_text(e.get("caption")),
                _to_text(e.get("content")),
                _to_text(e.get("context_before")),
                _to_text(e.get("context_after")),
            ]
            text = "\n".join([x for x in fields if x]).strip()
            if not text:
                continue
            if len(text) > max_chars:
                text = text[:max_chars]
            chunks.append(Chunk(chunk_id=element_id, doc_id=str(doc_id), text=text))
    return chunks


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


def span_overlap(span: str, text: str) -> float:
    span = (span or "").strip()
    text = (text or "").strip()
    if not span or not text:
        return 0.0
    if span in text:
        return 1.0
    # longest matching block over span length
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


def reciprocal_rank_binary(hit_ranks: List[int]) -> float:
    if not hit_ranks:
        return 0.0
    return 1.0 / min(hit_ranks)


def evaluate_method(
    method: str,
    queries: List[Dict[str, Any]],
    chunks: List[Chunk],
    bm25: BM25Lite,
    doc_hub_prior: Dict[str, float],
    dense_matrix,
    vectorizer,
    top_k: int,
    overlap_threshold: float,
    graph_alpha: float,
) -> Dict[str, Any]:
    import numpy as np

    r_at_10 = 0.0
    mrr = 0.0
    per_query = []

    chunk_tokens = [tokenize(c.text) for c in chunks]

    for q in queries:
        qtxt = (q.get("query") or "").strip()
        spans = query_spans(q)
        if not qtxt or not spans:
            continue

        if method == "bm25":
            q_toks = tokenize(qtxt)
            scored = [(i, bm25.score(q_toks, i)) for i in range(len(chunks))]
        elif method == "dense":
            from sklearn.preprocessing import normalize as _normalize
            qv = _normalize(vectorizer.transform([qtxt]))  # must normalize for cosine sim
            scores = (dense_matrix @ qv.T).toarray().reshape(-1)
            scored = list(enumerate(scores.tolist()))
        elif method == "graph_hub_rerank":
            q_toks = tokenize(qtxt)
            raw_bm25 = [bm25.score(q_toks, i) for i in range(len(chunks))]
            # Normalize BM25 to [0,1] before mixing with hub prior (which is already [0,1]),
            # otherwise graph_alpha has negligible effect on large BM25 scores.
            bm25_min = min(raw_bm25)
            bm25_range = max(raw_bm25) - bm25_min
            scored = []
            for i, c in enumerate(chunks):
                norm_base = (raw_bm25[i] - bm25_min) / max(bm25_range, 1e-9)
                prior = doc_hub_prior.get(c.doc_id, 0.0)
                scored.append((i, norm_base + graph_alpha * prior))
        else:
            raise ValueError(method)

        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        top = ranked[:top_k]

        hit_ranks: List[int] = []
        best_overlap = 0.0
        for rank_idx, (ci, _s) in enumerate(top, start=1):
            ctext = chunks[ci].text
            ov = max(span_overlap(sp, ctext) for sp in spans)
            best_overlap = max(best_overlap, ov)
            if ov >= overlap_threshold:
                hit_ranks.append(rank_idx)

        hit10 = 1.0 if hit_ranks else 0.0
        rr = reciprocal_rank_binary(hit_ranks)
        r_at_10 += hit10
        mrr += rr
        per_query.append(
            {
                "query_id": q.get("query_id"),
                "hit_at_10": hit10,
                "rr": rr,
                "best_overlap_in_topk": round(best_overlap, 4),
            }
        )

    n = max(1, len(per_query))
    return {
        "n": len(per_query),
        "recall_at_10": round(r_at_10 / n, 4),
        "mrr": round(mrr / n, 4),
        "per_query": per_query,
    }


def decision(graph_metrics: Dict[str, Any], bm25_metrics: Dict[str, Any]) -> Dict[str, Any]:
    g_r10 = graph_metrics["recall_at_10"]
    g_mrr = graph_metrics["mrr"]
    b_r10 = bm25_metrics["recall_at_10"]
    b_mrr = bm25_metrics["mrr"]

    delta_r10 = g_r10 - b_r10
    delta_mrr = g_mrr - b_mrr
    continue_expand = (delta_r10 >= 0.05) or (delta_mrr >= 0.03)
    return {
        "delta_recall_at_10_vs_bm25": round(delta_r10, 4),
        "delta_mrr_vs_bm25": round(delta_mrr, 4),
        "continue_expand": continue_expand,
        "rule": "continue if Recall@10 >= BM25+0.05 OR MRR >= BM25+0.03",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Phase-0 locked A/B retrieval evaluation")
    ap.add_argument("--q1", type=Path, default=Path("data/l1_dual_evidence_queries_v4_4_run1_pass.jsonl"))
    ap.add_argument("--q2", type=Path, default=Path("data/l1_dual_evidence_queries_v3_pass.jsonl"))
    ap.add_argument("--q3", type=Path, default=None, help="Optional extra query file (e.g. data111/l1_img_run_20.jsonl)")
    ap.add_argument("--elements", type=Path, default=Path("data111/multimodal_elements_enriched.json"))
    ap.add_argument("--hubs", type=Path, default=Path("data111/latex_graph_hubs (1).json"))
    ap.add_argument("--output", type=Path, default=Path("data/phase0_eval_report.json"))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--overlap-threshold", type=float, default=0.5)
    ap.add_argument("--graph-alpha", type=float, default=0.6)
    ap.add_argument("--max-chars", type=int, default=1800)
    args = ap.parse_args()

    rows = load_jsonl(args.q1) + load_jsonl(args.q2)
    if args.q3 is not None and args.q3.exists():
        rows += load_jsonl(args.q3)
    queries = dedupe_queries(rows)
    chunks = build_chunks(args.elements, max_chars=args.max_chars)
    doc_hub_prior = load_doc_hub_prior(args.hubs)

    if not queries:
        raise SystemExit("No queries loaded")
    if not chunks:
        raise SystemExit("No chunks loaded")

    chunk_tokens = [tokenize(c.text) for c in chunks]
    bm25 = BM25Lite(chunk_tokens)

    # Dense baseline via TF-IDF cosine (deterministic, no remote dependency)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", min_df=1)
    dense_matrix = vectorizer.fit_transform([c.text for c in chunks])
    dense_matrix = normalize(dense_matrix)

    metrics_bm25 = evaluate_method(
        "bm25", queries, chunks, bm25, doc_hub_prior, dense_matrix, vectorizer,
        top_k=args.top_k, overlap_threshold=args.overlap_threshold, graph_alpha=args.graph_alpha,
    )
    metrics_dense = evaluate_method(
        "dense", queries, chunks, bm25, doc_hub_prior, dense_matrix, vectorizer,
        top_k=args.top_k, overlap_threshold=args.overlap_threshold, graph_alpha=args.graph_alpha,
    )
    metrics_graph = evaluate_method(
        "graph_hub_rerank", queries, chunks, bm25, doc_hub_prior, dense_matrix, vectorizer,
        top_k=args.top_k, overlap_threshold=args.overlap_threshold, graph_alpha=args.graph_alpha,
    )

    report = {
        "config": {
            "q1": str(args.q1),
            "q2": str(args.q2),
            "elements": str(args.elements),
            "hubs": str(args.hubs),
            "top_k": args.top_k,
            "overlap_threshold": args.overlap_threshold,
            "graph_alpha": args.graph_alpha,
            "max_chars": args.max_chars,
            "n_queries": len(queries),
            "n_chunks": len(chunks),
        },
        "metrics": {
            "bm25": {k: v for k, v in metrics_bm25.items() if k != "per_query"},
            "dense": {k: v for k, v in metrics_dense.items() if k != "per_query"},
            "graph_hub_rerank": {k: v for k, v in metrics_graph.items() if k != "per_query"},
        },
        "decision": decision(metrics_graph, metrics_bm25),
        "details": {
            "bm25": metrics_bm25["per_query"],
            "dense": metrics_dense["per_query"],
            "graph_hub_rerank": metrics_graph["per_query"],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[phase0] report written:", args.output)
    for name in ["bm25", "dense", "graph_hub_rerank"]:
        m = report["metrics"][name]
        print(f"  {name:16s} n={m['n']}  Recall@10={m['recall_at_10']:.4f}  MRR={m['mrr']:.4f}")
    print("[phase0] decision:", report["decision"])


if __name__ == "__main__":
    main()
