#!/usr/bin/env python3
"""Evaluate evidence localization on generated queries.

Compares Document Graph retrieval against BM25 and dense-retrieval baselines.

Metrics:
  - Recall@k  (k = 1, 5, 10)
  - MRR (Mean Reciprocal Rank)
  - Hit@1

Usage:
    # Graph-based (uses hub graph nodes as the retrieval corpus)
    python scripts/evaluate_evidence_localization.py \\
        --queries data/l1_dual_evidence_queries_hub_enriched_v1.jsonl \\
        --graph data/latex_graph_hubs.json \\
        --elements data/multimodal_elements.json \\
        --mode graph \\
        --output data/eval_evidence_localization_graph.json

    # BM25 baseline
    python scripts/evaluate_evidence_localization.py \\
        --queries data/l1_dual_evidence_queries_hub_enriched_v1.jsonl \\
        --elements data/multimodal_elements.json \\
        --mode bm25 \\
        --output data/eval_evidence_localization_bm25.json

    # C-Pool evaluation (universal queries against document graph)
    python scripts/evaluate_evidence_localization.py \\
        --queries data/c_pool_universal_queries.json \\
        --c-pool \\
        --graph data/latex_graph_hubs.json \\
        --elements data/multimodal_elements.json \\
        --mode graph \\
        --output data/eval_cpool_localization.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "from",
    "into", "through", "during", "before", "after", "above", "below",
    "up", "down", "out", "off", "over", "under", "between", "and", "or",
    "but", "if", "because", "as", "until", "while", "although", "since",
    "so", "yet", "both", "either", "neither", "not", "no", "nor",
    "it", "its", "this", "that", "these", "those", "i", "we", "you",
    "he", "she", "they", "what", "which", "who", "how", "when", "where",
    "why", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "than", "then", "too", "very", "just", "only",
}


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[a-z0-9]+", text.lower())
            if t.lower() not in _STOPWORDS and len(t) > 1]


def tf_idf_score(query_tokens: List[str], doc_tokens: List[str],
                 idf: Dict[str, float]) -> float:
    """Simple TF-IDF dot-product score (BM25 approximation)."""
    tf: Dict[str, int] = defaultdict(int)
    for t in doc_tokens:
        tf[t] += 1
    n = max(len(doc_tokens), 1)
    score = 0.0
    for qt in set(query_tokens):
        score += (tf[qt] / n) * idf.get(qt, 0.0)
    return score


def bm25_score(query_tokens: List[str], doc_tokens: List[str],
               idf: Dict[str, float], avg_dl: float,
               k1: float = 1.5, b: float = 0.75) -> float:
    """BM25 scoring."""
    tf: Dict[str, int] = defaultdict(int)
    for t in doc_tokens:
        tf[t] += 1
    dl = len(doc_tokens)
    score = 0.0
    for qt in set(query_tokens):
        f = tf[qt]
        idf_v = idf.get(qt, 0.0)
        score += idf_v * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(avg_dl, 1)))
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Corpus builder
# ─────────────────────────────────────────────────────────────────────────────

def build_corpus(elements_path: Path, graph_path: Optional[Path],
                 mode: str) -> List[Dict[str, Any]]:
    """Build retrieval corpus from elements and optionally the graph.

    Each corpus entry has: id, text, element_type, doc_id.

    mode:
      "bm25"  — all elements in multimodal_elements.json
      "graph" — hub nodes from latex_graph_hubs.json (bridge_hubs +
                adjacent_backbone_bridges) plus their linked elements
    """
    # Load all elements
    all_elements: Dict[str, Dict] = {}
    if elements_path.exists():
        raw = json.loads(elements_path.read_text())
        docs = raw.get("documents", raw) if isinstance(raw, dict) else raw
        if isinstance(docs, dict):
            for doc_id, doc_data in docs.items():
                for elem in doc_data.get("elements", []):
                    eid = elem.get("element_id", "")
                    if eid:
                        all_elements[eid] = {**elem, "doc_id": doc_id}
        elif isinstance(docs, list):
            for elem in docs:
                eid = elem.get("element_id", "")
                if eid:
                    all_elements[eid] = elem

    corpus: List[Dict[str, Any]] = []

    if mode == "bm25":
        for eid, elem in all_elements.items():
            text_parts = [
                elem.get("caption", "") or "",
                elem.get("content", "") or "",
                elem.get("context_before", "") or "",
                elem.get("context_after", "") or "",
            ]
            corpus.append({
                "id": eid,
                "text": " ".join(text_parts),
                "element_type": elem.get("element_type", ""),
                "doc_id": elem.get("doc_id", ""),
            })

    elif mode == "graph":
        if not graph_path or not graph_path.exists():
            print("WARNING: --graph not provided or not found; falling back to BM25 corpus")
            return build_corpus(elements_path, None, "bm25")

        graph_data = json.loads(graph_path.read_text())
        hub_elem_ids: Set[str] = set()

        # Collect element IDs from bridge_hubs
        for hub in graph_data.get("bridge_hubs", []):
            mapped = hub.get("mapped_element_id")
            if mapped:
                hub_elem_ids.add(mapped)

        # Collect element IDs from adjacent_backbone_bridges (candidate pairs)
        for cand in graph_data.get("adjacent_backbone_bridges", []):
            for side in ("element_a_id", "element_b_id"):
                eid = cand.get(side)
                if eid:
                    hub_elem_ids.add(eid)

        # Build corpus from hub elements only (smaller, focused)
        selected_ids = hub_elem_ids if hub_elem_ids else set(all_elements.keys())
        for eid in selected_ids:
            elem = all_elements.get(eid)
            if not elem:
                continue
            text_parts = [
                elem.get("caption", "") or "",
                elem.get("content", "") or "",
                elem.get("context_before", "") or "",
            ]
            corpus.append({
                "id": eid,
                "text": " ".join(text_parts),
                "element_type": elem.get("element_type", ""),
                "doc_id": elem.get("doc_id", ""),
            })

    return corpus


# ─────────────────────────────────────────────────────────────────────────────
# IDF builder
# ─────────────────────────────────────────────────────────────────────────────

def build_idf(corpus: List[Dict[str, Any]]) -> Tuple[Dict[str, float], float]:
    """Compute IDF weights and average document length."""
    df: Dict[str, int] = defaultdict(int)
    total_dl = 0
    n = len(corpus)
    for doc in corpus:
        tokens = tokenize(doc["text"])
        total_dl += len(tokens)
        for t in set(tokens):
            df[t] += 1
    avg_dl = total_dl / max(n, 1)
    idf = {t: math.log((n - freq + 0.5) / (freq + 0.5) + 1)
           for t, freq in df.items()}
    return idf, avg_dl


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_bm25(query: str, corpus: List[Dict[str, Any]],
                  idf: Dict[str, float], avg_dl: float,
                  top_k: int = 10) -> List[Tuple[str, float]]:
    """Return top-k (element_id, score) by BM25."""
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    scored = []
    for doc in corpus:
        tokens = tokenize(doc["text"])
        s = bm25_score(q_tokens, tokens, idf, avg_dl)
        if s > 0:
            scored.append((doc["id"], s))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    hits = sum(1 for r in retrieved_ids[:k] if r in relevant_ids)
    return hits / max(len(relevant_ids), 1)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    for rank, r in enumerate(retrieved_ids, start=1):
        if r in relevant_ids:
            return 1.0 / rank
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Query loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_generated_queries(path: Path) -> List[Dict[str, Any]]:
    """Load queries from a .jsonl file (generate_multihop_l1_queries output)."""
    queries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    queries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return queries


def load_cpool_queries(path: Path) -> List[Dict[str, Any]]:
    """Load queries from C-Pool JSON file."""
    data = json.loads(path.read_text())
    return data.get("queries", [])


def get_relevant_ids_from_query(q: Dict[str, Any]) -> Set[str]:
    """Extract the ground-truth relevant element IDs from a query record."""
    relevant: Set[str] = set()

    # From element_ids (direct pair elements)
    for eid in q.get("element_ids", []):
        if eid:
            relevant.add(str(eid))

    # From required_evidence_spans
    for span in q.get("required_evidence_spans", []) or []:
        if isinstance(span, dict):
            eid = span.get("element_id", "")
            if eid:
                relevant.add(str(eid))

    return relevant


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(queries: List[Dict[str, Any]], corpus: List[Dict[str, Any]],
             idf: Dict[str, float], avg_dl: float,
             ks: List[int] = None) -> Dict[str, Any]:
    if ks is None:
        ks = [1, 5, 10]

    total = 0
    recall_sums: Dict[int, float] = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    hit1_sum = 0.0
    per_query: List[Dict[str, Any]] = []
    skipped = 0

    for q in queries:
        query_text = q.get("query", "")
        relevant = get_relevant_ids_from_query(q)
        if not query_text or not relevant:
            skipped += 1
            continue

        retrieved = retrieve_bm25(query_text, corpus, idf, avg_dl, top_k=max(ks))
        retrieved_ids = [r[0] for r in retrieved]

        rr = reciprocal_rank(retrieved_ids, relevant)
        mrr_sum += rr
        hit1_sum += 1.0 if (retrieved_ids and retrieved_ids[0] in relevant) else 0.0

        q_recalls = {}
        for k in ks:
            r_k = recall_at_k(retrieved_ids, relevant, k)
            recall_sums[k] += r_k
            q_recalls[f"recall@{k}"] = round(r_k, 4)

        per_query.append({
            "query_id": q.get("query_id", q.get("id", "")),
            "query": query_text[:120],
            "relevant_ids": sorted(relevant),
            "retrieved_top5": retrieved_ids[:5],
            "rr": round(rr, 4),
            **q_recalls,
        })
        total += 1

    if total == 0:
        return {"error": "no evaluable queries", "skipped": skipped}

    result: Dict[str, Any] = {
        "total_queries": total,
        "skipped_queries": skipped,
        "mrr": round(mrr_sum / total, 4),
        "hit@1": round(hit1_sum / total, 4),
    }
    for k in ks:
        result[f"recall@{k}"] = round(recall_sums[k] / total, 4)

    result["per_query"] = per_query
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate evidence localization: Document Graph vs BM25 baseline."
    )
    ap.add_argument("--queries", required=True,
                    help="Path to .jsonl query file or C-Pool JSON")
    ap.add_argument("--elements", required=True,
                    help="Path to multimodal_elements.json")
    ap.add_argument("--graph", default=None,
                    help="Path to latex_graph_hubs.json (required for --mode graph)")
    ap.add_argument("--mode", choices=["bm25", "graph"], default="bm25",
                    help="Retrieval mode: 'bm25' (all elements) or 'graph' (hub nodes only)")
    ap.add_argument("--output", default=None,
                    help="Output JSON path (default: auto-named)")
    ap.add_argument("--c-pool", action="store_true",
                    help="Input is a C-Pool JSON file instead of .jsonl")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10],
                    help="k values for Recall@k (default: 1 5 10)")
    ap.add_argument("--pass-only", action="store_true",
                    help="Only evaluate queries where qc_pass=true")
    args = ap.parse_args()

    queries_path = Path(args.queries)
    elements_path = Path(args.elements)
    graph_path = Path(args.graph) if args.graph else None

    print(f"Loading queries from: {queries_path}")
    if args.c_pool:
        queries = load_cpool_queries(queries_path)
    else:
        queries = load_generated_queries(queries_path)

    if args.pass_only:
        before = len(queries)
        queries = [q for q in queries if q.get("qc_pass", True)]
        print(f"  Filtered to qc_pass=true: {len(queries)}/{before}")

    print(f"Loaded {len(queries)} queries")

    print(f"Building corpus (mode={args.mode}) ...")
    corpus = build_corpus(elements_path, graph_path, args.mode)
    print(f"  Corpus size: {len(corpus)} documents")

    print("Building IDF ...")
    idf, avg_dl = build_idf(corpus)
    print(f"  Vocabulary size: {len(idf)}, avg doc length: {avg_dl:.1f} tokens")

    print("Running evaluation ...")
    results = evaluate(queries, corpus, idf, avg_dl, ks=args.ks)

    # Print summary
    print("\n── Evaluation Results ──────────────────────────────")
    print(f"  Mode:         {args.mode}")
    print(f"  Total queries: {results.get('total_queries', 0)}")
    print(f"  Skipped:       {results.get('skipped_queries', 0)}")
    print(f"  MRR:           {results.get('mrr', 0):.4f}")
    print(f"  Hit@1:         {results.get('hit@1', 0):.4f}")
    for k in args.ks:
        print(f"  Recall@{k}:     {results.get(f'recall@{k}', 0):.4f}")
    print("────────────────────────────────────────────────────")

    # Save output
    if args.output:
        out_path = Path(args.output)
    else:
        stem = queries_path.stem
        out_path = queries_path.parent / f"eval_evidence_loc_{stem}_{args.mode}.json"

    out_data = {
        "mode": args.mode,
        "queries_file": str(queries_path),
        "corpus_size": len(corpus),
        "metrics": {k: v for k, v in results.items() if k != "per_query"},
        "per_query": results.get("per_query", []),
    }
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
