#!/usr/bin/env python3
"""Evaluate C-Pool universal queries: keyword_boost + graph vs BM25 baseline.

Compares retrieval quality on the merged C-Pool (78 queries, 30 proxy-supported)
across multiple configurations:

  A. BM25 baseline (no keyword_boost, old hubs)
  B. BM25 baseline (keyword_boost enriched corpus, keyword_boost hubs)
  C. graph_neighbor_prop (keyword_boost hubs)
  D. graph_hub_rerank (keyword_boost hubs)
  E. graph_full (keyword_boost hubs, tuned weights: hw=0.15, nw=1.00, cw=0)

This script reuses the eval infrastructure from run_phase0_eval_ab.py.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.text_utils import tokenize_for_retrieval as tokenize
from src.retrieval import BM25Lite

# TOKEN_RE, tokenize(), BM25Lite moved to src.utils.text_utils / src.retrieval


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _to_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v) if isinstance(v, (int, float, bool)) else ""


def build_chunks(
    elements_json: Path,
    max_chars: int = 1800,
    section_enrich_path: Optional[Path] = None,
) -> List[Chunk]:
    data = json.loads(elements_json.read_text(encoding="utf-8"))
    docs = data.get("documents", {}) or {}
    chunks: List[Chunk] = []
    for doc_id, d in docs.items():
        elements = d.get("elements", {}) or {}
        for element_id, e in elements.items():
            fields = [
                _to_text(e.get("caption")),
                _to_text(e.get("content")),
                " ".join(filter(None, [_to_text(e.get("context_before")),
                                       _to_text(e.get("context_after"))])),
                _to_text(e.get("enriched_title")),
                _to_text(e.get("enriched_content")),
            ]
            text = "\n".join(x for x in fields if x).strip()
            if not text:
                continue
            if len(text) > max_chars:
                text = text[:max_chars]
            chunks.append(Chunk(chunk_id=element_id, doc_id=str(doc_id), text=text))

    # Optionally load section enrichment as additional chunks
    if section_enrich_path and section_enrich_path.exists():
        sec_data = json.loads(section_enrich_path.read_text(encoding="utf-8"))
        sec_rows = sec_data.get("sections", []) if isinstance(sec_data, dict) else []
        n_sec = 0
        for row in sec_rows:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("section_id", "") or "").strip()
            doc_id = str(row.get("doc_id", "") or "").strip()
            if not sid or not doc_id:
                continue
            sec_title = _to_text(row.get("enriched_title")) or _to_text(row.get("section_title"))
            sec_content = _to_text(row.get("enriched_content"))
            sec_metadata = row.get("enriched_metadata") or {}
            sec_keywords = sec_metadata.get("keywords", [])
            kw_str = ", ".join(str(k) for k in sec_keywords[:8]) if sec_keywords else ""
            fields = [sec_title, kw_str, sec_content]
            text = "\n".join(x for x in fields if x).strip()
            if not text:
                continue
            if len(text) > max_chars:
                text = text[:max_chars]
            chunks.append(Chunk(chunk_id=sid, doc_id=doc_id, text=text))
            n_sec += 1
        if n_sec:
            print(f"  Loaded {n_sec} section chunks from {section_enrich_path.name}")

    return chunks


def load_element_hub_prior(hub_candidates_json: Path) -> Dict[str, float]:
    if not hub_candidates_json.exists():
        return {}
    obj = json.loads(hub_candidates_json.read_text(encoding="utf-8"))
    pairs = obj.get("pairs", []) or []
    prior: Dict[str, float] = {}
    for p in pairs:
        hs = float(p.get("hub_score", 0))
        for eid in (p.get("element_a_id"), p.get("element_b_id")):
            if eid:
                prior[eid] = max(prior.get(eid, 0.0), hs)
    # Normalize to [0,1]
    if prior:
        mx = max(prior.values())
        if mx > 1e-9:
            prior = {k: v / mx for k, v in prior.items()}
    return prior


def load_element_adjacency(
    hub_candidates_json: Path,
) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, float]]]:
    if not hub_candidates_json.exists():
        return {}, {}
    obj = json.loads(hub_candidates_json.read_text(encoding="utf-8"))
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    weights: Dict[str, Dict[str, float]] = defaultdict(dict)
    for p in obj.get("pairs", []):
        a = str(p.get("element_a_id", "")).strip()
        b = str(p.get("element_b_id", "")).strip()
        if a and b:
            w = float(p.get("pair_score", p.get("hub_score", 1.0)))
            adjacency[a].add(b)
            adjacency[b].add(a)
            weights[a][b] = w
            weights[b][a] = w
    # Also load adjacent_bridge_adjacency if present
    for adj in obj.get("adjacent_bridge_adjacency", []):
        a = str(adj.get("element_a_id", "")).strip()
        b = str(adj.get("element_b_id", "")).strip()
        if a and b:
            w = float(adj.get("bridge_score", 0.5))
            adjacency[a].add(b)
            adjacency[b].add(a)
            weights[a].setdefault(b, w)
            weights[b].setdefault(a, w)
    return dict(adjacency), dict(weights)


def load_citation_adjacency(citation_json: Path) -> Dict[str, Dict[str, Set[str]]]:
    if not citation_json.exists():
        return {}
    data = json.loads(citation_json.read_text(encoding="utf-8"))
    edges = data.get("edges", [])
    adj: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for e in edges:
        src = str(e.get("source", e.get("from", ""))).strip()
        tgt = str(e.get("target", e.get("to", ""))).strip()
        if src and tgt:
            adj[src]["cites"].add(tgt)
            adj[tgt]["cited_by"].add(src)
    return {k: dict(v) for k, v in adj.items()}


def query_element_ids(q: Dict[str, Any]) -> List[str]:
    """Extract ground-truth element IDs from query."""
    spans = q.get("required_evidence_spans", [])
    ids = []
    for sp in spans:
        eid = str(sp.get("element_id", "")).strip()
        if eid:
            ids.append(eid)
    return ids


def _bm25_norm_scores(bm25: BM25Lite, q_toks: List[str], n: int) -> List[float]:
    raw = [bm25.score(q_toks, i) for i in range(n)]
    lo, hi = min(raw), max(raw)
    rng = hi - lo
    if rng < 1e-9:
        return [0.0] * n
    return [(x - lo) / rng for x in raw]


def coverage_at_k(retrieved: List[str], gt: Set[str], k: int) -> float:
    if not gt:
        return 0.0
    hits = sum(1 for eid in retrieved[:k] if eid in gt)
    return hits / len(gt)


def run_eval(
    method: str,
    queries: List[Dict[str, Any]],
    chunks: List[Chunk],
    bm25: BM25Lite,
    chunk_id_to_idx: Dict[str, int],
    element_hub_prior: Dict[str, float],
    element_adjacency: Dict[str, Set[str]],
    element_adj_weights: Dict[str, Dict[str, float]],
    citation_adjacency: Dict[str, Dict[str, Set[str]]],
    top_k: int = 10,
    graph_alpha: float = 0.2,
    neighbor_decay: float = 0.5,
    citation_decay: float = 0.3,
    rerank_topn: int = 100,
    hub_weight: float | None = None,
    nprop_weight: float | None = None,
    cite_weight: float | None = None,
) -> Dict[str, Any]:
    """Run a single retrieval method on C-Pool queries."""
    n_chunks = len(chunks)
    recall_sums = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0, 20: 0.0}
    mrr = 0.0
    per_query = []

    for q in queries:
        qtxt = (q.get("query") or "").strip()
        gt_eids = set(query_element_ids(q))
        if not qtxt or not gt_eids:
            continue

        q_toks = tokenize(qtxt)

        if method == "bm25":
            scored = [(i, bm25.score(q_toks, i)) for i in range(n_chunks)]

        elif method == "graph_hub_rerank":
            norm_bm25 = _bm25_norm_scores(bm25, q_toks, n_chunks)
            scored = []
            for i in range(n_chunks):
                prior = element_hub_prior.get(chunks[i].chunk_id, 0.0)
                scored.append((i, norm_bm25[i] + graph_alpha * prior))

        elif method == "graph_neighbor_prop":
            norm_bm25 = _bm25_norm_scores(bm25, q_toks, n_chunks)
            ranked = sorted(range(n_chunks), key=lambda i: norm_bm25[i], reverse=True)
            rerank_n = max(top_k, min(rerank_topn, n_chunks))
            neighbor_boost = [0.0] * n_chunks
            for seed_idx in ranked[:rerank_n]:
                seed_id = chunks[seed_idx].chunk_id
                seed_score = norm_bm25[seed_idx]
                seed_w = element_adj_weights.get(seed_id, {})
                for nbr_id in element_adjacency.get(seed_id, set()):
                    edge_w = seed_w.get(nbr_id, 1.0)
                    nbr_idx = chunk_id_to_idx.get(nbr_id)
                    if nbr_idx is not None:
                        neighbor_boost[nbr_idx] = min(
                            neighbor_boost[nbr_idx] + seed_score * neighbor_decay * edge_w,
                            neighbor_decay,
                        )
            scored = [(i, norm_bm25[i] + neighbor_boost[i]) for i in range(n_chunks)]

        elif method == "graph_full":
            norm_bm25 = _bm25_norm_scores(bm25, q_toks, n_chunks)
            ranked = sorted(range(n_chunks), key=lambda i: norm_bm25[i], reverse=True)
            rerank_n = max(top_k, min(rerank_topn, n_chunks))

            _hw = hub_weight if hub_weight is not None else graph_alpha
            _nw = nprop_weight if nprop_weight is not None else 1.0
            _cw = cite_weight if cite_weight is not None else citation_decay

            # hub boost
            hub_boost = [0.0] * n_chunks
            if _hw > 1e-9:
                for i in range(n_chunks):
                    hub_boost[i] = element_hub_prior.get(chunks[i].chunk_id, 0.0)

            # neighbor prop
            neighbor_boost = [0.0] * n_chunks
            if _nw > 1e-9:
                for seed_idx in ranked[:rerank_n]:
                    seed_id = chunks[seed_idx].chunk_id
                    seed_score = norm_bm25[seed_idx]
                    seed_w = element_adj_weights.get(seed_id, {})
                    for nbr_id in element_adjacency.get(seed_id, set()):
                        edge_w = seed_w.get(nbr_id, 1.0)
                        nbr_idx = chunk_id_to_idx.get(nbr_id)
                        if nbr_idx is not None:
                            neighbor_boost[nbr_idx] = min(
                                neighbor_boost[nbr_idx] + seed_score * neighbor_decay * edge_w,
                                neighbor_decay,
                            )

            # citation walk
            cite_boost: Dict[str, float] = defaultdict(float)
            if _cw > 1e-9:
                doc_score: Dict[str, float] = defaultdict(float)
                for i, c in enumerate(chunks):
                    doc_score[c.doc_id] = max(doc_score[c.doc_id], norm_bm25[i])
                for doc_id, ds in doc_score.items():
                    if ds < 0.3:
                        continue
                    adj = citation_adjacency.get(doc_id, {})
                    for nbr in adj.get("cites", set()):
                        cite_boost[nbr] += ds * citation_decay
                    for nbr in adj.get("cited_by", set()):
                        cite_boost[nbr] += ds * citation_decay * 0.5
                if cite_boost:
                    mx = max(cite_boost.values())
                    if mx > 1e-9:
                        cite_boost = {k: v / mx for k, v in cite_boost.items()}

            scored = []
            for i, c in enumerate(chunks):
                s = (
                    norm_bm25[i]
                    + _hw * hub_boost[i]
                    + _nw * neighbor_boost[i]
                    + _cw * cite_boost.get(c.doc_id, 0.0)
                )
                scored.append((i, s))

        else:
            raise ValueError(f"Unknown method: {method}")

        # Rank and compute metrics
        ranked_all = sorted(scored, key=lambda x: x[1], reverse=True)
        retrieved_ids = [chunks[ci].chunk_id for ci, _ in ranked_all]

        # MRR
        rr = 0.0
        for rank_idx, eid in enumerate(retrieved_ids[:top_k], start=1):
            if eid in gt_eids:
                rr = 1.0 / rank_idx
                break
        mrr += rr

        for k in (1, 3, 5, 10, 20):
            recall_sums[k] += coverage_at_k(retrieved_ids, gt_eids, k)

        per_query.append({
            "query_id": q.get("query_id"),
            "query": qtxt[:80],
            "category": q.get("category", ""),
            "rr": round(rr, 4),
            "recall_at_10": round(coverage_at_k(retrieved_ids, gt_eids, 10), 4),
            "n_gt": len(gt_eids),
        })

    n = max(1, len(per_query))
    return {
        "n": len(per_query),
        "recall_at_1": round(recall_sums[1] / n, 4),
        "recall_at_3": round(recall_sums[3] / n, 4),
        "recall_at_5": round(recall_sums[5] / n, 4),
        "recall_at_10": round(recall_sums[10] / n, 4),
        "recall_at_20": round(recall_sums[20] / n, 4),
        "mrr": round(mrr / n, 4),
        "per_query": per_query,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="C-Pool keyword_boost + graph evaluation")
    ap.add_argument("--queries", type=Path,
                    default=Path("data/05_eval/m2/c_pool_universal_queries_proxy_merged.jsonl"))
    ap.add_argument("--elements", type=Path,
                    default=Path("data/02_enriched/multimodal_elements_enriched.json"))
    ap.add_argument("--hub-candidates-old", type=Path,
                    default=Path("data/02_enriched/hub_candidates_enriched_v3.json"),
                    help="Original hubs (no keyword_boost)")
    ap.add_argument("--hub-candidates-kb", type=Path,
                    default=Path("data/05_eval/m2/hub_candidates_enriched_keyword_boost_full_2026-03-24.json"),
                    help="keyword_boost enhanced hubs")
    ap.add_argument("--citation-graph", type=Path,
                    default=Path("data/01_graphs/citation_graph.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("data/05_eval/m2/_eval_cpool_merged_keyword_boost_graph.json"))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--section-enrich", type=Path, default=None,
                    help="Optional section enrichment JSON (section_nodes_enriched.json). "
                         "Section nodes are added as retrieval chunks to the corpus.")
    args = ap.parse_args()

    # Load queries
    queries = load_jsonl(args.queries)
    print(f"Loaded {len(queries)} proxy C-Pool queries")

    # Load chunks
    chunks = build_chunks(args.elements, section_enrich_path=args.section_enrich)
    chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(chunks)}
    chunk_tokens = [tokenize(c.text) for c in chunks]
    bm25 = BM25Lite(chunk_tokens)
    print(f"Built BM25 index: {len(chunks)} chunks")

    # Load old hubs (no keyword_boost)
    elem_prior_old = load_element_hub_prior(args.hub_candidates_old)
    elem_adj_old, elem_adj_w_old = load_element_adjacency(args.hub_candidates_old)
    print(f"Old hubs: {len(elem_prior_old)} priors, "
          f"{sum(len(v) for v in elem_adj_old.values())//2} edges")

    # Load keyword_boost hubs
    elem_prior_kb = load_element_hub_prior(args.hub_candidates_kb)
    elem_adj_kb, elem_adj_w_kb = load_element_adjacency(args.hub_candidates_kb)
    print(f"KB hubs: {len(elem_prior_kb)} priors, "
          f"{sum(len(v) for v in elem_adj_kb.values())//2} edges")

    # Load citation graph
    cite_adj = load_citation_adjacency(args.citation_graph)
    print(f"Citation graph: {len(cite_adj)} docs")

    # Common eval kwargs
    common = dict(
        queries=queries, chunks=chunks, bm25=bm25,
        chunk_id_to_idx=chunk_id_to_idx, top_k=args.top_k,
        citation_adjacency=cite_adj,
    )

    results = {}

    # === A. BM25 baseline ===
    print("\n[A] BM25 baseline...")
    results["bm25"] = run_eval(
        method="bm25",
        element_hub_prior={}, element_adjacency={}, element_adj_weights={},
        **common,
    )

    # === B. graph_neighbor_prop (old hubs, no keyword_boost) ===
    print("[B] graph_neighbor_prop (old hubs)...")
    results["neighbor_prop_old"] = run_eval(
        method="graph_neighbor_prop",
        element_hub_prior=elem_prior_old,
        element_adjacency=elem_adj_old,
        element_adj_weights=elem_adj_w_old,
        **common,
    )

    # === C. graph_neighbor_prop (keyword_boost hubs) ===
    print("[C] graph_neighbor_prop (keyword_boost)...")
    results["neighbor_prop_kb"] = run_eval(
        method="graph_neighbor_prop",
        element_hub_prior=elem_prior_kb,
        element_adjacency=elem_adj_kb,
        element_adj_weights=elem_adj_w_kb,
        **common,
    )

    # === D. graph_hub_rerank (old hubs) ===
    print("[D] graph_hub_rerank (old hubs)...")
    results["hub_rerank_old"] = run_eval(
        method="graph_hub_rerank",
        element_hub_prior=elem_prior_old,
        element_adjacency=elem_adj_old,
        element_adj_weights=elem_adj_w_old,
        **common,
    )

    # === E. graph_hub_rerank (keyword_boost hubs) ===
    print("[E] graph_hub_rerank (keyword_boost)...")
    results["hub_rerank_kb"] = run_eval(
        method="graph_hub_rerank",
        element_hub_prior=elem_prior_kb,
        element_adjacency=elem_adj_kb,
        element_adj_weights=elem_adj_w_kb,
        **common,
    )

    # === F. graph_full (keyword_boost, tuned weights) ===
    print("[F] graph_full (keyword_boost, hw=0.15 nw=1.0 cw=0)...")
    results["graph_full_kb_tuned"] = run_eval(
        method="graph_full",
        element_hub_prior=elem_prior_kb,
        element_adjacency=elem_adj_kb,
        element_adj_weights=elem_adj_w_kb,
        hub_weight=0.15, nprop_weight=1.0, cite_weight=0.0,
        **common,
    )

    # === G. graph_full (old hubs, default weights) ===
    print("[G] graph_full (old hubs, default)...")
    results["graph_full_old"] = run_eval(
        method="graph_full",
        element_hub_prior=elem_prior_old,
        element_adjacency=elem_adj_old,
        element_adj_weights=elem_adj_w_old,
        **common,
    )

    # === Summary ===
    print("\n" + "=" * 80)
    print("C-Pool Merged (78 queries, 30 proxy-supported) Evaluation Results")
    print("=" * 80)
    bm25_r10 = results["bm25"]["recall_at_10"]
    bm25_mrr = results["bm25"]["mrr"]
    header = f"{'Method':<35} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@20':>6} {'MRR':>6} {'ΔR@10':>7} {'ΔMRR':>7}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        dr10 = r["recall_at_10"] - bm25_r10
        dmrr = r["mrr"] - bm25_mrr
        mark_r = "↑" if dr10 > 0.001 else ("↓" if dr10 < -0.001 else " ")
        mark_m = "↑" if dmrr > 0.001 else ("↓" if dmrr < -0.001 else " ")
        print(f"{name:<35} {r['recall_at_1']:>6.4f} {r['recall_at_5']:>6.4f} "
              f"{r['recall_at_10']:>6.4f} {r['recall_at_20']:>6.4f} {r['mrr']:>6.4f} "
              f"{dr10:>+7.4f}{mark_r} {dmrr:>+7.4f}{mark_m}")

    # Per-category breakdown
    print("\n--- Per-category breakdown (R@10 / MRR) ---")
    cats = sorted(set(pq["category"] for pq in results["bm25"]["per_query"] if pq.get("category")))
    cat_header = f"{'Method':<35} " + " ".join(f"{c:>22}" for c in cats)
    print(cat_header)
    for name, r in results.items():
        cat_metrics: Dict[str, List[float]] = {}
        for pq in r["per_query"]:
            cat = pq.get("category", "")
            if cat not in cat_metrics:
                cat_metrics[cat] = [0.0, 0.0, 0]
            cat_metrics[cat][0] += pq["recall_at_10"]
            cat_metrics[cat][1] += pq["rr"]
            cat_metrics[cat][2] += 1
        parts = []
        for c in cats:
            if c in cat_metrics and cat_metrics[c][2] > 0:
                n_c = cat_metrics[c][2]
                r10 = cat_metrics[c][0] / n_c
                m = cat_metrics[c][1] / n_c
                parts.append(f"R@10={r10:.3f} MRR={m:.3f}")
            else:
                parts.append("—")
        print(f"{name:<35} " + " ".join(f"{p:>22}" for p in parts))

    # Decision gate
    best_graph = max(
        [(n, r) for n, r in results.items() if n != "bm25"],
        key=lambda x: x[1]["mrr"],
    )
    best_name, best_r = best_graph
    decision = {
        "best_graph_method": best_name,
        "delta_recall_at_10_vs_bm25": round(best_r["recall_at_10"] - bm25_r10, 4),
        "delta_mrr_vs_bm25": round(best_r["mrr"] - bm25_mrr, 4),
        "continue_expand": (
            (best_r["recall_at_10"] - bm25_r10) >= 0.05
            or (best_r["mrr"] - bm25_mrr) >= 0.03
        ),
        "rule": "continue if R@10 >= BM25+0.05 OR MRR >= BM25+0.03",
    }
    print(f"\n--- Decision ---")
    print(f"Best graph method: {best_name}")
    print(f"ΔR@10 vs BM25: {decision['delta_recall_at_10_vs_bm25']:+.4f}")
    print(f"ΔMRR vs BM25: {decision['delta_mrr_vs_bm25']:+.4f}")
    print(f"Continue expand: {decision['continue_expand']}")

    # Save
    # Strip per_query for compact output
    compact_results = {}
    for name, r in results.items():
        compact_results[name] = {k: v for k, v in r.items() if k != "per_query"}

    report = {
        "config": {
            "queries_file": str(args.queries),
            "elements_file": str(args.elements),
            "hub_candidates_old": str(args.hub_candidates_old),
            "hub_candidates_kb": str(args.hub_candidates_kb),
            "n_queries": len(queries),
            "n_chunks": len(chunks),
            "n_hub_prior_old": len(elem_prior_old),
            "n_hub_prior_kb": len(elem_prior_kb),
            "n_adj_edges_old": sum(len(v) for v in elem_adj_old.values()) // 2,
            "n_adj_edges_kb": sum(len(v) for v in elem_adj_kb.values()) // 2,
        },
        "metrics": compact_results,
        "decision": decision,
        "keyword_boost_impact": {
            "neighbor_prop": {
                "old_r10": results["neighbor_prop_old"]["recall_at_10"],
                "kb_r10": results["neighbor_prop_kb"]["recall_at_10"],
                "delta_r10": round(results["neighbor_prop_kb"]["recall_at_10"] - results["neighbor_prop_old"]["recall_at_10"], 4),
                "old_mrr": results["neighbor_prop_old"]["mrr"],
                "kb_mrr": results["neighbor_prop_kb"]["mrr"],
                "delta_mrr": round(results["neighbor_prop_kb"]["mrr"] - results["neighbor_prop_old"]["mrr"], 4),
            },
            "hub_rerank": {
                "old_r10": results["hub_rerank_old"]["recall_at_10"],
                "kb_r10": results["hub_rerank_kb"]["recall_at_10"],
                "delta_r10": round(results["hub_rerank_kb"]["recall_at_10"] - results["hub_rerank_old"]["recall_at_10"], 4),
                "old_mrr": results["hub_rerank_old"]["mrr"],
                "kb_mrr": results["hub_rerank_kb"]["mrr"],
                "delta_mrr": round(results["hub_rerank_kb"]["mrr"] - results["hub_rerank_old"]["mrr"], 4),
            },
            "graph_full": {
                "old_r10": results["graph_full_old"]["recall_at_10"],
                "kb_tuned_r10": results["graph_full_kb_tuned"]["recall_at_10"],
                "delta_r10": round(results["graph_full_kb_tuned"]["recall_at_10"] - results["graph_full_old"]["recall_at_10"], 4),
                "old_mrr": results["graph_full_old"]["mrr"],
                "kb_tuned_mrr": results["graph_full_kb_tuned"]["mrr"],
                "delta_mrr": round(results["graph_full_kb_tuned"]["mrr"] - results["graph_full_old"]["mrr"], 4),
            },
        },
        "per_query_details": {name: r["per_query"] for name, r in results.items()},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
