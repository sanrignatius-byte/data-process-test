#!/usr/bin/env python3
"""build_crossdoc_gold57.py
===========================
Build element-level + chunk-level cross-document edges for the gold-57 doc set,
using Qwen3-Embedding-0.6B similarity + BBL citation boost.

Key differences from build_typed_crossdoc_edges.py:
  - Gold-57 filtered only (faster, focused on retrieval eval set)
  - Chunks loaded from paragraph_chunks (enriched text, has element_ids)
    instead of chunk_virtual_nodes_v2 (raw text)
  - Chunk-to-element mapping extracted so edges can be projected to element
    space downstream (for eval_graph_topk_rerank.py typed_crossdoc source)

Output JSON schema:
{
  "metadata": {...},
  "stats": {...},
  "edges": [
    {"source": pid_a, "target": pid_b, "source_doc": ..., "target_doc": ...,
     "type": "figure"|"formula"|"table"|"chunk",
     "similarity": float, "boosted_similarity": float, "cite_boosted": bool}
  ],
  "chunk_contains_element": {chunk_id: [element_id, ...]}   # for downstream projection
}

Usage:
    python scripts/build_crossdoc_gold57.py \\
        --output data/01_graphs/crossdoc_gold57.json

    # CPU-only quick run (no embedding, citation-only):
    python scripts/build_crossdoc_gold57.py \\
        --citation-only --output data/01_graphs/crossdoc_gold57_citation_only.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Data loading ─────────────────────────────────────────────────────────────

def load_gold_docs(qrels_path: Path) -> Set[str]:
    gold = set()
    with open(qrels_path) as f:
        for line in f:
            row = json.loads(line)
            pid = row.get("passage_id", "")
            if pid:
                gold.add(pid.rsplit("_", 2)[0])
    return gold


def load_corpus_typed(corpus_path: Path, types: Set[str],
                       gold_docs: Set[str]) -> Tuple[List[str], List[str], List[str]]:
    pids, docs, texts = [], [], []
    with open(corpus_path) as f:
        for line in f:
            row = json.loads(line)
            t = row.get("type", "")
            doc = row.get("doc_id", "")
            if t not in types or doc not in gold_docs:
                continue
            pids.append(row["passage_id"])
            docs.append(doc)
            texts.append(row.get("text", ""))
    return pids, docs, texts


def load_chunk_nodes_enriched(chunk_graph_path: Path,
                               gold_docs: Set[str]
                               ) -> Tuple[List[str], List[str], List[str], Dict[str, List[str]]]:
    """Load chunk nodes from paragraph_chunks (enriched text). Returns:
       pids, doc_ids, texts, chunk_to_element_ids
    """
    data = json.loads(chunk_graph_path.read_bytes())
    pids, docs, texts = [], [], []
    chunk_to_elems: Dict[str, List[str]] = {}
    for doc_id, doc in data.get("documents", {}).items():
        if doc_id not in gold_docs:
            continue
        nodes = doc.get("nodes", {})
        iterable = nodes.items() if isinstance(nodes, dict) else (
            (n.get("chunk_id", n.get("id", "")), n) for n in nodes)
        for cid, n in iterable:
            if not cid or not isinstance(n, dict):
                continue
            text = n.get("text", "")
            if len(text.strip()) < 30:
                continue
            eids = n.get("element_ids", [])
            pids.append(str(cid))
            docs.append(doc_id)
            texts.append(text)
            chunk_to_elems[str(cid)] = list(eids)
    return pids, docs, texts, chunk_to_elems


def load_citation_adj(citation_path: Path,
                       gold_docs: Set[str]) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = defaultdict(set)
    obj = json.loads(citation_path.read_bytes())
    for e in obj.get("edges", []):
        a, b = str(e.get("source", "")), str(e.get("target", ""))
        if a and b and a != b and a in gold_docs and b in gold_docs:
            adj[a].add(b)
            adj[b].add(a)
    return adj


# ── Edge building ─────────────────────────────────────────────────────────────

def cosine_matrix(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1e-9, norms)
    return (embs / norms) @ (embs / norms).T


def build_edges(pids: List[str], doc_ids: List[str], embs: np.ndarray,
                type_name: str, top_k: int, threshold: float,
                cite_adj: Dict[str, Set[str]], cite_boost: float) -> List[dict]:
    sim = cosine_matrix(embs)
    n = len(pids)
    edges = []
    for i in range(n):
        doc_i = doc_ids[i]
        # collect top-K cross-doc similarities for node i
        scores: List[Tuple[float, int]] = []
        for j in range(n):
            if i == j or doc_ids[j] == doc_i:
                continue
            s = float(sim[i, j])
            if s >= threshold:
                scores.append((s, j))
        scores.sort(reverse=True)
        for raw_sim, j in scores[:top_k]:
            cited = bool(cite_boost > 0 and doc_ids[j] in cite_adj.get(doc_i, set()))
            boosted = raw_sim + (cite_boost if cited else 0.0)
            edges.append({
                "source": pids[i], "target": pids[j],
                "source_doc": doc_i, "target_doc": doc_ids[j],
                "type": type_name,
                "similarity": round(raw_sim, 4),
                "boosted_similarity": round(boosted, 4),
                "cite_boosted": cited,
            })
    return edges


def build_citation_only_edges(pids: List[str], doc_ids: List[str],
                               type_name: str,
                               cite_adj: Dict[str, Set[str]]) -> List[dict]:
    """Build edges only for BBL-cited doc pairs (no embedding)."""
    # group by doc
    doc_to_pids: Dict[str, List[str]] = defaultdict(list)
    for pid, doc in zip(pids, doc_ids):
        doc_to_pids[doc].append(pid)
    edges = []
    seen: Set[Tuple[str, str]] = set()
    for doc_a, nbrs in cite_adj.items():
        for doc_b in nbrs:
            key = (min(doc_a, doc_b), max(doc_a, doc_b))
            if key in seen:
                continue
            seen.add(key)
            for pa in doc_to_pids.get(doc_a, []):
                for pb in doc_to_pids.get(doc_b, []):
                    edges.append({
                        "source": pa, "target": pb,
                        "source_doc": doc_a, "target_doc": doc_b,
                        "type": type_name,
                        "similarity": 0.0,
                        "boosted_similarity": 0.0,
                        "cite_boosted": True,
                    })
    return edges


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qrels", default="data/03_queries/M4query_v1/qrels.jsonl")
    ap.add_argument("--corpus", default="data/03_queries/M4query_v1/corpus.jsonl",
                    help="Enriched element corpus JSONL")
    ap.add_argument("--chunk-graph", default="data/01_graphs/paragraph_chunks_n400_trial57_enriched.json",
                    help="Enriched chunk graph with element_ids (from build_paragraph_chunks.py)")
    ap.add_argument("--citation-graph", default="data/01_graphs/citation_graph.json")
    ap.add_argument("--output", default="data/01_graphs/crossdoc_gold57.json")
    ap.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--download-root", default="models")
    ap.add_argument("--types", nargs="+", default=["figure", "formula", "table"])
    ap.add_argument("--include-chunks", action="store_true",
                    help="Also build chunk-chunk cross-doc edges")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--cite-boost", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--citation-only", action="store_true",
                    help="Skip embedding, build BBL citation edges only (fast CPU check)")
    args = ap.parse_args()

    base = PROJECT_ROOT
    gold_docs = load_gold_docs(base / args.qrels)
    print(f"[info] gold docs: {len(gold_docs)}", flush=True)

    cite_adj = load_citation_adj(base / args.citation_graph, gold_docs)
    print(f"[info] citation pairs in gold set: "
          f"{sum(len(v) for v in cite_adj.values())//2}", flush=True)

    model = None
    if not args.citation_only:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[info] device={device}", flush=True)
        try:
            from modelscope import snapshot_download
            model_dir = snapshot_download(args.model_name, cache_dir=args.download_root)
        except Exception:
            candidates = [
                base / args.download_root / args.model_name,
                base / "models" / args.model_name.replace("/", "___").replace(".", "___"),
                Path(args.download_root) / args.model_name.replace("/", "___").replace(".", "___"),
            ]
            model_dir = None
            for c in candidates:
                if c.exists():
                    model_dir = str(c); break
            if model_dir is None:
                print("[fatal] model not found", file=sys.stderr); sys.exit(2)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(model_dir), device=device)
        # Apply max-length via tokenizer; SentenceTransformer.encode() no longer accepts max_length kwarg
        try:
            model.max_seq_length = args.max_length
            if hasattr(model, "tokenizer") and model.tokenizer is not None:
                model.tokenizer.model_max_length = args.max_length
        except Exception as _e:
            print(f"[warn] could not set max_length={args.max_length}: {_e}", flush=True)
        print(f"[info] model loaded from {model_dir} (max_seq_length={getattr(model, 'max_seq_length', 'n/a')})", flush=True)

    all_edges: List[dict] = []
    type_stats: Dict[str, int] = {}
    chunk_to_elems: Dict[str, List[str]] = {}

    def encode(texts):
        embs = []
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            e = model.encode(batch, batch_size=args.batch_size,
                             show_progress_bar=False, normalize_embeddings=True)
            embs.append(e)
        return np.vstack(embs)

    # Element types
    for typ in args.types:
        pids, docs, texts = load_corpus_typed(base / args.corpus, {typ}, gold_docs)
        if not pids:
            print(f"[warn] {typ}: 0 passages", flush=True); continue
        print(f"[info] {typ}: {len(pids)} passages", flush=True)
        if args.citation_only:
            edges = build_citation_only_edges(pids, docs, typ, cite_adj)
        else:
            t0 = time.time()
            embs = encode(texts)
            print(f"[info] {typ}: encoded in {time.time()-t0:.1f}s", flush=True)
            edges = build_edges(pids, docs, embs, typ,
                                args.top_k, args.threshold, cite_adj, args.cite_boost)
        print(f"[info] {typ}: {len(edges)} edges", flush=True)
        all_edges.extend(edges)
        type_stats[typ] = len(edges)

    # Chunk level
    if args.include_chunks:
        c_pids, c_docs, c_texts, chunk_to_elems = load_chunk_nodes_enriched(
            base / args.chunk_graph, gold_docs)
        print(f"[info] chunks (gold 57): {len(c_pids)}", flush=True)
        if c_pids:
            if args.citation_only:
                edges = build_citation_only_edges(c_pids, c_docs, "chunk", cite_adj)
            else:
                t0 = time.time()
                embs = encode(c_texts)
                print(f"[info] chunks: encoded in {time.time()-t0:.1f}s", flush=True)
                edges = build_edges(c_pids, c_docs, embs, "chunk",
                                    args.top_k, args.threshold, cite_adj, args.cite_boost)
            print(f"[info] chunks: {len(edges)} edges", flush=True)
            all_edges.extend(edges)
            type_stats["chunk"] = len(edges)

    # Stats
    cite_count = sum(1 for e in all_edges if e["cite_boosted"])
    doc_pairs: Set[Tuple[str, str]] = set()
    for e in all_edges:
        a, b = e["source_doc"], e["target_doc"]
        doc_pairs.add((min(a, b), max(a, b)))

    # BBL-only pairs not covered by embedding (cite_adj coverage check)
    all_cite_pairs = set()
    for a, nbrs in cite_adj.items():
        for b in nbrs:
            all_cite_pairs.add((min(a, b), max(a, b)))
    embed_cite_pairs = {(min(e["source_doc"], e["target_doc"]),
                         max(e["source_doc"], e["target_doc"]))
                        for e in all_edges if e["cite_boosted"]}
    missing_bbl = all_cite_pairs - embed_cite_pairs
    print(f"[info] BBL pairs without embedding edge: {len(missing_bbl)} "
          f"(below threshold {args.threshold})", flush=True)

    out = {
        "metadata": {
            "model": args.model_name if not args.citation_only else "citation_only",
            "corpus": str(base / args.corpus),
            "chunk_graph": str(base / args.chunk_graph),
            "gold_docs": len(gold_docs),
            "threshold": args.threshold,
            "top_k": args.top_k,
            "cite_boost": args.cite_boost,
            "types": args.types + (["chunk"] if args.include_chunks else []),
            "citation_only": args.citation_only,
        },
        "stats": {
            "num_edges_total": len(all_edges),
            "num_edges_cite_boosted": cite_count,
            "num_doc_pairs_connected": len(doc_pairs),
            "bbl_pairs_total": len(all_cite_pairs),
            "bbl_pairs_covered_by_embed": len(embed_cite_pairs),
            "bbl_pairs_below_threshold": len(missing_bbl),
            "edges_per_type": type_stats,
        },
        "chunk_contains_element": chunk_to_elems,
        "edges": all_edges,
    }
    out_path = base / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    print(f"[done] wrote {out_path} ({len(all_edges)} edges)", flush=True)


if __name__ == "__main__":
    main()
