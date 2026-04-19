#!/usr/bin/env python3
"""Build typed cross-document similarity edges using passage-level embeddings.

For each type in {figure, formula, table, chunk}, compute cross-doc pairwise
cosine similarity and keep top-K edges per node above threshold. Optionally
boost by bbl citation graph (if src_doc cites tgt_doc or vice-versa).

Output schema (pid-level, simpler than section-level cross_doc_sim_edges.json):
{
  "metadata": {...},
  "edges": [
    {"source": pid_a, "target": pid_b, "type": "figure",
     "similarity": 0.82, "cite_boosted": true/false, "boosted_sim": 0.92}
  ],
  "stats": {...}
}
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


def load_jsonl(path: Path) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_corpus_typed(corpus_path: Path, types: Set[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
    pids, docs, texts, tps = [], [], [], []
    for row in load_jsonl(corpus_path):
        t = row.get("type", "")
        if t not in types:
            continue
        pids.append(str(row["passage_id"]))
        docs.append(str(row.get("doc_id", "")))
        texts.append(str(row.get("text", "")))
        tps.append(t)
    return pids, docs, texts, tps


def load_chunk_nodes(chunk_graph_path: Path) -> Tuple[List[str], List[str], List[str]]:
    """Return chunk_id, doc_id, text for every chunk node in v2 graph."""
    data = json.loads(chunk_graph_path.read_bytes())
    pids, docs, texts = [], [], []
    for doc_id, doc in data.get("documents", {}).items():
        nodes = doc.get("nodes") or {}
        iterable = nodes.items() if isinstance(nodes, dict) else ((n.get("id") or n.get("chunk_id"), n) for n in nodes)
        for cid, n in iterable:
            if not cid:
                continue
            if not isinstance(n, dict):
                continue
            # chunk nodes have 'chunk_id' field in dict form
            if "chunk_id" not in n and n.get("type") != "chunk":
                continue
            text = n.get("text") or n.get("content") or ""
            if len(text.strip()) < 30:
                continue
            pids.append(str(cid))
            docs.append(doc_id)
            texts.append(text)
    return pids, docs, texts


def load_citation_adj(citation_path: Path) -> Dict[str, Set[str]]:
    obj = json.loads(citation_path.read_bytes())
    adj: Dict[str, Set[str]] = defaultdict(set)
    for e in obj.get("edges", []) or []:
        a = str(e.get("source", ""))
        b = str(e.get("target", ""))
        if a and b and a != b:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def encode_texts(model, texts: List[str], batch_size: int, is_query: bool, max_length: int) -> np.ndarray:
    prompt = None
    if is_query:
        prompt = "Retrieve relevant passages: "
    embs = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]
        if prompt:
            batch = [prompt + t for t in batch]
        vec = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )
        embs.append(vec)
    return np.vstack(embs) if embs else np.zeros((0, 1))


def build_edges_for_type(
    pids: List[str],
    docs: List[str],
    embs: np.ndarray,
    type_name: str,
    top_k: int,
    threshold: float,
    cite_adj: Dict[str, Set[str]],
    cite_boost: float,
) -> List[dict]:
    """Top-K per node, cross-doc only, above threshold."""
    if len(pids) == 0:
        return []
    print(f"[info] type={type_name}: {len(pids)} nodes, computing similarities...", flush=True)
    n = len(pids)
    # Compute full sim matrix (2809 max, fine in memory: 2809*2809*4 = 31MB)
    sim = embs @ embs.T  # (n, n)
    # Build edges
    edges: List[dict] = []
    doc_arr = np.array(docs)
    for i in range(n):
        doc_i = docs[i]
        # Mask: only cross-doc (doc != doc_i) and above threshold, skip self
        row = sim[i].copy()
        row[i] = -1.0
        # Apply cite boost before top-k selection
        boosted = row.copy()
        if cite_boost > 0 and doc_i in cite_adj:
            cited = cite_adj[doc_i]
            if cited:
                cite_mask = np.array([(d in cited) for d in docs])
                boosted = np.where(cite_mask, row + cite_boost, row)
        # Mask out same-doc
        same_doc = (doc_arr == doc_i)
        boosted = np.where(same_doc, -1.0, boosted)
        # Threshold on raw sim (not boosted) to avoid low-quality matches
        valid_mask = (row >= threshold) & (~same_doc)
        if not valid_mask.any():
            continue
        # Select top-k by boosted score
        boosted_valid = np.where(valid_mask, boosted, -1.0)
        if top_k >= valid_mask.sum():
            top_idx = np.where(valid_mask)[0]
        else:
            top_idx = np.argpartition(-boosted_valid, top_k - 1)[:top_k]
            top_idx = [j for j in top_idx if valid_mask[j]]
        for j in top_idx:
            if j == i:
                continue
            sim_val = float(row[j])
            boosted_val = float(boosted[j])
            cited_flag = (cite_boost > 0) and (docs[j] in cite_adj.get(doc_i, set()))
            edges.append({
                "source": pids[i],
                "target": pids[j],
                "source_doc": doc_i,
                "target_doc": docs[j],
                "type": type_name,
                "similarity": round(sim_val, 4),
                "boosted_similarity": round(boosted_val, 4),
                "cite_boosted": cited_flag,
            })
    return edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/05_eval/dense_retrieval/augmented_v2/corpus_v1_enriched.jsonl")
    ap.add_argument("--chunk-graph", default="data/01_graphs/chunk_virtual_nodes_v2.json")
    ap.add_argument("--citation-graph", default="data/01_graphs/citation_graph.json")
    ap.add_argument("--output", default="data/01_graphs/typed_crossdoc_edges.json")
    ap.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--download-root", default="models")
    ap.add_argument("--types", nargs="+", default=["figure", "formula", "table"],
                    help="Which element types to build (from corpus)")
    ap.add_argument("--include-chunks", action="store_true",
                    help="Also build chunk-chunk cross-doc edges (from chunk graph)")
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--cite-boost", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    # ── Lazy import + model load ──
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}", flush=True)
    try:
        from modelscope import snapshot_download
        model_dir = snapshot_download(args.model_name, cache_dir=args.download_root)
    except Exception:
        # Fall back to local path
        local = Path(args.download_root) / args.model_name.replace("/", "_").replace(".", "_")
        # common pre-downloaded location
        candidates = [
            Path(args.download_root) / args.model_name,
            Path(args.download_root) / args.model_name.replace("/", "___").replace(".", "___"),
            Path("models") / args.model_name.replace("/", "___").replace(".", "___"),
        ]
        model_dir = None
        for c in candidates:
            if c.exists():
                model_dir = str(c); break
        if model_dir is None:
            print("[fatal] could not resolve model path", file=sys.stderr); sys.exit(2)
    # Fix Qwen3 subdir naming convention on disk
    on_disk = Path(model_dir)
    if not (on_disk / "config.json").exists():
        alt = on_disk.parent / on_disk.name.replace(".", "___")
        if (alt / "config.json").exists():
            model_dir = str(alt)
    print(f"[info] loading SentenceTransformer from {model_dir}", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_dir, device=device)

    # ── Load citation adj ──
    cite_path = Path(args.citation_graph)
    cite_adj = load_citation_adj(cite_path) if cite_path.exists() else {}
    print(f"[info] citation adjacency: {len(cite_adj)} docs, "
          f"{sum(len(v) for v in cite_adj.values())//2} undirected edges", flush=True)

    # ── Load & encode typed passages ──
    all_edges: List[dict] = []
    type_stats: Dict[str, int] = {}

    corpus_path = Path(args.corpus)
    for typ in args.types:
        pids, docs, texts, _ = load_corpus_typed(corpus_path, {typ})
        if not pids:
            print(f"[warn] type={typ}: no passages", flush=True); continue
        t0 = time.time()
        embs = encode_texts(model, texts, batch_size=args.batch_size,
                            is_query=False, max_length=args.max_length)
        print(f"[info] type={typ}: encoded {len(pids)} in {time.time()-t0:.1f}s", flush=True)
        edges = build_edges_for_type(
            pids, docs, embs, typ,
            top_k=args.top_k, threshold=args.threshold,
            cite_adj=cite_adj, cite_boost=args.cite_boost,
        )
        print(f"[info] type={typ}: {len(edges)} edges", flush=True)
        all_edges.extend(edges)
        type_stats[typ] = len(edges)

    # ── Chunk-chunk (optional) ──
    if args.include_chunks:
        pids, docs, texts = load_chunk_nodes(Path(args.chunk_graph))
        print(f"[info] chunks: {len(pids)} nodes", flush=True)
        t0 = time.time()
        embs = encode_texts(model, texts, batch_size=args.batch_size,
                            is_query=False, max_length=args.max_length)
        print(f"[info] chunks: encoded in {time.time()-t0:.1f}s", flush=True)
        edges = build_edges_for_type(
            pids, docs, embs, "chunk",
            top_k=args.top_k, threshold=args.threshold,
            cite_adj=cite_adj, cite_boost=args.cite_boost,
        )
        print(f"[info] chunks: {len(edges)} edges", flush=True)
        all_edges.extend(edges)
        type_stats["chunk"] = len(edges)

    # ── Stats ──
    cite_count = sum(1 for e in all_edges if e.get("cite_boosted"))
    doc_pairs: Set[Tuple[str, str]] = set()
    for e in all_edges:
        a, b = e["source_doc"], e["target_doc"]
        doc_pairs.add((min(a, b), max(a, b)))

    out = {
        "metadata": {
            "model": args.model_name,
            "corpus": str(corpus_path),
            "threshold": args.threshold,
            "top_k": args.top_k,
            "cite_boost": args.cite_boost,
            "types": args.types + (["chunk"] if args.include_chunks else []),
        },
        "stats": {
            "num_edges_total": len(all_edges),
            "num_edges_cite_boosted": cite_count,
            "num_doc_pairs_connected": len(doc_pairs),
            "edges_per_type": type_stats,
        },
        "edges": all_edges,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[info] wrote {out_path}", flush=True)
    print(f"[info] stats: {json.dumps(out['stats'], indent=2)}", flush=True)


if __name__ == "__main__":
    main()
