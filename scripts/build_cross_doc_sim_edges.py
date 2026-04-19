#!/usr/bin/env python3
"""
build_cross_doc_sim_edges.py
============================
Build cross-document virtual edges by computing cosine similarity between
LLM section summaries across the 53 small-scale docs.

Steps
-----
1. Load section summaries from llm_summary_virtual_nodes.json
2. Encode all summaries with Qwen3-Embedding (or fallback to TF-IDF)
3. For each summary, find top-K most similar summaries in OTHER docs
4. Keep pairs above a similarity threshold
5. Output cross_doc_sim_edges.json

Output schema
-------------
{
  "metadata": {...},
  "edges": [
    {
      "source": "<doc_id>_section_<idx>",   # summary node id
      "target": "<doc_id>_section_<idx>",
      "source_doc": "<doc_id>",
      "target_doc": "<doc_id>",
      "source_text": "...",   # truncated
      "target_text": "...",
      "similarity": 0.82,
      "edge_type": "cross_doc_section_sim"
    }, ...
  ],
  "stats": {...}
}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── helpers ──────────────────────────────────────────────────────────────────

def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-9)


def batch_encode(texts: List[str], model_path: str, batch_size: int = 32, max_length: int = 512, device: str = "cuda") -> List[List[float]]:
    """Encode texts with sentence-transformers / Qwen3-Embedding."""
    from sentence_transformers import SentenceTransformer
    import torch

    print(f"[encode] loading model from {model_path} ...", flush=True)
    model = SentenceTransformer(model_path, trust_remote_code=True, device=device)
    # max_length must be set via tokenizer, not encode()
    try:
        model.tokenizer.model_max_length = max_length
        model.max_seq_length = max_length
    except Exception:
        pass
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def tfidf_encode(texts: List[str]) -> List[List[float]]:
    """Lightweight CPU fallback: TF-IDF vectors."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    print("[encode] using TF-IDF fallback ...", flush=True)
    vec = TfidfVectorizer(max_features=8192, sublinear_tf=True)
    mat = vec.fit_transform(texts).toarray().tolist()
    return mat


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-graph", default="data/01_graphs/llm_summary_virtual_nodes.json")
    ap.add_argument("--output", default="data/01_graphs/cross_doc_sim_edges.json")
    ap.add_argument("--model-path", default="models/Qwen3-Embedding-4B")
    ap.add_argument("--citation-graph", default="data/01_graphs/citation_graph.json",
                    help="citation_graph.json for doc-level citation confidence boost")
    ap.add_argument("--citation-boost", type=float, default=0.10,
                    help="Additive confidence boost when two docs have a citation link")
    ap.add_argument("--top-k", type=int, default=5, help="Max cross-doc neighbors per summary node")
    ap.add_argument("--threshold", type=float, default=0.70, help="Min cosine similarity to keep edge (applied BEFORE citation boost)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use-tfidf", action="store_true", help="Skip model, use TF-IDF (CPU debug)")
    ap.add_argument("--level", choices=["section", "chunk", "both"], default="section",
                    help="Which summary level to use for cross-doc edges")
    args = ap.parse_args()

    summary_path = PROJECT_ROOT / args.summary_graph
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] reading {summary_path} ...", flush=True)
    data = json.loads(summary_path.read_bytes())
    documents: Dict[str, Any] = data.get("documents", {})
    print(f"[load] {len(documents)} docs", flush=True)

    # ── collect summary nodes ──
    nodes: List[Dict[str, Any]] = []   # {node_id, doc_id, text, level}

    for doc_id, doc in documents.items():
        if args.level in ("section", "both"):
            for sec_id, sec in doc.get("section_summaries", {}).items():
                text = (sec.get("text") or sec.get("summary_text", "")).strip()
                if not text:
                    continue
                node_id = f"{doc_id}_secsummary_{sec_id}"
                nodes.append({"node_id": node_id, "doc_id": doc_id,
                               "text": text, "level": "section",
                               "section_title": sec.get("source_section") or sec.get("section_title", "")})

        if args.level in ("chunk", "both"):
            for chunk_id, chunk in doc.get("chunk_summaries", {}).items():
                text = (chunk.get("text") or chunk.get("summary_text", "")).strip()
                if not text:
                    continue
                node_id = f"{doc_id}_chunksummary_{chunk_id}"
                nodes.append({"node_id": node_id, "doc_id": doc_id,
                               "text": text, "level": "chunk",
                               "chunk_idx": chunk.get("chunk_idx", -1)})

    print(f"[nodes] collected {len(nodes)} summary nodes", flush=True)
    if not nodes:
        print("[error] no nodes found, check summary graph format")
        sys.exit(1)

    texts = [n["text"] for n in nodes]

    # ── load citation graph → doc-level citation set ──
    citation_pairs: Dict[Tuple[str, str], float] = {}  # (min_doc, max_doc) -> confidence
    citation_path = PROJECT_ROOT / args.citation_graph
    if citation_path.exists() and args.citation_boost > 0:
        print(f"[cite] loading citation graph from {citation_path} ...", flush=True)
        cite_data = json.loads(citation_path.read_bytes())
        for edge in cite_data.get("edges", []):
            src = str(edge.get("source", "")).strip()
            tgt = str(edge.get("target", "")).strip()
            if not src or not tgt or src == tgt:
                continue
            conf = float(edge.get("confidence", 1.0))
            key = (min(src, tgt), max(src, tgt))
            citation_pairs[key] = max(citation_pairs.get(key, 0.0), conf)
        print(f"[cite] {len(citation_pairs)} citation doc-pairs loaded", flush=True)
    else:
        print("[cite] no citation graph or boost=0, skipping", flush=True)

    if args.use_tfidf:
        embeddings = tfidf_encode(texts)
    else:
        embeddings = batch_encode(texts, args.model_path,
                                  batch_size=args.batch_size,
                                  max_length=args.max_length,
                                  device=args.device)

    print(f"[embed] done, dim={len(embeddings[0])}", flush=True)

    # ── build cross-doc edges ──
    # group node indices by doc
    doc_to_indices: Dict[str, List[int]] = {}
    for i, n in enumerate(nodes):
        doc_to_indices.setdefault(n["doc_id"], []).append(i)

    edges: List[Dict[str, Any]] = []
    total_comparisons = 0

    for i, node_a in enumerate(nodes):
        doc_a = node_a["doc_id"]
        emb_a = embeddings[i]

        # only compare against nodes from OTHER docs
        candidates: List[Tuple[float, int]] = []
        for doc_b, indices_b in doc_to_indices.items():
            if doc_b == doc_a:
                continue
            # citation boost: additive if these two docs cite each other
            cite_key = (min(doc_a, doc_b), max(doc_a, doc_b))
            cite_conf = citation_pairs.get(cite_key, 0.0)
            cite_add = args.citation_boost * cite_conf  # 0 if no citation
            for j in indices_b:
                sim = cosine_sim(emb_a, embeddings[j])
                total_comparisons += 1
                # apply threshold on raw sim (before boost) to avoid noise
                if sim >= args.threshold:
                    candidates.append((sim + cite_add, j, sim, cite_add > 0))

        # keep top-k by boosted score
        candidates.sort(key=lambda x: -x[0])
        for boosted_sim, j, raw_sim, has_cite in candidates[: args.top_k]:
            node_b = nodes[j]
            edges.append({
                "source": node_a["node_id"],
                "target": node_b["node_id"],
                "source_doc": node_a["doc_id"],
                "target_doc": node_b["doc_id"],
                "source_section": node_a.get("section_title", ""),
                "target_section": node_b.get("section_title", ""),
                "source_text_preview": node_a["text"][:120],
                "target_text_preview": node_b["text"][:120],
                "similarity": round(boosted_sim, 4),
                "raw_similarity": round(raw_sim, 4),
                "citation_boosted": has_cite,
                "level": node_a["level"],
                "edge_type": "cross_doc_section_sim",
            })

        if (i + 1) % 100 == 0:
            print(f"[progress] {i+1}/{len(nodes)} nodes processed, {len(edges)} edges so far", flush=True)

    print(f"[edges] {len(edges)} cross-doc edges (from {total_comparisons} comparisons)", flush=True)

    # ── stats ──
    doc_pairs_covered: set = set()
    for e in edges:
        da, db = e["source_doc"], e["target_doc"]
        doc_pairs_covered.add((min(da, db), max(da, db)))

    sim_values = [e["similarity"] for e in edges]
    citation_boosted_count = sum(1 for e in edges if e.get("citation_boosted"))
    stats = {
        "num_summary_nodes": len(nodes),
        "num_docs": len(documents),
        "num_cross_doc_edges": len(edges),
        "num_citation_boosted_edges": citation_boosted_count,
        "num_doc_pairs_connected": len(doc_pairs_covered),
        "threshold": args.threshold,
        "citation_boost": args.citation_boost,
        "top_k_per_node": args.top_k,
        "level": args.level,
        "sim_mean": round(sum(sim_values) / max(1, len(sim_values)), 4),
        "sim_min": round(min(sim_values), 4) if sim_values else 0,
        "sim_max": round(max(sim_values), 4) if sim_values else 0,
    }
    print(f"[stats] doc pairs connected: {len(doc_pairs_covered)} / {len(documents)*(len(documents)-1)//2} possible", flush=True)

    output = {
        "metadata": {
            "model_path": args.model_path if not args.use_tfidf else "tfidf",
            "level": args.level,
            "threshold": args.threshold,
            "citation_boost": args.citation_boost,
            "top_k": args.top_k,
        },
        "edges": edges,
        "stats": stats,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] saved to {output_path}", flush=True)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
