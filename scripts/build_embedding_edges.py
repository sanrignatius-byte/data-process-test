#!/usr/bin/env python3
"""Build embedding-based semantic edges between document elements.

Computes embeddings for all multimodal elements (using text: caption + context),
then finds high-similarity pairs to create virtual edges that complement
the existing LaTeX reference-based graph structure.

Outputs a JSON file compatible with run_phase0_eval_ab.py --embedding-edges.

Usage:
    python scripts/build_embedding_edges.py \
        --elements data111/multimodal_elements_enriched.json \
        --output data/embedding_edges.json \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --threshold 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────
# Element text extraction
# ──────────────────────────────────────────────────────────────

def _element_text(elem: Dict[str, Any], use_enriched: bool = True) -> str:
    """Build a text representation for an element (for embedding).

    Combines caption + context + enriched_content (if available).
    For formulas: uses LaTeX content + surrounding context.
    """
    parts: List[str] = []

    # Caption / label
    caption = (elem.get("caption") or "").strip()
    if caption:
        parts.append(caption)

    # Enriched content (visual description from MoDora)
    if use_enriched:
        enriched = (elem.get("enriched_content") or "").strip()
        if enriched:
            parts.append(enriched)

    # Raw content (LaTeX for formulas, text for paragraphs)
    content = (elem.get("content") or "").strip()
    if content and content != caption:
        # For formulas, content is LaTeX — include but limit length
        parts.append(content[:500])

    # Context before/after (surrounding paragraphs)
    for ctx_key in ("context_before", "context_after"):
        ctx = (elem.get(ctx_key) or "").strip()
        if ctx:
            parts.append(ctx[:300])

    text = " ".join(parts).strip()
    if not text:
        # Fallback: element_id itself
        text = elem.get("element_id", "unknown")
    return text


def load_elements(elements_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load multimodal elements, return {element_id: element_dict}."""
    raw = json.loads(elements_path.read_text(encoding="utf-8"))
    docs = raw.get("documents", {})
    elements: Dict[str, Dict[str, Any]] = {}
    for doc_id, doc in docs.items():
        doc_elements = doc.get("elements", {})
        if isinstance(doc_elements, dict):
            for eid, edata in doc_elements.items():
                edata["doc_id"] = doc_id
                elements[eid] = edata
        elif isinstance(doc_elements, list):
            for edata in doc_elements:
                eid = edata.get("element_id", "")
                edata["doc_id"] = doc_id
                elements[eid] = edata
    return elements


# ──────────────────────────────────────────────────────────────
# Embedding computation
# ──────────────────────────────────────────────────────────────

def compute_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> np.ndarray:
    """Compute embeddings using sentence-transformers.

    Returns: (N, D) numpy array of L2-normalized embeddings.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        print("  Install: pip install sentence-transformers torch")
        sys.exit(1)

    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  Loading model: {model_name} (device={device})")
    model = SentenceTransformer(model_name, device=device)

    print(f"  Computing embeddings for {len(texts)} elements (batch_size={batch_size})...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — shape: {embeddings.shape}")
    return embeddings


# ──────────────────────────────────────────────────────────────
# Edge discovery
# ──────────────────────────────────────────────────────────────

def find_embedding_edges(
    element_ids: List[str],
    elements: Dict[str, Dict[str, Any]],
    embeddings: np.ndarray,
    threshold: float = 0.8,
    max_edges_per_element: int = 10,
    same_doc_only: bool = False,
    cross_doc_only: bool = False,
    existing_edges: Optional[Set[Tuple[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Find high-similarity element pairs above threshold.

    Returns list of edge dicts: {element_a, element_b, similarity, edge_type, ...}
    """
    if existing_edges is None:
        existing_edges = set()

    n = len(element_ids)
    eid_to_idx = {eid: i for i, eid in enumerate(element_ids)}

    # Compute pairwise similarity matrix (cosine, since embeddings are L2-normalized)
    # For large N, use batched dot product to avoid memory issues
    print(f"  Computing pairwise similarities ({n}×{n})...")
    edges: List[Dict[str, Any]] = []

    # Track per-element edge count
    edge_count: Dict[str, int] = defaultdict(int)

    # Process in blocks to limit memory
    block_size = 500
    for i_start in range(0, n, block_size):
        i_end = min(i_start + block_size, n)
        block_emb = embeddings[i_start:i_end]  # (block, D)
        sim_block = block_emb @ embeddings.T  # (block, N)

        for local_i, global_i in enumerate(range(i_start, i_end)):
            eid_a = element_ids[global_i]
            doc_a = elements[eid_a].get("doc_id", "")
            type_a = elements[eid_a].get("element_type", "")

            # Get similarities, filter
            sims = sim_block[local_i]

            # Find indices above threshold (excluding self)
            candidates = []
            for j in range(n):
                if j == global_i:
                    continue
                if sims[j] < threshold:
                    continue
                candidates.append((j, float(sims[j])))

            # Sort by similarity descending
            candidates.sort(key=lambda x: -x[1])

            for j, sim_val in candidates:
                if edge_count[eid_a] >= max_edges_per_element:
                    break

                eid_b = element_ids[j]
                doc_b = elements[eid_b].get("doc_id", "")
                type_b = elements[eid_b].get("element_type", "")

                # Same/cross doc filter
                is_same_doc = (doc_a == doc_b)
                if same_doc_only and not is_same_doc:
                    continue
                if cross_doc_only and is_same_doc:
                    continue

                # Skip self-type same-doc (paragraph-paragraph in same doc is too noisy)
                if is_same_doc and type_a == type_b == "paragraph":
                    continue

                # Canonical ordering to avoid duplicates
                edge_key = (min(eid_a, eid_b), max(eid_a, eid_b))
                if edge_key in existing_edges:
                    continue
                existing_edges.add(edge_key)

                edge_count[eid_a] += 1
                edge_count[eid_b] += 1

                edges.append({
                    "element_a_id": eid_a,
                    "element_b_id": eid_b,
                    "doc_a": doc_a,
                    "doc_b": doc_b,
                    "type_a": type_a,
                    "type_b": type_b,
                    "similarity": round(sim_val, 4),
                    "is_cross_doc": not is_same_doc,
                    "edge_type": "embedding_similarity",
                })

    print(f"  Found {len(edges)} embedding edges (threshold={threshold})")
    return edges


# ──────────────────────────────────────────────────────────────
# Inspection / analysis
# ──────────────────────────────────────────────────────────────

def inspect_edges(
    edges: List[Dict[str, Any]],
    elements: Dict[str, Dict[str, Any]],
    top_k: int = 30,
) -> None:
    """Print top-K edges for manual inspection."""
    sorted_edges = sorted(edges, key=lambda e: -e["similarity"])
    print(f"\n{'='*80}")
    print(f"Top-{top_k} embedding edges (sorted by similarity)")
    print(f"{'='*80}")

    from collections import Counter
    combo_counter = Counter()
    cross_doc_count = 0

    for i, edge in enumerate(sorted_edges[:top_k]):
        ea = edge["element_a_id"]
        eb = edge["element_b_id"]
        sim = edge["similarity"]
        cross = "CROSS-DOC" if edge["is_cross_doc"] else "same-doc"
        ta = edge["type_a"]
        tb = edge["type_b"]
        combo = "+".join(sorted([ta, tb]))
        combo_counter[combo] += 1
        if edge["is_cross_doc"]:
            cross_doc_count += 1

        cap_a = (elements.get(ea, {}).get("caption") or "")[:60]
        cap_b = (elements.get(eb, {}).get("caption") or "")[:60]

        print(f"\n  [{i+1}] sim={sim:.4f} | {combo} | {cross}")
        print(f"      A: {ea}")
        print(f"         {cap_a}")
        print(f"      B: {eb}")
        print(f"         {cap_b}")

    # Summary stats
    print(f"\n{'='*80}")
    print(f"Summary of all {len(edges)} edges:")
    all_combos = Counter()
    all_cross = 0
    for e in edges:
        combo = "+".join(sorted([e["type_a"], e["type_b"]]))
        all_combos[combo] += 1
        if e["is_cross_doc"]:
            all_cross += 1
    print(f"  By modality combo: {dict(all_combos)}")
    print(f"  Cross-doc: {all_cross} ({100*all_cross/max(len(edges),1):.1f}%)")
    print(f"  Same-doc: {len(edges)-all_cross} ({100*(len(edges)-all_cross)/max(len(edges),1):.1f}%)")

    # Similarity distribution
    sims = [e["similarity"] for e in edges]
    if sims:
        print(f"  Similarity: min={min(sims):.4f}, max={max(sims):.4f}, "
              f"mean={sum(sims)/len(sims):.4f}, median={sorted(sims)[len(sims)//2]:.4f}")


# ──────────────────────────────────────────────────────────────
# Load existing graph edges for comparison
# ──────────────────────────────────────────────────────────────

def load_existing_graph_edges(
    hub_candidates_path: Optional[Path],
) -> Set[Tuple[str, str]]:
    """Load existing element adjacency to mark overlap."""
    existing: Set[Tuple[str, str]] = set()
    if hub_candidates_path is None or not hub_candidates_path.exists():
        return existing

    obj = json.loads(hub_candidates_path.read_text(encoding="utf-8"))
    for p in obj.get("pairs", []):
        a = str(p.get("element_a_id", "")).strip()
        b = str(p.get("element_b_id", "")).strip()
        if a and b:
            existing.add((min(a, b), max(a, b)))

    for ab in obj.get("adjacent_bridge_adjacency", []):
        for ei in ab.get("elements_i", []):
            for ej in ab.get("elements_j", []):
                ei, ej = str(ei).strip(), str(ej).strip()
                if ei and ej:
                    existing.add((min(ei, ej), max(ei, ej)))

    return existing


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build embedding-based semantic edges between document elements"
    )
    ap.add_argument(
        "--elements", type=Path,
        default=Path("data111/multimodal_elements_enriched.json"),
        help="Multimodal elements JSON (enriched preferred)",
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("data/embedding_edges.json"),
        help="Output embedding edges JSON",
    )
    ap.add_argument(
        "--model", type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name",
    )
    ap.add_argument(
        "--threshold", type=float, default=0.80,
        help="Cosine similarity threshold for creating an edge (default: 0.80)",
    )
    ap.add_argument(
        "--max-edges-per-element", type=int, default=10,
        help="Maximum edges per element to prevent hub explosion",
    )
    ap.add_argument(
        "--batch-size", type=int, default=64,
        help="Encoding batch size",
    )
    ap.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda / cpu / mps (auto-detect if omitted)",
    )
    ap.add_argument(
        "--hub-candidates", type=Path, default=None,
        help="Existing hub candidates JSON (to exclude already-known edges)",
    )
    ap.add_argument(
        "--inspect", action="store_true",
        help="Print top edges for manual inspection",
    )
    ap.add_argument(
        "--top-k", type=int, default=30,
        help="Number of top edges to inspect",
    )
    ap.add_argument(
        "--same-doc-only", action="store_true",
        help="Only create intra-document edges",
    )
    ap.add_argument(
        "--cross-doc-only", action="store_true",
        help="Only create cross-document edges",
    )
    ap.add_argument(
        "--no-enriched", action="store_true",
        help="Do not use enriched_content in text representation",
    )
    ap.add_argument(
        "--save-embeddings", type=Path, default=None,
        help="Save computed embeddings as .npy for reuse",
    )
    ap.add_argument(
        "--load-embeddings", type=Path, default=None,
        help="Load pre-computed embeddings from .npy",
    )
    args = ap.parse_args()

    print("=" * 60)
    print("Build Embedding Edges")
    print("=" * 60)

    # 1. Load elements
    print(f"\n[1/4] Loading elements from {args.elements}")
    elements = load_elements(args.elements)
    print(f"  Loaded {len(elements)} elements")

    # Sorted element IDs for consistent ordering
    element_ids = sorted(elements.keys())

    # Build text representations
    use_enriched = not args.no_enriched
    texts = [_element_text(elements[eid], use_enriched=use_enriched) for eid in element_ids]
    avg_len = sum(len(t) for t in texts) / max(len(texts), 1)
    print(f"  Average text length: {avg_len:.0f} chars (enriched={use_enriched})")

    # 2. Compute or load embeddings
    if args.load_embeddings and args.load_embeddings.exists():
        print(f"\n[2/4] Loading pre-computed embeddings from {args.load_embeddings}")
        embeddings = np.load(str(args.load_embeddings))
        assert embeddings.shape[0] == len(element_ids), (
            f"Embedding count mismatch: {embeddings.shape[0]} vs {len(element_ids)} elements"
        )
    else:
        print(f"\n[2/4] Computing embeddings with {args.model}")
        embeddings = compute_embeddings(
            texts, args.model,
            batch_size=args.batch_size,
            device=args.device,
        )

    if args.save_embeddings:
        args.save_embeddings.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(args.save_embeddings), embeddings)
        print(f"  Saved embeddings to {args.save_embeddings}")

    # 3. Load existing edges (optional, to avoid duplicates)
    existing_edges: Set[Tuple[str, str]] = set()
    if args.hub_candidates:
        print(f"\n[3/4] Loading existing graph edges from {args.hub_candidates}")
        existing_edges = load_existing_graph_edges(args.hub_candidates)
        print(f"  Loaded {len(existing_edges)} existing edges (will be excluded)")
    else:
        print(f"\n[3/4] No existing edges to exclude")

    # 4. Find embedding edges
    print(f"\n[4/4] Finding embedding edges (threshold={args.threshold})")
    edges = find_embedding_edges(
        element_ids=element_ids,
        elements=elements,
        embeddings=embeddings,
        threshold=args.threshold,
        max_edges_per_element=args.max_edges_per_element,
        same_doc_only=args.same_doc_only,
        cross_doc_only=args.cross_doc_only,
        existing_edges=existing_edges,
    )

    # Inspect if requested
    if args.inspect:
        inspect_edges(edges, elements, top_k=args.top_k)

    # Save output
    output_data = {
        "metadata": {
            "model": args.model,
            "threshold": args.threshold,
            "max_edges_per_element": args.max_edges_per_element,
            "n_elements": len(element_ids),
            "n_edges": len(edges),
            "embedding_dim": int(embeddings.shape[1]),
            "use_enriched": use_enriched,
            "same_doc_only": args.same_doc_only,
            "cross_doc_only": args.cross_doc_only,
            "existing_edges_excluded": len(existing_edges),
        },
        "edges": edges,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Wrote {len(edges)} edges → {args.output}")

    # Quick summary
    from collections import Counter
    combos = Counter("+".join(sorted([e["type_a"], e["type_b"]])) for e in edges)
    cross = sum(1 for e in edges if e["is_cross_doc"])
    print(f"\n  Summary:")
    print(f"    By combo: {dict(combos)}")
    print(f"    Cross-doc: {cross}, Same-doc: {len(edges)-cross}")
    if edges:
        sims = [e["similarity"] for e in edges]
        print(f"    Similarity range: [{min(sims):.3f}, {max(sims):.3f}]")


if __name__ == "__main__":
    main()
