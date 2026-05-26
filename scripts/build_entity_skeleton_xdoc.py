#!/usr/bin/env python3
"""Entity-skeleton cross-document reranker (REGENT-inspired).

Reads cross-doc similarity edges (CLIP-based, recall-only) and enriched elements,
extracts key entities as a "semantic skeleton," then reranks cross-doc pairs using
entity overlap + embedding similarity fusion.

Output: reranked cross-doc edges with entity_skeleton_score, suitable for
promotion to hard multi-hop graph edges.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Entity extraction patterns (zero-cost, rule-based) ──────────────────────

# Method name indicators
METHOD_PATTERNS = [
    r"\b(CNN|RNN|LSTM|GRU|BERT|GPT|ResNet|ViT|Transformer|GAN|VAE|RL|SGD|Adam)\b",
    r"\b(?:random forest|SVM|logistic regression|gradient boosting|XGBoost|k-means|PCA)\b",
    r"\b(?:fine.?tun(?:e|ing)|pre.?train(?:ed|ing)|transfer learning|meta.?learning)\b",
]

# Dataset indicators
DATASET_PATTERNS = [
    r"\b(ImageNet|CIFAR|MNIST|COCO|SQuAD|GLUE|SuperGLUE|WMT|LibriSpeech|WikiText)\b",
    r"\b(?:Adult|COMPAS|CelebA|FFHQ|LAION|CommonCrawl|PubMed|ArXiv)\b",
    r"\b(?:OpenStreetMap|SpaceNet|OpenEarthMap|Sentinel|Landsat)\b",
]

# Metric indicators
METRIC_PATTERNS = [
    r"\b(?:accuracy|precision|recall|F1|AUC|BLEU|ROUGE|METEOR|perplexity|MAE|MSE|RMSE)\b",
    r"\b(?:demographic parity|equalized odds|equal opportunity|calibration|fairness)\b",
    r"\b(?:throughput|latency|FLOPS|parameters?|inference time)\b",
]

# Formula variable indicators
FORMULA_PATTERNS = [
    r"\b(?:\\mathcal\{[LW]\}|\\theta|\\lambda|\\alpha|\\beta|\\gamma|\\epsilon)\b",
    r"\b(?:loss function|objective|regularization|gradient|convergence)\b",
]

# Generic stop entities (too common to be discriminative)
STOP_ENTITIES = {
    "model", "method", "approach", "result", "figure", "table", "data",
    "performance", "experiment", "training", "test", "validation",
    "baseline", "proposed", "comparison", "paper", "section",
}


def extract_entities(text: str) -> Set[str]:
    """Extract method/dataset/metric/formula entities from text (rule-based)."""
    entities: Set[str] = set()
    all_patterns = METHOD_PATTERNS + DATASET_PATTERNS + METRIC_PATTERNS + FORMULA_PATTERNS
    for pat in all_patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            entity = match.group(0).lower().strip()
            if entity not in STOP_ENTITIES and len(entity) >= 2:
                entities.add(entity)
    return entities


def element_to_text(el: Dict[str, Any]) -> str:
    """Combine all text fields of an element."""
    parts = []
    for field in ("caption", "content", "enriched_content", "enriched_title",
                   "context_before", "context_after"):
        val = el.get(field, "")
        if val:
            parts.append(str(val))
    return " ".join(parts)


def load_elements(elements_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load elements, handling both flat and documents-format."""
    with open(elements_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    elements: Dict[str, Dict[str, Any]] = {}
    docs = data.get("documents", {})
    if docs:
        for doc_id, doc in docs.items():
            for eid, el in doc.get("elements", {}).items():
                elements[eid] = el
    elif isinstance(data, dict) and "elements" in data:
        for eid, el in data["elements"].items():
            elements[eid] = el
    elif isinstance(data, list):
        for item in data:
            eid = item.get("element_id", item.get("id", ""))
            if eid:
                elements[eid] = item
    return elements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cross-doc-edges",
        default="data/01_graphs/cross_doc_sim_edges.json",
        help="Cross-doc similarity edges (CLIP-based)",
    )
    parser.add_argument(
        "--elements",
        default="data/02_enriched/multimodal_elements_enriched.json",
        help="Enriched multimodal elements",
    )
    parser.add_argument(
        "--min-entity-overlap",
        type=int,
        default=1,
        help="Minimum shared entities for a pair to be promoted",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    # Output dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = ROOT / f"data/05_eval/entity_skeleton_xdoc_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load cross-doc edges
    edges_path = ROOT / args.cross_doc_edges
    if not edges_path.exists():
        print(f"ERROR: edges file not found: {edges_path}")
        sys.exit(1)
    with open(edges_path, "r", encoding="utf-8") as f:
        edges_data = json.load(f)
    edges = edges_data.get("edges", edges_data if isinstance(edges_data, list) else [])
    print(f"Loaded {len(edges)} cross-doc edges")

    # Load elements
    elements_path = ROOT / args.elements
    if not elements_path.exists():
        print(f"ERROR: elements file not found: {elements_path}")
        sys.exit(1)
    elements = load_elements(elements_path)
    print(f"Loaded {len(elements)} elements")

    if args.limit:
        edges = edges[: args.limit]
        print(f"Limited to {len(edges)} edges")

    # Extract entities from edge text previews (section-level nodes, not elements)
    print("\n--- Extracting entities ---")
    edge_entities: Dict[str, Set[str]] = {}
    # Also build per-doc entity indices by aggregating section entities
    doc_entities: Dict[str, Set[str]] = defaultdict(set)

    for edge in edges:
        for key, text_key in [("source", "source_text_preview"),
                               ("target", "target_text_preview")]:
            node_id = edge.get(key, "")
            if node_id and node_id not in edge_entities:
                text = edge.get(text_key, "")
                edge_entities[node_id] = extract_entities(text)
                doc_id = edge.get(f"{key}_doc", "")
                if doc_id:
                    doc_entities[doc_id] |= edge_entities[node_id]

    # Compute entity skeleton scores
    print("--- Computing entity skeleton scores ---")
    results: List[Dict[str, Any]] = []
    stats = defaultdict(int)

    for idx, edge in enumerate(edges, 1):
        src_id = edge.get("source", "")
        tgt_id = edge.get("target", "")
        src_doc = edge.get("source_doc", "")
        tgt_doc = edge.get("target_doc", "")
        src_ents = edge_entities.get(src_id, set())
        tgt_ents = edge_entities.get(tgt_id, set())

        # Entity overlap metrics
        shared = src_ents & tgt_ents
        union = src_ents | tgt_ents
        jaccard = len(shared) / len(union) if union else 0.0
        overlap_count = len(shared)

        # REGENT-style entity skeleton score
        # Base on overlap density and semantic importance
        sim_score = edge.get("similarity", edge.get("score", 0.0))
        entity_score = overlap_count * 0.15 + jaccard * 0.5
        skeleton_score = 0.3 * sim_score + 0.7 * entity_score

        # Fusion: combine CLIP similarity + entity overlap
        fusion_score = (
            0.3 * sim_score +          # CLIP visual similarity (recall)
            0.4 * entity_score +        # Entity overlap (precision signal)
            0.3 * (min(overlap_count / 3, 1.0))  # Saturation bonus
        )

        # Quality tier
        if overlap_count >= 3 and jaccard >= 0.1:
            tier = "strong"
        elif overlap_count >= 1:
            tier = "weak"
        else:
            tier = "recall_only"

        stats[f"tier_{tier}"] += 1
        stats[f"entity_overlap_{overlap_count}"] += 1

        result = {
            **edge,
            "source_entities": sorted(src_ents),
            "target_entities": sorted(tgt_ents),
            "shared_entities": sorted(shared),
            "entity_overlap_count": overlap_count,
            "entity_jaccard": round(jaccard, 4),
            "entity_skeleton_score": round(entity_score, 4),
            "fusion_score": round(fusion_score, 4),
            "quality_tier": tier,
        }

        if overlap_count >= args.min_entity_overlap:
            results.append(result)

        if idx % 500 == 0 or idx == len(edges):
            print(
                f"[{idx:05d}/{len(edges):05d}] "
                f"strong={stats['tier_strong']} "
                f"weak={stats['tier_weak']} "
                f"recall_only={stats['tier_recall_only']}"
            )

    # Sort by fusion score
    results.sort(key=lambda x: x["fusion_score"], reverse=True)

    # Output
    output_path = out_dir / "entity_skeleton_reranked.json"
    promoted_path = out_dir / "entity_skeleton_strong.json"

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_input_edges": len(edges),
        "total_promoted": len(results),
        "stats": dict(stats),
        "tier_counts": {
            "strong": stats["tier_strong"],
            "weak": stats["tier_weak"],
            "recall_only": stats["tier_recall_only"],
        },
        "edges": results,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Strong-only subset for hard edge promotion
    strong = [r for r in results if r["quality_tier"] == "strong"]
    with promoted_path.open("w", encoding="utf-8") as f:
        json.dump({
            "edges": strong,
            "count": len(strong),
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults:")
    print(f"  Total edges: {len(edges)}")
    print(f"  Promoted (entity overlap >= {args.min_entity_overlap}): {len(results)}")
    print(f"  Strong tier (overlap>=3, jaccard>=0.1): {stats['tier_strong']}")
    print(f"  Weak tier (overlap>=1): {stats['tier_weak']}")
    print(f"  Recall-only (no entity overlap): {stats['tier_recall_only']}")
    print(f"  Output: {output_path}")
    print(f"  Strong subset: {promoted_path}")


if __name__ == "__main__":
    main()
