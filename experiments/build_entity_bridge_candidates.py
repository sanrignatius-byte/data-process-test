#!/usr/bin/env python3
"""
Build cross-document element pairs linked by shared enriched entities (keywords).

Instead of matching elements by explicit numbering (resolver v1, 0/120 strong_chain),
this flips the approach:
  1. Paper A and Paper B share semantic entities (methods, datasets, metrics)
  2. Elements in A and B that mention the SAME entity are linked through that entity
  3. The entity itself IS the bridge

This avoids the "Figure 3 in paper A != Figure 3 in paper B" problem entirely,
because elements are linked by what they're about, not by their number.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ENRICHED = ROOT / "data/02_enriched/multimodal_elements_enriched.json"
DEFAULT_CITATION_GRAPH = ROOT / "data/01_graphs/citation_graph.json"


def load_enriched(path: Path) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]], dict[str, set[str]]]:
    """Load enriched elements, return (paper_kws, elem_info, elem_kws)."""
    with open(path) as f:
        data = json.load(f)

    paper_kws: dict[str, set[str]] = defaultdict(set)
    elem_info: dict[str, dict[str, Any]] = {}
    elem_kws: dict[str, set[str]] = {}

    for doc_id, doc in data.get("documents", {}).items():
        for elem_id, elem in doc["elements"].items():
            meta = elem.get("enriched_metadata", {}) or {}
            # Only use enriched_metadata.keywords (already curated per-element by the enrichment pipeline)
            kws = set()
            for kw in meta.get("keywords", []):
                kw = kw.strip().lower()
                if len(kw) >= 4 and kw not in VISUAL_PATTERNS:
                    kws.add(kw)

            paper_kws[doc_id].update(kws)
            elem_kws[elem_id] = kws
            elem_info[elem_id] = {
                "doc_id": doc_id,
                "element_type": elem["element_type"],
                "caption": elem.get("caption", ""),
                "content": elem.get("content", ""),
                "enriched_title": elem.get("enriched_title", ""),
                "enriched_content": elem.get("enriched_content", ""),
                "enriched_metadata": meta,
                "image_path": elem.get("image_path", ""),
            }
    return dict(paper_kws), elem_info, elem_kws


def compute_idf(paper_kws: dict[str, set[str]], elem_kws: dict[str, set[str]]) -> dict[str, float]:
    """Compute IDF: log(N / df(kw)). Rare keywords are more specific bridges."""
    N = len(paper_kws)
    df: dict[str, int] = defaultdict(int)
    for _elem_id, kws in elem_kws.items():
        for kw in kws:
            df[kw] += 1
    idf = {}
    for kw, d in df.items():
        idf[kw] = math.log((N + 1) / (d + 1)) + 1.0  # smooth
    return idf


# Visual-only patterns to downweight (they don't bridge research content)
VISUAL_PATTERNS = {
    "histogram", "bar chart", "line chart", "scatter plot", "heatmap", "box plot",
    "pie chart", "screenshot", "diagram", "flowchart", "error bars", "error bar",
    "confusion matrix", "time series", "frequency", "distribution", "density",
    "standard deviation", "mean", "median", "quartile", "whiskers", "boxplot",
    "x-axis", "y-axis", "colorbar", "legend", "title", "label", "caption",
    "grayscale", "pose", "object size", "bin counts", "bins", "missing content",
    "scientific paper", "table", "marker", "icon", "glyph", "typographic",
    "remark marker", "square symbol", "document icon", "hollow square",
    "num faces", "female", "male",  # demographic labels, not research bridges
    "outline", "square", "placeholder", "icon", "symbol", "marker",
    "typo", "glyph", "separator", "background", "border", "frame",
    "row", "column", "cell", "spacing", "alignment", "padding",
    "font", "bold", "italic", "underline", "strikethrough",
}


def is_semantic_kw(kw: str) -> bool:
    """Filter out purely visual/format keywords."""
    return kw.lower() not in VISUAL_PATTERNS and len(kw) >= 3


def load_citation_edges(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load citation graph edges, return {(source, target): edge_info}."""
    if not path.exists():
        return {}
    with open(path) as f:
        cg = json.load(f)
    edges = {}
    for e in cg.get("edges", []):
        key = (e["source"], e["target"])
        edges[key] = e
    return edges


def build_entity_bridged_pairs(
    paper_kws: dict[str, set[str]],
    elem_info: dict[str, dict[str, Any]],
    elem_kws: dict[str, set[str]],
    idf: dict[str, float],
    citation_edges: dict[tuple[str, str], dict[str, Any]],
    min_shared_entities: int = 2,
    min_elem_entity_overlap: int = 1,
    min_idf: float = 2.0,
    max_per_pair: int = 10,
    total_limit: int = 200,
) -> list[dict[str, Any]]:
    """Build entity-bridged element pairs across papers."""

    # Group elements by doc
    doc_elems: dict[str, list[str]] = defaultdict(list)
    for eid, info in elem_info.items():
        doc_elems[info["doc_id"]].append(eid)

    # For each paper pair with shared entities, find element pairs
    candidates: list[dict[str, Any]] = []
    doc_ids = sorted(paper_kws.keys())

    for i, doc_a in enumerate(doc_ids):
        for doc_b in doc_ids[i + 1:]:
            shared = paper_kws[doc_a] & paper_kws[doc_b]
            semantic_shared = {kw for kw in shared
                              if is_semantic_kw(kw) and idf.get(kw, 0.0) >= min_idf}
            if len(semantic_shared) < min_shared_entities:
                continue

            # Score shared entities by IDF (already filtered)
            sorted_shared = sorted(semantic_shared, key=lambda kw: -idf.get(kw, 0.0))
            bridge_score = sum(idf.get(kw, 0.0) for kw in semantic_shared)

            # Check if there's a citation link between these papers
            cite_key = (doc_a, doc_b)
            cite_rev = (doc_b, doc_a)
            cite_info = citation_edges.get(cite_key) or citation_edges.get(cite_rev)
            cite_prob = cite_info.get("confidence", 0.0) if cite_info else None
            cite_direction = "forward" if citation_edges.get(cite_key) else (
                "reverse" if citation_edges.get(cite_rev) else "none"
            )

            # For each entity, find elements in both papers that mention it
            pair_scores: dict[tuple[str, str], float] = defaultdict(float)
            pair_entities: dict[tuple[str, str], list[str]] = defaultdict(list)

            for kw in sorted_shared:
                # Which elements in doc A mention this keyword?
                a_elems = [e for e in doc_elems[doc_a]
                          if kw in elem_kws.get(e, set())]
                b_elems = [e for e in doc_elems[doc_b]
                          if kw in elem_kws.get(e, set())]
                if a_elems and b_elems:
                    for ea in a_elems:
                        for eb in b_elems:
                            pair = (ea, eb)
                            kw_weight = idf.get(kw, 1.0)
                            pair_scores[pair] += kw_weight
                            pair_entities[pair].append(kw)

            if not pair_scores:
                continue

            # Filter: element pairs must share >= min_elem_entity_overlap keywords
            pair_scores = {
                k: v for k, v in pair_scores.items()
                if len(pair_entities[k]) >= min_elem_entity_overlap
            }
            if not pair_scores:
                continue

            # Take top pairs for this paper pair
            sorted_pairs = sorted(pair_scores.items(), key=lambda x: -x[1])[:max_per_pair]
            for (ea, eb), score in sorted_pairs:
                entities = pair_entities[(ea, eb)]
                info_a = elem_info[ea]
                info_b = elem_info[eb]

                # Determine stratum by keyword specificity
                max_idf = max(idf.get(kw, 0.0) for kw in entities)
                avg_idf = sum(idf.get(kw, 0.0) for kw in entities) / len(entities)
                if avg_idf >= 3.0:
                    stratum = "entity_high_specificity"
                elif avg_idf >= 2.0:
                    stratum = "entity_medium_specificity"
                else:
                    stratum = "entity_low_specificity"

                pair_type = f"{info_a['element_type']}+{info_b['element_type']}"
                bridge_text = (
                    f"Shared entities: {', '.join(entities)}\n"
                    f"Paper [{doc_a}] and [{doc_b}] both discuss these concepts.\n\n"
                    f"Source element [{ea}] enriched: {info_a['enriched_title']}\n"
                    f"  {info_a['enriched_content'][:300]}\n\n"
                    f"Target element [{eb}] enriched: {info_b['enriched_title']}\n"
                    f"  {info_b['enriched_content'][:300]}"
                )

                candidates.append({
                    "candidate_id": f"entity_bridge_{len(candidates)+1:05d}",
                    "judge_index": len(candidates) + 1,
                    "target_stratum": stratum,
                    "target_anchor_reason": f"shared_entity_{len(entities)}",
                    "source_doc": doc_a,
                    "target_doc": doc_b,
                    "source_element_id": ea,
                    "target_element_id": eb,
                    "source_element_type": info_a["element_type"],
                    "target_element_type": info_b["element_type"],
                    "pair_type": pair_type,
                    "source_caption_or_content": info_a["caption"] or info_a["content"],
                    "target_caption_or_content": info_b["caption"] or info_b["content"],
                    "citation_bridge_text": bridge_text,
                    "section_title": "",
                    "citation_probability": cite_prob,
                    "source_resolution_method": "entity_bridge",
                    "target_resolution_method": "entity_bridge",
                    "target_resolution_score": round(score, 4),
                    "target_resolution_detail": {
                        "ref_text": "; ".join(entities),
                        "ref_window_before": f"Entity bridge score: {score:.3f}, avg_idf: {avg_idf:.3f}",
                        "ref_window_after": f"Bridge paper score: {bridge_score:.3f}, cite: {cite_direction}",
                    },
                    "citation_fanout": len(entities),
                    "source_fanout_penalty": 1.0,
                    "quality_score": round(score / 20.0, 5),  # normalized
                    "element_a_image_path": info_a["image_path"],
                    "element_b_image_path": info_b["image_path"],
                    "question_for_judge": (
                        f"Do element [{ea}] from [{doc_a}] and element [{eb}] from [{doc_b}] "
                        f"both specifically discuss the shared entity/concept [{entities[0]}] "
                        f"(and possibly {', '.join(entities[1:4])}) such that they form a "
                        f"semantically meaningful cross-document evidence chain? "
                        f"The bridge is the shared research concept(s), not a citation paragraph."
                    ),
                    "_meta": {
                        "shared_entities": entities,
                        "entity_bridge_score": score,
                        "avg_idf": round(avg_idf, 3),
                        "paper_bridge_score": round(bridge_score, 3),
                        "citation_direction": cite_direction,
                    },
                })

            if len(candidates) >= total_limit:
                break
        if len(candidates) >= total_limit:
            break

    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Build entity-bridged cross-document element candidates"
    )
    parser.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    parser.add_argument("--citation-graph", default=str(DEFAULT_CITATION_GRAPH))
    parser.add_argument("--min-shared-entities", type=int, default=2,
                       help="Min shared semantic entities for a paper pair to be considered")
    parser.add_argument("--min-elem-entity-overlap", type=int, default=2,
                       help="Min shared entities for an element pair to be a candidate (2+ means less noise)")
    parser.add_argument("--min-idf", type=float, default=2.0,
                       help="Minimum IDF for a keyword to be used for element linking")
    parser.add_argument("--max-per-pair", type=int, default=10,
                       help="Max element pairs per paper pair")
    parser.add_argument("--limit", type=int, default=200,
                       help="Max total candidates")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    enriched_path = Path(args.enriched)
    if not enriched_path.is_absolute():
        enriched_path = ROOT / enriched_path
    cite_path = Path(args.citation_graph)
    if not cite_path.is_absolute():
        cite_path = ROOT / cite_path

    print(f"Loading enriched elements from {enriched_path}")
    paper_kws, elem_info, elem_kws = load_enriched(enriched_path)
    print(f"  Papers: {len(paper_kws)}")
    print(f"  Elements: {len(elem_info)}")
    print(f"  Unique keywords: {len(set().union(*elem_kws.values()))}")

    idf = compute_idf(paper_kws, elem_kws)
    cite_edges = load_citation_edges(cite_path)
    print(f"  Citation edges: {len(cite_edges)}")

    candidates = build_entity_bridged_pairs(
        paper_kws, elem_info, elem_kws, idf, cite_edges,
        min_shared_entities=args.min_shared_entities,
        min_elem_entity_overlap=args.min_elem_entity_overlap,
        min_idf=args.min_idf,
        max_per_pair=args.max_per_pair,
        total_limit=args.limit,
    )

    # Output
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir:
        out_dir = ROOT / args.output_dir
    else:
        out_dir = ROOT / f"data/05_eval/entity_bridge_candidates_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pack_path = out_dir / "judge_pack.jsonl"

    with open(pack_path, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Stats
    from collections import Counter
    strata = Counter(c["target_stratum"] for c in candidates)
    pair_types = Counter(c["pair_type"] for c in candidates)
    n_pairs = len(set((c["source_doc"], c["target_doc"]) for c in candidates))

    summary = {
        "status": "ok",
        "created_at": ts,
        "method": "entity_bridge",
        "config": {
            "min_shared_entities": args.min_shared_entities,
            "min_elem_entity_overlap": args.min_elem_entity_overlap,
            "min_idf": args.min_idf,
            "max_per_pair": args.max_per_pair,
            "limit": args.limit,
        },
        "sources": {
            "enriched": str(enriched_path.relative_to(ROOT)),
            "citation_graph": str(cite_path.relative_to(ROOT)) if cite_path.exists() else "none",
        },
        "total_candidates": len(candidates),
        "unique_paper_pairs": n_pairs,
        "unique_source_docs": len(set(c["source_doc"] for c in candidates)),
        "unique_target_docs": len(set(c["target_doc"] for c in candidates)),
        "by_stratum": dict(strata),
        "by_pair_type": dict(pair_types),
        "output_dir": str(out_dir.relative_to(ROOT)),
        "output_file": str(pack_path.relative_to(ROOT)),
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n=== Entity-Bridge Candidates ===")
    print(f"Total candidates: {len(candidates)}")
    print(f"Paper pairs: {n_pairs}")
    print(f"Strata: {dict(strata)}")
    print(f"Pair types: {dict(pair_types)}")
    print(f"\nOutput: {pack_path}")

    # Print some examples
    print(f"\n=== Sample candidates ===")
    for c in candidates[:5]:
        meta = c["_meta"]
        print(f"\n{c['candidate_id']} ({c['target_stratum']})")
        print(f"  {c['source_doc']} <-> {c['target_doc']}")
        print(f"  Source: {c['source_element_id']}")
        print(f"  Target: {c['target_element_id']}")
        print(f"  Entities: {meta['shared_entities'][:8]}")
        print(f"  Score: {meta['entity_bridge_score']:.3f}, IDF: {meta['avg_idf']:.3f}")
        print(f"  Cite: {meta['citation_direction']}")


if __name__ == "__main__":
    main()
