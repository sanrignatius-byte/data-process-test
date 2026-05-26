#!/usr/bin/env python3
"""
Build cross-document long chains for the 53-paper experiment subset
using the entity-bridge approach.

Pipeline:
  1. Load enriched keywords for the 53 docs (from mineru_topology_graph_v1)
  2. Build entity-bridge element pairs (cross-doc, linked by shared keywords)
  3. Build paper-level entity graph (nodes=papers, edges=shared entities)
  4. Enumerate multi-hop paper paths (2-hop cross-doc chains)
  5. Select specific element sequences for each chain
  6. Output chain materials

Cost: $0 (no LLM calls — uses existing enriched keywords only)
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
DEFAULT_TOPOLOGY_GRAPH = ROOT / "data/05_eval/mineru_topology_graph_v1_latest/graph.json"
DEFAULT_TOPOLOGY_SUMMARY = ROOT / "data/05_eval/mineru_topology_graph_v1_latest/summary.json"

VISUAL_PATTERNS = {
    "histogram", "bar chart", "line chart", "scatter plot", "heatmap", "box plot",
    "pie chart", "screenshot", "diagram", "flowchart", "error bars", "error bar",
    "confusion matrix", "time series", "frequency", "distribution", "density",
    "standard deviation", "mean", "median", "quartile", "whiskers", "boxplot",
    "x-axis", "y-axis", "colorbar", "legend", "title", "label", "caption",
    "grayscale", "pose", "object size", "bin counts", "bins", "missing content",
    "scientific paper", "table", "marker", "icon", "glyph", "typographic",
    "remark marker", "square symbol", "document icon", "hollow square",
    "num faces", "female", "male",
    "outline", "square", "placeholder", "icon", "symbol", "marker",
    "typo", "glyph", "separator", "background", "border", "frame",
    "row", "column", "cell", "spacing", "alignment", "padding",
    "font", "bold", "italic", "underline", "strikethrough",
}


def get_53_docs() -> list[str]:
    """Extract the 53 doc IDs from the MinerU topology graph summary."""
    with open(DEFAULT_TOPOLOGY_SUMMARY) as f:
        s = json.load(f)
    return sorted(s["backbone_reachability"]["component_counts"].keys())


def load_enriched_keywords(path: Path, doc_filter: set[str] | None = None):
    """Load enriched keywords for docs, return (paper_kws, elem_info, elem_kws)."""
    with open(path) as f:
        data = json.load(f)

    paper_kws: dict[str, set[str]] = defaultdict(set)
    elem_info: dict[str, dict[str, Any]] = {}
    elem_kws: dict[str, set[str]] = {}

    for doc_id, doc in data.get("documents", {}).items():
        if doc_filter and doc_id not in doc_filter:
            continue
        for elem_id, elem in doc["elements"].items():
            meta = elem.get("enriched_metadata", {}) or {}
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
                "image_path": elem.get("image_path", ""),
            }
    return dict(paper_kws), elem_info, elem_kws


def compute_idf(elem_kws: dict[str, set[str]], N: int) -> dict[str, float]:
    df: dict[str, int] = defaultdict(int)
    for kws in elem_kws.values():
        for kw in kws:
            df[kw] += 1
    return {kw: math.log((N + 1) / (d + 1)) + 1.0 for kw, d in df.items()}


def build_entity_bridge_pairs(
    paper_kws: dict[str, set[str]],
    elem_info: dict[str, dict[str, Any]],
    elem_kws: dict[str, set[str]],
    idf: dict[str, float],
    min_idf: float = 2.5,
    min_elem_overlap: int = 2,
) -> list[dict[str, Any]]:
    """Build cross-doc element pairs linked by shared high-IDF keywords."""

    doc_elems: dict[str, list[str]] = defaultdict(list)
    for eid, info in elem_info.items():
        doc_elems[info["doc_id"]].append(eid)

    pairs: list[dict[str, Any]] = []
    doc_ids = sorted(paper_kws.keys())

    for i, doc_a in enumerate(doc_ids):
        for doc_b in doc_ids[i + 1:]:
            shared = paper_kws[doc_a] & paper_kws[doc_b]
            qualified = {kw for kw in shared if idf.get(kw, 0) >= min_idf}
            if len(qualified) < 2:
                continue

            sorted_shared = sorted(qualified, key=lambda k: -idf.get(k, 0))
            # Score each element pair by keyword overlap
            pair_scores: dict[tuple[str, str], float] = defaultdict(float)
            pair_entities: dict[tuple[str, str], list[str]] = defaultdict(list)

            for kw in sorted_shared:
                a_elems = [e for e in doc_elems[doc_a] if kw in elem_kws.get(e, set())]
                b_elems = [e for e in doc_elems[doc_b] if kw in elem_kws.get(e, set())]
                if a_elems and b_elems:
                    kw_weight = idf.get(kw, 1.0)
                    for ea in a_elems:
                        for eb in b_elems:
                            pair_scores[(ea, eb)] += kw_weight
                            pair_entities[(ea, eb)].append(kw)

            pair_scores = {k: v for k, v in pair_scores.items()
                          if len(pair_entities[k]) >= min_elem_overlap}

            for (ea, eb), score in sorted(pair_scores.items(), key=lambda x: -x[1])[:5]:
                info_a = elem_info[ea]
                info_b = elem_info[eb]
                entities = pair_entities[(ea, eb)]
                pairs.append({
                    "source_doc": doc_a,
                    "target_doc": doc_b,
                    "source_element_id": ea,
                    "target_element_id": eb,
                    "source_element_type": info_a["element_type"],
                    "target_element_type": info_b["element_type"],
                    "shared_entities": entities,
                    "bridge_score": round(score, 3),
                    "source_caption": info_a["caption"] or info_a["content"],
                    "target_caption": info_b["caption"] or info_b["content"],
                })
    return pairs


def build_paper_entity_graph(
    pairs: list[dict[str, Any]]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Build paper-level graph: paper -> {neighbor -> [entity-bridge pairs]}."""
    graph: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        a, b = p["source_doc"], p["target_doc"]
        graph[a][b].append(p)
        graph[b][a].append(p)
    return dict(graph)


def orient_bridge_pair(
    pair: dict[str, Any], from_doc: str, to_doc: str
) -> tuple[str, str] | None:
    """Return element ids oriented for from_doc -> to_doc.

    Entity bridge pairs are stored once with source_doc/target_doc, while the
    paper graph is undirected. Every hop must therefore orient the element ids
    to match the enumerated paper path before rendering chain materials.
    """
    if pair["source_doc"] == from_doc and pair["target_doc"] == to_doc:
        return pair["source_element_id"], pair["target_element_id"]
    if pair["source_doc"] == to_doc and pair["target_doc"] == from_doc:
        return pair["target_element_id"], pair["source_element_id"]
    return None


def enumerate_cross_doc_chains(
    paper_graph: dict[str, dict[str, list[dict[str, Any]]]],
    max_hops: int = 2,
    max_chains: int = 200,
) -> list[dict[str, Any]]:
    """Enumerate multi-hop cross-doc paths. max_hops=2 means 3-paper chains (2 cross-doc hops)."""
    chains: list[dict[str, Any]] = []

    for doc_a in sorted(paper_graph.keys()):
        for doc_b in sorted(paper_graph[doc_a].keys()):
            if doc_b <= doc_a:
                continue

            # 1-hop chain: A → B (the basic entity-bridge pair)
            bridge_pairs_ab = paper_graph[doc_a][doc_b]
            for bp in bridge_pairs_ab[:3]:  # top 3 element pairs per paper pair
                oriented = orient_bridge_pair(bp, doc_a, doc_b)
                if oriented is None:
                    continue
                from_elem, to_elem = oriented
                chains.append({
                    "chain_id": f"eb1_{doc_a}_{doc_b}_{from_elem}_{to_elem}",
                    "paper_path": [doc_a, doc_b],
                    "cross_doc_hops": 1,
                    "hops": [{
                        "from_doc": doc_a,
                        "to_doc": doc_b,
                        "from_element": from_elem,
                        "to_element": to_elem,
                        "bridge_type": "entity_bridge",
                        "shared_entities": bp["shared_entities"],
                        "bridge_score": bp["bridge_score"],
                    }],
                    "total_score": bp["bridge_score"],
                })

            if max_hops < 2:
                continue

            # 2-hop chain: A → B → C
            for doc_c in sorted(paper_graph[doc_b].keys()):
                if doc_c == doc_a:
                    continue
                bridge_pairs_bc = paper_graph[doc_b][doc_c]

                for bp_ab in bridge_pairs_ab[:2]:
                    for bp_bc in bridge_pairs_bc[:2]:
                        oriented_ab = orient_bridge_pair(bp_ab, doc_a, doc_b)
                        oriented_bc = orient_bridge_pair(bp_bc, doc_b, doc_c)
                        if oriented_ab is None or oriented_bc is None:
                            continue
                        ab_from_elem, ab_to_elem = oriented_ab
                        bc_from_elem, bc_to_elem = oriented_bc
                        score = bp_ab["bridge_score"] + bp_bc["bridge_score"]
                        # Check if the "joint" paper B elements share entities
                        b_elem_ab = set(bp_ab["shared_entities"])
                        b_elem_bc = set(bp_bc["shared_entities"])
                        joint_entities = b_elem_ab & b_elem_bc

                        chains.append({
                            "chain_id": f"eb2_{doc_a}_{doc_b}_{doc_c}_{len(chains)}",
                            "paper_path": [doc_a, doc_b, doc_c],
                            "cross_doc_hops": 2,
                            "hops": [
                                {
                                    "from_doc": doc_a,
                                    "to_doc": doc_b,
                                    "from_element": ab_from_elem,
                                    "to_element": ab_to_elem,
                                    "bridge_type": "entity_bridge",
                                    "shared_entities": bp_ab["shared_entities"],
                                    "bridge_score": bp_ab["bridge_score"],
                                },
                                {
                                    "from_doc": doc_b,
                                    "to_doc": doc_c,
                                    "from_element": bc_from_elem,
                                    "to_element": bc_to_elem,
                                    "bridge_type": "entity_bridge",
                                    "shared_entities": bp_bc["shared_entities"],
                                    "bridge_score": bp_bc["bridge_score"],
                                },
                            ],
                            "joint_entities_at_b": sorted(joint_entities),
                            "total_score": round(score, 3),
                        })

                if len(chains) >= max_chains:
                    break
            if len(chains) >= max_chains:
                break
        if len(chains) >= max_chains:
            break

    chains.sort(key=lambda c: -c["total_score"])
    return chains[:max_chains]


def render_chain_material(chain: dict[str, Any], elem_info: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Render a chain into structured M4 material format."""
    elements = []
    seen_elements: set[str] = set()
    bridge_texts = []

    def add_element(doc_id: str, element_id: str, role: str) -> None:
        if element_id in seen_elements:
            return
        info = elem_info.get(element_id, {})
        elements.append({
            "doc_id": doc_id,
            "element_id": element_id,
            "element_type": info.get("element_type", ""),
            "caption": info.get("caption", "")[:300],
            "enriched_title": info.get("enriched_title", ""),
            "role": role,
        })
        seen_elements.add(element_id)

    last_hop = chain["hops"][-1]
    for hop in chain["hops"]:
        add_element(
            hop["from_doc"],
            hop["from_element"],
            "chain_start" if not elements else "chain_joint",
        )
        add_element(
            hop["to_doc"],
            hop["to_element"],
            "chain_end" if hop == last_hop else "chain_joint",
        )

        bridge_texts.append({
            "from_doc": hop["from_doc"],
            "to_doc": hop["to_doc"],
            "from_element_id": hop["from_element"],
            "to_element_id": hop["to_element"],
            "shared_entities": hop["shared_entities"],
            "bridge_score": hop["bridge_score"],
            "bridge_description": (
                f"Papers [{hop['from_doc']}] and [{hop['to_doc']}] both discuss: "
                f"{', '.join(hop['shared_entities'][:5])}"
            ),
        })

    return {
        "chain_id": chain["chain_id"],
        "paper_path": chain["paper_path"],
        "cross_doc_hops": chain["cross_doc_hops"],
        "total_score": chain["total_score"],
        "elements": elements,
        "bridges": bridge_texts,
        "joint_entities": chain.get("joint_entities_at_b", []),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build entity-bridge cross-document long chains for the 53-paper subset"
    )
    parser.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    parser.add_argument("--min-idf", type=float, default=2.5)
    parser.add_argument("--min-elem-overlap", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=2,
                       help="Max cross-doc hops (2 = 3-paper chains)")
    parser.add_argument("--max-chains", type=int, default=200)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    enriched_path = Path(args.enriched)
    if not enriched_path.is_absolute():
        enriched_path = ROOT / enriched_path

    # 1. Get the 53 doc IDs
    docs_53 = set(get_53_docs())
    print(f"Target docs (old_53 subset): {len(docs_53)}")

    # 2. Load enriched keywords for those 53 docs
    paper_kws, elem_info, elem_kws = load_enriched_keywords(enriched_path, doc_filter=docs_53)
    print(f"Docs with enriched data: {len(paper_kws)}")
    print(f"Total elements: {len(elem_info)}")

    # 3. Compute IDF on the 53-doc corpus
    idf = compute_idf(elem_kws, N=len(paper_kws))
    print(f"Unique keywords: {len(idf)}")

    # 4. Build entity-bridge pairs
    pairs = build_entity_bridge_pairs(
        paper_kws, elem_info, elem_kws, idf,
        min_idf=args.min_idf,
        min_elem_overlap=args.min_elem_overlap,
    )
    print(f"Entity-bridge element pairs: {len(pairs)}")
    paper_pairs = len(set((p["source_doc"], p["target_doc"]) for p in pairs))
    print(f"Unique paper pairs with bridges: {paper_pairs}")

    # 5. Build paper-level entity graph
    paper_graph = build_paper_entity_graph(pairs)
    print(f"Papers in entity graph: {len(paper_graph)}")

    # 6. Enumerate cross-doc chains
    chains = enumerate_cross_doc_chains(
        paper_graph, max_hops=args.max_hops, max_chains=args.max_chains
    )
    hop1 = sum(1 for c in chains if c["cross_doc_hops"] == 1)
    hop2 = sum(1 for c in chains if c["cross_doc_hops"] == 2)
    print(f"Chains: {len(chains)} total ({hop1} 1-hop, {hop2} 2-hop)")

    # 7. Render chain materials
    materials = [render_chain_material(c, elem_info) for c in chains]

    # 8. Output
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir:
        out_dir = ROOT / args.output_dir
    else:
        out_dir = ROOT / f"data/05_eval/entity_bridge_chains_53_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write entity-bridge pairs
    pairs_path = out_dir / "entity_bridge_pairs.jsonl"
    with open(pairs_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Write chain materials
    chains_path = out_dir / "chains.jsonl"
    with open(chains_path, "w", encoding="utf-8") as f:
        for m in materials:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter
    chain_lengths = Counter(c["cross_doc_hops"] for c in chains)
    chain_docs = set()
    for c in chains:
        for d in c["paper_path"]:
            chain_docs.add(d)

    # Distribution of shared entities across bridges
    all_entities = Counter()
    for p in pairs:
        for e in p["shared_entities"]:
            all_entities[e] += 1

    summary = {
        "status": "ok",
        "created_at": ts,
        "method": "entity_bridge_chains",
        "corpus": "old_53",
        "config": {
            "min_idf": args.min_idf,
            "min_elem_overlap": args.min_elem_overlap,
            "max_hops": args.max_hops,
        },
        "entity_bridge_pairs": len(pairs),
        "unique_paper_pairs_with_bridges": paper_pairs,
        "papers_in_entity_graph": len(paper_graph),
        "total_chains": len(chains),
        "by_hop_count": {str(k): v for k, v in chain_lengths.items()},
        "papers_in_chains": len(chain_docs),
        "top_bridge_entities": all_entities.most_common(30),
        "output_dir": str(out_dir.relative_to(ROOT)),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n=== Entity-Bridge Chains (old_53) ===")
    print(f"Entity-bridge pairs: {len(pairs)}")
    print(f"Paper pairs with bridges: {paper_pairs}")
    print(f"Chains total: {len(chains)}")
    print(f"  1-hop: {hop1}")
    print(f"  2-hop: {hop2}")
    print(f"Papers connected in chains: {len(chain_docs)}/{len(docs_53)}")
    print(f"\nTop bridge entities:")
    for kw, cnt in all_entities.most_common(15):
        print(f"  {kw}: {cnt}")
    print(f"\nOutput: {out_dir}")

    # Print a few example chains
    print(f"\n=== Example chains ===")
    for c in chains[:5]:
        path = " → ".join(c["paper_path"])
        n_entities = len(c["hops"][0]["shared_entities"])
        print(f"\n{c['chain_id']}")
        print(f"  Path: {path}")
        print(f"  Hops: {c['cross_doc_hops']}, Score: {c['total_score']}")
        print(f"  Bridge 1: {c['hops'][0]['shared_entities'][:5]} ({n_entities} total)")
        if c['cross_doc_hops'] >= 2:
            n2 = len(c['hops'][1]['shared_entities'])
            print(f"  Bridge 2: {c['hops'][1]['shared_entities'][:5]} ({n2} total)")
            print(f"  Joint entities at B: {c.get('joint_entities_at_b', [])[:5]}")


if __name__ == "__main__":
    main()
