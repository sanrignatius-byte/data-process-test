#!/usr/bin/env python3
"""Score hub nodes on the realworld multimodal graph.

Loads `data/01_graphs/realworld_multimodal_elements.json` (produced by
`build_realworld_graph.py`) and computes per-element bridge / connectivity
scores. No academic-paper assumptions — works on 10-Ks, datasheets, RFCs,
legal text alike.

Hub-first philosophy (mirrors `prune_and_score_graph.py` for LaTeX):
    bridge_role        = #distinct neighbour modalities (table/figure/formula)
    cross_modal_degree = #neighbours whose element_type differs from self
    total_degree       = in + out
    score = bridge_role*15 + cross_modal_degree*3 + total_degree*1

Output: `data/01_graphs/realworld_hub_scores.json` containing
    summary       : counts and quantile cuts
    bridge_hubs   : nodes with ≥2 distinct cross-modal neighbours
    top_per_doc   : the 20% highest-scoring nodes per document
"""

import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data/01_graphs/realworld_multimodal_elements.json"
DEFAULT_OUT = ROOT / "data/01_graphs/realworld_hub_scores.json"

MODALITY_TYPES = {"figure", "table", "formula"}
BRIDGEABLE_TYPES = {"text", "section"}  # only these can act as bridges


def build_adjacency(graph: dict) -> tuple[dict, dict, dict]:
    """Return (out_adj, in_adj, edge_type_to_neighbours)."""
    out_adj: dict[str, set[str]] = defaultdict(set)
    in_adj: dict[str, set[str]] = defaultdict(set)
    edge_typed: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for e in graph["edges"]:
        s, t, et = e["source_id"], e["target_id"], e["edge_type"]
        out_adj[s].add(t)
        in_adj[t].add(s)
        edge_typed[s][et].add(t)
        edge_typed[t][et].add(s)
    return out_adj, in_adj, edge_typed


def score_node(
    nid: str,
    elem: dict,
    out_adj: dict,
    in_adj: dict,
    elem_index: dict,
) -> dict:
    self_type = elem["element_type"]
    neighbours = out_adj.get(nid, set()) | in_adj.get(nid, set())
    in_deg = len(in_adj.get(nid, set()))
    out_deg = len(out_adj.get(nid, set()))
    total = in_deg + out_deg

    neighbour_modalities = set()
    cross_modal_degree = 0
    neighbour_type_counts = Counter()
    for nb_id in neighbours:
        nb = elem_index.get(nb_id)
        if not nb:
            continue
        nb_type = nb["element_type"]
        neighbour_type_counts[nb_type] += 1
        if nb_type in MODALITY_TYPES:
            neighbour_modalities.add(nb_type)
        if nb_type != self_type:
            cross_modal_degree += 1

    bridge_role = len(neighbour_modalities)
    # Bridge-first scoring. Tunable:
    #   modality coverage matters most (×15) — paragraphs linking
    #   figure+table+formula are the rarest, most useful hubs.
    score = bridge_role * 15 + cross_modal_degree * 3 + total * 1

    # Type-specific bonus: a section that *contains* many modalities counts
    # almost as much as a paragraph that *references* them.
    if self_type == "section":
        score += min(bridge_role, 3) * 5

    return {
        "element_id": nid,
        "doc_id": elem.get("doc_id", ""),
        "element_type": self_type,
        "in_degree": in_deg,
        "out_degree": out_deg,
        "total_degree": total,
        "neighbour_modalities": sorted(neighbour_modalities),
        "modality_coverage": bridge_role,
        "cross_modal_degree": cross_modal_degree,
        "neighbour_type_counts": dict(neighbour_type_counts),
        "score": score,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_IN))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--top-quantile", type=float, default=0.20,
                    help="Per-doc top fraction to mark as hub (default 0.20)")
    ap.add_argument("--min-bridge-modalities", type=int, default=2,
                    help="A node is a bridge_hub if its modality_coverage ≥ N")
    args = ap.parse_args()

    g = json.load(open(args.input))
    out_adj, in_adj, _ = build_adjacency(g)

    elem_index: dict[str, dict] = {}
    docs = g["documents"]
    for did, doc in docs.items():
        for eid, e in doc["elements"].items():
            elem_index[eid] = e

    scored: list[dict] = []
    for nid, elem in elem_index.items():
        s = score_node(nid, elem, out_adj, in_adj, elem_index)
        if s["total_degree"] == 0:
            continue
        scored.append(s)

    # Per-doc top quantile cut
    by_doc = defaultdict(list)
    for s in scored:
        by_doc[s["doc_id"]].append(s)
    top_per_doc: list[dict] = []
    for did, lst in by_doc.items():
        lst.sort(key=lambda x: -x["score"])
        k = max(1, int(len(lst) * args.top_quantile))
        top_per_doc.extend(lst[:k])

    bridge_hubs = [s for s in scored if s["modality_coverage"] >= args.min_bridge_modalities]
    bridge_hubs.sort(key=lambda x: -x["score"])

    # ── anchor_hubs: high-in-degree tables/sections that aren't bridges ────
    # For docs with no formula/figure mix (10-Ks, legal text), the most useful
    # "core" elements are heavily-referenced tables / sections.  We surface
    # them in a separate channel so downstream tasks can choose either
    # bridge or anchor depending on document genre.
    bridge_ids = {h["element_id"] for h in bridge_hubs}
    anchor_candidates = []
    for s in scored:
        if s["element_id"] in bridge_ids:
            continue
        if s["element_type"] not in ("table", "section"):
            continue
        # Anchor signal: in-degree is what matters (others point at us).
        # Score combines in_degree (×5) + total_degree (×1) so a table cited
        # from many paragraphs out-ranks a section that just contains many.
        anchor_score = s["in_degree"] * 5 + s["total_degree"]
        if anchor_score < 10:
            continue
        anchor_candidates.append({**s, "anchor_score": anchor_score})
    anchor_candidates.sort(key=lambda x: -x["anchor_score"])

    # Decorate hubs with brief content snippets for human inspection
    def _decorate(node: dict) -> dict:
        e = elem_index.get(node["element_id"], {})
        snippet = ""
        if e.get("element_type") == "text":
            snippet = (e.get("content") or "")[:120]
        elif e.get("element_type") == "section":
            snippet = (e.get("label") or e.get("content") or "")[:120]
        elif e.get("element_type") in ("figure", "table"):
            snippet = (e.get("caption") or e.get("label") or "")[:120]
        elif e.get("element_type") == "formula":
            snippet = (e.get("content") or "")[:120]
        return {**node, "snippet": snippet}

    bridge_hubs = [_decorate(h) for h in bridge_hubs]
    top_per_doc = [_decorate(h) for h in top_per_doc]
    anchor_hubs = [_decorate(h) for h in anchor_candidates]

    summary = {
        "input": str(args.input),
        "total_scored_nodes": len(scored),
        "by_type_scored": dict(Counter(s["element_type"] for s in scored)),
        "by_type_bridge_hubs": dict(Counter(s["element_type"] for s in bridge_hubs)),
        "by_type_top_per_doc": dict(Counter(s["element_type"] for s in top_per_doc)),
        "bridge_hub_count": len(bridge_hubs),
        "anchor_hub_count": len(anchor_hubs),
        "by_type_anchor_hubs": dict(Counter(s["element_type"] for s in anchor_hubs)),
        "anchor_hubs_per_doc": dict(Counter(h["doc_id"] for h in anchor_hubs)),
        "top_per_doc_count": len(top_per_doc),
        "top_quantile": args.top_quantile,
        "min_bridge_modalities": args.min_bridge_modalities,
        "score_quartiles": {
            "p50": sorted(s["score"] for s in scored)[len(scored)//2] if scored else 0,
            "p75": sorted(s["score"] for s in scored)[int(len(scored)*0.75)] if scored else 0,
            "p90": sorted(s["score"] for s in scored)[int(len(scored)*0.90)] if scored else 0,
            "p99": sorted(s["score"] for s in scored)[int(len(scored)*0.99)] if scored else 0,
            "max": max(s["score"] for s in scored) if scored else 0,
        },
        "bridge_hubs_per_doc": dict(Counter(h["doc_id"] for h in bridge_hubs)),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "summary": summary,
        "bridge_hubs": bridge_hubs,
        "anchor_hubs": anchor_hubs,
        "top_per_doc": top_per_doc,
    }, ensure_ascii=False, indent=2))

    print(f"Output: {out_path}")
    print(f"Scored nodes (degree>0): {len(scored)}")
    print(f"Bridge hubs (≥{args.min_bridge_modalities} modalities): {len(bridge_hubs)}")
    print(f"Anchor hubs (high-in-degree tables/sections, non-bridge): {len(anchor_hubs)}")
    print(f"Top-{int(args.top_quantile*100)}% per doc: {len(top_per_doc)}")
    print(f"Score quartiles: {summary['score_quartiles']}")
    print(f"By type (bridge_hubs): {summary['by_type_bridge_hubs']}")
    print(f"By type (anchor_hubs): {summary['by_type_anchor_hubs']}")


if __name__ == "__main__":
    main()
