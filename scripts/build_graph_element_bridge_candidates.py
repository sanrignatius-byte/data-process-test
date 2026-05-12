#!/usr/bin/env python3
"""Build graph-derived 2/3-element candidates with explicit bridge nodes.

This is a fuller resource-prep lane than ``analyze_latex_graph_topology.py``:

* 2 multimodal elements connected by 0-2 bridge nodes, where 0 means a direct
  element-ref edge.
* 3 multimodal elements connected by 0-4 bridge nodes, keeping direct
  element-ref segments when they appear.
* Candidates are scored structurally, then the top fraction per kind is kept.
* A missing-enrichment subset is emitted for elements not already covered by
  existing MoDora-style enriched overlays.

The output is intentionally pair-file-like: ``pairs`` contains ``element_a``,
``element_b``, and ``node_group`` so existing injection/generation utilities can
reuse it, while ``bridge_nodes`` keeps the extra graph evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_latex_graph_topology import (  # noqa: E402
    ELEMENT_MODALITIES,
    build_cross_doc_edges,
    build_multimodal_index,
    build_topology_graph,
)


BRIDGE_NODE_TYPES = {"paragraph", "section", "subsection", "subsubsection"}
ENRICH_KEYS = ("enriched_title", "enriched_metadata", "enriched_content", "enrichment_issues")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_elements(doc: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    elements = doc.get("elements", {})
    if isinstance(elements, dict):
        yield from elements.items()
    elif isinstance(elements, list):
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            eid = str(elem.get("element_id", "") or "").strip()
            if eid:
                yield eid, elem


def build_element_index(mm_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for doc in (mm_data.get("documents") or {}).values():
        for eid, elem in iter_elements(doc):
            idx[eid] = elem
    return idx


def build_enriched_index(paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            print(f"[warn] enriched overlay missing, skipping: {path}")
            continue
        data = load_json(path)
        added = 0
        for doc in (data.get("documents") or {}).values():
            for eid, elem in iter_elements(doc):
                if any(elem.get(k) for k in ENRICH_KEYS[:3]):
                    idx[eid] = elem
                    added += 1
        print(f"[info] enriched overlay {path}: +{added} usable entries")
    print(f"[info] enriched index total: {len(idx)}")
    return idx


def norm_type(node_type: str) -> str:
    return "formula" if node_type in {"equation", "formula"} else node_type


def make_element_dict(elem: Dict[str, Any], enriched_idx: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    eid = str(elem.get("element_id", "") or "")
    out = {
        "element_id": eid,
        "doc_id": elem.get("doc_id", ""),
        "element_type": elem.get("element_type", ""),
        "caption": elem.get("caption", "") or "",
        "content": elem.get("content", "") or "",
        "image_path": elem.get("image_path", "") or "",
        "context_before": (elem.get("context_before", "") or "")[:300],
        "context_after": (elem.get("context_after", "") or "")[:300],
    }
    src = enriched_idx.get(eid)
    if src:
        for key in ENRICH_KEYS:
            if key in src:
                out[key] = src[key]
    return out


def bridge_text(node: Any) -> str:
    if getattr(node, "text_snippet", None):
        return str(node.text_snippet)
    if getattr(node, "section_title", None):
        return str(node.section_title)
    return str(getattr(node, "label", "") or "")


def bridge_quality(text: str) -> float:
    text = " ".join((text or "").split())
    if len(text) < 30:
        return 0.15
    words = text.split()
    verbs = sum(1 for w in words if w.lower().endswith(("s", "ed", "ing")) or w.lower() in {
        "is", "are", "was", "were", "shows", "uses", "compares", "indicates",
        "demonstrates", "explains", "requires", "improves", "reduces",
    })
    ref_bonus = 0.15 if any(tok in text.lower() for tok in ("figure", "table", "eq.", "equation", "results")) else 0.0
    length_score = min(1.0, len(words) / 80.0)
    verb_score = min(1.0, verbs / max(4.0, len(words) / 12.0))
    return max(0.05, min(1.0, 0.45 * length_score + 0.40 * verb_score + ref_bonus))


def edge_type_index(edges: List[Any]) -> Dict[Tuple[str, str], Set[str]]:
    idx: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for e in edges:
        idx[(e.source_id, e.target_id)].add(e.edge_type)
        idx[(e.target_id, e.source_id)].add(e.edge_type)
    return idx


def build_undirected_adj(edges: List[Any]) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = defaultdict(set)
    for e in edges:
        adj[e.source_id].add(e.target_id)
        adj[e.target_id].add(e.source_id)
    return adj


def canonical_path(path: List[str]) -> Tuple[str, ...]:
    fwd = tuple(path)
    rev = tuple(reversed(path))
    return fwd if fwd <= rev else rev


def candidate_signature(element_ids: List[str], bridge_ids: List[str], direct_edges: List[Tuple[str, str]]) -> Tuple[Any, ...]:
    elem_sig = tuple(sorted(element_ids))
    bridge_sig = tuple(sorted(bridge_ids))
    direct_sig = tuple(sorted(tuple(sorted(x)) for x in direct_edges))
    return elem_sig, bridge_sig, direct_sig


def path_score(
    path: List[str],
    nodes: Dict[str, Any],
    eidx: Dict[Tuple[str, str], Set[str]],
    enriched_idx: Dict[str, Dict[str, Any]],
) -> float:
    elem_nodes = [nid for nid in path if nodes[nid].node_type in ELEMENT_MODALITIES]
    bridge_nodes = [nid for nid in path if nodes[nid].node_type in BRIDGE_NODE_TYPES]
    modalities = {norm_type(nodes[nid].node_type) for nid in elem_nodes}
    modality_score = len(modalities) / max(1, len(elem_nodes))
    bridge_score = (
        sum(bridge_quality(bridge_text(nodes[nid])) for nid in bridge_nodes) / max(1, len(bridge_nodes))
        if bridge_nodes else 0.45
    )
    direct_count = 0
    edge_bonus = 0.0
    for a, b in zip(path, path[1:]):
        types = eidx.get((a, b), set())
        if "element_ref" in types:
            direct_count += 1
            edge_bonus += 0.18
        if "paragraph_ref" in types:
            edge_bonus += 0.10
        if "backbone" in types:
            edge_bonus += 0.06
        if "cross_doc_cite" in types:
            edge_bonus += 0.08
        if any(t.startswith("section_contains") for t in types):
            edge_bonus += 0.04
    positions = [nodes[nid].position_idx for nid in elem_nodes if isinstance(nodes[nid].position_idx, int)]
    if len(positions) >= 2:
        pos_span = max(positions) - min(positions)
        distance_score = 1.0 / (1.0 + math.log1p(max(0, pos_span)))
    else:
        distance_score = 0.35
    enriched_score = sum(
        1 for nid in elem_nodes
        if nodes[nid].mapped_element_id and nodes[nid].mapped_element_id in enriched_idx
    ) / max(1, len(elem_nodes))
    compact_score = 1.0 / max(1.0, len(path) - 1)
    return round(
        0.28 * modality_score
        + 0.24 * bridge_score
        + 0.16 * min(1.0, edge_bonus)
        + 0.14 * distance_score
        + 0.10 * enriched_score
        + 0.08 * compact_score
        + 0.03 * min(1, direct_count),
        6,
    )


def neighbor_sort_key(nid: str, cur: str, nodes: Dict[str, Any], eidx: Dict[Tuple[str, str], Set[str]]) -> Tuple[int, int, str]:
    ntype = nodes[nid].node_type
    types = eidx.get((cur, nid), set())
    if "element_ref" in types:
        pri = 0
    elif ntype in ELEMENT_MODALITIES:
        pri = 1
    elif "paragraph_ref" in types:
        pri = 2
    elif "backbone" in types:
        pri = 3
    elif ntype == "paragraph":
        pri = 4
    elif ntype in {"section", "subsection", "subsubsection"}:
        pri = 5
    else:
        pri = 8
    return pri, len(eidx.get((cur, nid), ())), nid


def enumerate_paths(
    nodes: Dict[str, Any],
    adj: Dict[str, Set[str]],
    eidx: Dict[Tuple[str, str], Set[str]],
    enriched_idx: Dict[str, Dict[str, Any]],
    max_edges: int,
    neighbor_cap: int,
    per_start_cap: int,
    doc_limit: int = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    element_nodes = [
        nid for nid, n in nodes.items()
        if n.node_type in ELEMENT_MODALITIES and n.mapped_element_id
    ]
    if doc_limit > 0:
        doc_allow = set(sorted({nodes[nid].doc_id for nid in element_nodes})[:doc_limit])
        element_nodes = [nid for nid in element_nodes if nodes[nid].doc_id in doc_allow]
    element_nodes.sort(key=lambda nid: (nodes[nid].doc_id, nodes[nid].position_idx or 10**9, nid))

    raw: Dict[str, List[Dict[str, Any]]] = {"2elem": [], "3elem": []}
    seen: Set[Tuple[Any, ...]] = set()

    def record(path: List[str]) -> None:
        elem_nodes = [nid for nid in path if nodes[nid].node_type in ELEMENT_MODALITIES]
        bridge_nodes = [nid for nid in path if nodes[nid].node_type in BRIDGE_NODE_TYPES]
        elem_ids = [nodes[nid].mapped_element_id for nid in elem_nodes if nodes[nid].mapped_element_id]
        if len(elem_nodes) not in (2, 3) or len(set(elem_ids)) != len(elem_nodes):
            return
        modalities = {norm_type(nodes[nid].node_type) for nid in elem_nodes}
        if len(modalities) < 2:
            return
        direct_edges: List[Tuple[str, str]] = []
        for a, b in zip(path, path[1:]):
            if "element_ref" in eidx.get((a, b), set()):
                direct_edges.append((a, b))
        bridge_count = len(bridge_nodes)
        if len(elem_nodes) == 2:
            if bridge_count not in (1, 2) and not direct_edges:
                return
            kind = "2elem"
        else:
            if not (2 <= bridge_count <= 4) and not direct_edges:
                return
            kind = "3elem"
        sig = (kind,) + candidate_signature(elem_ids, bridge_nodes, direct_edges)
        if sig in seen:
            return
        seen.add(sig)
        docs = sorted({nodes[nid].doc_id for nid in elem_nodes})
        raw[kind].append({
            "kind": kind,
            "doc_id": docs[0] if len(docs) == 1 else "cross_doc",
            "doc_ids": docs,
            "path_node_ids": list(path),
            "path_node_types": [nodes[nid].node_type for nid in path],
            "element_node_ids": elem_nodes,
            "element_ids": elem_ids,
            "bridge_node_ids": bridge_nodes,
            "bridge_count": bridge_count,
            "direct_element_ref_count": len(direct_edges),
            "direct_element_ref_edges": direct_edges,
            "modalities_in_path": sorted(modalities),
            "hop_distance": len(path) - 1,
            "score": path_score(path, nodes, eidx, enriched_idx),
            "is_cross_doc": len(docs) > 1,
        })

    for start_idx, start in enumerate(element_nodes, 1):
        per_start_added_before = len(raw["2elem"]) + len(raw["3elem"])

        def dfs(path: List[str], elem_count: int, bridge_count: int) -> None:
            if len(path) - 1 >= max_edges:
                return
            if (len(raw["2elem"]) + len(raw["3elem"]) - per_start_added_before) >= per_start_cap:
                return
            cur = path[-1]
            neigh = [n for n in adj.get(cur, set()) if n in nodes and n not in path]
            neigh.sort(key=lambda n: neighbor_sort_key(n, cur, nodes, eidx))
            for nb in neigh[:neighbor_cap]:
                ntype = nodes[nb].node_type
                next_elem = elem_count + (1 if ntype in ELEMENT_MODALITIES else 0)
                next_bridge = bridge_count + (1 if ntype in BRIDGE_NODE_TYPES else 0)
                if next_elem > 3 or next_bridge > 4:
                    continue
                if ntype not in ELEMENT_MODALITIES and ntype not in BRIDGE_NODE_TYPES:
                    continue
                if ntype in ELEMENT_MODALITIES and not nodes[nb].mapped_element_id:
                    continue
                new_path = path + [nb]
                if ntype in ELEMENT_MODALITIES and next_elem in (2, 3):
                    record(new_path)
                if next_elem < 3:
                    dfs(new_path, next_elem, next_bridge)

        dfs([start], 1, 0)
        if start_idx % 5000 == 0:
            print(f"[enum] starts={start_idx}/{len(element_nodes)} raw2={len(raw['2elem'])} raw3={len(raw['3elem'])}")

    return raw


def select_top(
    raw: List[Dict[str, Any]],
    frac: float,
    max_keep: int,
    bridge_min: int,
    bridge_max: int,
    direct_frac: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Keep top bridge-rich candidates plus a small direct-element-ref bonus set."""
    def sort_key(x: Dict[str, Any]) -> Tuple[float, int, int]:
        return (x["score"], -x["hop_distance"], -x["bridge_count"])

    main = [x for x in raw if bridge_min <= int(x.get("bridge_count", 0)) <= bridge_max]
    direct = [
        x for x in raw
        if int(x.get("direct_element_ref_count", 0)) > 0
        and not (bridge_min <= int(x.get("bridge_count", 0)) <= bridge_max)
    ]
    main_ordered = sorted(main, key=sort_key, reverse=True)
    direct_ordered = sorted(direct, key=sort_key, reverse=True)

    main_keep = max(1, int(math.ceil(len(main_ordered) * frac))) if main_ordered else 0
    direct_keep = int(math.ceil(len(direct_ordered) * direct_frac)) if direct_ordered and direct_frac > 0 else 0
    if max_keep > 0:
        main_keep = min(main_keep, max_keep)
        direct_keep = min(direct_keep, max(0, max_keep - main_keep))

    selected = main_ordered[:main_keep] + direct_ordered[:direct_keep]
    return selected, {
        "raw_main": len(main_ordered),
        "raw_direct_bonus": len(direct_ordered),
        "kept_main": main_keep,
        "kept_direct_bonus": direct_keep,
    }


def build_pair_payload(
    selected: List[Dict[str, Any]],
    nodes: Dict[str, Any],
    element_idx: Dict[str, Dict[str, Any]],
    enriched_idx: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    counter: Dict[str, int] = defaultdict(int)
    for cand in selected:
        element_nodes = cand["element_node_ids"]
        elements = []
        for nid in element_nodes:
            eid = nodes[nid].mapped_element_id
            elem = element_idx.get(eid)
            if elem:
                elements.append(make_element_dict(elem, enriched_idx))
        if len(elements) != len(element_nodes):
            continue
        doc_key = cand["doc_id"]
        counter[doc_key] += 1
        pair_id = f"{doc_key}_{cand['kind']}_graph_bridge_{counter[doc_key]}"
        bridge_nodes = []
        for nid in cand["bridge_node_ids"]:
            node = nodes[nid]
            bridge_nodes.append({
                "node_id": nid,
                "node_type": node.node_type,
                "doc_id": node.doc_id,
                "label": node.label,
                "line_no": node.line_no,
                "line_no_end": node.line_no_end,
                "text": bridge_text(node)[:1000],
                "quality": bridge_quality(bridge_text(node)),
            })
        pair = {
            "pair_id": pair_id,
            "candidate_kind": cand["kind"],
            "doc_id": cand["doc_id"],
            "doc_ids": cand["doc_ids"],
            "element_a_id": elements[0]["element_id"],
            "element_b_id": elements[-1]["element_id"],
            "element_a_type": elements[0]["element_type"],
            "element_b_type": elements[-1]["element_type"],
            "pair_type": "+".join(sorted({e["element_type"] for e in elements})),
            "hop_distance": cand["hop_distance"],
            "bridge_count": cand["bridge_count"],
            "direct_element_ref_count": cand["direct_element_ref_count"],
            "direct_element_ref_edges": cand["direct_element_ref_edges"],
            "path": [nodes[nid].mapped_element_id if nodes[nid].node_type in ELEMENT_MODALITIES else nid for nid in cand["path_node_ids"]],
            "path_node_ids": cand["path_node_ids"],
            "path_node_types": cand["path_node_types"],
            "modalities_in_path": cand["modalities_in_path"],
            "quality_score": cand["score"],
            "is_cross_doc": cand["is_cross_doc"],
            "element_a": elements[0],
            "element_b": elements[-1],
            "node_group": elements,
            "bridge_nodes": bridge_nodes,
            "bridge_contexts": [{"node_id": b["node_id"], "text": b["text"]} for b in bridge_nodes if b["text"]],
            "overlap_with_existing_l1": False,
        }
        pairs.append(pair)
    return pairs


def write_missing_subset(
    pairs: List[Dict[str, Any]],
    mm_data: Dict[str, Any],
    enriched_idx: Dict[str, Dict[str, Any]],
    output: Path,
) -> Dict[str, Any]:
    needed: Set[str] = set()
    for pair in pairs:
        for elem in pair.get("node_group", []):
            eid = str(elem.get("element_id", "") or "")
            if eid and eid not in enriched_idx:
                needed.add(eid)
    docs_out: Dict[str, Any] = {}
    for doc_id, doc in (mm_data.get("documents") or {}).items():
        elems = {eid: elem for eid, elem in iter_elements(doc) if eid in needed}
        if not elems:
            continue
        docs_out[doc_id] = {
            "doc_id": doc_id,
            "num_elements": len(elems),
            "num_edges": 0,
            "elements": elems,
            "edges": [],
            "multimodal_pairs": [],
        }
    payload = {
        "metadata": {
            "source": "graph_element_bridge_candidates_missing_enrichment",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_docs": len(docs_out),
            "total_elements": sum(len(d["elements"]) for d in docs_out.values()),
        },
        "documents": docs_out,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload["metadata"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latex-graph", default="data/01_graphs/latex_reference_graph_v2.json")
    ap.add_argument("--elements", default="data/01_graphs/multimodal_elements_v2.json")
    ap.add_argument("--citation-graph", default="data/01_graphs/citation_graph.json")
    ap.add_argument("--mineru-output", default="data/00_raw/mineru_output")
    ap.add_argument("--enriched-elements", action="append", default=[])
    ap.add_argument("--output", default="data/02_enriched/graph_bridge_element_candidates_20260512.json")
    ap.add_argument("--missing-elements-output", default="data/02_enriched/graph_bridge_missing_elements_20260512.json")
    ap.add_argument("--two-frac", type=float, default=0.25)
    ap.add_argument("--three-frac", type=float, default=0.20)
    ap.add_argument("--direct-frac", type=float, default=0.05,
                    help="Extra fraction of direct element-ref candidates to retain outside bridge-count targets.")
    ap.add_argument("--max-edges", type=int, default=6)
    ap.add_argument("--neighbor-cap", type=int, default=48)
    ap.add_argument("--per-start-cap", type=int, default=24)
    ap.add_argument("--max-keep-two", type=int, default=0)
    ap.add_argument("--max-keep-three", type=int, default=0)
    ap.add_argument("--doc-limit", type=int, default=0, help="Debug only: restrict to first N docs by doc_id")
    args = ap.parse_args()

    latex_data = load_json(Path(args.latex_graph))
    mm_data = load_json(Path(args.elements))
    citation_data = load_json(Path(args.citation_graph)) if Path(args.citation_graph).exists() else {"edges": []}
    enriched_idx = build_enriched_index([Path(p) for p in args.enriched_elements])

    print("[build] multimodal index")
    mm_index = build_multimodal_index(mm_data, mineru_output_dir=args.mineru_output)
    print("[build] topology graph")
    nodes, edges, out_adj, in_adj, doc_stats = build_topology_graph(latex_data, mm_index)
    cross_edges, _cite_pairs = build_cross_doc_edges(citation_data, nodes, out_adj, in_adj)
    edges.extend(cross_edges)
    adj = build_undirected_adj(edges)
    eidx = edge_type_index(edges)
    element_idx = build_element_index(mm_data)

    print(f"[build] nodes={len(nodes)} edges={len(edges)} cross_doc_edges={len(cross_edges)}")
    print("[enum] graph bridge paths")
    raw = enumerate_paths(
        nodes=nodes,
        adj=adj,
        eidx=eidx,
        enriched_idx=enriched_idx,
        max_edges=args.max_edges,
        neighbor_cap=args.neighbor_cap,
        per_start_cap=args.per_start_cap,
        doc_limit=args.doc_limit,
    )
    selected_2, policy_2 = select_top(
        raw["2elem"], args.two_frac, args.max_keep_two,
        bridge_min=1, bridge_max=2, direct_frac=args.direct_frac,
    )
    selected_3, policy_3 = select_top(
        raw["3elem"], args.three_frac, args.max_keep_three,
        bridge_min=2, bridge_max=4, direct_frac=args.direct_frac,
    )
    selected = selected_2 + selected_3
    pairs = build_pair_payload(selected, nodes, element_idx, enriched_idx)
    missing_meta = write_missing_subset(pairs, mm_data, enriched_idx, Path(args.missing_elements_output))

    summary = {
        "raw_candidates": {"2elem": len(raw["2elem"]), "3elem": len(raw["3elem"])},
        "selected_candidates": {"2elem": len(selected_2), "3elem": len(selected_3), "total": len(pairs)},
        "selection_policy": {"2elem": policy_2, "3elem": policy_3, "direct_frac": args.direct_frac},
        "by_kind": dict(Counter(p["candidate_kind"] for p in pairs)),
        "by_bridge_count": dict(Counter(str(p["bridge_count"]) for p in pairs)),
        "by_direct_element_ref_count": dict(Counter(str(p["direct_element_ref_count"]) for p in pairs)),
        "by_pair_type": dict(Counter(p["pair_type"] for p in pairs)),
        "docs_covered": len({d for p in pairs for d in p.get("doc_ids", [])}),
        "cross_doc": sum(bool(p.get("is_cross_doc")) for p in pairs),
        "missing_enrichment": missing_meta,
    }
    payload = {
        "metadata": {
            "source": "build_graph_element_bridge_candidates.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "latex_graph": args.latex_graph,
                "elements": args.elements,
                "citation_graph": args.citation_graph,
                "enriched_elements": args.enriched_elements,
            },
            "selection": {
                "two_element_top_fraction": args.two_frac,
                "three_element_top_fraction": args.three_frac,
                "direct_element_ref_bonus_fraction": args.direct_frac,
                "max_edges": args.max_edges,
                "neighbor_cap": args.neighbor_cap,
                "per_start_cap": args.per_start_cap,
            },
        },
        "summary": summary,
        "pairs": pairs,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print("Graph element+bridge candidate build complete")
    print(f"Raw 2elem/3elem:      {len(raw['2elem'])} / {len(raw['3elem'])}")
    print(f"Selected 2elem/3elem: {len(selected_2)} / {len(selected_3)}")
    print(f"Pairs output:         {len(pairs)}")
    print(f"Docs covered:         {summary['docs_covered']}")
    print(f"Missing enrich elems: {missing_meta['total_elements']} across {missing_meta['total_docs']} docs")
    print(f"Output:               {out}")
    print(f"Missing subset:       {args.missing_elements_output}")
    print("=" * 72)


if __name__ == "__main__":
    main()
