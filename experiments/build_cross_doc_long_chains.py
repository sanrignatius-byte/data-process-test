#!/usr/bin/env python3
"""Build cross-document long chains — combine entity-bridge chains + topology graph.

Start from entity-bridge chains (3-element, 3-paper seeds with 2 cross-doc bridges),
then extend each endpoint by walking intra-doc topology graph to add multimodal elements.
Target: 50 chains with >=5 figure/table/formula elements, >=2 papers.
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOPOLOGY_GRAPH = ROOT / "data/05_eval/mineru_topology_graph_v1_latest/mineru_topology_graph_v1.json"
EB_CHAINS = ROOT / "data/05_eval/entity_bridge_chains_53_fixed_20260522T0910Z/chains.jsonl"

MULTIMODAL_TYPES = {"figure", "table", "formula"}
MIN_ELEMENTS = 5
TARGET_CHAINS = 50


def load_topology(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_mm_graph(topo: dict) -> tuple[dict[str, dict], dict[str, dict[str, list[tuple[str, str]]]], dict[str, str]]:
    """Build multimodal subgraph: node_index, doc_adj, element_to_node_id."""
    node_index: dict[str, dict] = {}
    node_id_to_eid: dict[str, str] = {}
    eid_to_node_id: dict[str, str] = {}

    for n in topo["nodes"]:
        ntype = n.get("node_type", "")
        eid = n.get("element_id", "")
        if ntype in MULTIMODAL_TYPES and eid:
            node_index[eid] = {
                "element_id": eid,
                "doc_id": n.get("doc_id", ""),
                "node_type": ntype,
                "label": n.get("label", "")[:200],
                "text_snippet": n.get("text_snippet", "")[:200],
            }
            node_id_to_eid[n["node_id"]] = eid
            eid_to_node_id[eid] = n["node_id"]

    doc_adj: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for edge in topo["edges"]:
        s_eid = node_id_to_eid.get(edge["source_id"])
        t_eid = node_id_to_eid.get(edge["target_id"])
        if not s_eid or not t_eid or s_eid == t_eid:
            continue
        doc_id = edge.get("doc_id", "")
        etype = edge.get("edge_type", "unknown")
        doc_adj[doc_id][s_eid].append((t_eid, etype))
        doc_adj[doc_id][t_eid].append((s_eid, etype))  # undirected

    return node_index, {k: dict(v) for k, v in doc_adj.items()}, eid_to_node_id


def find_neighbors(
    eid: str, doc_id: str, doc_adj: dict[str, dict[str, list[tuple[str, str]]]],
    max_hops: int,
) -> list[dict]:
    """BFS from eid in doc_id to find chains of connected multimodal elements.

    Returns list of paths {elements: [eid, ...], bridges: [{from,to,doc_id,edge_type}]}
    """
    adj = doc_adj.get(doc_id, {})
    paths: list[dict] = []
    visited: dict[str, int] = {eid: 0}
    queue = deque()

    for neighbor, etype in adj.get(eid, []):
        queue.append((neighbor, [eid, neighbor],
                      [{"type": "intra_doc", "from": eid, "to": neighbor,
                        "doc_id": doc_id, "edge_type": etype}], 1))
        visited[neighbor] = 1

    while queue:
        node, elem_path, bridge_path, depth = queue.popleft()
        if depth >= 2:
            paths.append({"elements": list(elem_path), "bridges": list(bridge_path)})
        if depth >= max_hops:
            continue
        for neighbor, etype in adj.get(node, []):
            if neighbor in elem_path:
                continue
            new_depth = depth + 1
            if neighbor in visited and visited[neighbor] < new_depth:
                continue
            visited[neighbor] = new_depth
            queue.append((neighbor, elem_path + [neighbor],
                          bridge_path + [{"type": "intra_doc", "from": node, "to": neighbor,
                                          "doc_id": doc_id, "edge_type": etype}],
                          new_depth))
    return paths


def score_chain(elements: list[dict], bridges: list[dict], entity_bridges: list[dict]) -> float:
    s = 0.0
    n = len(elements)
    s += min(n - MIN_ELEMENTS + 1, 12) * 2.0
    types = {e.get("node_type", e.get("element_type", "")) for e in elements}
    s += len(types) * 5.0
    tbl = sum(1 for e in elements if e.get("node_type", e.get("element_type", "")) == "table")
    s += min(tbl, 3) * 3.0
    s += len(bridges) * 1.5
    papers = {e.get("doc_id", "") for e in elements}
    s += min(len(papers), 4) * 6.0
    xdoc = sum(1 for b in bridges if b.get("type", "") == "cross_doc_entity")
    if xdoc == 0:
        # Entity-bridge bridges from chains.jsonl don't have "type" field;
        # they're all cross-doc by definition
        xdoc = sum(1 for b in bridges if "from_doc" in b and "to_doc" in b)
    s += xdoc * 5.0
    eb_scores = [eb.get("bridge_score", 0) for eb in entity_bridges]
    s += min(sum(eb_scores), 30) * 0.3
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", default=str(TOPOLOGY_GRAPH))
    parser.add_argument("--eb-chains", default=str(EB_CHAINS))
    parser.add_argument("--min-elements", type=int, default=MIN_ELEMENTS)
    parser.add_argument("--max-intra-hops", type=int, default=2)
    parser.add_argument("--target", type=int, default=TARGET_CHAINS)
    parser.add_argument("--output", default="data/05_eval/cross_doc_long_chains_v3.json")
    parser.add_argument("--output-jsonl", default="data/05_eval/cross_doc_long_chains_v3.jsonl")
    args = parser.parse_args()

    print("Loading topology graph...")
    topo = load_topology(Path(args.topology))
    node_index, doc_adj, eid_to_node = build_mm_graph(topo)
    print(f"  MM elements: {len(node_index)} in {len(doc_adj)} docs")

    print("Loading entity-bridge chains...")
    eb_chains = load_jsonl(Path(args.eb_chains))
    print(f"  EB chains: {len(eb_chains)}")

    print("Building long chains...")
    all_chains: list[dict] = []
    papers_used: set[str] = set()
    seeds_used = 0

    for eb in eb_chains:
        eb_elems = eb.get("elements", [])
        eb_bridges = eb.get("bridges", [])
        eb_mm = [e for e in eb_elems if e.get("element_type") in MULTIMODAL_TYPES]

        if len(eb_mm) < 2:
            continue

        # Normalize eb element keys to match topology naming
        for e in eb_mm:
            if "element_type" in e and "node_type" not in e:
                e["node_type"] = e["element_type"]

        seeds_used += 1

        # For each endpoint element, find topology-graph neighbors in its paper
        extensions_by_elem: dict[str, list[dict]] = {}
        for e in eb_mm:
            eid = e.get("element_id", "")
            doc_id = e.get("doc_id", "")
            if eid in eid_to_node:
                paths = find_neighbors(eid, doc_id, doc_adj, args.max_intra_hops)
                extensions_by_elem[eid] = paths
            else:
                extensions_by_elem[eid] = []

        # Pick one extension path for each endpoint (try all combinations)
        # For efficiency, limit to 2 extensions max
        extendable = [(eid, paths) for eid, paths in extensions_by_elem.items() if paths]
        if not extendable:
            continue

        # Try adding extensions one at a time until we reach >= MIN_ELEMENTS
        best_chain = None
        best_score = -1

        # Strategy: extend from each endpoint, pick the best combination
        for eid, paths in extendable[:1]:  # just take first extendable endpoint
            for path in paths[:5]:  # top 5 paths
                # Build chain: normalized eb_elements + path extension
                chain_elems = list(eb_mm)
                # Add path elements
                for peid in path["elements"][1:]:  # skip first (it's our eb element)
                    if peid in node_index:
                        ni = node_index[peid]
                        if not any(existing.get("element_id") == peid for existing in chain_elems):
                            chain_elems.append({
                                "element_id": peid,
                                "doc_id": ni["doc_id"],
                                "node_type": ni["node_type"],
                                "label": ni["label"],
                                "text_snippet": ni.get("text_snippet", ""),
                            })

                # Bridges: eb bridges + path bridges
                chain_bridges = list(eb_bridges)
                for pb in path["bridges"]:
                    chain_bridges.append(pb)

                # Filter: only multimodal, dedup
                seen = set()
                deduped = []
                for e in chain_elems:
                    ek = e.get("element_id", "")
                    if ek and ek not in seen:
                        seen.add(ek)
                        deduped.append(e)
                chain_elems = deduped

                if len(chain_elems) < args.min_elements:
                    continue

                papers_in_chain = {e.get("doc_id", "") for e in chain_elems}
                if len(papers_in_chain) < 2:
                    continue

                sc = score_chain(chain_elems, chain_bridges, eb_bridges)
                if sc > best_score:
                    best_score = sc
                    best_chain = {
                        "elements": chain_elems,
                        "bridges": chain_bridges,
                        "entity_bridge_score": sum(eb.get("bridge_score", 0) for eb in eb_bridges),
                        "shared_entities": [e for eb in eb_bridges for e in eb.get("shared_entities", [])],
                        "element_types": [e["node_type"] for e in chain_elems],
                        "bridge_types": [b.get("type", "cross_doc_entity") for b in chain_bridges],
                        "papers": sorted(papers_in_chain),
                        "seed_chain_id": eb.get("chain_id", ""),
                        "score": sc,
                    }

        if best_chain:
            all_chains.append(best_chain)
            papers_used.update(best_chain["papers"])

    # Sort, dedup, limit per paper set
    all_chains.sort(key=lambda c: c["score"], reverse=True)

    seen_sets = set()
    seen_paper_sets: dict[str, int] = defaultdict(int)
    final_chains = []
    for c in all_chains:
        if len(final_chains) >= args.target:
            break
        es = frozenset(e["element_id"] for e in c["elements"])
        if es not in seen_sets:
            pp = "|".join(c["papers"])
            if seen_paper_sets[pp] >= 4:
                continue
            seen_sets.add(es)
            seen_paper_sets[pp] += 1
            c["chain_id"] = f"long_xdoc_{len(final_chains):04d}"
            final_chains.append(c)

    print(f"\n  Raw chains: {len(all_chains)}, Unique: {len(final_chains)}")
    print(f"  Seeds used: {seeds_used}/{len(eb_chains)}")
    print(f"  Papers used: {len(papers_used)}")

    if final_chains:
        from collections import Counter
        ec = Counter()
        bc = Counter()
        tc = Counter()
        pc = Counter()
        for c in final_chains:
            ec[len(c["elements"])] += 1
            bc[len(c["bridges"])] += 1
            pc[len(c["papers"])] += 1
            tc.update(c["element_types"])

        print(f"  Elements: min={min(ec)}, max={max(ec)}, dist={dict(sorted(ec.items()))}")
        print(f"  Bridges: min={min(bc)}, max={max(bc)}, dist={dict(sorted(bc.items()))}")
        print(f"  Papers: dist={dict(sorted(pc.items()))}")
        print(f"  Element types: {dict(tc)}")

    # Save
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({
            "config": {"min_elements": args.min_elements, "max_intra_hops": args.max_intra_hops},
            "chains": final_chains,
            "stats": {"total": len(final_chains), "papers_used": len(papers_used)},
        }, f, ensure_ascii=False, indent=2)

    output_jsonl = Path(args.output_jsonl)
    with open(output_jsonl, "w") as f:
        for c in final_chains:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nSaved to {output}")

    # Show sample chains
    print("\n=== Top 3 Chains ===")
    for c in final_chains[:3]:
        xdoc = sum(1 for b in c["bridges"] if b.get("type", "cross_doc_entity") == "cross_doc_entity")
        intra = sum(1 for b in c["bridges"] if b.get("type", "cross_doc_entity") == "intra_doc")
        print(f"\n{c['chain_id']}: score={c['score']:.1f}, papers={c['papers']}, "
              f"elems={len(c['elements'])}, bridges={intra}intra+{xdoc}xdoc")
        print(f"  Seed: {c.get('seed_chain_id', '')}")
        for e in c["elements"]:
            label = e.get('label') or e.get('enriched_title', '') or e.get('text_snippet', '')
            ntype = e.get('node_type', e.get('element_type', '?'))
            print(f"  [{e.get('doc_id','')}] {ntype}: {label[:100]}")
        for b in c["bridges"]:
            if b.get("type", "cross_doc_entity") == "cross_doc_entity":
                print(f"  CROSS: {b['from_doc']}→{b['to_doc']}: {b.get('shared_entities',[])[:3]}")
            else:
                print(f"  intra: {b['from']}→{b['to']} [{b.get('edge_type','?')}]")


if __name__ == "__main__":
    main()
