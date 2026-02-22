#!/usr/bin/env python3
"""Build long-chain candidates from latex_cross_modal_pairs.json.

This script serves two use cases:
1) Pair slicing: derive endpoint pairs from long citation chains (hop >= 2),
   output in the same schema expected by generate_multihop_l1_queries.py.
2) Native chain QA: emit full-chain records preserving intermediate nodes.

Input:
  data/latex_cross_modal_pairs.json   (dict with "pairs")

Outputs:
  data/latex_long_chain_pairs.json    (dict with "pairs")
  data/latex_long_chains.json         (dict with "chains")
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def normalize_pair_type(a_type: str, b_type: str) -> str:
    types = sorted([a_type, b_type])
    return f"{types[0]}+{types[1]}"


def canonical_edge(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def canonical_path(path: Iterable[str]) -> Tuple[str, ...]:
    p = tuple(path)
    r = tuple(reversed(p))
    return p if p <= r else r


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def build_doc_graph(
    doc_pairs: List[Dict[str, Any]],
) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Return adjacency, edge_map, and node_map for one document."""
    adj: Dict[str, Set[str]] = defaultdict(set)
    edge_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    node_map: Dict[str, Dict[str, Any]] = {}

    for p in doc_pairs:
        a = p["element_a_id"]
        b = p["element_b_id"]
        ek = canonical_edge(a, b)

        # Keep the best-quality edge metadata if duplicates exist.
        prev = edge_map.get(ek)
        if prev is None or p.get("quality_score", 0.0) > prev.get("quality_score", 0.0):
            edge_map[ek] = p

        adj[a].add(b)
        adj[b].add(a)

        if a not in node_map:
            node_map[a] = {
                "element_id": a,
                "element_type": p["element_a_type"],
                **(p.get("element_a") or {}),
            }
        if b not in node_map:
            node_map[b] = {
                "element_id": b,
                "element_type": p["element_b_type"],
                **(p.get("element_b") or {}),
            }

    return adj, edge_map, node_map


def chain_quality(path: List[str], edge_map: Dict[Tuple[str, str], Dict[str, Any]]) -> float:
    """Conservative quality: min edge quality with a mild hop penalty."""
    edge_q = []
    for i in range(len(path) - 1):
        e = edge_map[canonical_edge(path[i], path[i + 1])]
        edge_q.append(float(e.get("quality_score", 0.0)))
    if not edge_q:
        return 0.0
    hops = len(path) - 1
    q = min(edge_q) - 0.03 * max(0, hops - 1)
    return max(0.0, round(q, 4))


def aggregate_edge_contexts(
    path: List[str],
    edge_map: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for i in range(len(path) - 1):
        s = path[i]
        t = path[i + 1]
        e = edge_map[canonical_edge(s, t)]
        ecs = e.get("edge_contexts") or []
        if ecs:
            # Keep all edge contexts for this segment.
            for ec in ecs:
                contexts.append({
                    "source": ec.get("source", s),
                    "target": ec.get("target", t),
                    "ref_text": ec.get("ref_text", ""),
                    "context_snippet": ec.get("context_snippet", ""),
                    "segment_index": i,
                })
        else:
            contexts.append({
                "source": s,
                "target": t,
                "ref_text": "",
                "context_snippet": "",
                "segment_index": i,
            })
    return contexts


def aggregate_bridge(path: List[str], edge_map: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    parts: List[str] = []
    strategies: List[str] = []
    confs: List[float] = []
    labels: List[str] = []
    for i in range(len(path) - 1):
        e = edge_map[canonical_edge(path[i], path[i + 1])]
        lb = e.get("latex_bridge") or {}
        txt = (lb.get("bridge_text") or "").strip()
        if txt:
            parts.append(txt[:300])
        st = lb.get("strategy")
        if st:
            strategies.append(st)
        for ck in ("match_conf_a", "match_conf_b"):
            if ck in lb:
                try:
                    confs.append(float(lb.get(ck, 0.0)))
                except Exception:
                    pass
        if lb.get("label_a"):
            labels.append(lb["label_a"])
        if lb.get("label_b"):
            labels.append(lb["label_b"])

    return {
        "bridge_text": "\n\n".join(dedupe_keep_order(parts))[:2000],
        "strategy": "+".join(dedupe_keep_order(strategies)) if strategies else "chain",
        "labels": dedupe_keep_order(labels),
        "avg_match_conf": round(sum(confs) / len(confs), 4) if confs else 0.0,
    }


def enumerate_doc_chains(
    doc_id: str,
    adj: Dict[str, Set[str]],
    edge_map: Dict[Tuple[str, str], Dict[str, Any]],
    node_map: Dict[str, Dict[str, Any]],
    min_hops: int,
    max_hops: int,
    max_chains_per_doc: int,
) -> List[Dict[str, Any]]:
    chains: List[Dict[str, Any]] = []
    seen_paths: Set[Tuple[str, ...]] = set()
    node_ids = sorted(node_map.keys())

    def dfs(cur: str, path: List[str]) -> None:
        if len(path) - 1 > max_hops:
            return

        hops = len(path) - 1
        if min_hops <= hops <= max_hops:
            cpath = canonical_path(path)
            if cpath not in seen_paths:
                seen_paths.add(cpath)
                a = path[0]
                b = path[-1]
                ta = node_map[a]["element_type"]
                tb = node_map[b]["element_type"]
                if ta != tb:
                    chains.append({
                        "doc_id": doc_id,
                        "path": list(path),
                        "hop_distance": hops,
                        "element_a_id": a,
                        "element_b_id": b,
                        "element_a_type": ta,
                        "element_b_type": tb,
                        "pair_type": normalize_pair_type(ta, tb),
                        "intermediate_ids": path[1:-1],
                        "intermediate_elements": [node_map[mid] for mid in path[1:-1] if mid in node_map],
                        "edge_contexts": aggregate_edge_contexts(path, edge_map),
                        "latex_bridge": aggregate_bridge(path, edge_map),
                        "quality_score": chain_quality(path, edge_map),
                        "element_a": node_map[a],
                        "element_b": node_map[b],
                    })
                    if len(chains) >= max_chains_per_doc:
                        return

        if hops == max_hops:
            return

        for nb in sorted(adj.get(cur, [])):
            if nb in path:
                continue
            dfs(nb, path + [nb])
            if len(chains) >= max_chains_per_doc:
                return

    for s in node_ids:
        dfs(s, [s])
        if len(chains) >= max_chains_per_doc:
            break

    return chains


def select_endpoint_pairs(
    chains: List[Dict[str, Any]],
    direct_endpoints: Set[Tuple[str, str, str]],
    only_novel_endpoints: bool,
    prefer_longer: bool,
) -> List[Dict[str, Any]]:
    """Pick one best chain per endpoint pair and emit pair-compatible records."""
    best: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    multi_counts: Counter = Counter()

    for c in chains:
        doc_id = c["doc_id"]
        a = c["element_a_id"]
        b = c["element_b_id"]
        ek = canonical_edge(a, b)
        key = (doc_id, ek[0], ek[1])

        if only_novel_endpoints and key in direct_endpoints:
            continue

        multi_counts[key] += 1
        cur = best.get(key)
        if cur is None:
            best[key] = c
            continue

        # Ranking policy
        if prefer_longer:
            new_rank = (c["quality_score"], c["hop_distance"])
            old_rank = (cur["quality_score"], cur["hop_distance"])
        else:
            new_rank = (c["quality_score"], -c["hop_distance"])
            old_rank = (cur["quality_score"], -cur["hop_distance"])
        if new_rank > old_rank:
            best[key] = c

    selected = []
    for idx, (key, c) in enumerate(sorted(best.items()), 1):
        doc_id, a, b = key
        out = {
            "pair_id": f"{doc_id}_lc_{idx:04d}",
            "doc_id": doc_id,
            "element_a_id": a,
            "element_b_id": b,
            "element_a_type": c["element_a_type"] if c["element_a_id"] == a else c["element_b_type"],
            "element_b_type": c["element_b_type"] if c["element_b_id"] == b else c["element_a_type"],
            "pair_type": normalize_pair_type(
                c["element_a_type"] if c["element_a_id"] == a else c["element_b_type"],
                c["element_b_type"] if c["element_b_id"] == b else c["element_a_type"],
            ),
            "hop_distance": c["hop_distance"],
            "path": c["path"],
            "quality_score": c["quality_score"],
            "overlap_with_existing_l1": False,
            "element_a": c["element_a"] if c["element_a_id"] == a else c["element_b"],
            "element_b": c["element_b"] if c["element_b_id"] == b else c["element_a"],
            "intermediate_ids": c.get("intermediate_ids", []),
            "intermediate_elements": c.get("intermediate_elements", []),
            "edge_contexts": c["edge_contexts"],
            "latex_bridge": c["latex_bridge"],
            "chain_count_for_endpoints": multi_counts[key],
        }
        selected.append(out)

    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description="Build long-chain candidates from latex cross-modal pairs")
    ap.add_argument("--input", default="data/latex_cross_modal_pairs.json", help="Input pair JSON")
    ap.add_argument("--output-pairs", default="data/latex_long_chain_pairs.json", help="Output pair-sliced candidates")
    ap.add_argument("--output-chains", default="data/latex_long_chains.json", help="Output native chain records")
    ap.add_argument("--min-hops", type=int, default=2, help="Minimum chain hops")
    ap.add_argument("--max-hops", type=int, default=4, help="Maximum chain hops")
    ap.add_argument("--max-chains-per-doc", type=int, default=800, help="Cap chains per doc")
    ap.add_argument("--min-quality", type=float, default=0.0, help="Drop chains below quality")
    ap.add_argument("--only-novel-endpoints", action="store_true", help="Keep only endpoints not present in direct pairs")
    ap.add_argument("--prefer-longer", action="store_true", help="When tie-breaking endpoint chains, favor longer hop")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    data = json.loads(in_path.read_text(encoding="utf-8"))
    in_pairs = data.get("pairs", [])
    if not in_pairs:
        raise RuntimeError("Input has no pairs")

    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in in_pairs:
        by_doc[p["doc_id"]].append(p)

    direct_endpoints: Set[Tuple[str, str, str]] = set()
    for p in in_pairs:
        a, b = canonical_edge(p["element_a_id"], p["element_b_id"])
        direct_endpoints.add((p["doc_id"], a, b))

    all_chains: List[Dict[str, Any]] = []
    for doc_id, doc_pairs in sorted(by_doc.items()):
        adj, edge_map, node_map = build_doc_graph(doc_pairs)
        if not adj:
            continue
        chains = enumerate_doc_chains(
            doc_id=doc_id,
            adj=adj,
            edge_map=edge_map,
            node_map=node_map,
            min_hops=args.min_hops,
            max_hops=args.max_hops,
            max_chains_per_doc=args.max_chains_per_doc,
        )
        all_chains.extend(chains)

    # Quality filter
    all_chains = [c for c in all_chains if c.get("quality_score", 0.0) >= args.min_quality]

    # Assign chain_id after filtering
    for i, c in enumerate(all_chains):
        c["chain_id"] = f"{c['doc_id']}_chain_{i:05d}"

    pair_candidates = select_endpoint_pairs(
        chains=all_chains,
        direct_endpoints=direct_endpoints,
        only_novel_endpoints=args.only_novel_endpoints,
        prefer_longer=args.prefer_longer,
    )

    # Build reports
    chain_hops = Counter(c["hop_distance"] for c in all_chains)
    chain_types = Counter(c["pair_type"] for c in all_chains)
    pair_hops = Counter(p["hop_distance"] for p in pair_candidates)
    pair_types = Counter(p["pair_type"] for p in pair_candidates)
    pair_docs = Counter(p["doc_id"] for p in pair_candidates)

    out_pairs = {
        "metadata": {
            "source": str(in_path),
            "min_hops": args.min_hops,
            "max_hops": args.max_hops,
            "max_chains_per_doc": args.max_chains_per_doc,
            "min_quality": args.min_quality,
            "only_novel_endpoints": args.only_novel_endpoints,
            "prefer_longer": args.prefer_longer,
            "num_input_pairs": len(in_pairs),
            "num_output_pairs": len(pair_candidates),
            "num_output_chains": len(all_chains),
        },
        "summary": {
            "pair_hop_distribution": dict(sorted(pair_hops.items())),
            "pair_type_distribution": dict(sorted(pair_types.items())),
            "top_docs": dict(pair_docs.most_common(20)),
        },
        "pairs": pair_candidates,
    }

    out_chains = {
        "metadata": out_pairs["metadata"],
        "summary": {
            "chain_hop_distribution": dict(sorted(chain_hops.items())),
            "chain_type_distribution": dict(sorted(chain_types.items())),
        },
        "chains": all_chains,
    }

    out_pairs_path = Path(args.output_pairs)
    out_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    out_pairs_path.write_text(json.dumps(out_pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    out_chains_path = Path(args.output_chains)
    out_chains_path.parent.mkdir(parents=True, exist_ok=True)
    out_chains_path.write_text(json.dumps(out_chains, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] long-chain build complete")
    print(f"  input pairs      : {len(in_pairs)}")
    print(f"  output chains    : {len(all_chains)}")
    print(f"  output pair slice: {len(pair_candidates)}")
    print(f"  chains by hop    : {dict(sorted(chain_hops.items()))}")
    print(f"  pairs by hop     : {dict(sorted(pair_hops.items()))}")
    print(f"  pairs by type    : {dict(sorted(pair_types.items()))}")
    print(f"  output pairs file: {out_pairs_path}")
    print(f"  output chains file: {out_chains_path}")


if __name__ == "__main__":
    main()
