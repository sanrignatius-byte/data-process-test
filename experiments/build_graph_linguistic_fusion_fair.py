#!/usr/bin/env python3
"""Fair-baseline comparison for linguistic vs raw CLIP cross-doc edges.

Builds and compares four graph variants on the SAME element resolution rule:

  A. empty       — graph + 0 cross-doc edges (the old "baseline")
  B. raw_100     — graph + the same 100 raw CLIP section-level edges that were
                   fed into the linguistic judge (matched-pair control)
  C. linguistic  — graph + 43 strong+weak linguistic edges (subset of B)
  D. raw_all     — graph + all 2467 raw CLIP section-level edges (full corpus)

Element resolution uses the same cartesian-product rule across all variants,
so any chain-count difference comes from the edge set alone — not from how
section-level edges are projected to element pairs.

What this answers / doesn't answer:
  - Answers: "Does linguistic filtering of a matched 100-edge pool change
    chain count / modality coverage vs no filter?"  (B vs C)
  - Answers: "Does the empty baseline radically understate what raw CLIP
    alone would already produce?"  (A vs B, A vs D)
  - Does NOT answer: chain *quality*. Chain count is topological reachability.
    Quality requires running a judge (e.g. chunk-bridge judge) on the chains
    output; that's a separate step.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_raw_clip_edges(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["edges"]


def load_elements(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    elements: Dict[str, Dict[str, Any]] = {}
    for doc_id, doc in data.get("documents", {}).items():
        for eid, el in doc.get("elements", {}).items():
            elements[eid] = el
    return elements


def load_hub_candidates(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def doc_id_of(eid: str) -> str:
    for tag in ("_figure_", "_table_", "_formula_"):
        if tag in eid:
            return eid.split(tag)[0]
    return eid


def build_intra_doc_adjacency(
    elements: Dict[str, Dict[str, Any]],
    hub_candidates: Dict[str, Any],
) -> Dict[str, Set[str]]:
    """Same intra-doc backbone as the original fusion script."""
    adj: Dict[str, Set[str]] = defaultdict(set)

    pairs = hub_candidates.get("pairs", hub_candidates.get("candidates", []))
    if not pairs and isinstance(hub_candidates, list):
        pairs = hub_candidates
    for pair in pairs:
        path = pair.get("path", [])
        for i in range(len(path) - 1):
            adj[path[i]].add(path[i + 1])
            adj[path[i + 1]].add(path[i])

    doc_of = {eid: doc_id_of(eid) for eid in elements}
    for eid in elements:
        doc = doc_of[eid]
        for other in elements:
            if doc_of.get(other) != doc or other == eid:
                continue
            ta = elements[eid].get("element_type", "")
            tb = elements[other].get("element_type", "")
            if ta != tb and ta in ("figure", "table", "formula") and tb in ("figure", "table", "formula"):
                adj[eid].add(other)

    return adj


def inject_xdoc_edges(
    adj: Dict[str, Set[str]],
    edges: List[Dict[str, Any]],
    elements: Dict[str, Dict[str, Any]],
) -> int:
    """Cartesian-product projection of section-level edges to element pairs.

    Same rule for every variant so the comparison is apples-to-apples.
    """
    by_doc: Dict[str, List[str]] = defaultdict(list)
    for eid, el in elements.items():
        if el.get("element_type") in ("figure", "table", "formula"):
            by_doc[doc_id_of(eid)].append(eid)

    added_pairs: Set[Tuple[str, str]] = set()
    for edge in edges:
        sd = edge.get("source_doc", "")
        td = edge.get("target_doc", "")
        if not sd or not td or sd == td:
            continue
        for se in by_doc.get(sd, []):
            for te in by_doc.get(td, []):
                if se == te:
                    continue
                key = (se, te) if se < te else (te, se)
                if key in added_pairs:
                    continue
                added_pairs.add(key)
                adj[se].add(te)
                adj[te].add(se)
    return len(added_pairs)


def find_cross_doc_chains(
    adj: Dict[str, Set[str]],
    elements: Dict[str, Dict[str, Any]],
    min_hops: int,
    max_hops: int,
    max_chains: int,
    seed: int,
) -> List[Dict[str, Any]]:
    valid = {eid for eid, el in elements.items()
             if el.get("element_type") in ("figure", "table", "formula")}
    doc_of = {eid: doc_id_of(eid) for eid in elements}
    chains: List[Dict[str, Any]] = []

    starts = list(valid)
    rng = random.Random(seed)
    rng.shuffle(starts)

    for start in starts:
        if len(chains) >= max_chains:
            break
        start_doc = doc_of[start]
        start_type = elements[start].get("element_type", "")

        visited = {start}
        queue: deque = deque([(start, [start])])
        while queue and len(chains) < max_chains:
            node, path = queue.popleft()
            if len(path) > max_hops:
                continue
            for nb in adj.get(node, set()):
                if nb in visited or nb not in valid:
                    continue
                visited.add(nb)
                new_path = path + [nb]
                if len(new_path) - 1 >= min_hops:
                    end_type = elements[nb].get("element_type", "")
                    if start_type != end_type and doc_of[nb] != start_doc:
                        chains.append({
                            "path": new_path,
                            "hop_count": len(new_path) - 1,
                            "modality_pair": f"{start_type}+{end_type}",
                        })
                if len(new_path) <= max_hops:
                    queue.append((nb, new_path))
    return chains


def summarize(chains: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not chains:
        return {"count": 0, "modality_dist": {}, "hop_dist": {}}
    mod = Counter(c["modality_pair"] for c in chains)
    hop = Counter(c["hop_count"] for c in chains)
    return {
        "count": len(chains),
        "modality_dist": dict(mod.most_common(10)),
        "hop_dist": {str(k): v for k, v in sorted(hop.items())},
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--linguistic-edges", required=True)
    ap.add_argument("--raw-clip-edges",
                    default="data/01_graphs/cross_doc_sim_edges.json")
    ap.add_argument("--elements",
                    default="data/02_enriched/multimodal_elements_enriched.json")
    ap.add_argument("--hub-candidates",
                    default="data/02_enriched/hub_candidates_enriched_v4_intra_doc.json")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--min-hops", type=int, default=2)
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--max-chains", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else \
        ROOT / f"data/05_eval/graph_linguistic_fusion_fair_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading inputs...")
    lingu_all = load_jsonl(Path(args.linguistic_edges))
    raw_all = load_raw_clip_edges(ROOT / args.raw_clip_edges)
    elements = load_elements(ROOT / args.elements)
    hub_candidates = load_hub_candidates(ROOT / args.hub_candidates)
    print(f"  linguistic edges: {len(lingu_all)}")
    print(f"  raw CLIP edges total: {len(raw_all)}")
    print(f"  elements: {len(elements)}")

    # The matched-pair raw subset = the 100 raw edges that were fed into the judge.
    lingu_keys = {(e["source"], e["target"]) for e in lingu_all}
    raw_100 = [e for e in raw_all if (e["source"], e["target"]) in lingu_keys]
    print(f"  raw matched subset: {len(raw_100)} / {len(lingu_all)}")

    lingu_strong_weak = [e for e in lingu_all
                         if e.get("linguistic_quality_tier") in ("strong", "weak")]
    print(f"  linguistic strong+weak: {len(lingu_strong_weak)}")

    variants = {
        "A_empty": [],
        "B_raw_matched_100": raw_100,
        "C_linguistic_strong_weak": lingu_strong_weak,
        "D_raw_all": raw_all,
    }

    results: Dict[str, Any] = {}
    chains_out: Dict[str, List[Dict[str, Any]]] = {}
    for name, edges in variants.items():
        print(f"\n=== Variant {name}  (input edges = {len(edges)}) ===")
        intra = build_intra_doc_adjacency(elements, hub_candidates)
        injected = inject_xdoc_edges(intra, edges, elements)
        nodes = len(intra)
        edges_total = sum(len(v) for v in intra.values()) // 2
        print(f"  graph: {nodes} nodes, {edges_total} edges (injected element pairs: {injected})")

        chains = find_cross_doc_chains(
            intra, elements,
            min_hops=args.min_hops, max_hops=args.max_hops,
            max_chains=args.max_chains, seed=args.seed,
        )
        summary = summarize(chains)
        print(f"  chains: {summary['count']}")
        if summary["modality_dist"]:
            top = ", ".join(f"{k}={v}" for k, v in
                            list(summary["modality_dist"].items())[:3])
            print(f"  top modality: {top}")

        results[name] = {
            "input_edges": len(edges),
            "injected_element_pairs": injected,
            "graph_nodes": nodes,
            "graph_edges": edges_total,
            **summary,
        }
        chains_out[name] = chains

    # Write per-variant chains
    for name, chains in chains_out.items():
        (out_dir / f"chains_{name}.json").write_text(
            json.dumps({"variant": name, "count": len(chains), "chains": chains})
        )

    # Headline comparison
    print("\n=== Fair Comparison ===")
    print(f"  {'variant':28s}  edges  injected   chains")
    for name in variants:
        r = results[name]
        print(f"  {name:28s}  {r['input_edges']:5d}  {r['injected_element_pairs']:8d}  {r['count']:6d}")

    summary_path = out_dir / "fair_summary.json"
    summary_path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "min_hops": args.min_hops,
        "max_hops": args.max_hops,
        "max_chains_cap": args.max_chains,
        "variants": results,
        "note": (
            "Chain counts are BFS-reachability under a cap of "
            f"{args.max_chains} chains. Cartesian-product element resolution "
            "is identical across variants. Chain *quality* is not measured "
            "here — run chunk-bridge judge on chains_*.json to assess that."
        ),
    }, indent=2))
    print(f"\nWritten: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
