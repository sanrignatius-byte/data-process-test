#!/usr/bin/env python3
"""Build hub scores and multi-hop candidates for the MinerU topology graph."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPOLOGY_DIR = ROOT / "data/05_eval/mineru_topology_graph_v1_latest"
DEFAULT_VL_DIR = ROOT / "data/05_eval/mineru_vl_edges_v1_latest"

ELEMENT_TYPES = {"figure", "table", "formula"}
BRIDGE_TYPES = {"paragraph", "section"}
NON_BACKBONE_TYPES = {
    "element_ref",
    "text_describes_figure",
    "figure_described_by_text",
    "visual_similarity",
    "cross_doc_visual_sim",
    "formula_similarity",
    "section_contains_element",
    "section_contains_paragraph",
    "same_page_cross_type",
}
EDGE_PRIORITY = {
    "element_ref": 100,
    "text_describes_figure": 95,
    "figure_described_by_text": 94,
    "cross_doc_visual_sim": 90,
    "visual_similarity": 85,
    "formula_similarity": 80,
    "section_contains_element": 70,
    "section_contains_paragraph": 60,
    "same_page_cross_type": 20,
    "backbone": 10,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compact_text(value: Any, limit: int = 240) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.split())[:limit]


def load_graph(topology_path: Path, vl_edges_path: Path | None) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    topo = read_json(topology_path)
    nodes = {str(node["node_id"]): node for node in topo.get("nodes", [])}
    edges = list(topo.get("edges", []) or [])
    loaded_vl = 0
    skipped_vl = 0
    if vl_edges_path and vl_edges_path.exists():
        for edge in iter_jsonl(vl_edges_path):
            if edge.get("source_id") in nodes and edge.get("target_id") in nodes:
                edges.append(edge)
                loaded_vl += 1
            else:
                skipped_vl += 1
    meta = {
        "source_topology": str(topology_path),
        "source_vl_edges": str(vl_edges_path) if vl_edges_path else "",
        "loaded_vl_edges": loaded_vl,
        "skipped_vl_edges": skipped_vl,
    }
    return nodes, edges, meta


def build_adjacency(edges: list[dict[str, Any]], max_neighbors: int) -> tuple[dict[str, set[str]], dict[tuple[str, str], list[dict[str, Any]]]]:
    neighbor_edges: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    degree_weight: dict[tuple[str, str], float] = defaultdict(float)
    raw_adj: dict[str, set[str]] = defaultdict(set)

    for edge in edges:
        src = str(edge.get("source_id"))
        tgt = str(edge.get("target_id"))
        if not src or not tgt or src == tgt:
            continue
        etype = str(edge.get("edge_type") or "")
        weight = float(edge.get("weight") or 1.0)
        score = EDGE_PRIORITY.get(etype, 1) + weight
        for a, b in ((src, tgt), (tgt, src)):
            raw_adj[a].add(b)
            neighbor_edges[(a, b)].append(edge)
            degree_weight[(a, b)] = max(degree_weight[(a, b)], score)

    if max_neighbors <= 0:
        return raw_adj, neighbor_edges

    adj: dict[str, set[str]] = {}
    for src, neighbors in raw_adj.items():
        ordered = sorted(
            neighbors,
            key=lambda nb: (degree_weight.get((src, nb), 0.0), -len(raw_adj.get(nb, set())), nb),
            reverse=True,
        )
        adj[src] = set(ordered[:max_neighbors])
    return adj, neighbor_edges


def pagerank(
    node_ids: list[str],
    adj: dict[str, set[str]],
    damping: float = 0.85,
    max_iter: int = 40,
    tol: float = 1e-9,
) -> dict[str, float]:
    n = len(node_ids)
    if n == 0:
        return {}
    rank = {nid: 1.0 / n for nid in node_ids}
    reverse: dict[str, set[str]] = defaultdict(set)
    for src, tgts in adj.items():
        for tgt in tgts:
            reverse[tgt].add(src)
    for _ in range(max_iter):
        dangling = sum(rank[nid] for nid in node_ids if not adj.get(nid))
        new_rank: dict[str, float] = {}
        diff = 0.0
        for nid in node_ids:
            value = (1.0 - damping) / n + damping * dangling / n
            for src in reverse.get(nid, set()):
                deg = len(adj.get(src, set()))
                if deg:
                    value += damping * rank[src] / deg
            new_rank[nid] = value
            diff += abs(value - rank[nid])
        rank = new_rank
        if diff < tol:
            break
    return rank


def zscores(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    nums = list(values.values())
    mean = sum(nums) / len(nums)
    var = sum((v - mean) ** 2 for v in nums) / max(1, len(nums))
    std = math.sqrt(var) or 1.0
    return {k: (v - mean) / std for k, v in values.items()}


def keyword_boost(node: dict[str, Any]) -> tuple[float, str]:
    text = " ".join([
        str(node.get("label") or ""),
        str(node.get("section_title") or ""),
        str(node.get("text_snippet") or ""),
    ]).lower()
    rules = [
        (r"\b(introduction|overview|summary)\b", 0.15, "overview"),
        (r"\b(method|approach|framework|architecture|pipeline|algorithm)\b", 0.12, "method"),
        (r"\b(experiment|evaluation|result|ablation|comparison)\b", 0.10, "evaluation"),
        (r"\b(discussion|analysis|conclusion)\b", 0.06, "discussion"),
    ]
    for pat, score, label in rules:
        if re.search(pat, text):
            return score, label
    return 0.0, ""


def compute_hubs(nodes: dict[str, dict[str, Any]], adj: dict[str, set[str]], top_k: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
    node_ids = sorted(nodes)
    pr = pagerank(node_ids, adj)
    pr_z = zscores(pr)
    max_pos_by_doc: dict[str, int] = defaultdict(int)
    for node in nodes.values():
        if isinstance(node.get("position_idx"), int):
            max_pos_by_doc[str(node.get("doc_id"))] = max(max_pos_by_doc[str(node.get("doc_id"))], int(node["position_idx"]))

    hubs: list[dict[str, Any]] = []
    score_by_node: dict[str, float] = {}
    for nid in node_ids:
        node = nodes[nid]
        neighbors = adj.get(nid, set())
        if not neighbors:
            continue
        neighbor_types = {nodes[nb]["node_type"] for nb in neighbors if nb in nodes}
        modality_diversity = len(neighbor_types & (ELEMENT_TYPES | {"paragraph"}))
        diversity_bonus = 0.3 * (modality_diversity / 4.0)
        max_pos = max_pos_by_doc.get(str(node.get("doc_id")), 0)
        pos = node.get("position_idx")
        backbone_bonus = 0.0
        if isinstance(pos, int) and max_pos > 0:
            backbone_bonus = 0.2 * (pos / max_pos)
        kboost, ksource = keyword_boost(node)
        connectivity_bonus = 0.1 * math.log1p(len(neighbors))
        raw_score = pr_z.get(nid, 0.0) + diversity_bonus + backbone_bonus + kboost + connectivity_bonus
        score_by_node[nid] = raw_score
        hubs.append({
            "node_id": nid,
            "doc_id": node.get("doc_id"),
            "node_type": node.get("node_type"),
            "label": node.get("label") or "",
            "text_preview": compact_text(node.get("text_snippet"), 220),
            "page_idx": node.get("page_idx"),
            "position_idx": node.get("position_idx"),
            "pagerank": round(pr.get(nid, 0.0), 10),
            "pagerank_z": round(pr_z.get(nid, 0.0), 6),
            "degree": len(neighbors),
            "modality_diversity": modality_diversity,
            "keyword_boost": kboost,
            "keyword_source": ksource,
            "hub_score": round(raw_score, 6),
        })

    hubs.sort(key=lambda row: row["hub_score"], reverse=True)
    return hubs[:top_k] if top_k > 0 else hubs, score_by_node


def best_edge(edge_index: dict[tuple[str, str], list[dict[str, Any]]], src: str, tgt: str) -> dict[str, Any]:
    candidates = edge_index.get((src, tgt), [])
    if not candidates:
        return {"edge_type": "unknown", "weight": 0.0, "metadata": {}}
    return max(
        candidates,
        key=lambda e: (EDGE_PRIORITY.get(str(e.get("edge_type") or ""), 0), float(e.get("weight") or 0.0)),
    )


def path_signature(path: list[str]) -> tuple[str, ...]:
    rev = list(reversed(path))
    return tuple(path) if tuple(path) <= tuple(rev) else tuple(rev)


def score_path(
    path: list[str],
    edge_types: list[str],
    nodes: dict[str, dict[str, Any]],
    score_by_node: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    endpoint_score = (score_by_node.get(path[0], 0.0) + score_by_node.get(path[-1], 0.0)) / 2.0
    types = [str(nodes[n].get("node_type")) for n in path]
    modality_diversity = len(set(types)) / 4.0
    text_len = sum(len(str(nodes[n].get("text_snippet") or "")) for n in path if nodes[n].get("node_type") in BRIDGE_TYPES)
    bridge_richness = min(1.0, text_len / 1000.0)
    doc_ids = {str(nodes[n].get("doc_id")) for n in path}
    page_span = None
    if len(doc_ids) == 1:
        pages = [nodes[n].get("page_idx") for n in path if isinstance(nodes[n].get("page_idx"), int)]
        if len(pages) >= 2:
            page_span = max(pages) - min(pages)
    long_gap_penalty = ((page_span or 0) / 10.0) if page_span is not None else 0.0
    edge_type_bonus = 0.0
    if "element_ref" in edge_types:
        edge_type_bonus += 0.15
    if "text_describes_figure" in edge_types or "figure_described_by_text" in edge_types:
        edge_type_bonus += 0.15
    if "cross_doc_visual_sim" in edge_types:
        edge_type_bonus += 0.12
    score = endpoint_score + modality_diversity + bridge_richness - long_gap_penalty + edge_type_bonus
    return score, {
        "modality_diversity": round(modality_diversity, 4),
        "bridge_richness": round(bridge_richness, 4),
        "page_span": page_span,
        "long_gap_penalty": round(long_gap_penalty, 4),
        "edge_type_bonus": round(edge_type_bonus, 4),
    }


def enumerate_candidates(
    nodes: dict[str, dict[str, Any]],
    hubs: list[dict[str, Any]],
    adj: dict[str, set[str]],
    edge_index: dict[tuple[str, str], list[dict[str, Any]]],
    score_by_node: dict[str, float],
    max_hops: int,
    min_hops: int,
    max_candidates: int,
    per_seed_cap: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    seed_counts: Counter[str] = Counter()

    def maybe_add(seed_id: str, path: list[str]) -> None:
        if len(path) < 2 or len(path) - 1 > max_hops:
            return
        if len(path) - 1 < min_hops:
            return
        if seed_counts[seed_id] >= per_seed_cap or len(candidates) >= max_candidates:
            return
        sig = path_signature(path)
        if sig in seen:
            return
        endpoint_types = [str(nodes[path[0]].get("node_type")), str(nodes[path[-1]].get("node_type"))]
        if endpoint_types[0] == endpoint_types[1]:
            return
        if not any(nodes[n].get("node_type") in ELEMENT_TYPES for n in path):
            return
        if not (endpoint_types[0] in ELEMENT_TYPES or endpoint_types[1] in ELEMENT_TYPES):
            return
        edge_rows = [best_edge(edge_index, a, b) for a, b in zip(path, path[1:])]
        edge_types = [str(e.get("edge_type") or "") for e in edge_rows]
        if not any(et in NON_BACKBONE_TYPES and et != "backbone" for et in edge_types):
            return
        if len({str(nodes[n].get("node_type")) for n in path}) < 2:
            return
        score, parts = score_path(path, edge_types, nodes, score_by_node)
        bridge_texts = [
            {
                "node_id": nid,
                "node_type": nodes[nid].get("node_type"),
                "text": compact_text(nodes[nid].get("text_snippet"), 500),
            }
            for nid in path
            if nodes[nid].get("node_type") in BRIDGE_TYPES and nodes[nid].get("text_snippet")
        ]
        seen.add(sig)
        seed_counts[seed_id] += 1
        candidates.append({
            "candidate_id": f"cand_{len(candidates) + 1:05d}",
            "doc_id": nodes[seed_id].get("doc_id"),
            "seed_node_id": seed_id,
            "path": path,
            "path_node_types": [nodes[n].get("node_type") for n in path],
            "endpoint_types": endpoint_types,
            "hop_count": len(path) - 1,
            "edge_types": edge_types,
            "score": round(score, 6),
            "score_parts": parts,
            "bridge_texts": bridge_texts,
            "is_cross_doc": len({nodes[n].get("doc_id") for n in path}) > 1,
        })

    for hub in hubs:
        seed_id = hub["node_id"]
        stack: list[tuple[str, list[str]]] = [(seed_id, [seed_id])]
        while stack and len(candidates) < max_candidates and seed_counts[seed_id] < per_seed_cap:
            cur, path = stack.pop()
            if len(path) - 1 >= max_hops:
                maybe_add(seed_id, path)
                continue
            for nb in sorted(adj.get(cur, set()), key=lambda x: (-score_by_node.get(x, 0.0), x)):
                if nb in path or nb not in nodes:
                    continue
                next_path = path + [nb]
                maybe_add(seed_id, next_path)
                if len(next_path) - 1 < max_hops:
                    stack.append((nb, next_path))
                if len(candidates) >= max_candidates or seed_counts[seed_id] >= per_seed_cap:
                    break

    candidates.sort(key=lambda row: row["score"], reverse=True)
    for idx, cand in enumerate(candidates, start=1):
        cand["candidate_id"] = f"cand_{idx:05d}"
    return candidates[:max_candidates]


def balance_candidates(
    candidates: list[dict[str, Any]],
    max_candidates: int,
    min_two_hop_ratio: float,
    min_three_hop_ratio: float,
) -> list[dict[str, Any]]:
    """Select a score-ordered but hop-balanced final candidate set."""
    by_hop: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cand in sorted(candidates, key=lambda row: row["score"], reverse=True):
        by_hop[int(cand.get("hop_count") or 0)].append(cand)

    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()

    def take_from(hop: int, count: int) -> None:
        for cand in by_hop.get(hop, []):
            if len(selected) >= max_candidates or count <= 0:
                return
            ident = id(cand)
            if ident in selected_ids:
                continue
            selected.append(cand)
            selected_ids.add(ident)
            count -= 1

    take_from(2, math.ceil(max_candidates * min_two_hop_ratio))
    take_from(3, math.ceil(max_candidates * min_three_hop_ratio))
    for cand in sorted(candidates, key=lambda row: row["score"], reverse=True):
        if len(selected) >= max_candidates:
            break
        ident = id(cand)
        if ident not in selected_ids:
            selected.append(cand)
            selected_ids.add(ident)

    selected.sort(key=lambda row: row["score"], reverse=True)
    for idx, cand in enumerate(selected, start=1):
        cand["candidate_id"] = f"cand_{idx:05d}"
    return selected[:max_candidates]


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MinerU Hub Candidates v1",
        "",
        f"- topology: `{summary['source_topology']}`",
        f"- VL edges loaded: **{summary['loaded_vl_edges']}**",
        f"- hubs: **{summary['hub_count']}**",
        f"- candidates: **{summary['candidate_count']}**",
        f"- candidate edge types: `{summary['candidate_edge_type_counts']}`",
        f"- hop distribution: `{summary['hop_count_distribution']}`",
        f"- cross-doc candidates: **{summary['cross_doc_candidates']}**",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_latest_symlink(out_dir: Path) -> None:
    latest = ROOT / "data/05_eval/mineru_hub_candidates_v1_latest"
    try:
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        if not latest.exists():
            latest.symlink_to(out_dir.resolve())
    except OSError as exc:
        print(f"[warn] could not update latest symlink: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MinerU hub scores and candidates v1")
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY_DIR / "mineru_topology_graph_v1.json"))
    parser.add_argument("--vl-edges", default=str(DEFAULT_VL_DIR / "mineru_vl_edges_v1.jsonl"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--top-k-hubs", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--min-hops", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--per-seed-cap", type=int, default=50)
    parser.add_argument("--pool-multiplier", type=int, default=8)
    parser.add_argument("--min-two-hop-ratio", type=float, default=0.50)
    parser.add_argument("--min-three-hop-ratio", type=float, default=0.30)
    parser.add_argument("--max-neighbors", type=int, default=40)
    args = parser.parse_args()

    topology_path = Path(args.topology)
    vl_edges_path = Path(args.vl_edges) if args.vl_edges else None
    if not topology_path.exists():
        raise FileNotFoundError(topology_path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / f"data/05_eval/mineru_hub_candidates_v1_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes, edges, source_meta = load_graph(topology_path, vl_edges_path)
    adj, edge_index = build_adjacency(edges, args.max_neighbors)
    for nid in nodes:
        adj.setdefault(nid, set())
    hubs, score_by_node = compute_hubs(nodes, adj, args.top_k_hubs)
    seed_hubs = hubs[:args.seed_count]
    pool_limit = max(args.max_candidates, args.max_candidates * max(1, args.pool_multiplier))
    candidate_pool = enumerate_candidates(
        nodes,
        seed_hubs,
        adj,
        edge_index,
        score_by_node,
        args.max_hops,
        args.min_hops,
        pool_limit,
        args.per_seed_cap,
    )
    candidates = balance_candidates(
        candidate_pool,
        args.max_candidates,
        args.min_two_hop_ratio,
        args.min_three_hop_ratio,
    )

    hub_payload = {
        "metadata": {
            "builder": "mineru_hub_candidates_v1",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **source_meta,
            "top_k_hubs": args.top_k_hubs,
        },
        "hubs": hubs,
    }
    candidate_payload = {
        "metadata": {
            "builder": "mineru_hub_candidates_v1",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **source_meta,
            "num_seeds": len(seed_hubs),
            "min_hops": args.min_hops,
            "max_hops": args.max_hops,
            "max_candidates": args.max_candidates,
            "candidate_pool_size": len(candidate_pool),
            "max_neighbors": args.max_neighbors,
        },
        "hubs": seed_hubs,
        "candidates": candidates,
    }
    edge_type_counter: Counter[str] = Counter()
    for cand in candidates:
        edge_type_counter.update(cand.get("edge_types") or [])
    summary = {
        "builder": "mineru_hub_candidates_v1",
        "created_at": candidate_payload["metadata"]["created_at"],
        **source_meta,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "hub_count": len(hubs),
        "seed_count": len(seed_hubs),
        "candidate_count": len(candidates),
        "candidate_pool_size": len(candidate_pool),
        "cross_doc_candidates": sum(1 for c in candidates if c.get("is_cross_doc")),
        "hop_count_distribution": dict(Counter(c.get("hop_count") for c in candidates)),
        "candidate_edge_type_counts": dict(edge_type_counter),
        "top_hub_score": hubs[0]["hub_score"] if hubs else None,
    }

    (out_dir / "mineru_hub_scores_v1.json").write_text(json.dumps(hub_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "mineru_multihop_candidates_v1.json").write_text(json.dumps(candidate_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir, summary)
    update_latest_symlink(out_dir)

    print(f"[ok] wrote {out_dir / 'mineru_multihop_candidates_v1.json'}")
    print(f"hubs={len(hubs)} candidates={len(candidates)} cross_doc={summary['cross_doc_candidates']}")


if __name__ == "__main__":
    main()
