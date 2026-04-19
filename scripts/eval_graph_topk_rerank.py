#!/usr/bin/env python3
"""
eval_graph_topk_rerank.py
=========================
Apply graph-aware reranking to an existing dense top-K ranking.

This script is intentionally narrow:
  - it does NOT rebuild the index
  - it does NOT expand the candidate pool
  - it only reranks passages already present in the input top-K

Current use case:
  rebuilt M4query_v1 dense ranking -> graph rerank within top-100

Methods:
  baseline
    Keep the input ranking unchanged.

  graph_static_prior
    Add a small static graph prior based on each passage's normalized graph
    degree in the chunk-induced passage graph.

  graph_neighbor_prop
    1-hop propagation: higher-ranked passages pass score to lower-ranked graph
    neighbors already present in top-K.

  graph_static_plus_neighbor
    Combine the two signals above.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

VIRTUAL_EDGE_TYPES = ("same_chunk", "chunk_sequence", "same_section_nonseq")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rankings(path: Path, top_k: int) -> Dict[str, List[str]]:
    rankings: Dict[str, List[str]] = {}
    for row in load_jsonl(path):
        qid = str(row["query_id"])
        pids = [str(pid) for pid in row.get("top_k", [])][:top_k]
        rankings[qid] = pids
    return rankings


def load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for row in load_jsonl(path):
        qid = str(row.get("query_id") or row.get("qid") or "")
        pid = str(row.get("passage_id") or row.get("pid") or "")
        if not qid or not pid:
            continue
        try:
            rel = int(row.get("relevance", 1))
        except (TypeError, ValueError):
            rel = 1
        qrels[qid][pid] = rel
    return dict(qrels)


def load_explicit_adjacency(
    hub_candidates_path: Path,
    valid_pids: Set[str],
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, Any]]:
    """Load original element-level bridge edges from enriched hub candidates.

    This is the old high-precision graph layer:
      - hub candidate pairs, weighted by quality_score
      - adjacent backbone bridge adjacency, weighted with a fixed moderate prior
    """
    print(f"[info] loading explicit graph from {hub_candidates_path} ...", flush=True)
    obj = json.loads(hub_candidates_path.read_text(encoding="utf-8"))

    weighted: Dict[str, Dict[str, float]] = defaultdict(dict)
    edge_type_counts: Counter[str] = Counter()

    for pair in obj.get("pairs", []) or []:
        a = str(pair.get("element_a_id", "") or "").strip()
        b = str(pair.get("element_b_id", "") or "").strip()
        if not a or not b or a == b:
            continue
        if a not in valid_pids or b not in valid_pids:
            continue
        qs = float(pair.get("quality_score", 1.0) or 1.0)
        weighted[a][b] = max(weighted[a].get(b, 0.0), qs)
        weighted[b][a] = max(weighted[b].get(a, 0.0), qs)
        edge_type_counts["hub_pair"] += 2

    for bridge in obj.get("adjacent_bridge_adjacency", []) or []:
        left = [str(x).strip() for x in (bridge.get("elements_i") or []) if str(x).strip()]
        right = [str(x).strip() for x in (bridge.get("elements_j") or []) if str(x).strip()]
        for a in left:
            if a not in valid_pids:
                continue
            for b in right:
                if b not in valid_pids or a == b:
                    continue
                weighted[a][b] = max(weighted[a].get(b, 0.0), 0.6)
                weighted[b][a] = max(weighted[b].get(a, 0.0), 0.6)
                edge_type_counts["adjacent_bridge"] += 2

    adjacency: Dict[str, List[Tuple[str, float]]] = {}
    degs: List[int] = []
    for pid, nbrs in weighted.items():
        merged = sorted(nbrs.items(), key=lambda x: (x[1], x[0]), reverse=True)
        adjacency[pid] = [(nbr, float(w)) for nbr, w in merged]
        degs.append(len(merged))
    degs.sort()

    graph_stats = {
        "num_explicit_graph_pids": len(adjacency),
        "explicit_graph_mapping_pct": round(len(adjacency) / max(1, len(valid_pids)) * 100, 2),
        "explicit_edge_counts_directed": dict(edge_type_counts),
        "explicit_degree_stats": {
            "median_degree": float(degs[len(degs) // 2]) if degs else 0.0,
            "mean_degree": float(sum(degs) / len(degs)) if degs else 0.0,
            "max_degree": float(max(degs)) if degs else 0.0,
        },
    }
    return adjacency, graph_stats


def build_typed_adjacency(
    chunk_graph_path: Path,
    elements_graph_path: Path,
    valid_pids: Set[str],
    include_edge_types: Set[str],
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, Any]]:
    """Build typed pid-level adjacency from the chunk graph.

    Edge semantics:
      - same_chunk: two element passages mapped to the same merged chunk
      - chunk_sequence: two passages whose mapped chunks are directly adjacent
      - same_section_nonseq: two passages in different chunks but same section
    """
    print(f"[info] loading chunk graph from {chunk_graph_path} ...", flush=True)
    chunk_data = json.loads(chunk_graph_path.read_bytes())
    print(f"[info] loading multimodal elements from {elements_graph_path} ...", flush=True)
    elem_data = json.loads(elements_graph_path.read_bytes())

    doc_pos_to_chunk: Dict[str, Dict[int, str]] = {}
    doc_chunk_sequence: Dict[str, Dict[str, Set[str]]] = {}
    doc_section_to_chunks: Dict[str, Dict[str, List[str]]] = {}
    for doc_id, doc in chunk_data.get("documents", {}).items():
        pos_to_chunk: Dict[int, str] = {}
        chunk_seq: Dict[str, Set[str]] = defaultdict(set)
        section_to_chunks: Dict[str, List[str]] = defaultdict(list)

        for nid, node in doc.get("nodes", {}).items():
            for pidx in node.get("paragraph_indices", []):
                pos_to_chunk[pidx] = nid

        for edge in doc.get("edges", []):
            rel = edge.get("relation", "")
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if rel == "chunk_sequence":
                chunk_seq[src].add(tgt)
                chunk_seq[tgt].add(src)
            elif rel == "section_contains_chunk":
                section_to_chunks[src].append(tgt)

        doc_pos_to_chunk[doc_id] = pos_to_chunk
        doc_chunk_sequence[doc_id] = chunk_seq
        doc_section_to_chunks[doc_id] = section_to_chunks

    doc_type0_to_pos: Dict[str, Dict[Tuple[str, int], int]] = {}
    doc_elemid_to_pos: Dict[str, Dict[str, int]] = {}
    for doc_id, doc in elem_data.get("documents", {}).items():
        type0_map: Dict[Tuple[str, int], int] = {}
        eid_map: Dict[str, int] = {}
        for elem_id, elem in doc.get("elements", {}).items():
            pidx = elem.get("position_idx")
            if pidx is None:
                continue
            eid_map[elem_id] = pidx
            parts = elem_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                one_idx = int(parts[1])
                prefix = parts[0]
                etype = (
                    prefix[len(doc_id) + 1:]
                    if prefix.startswith(doc_id + "_")
                    else prefix.rsplit("_", 1)[-1]
                )
                type0_map[(etype, one_idx - 1)] = pidx
        doc_type0_to_pos[doc_id] = type0_map
        doc_elemid_to_pos[doc_id] = eid_map

    type_alias = {"fig": "figure"}
    pid_to_chunk: Dict[str, str] = {}
    pid_doc: Dict[str, str] = {}
    for pid in valid_pids:
        for doc_id in doc_type0_to_pos:
            if not pid.startswith(doc_id + "_"):
                continue
            pid_doc[pid] = doc_id
            rest = pid[len(doc_id) + 1:]
            if rest.startswith(doc_id + "_"):
                rest = rest[len(doc_id) + 1:]
            parts = rest.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                raw_type = parts[0]
                n = int(parts[1])
                etype = type_alias.get(raw_type, raw_type)
                pidx = doc_type0_to_pos[doc_id].get((etype, n))
                if pidx is not None:
                    chunk_id = doc_pos_to_chunk.get(doc_id, {}).get(pidx)
                    if chunk_id:
                        pid_to_chunk[pid] = chunk_id

                if pid not in pid_to_chunk:
                    elem_id = f"{doc_id}_{raw_type}_{n}"
                    pidx2 = doc_elemid_to_pos[doc_id].get(elem_id)
                    if pidx2 is not None:
                        chunk_id2 = doc_pos_to_chunk.get(doc_id, {}).get(pidx2)
                        if chunk_id2:
                            pid_to_chunk[pid] = chunk_id2
            break

    chunk_to_pids: Dict[str, List[str]] = defaultdict(list)
    for pid, chunk_id in pid_to_chunk.items():
        chunk_to_pids[chunk_id].append(pid)

    typed_neighbors: Dict[str, Dict[str, Set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    edge_type_counts: Counter[str] = Counter()
    for pid, chunk_id in pid_to_chunk.items():
        doc_id = pid_doc[pid]

        if "same_chunk" in include_edge_types:
            for nb_pid in chunk_to_pids.get(chunk_id, []):
                if nb_pid != pid:
                    typed_neighbors[pid]["same_chunk"].add(nb_pid)
                    edge_type_counts["same_chunk"] += 1

        if "chunk_sequence" in include_edge_types:
            for nb_chunk in doc_chunk_sequence.get(doc_id, {}).get(chunk_id, set()):
                for nb_pid in chunk_to_pids.get(nb_chunk, []):
                    typed_neighbors[pid]["chunk_sequence"].add(nb_pid)
                    edge_type_counts["chunk_sequence"] += 1

        if "same_section_nonseq" in include_edge_types:
            for _, chunks in doc_section_to_chunks.get(doc_id, {}).items():
                if chunk_id not in chunks:
                    continue
                for nb_chunk in chunks:
                    if nb_chunk == chunk_id:
                        continue
                    if nb_chunk in doc_chunk_sequence.get(doc_id, {}).get(chunk_id, set()):
                        continue
                    for nb_pid in chunk_to_pids.get(nb_chunk, []):
                        typed_neighbors[pid]["same_section_nonseq"].add(nb_pid)
                        edge_type_counts["same_section_nonseq"] += 1

    adjacency: Dict[str, List[Tuple[str, float]]] = {}
    degree_stats: Dict[str, Dict[str, float]] = {}
    for edge_type in VIRTUAL_EDGE_TYPES:
        vals = [
            len(typed_neighbors[pid].get(edge_type, set()))
            for pid in pid_to_chunk
        ]
        vals.sort()
        if vals:
            degree_stats[edge_type] = {
                "median_degree": float(vals[len(vals) // 2]),
                "mean_degree": float(sum(vals) / len(vals)),
                "max_degree": float(max(vals)),
            }
        else:
            degree_stats[edge_type] = {
                "median_degree": 0.0,
                "mean_degree": 0.0,
                "max_degree": 0.0,
            }

    for pid, by_type in typed_neighbors.items():
        merged: List[Tuple[str, float]] = []
        seen: Set[str] = set()
        for edge_type in VIRTUAL_EDGE_TYPES:
            for nb_pid in by_type.get(edge_type, set()):
                if nb_pid == pid or nb_pid in seen:
                    continue
                seen.add(nb_pid)
                merged.append((nb_pid, 1.0))
        if merged:
            adjacency[pid] = merged

    graph_stats = {
        "num_virtual_graph_pids": len(pid_to_chunk),
        "virtual_graph_mapping_pct": round(len(pid_to_chunk) / max(1, len(valid_pids)) * 100, 2),
        "virtual_edge_types_active": sorted(include_edge_types),
        "virtual_edge_counts_directed": dict(edge_type_counts),
        "virtual_edge_type_degree_stats": degree_stats,
    }
    # expose mappings for cross-doc adjacency builder
    extra_mappings = {
        "pid_to_chunk": pid_to_chunk,
        "chunk_to_pids": dict(chunk_to_pids),
        "doc_section_to_chunks": doc_section_to_chunks,
    }
    return adjacency, graph_stats, extra_mappings


def load_cross_doc_adjacency(
    cross_doc_edges_path: Path,
    valid_pids: Set[str],
    pid_to_chunk: Dict[str, str],
    chunk_to_pids: Dict[str, List[str]],
    doc_section_to_chunks: Dict[str, Dict[str, List[str]]],
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, Any]]:
    """Build cross-document adjacency from pre-computed section-similarity edges.

    Maps: summary_node_id -> set of corpus passage PIDs belonging to that section/chunk,
    then creates weighted passage-level adjacency between passages in connected sections.
    """
    print(f"[info] loading cross-doc sim edges from {cross_doc_edges_path} ...", flush=True)
    obj = json.loads(cross_doc_edges_path.read_bytes())
    edges = obj.get("edges", [])
    print(f"[info] {len(edges)} cross-doc sim edges loaded", flush=True)

    # Build section_title -> chunk_ids mapping across all docs for fast lookup
    # doc_section_to_chunks: {doc_id: {section_title: [chunk_id, ...]}}
    # chunk_to_pids: {chunk_id: [pid, ...]}

    def summary_node_to_pids(node_id: str, doc_hint: str = "", section_hint: str = "") -> List[str]:
        """Resolve a summary node ID (or explicit (doc, section) hint) to corpus passage PIDs."""
        # Prefer explicit hints from edge metadata (most reliable)
        if doc_hint and section_hint and doc_hint in doc_section_to_chunks:
            sec_chunks = doc_section_to_chunks[doc_hint].get(section_hint, [])
            if not sec_chunks:
                for stitle, chunks in doc_section_to_chunks[doc_hint].items():
                    if section_hint.lower() in stitle.lower() or stitle.lower() in section_hint.lower():
                        sec_chunks = chunks
                        break
            if sec_chunks:
                pids: List[str] = []
                for cid in sec_chunks:
                    pids.extend(p for p in chunk_to_pids.get(cid, []) if p in valid_pids)
                if pids:
                    return pids
        # Format: {doc_id}_secsummary_{sec_key}  or  {doc_id}_chunksummary_{chunk_id}
        parts = node_id.split("_", 1)
        if len(parts) < 2:
            return []
        for doc_id in doc_section_to_chunks:
            if not node_id.startswith(doc_id + "_"):
                continue
            rest = node_id[len(doc_id) + 1:]
            if rest.startswith("secsummary_"):
                sec_key = rest[len("secsummary_"):]
                sec_chunks = doc_section_to_chunks[doc_id].get(sec_key, [])
                if not sec_chunks:
                    for stitle, chunks in doc_section_to_chunks[doc_id].items():
                        if sec_key.lower() in stitle.lower() or stitle.lower() in sec_key.lower():
                            sec_chunks = chunks
                            break
                pids: List[str] = []
                for cid in sec_chunks:
                    pids.extend(p for p in chunk_to_pids.get(cid, []) if p in valid_pids)
                return pids
            elif rest.startswith("chunksummary_"):
                chunk_id = rest[len("chunksummary_"):]
                return [p for p in chunk_to_pids.get(chunk_id, []) if p in valid_pids]
        return []

    weighted: Dict[str, Dict[str, float]] = defaultdict(dict)
    edge_count = 0
    pid_pair_count = 0
    doc_pairs_used: Set[Tuple[str, str]] = set()

    for edge in edges:
        sim = float(edge.get("similarity", 0.0))
        pids_a = summary_node_to_pids(
            edge["source"],
            doc_hint=edge.get("source_doc", ""),
            section_hint=edge.get("source_section", ""),
        )
        pids_b = summary_node_to_pids(
            edge["target"],
            doc_hint=edge.get("target_doc", ""),
            section_hint=edge.get("target_section", ""),
        )
        if not pids_a or not pids_b:
            continue
        edge_count += 1
        doc_a = edge.get("source_doc", "")
        doc_b = edge.get("target_doc", "")
        if doc_a and doc_b:
            doc_pairs_used.add((min(doc_a, doc_b), max(doc_a, doc_b)))
        for pa in pids_a:
            for pb in pids_b:
                if pa == pb:
                    continue
                w = sim
                weighted[pa][pb] = max(weighted[pa].get(pb, 0.0), w)
                weighted[pb][pa] = max(weighted[pb].get(pa, 0.0), w)
                pid_pair_count += 1

    adjacency: Dict[str, List[Tuple[str, float]]] = {
        pid: sorted(nbrs.items(), key=lambda x: -x[1])
        for pid, nbrs in weighted.items()
    }
    stats = {
        "num_cross_doc_graph_pids": len(adjacency),
        "cross_doc_graph_mapping_pct": round(len(adjacency) / max(1, len(valid_pids)) * 100, 2),
        "cross_doc_sim_edges_resolved": edge_count,
        "cross_doc_pid_pairs": pid_pair_count,
        "cross_doc_pairs_connected": len(doc_pairs_used),
    }
    print(f"[info] cross-doc adjacency: {len(adjacency)} pids, {len(doc_pairs_used)} doc pairs connected", flush=True)
    return adjacency, stats


def load_typed_crossdoc_adjacency(
    edges_path: Path,
    valid_pids: Set[str],
    include_types: Set[str],
    use_boost: bool = False,
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, Any]]:
    """Load typed cross-doc element similarity edges (pid-level) into adjacency."""
    print(f"[info] loading typed cross-doc edges from {edges_path} ...", flush=True)
    if not edges_path.exists():
        print(f"[warn] typed cross-doc edges file not found: {edges_path}", flush=True)
        return {}, {"typed_crossdoc_edges_loaded": 0}
    obj = json.loads(edges_path.read_bytes())
    edges = obj.get("edges", [])
    weighted: Dict[str, Dict[str, float]] = defaultdict(dict)
    per_type: Counter = Counter()
    cite_boosted = 0
    doc_pairs: Set[Tuple[str, str]] = set()
    for e in edges:
        t = e.get("type", "")
        if include_types and t not in include_types:
            continue
        a = str(e.get("source", ""))
        b = str(e.get("target", ""))
        if not a or not b or a == b:
            continue
        if a not in valid_pids or b not in valid_pids:
            continue
        w_key = "boosted_similarity" if use_boost else "similarity"
        w = float(e.get(w_key, e.get("similarity", 0.0)))
        weighted[a][b] = max(weighted[a].get(b, 0.0), w)
        weighted[b][a] = max(weighted[b].get(a, 0.0), w)
        per_type[t] += 1
        if e.get("cite_boosted"):
            cite_boosted += 1
        da, db = e.get("source_doc", ""), e.get("target_doc", "")
        if da and db:
            doc_pairs.add((min(da, db), max(da, db)))
    adjacency: Dict[str, List[Tuple[str, float]]] = {
        pid: sorted(nbrs.items(), key=lambda x: -x[1])
        for pid, nbrs in weighted.items()
    }
    stats = {
        "typed_crossdoc_edges_loaded": sum(per_type.values()),
        "typed_crossdoc_edges_per_type": dict(per_type),
        "typed_crossdoc_pids_in_graph": len(adjacency),
        "typed_crossdoc_pids_mapping_pct": round(len(adjacency) / max(1, len(valid_pids)) * 100, 2),
        "typed_crossdoc_doc_pairs_connected": len(doc_pairs),
        "typed_crossdoc_cite_boosted_edges": cite_boosted,
        "typed_crossdoc_use_boost_weight": use_boost,
    }
    print(f"[info] typed crossdoc: {len(adjacency)} pids, "
          f"{len(doc_pairs)} doc pairs, per-type {dict(per_type)}", flush=True)
    return adjacency, stats


def load_summary_shadow_index(
    path: Path,
    valid_pids: Set[str],
) -> Tuple[Dict[str, List[str]], Dict[str, List[Tuple[str, float]]], Dict[str, Any]]:
    """Load pre-built summary shadow index.

    Returns:
        summary_to_pids: summary_id -> list of child pids (only those in valid_pids)
        query_to_summary_scores: query_id -> [(summary_id, score), ...]
        stats: passthrough stats dict
    """
    print(f"[info] loading summary shadow index from {path} ...", flush=True)
    data = json.loads(path.read_bytes())
    raw_sum2pids: Dict[str, List[str]] = data.get("summary_to_pids") or {}
    summary_to_pids: Dict[str, List[str]] = {}
    for sid, pids in raw_sum2pids.items():
        kept = [p for p in pids if p in valid_pids]
        if kept:
            summary_to_pids[sid] = kept
    raw_q2s: Dict[str, List[List[Any]]] = data.get("query_summary_scores") or {}
    query_to_summary_scores: Dict[str, List[Tuple[str, float]]] = {}
    for qid, pairs in raw_q2s.items():
        kept_pairs = [
            (str(sid), float(sc))
            for sid, sc in pairs
            if str(sid) in summary_to_pids
        ]
        query_to_summary_scores[str(qid)] = kept_pairs
    stats = {
        "summary_shadow_index_path": str(path),
        "summary_shadow_active_summaries": len(summary_to_pids),
        "summary_shadow_total_summaries": len(raw_sum2pids),
        "summary_shadow_avg_pids_per_summary": round(
            sum(len(v) for v in summary_to_pids.values()) / max(1, len(summary_to_pids)), 3
        ),
        "summary_shadow_num_queries": len(query_to_summary_scores),
        "summary_shadow_avg_summaries_per_query": round(
            sum(len(v) for v in query_to_summary_scores.values()) / max(1, len(query_to_summary_scores)), 3
        ),
    }
    return summary_to_pids, query_to_summary_scores, stats


def compute_summary_shadow_boost(
    pids: Sequence[str],
    query_summaries: Sequence[Tuple[str, float]],
    summary_to_pids: Dict[str, List[str]],
    weight: float,
    cap: float,
    topn: int,
    child_alpha: float = 1.0,
    max_children: int = 0,
) -> Dict[str, float]:
    """For one query, accumulate boost onto child pids.

    Boost formula (fine-grained protecting):
        contrib = weight * cos(query, summary) / children_count^child_alpha
    child_alpha=1 fully normalizes a sibling group (no net advantage vs. a
    singleton summary); child_alpha=0 disables normalization (old behavior).
    """
    pid_set = set(pids)
    boost: Dict[str, float] = defaultdict(float)
    used = query_summaries[:topn] if topn > 0 else query_summaries
    for sid, sc in used:
        children = summary_to_pids.get(sid)
        if not children:
            continue
        if max_children and len(children) > max_children:
            continue
        denom = max(1, len(children)) ** max(0.0, float(child_alpha))
        contrib = weight * float(sc) / denom
        for pid in children:
            if pid in pid_set:
                boost[pid] = min(boost[pid] + contrib, cap)
    return dict(boost)


def merge_adjacencies(
    parts: Sequence[Dict[str, Sequence[Any]]],
    combine: str = "max",
) -> Dict[str, List[Tuple[str, float]]]:
    """Merge multiple adjacencies into one.

    combine='max' : per-edge weight is max across sources (legacy, source-agnostic)
    combine='sum' : per-edge weight is sum across sources (strong+weak accumulate)
    """
    weighted: Dict[str, Dict[str, float]] = defaultdict(dict)
    for adjacency in parts:
        for pid, nbrs in adjacency.items():
            for nbr_pid, weight in iter_neighbors(nbrs):
                if nbr_pid == pid:
                    continue
                w = float(weight)
                cur = weighted[pid].get(nbr_pid, 0.0)
                if combine == "sum":
                    weighted[pid][nbr_pid] = cur + w
                else:  # max
                    weighted[pid][nbr_pid] = max(cur, w)
    merged: Dict[str, List[Tuple[str, float]]] = {}
    for pid, nbrs in weighted.items():
        merged[pid] = sorted(nbrs.items(), key=lambda x: (x[1], x[0]), reverse=True)
    return merged


def scale_adjacency_weights(
    adjacency: Dict[str, Sequence[Any]],
    factor: float,
) -> Dict[str, List[Tuple[str, float]]]:
    """Multiply every edge weight by `factor`. Returns a new adjacency."""
    if factor == 1.0:
        # still normalize to List[Tuple[str,float]]
        out: Dict[str, List[Tuple[str, float]]] = {}
        for pid, nbrs in adjacency.items():
            out[pid] = [(nbr, float(w)) for nbr, w in iter_neighbors(nbrs)]
        return out
    scaled: Dict[str, List[Tuple[str, float]]] = {}
    for pid, nbrs in adjacency.items():
        scaled[pid] = [(nbr, float(w) * factor) for nbr, w in iter_neighbors(nbrs)]
    return scaled


def compute_graph_weighted_prior(
    adjacency: Dict[str, Sequence[Any]],
) -> Dict[str, float]:
    """Prior based on sum-of-weights degree (weight-aware version of static prior)."""
    wdeg: Dict[str, float] = {}
    for pid, nbrs in adjacency.items():
        s = 0.0
        seen: Set[str] = set()
        for nbr, w in iter_neighbors(nbrs):
            if nbr in seen:
                continue
            seen.add(nbr)
            s += float(w)
        wdeg[pid] = s
    if not wdeg:
        return {}
    max_w = max(wdeg.values())
    if max_w <= 0:
        return {pid: 0.0 for pid in wdeg}
    denom = math.log1p(max_w)
    return {
        pid: (math.log1p(w) / denom if w > 0 else 0.0)
        for pid, w in wdeg.items()
    }


def build_rank_scores(pids: Sequence[str], mode: str) -> Dict[str, float]:
    n = max(1, len(pids))
    scores: Dict[str, float] = {}
    for rank, pid in enumerate(pids):
        if mode == "linear":
            score = (n - rank) / n
        elif mode == "reciprocal":
            score = 1.0 / (rank + 1)
        elif mode == "log":
            score = 1.0 / math.log2(rank + 2)
        else:
            raise ValueError(f"unsupported score mode: {mode}")
        scores[pid] = score
    return scores


def iter_neighbors(nbrs: Sequence[Any]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for entry in nbrs:
        if isinstance(entry, str):
            out.append((entry, 1.0))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 1:
            nbr = str(entry[0])
            weight = float(entry[1]) if len(entry) > 1 else 1.0
            out.append((nbr, weight))
    return out


def compute_graph_static_prior(
    adjacency: Dict[str, Sequence[Any]],
) -> Dict[str, float]:
    degree_map = {
        pid: len({nbr for nbr, _ in iter_neighbors(nbrs)})
        for pid, nbrs in adjacency.items()
    }
    if not degree_map:
        return {}
    max_degree = max(degree_map.values())
    if max_degree <= 0:
        return {pid: 0.0 for pid in degree_map}
    denom = math.log1p(max_degree)
    return {
        pid: (math.log1p(deg) / denom if deg > 0 else 0.0)
        for pid, deg in degree_map.items()
    }


def apply_static_prior(
    pids: Sequence[str],
    base_scores: Dict[str, float],
    prior: Dict[str, float],
    weight: float,
) -> Tuple[List[str], Dict[str, float]]:
    new_scores = {
        pid: base_scores[pid] + weight * prior.get(pid, 0.0)
        for pid in pids
    }
    reranked = sorted(
        pids,
        key=lambda pid: (new_scores[pid], base_scores[pid]),
        reverse=True,
    )
    return reranked, new_scores


def compute_neighbor_boost(
    pids: Sequence[str],
    base_scores: Dict[str, float],
    adjacency: Dict[str, Sequence[Any]],
    rerank_topn: int,
    neighbor_decay: float,
    boost_cap: float,
) -> Dict[str, float]:
    rank_index = {pid: idx for idx, pid in enumerate(pids)}
    boost: Dict[str, float] = defaultdict(float)

    for seed_rank, seed_pid in enumerate(pids[:rerank_topn]):
        seed_score = base_scores[seed_pid]
        for nbr_pid, edge_w in iter_neighbors(adjacency.get(seed_pid, [])):
            nbr_rank = rank_index.get(nbr_pid)
            if nbr_rank is None or nbr_rank <= seed_rank:
                continue
            boost[nbr_pid] = min(
                boost[nbr_pid] + seed_score * neighbor_decay * edge_w,
                boost_cap,
            )

    return dict(boost)


def apply_neighbor_prop(
    pids: Sequence[str],
    base_scores: Dict[str, float],
    adjacency: Dict[str, Sequence[Any]],
    rerank_topn: int,
    neighbor_decay: float,
    boost_cap: float,
    static_prior: Dict[str, float] | None = None,
    static_weight: float = 0.0,
) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
    boost = compute_neighbor_boost(
        pids=pids,
        base_scores=base_scores,
        adjacency=adjacency,
        rerank_topn=rerank_topn,
        neighbor_decay=neighbor_decay,
        boost_cap=boost_cap,
    )
    new_scores: Dict[str, float] = {}
    for pid in pids:
        score = base_scores[pid] + boost.get(pid, 0.0)
        if static_prior and static_weight > 0:
            score += static_weight * static_prior.get(pid, 0.0)
        new_scores[pid] = score

    reranked = sorted(
        pids,
        key=lambda pid: (new_scores[pid], base_scores[pid]),
        reverse=True,
    )
    return reranked, new_scores, boost


def compute_metrics(
    ranking: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    ks: Tuple[int, ...] = (1, 2, 5, 10, 100),
) -> Dict[str, float]:
    recalls: Dict[int, List[float]] = {k: [] for k in ks}
    ndcgs: Dict[int, List[float]] = {k: [] for k in ks}
    hits: Dict[int, List[float]] = {k: [] for k in ks}
    first_hit_ranks: List[float] = []
    mrrs: List[float] = []

    for qid, pids in ranking.items():
        gold = qrels.get(qid) or {}
        gold_set = {pid for pid, rel in gold.items() if rel > 0}
        if not gold_set:
            continue

        rr = 0.0
        first_hit = 0
        for i, pid in enumerate(pids):
            if pid in gold_set:
                rr = 1.0 / (i + 1)
                first_hit = i + 1
                break
        mrrs.append(rr)
        first_hit_ranks.append(float(first_hit))

        for k in ks:
            hits_k = len(set(pids[:k]) & gold_set)
            recalls[k].append(hits_k / len(gold_set))
            hits[k].append(1.0 if hits_k > 0 else 0.0)

            dcg = 0.0
            for i, pid in enumerate(pids[:k]):
                if pid in gold_set:
                    dcg += 1.0 / math.log2(i + 2)
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold_set))))
            ndcgs[k].append(dcg / ideal if ideal > 0 else 0.0)

    out: Dict[str, float] = {
        "num_queries_evaluated": float(len(mrrs)),
        "mrr": float(sum(mrrs) / len(mrrs)) if mrrs else 0.0,
        "mean_first_hit_rank": float(sum(first_hit_ranks) / len(first_hit_ranks))
        if first_hit_ranks
        else 0.0,
        "median_first_hit_rank": float(statistics.median(first_hit_ranks))
        if first_hit_ranks
        else 0.0,
    }
    for k in ks:
        out[f"recall@{k}"] = float(sum(recalls[k]) / len(recalls[k])) if recalls[k] else 0.0
        out[f"ndcg@{k}"] = float(sum(ndcgs[k]) / len(ndcgs[k])) if ndcgs[k] else 0.0
        out[f"hit@{k}"] = float(sum(hits[k]) / len(hits[k])) if hits[k] else 0.0
    return out


def dump_rankings(path: Path, ranking: Dict[str, List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, pids in ranking.items():
            f.write(json.dumps({"query_id": qid, "top_k": pids}, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Graph rerank within an existing top-K ranking")
    ap.add_argument("--ranking", required=True, help="Input ranking JSONL from dense retrieval")
    ap.add_argument("--qrels", required=True, help="Qrels JSONL")
    ap.add_argument("--corpus", required=True, help="Corpus JSONL")
    ap.add_argument("--chunk-graph", default="data/01_graphs/chunk_virtual_nodes.json")
    ap.add_argument("--elements-graph", default="data/01_graphs/multimodal_elements.json")
    ap.add_argument("--summary-graph", default="")
    ap.add_argument("--output-dir", required=True, help="Directory for rankings + metrics")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--rerank-topn", type=int, default=100)
    ap.add_argument("--score-mode", choices=["linear", "reciprocal", "log"], default="linear")
    ap.add_argument("--static-weight", type=float, default=0.15)
    ap.add_argument("--neighbor-decay", type=float, default=0.20)
    ap.add_argument("--boost-cap", type=float, default=0.20)
    ap.add_argument(
        "--edge-types",
        nargs="+",
        choices=list(VIRTUAL_EDGE_TYPES),
        default=list(VIRTUAL_EDGE_TYPES),
        help="Which virtual typed graph edges to keep when virtual graph source is enabled",
    )
    ap.add_argument(
        "--graph-sources",
        nargs="+",
        choices=["explicit", "virtual", "cross_doc", "summary", "typed_crossdoc"],
        required=True,
        help="Which graph layers to use: explicit bridge edges, virtual chunk edges, cross-doc sim edges, summary shadow boost, typed cross-doc element similarity. Required to avoid silently running on the wrong graph layer.",
    )
    ap.add_argument(
        "--summary-shadow-index",
        default="data/01_graphs/summary_shadow_index.json",
        help="Pre-built summary shadow index (output of build_summary_shadow_index.py)",
    )
    ap.add_argument(
        "--summary-shadow-weight",
        type=float,
        default=0.10,
        help="Multiplier on per-query summary->pid shadow boost",
    )
    ap.add_argument(
        "--summary-shadow-topn",
        type=int,
        default=10,
        help="Per query: only use top-N summaries from the shadow index",
    )
    ap.add_argument(
        "--summary-shadow-cap",
        type=float,
        default=0.20,
        help="Per-pid cap on accumulated shadow boost",
    )
    ap.add_argument(
        "--summary-shadow-child-alpha",
        type=float,
        default=1.0,
        help="Boost divided by children_count^alpha. alpha=1 full normalize (anti-group), 0.5 mild, 0 off.",
    )
    ap.add_argument(
        "--summary-shadow-max-children",
        type=int,
        default=0,
        help="At rerank time skip summaries whose children_count exceeds this (0=no filter).",
    )
    ap.add_argument(
        "--hub-candidates",
        default="",
        help="Enriched hub candidate JSON for original explicit bridge edges",
    )
    ap.add_argument(
        "--cross-doc-edges",
        default="data/01_graphs/cross_doc_sim_edges.json",
        help="Cross-doc section similarity edges JSON (output of build_cross_doc_sim_edges.py)",
    )
    ap.add_argument("--explicit-weight", type=float, default=1.0,
                    help="Per-source scale on explicit bridge edge weights before merge")
    ap.add_argument("--virtual-weight", type=float, default=1.0,
                    help="Per-source scale on virtual chunk edge weights before merge")
    ap.add_argument("--crossdoc-weight", type=float, default=1.0,
                    help="Per-source scale on cross-doc edge weights before merge")
    ap.add_argument("--typed-crossdoc-weight", type=float, default=1.0,
                    help="Per-source scale on typed-crossdoc edge weights before merge")
    ap.add_argument("--typed-crossdoc-edges", default="data/01_graphs/typed_crossdoc_edges.json",
                    help="Typed cross-doc edges JSON (output of build_typed_crossdoc_edges.py)")
    ap.add_argument("--typed-crossdoc-types", nargs="+", default=["figure","formula","table","chunk"],
                    help="Which edge types to load from typed_crossdoc_edges")
    ap.add_argument("--typed-crossdoc-use-boost", action="store_true",
                    help="Use 'boosted_similarity' (includes citation boost) as edge weight")
    ap.add_argument("--merge-combine", choices=["max", "sum"], default="max",
                    help="How to combine an edge that appears in multiple sources")
    ap.add_argument("--prior-mode", choices=["degree", "weighted"], default="degree",
                    help="Static prior: 'degree' (legacy, source-agnostic) or 'weighted' (sum-of-weights)")
    args = ap.parse_args()

    ranking_path = PROJECT_ROOT / args.ranking if not Path(args.ranking).is_absolute() else Path(args.ranking)
    qrels_path = PROJECT_ROOT / args.qrels if not Path(args.qrels).is_absolute() else Path(args.qrels)
    corpus_path = PROJECT_ROOT / args.corpus if not Path(args.corpus).is_absolute() else Path(args.corpus)
    chunk_graph_path = PROJECT_ROOT / args.chunk_graph if not Path(args.chunk_graph).is_absolute() else Path(args.chunk_graph)
    elements_graph_path = PROJECT_ROOT / args.elements_graph if not Path(args.elements_graph).is_absolute() else Path(args.elements_graph)
    hub_candidates_path = (
        PROJECT_ROOT / args.hub_candidates
        if args.hub_candidates and not Path(args.hub_candidates).is_absolute()
        else Path(args.hub_candidates)
        if args.hub_candidates
        else None
    )
    output_dir = PROJECT_ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rankings = load_rankings(ranking_path, top_k=args.top_k)
    qrels = load_qrels(qrels_path)
    corpus_rows = load_jsonl(corpus_path)
    valid_pids = {str(row["passage_id"]) for row in corpus_rows}

    adjacency_parts: List[Dict[str, Sequence[Any]]] = []
    adjacency_stats: Dict[str, Any] = {
        "num_valid_pids": len(valid_pids),
        "graph_sources": sorted(args.graph_sources),
    }
    if "explicit" in args.graph_sources:
        if hub_candidates_path is None:
            raise ValueError("--hub-candidates is required when --graph-sources includes explicit")
        explicit_adj, explicit_stats = load_explicit_adjacency(
            hub_candidates_path=hub_candidates_path,
            valid_pids=valid_pids,
        )
        adjacency_parts.append(scale_adjacency_weights(explicit_adj, args.explicit_weight))
        adjacency_stats.update(explicit_stats)
        adjacency_stats["explicit_weight"] = args.explicit_weight
    extra_mappings: Dict[str, Any] = {}
    if "virtual" in args.graph_sources or "cross_doc" in args.graph_sources:
        virtual_adj, virtual_stats, extra_mappings = build_typed_adjacency(
            chunk_graph_path=chunk_graph_path,
            elements_graph_path=elements_graph_path,
            valid_pids=valid_pids,
            include_edge_types=set(args.edge_types) if "virtual" in args.graph_sources else set(),
        )
        if "virtual" in args.graph_sources:
            adjacency_parts.append(scale_adjacency_weights(virtual_adj, args.virtual_weight))
            adjacency_stats.update(virtual_stats)
            adjacency_stats["virtual_weight"] = args.virtual_weight
    if "cross_doc" in args.graph_sources:
        cross_doc_path = (
            PROJECT_ROOT / args.cross_doc_edges
            if not Path(args.cross_doc_edges).is_absolute()
            else Path(args.cross_doc_edges)
        )
        cross_adj, cross_stats = load_cross_doc_adjacency(
            cross_doc_edges_path=cross_doc_path,
            valid_pids=valid_pids,
            pid_to_chunk=extra_mappings.get("pid_to_chunk", {}),
            chunk_to_pids=extra_mappings.get("chunk_to_pids", {}),
            doc_section_to_chunks=extra_mappings.get("doc_section_to_chunks", {}),
        )
        adjacency_parts.append(scale_adjacency_weights(cross_adj, args.crossdoc_weight))
        adjacency_stats.update(cross_stats)
        adjacency_stats["crossdoc_weight"] = args.crossdoc_weight
    if "typed_crossdoc" in args.graph_sources:
        tcd_path = (
            PROJECT_ROOT / args.typed_crossdoc_edges
            if not Path(args.typed_crossdoc_edges).is_absolute()
            else Path(args.typed_crossdoc_edges)
        )
        tcd_adj, tcd_stats = load_typed_crossdoc_adjacency(
            edges_path=tcd_path,
            valid_pids=valid_pids,
            include_types=set(args.typed_crossdoc_types),
            use_boost=args.typed_crossdoc_use_boost,
        )
        adjacency_parts.append(scale_adjacency_weights(tcd_adj, args.typed_crossdoc_weight))
        adjacency_stats.update(tcd_stats)
        adjacency_stats["typed_crossdoc_weight"] = args.typed_crossdoc_weight

    summary_to_pids: Dict[str, List[str]] = {}
    query_summary_scores: Dict[str, List[Tuple[str, float]]] = {}
    use_summary_shadow = "summary" in args.graph_sources
    if use_summary_shadow:
        sum_idx_path = (
            PROJECT_ROOT / args.summary_shadow_index
            if not Path(args.summary_shadow_index).is_absolute()
            else Path(args.summary_shadow_index)
        )
        summary_to_pids, query_summary_scores, sum_stats = load_summary_shadow_index(
            path=sum_idx_path,
            valid_pids=valid_pids,
        )
        adjacency_stats.update(sum_stats)
        adjacency_stats["summary_shadow_weight"] = args.summary_shadow_weight
        adjacency_stats["summary_shadow_topn"] = args.summary_shadow_topn
        adjacency_stats["summary_shadow_cap"] = args.summary_shadow_cap
        adjacency_stats["summary_shadow_child_alpha"] = args.summary_shadow_child_alpha
        adjacency_stats["summary_shadow_max_children"] = args.summary_shadow_max_children

    if not adjacency_parts and not use_summary_shadow:
        raise ValueError("at least one graph source must be enabled")

    # Allow summary-only run: synthesize an empty adjacency in that case
    if not adjacency_parts:
        adjacency_parts.append({})

    adjacency = merge_adjacencies(adjacency_parts, combine=args.merge_combine)
    adjacency_stats["num_union_graph_pids"] = len(adjacency)
    adjacency_stats["union_graph_mapping_pct"] = round(
        len(adjacency) / max(1, len(valid_pids)) * 100, 2
    )
    adjacency_stats["merge_combine"] = args.merge_combine
    adjacency_stats["prior_mode"] = args.prior_mode
    if args.prior_mode == "weighted":
        static_prior = compute_graph_weighted_prior(adjacency)
    else:
        static_prior = compute_graph_static_prior(adjacency)

    methods_rankings: Dict[str, Dict[str, List[str]]] = {"baseline": rankings}
    query_stats: Dict[str, Dict[str, Any]] = {}
    static_rankings: Dict[str, List[str]] = {}
    neighbor_rankings: Dict[str, List[str]] = {}
    combined_rankings: Dict[str, List[str]] = {}
    summary_shadow_rankings: Dict[str, List[str]] = {}
    summary_plus_static_rankings: Dict[str, List[str]] = {}

    queries_with_static_overlap = 0
    queries_with_neighbor_signal = 0
    queries_with_summary_signal = 0
    boosted_candidates_counts: List[int] = []
    mapped_candidates_counts: List[int] = []
    summary_boosted_candidates_counts: List[int] = []

    for qid, pids in rankings.items():
        base_scores = build_rank_scores(pids, mode=args.score_mode)
        mapped_count = sum(1 for pid in pids if pid in adjacency)
        mapped_candidates_counts.append(mapped_count)

        static_overlap = any(pid in static_prior for pid in pids)
        if static_overlap:
            queries_with_static_overlap += 1

        reranked_static, _ = apply_static_prior(
            pids=pids,
            base_scores=base_scores,
            prior=static_prior,
            weight=args.static_weight,
        )
        static_rankings[qid] = reranked_static

        reranked_neighbor, _, boost = apply_neighbor_prop(
            pids=pids,
            base_scores=base_scores,
            adjacency=adjacency,
            rerank_topn=min(args.rerank_topn, len(pids)),
            neighbor_decay=args.neighbor_decay,
            boost_cap=args.boost_cap,
        )
        if boost:
            queries_with_neighbor_signal += 1
        boosted_candidates_counts.append(len(boost))
        neighbor_rankings[qid] = reranked_neighbor

        reranked_combined, _, _ = apply_neighbor_prop(
            pids=pids,
            base_scores=base_scores,
            adjacency=adjacency,
            rerank_topn=min(args.rerank_topn, len(pids)),
            neighbor_decay=args.neighbor_decay,
            boost_cap=args.boost_cap,
            static_prior=static_prior,
            static_weight=args.static_weight,
        )
        combined_rankings[qid] = reranked_combined

        # ---- Summary shadow boost (per-query, no candidate expansion) ----
        if use_summary_shadow:
            sb = compute_summary_shadow_boost(
                pids=pids,
                query_summaries=query_summary_scores.get(qid, []),
                summary_to_pids=summary_to_pids,
                weight=args.summary_shadow_weight,
                cap=args.summary_shadow_cap,
                topn=args.summary_shadow_topn,
                child_alpha=args.summary_shadow_child_alpha,
                max_children=args.summary_shadow_max_children,
            )
            if sb:
                queries_with_summary_signal += 1
            summary_boosted_candidates_counts.append(len(sb))

            shadow_scores = {pid: base_scores[pid] + sb.get(pid, 0.0) for pid in pids}
            summary_shadow_rankings[qid] = sorted(
                pids,
                key=lambda pid: (shadow_scores[pid], base_scores[pid]),
                reverse=True,
            )
            # combine: explicit static prior + summary shadow boost
            combined_scores = {
                pid: base_scores[pid]
                + args.static_weight * static_prior.get(pid, 0.0)
                + sb.get(pid, 0.0)
                for pid in pids
            }
            summary_plus_static_rankings[qid] = sorted(
                pids,
                key=lambda pid: (combined_scores[pid], base_scores[pid]),
                reverse=True,
            )

        query_stats[qid] = {
            "mapped_candidates": mapped_count,
            "boosted_candidates": len(boost),
            "static_overlap": static_overlap,
            "summary_boosted_candidates": (
                len(sb) if use_summary_shadow else 0
            ),
        }

    methods_rankings["graph_static_prior"] = static_rankings
    methods_rankings["graph_neighbor_prop"] = neighbor_rankings
    methods_rankings["graph_static_plus_neighbor"] = combined_rankings
    if use_summary_shadow:
        methods_rankings["graph_summary_shadow"] = summary_shadow_rankings
        methods_rankings["graph_summary_plus_static"] = summary_plus_static_rankings

    summary: Dict[str, Any] = {
        "config": {
            "top_k": args.top_k,
            "rerank_topn": args.rerank_topn,
            "score_mode": args.score_mode,
            "static_weight": args.static_weight,
            "neighbor_decay": args.neighbor_decay,
            "boost_cap": args.boost_cap,
            "graph_sources": sorted(args.graph_sources),
            "edge_types": sorted(args.edge_types),
            "ranking": str(ranking_path),
            "qrels": str(qrels_path),
            "corpus": str(corpus_path),
            "chunk_graph": str(chunk_graph_path),
            "elements_graph": str(elements_graph_path),
            "hub_candidates": str(hub_candidates_path) if hub_candidates_path else "",
        },
        "graph_stats": {
            **adjacency_stats,
            "num_queries": len(rankings),
            "queries_with_static_overlap": queries_with_static_overlap,
            "queries_with_static_overlap_pct": round(
                queries_with_static_overlap / max(1, len(rankings)) * 100, 2
            ),
            "queries_with_neighbor_signal": queries_with_neighbor_signal,
            "queries_with_neighbor_signal_pct": round(
                queries_with_neighbor_signal / max(1, len(rankings)) * 100, 2
            ),
            "avg_mapped_candidates_per_query": round(
                sum(mapped_candidates_counts) / max(1, len(mapped_candidates_counts)), 3
            ),
            "avg_boosted_candidates_per_query": round(
                sum(boosted_candidates_counts) / max(1, len(boosted_candidates_counts)), 3
            ),
            "queries_with_summary_signal": queries_with_summary_signal,
            "queries_with_summary_signal_pct": round(
                queries_with_summary_signal / max(1, len(rankings)) * 100, 2
            ),
            "avg_summary_boosted_candidates_per_query": round(
                sum(summary_boosted_candidates_counts) / max(1, len(summary_boosted_candidates_counts)), 3
            ) if summary_boosted_candidates_counts else 0.0,
        },
        "results": {},
    }

    for method, method_ranking in methods_rankings.items():
        metrics = compute_metrics(method_ranking, qrels)
        metrics["num_queries_total"] = float(len(rankings))
        summary["results"][method] = metrics
        dump_rankings(output_dir / f"ranking_{method}.jsonl", method_ranking)
        with open(output_dir / f"metrics_{method}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(output_dir / "query_graph_stats.json", "w", encoding="utf-8") as f:
        json.dump(query_stats, f, indent=2, ensure_ascii=False)
    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("== Graph top-k rerank summary ==")
    print(json.dumps(summary["graph_stats"], indent=2, ensure_ascii=False))
    print()
    for method, metrics in summary["results"].items():
        print(
            f"{method:>26}  "
            f"R@1={metrics['recall@1']:.4f}  "
            f"R@2={metrics['recall@2']:.4f}  "
            f"R@5={metrics['recall@5']:.4f}  "
            f"R@10={metrics['recall@10']:.4f}  "
            f"R@100={metrics['recall@100']:.4f}  "
            f"MRR={metrics['mrr']:.4f}  "
            f"NDCG@10={metrics['ndcg@10']:.4f}"
        )


if __name__ == "__main__":
    main()
