#!/usr/bin/env python3
"""
eval_graph_expanded_retrieval.py
=================================
Graph-expanded retrieval: take a dense top-K ranking (from eval_dense_retrieval.py)
and expand each result via graph edges from chunk_virtual_nodes.json and
llm_summary_virtual_nodes.json. Re-score with decay and re-evaluate metrics.

WHY MRR IS LOW (~0.10 on v1):
  - Every query has EXACTLY 3-5 positive passages (multi-hop, cross-modal)
  - 697/1541 positives are figures  → text=[Image: xxx.jpg], hard to rank high
  - 766/1541 positives are formulas → raw LaTeX, partially matching context
  - Even after v1 enrichment, 219/473 queries have NO hit in top-100
  - Median first-hit rank = 8: when found, model knows roughly where to look,
    but MRR penalises rank-8 hits as 1/8 = 0.125 only
  ⟹ Two complementary fixes:
    (A) Graph expansion (this script): hop from a found passage to its
        neighbours (same chunk, same section) to surface more positives.
    (B) Multimodal embeddings (future): encode figure images with CLIP /
        Qwen-VL and do image↔text retrieval for figure positives.

Graph expansion strategy:
  For each retrieved passage p at rank r with score s:
    - paragraph → parent chunk  (relation: chunk_contains_paragraph, reversed)
    - chunk → sibling chunks    (relation: chunk_sequence)
    - chunk → section           (relation: section_contains_chunk, reversed)
    - section → other chunks    (relation: section_contains_chunk)
    - chunk → its llm_summary   (relation: summary_of, reversed)
    - summary → its chunk       (relation: summary_of)
  Each neighbour gets score: s * decay^hop_distance
  If a passage_id is already in the top-K with a higher score, skip.

Usage (CPU, runs on login node):
  python scripts/eval_graph_expanded_retrieval.py \\
      --ranking data/05_eval/dense_retrieval/ranking_v1_enriched.jsonl \\
      --qrels   data/05_eval/dense_retrieval/augmented/qrels_v1.jsonl \\
      --corpus  data/05_eval/dense_retrieval/augmented/corpus_v1_enriched.jsonl \\
      --chunk-graph   data/01_graphs/chunk_virtual_nodes.json \\
      --summary-graph data/01_graphs/llm_summary_virtual_nodes.json \\
      --output  data/05_eval/dense_retrieval/eval_graph_expanded.json \\
      --decay 0.7 --max-hops 2 --top-k 100
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# ── IO ────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_adjacency(
    chunk_graph_path: Path,
    summary_graph_path: Path,
    elements_graph_path: Path,
    valid_pids: Set[str],
) -> Dict[str, List[Tuple[str, float]]]:
    """Build pid → [neighbour_pid] adjacency bridging corpus element pids to the
    chunk graph.

    THE NAMESPACE PROBLEM:
      - Corpus passage ids look like: ``1104.3913_1104.3913_formula_1``
        (format: {doc_id}_{element_id} where element_id={doc_id}_{type}_{n})
      - Chunk graph node ids look like: ``1104.3913_chunk_7``
      - These two namespaces never overlap → naive filter gives 0 neighbours.

    THE BRIDGE (position_idx):
      1. multimodal_elements.json  →  element_id  →  position_idx
      2. chunk_virtual_nodes.json  →  chunk paragraph_indices  →  position_idx → chunk_id
      3. corpus_pid  →  element_id  →  position_idx  →  chunk_id
      4. Two corpus pids are connected iff they map to the same chunk *or* to
         adjacent chunks (via chunk_sequence edge) *or* to chunks in the same
         section (via section_contains_chunk).

    Returns a dict: pid → list of (neighbour_pid).  (hop weight applied later)
    """
    adj: Dict[str, List[str]] = defaultdict(list)

    # ── Step 1 : load chunk graph, build per-doc position→chunk index ─────────
    print(f"[info] loading chunk graph from {chunk_graph_path} ...", flush=True)
    t0 = time.time()
    chunk_data = json.loads(chunk_graph_path.read_bytes())
    print(f"[info] loaded in {time.time()-t0:.1f}s", flush=True)

    # per-doc: pos_idx → chunk_id
    doc_pos_to_chunk: Dict[str, Dict[int, str]] = {}
    # per-doc: chunk_id → set of adjacent chunk_ids (same-section + sequence)
    doc_chunk_neighbors: Dict[str, Dict[str, Set[str]]] = {}

    for doc_id, doc in chunk_data.get("documents", {}).items():
        pos_to_chunk: Dict[int, str] = {}
        for nid, node in doc.get("nodes", {}).items():
            for pidx in node.get("paragraph_indices", []):
                pos_to_chunk[pidx] = nid
        doc_pos_to_chunk[doc_id] = pos_to_chunk

        section_to_chunks: Dict[str, List[str]] = defaultdict(list)
        chunk_nbrs: Dict[str, Set[str]] = defaultdict(set)

        for edge in doc.get("edges", []):
            rel = edge.get("relation", "")
            src, tgt = edge.get("source", ""), edge.get("target", "")
            if rel == "section_contains_chunk":
                section_to_chunks[src].append(tgt)
            elif rel == "chunk_sequence":
                chunk_nbrs[src].add(tgt)
                chunk_nbrs[tgt].add(src)

        # same-section chunks are also neighbours
        for sec_title, chunks in section_to_chunks.items():
            for c in chunks:
                for c2 in chunks:
                    if c != c2:
                        chunk_nbrs[c].add(c2)

        doc_chunk_neighbors[doc_id] = chunk_nbrs

    # ── Step 2 : load multimodal elements, build lookup tables ───────────────
    print(f"[info] loading multimodal elements from {elements_graph_path} ...", flush=True)
    elem_data = json.loads(elements_graph_path.read_bytes())

    # Two lookup strategies (both tried, first wins):
    #   A) Direct element_id match: corpus pid = {doc_id}_{element_id}
    #   B) 0-indexed match: corpus uses 0-based numbering, elements use 1-based
    #      e.g. corpus formula_0 → elements formula_1; corpus fig_N → elements figure_{N+1}
    #   Type alias: corpus "fig" → elements "figure"

    # Build (doc_id, element_type_normalised, zero_idx) → position_idx
    doc_type0_to_pos: Dict[str, Dict[Tuple[str, int], int]] = {}
    # Also exact element_id → position_idx (for strategy A)
    doc_elemid_to_pos: Dict[str, Dict[str, int]] = {}

    for doc_id, doc in elem_data.get("documents", {}).items():
        type0_map: Dict[Tuple[str, int], int] = {}
        eid_map: Dict[str, int] = {}
        for elem_id, elem in doc.get("elements", {}).items():
            pidx = elem.get("position_idx")
            if pidx is None:
                continue
            eid_map[elem_id] = pidx
            # parse: last token = 1-based number
            parts = elem_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                one_idx = int(parts[1])
                # determine type from the prefix: {doc_id}_{type}
                prefix = parts[0]
                etype = prefix[len(doc_id) + 1:] if prefix.startswith(doc_id + "_") else prefix.rsplit("_", 1)[-1]
                type0_map[(etype, one_idx - 1)] = pidx  # store as 0-indexed
        doc_type0_to_pos[doc_id] = type0_map
        doc_elemid_to_pos[doc_id] = eid_map

    _TYPE_ALIAS = {"fig": "figure"}  # corpus uses "fig", elements use "figure"

    # doc_id → { corpus_element_id → chunk_id }
    doc_elem_to_chunk: Dict[str, Dict[str, str]] = {}
    for doc_id in elem_data.get("documents", {}).keys():
        pos_to_chunk = doc_pos_to_chunk.get(doc_id, {})
        type0_map = doc_type0_to_pos.get(doc_id, {})
        eid_map = doc_elemid_to_pos.get(doc_id, {})
        cmap: Dict[str, str] = {}

        # Populated lazily during corpus-pid mapping below; store lookup funcs
        doc_elem_to_chunk[doc_id] = cmap
        # attach helper data as attributes for later access
        doc_elem_to_chunk[doc_id + "__type0"] = type0_map   # type: ignore[assignment]
        doc_elem_to_chunk[doc_id + "__eid"]   = eid_map     # type: ignore[assignment]
        doc_elem_to_chunk[doc_id + "__p2c"]   = pos_to_chunk  # type: ignore[assignment]

    # ── Step 3 : map corpus pids to chunk_ids ─────────────────────────────────
    # corpus pid format: {doc_id}_{doc_id}_{type}_{N}  (N is 0-indexed)
    # element_id format: {doc_id}_{type}_{K}           (K is 1-indexed)
    # type alias: corpus "fig" == elements "figure"
    pid_to_chunk: Dict[str, str] = {}
    for pid in valid_pids:
        for doc_id in doc_type0_to_pos:
            if not pid.startswith(doc_id + "_"):
                continue
            # strip {doc_id}_ prefix to get the element part: {doc_id}_{type}_{N}
            rest = pid[len(doc_id) + 1:]
            # rest should start with doc_id again → strip again
            if rest.startswith(doc_id + "_"):
                rest = rest[len(doc_id) + 1:]
            # rest = {type}_{N}
            parts = rest.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                raw_type = parts[0]
                n = int(parts[1])
                etype = _TYPE_ALIAS.get(raw_type, raw_type)
                pos_to_chunk = doc_elem_to_chunk[doc_id + "__p2c"]   # type: ignore

                # Strategy B: 0-indexed match
                type0_map = doc_elem_to_chunk[doc_id + "__type0"]     # type: ignore
                pidx = type0_map.get((etype, n))
                if pidx is not None:
                    chunk_id = pos_to_chunk.get(pidx)
                    if chunk_id:
                        pid_to_chunk[pid] = chunk_id

                # Strategy A: direct element_id match (fallback)
                if pid not in pid_to_chunk:
                    elem_id = f"{doc_id}_{raw_type}_{n}"
                    eid_map = doc_elem_to_chunk[doc_id + "__eid"]     # type: ignore
                    pidx2 = eid_map.get(elem_id)
                    if pidx2 is not None:
                        chunk_id2 = pos_to_chunk.get(pidx2)
                        if chunk_id2:
                            pid_to_chunk[pid] = chunk_id2
            break

    print(f"[info] mapped {len(pid_to_chunk)}/{len(valid_pids)} corpus pids to chunks", flush=True)

    # ── Step 4 : build chunk_id → [corpus_pids in that chunk] ────────────────
    chunk_to_pids: Dict[str, List[str]] = defaultdict(list)
    for pid, chunk_id in pid_to_chunk.items():
        chunk_to_pids[chunk_id].append(pid)

    # ── Step 5 : connect corpus pids via shared chunk or adjacent chunks ──────
    for pid, chunk_id in pid_to_chunk.items():
        doc_id = pid.split("_")[0]  # works for "1104.3913" prefix
        # find full doc_id robustly
        for d in doc_chunk_neighbors:
            if pid.startswith(d + "_"):
                doc_id = d
                break

        # same-chunk neighbours (other corpus pids in the same chunk)
        for nb_pid in chunk_to_pids.get(chunk_id, []):
            if nb_pid != pid:
                adj[pid].append(nb_pid)

        # neighboring-chunk neighbours
        for nb_chunk in doc_chunk_neighbors.get(doc_id, {}).get(chunk_id, set()):
            for nb_pid in chunk_to_pids.get(nb_chunk, []):
                adj[pid].append(nb_pid)

    # ── Summary graph ─────────────────────────────────────────────────────────
    # Summary virtual nodes (llm_summary_virtual_nodes) have ids like
    # "{doc_id}_section_summary_{i}" or "{doc_id}_chunk_summary_{i}".
    # They are NOT corpus passages so they can't appear in valid_pids, but
    # we can use them as intermediate hops: if summary S is "summary_of" chunk C,
    # and C contains corpus pid P, then S is a proxy for P's neighbours.
    # Concretely: build summary_id → [corpus pids whose chunk the summary covers].
    if summary_graph_path.exists():
        print(f"[info] loading summary graph ...", flush=True)
        sum_data = json.loads(summary_graph_path.read_bytes())
        # summary_of edge: source=summary_id, target=chunk_id (or section title)
        summary_to_chunk: Dict[str, str] = {}
        for doc_id, doc in sum_data.get("documents", {}).items():
            for edge in doc.get("edges", []):
                if edge.get("relation") == "summary_of":
                    summary_to_chunk[edge["source"]] = edge["target"]
        # summary_id → corpus pids
        for summary_id, chunk_id in summary_to_chunk.items():
            pids_in_chunk = chunk_to_pids.get(chunk_id, [])
            for pid in pids_in_chunk:
                for nb_pid in pids_in_chunk:
                    if nb_pid != pid:
                        adj[pid].append(nb_pid)
                # also connect to pids in neighbor chunks bridged via the summary
                doc_id = chunk_id.rsplit("_chunk_", 1)[0] if "_chunk_" in chunk_id else ""
                for nb_chunk in doc_chunk_neighbors.get(doc_id, {}).get(chunk_id, set()):
                    for nb_pid in chunk_to_pids.get(nb_chunk, []):
                        adj[pid].append(nb_pid)

    # Filter to valid corpus pids only; deduplicate
    filtered: Dict[str, List[str]] = {}
    for pid, neighbours in adj.items():
        if pid not in valid_pids:
            continue
        valid_nb = list(dict.fromkeys(n for n in neighbours if n in valid_pids and n != pid))
        if valid_nb:
            filtered[pid] = valid_nb

    print(f"[info] adjacency built: {len(filtered)} nodes with neighbours "
          f"(total entries: {sum(len(v) for v in filtered.values())})", flush=True)
    return filtered


# ── Graph expansion ───────────────────────────────────────────────────────────

def expand_ranking(
    original_ranking: List[str],
    original_scores: Dict[str, float],
    adj: Dict[str, List[str]],
    decay: float,
    max_hops: int,
    top_k: int,
) -> List[str]:
    """BFS-expand original_ranking via adj, propagate scores with decay.

    Returns re-ranked list of up to top_k passage_ids.
    """
    # Start: scores from dense retrieval
    scores: Dict[str, float] = dict(original_scores)
    visited: Set[str] = set(original_ranking)
    frontier: List[Tuple[str, float]] = [(pid, original_scores[pid]) for pid in original_ranking]

    for hop in range(1, max_hops + 1):
        next_frontier: List[Tuple[str, float]] = []
        for pid, score in frontier:
            for nb in adj.get(pid, []):
                new_score = score * (decay ** hop)
                if nb not in visited:
                    visited.add(nb)
                    scores[nb] = new_score
                    next_frontier.append((nb, new_score))
                elif new_score > scores.get(nb, 0.0):
                    scores[nb] = new_score
        frontier = next_frontier
        if not frontier:
            break

    # Sort all candidates by score descending
    sorted_pids = sorted(scores.keys(), key=lambda p: -scores[p])
    return sorted_pids[:top_k]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    ranking: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    ks: Tuple[int, ...] = (1, 5, 10, 100),
) -> Dict[str, float]:
    import numpy as np
    recalls: Dict[int, List[float]] = {k: [] for k in ks}
    ndcgs: Dict[int, List[float]] = {k: [] for k in ks}
    mrrs: List[float] = []

    for qid, pids in ranking.items():
        gold = {pid for pid, rel in (qrels.get(qid) or {}).items() if rel > 0}
        if not gold:
            continue
        rr = 0.0
        for i, pid in enumerate(pids):
            if pid in gold:
                rr = 1.0 / (i + 1)
                break
        mrrs.append(rr)
        for k in ks:
            hits = len(set(pids[:k]) & gold)
            recalls[k].append(hits / len(gold))
            dcg = sum(1.0 / math.log2(i + 2) for i, pid in enumerate(pids[:k]) if pid in gold)
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold))))
            ndcgs[k].append(dcg / ideal if ideal > 0 else 0.0)

    out: Dict[str, float] = {
        "num_queries": float(len(mrrs)),
        "mrr": float(sum(mrrs) / len(mrrs)) if mrrs else 0.0,
    }
    for k in ks:
        out[f"recall@{k}"] = float(sum(recalls[k]) / len(recalls[k])) if recalls[k] else 0.0
        out[f"ndcg@{k}"] = float(sum(ndcgs[k]) / len(ndcgs[k])) if ndcgs[k] else 0.0
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Graph-expanded retrieval evaluation")
    ap.add_argument("--ranking", default="data/05_eval/dense_retrieval/ranking_v1_enriched.jsonl")
    ap.add_argument("--qrels",   default="data/05_eval/dense_retrieval/augmented/qrels_v1.jsonl")
    ap.add_argument("--corpus",  default="data/05_eval/dense_retrieval/augmented/corpus_v1_enriched.jsonl")
    ap.add_argument("--chunk-graph",    default="data/01_graphs/chunk_virtual_nodes.json")
    ap.add_argument("--summary-graph",  default="data/01_graphs/llm_summary_virtual_nodes.json")
    ap.add_argument("--elements-graph", default="data/03_queries/M4query_v1/graphs/multimodal_elements.json",
                    help="multimodal_elements.json with element position_idx for namespace bridge")
    ap.add_argument("--output",  default="data/05_eval/dense_retrieval/eval_graph_expanded.json")
    ap.add_argument("--save-ranking", default=None)
    ap.add_argument("--decay",    type=float, default=0.7,
                    help="Score decay per hop (default: 0.7)")
    ap.add_argument("--max-hops", type=int, default=2,
                    help="Max BFS hops from each dense result (default: 2)")
    ap.add_argument("--top-k",   type=int, default=100)
    ap.add_argument("--decay-sweep", action="store_true",
                    help="Run grid search over decay in {0.3, 0.5, 0.7, 0.9}")
    args = ap.parse_args()

    # ── Load corpus (for valid pid set) ──
    print("[info] loading corpus ...", flush=True)
    corpus_rows = load_jsonl(Path(args.corpus))
    valid_pids: Set[str] = {r["passage_id"] for r in corpus_rows}
    print(f"[info] {len(valid_pids)} corpus passages", flush=True)

    # ── Load qrels ──
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for row in load_jsonl(Path(args.qrels)):
        qrels[str(row["query_id"])][str(row["passage_id"])] = int(row.get("relevance", 1))

    # ── Load dense ranking ──
    print("[info] loading dense ranking ...", flush=True)
    dense_ranking: Dict[str, List[str]] = {}
    dense_scores: Dict[str, Dict[str, float]] = {}
    for row in load_jsonl(Path(args.ranking)):
        qid = str(row["query_id"])
        pids = row["top_k"][:args.top_k]
        dense_ranking[qid] = pids
        # Assign monotonically decreasing scores (rank-based, no raw scores saved)
        dense_scores[qid] = {pid: 1.0 / (rank + 1) for rank, pid in enumerate(pids)}
    print(f"[info] {len(dense_ranking)} queries in ranking", flush=True)

    # ── Build graph adjacency ──
    adj = build_adjacency(
        Path(args.chunk_graph),
        Path(args.summary_graph),
        Path(args.elements_graph),
        valid_pids,
    )

    # ── Baseline metrics (dense only) ──
    baseline_metrics = compute_metrics(dense_ranking, qrels)
    print(f"\n── Dense baseline (no expansion) ──")
    for k in (1, 5, 10, 100):
        print(f"  Recall@{k:<3} = {baseline_metrics[f'recall@{k}']:.4f}   NDCG@{k:<3} = {baseline_metrics[f'ndcg@{k}']:.4f}")
    print(f"  MRR = {baseline_metrics['mrr']:.4f}")

    # ── Decay sweep or single run ──
    decays = [0.3, 0.5, 0.7, 0.9] if args.decay_sweep else [args.decay]
    best_metrics = None
    best_decay = args.decay
    best_ranking: Dict[str, List[str]] = {}

    for decay in decays:
        expanded: Dict[str, List[str]] = {}
        for qid, orig_pids in dense_ranking.items():
            expanded[qid] = expand_ranking(
                orig_pids,
                dense_scores[qid],
                adj,
                decay=decay,
                max_hops=args.max_hops,
                top_k=args.top_k,
            )

        m = compute_metrics(expanded, qrels)
        marker = " ←" if len(decays) > 1 else ""
        if best_metrics is None or m["recall@100"] > best_metrics.get("recall@100", 0):
            best_metrics = m
            best_decay = decay
            best_ranking = expanded
            if len(decays) > 1:
                marker = " ← best so far"

        print(f"\n── decay={decay}, max_hops={args.max_hops} ──{marker}")
        for k in (1, 5, 10, 100):
            delta_r = m[f"recall@{k}"] - baseline_metrics[f"recall@{k}"]
            delta_n = m[f"ndcg@{k}"] - baseline_metrics[f"ndcg@{k}"]
            print(f"  Recall@{k:<3} = {m[f'recall@{k}']:.4f}  ({delta_r:+.4f})   "
                  f"NDCG@{k:<3} = {m[f'ndcg@{k}']:.4f}  ({delta_n:+.4f})")
        delta_mrr = m["mrr"] - baseline_metrics["mrr"]
        print(f"  MRR = {m['mrr']:.4f}  ({delta_mrr:+.4f})")

    # ── Save best result ──
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "config": {
            "ranking_file": args.ranking,
            "corpus_file": args.corpus,
            "chunk_graph": args.chunk_graph,
            "summary_graph": args.summary_graph,
            "best_decay": best_decay,
            "max_hops": args.max_hops,
            "top_k": args.top_k,
        },
        "baseline": baseline_metrics,
        "graph_expanded": best_metrics,
        "delta": {
            k: round(best_metrics[k] - baseline_metrics[k], 6)
            for k in best_metrics if k != "num_queries"
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[info] saved to {out_path}")

    if args.save_ranking:
        rp = Path(args.save_ranking)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w", encoding="utf-8") as f:
            for qid, pids in best_ranking.items():
                f.write(json.dumps({"query_id": qid, "top_k": pids}) + "\n")
        print(f"[info] saved expanded ranking to {rp}")


if __name__ == "__main__":
    main()
