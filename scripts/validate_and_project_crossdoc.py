#!/usr/bin/env python3
"""validate_and_project_crossdoc.py
====================================
Validate the cross-doc edges produced by build_crossdoc_gold57.py and project
chunk<->chunk cross-doc edges down to element<->element edges via
chunk_contains_element, so they become usable by eval_graph_topk_rerank.py
(which only understands element-level pid graphs).

Two outputs:
  1. {out}_report.json   — validation report
  2. {out}_projected.json — typed_crossdoc-format file that merges:
       a) the original figure/formula/table element edges (as-is)
       b) new element-element edges projected from chunk-chunk edges,
          assigned type = "chunk_mediated"

Validation checks
-----------------
  - doc-pair coverage: how many (doc_i, doc_j) among gold-57 are connected?
  - BBL overlap: among connected pairs, how many are BBL-cited?  How many
    BBL-cited gold pairs remain uncovered?
  - per-type breakdown: figure/formula/table/chunk edge counts, unique pids
  - top-100 query coverage: for each query, how many top-100 elements have
    at least one cross-doc neighbor (either direct element edge or
    chunk-mediated via its parent chunk)?
  - mediated pair blow-up: average chunk-pair fanout (how many element pairs
    does one chunk pair project to?)

Correctness heuristic for "跨法": a projection is "reasonable" when:
  - chunk-mediated element pairs preferentially connect same-type or
    citation-consistent elements
  - mediated pair fanout has a bounded cap (we enforce --chunk-project-cap)
  - coverage goes up (doc-pair / query-level) without inflating below-threshold
    edges
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gold_docs_from_qrels(qrels_path: Path) -> Set[str]:
    out = set()
    for row in load_jsonl(qrels_path):
        pid = str(row.get("passage_id", ""))
        if pid:
            out.add(pid.rsplit("_", 2)[0])
    return out


def load_bbl_pairs(citation_path: Path, gold: Set[str]) -> Set[Tuple[str, str]]:
    obj = json.loads(citation_path.read_bytes())
    pairs: Set[Tuple[str, str]] = set()
    for e in obj.get("edges", []):
        a, b = str(e.get("source", "")), str(e.get("target", ""))
        if a and b and a != b and a in gold and b in gold:
            pairs.add((min(a, b), max(a, b)))
    return pairs


def load_element_doc_map(corpus_path: Path, gold: Set[str]) -> Dict[str, str]:
    """pid -> doc_id for element corpus."""
    out = {}
    for row in load_jsonl(corpus_path):
        pid = str(row["passage_id"])
        doc = str(row.get("doc_id", ""))
        if doc in gold:
            out[pid] = doc
    return out


def load_top100(ranking_path: Path, top_k: int = 100) -> Dict[str, List[str]]:
    out = {}
    for row in load_jsonl(ranking_path):
        qid = str(row["query_id"])
        pids = [str(p) for p in (row.get("top_k") or row.get("passage_ids") or [])][:top_k]
        out[qid] = pids
    return out


# ── projection ────────────────────────────────────────────────────────────────

def project_chunk_edges(
    chunk_edges: List[dict],
    chunk_to_elems: Dict[str, List[str]],
    valid_elem_pids: Set[str],
    cap_per_pair: int,
    min_similarity: float,
) -> List[dict]:
    """For each chunk<->chunk edge, emit element<->element edges for every
    (e_a in chunk_A) x (e_b in chunk_B) combination, bounded by cap_per_pair.

    Returned edges use type="chunk_mediated" and carry:
      - parent_chunks: [chunk_A, chunk_B]
      - similarity / boosted_similarity copied from chunk edge
      - cite_boosted copied
    """
    out: List[dict] = []
    for e in chunk_edges:
        sim = float(e.get("similarity", 0.0))
        if sim < min_similarity:
            continue
        ca, cb = str(e["source"]), str(e["target"])
        ea_list = [x for x in chunk_to_elems.get(ca, []) if x in valid_elem_pids]
        eb_list = [x for x in chunk_to_elems.get(cb, []) if x in valid_elem_pids]
        if not ea_list or not eb_list:
            continue
        # simple cap: trim list lengths so |A|*|B| <= cap_per_pair
        while len(ea_list) * len(eb_list) > cap_per_pair:
            if len(ea_list) >= len(eb_list):
                ea_list = ea_list[: max(1, len(ea_list) - 1)]
            else:
                eb_list = eb_list[: max(1, len(eb_list) - 1)]
        for a in ea_list:
            for b in eb_list:
                if a == b:
                    continue
                out.append({
                    "source": a, "target": b,
                    "source_doc": e.get("source_doc", ""),
                    "target_doc": e.get("target_doc", ""),
                    "type": "chunk_mediated",
                    "similarity": round(sim, 4),
                    "boosted_similarity": round(float(e.get("boosted_similarity", sim)), 4),
                    "cite_boosted": bool(e.get("cite_boosted", False)),
                    "parent_chunks": [ca, cb],
                })
    return out


# ── validation ────────────────────────────────────────────────────────────────

def analyze(
    crossdoc: dict,
    bbl_pairs: Set[Tuple[str, str]],
    elem_doc: Dict[str, str],
    top100: Dict[str, List[str]],
    projected_element_edges: List[dict],
) -> dict:
    edges = crossdoc["edges"]
    chunk_to_elems = crossdoc.get("chunk_contains_element", {})

    # per-type
    type_counts: Counter = Counter(e["type"] for e in edges)
    type_doc_pairs: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    type_cite: Counter = Counter()
    for e in edges:
        pair = (min(e["source_doc"], e["target_doc"]),
                max(e["source_doc"], e["target_doc"]))
        type_doc_pairs[e["type"]].add(pair)
        if e.get("cite_boosted"):
            type_cite[e["type"]] += 1

    # aggregate doc pair coverage
    all_pairs: Set[Tuple[str, str]] = set()
    for s in type_doc_pairs.values():
        all_pairs |= s
    bbl_covered = all_pairs & bbl_pairs
    bbl_missing = bbl_pairs - all_pairs

    # chunk projection fanout stats
    fanout: List[int] = []
    for e in edges:
        if e["type"] != "chunk":
            continue
        a_elems = len(chunk_to_elems.get(str(e["source"]), []))
        b_elems = len(chunk_to_elems.get(str(e["target"]), []))
        fanout.append(a_elems * b_elems)
    fanout_stats = {}
    if fanout:
        fanout_sorted = sorted(fanout)
        fanout_stats = {
            "chunk_pairs": len(fanout),
            "mean_fanout": round(sum(fanout) / len(fanout), 2),
            "p50_fanout": fanout_sorted[len(fanout_sorted) // 2],
            "p95_fanout": fanout_sorted[int(len(fanout_sorted) * 0.95)],
            "max_fanout": max(fanout),
        }

    # build element adjacency (from element edges + projected chunk-mediated edges)
    elem_adj: Dict[str, Set[str]] = defaultdict(set)
    for e in edges:
        if e["type"] in {"figure", "formula", "table"}:
            a, b = str(e["source"]), str(e["target"])
            if a in elem_doc and b in elem_doc:
                elem_adj[a].add(b)
                elem_adj[b].add(a)
    elem_adj_after_projection = {k: set(v) for k, v in elem_adj.items()}
    for e in projected_element_edges:
        a, b = str(e["source"]), str(e["target"])
        if a in elem_doc and b in elem_doc:
            elem_adj_after_projection.setdefault(a, set()).add(b)
            elem_adj_after_projection.setdefault(b, set()).add(a)

    # query-level coverage (top-100 in-graph density)
    def query_cov(adj: Dict[str, Set[str]]) -> dict:
        cnt_q_any = cnt_q_cross = 0
        per_query_cross = []
        for qid, pids in top100.items():
            in_top = set(pids)
            has_any = False
            has_cross_in_top = 0
            for p in pids:
                nbrs = adj.get(p, set())
                if nbrs:
                    has_any = True
                has_cross_in_top += sum(1 for nb in nbrs if nb in in_top)
            if has_any:
                cnt_q_any += 1
            if has_cross_in_top > 0:
                cnt_q_cross += 1
            per_query_cross.append(has_cross_in_top)
        sorted_pq = sorted(per_query_cross)
        return {
            "num_queries": len(top100),
            "queries_with_any_neighbor_in_top100": cnt_q_any,
            "queries_with_in_top100_cross_pair": cnt_q_cross,
            "mean_cross_pairs_in_top100": round(sum(per_query_cross) / max(1, len(per_query_cross)), 2),
            "p50_cross_pairs_in_top100": sorted_pq[len(sorted_pq) // 2] if sorted_pq else 0,
            "p95_cross_pairs_in_top100": sorted_pq[int(len(sorted_pq) * 0.95)] if sorted_pq else 0,
        }

    return {
        "num_edges_total": len(edges),
        "num_edges_projected_chunk_mediated": len(projected_element_edges),
        "edges_per_type": dict(type_counts),
        "cite_boosted_per_type": dict(type_cite),
        "doc_pairs_per_type": {k: len(v) for k, v in type_doc_pairs.items()},
        "doc_pairs_total": len(all_pairs),
        "bbl_pairs_total": len(bbl_pairs),
        "bbl_pairs_covered": len(bbl_covered),
        "bbl_pairs_missing": len(bbl_missing),
        "bbl_pairs_missing_examples": list(list(bbl_missing)[:10]),
        "chunk_fanout": fanout_stats,
        "query_coverage_elem_only": query_cov(elem_adj),
        "query_coverage_elem_plus_chunk_mediated": query_cov(elem_adj_after_projection),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossdoc", required=True, help="build_crossdoc_gold57.py output json")
    ap.add_argument("--qrels", default="data/03_queries/M4query_v1/qrels.jsonl")
    ap.add_argument("--corpus", default="data/03_queries/M4query_v1/corpus.jsonl")
    ap.add_argument("--ranking", default="data/05_eval/dense_retrieval/rebuilt_20260417/ranking_qwen3_4B.jsonl")
    ap.add_argument("--citation-graph", default="data/01_graphs/citation_graph.json")
    ap.add_argument("--output-prefix", required=True,
                    help="Prefix for output files: <prefix>_report.json, <prefix>_projected.json")
    ap.add_argument("--chunk-project-cap", type=int, default=20,
                    help="Max element pairs projected from one chunk pair")
    ap.add_argument("--chunk-project-min-sim", type=float, default=0.0,
                    help="Skip chunk pairs below this raw similarity")
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    base = PROJECT_ROOT
    gold = load_gold_docs_from_qrels(base / args.qrels)
    print(f"[info] gold docs: {len(gold)}", flush=True)

    bbl = load_bbl_pairs(base / args.citation_graph, gold)
    print(f"[info] BBL pairs (gold only): {len(bbl)}", flush=True)

    elem_doc = load_element_doc_map(base / args.corpus, gold)
    print(f"[info] gold element passages: {len(elem_doc)}", flush=True)

    top100 = load_top100(base / args.ranking, top_k=args.top_k)
    print(f"[info] queries with ranking: {len(top100)}", flush=True)

    crossdoc = json.loads(Path(args.crossdoc).read_bytes())
    edges = crossdoc["edges"]
    chunk_to_elems: Dict[str, List[str]] = crossdoc.get("chunk_contains_element", {})
    print(f"[info] crossdoc edges: {len(edges)}; chunk->element map: {len(chunk_to_elems)}",
          flush=True)

    # project chunk edges → element edges
    chunk_edges = [e for e in edges if e["type"] == "chunk"]
    projected = project_chunk_edges(
        chunk_edges=chunk_edges,
        chunk_to_elems=chunk_to_elems,
        valid_elem_pids=set(elem_doc.keys()),
        cap_per_pair=args.chunk_project_cap,
        min_similarity=args.chunk_project_min_sim,
    )
    print(f"[info] projected chunk-mediated element edges: {len(projected)}", flush=True)

    # validation analysis
    report = analyze(crossdoc, bbl, elem_doc, top100, projected)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    # save outputs
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    report_path = prefix.with_name(prefix.name + "_report.json")
    report_path.write_text(json.dumps({
        "metadata": {
            "crossdoc": str(args.crossdoc),
            "qrels": str(args.qrels),
            "ranking": str(args.ranking),
            "chunk_project_cap": args.chunk_project_cap,
            "chunk_project_min_sim": args.chunk_project_min_sim,
        },
        "source_stats": crossdoc.get("stats", {}),
        "validation": report,
    }, indent=2, ensure_ascii=False))
    print(f"[done] report -> {report_path}", flush=True)

    # build projected typed_crossdoc-format file:
    # keep element (figure/formula/table) edges + projected chunk_mediated edges
    elem_typed = [e for e in edges if e["type"] in {"figure", "formula", "table"}]
    out_edges = elem_typed + projected

    out_obj = {
        "metadata": {
            "source": str(args.crossdoc),
            "includes": ["figure", "formula", "table", "chunk_mediated"],
            "chunk_project_cap": args.chunk_project_cap,
            "chunk_project_min_sim": args.chunk_project_min_sim,
        },
        "stats": {
            "num_edges_total": len(out_edges),
            "num_edges_projected": len(projected),
            "num_edges_element_direct": len(elem_typed),
        },
        "edges": out_edges,
    }
    proj_path = prefix.with_name(prefix.name + "_projected.json")
    proj_path.write_text(json.dumps(out_obj, ensure_ascii=False))
    print(f"[done] projected edges -> {proj_path} ({len(out_edges)} edges)", flush=True)


if __name__ == "__main__":
    main()
