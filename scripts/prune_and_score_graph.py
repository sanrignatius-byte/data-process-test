#!/usr/bin/env python3
"""图清洗 + hub 打分：方案C 第一步。

做两件事：
  1. 清洗：删掉没有 multimodal elements 匹配的文档、删掉两端 label 都无法映射到
     multimodal element 的 ref edges（保留至少一端能映射的边）
  2. Hub 打分：在清洗后的图上用 analyze_latex_graph_topology.py 的 hub scoring
     逻辑给每个节点打分，选 top 20% 作为高质量 hub

输入：
  - data/01_graphs/latex_reference_graph_v2.json
  - data/01_graphs/multimodal_elements_v2.json
  - data/00_raw/mineru_output/

输出：
  - data/01_graphs/pruned_graph_v2.json       — 清洗后的图（文档级）
  - data/01_graphs/hub_scores_v2.json         — 所有节点 hub 分数 + top 20%
  - data/01_graphs/prune_report_v2.json       — 清洗统计报告

用法:
    cd /projects/myyyx1/data-process-test
    python3 scripts/prune_and_score_graph.py

    # 自定义 top 比例
    python3 scripts/prune_and_score_graph.py --top-ratio 0.20

    # 只看报告不写图
    python3 scripts/prune_and_score_graph.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import variance
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

# ── Reuse analyze_latex_graph_topology logic ──────────────────────────────────
# Import the heavy topology builder instead of reimplementing
sys.path.insert(0, str(Path(__file__).parent))

from analyze_latex_graph_topology import (
    Node,
    Edge,
    ELEMENT_MODALITIES,
    build_multimodal_index,
    build_topology_graph,
    compute_hubs,
    compute_bridge_hubs,
    _build_global_keyword_boost_map,
)


# ── Phase 1: Prune ───────────────────────────────────────────────────────────

def prune_graph(
    latex_data: Dict[str, Any],
    mm_data: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """清洗 LaTeX reference graph：
    1. 删掉没有 multimodal elements 的文档
    2. 对保留的文档，删掉两端 label 都不是 element modality 的 edges
       （保留至少一端是 fig/tab/eq 类型的 label）

    Returns:
        pruned_latex_data: 清洗后的 latex graph (同结构)
        report: 清洗统计
    """
    rg_docs = latex_data.get("documents", {})
    me_docs = mm_data.get("documents", {})

    both_ids = set(rg_docs.keys()) & set(me_docs.keys())

    stats = {
        "original_docs": len(rg_docs),
        "multimodal_docs": len(me_docs),
        "overlap_docs": len(both_ids),
        "removed_docs_no_mineru": len(set(rg_docs.keys()) - both_ids),
    }

    # Element-type label prefixes (used to detect if a label represents a
    # figure/table/equation even when label_type metadata is missing)
    _ELEM_PREFIXES = (
        "fig:", "f:", "figure:",
        "tab:", "t:", "tbl:", "table:",
        "eq:", "eqn:", "equation:", "formula:",
    )

    def _is_element_label(label_key: str, label_info: Dict[str, Any]) -> bool:
        """Check if a label corresponds to an element modality."""
        ltype = str(label_info.get("label_type", "")).lower()
        if ltype in {"figure", "table", "equation", "formula"}:
            return True
        lk = label_key.lower()
        return any(lk.startswith(p) for p in _ELEM_PREFIXES)

    pruned_docs = {}
    total_labels_before = 0
    total_labels_after = 0
    total_edges_before = 0
    total_edges_after = 0
    docs_removed_empty_edges = 0

    for doc_id in sorted(both_ids):
        rg_doc = rg_docs[doc_id]
        labels = rg_doc.get("labels", {})
        edges = rg_doc.get("edges", [])

        total_labels_before += len(labels)
        total_edges_before += len(edges)

        # Build set of element-type labels in this doc
        element_labels = set()
        for lk, li in labels.items():
            if _is_element_label(lk, li):
                element_labels.add(lk)

        # Keep only edges where at least one endpoint is an element label
        kept_edges = []
        for edge in edges:
            src = edge.get("source_label", "")
            tgt = edge.get("target_label", "")
            if src in element_labels or tgt in element_labels:
                kept_edges.append(edge)

        # Collect labels that appear in kept edges
        used_labels = set()
        for edge in kept_edges:
            used_labels.add(edge.get("source_label", ""))
            used_labels.add(edge.get("target_label", ""))
        # Also keep all element labels (even if no edges reference them)
        used_labels |= element_labels

        # Prune labels not used by any kept edge and not element-type
        kept_labels = {lk: li for lk, li in labels.items()
                       if lk in used_labels or lk in element_labels}

        total_labels_after += len(kept_labels)
        total_edges_after += len(kept_edges)

        if len(kept_edges) == 0:
            docs_removed_empty_edges += 1
            # Still keep the doc if it has element labels (for future
            # enrichment), but flag it
            if not element_labels:
                continue

        pruned_doc = dict(rg_doc)  # shallow copy
        pruned_doc["labels"] = kept_labels
        pruned_doc["edges"] = kept_edges
        pruned_doc["num_labels"] = len(kept_labels)
        pruned_doc["num_edges"] = len(kept_edges)
        pruned_docs[doc_id] = pruned_doc

    stats.update({
        "pruned_docs": len(pruned_docs),
        "docs_with_zero_edges_after_prune": docs_removed_empty_edges,
        "labels_before": total_labels_before,
        "labels_after": total_labels_after,
        "labels_removed": total_labels_before - total_labels_after,
        "edges_before": total_edges_before,
        "edges_after": total_edges_after,
        "edges_removed": total_edges_before - total_edges_after,
        "edge_retention_rate": round(total_edges_after / max(total_edges_before, 1) * 100, 1),
    })

    pruned_latex = {
        "metadata": {
            **latex_data.get("metadata", {}),
            "pruned": True,
            "prune_stats": stats,
        },
        "documents": pruned_docs,
    }

    return pruned_latex, stats


# ── Phase 2: Hub scoring on pruned graph ──────────────────────────────────────

def score_hubs_on_pruned(
    pruned_latex: Dict[str, Any],
    mm_data: Dict[str, Any],
    mineru_output_dir: str,
    top_ratio: float = 0.20,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """在清洗后的图上用 analyze_latex_graph_topology 的打分逻辑算 hub scores。

    Returns:
        all_hubs:    所有 hub 节点（按 hub_score 排序）
        top_hubs:    top_ratio 的高分 hub
        score_stats: 打分统计
    """
    print("  Building multimodal index...")
    mm_index = build_multimodal_index(mm_data, mineru_output_dir=mineru_output_dir)
    print(f"    Real page coverage: {mm_index['real_page_coverage']}")

    print("  Building topology graph on pruned data...")
    nodes, edges, out_adj, in_adj, doc_stats = build_topology_graph(
        pruned_latex, mm_index
    )
    print(f"    Nodes: {len(nodes)}, Edges: {len(edges)}")

    # Node type distribution
    type_dist = Counter(n.node_type for n in nodes.values())
    print(f"    Node types: {dict(type_dist)}")

    # Element mapping stats
    element_nodes = [n for n in nodes.values() if n.node_type in ELEMENT_MODALITIES]
    mapped = sum(1 for n in element_nodes if n.mapped_element_id is not None)
    print(f"    Element nodes: {len(element_nodes)}, mapped: {mapped} "
          f"({100*mapped/max(len(element_nodes),1):.1f}%)")

    # Adaptive degree normalization
    all_degrees = sorted(
        len(out_adj.get(nid, ())) + len(in_adj.get(nid, ()))
        for nid in nodes
    )
    if all_degrees:
        p90_idx = max(0, int(0.90 * (len(all_degrees) - 1)))
        degree_norm = float(max(all_degrees[p90_idx], 4))
    else:
        degree_norm = 12.0
    print(f"    Degree norm (P90): {degree_norm}")

    # Build keyword boost maps
    keyword_boost_map, keyword_source_map = _build_global_keyword_boost_map(nodes, in_adj)

    # Compute ALL hub scores (top_k=0 means no limit)
    print("  Computing hub scores for all nodes...")
    all_hubs = compute_hubs(
        nodes, out_adj, in_adj,
        top_k=0,
        degree_norm=degree_norm,
        keyword_boost_map=keyword_boost_map,
        keyword_source_map=keyword_source_map,
    )
    print(f"    Total scored nodes: {len(all_hubs)}")

    # Compute bridge hubs (subset)
    # NOTE: compute_bridge_hubs uses bridges[:top_k], so top_k=0 returns empty.
    # Pass a huge number to get all.
    bridge_hubs = compute_bridge_hubs(
        nodes, out_adj, in_adj,
        top_k=999999,
        keyword_boost_map=keyword_boost_map,
        keyword_source_map=keyword_source_map,
    )
    print(f"    Bridge hubs: {len(bridge_hubs)}")

    # Select top ratio
    n_top = max(1, int(len(all_hubs) * top_ratio))
    top_hubs = all_hubs[:n_top]
    print(f"    Top {top_ratio*100:.0f}% hubs: {n_top}")

    # Hub statistics
    if all_hubs:
        scores = [h["hub_score"] for h in all_hubs]
        top_scores = [h["hub_score"] for h in top_hubs]
        score_stats = {
            "total_scored": len(all_hubs),
            "bridge_hubs": len(bridge_hubs),
            "top_ratio": top_ratio,
            "top_count": n_top,
            "score_min": round(min(scores), 4),
            "score_max": round(max(scores), 4),
            "score_mean": round(sum(scores) / len(scores), 4),
            "top_score_min": round(min(top_scores), 4),
            "top_score_max": round(max(top_scores), 4),
            "top_score_mean": round(sum(top_scores) / len(top_scores), 4),
            "top_hub_category_dist": dict(Counter(h["hub_category"] for h in top_hubs)),
            "top_node_type_dist": dict(Counter(h["node_type"] for h in top_hubs)),
            "top_docs_covered": len(set(h["doc_id"] for h in top_hubs)),
            "total_docs_in_graph": len(set(n.doc_id for n in nodes.values())),
            "degree_norm_p90": degree_norm,
            "element_nodes_total": len(element_nodes),
            "element_nodes_mapped": mapped,
        }
    else:
        score_stats = {"total_scored": 0}

    return all_hubs, top_hubs, score_stats


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prune graph + hub scoring for Method C")
    ap.add_argument("--latex-graph",
                    default="data/01_graphs/latex_reference_graph_v2.json")
    ap.add_argument("--elements",
                    default="data/01_graphs/multimodal_elements_v2.json")
    ap.add_argument("--mineru-output",
                    default="data/00_raw/mineru_output")
    ap.add_argument("--output-pruned",
                    default="data/01_graphs/pruned_graph_v2.json")
    ap.add_argument("--output-hubs",
                    default="data/01_graphs/hub_scores_v2.json")
    ap.add_argument("--output-report",
                    default="data/01_graphs/prune_report_v2.json")
    ap.add_argument("--top-ratio", type=float, default=0.20,
                    help="Fraction of hubs to keep as 'top' (default: 0.20)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print stats only, don't write files")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    print("=" * 60)
    print("=== Prune & Score Graph — Method C Step 1 ===")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    print("\n[1/3] Loading data...")
    latex_path = Path(args.latex_graph)
    elements_path = Path(args.elements)
    for p in (latex_path, elements_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    latex_data = json.loads(latex_path.read_text(encoding="utf-8"))
    mm_data = json.loads(elements_path.read_text(encoding="utf-8"))
    print(f"  LaTeX docs: {len(latex_data.get('documents', {}))}")
    print(f"  Multimodal docs: {len(mm_data.get('documents', {}))}")

    # ── Phase 1: Prune ────────────────────────────────────────────────────
    print("\n[2/3] Pruning graph...")
    pruned_latex, prune_stats = prune_graph(latex_data, mm_data)

    print(f"  Docs: {prune_stats['original_docs']} → {prune_stats['pruned_docs']}"
          f" (removed {prune_stats['removed_docs_no_mineru']} no-mineru)")
    print(f"  Labels: {prune_stats['labels_before']} → {prune_stats['labels_after']}"
          f" (removed {prune_stats['labels_removed']})")
    print(f"  Edges: {prune_stats['edges_before']} → {prune_stats['edges_after']}"
          f" (removed {prune_stats['edges_removed']}, "
          f"retention {prune_stats['edge_retention_rate']}%)")
    print(f"  Docs with 0 edges after prune: {prune_stats['docs_with_zero_edges_after_prune']}")

    # ── Phase 2: Hub scoring ──────────────────────────────────────────────
    print(f"\n[3/3] Hub scoring (top {args.top_ratio*100:.0f}%)...")
    all_hubs, top_hubs, score_stats = score_hubs_on_pruned(
        pruned_latex, mm_data,
        mineru_output_dir=args.mineru_output,
        top_ratio=args.top_ratio,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print("=== SUMMARY ===")
    print(f"  Pruned graph: {prune_stats['pruned_docs']} docs, "
          f"{prune_stats['edges_after']} edges")
    print(f"  Hub scores: {score_stats.get('total_scored', 0)} nodes scored")
    print(f"  Bridge hubs: {score_stats.get('bridge_hubs', 0)}")
    print(f"  Top {args.top_ratio*100:.0f}%: {score_stats.get('top_count', 0)} hubs "
          f"covering {score_stats.get('top_docs_covered', 0)} docs")
    if score_stats.get('top_hub_category_dist'):
        print(f"  Top hub categories: {score_stats['top_hub_category_dist']}")
    if score_stats.get('top_node_type_dist'):
        print(f"  Top node types: {score_stats['top_node_type_dist']}")
    print(f"  Score range (top): [{score_stats.get('top_score_min', 0):.2f}, "
          f"{score_stats.get('top_score_max', 0):.2f}], "
          f"mean={score_stats.get('top_score_mean', 0):.2f}")
    print(f"  Elapsed: {elapsed:.1f}s")

    # ── Write outputs ─────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    out_pruned = Path(args.output_pruned)
    out_hubs = Path(args.output_hubs)
    out_report = Path(args.output_report)

    for p in (out_pruned, out_hubs, out_report):
        p.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting pruned graph → {out_pruned}")
    with open(out_pruned, "w", encoding="utf-8") as f:
        json.dump(pruned_latex, f, ensure_ascii=False)
    print(f"  Size: {out_pruned.stat().st_size / 1e6:.1f} MB")

    print(f"Writing hub scores → {out_hubs}")
    hub_output = {
        "metadata": {
            "top_ratio": args.top_ratio,
            "top_count": len(top_hubs),
            "total_scored": len(all_hubs),
            "score_stats": score_stats,
        },
        "all_hubs": all_hubs,
        "top_hubs": top_hubs,
    }
    with open(out_hubs, "w", encoding="utf-8") as f:
        json.dump(hub_output, f, ensure_ascii=False, indent=1)
    print(f"  Size: {out_hubs.stat().st_size / 1e6:.1f} MB")

    print(f"Writing report → {out_report}")
    report = {
        "prune_stats": prune_stats,
        "score_stats": score_stats,
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nDone! 喵")


if __name__ == "__main__":
    main()
