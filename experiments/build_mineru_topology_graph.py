#!/usr/bin/env python3
"""Build a unified topology graph from MinerU-only v1 artifacts.

This is the Phase 1 adapter in the pure-MinerU pipeline.  It keeps all work in
the experimental lane: v1 element dictionaries and edge JSONL records are
converted into stable node IDs, typed edge records, and adjacency lists that can
feed hub scoring, VL edge augmentation, and query generation.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V1_DIR = ROOT / "data/05_eval/mineru_only_graph_v1_latest"

NODE_TYPE_MAP = {
    "text": "paragraph",
    "section": "section",
    "figure": "figure",
    "table": "table",
    "formula": "formula",
}

BACKBONE_EDGE_TYPES = {"next_element", "prev_element"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compact_text(value: Any, limit: int = 500) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    return text[:limit]


def make_label(element: dict[str, Any], node_type: str) -> str:
    label = compact_text(element.get("label"), 120)
    if label:
        return label
    if node_type in {"figure", "table", "formula"}:
        prefix = {"figure": "Figure", "table": "Table", "formula": "Eq."}[node_type]
        number = element.get("number")
        if number is not None:
            return f"{prefix} {number}"
    caption = compact_text(element.get("caption"), 120)
    if caption:
        return caption
    return compact_text(element.get("content"), 120)


def node_metadata(element: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(element.get("metadata") or {})
    for key in (
        "element_id",
        "element_type",
        "number",
        "caption",
        "image_path",
        "bbox",
        "source",
        "quality_score",
        "context_before",
        "context_after",
        "referring_paragraphs",
    ):
        if key in element:
            metadata[key] = element.get(key)
    content = str(element.get("content") or "")
    metadata["content_length"] = len(content)
    if element.get("element_type") not in {"text", "section"}:
        metadata["content_preview"] = compact_text(content, 1000)
    return metadata


def build_nodes(v1_graph: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    element_to_node: dict[str, str] = {}
    doc_stats: dict[str, Any] = {}

    for doc_id, doc in sorted((v1_graph.get("documents") or {}).items()):
        counters: Counter[str] = Counter()
        elements = list((doc.get("elements") or {}).values())
        elements.sort(key=lambda e: (
            int(e.get("page_idx") or 0),
            int(e.get("position_idx") or 0),
            str(e.get("element_id") or ""),
        ))
        per_type: Counter[str] = Counter()
        for element in elements:
            v1_type = str(element.get("element_type") or "")
            node_type = NODE_TYPE_MAP.get(v1_type, v1_type or "unknown")
            idx = counters[node_type]
            counters[node_type] += 1
            node_id = f"{doc_id}::{node_type}::{idx:05d}"
            element_id = str(element.get("element_id") or "")
            element_to_node[element_id] = node_id
            per_type[node_type] += 1

            content = element.get("content") or ""
            section_level = None
            if isinstance(element.get("metadata"), dict):
                section_level = element["metadata"].get("text_level")
            node = {
                "node_id": node_id,
                "doc_id": doc_id,
                "node_type": node_type,
                "label": make_label(element, node_type),
                "text_snippet": compact_text(content, 500) if node_type in {"paragraph", "section", "formula"} else compact_text(element.get("caption") or content, 500),
                "page_idx": element.get("page_idx"),
                "position_idx": element.get("position_idx"),
                "element_id": element_id,
                "mapped_element_id": element_id,
                "section_level": section_level,
                "section_title": compact_text(content, 200) if node_type == "section" else None,
                "metadata": node_metadata(element),
            }
            nodes.append(node)

        doc_stats[doc_id] = {
            "nodes": sum(per_type.values()),
            "node_type_counts": dict(per_type),
            "v1_edges": int(doc.get("num_edges") or 0),
        }
    return nodes, element_to_node, doc_stats


def map_edge_type(
    edge: dict[str, Any],
    source_node: dict[str, Any],
    target_node: dict[str, Any],
) -> str:
    original = str(edge.get("edge_type") or "")
    if original in BACKBONE_EDGE_TYPES:
        return "backbone"
    if original in {"regex_reference", "co_reference"}:
        return "element_ref"
    if original == "section_contains":
        if target_node.get("node_type") == "paragraph":
            return "section_contains_paragraph"
        return "section_contains_element"
    return original


def build_edges(
    raw_edges: list[dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
    element_to_node: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, list[str]], dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    out_adj: dict[str, set[str]] = defaultdict(set)
    in_adj: dict[str, set[str]] = defaultdict(set)
    skipped: Counter[str] = Counter()

    for raw in raw_edges:
        src = element_to_node.get(str(raw.get("source_id") or ""))
        tgt = element_to_node.get(str(raw.get("target_id") or ""))
        if not src or not tgt:
            skipped["missing_endpoint"] += 1
            continue
        source_node = nodes_by_id[src]
        target_node = nodes_by_id[tgt]
        mapped_type = map_edge_type(raw, source_node, target_node)
        key = (src, tgt, mapped_type)
        if src == tgt:
            skipped["self_loop"] += 1
            continue
        if key in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(key)

        meta = dict(raw.get("metadata") or {})
        meta["original_edge_type"] = raw.get("edge_type")
        meta["original_source_id"] = raw.get("source_id")
        meta["original_target_id"] = raw.get("target_id")
        edge = {
            "source_id": src,
            "target_id": tgt,
            "doc_id": raw.get("doc_id") or source_node.get("doc_id"),
            "edge_type": mapped_type,
            "weight": float(raw.get("weight", 1.0)),
            "metadata": meta,
        }
        edges.append(edge)
        out_adj[src].add(tgt)
        in_adj[tgt].add(src)

    for node_id in nodes_by_id:
        out_adj.setdefault(node_id, set())
        in_adj.setdefault(node_id, set())

    return (
        edges,
        {k: sorted(v) for k, v in out_adj.items()},
        {k: sorted(v) for k, v in in_adj.items()},
        dict(skipped),
    )


def backbone_reachability(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    by_doc: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        if node.get("node_type") == "paragraph":
            by_doc[str(node.get("doc_id"))].append(node["node_id"])

    backbone: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge.get("edge_type") != "backbone":
            continue
        src, tgt = edge["source_id"], edge["target_id"]
        backbone[src].add(tgt)
        backbone[tgt].add(src)

    component_counts: dict[str, int] = {}
    largest_components: dict[str, int] = {}
    for doc_id, para_ids in by_doc.items():
        remaining = set(para_ids)
        sizes: list[int] = []
        while remaining:
            start = remaining.pop()
            q: deque[str] = deque([start])
            size = 1
            while q:
                cur = q.popleft()
                for nb in backbone.get(cur, set()):
                    if nb in remaining:
                        remaining.remove(nb)
                        q.append(nb)
                        size += 1
            sizes.append(size)
        component_counts[doc_id] = len(sizes)
        largest_components[doc_id] = max(sizes) if sizes else 0
    docs_single_component = sum(1 for v in component_counts.values() if v <= 1)
    return {
        "docs_with_paragraphs": len(by_doc),
        "docs_single_backbone_component": docs_single_component,
        "component_counts": component_counts,
        "largest_components": largest_components,
    }


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MinerU Topology Graph v1",
        "",
        f"- source: `{summary['source_v1_dir']}`",
        f"- docs processed: **{summary['docs_processed']}**",
        f"- nodes: **{summary['total_nodes']}** `{summary['node_type_counts']}`",
        f"- edges: **{summary['total_edges']}** `{summary['edge_type_counts']}`",
        f"- skipped edges: `{summary['skipped_edges']}`",
        "",
        "## Backbone",
        f"- docs with paragraph nodes: **{summary['backbone_reachability']['docs_with_paragraphs']}**",
        f"- docs with one paragraph backbone component: **{summary['backbone_reachability']['docs_single_backbone_component']}**",
        "",
        "## Notes",
        "- v1 `text` elements are topology `paragraph` nodes.",
        "- `next_element` and `prev_element` are normalized to `backbone`.",
        "- `regex_reference` and `co_reference` are normalized to `element_ref` with original type preserved in metadata.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_latest_symlink(out_dir: Path) -> None:
    latest = ROOT / "data/05_eval/mineru_topology_graph_v1_latest"
    try:
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        if not latest.exists():
            latest.symlink_to(out_dir.resolve())
    except OSError as exc:
        print(f"[warn] could not update latest symlink: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MinerU topology graph v1")
    parser.add_argument("--v1-dir", default=str(DEFAULT_V1_DIR))
    parser.add_argument("--elements", default="")
    parser.add_argument("--edges", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    v1_dir = Path(args.v1_dir)
    elements_path = Path(args.elements) if args.elements else v1_dir / "mineru_elements_v1.json"
    edges_path = Path(args.edges) if args.edges else v1_dir / "mineru_edges_v1.jsonl"
    if not elements_path.exists():
        raise FileNotFoundError(elements_path)
    if not edges_path.exists():
        raise FileNotFoundError(edges_path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / f"data/05_eval/mineru_topology_graph_v1_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    v1_graph = read_json(elements_path)
    raw_edges = iter_jsonl(edges_path)
    nodes, element_to_node, doc_stats = build_nodes(v1_graph)
    nodes_by_id = {node["node_id"]: node for node in nodes}
    edges, out_adj, in_adj, skipped_edges = build_edges(raw_edges, nodes_by_id, element_to_node)

    for edge in edges:
        doc_id = str(edge.get("doc_id") or "")
        if doc_id in doc_stats:
            doc_stats[doc_id]["edges"] = int(doc_stats[doc_id].get("edges", 0)) + 1

    summary = {
        "builder": "mineru_topology_graph_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_v1_dir": str(v1_dir),
        "docs_processed": len(doc_stats),
        "total_nodes": len(nodes),
        "node_type_counts": dict(Counter(node["node_type"] for node in nodes)),
        "total_edges": len(edges),
        "edge_type_counts": dict(Counter(edge["edge_type"] for edge in edges)),
        "skipped_edges": skipped_edges,
        "backbone_reachability": backbone_reachability(nodes, edges),
        "doc_stats": doc_stats,
    }
    graph = {
        "metadata": {
            "builder": "mineru_topology_graph_v1",
            "created_at": summary["created_at"],
            "source_elements": str(elements_path),
            "source_edges": str(edges_path),
        },
        "nodes": nodes,
        "edges": edges,
        "adjacency": {
            "out_adj": out_adj,
            "in_adj": in_adj,
        },
        "element_to_node": element_to_node,
    }

    (out_dir / "mineru_topology_graph_v1.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir, summary)
    update_latest_symlink(out_dir)

    print(f"[ok] wrote {out_dir / 'mineru_topology_graph_v1.json'}")
    print(f"nodes={summary['total_nodes']} edges={summary['total_edges']} skipped={skipped_edges}")


if __name__ == "__main__":
    main()
