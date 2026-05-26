#!/usr/bin/env python3
"""Build an experimental MinerU-only graph directly from MinerU artifacts.

This is the first migration prototype away from LaTeX-centric graph construction.
It parses `structure.json` directly, treats text as a first-class element, and
creates only PDF/MinerU-grounded local edges. No production files are modified.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MINERU_DIR = ROOT / "data/00_raw/mineru_output"
MODAL_TYPES = {"figure", "table", "formula"}

FIG_REF = re.compile(r"(?:Figure|Fig\.?)\s*(\d+)", re.IGNORECASE)
TABLE_REF = re.compile(r"Table\s*(\d+)", re.IGNORECASE)
EQ_REF = re.compile(r"(?:Equation|Eq(?:n)?\.?)\s*\(?(\d+)\)?|\((\d+)\)", re.IGNORECASE)
TAG_REF = re.compile(r"\\tag\s*\{\s*(\d+)\s*\}")


def normalize_type(raw_type: str) -> str:
    raw = (raw_type or "unknown").lower()
    if raw in {"image", "fig"}:
        return "figure"
    if raw in {"paragraph", "plain_text", "body_text"}:
        return "text"
    return raw


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, list):
        return " ".join(clean_text(item) for item in value).strip()
    if isinstance(value, dict):
        parts = []
        for key in ["text", "content", "caption", "image_caption", "table_caption"]:
            if key in value:
                text = clean_text(value[key])
                if text:
                    parts.append(text)
        if parts:
            return " ".join(parts).strip()
    return str(value).strip()


def extract_number(element_type: str, content: str, metadata: dict[str, Any], element_id: str) -> int | None:
    caption = clean_text(metadata.get("caption"))
    haystack = " ".join(part for part in [caption, content, element_id] if part)
    if element_type == "figure":
        match = FIG_REF.search(haystack)
        if match:
            return int(match.group(1))
    if element_type == "table":
        match = TABLE_REF.search(haystack)
        if match:
            return int(match.group(1))
    if element_type == "formula":
        match = TAG_REF.search(haystack)
        if match:
            return int(match.group(1))
        match = EQ_REF.search(haystack)
        if match:
            value = match.group(1) or match.group(2)
            if value:
                return int(value)
    return None


def richness(element: dict[str, Any]) -> int:
    return sum(len((element.get(key) or "").strip()) for key in ["content", "caption", "context_before", "context_after"])


def resolve_image_path(mineru_dir: Path, doc_id: str, raw_path: str | None) -> str:
    if not raw_path:
        return ""
    raw = Path(raw_path)
    if raw.is_absolute() and raw.exists():
        return str(raw)
    candidates = [
        mineru_dir / raw_path,
        mineru_dir / doc_id / raw_path,
        mineru_dir / doc_id / Path(raw_path).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return raw_path


def local_context(elements: list[dict[str, Any]], idx: int, direction: str, window: int) -> str:
    parts: list[str] = []
    rng = range(idx - 1, -1, -1) if direction == "before" else range(idx + 1, len(elements))
    for other_idx in rng:
        if len(parts) >= window:
            break
        other = elements[other_idx]
        if other.get("page_idx") != elements[idx].get("page_idx") and parts:
            break
        if other.get("element_type") == "text":
            text = other.get("content", "")
            if len(text) >= 20:
                parts.append(text)
        elif other.get("element_type") == "section":
            text = other.get("content", "")
            if text:
                parts.append(f"[Section: {text}]")
    if direction == "before":
        parts.reverse()
    return "\n\n".join(parts)[:1500]


def normalize_doc(mineru_dir: Path, structure_path: Path, context_window: int) -> dict[str, Any] | None:
    try:
        obj = json.loads(structure_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    doc_id = str(obj.get("doc_id") or structure_path.parent.name)
    raw_elements = obj.get("elements", []) if isinstance(obj, dict) else []
    elements: list[dict[str, Any]] = []
    type_seen: Counter[str] = Counter()
    for position_idx, raw in enumerate(raw_elements):
        if not isinstance(raw, dict):
            continue
        element_type = normalize_type(str(raw.get("type") or raw.get("element_type") or "unknown"))
        content = clean_text(raw.get("content"))
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        caption = clean_text(metadata.get("caption"))
        if element_type == "figure" and not caption and content.startswith("[Image:"):
            content_text = caption
        else:
            content_text = content
        type_seen[element_type] += 1
        element_id = str(raw.get("element_id") or f"{doc_id}_{element_type}_{type_seen[element_type]}")
        image_path = resolve_image_path(mineru_dir, doc_id, raw.get("image_path"))
        element = {
            "element_id": element_id,
            "doc_id": doc_id,
            "element_type": element_type,
            "number": extract_number(element_type, content, metadata, element_id),
            "content": content_text,
            "caption": caption,
            "page_idx": raw.get("page_idx", 0),
            "position_idx": position_idx,
            "bbox": raw.get("bbox"),
            "image_path": image_path,
            "source": "mineru.structure",
            "metadata": metadata,
        }
        elements.append(element)

    for idx, element in enumerate(elements):
        element["context_before"] = local_context(elements, idx, "before", context_window)
        element["context_after"] = local_context(elements, idx, "after", context_window)
        element["quality"] = {
            "has_content": bool(element.get("content")),
            "has_caption": bool(element.get("caption")),
            "has_context": bool(element.get("context_before") or element.get("context_after")),
            "has_image": bool(element.get("image_path")),
            "richness": richness(element),
        }
    return {"doc_id": doc_id, "total_pages": obj.get("total_pages"), "elements": elements}


def edge(source: str, target: str, edge_type: str, weight: float, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "source_id": source,
        "target_id": target,
        "edge_type": edge_type,
        "weight": round(weight, 4),
        "metadata": metadata or {},
    }


def build_edges(elements: list[dict[str, Any]], same_page_window: int) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(item: dict[str, Any]) -> None:
        key = (item["source_id"], item["target_id"], item["edge_type"])
        if item["source_id"] == item["target_id"] or key in seen:
            return
        seen.add(key)
        edges.append(item)

    sorted_elements = sorted(elements, key=lambda item: (item.get("page_idx") or 0, item.get("position_idx") or 0))
    for left, right in zip(sorted_elements, sorted_elements[1:]):
        if left.get("page_idx") == right.get("page_idx"):
            add(edge(left["element_id"], right["element_id"], "next_element", 0.4, {"source": "reading_order"}))
            add(edge(right["element_id"], left["element_id"], "prev_element", 0.4, {"source": "reading_order"}))

    by_number: dict[tuple[str, int], str] = {}
    for element in elements:
        number = element.get("number")
        if isinstance(number, int):
            by_number[(element["element_type"], number)] = element["element_id"]

    ref_patterns = [(FIG_REF, "figure"), (TABLE_REF, "table"), (EQ_REF, "formula")]
    for source in elements:
        text = " ".join(str(source.get(key) or "") for key in ["content", "caption", "context_before", "context_after"])
        if not text:
            continue
        for pattern, target_type in ref_patterns:
            for match in pattern.finditer(text):
                value = match.group(1) or (match.group(2) if len(match.groups()) > 1 else None)
                if not value:
                    continue
                target_id = by_number.get((target_type, int(value)))
                if target_id:
                    add(edge(source["element_id"], target_id, "regex_reference", 0.8, {"ref_text": match.group(0)}))

    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for element in elements:
        by_page[int(element.get("page_idx") or 0)].append(element)
    for page_elements in by_page.values():
        page_elements.sort(key=lambda item: item.get("position_idx") or 0)
        for idx, anchor in enumerate(page_elements):
            for other in page_elements[max(0, idx - same_page_window): idx + same_page_window + 1]:
                if anchor["element_id"] == other["element_id"]:
                    continue
                distance = abs((anchor.get("position_idx") or 0) - (other.get("position_idx") or 0))
                if distance == 0 or distance > same_page_window:
                    continue
                if anchor["element_type"] != other["element_type"]:
                    weight = max(0.1, 0.6 - distance * 0.08)
                    add(edge(anchor["element_id"], other["element_id"], "same_page_cross_type_window", weight, {"position_distance": distance}))
    return edges


def build_graph(mineru_dir: Path, context_window: int, same_page_window: int) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    docs: dict[str, Any] = {}
    all_edges: list[dict[str, Any]] = []
    skipped = 0
    for structure_path in sorted(mineru_dir.glob("*/structure.json")):
        normalized = normalize_doc(mineru_dir, structure_path, context_window)
        if not normalized:
            skipped += 1
            continue
        doc_edges = build_edges(normalized["elements"], same_page_window)
        docs[normalized["doc_id"]] = {
            "doc_id": normalized["doc_id"],
            "total_pages": normalized.get("total_pages"),
            "num_elements": len(normalized["elements"]),
            "num_edges": len(doc_edges),
            "elements": {item["element_id"]: item for item in normalized["elements"]},
            "edges": doc_edges,
        }
        for item in doc_edges:
            item["doc_id"] = normalized["doc_id"]
        all_edges.extend(doc_edges)

    type_counts: Counter[str] = Counter()
    edge_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    for doc in docs.values():
        for element in doc["elements"].values():
            type_counts[element["element_type"]] += 1
            quality = element.get("quality", {})
            for key, value in quality.items():
                if isinstance(value, bool) and value:
                    field_counts[key] += 1
        for item in doc["edges"]:
            edge_counts[item["edge_type"]] += 1
    summary = {
        "source": str(mineru_dir.relative_to(ROOT)) if mineru_dir.is_relative_to(ROOT) else str(mineru_dir),
        "docs": len(docs),
        "skipped_docs": skipped,
        "elements": sum(type_counts.values()),
        "element_type_counts": dict(type_counts),
        "edges": len(all_edges),
        "edge_type_counts": dict(edge_counts),
        "quality_true_counts": dict(field_counts),
    }
    return {"metadata": {"builder": "mineru_only_graph_v0", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, "documents": docs}, all_edges, summary


def write_outputs(out_dir: Path, graph: dict[str, Any], edges: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mineru_elements_v0.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "mineru_edges_v0.jsonl").open("w", encoding="utf-8") as handle:
        for item in edges:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MinerU-only Graph v0",
        "",
        f"- docs: **{summary['docs']}**",
        f"- elements: **{summary['elements']}** `{summary['element_type_counts']}`",
        f"- edges: **{summary['edges']}** `{summary['edge_type_counts']}`",
        f"- quality true counts: `{summary['quality_true_counts']}`",
        "",
        "## Notes",
        "- Experimental output only; no production `src/` files are modified.",
        "- Directly parses MinerU `structure.json` instead of LaTeX references.",
        "- Text elements are included when MinerU exposes them, but current raw structures are text-sparse; this must be fixed before full query generation.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build experimental MinerU-only graph v0")
    parser.add_argument("--mineru-dir", default=str(DEFAULT_MINERU_DIR))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--context-window", type=int, default=3)
    parser.add_argument("--same-page-window", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else ROOT / f"data/05_eval/mineru_only_graph_v0_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    graph, edges, summary = build_graph(Path(args.mineru_dir), args.context_window, args.same_page_window)
    write_outputs(out_dir, graph, edges, summary)
    latest = ROOT / "data/05_eval/mineru_only_graph_v0_latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_dir.resolve())
    except OSError:
        pass
    print(f"[ok] wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()