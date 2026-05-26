#!/usr/bin/env python3
"""Build MinerU-only graph v1 — reference LaTeX graph design, pure MinerU data.

Key differences from v0:
  - Parses content_list.json for paragraph-level text elements (~107/doc instead of 1)
  - Merges structure.json metadata (LaTeX, image_path) into content_list-derived elements
  - Builds richer edges: co-reference, caption-of, section-containment
  - Adds element quality scoring and referring_paragraphs tracking

Scope: old_53 experimental group (all hybrid_auto with full MinerU output).
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
DEFAULT_DOC_IDS_FILE = ROOT / "data/doc_lists/old_53_docs.txt"

# ---------------------------------------------------------------------------
# Regex patterns (from MultimodalRelationshipBuilder)
# ---------------------------------------------------------------------------
FIG_REF = re.compile(r"(?:Figure|Fig\.?)\s*(\d+)", re.IGNORECASE)
TABLE_REF = re.compile(r"Table\s*(\d+)", re.IGNORECASE)
EQ_REF = re.compile(r"(?:Equation|Eq(?:n)?\.?)\s*\(?(\d+)\)?", re.IGNORECASE)
SECTION_REF = re.compile(r"(?:Section|Sec\.?|§)\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
FIG_CAPTION = re.compile(r"(?:Figure|Fig\.?)\s*(\d+)\s*[:.]\s*(.*)", re.IGNORECASE)
TABLE_CAPTION = re.compile(r"Table\s*(\d+)\s*[:.]\s*(.*)", re.IGNORECASE)
EQ_LABEL = re.compile(r"\((\d+)\)\s*$")
TAG_REF = re.compile(r"\\tag\s*\{\s*(\d+)\s*\}")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, list):
        return " ".join(clean_text(v) for v in value).strip()
    if isinstance(value, dict):
        parts = [
            clean_text(value.get(k, ""))
            for k in ["text", "content", "caption", "image_caption", "table_caption"]
        ]
        return " ".join(p for p in parts if p).strip()
    return str(value).strip()


def extract_number(element_type: str, text: str, elem_id: str) -> int | None:
    haystack = f"{text} {elem_id}"
    if element_type == "figure":
        m = FIG_CAPTION.search(haystack) or FIG_REF.search(haystack)
        if m:
            return int(m.group(1))
    elif element_type == "table":
        m = TABLE_CAPTION.search(haystack) or TABLE_REF.search(haystack)
        if m:
            return int(m.group(1))
    elif element_type == "formula":
        m = TAG_REF.search(haystack) or EQ_LABEL.search(haystack) or EQ_REF.search(haystack)
        if m:
            return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_doc_ids(path: Path) -> list[str]:
    ids = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


def resolve_mineru_dir(doc_id: str, mineru_root: Path) -> Path | None:
    doc_dir = mineru_root / doc_id
    if not doc_dir.is_dir():
        return None
    inner = doc_dir / doc_id
    if not inner.is_dir():
        return None
    for mode in ("hybrid_auto", "auto"):
        mode_dir = inner / mode
        if mode_dir.is_dir():
            return mode_dir
    return None


def load_content_list(mode_dir: Path, doc_id: str) -> list[dict[str, Any]]:
    candidates = [
        mode_dir / f"{doc_id}_content_list.json",
        mode_dir / f"{doc_id}_content_list_v2.json",
        mode_dir / "content_list.json",
    ]
    for p in candidates:
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                return obj
    return []


def load_structure(doc_dir: Path) -> list[dict[str, Any]]:
    sp = doc_dir / "structure.json"
    if sp.exists():
        obj = json.loads(sp.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj.get("elements", [])
    return []


def resolve_image_path(mode_dir: Path, raw_path: str | None) -> str:
    if not raw_path:
        return ""
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return str(p)
    candidates = [mode_dir / raw_path, mode_dir / "images" / p.name]
    for c in candidates:
        if c.exists():
            return str(c)
    return raw_path


# ---------------------------------------------------------------------------
# Element building (merged from content_list + structure.json)
# ---------------------------------------------------------------------------

def normalize_type(raw: str) -> str:
    raw = (raw or "").lower()
    if raw in {"image", "figure", "fig"}:
        return "figure"
    if raw in {"equation", "formula"}:
        return "formula"
    if raw in {"paragraph", "plain_text", "body_text"}:
        return "text"
    return raw


def build_elements(
    doc_id: str,
    content_items: list[dict[str, Any]],
    struct_elements: list[dict[str, Any]],
    mode_dir: Path,
) -> list[dict[str, Any]]:
    """Merge content_list reading order with structure.json metadata."""
    elements: list[dict[str, Any]] = []

    # Index structure elements by type for sequential matching
    struct_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for se in struct_elements:
        st = normalize_type(se.get("type", ""))
        struct_by_type[st].append(se)
    struct_cursors: dict[str, int] = defaultdict(int)

    for pos_idx, item in enumerate(content_items):
        item_type = str(item.get("type", "")).lower()
        bbox = item.get("bbox")
        page_idx = int(item.get("page_idx", 0))
        text = clean_text(item.get("text") or item.get("content"))

        if item_type == "text":
            text_level = item.get("text_level")
            elem_type = "section" if (text_level is not None and text_level <= 2) else "text"
            if not text or len(text) < 10:
                continue
            elements.append({
                "element_id": f"{doc_id}_{elem_type}_{len(elements)}",
                "doc_id": doc_id,
                "element_type": elem_type,
                "number": None,
                "label": text[:80] if elem_type == "section" else "",
                "caption": "",
                "content": text,
                "image_path": "",
                "page_idx": page_idx,
                "position_idx": pos_idx,
                "bbox": bbox,
                "source": "mineru.content_list",
                "metadata": {"text_level": text_level},
            })

        elif item_type in ("image", "figure", "table"):
            elem_type = "table" if item_type == "table" else "figure"
            cursor = struct_cursors[elem_type]
            struct_match = None
            if cursor < len(struct_by_type.get(elem_type, [])):
                struct_match = struct_by_type[elem_type][cursor]
                struct_cursors[elem_type] += 1

            # Image path: prefer structure.json, fallback to content_list img_path
            raw_img = item.get("img_path") or item.get("image_path")
            if struct_match:
                raw_img = struct_match.get("image_path") or raw_img
            image_path = ""
            if raw_img:
                image_path = resolve_image_path(mode_dir, raw_img)

            # Caption: content_list nested fields first, then structure metadata
            caption = ""
            cap_field = "table_caption" if elem_type == "table" else "image_caption"
            raw_cap = item.get(cap_field)
            if isinstance(raw_cap, list) and raw_cap:
                caption = clean_text(raw_cap)
            elif raw_cap:
                caption = clean_text(raw_cap)
            if not caption and struct_match:
                smd = struct_match.get("metadata", {}) if isinstance(struct_match.get("metadata"), dict) else {}
                caption = clean_text(smd.get("caption"))
            if not caption:
                cap_match = (TABLE_CAPTION if elem_type == "table" else FIG_CAPTION).search(text)
                if cap_match:
                    caption = text

            # Content: table_body for tables, image description for figures
            content = text
            if elem_type == "table":
                tb = item.get("table_body")
                if tb:
                    content = clean_text(tb) if isinstance(tb, str) else str(tb)
            if not content and struct_match:
                content = clean_text(struct_match.get("content", ""))

            number = extract_number(elem_type, caption or content, "")
            if number is None and struct_match:
                number = extract_number(elem_type,
                    clean_text(struct_match.get("content", "")),
                    struct_match.get("element_id", ""))

            n_str = str(number) if number is not None else str(len(elements))
            elements.append({
                "element_id": f"{doc_id}_{elem_type}_{n_str}",
                "doc_id": doc_id,
                "element_type": elem_type,
                "number": number,
                "label": f"{'Table' if elem_type == 'table' else 'Figure'} {number}" if number is not None else "",
                "caption": caption,
                "content": content,
                "image_path": image_path,
                "page_idx": page_idx,
                "position_idx": pos_idx,
                "bbox": bbox,
                "source": "mineru.content_list+structure",
                "metadata": {
                    "struct_element_id": struct_match.get("element_id") if struct_match else None,
                    "struct_metadata": struct_match.get("metadata") if struct_match else {},
                },
            })

        elif item_type == "equation":
            cursor = struct_cursors["formula"]
            struct_match = None
            if cursor < len(struct_by_type.get("formula", [])):
                struct_match = struct_by_type["formula"][cursor]
                struct_cursors["formula"] += 1

            content = text
            latex_meta = {}
            if struct_match:
                smd = struct_match.get("metadata", {}) if isinstance(struct_match.get("metadata"), dict) else {}
                latex_meta = smd
                struct_content = clean_text(struct_match.get("content", ""))
                if struct_content and len(struct_content) > len(content):
                    content = struct_content

            number = extract_number("formula", content, "")
            if number is None and struct_match:
                number = extract_number("formula",
                    clean_text(struct_match.get("content", "")),
                    struct_match.get("element_id", ""))

            n_str = str(number) if number is not None else str(len(elements))
            elements.append({
                "element_id": f"{doc_id}_formula_{n_str}",
                "doc_id": doc_id,
                "element_type": "formula",
                "number": number,
                "label": f"Eq. {number}" if number is not None else "",
                "caption": "",
                "content": content,
                "image_path": "",
                "page_idx": page_idx,
                "position_idx": pos_idx,
                "bbox": bbox,
                "source": "mineru.content_list+structure",
                "metadata": {"latex": latex_meta.get("latex", ""),
                             "struct_element_id": struct_match.get("element_id") if struct_match else None},
            })

        elif item_type in ("page_footnote", "aside_text", "list", "header", "footer"):
            if text and len(text) >= 10:
                elements.append({
                    "element_id": f"{doc_id}_text_{len(elements)}",
                    "doc_id": doc_id,
                    "element_type": "text",
                    "number": None,
                    "label": "",
                    "caption": "",
                    "content": text,
                    "image_path": "",
                    "page_idx": page_idx,
                    "position_idx": pos_idx,
                    "bbox": bbox,
                    "source": "mineru.content_list",
                    "metadata": {"original_type": item_type},
                })

    return elements


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------

def add_context(elements: list[dict[str, Any]], window: int = 3) -> None:
    for idx, elem in enumerate(elements):
        before: list[str] = []
        after: list[str] = []
        for j in range(idx - 1, max(-1, idx - 1 - window * 2), -1):
            if j < 0:
                break
            other = elements[j]
            if other.get("page_idx") != elem.get("page_idx") and before:
                break
            ot = other.get("element_type", "")
            if ot in ("text", "section"):
                t = other.get("content", "")
                if len(t) >= 20:
                    before.append(t)
            elif ot == "section":
                t = other.get("content", "")
                if t:
                    before.append(f"[{t}]")
            if len(before) >= window:
                break
        for j in range(idx + 1, min(len(elements), idx + 1 + window * 2)):
            other = elements[j]
            if other.get("page_idx") != elem.get("page_idx") and after:
                break
            ot = other.get("element_type", "")
            if ot in ("text", "section"):
                t = other.get("content", "")
                if len(t) >= 20:
                    after.append(t)
            elif ot == "section":
                t = other.get("content", "")
                if t:
                    after.append(f"[{t}]")
            if len(after) >= window:
                break
        before.reverse()
        elem["context_before"] = "\n\n".join(before)[:1500]
        elem["context_after"] = "\n\n".join(after)[:1500]


# ---------------------------------------------------------------------------
# Referring paragraphs
# ---------------------------------------------------------------------------

def add_referring_paragraphs(elements: list[dict[str, Any]]) -> None:
    text_elems = [e for e in elements if e["element_type"] == "text"]
    for elem in elements:
        if elem["number"] is None:
            elem["referring_paragraphs"] = []
            continue
        etype = elem["element_type"]
        if etype == "figure":
            pat = re.compile(rf"\bFig(?:ure|\.)\s*{elem['number']}\b", re.IGNORECASE)
        elif etype == "table":
            pat = re.compile(rf"\bTable\s*{elem['number']}\b", re.IGNORECASE)
        elif etype == "formula":
            pat = re.compile(rf"(?:Eq(?:uation|n)?\.?\s*\(?{elem['number']}\)?|\({elem['number']}\))", re.IGNORECASE)
        else:
            elem["referring_paragraphs"] = []
            continue
        refs = [t["content"][:500] for t in text_elems if pat.search(t.get("content", ""))]
        elem["referring_paragraphs"] = refs[:10]


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def score_quality(elem: dict[str, Any]) -> float:
    score = 0.0
    if elem.get("caption"):
        score += 0.25
    if elem.get("context_before"):
        score += 0.15
    if elem.get("context_after"):
        score += 0.15
    if elem.get("referring_paragraphs"):
        score += min(0.25, len(elem["referring_paragraphs"]) * 0.08)
    if elem.get("number") is not None:
        score += 0.1
    if elem["element_type"] == "formula" and len(elem.get("content", "")) > 30:
        score += 0.1
    if elem["element_type"] == "figure" and elem.get("image_path"):
        score += 0.1
    return min(1.0, score)


# ---------------------------------------------------------------------------
# Edge building
# ---------------------------------------------------------------------------

def make_edge(src: str, tgt: str, etype: str, weight: float, **meta) -> dict[str, Any]:
    return {"source_id": src, "target_id": tgt, "edge_type": etype, "weight": round(weight, 4), "metadata": meta}


def build_edges(elements: list[dict[str, Any]], same_page_window: int = 5) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(edge: dict[str, Any]) -> None:
        key = (edge["source_id"], edge["target_id"], edge["edge_type"])
        if edge["source_id"] == edge["target_id"] or key in seen:
            return
        seen.add(key)
        edges.append(edge)

    # 1. Reading order edges
    sorted_elems = sorted(elements, key=lambda x: (x.get("page_idx", 0), x.get("position_idx", 0)))
    for left, right in zip(sorted_elems, sorted_elems[1:]):
        if left.get("page_idx") == right.get("page_idx"):
            add(make_edge(left["element_id"], right["element_id"], "next_element", 0.4, source="reading_order"))
            add(make_edge(right["element_id"], left["element_id"], "prev_element", 0.4, source="reading_order"))

    # 2. Regex reference edges
    by_number: dict[tuple[str, int], str] = {}
    for e in elements:
        n = e.get("number")
        if isinstance(n, int):
            by_number[(e["element_type"], n)] = e["element_id"]

    ref_patterns = [
        (FIG_REF, "figure"), (TABLE_REF, "table"),
        (EQ_REF, "formula"), (SECTION_REF, "section"),
    ]
    for src in elements:
        text = " ".join(str(src.get(k) or "") for k in ["content", "caption", "context_before", "context_after"])
        if not text:
            continue
        for pattern, target_type in ref_patterns:
            for m in pattern.finditer(text):
                try:
                    num = int(m.group(1).split(".")[0])
                except (ValueError, IndexError):
                    continue
                tgt = by_number.get((target_type, num))
                if tgt:
                    add(make_edge(src["element_id"], tgt, "regex_reference", 0.8, ref_text=m.group(0)))

    # 3. Co-reference edges (two elements mentioned in same paragraph)
    for src in elements:
        if src["element_type"] != "text":
            continue
        text = src.get("content", "")
        refs_found: list[tuple[str, int]] = []
        for pattern, target_type in ref_patterns:
            for m in pattern.finditer(text):
                try:
                    num = int(m.group(1).split(".")[0])
                except (ValueError, IndexError):
                    continue
                tgt = by_number.get((target_type, num))
                if tgt:
                    refs_found.append((tgt, num))
        # Remove duplicates preserving order
        seen_ids = set()
        unique_refs = []
        for rid, rnum in refs_found:
            if rid not in seen_ids:
                seen_ids.add(rid)
                unique_refs.append((rid, rnum))
        for i in range(len(unique_refs)):
            for j in range(i + 1, len(unique_refs)):
                add(make_edge(unique_refs[i][0], unique_refs[j][0], "co_reference", 0.6,
                              paragraph_id=src["element_id"]))
                add(make_edge(unique_refs[j][0], unique_refs[i][0], "co_reference", 0.6,
                              paragraph_id=src["element_id"]))

    # 4. Caption-of edges (nearby text that looks like a caption → figure/table)
    for idx, elem in enumerate(elements):
        if elem["element_type"] not in ("figure", "table"):
            continue
        if elem.get("caption"):
            continue  # already has caption
        for j in (idx - 1, idx + 1):
            if 0 <= j < len(elements):
                neighbor = elements[j]
                nt = neighbor.get("content", "")
                if neighbor["element_type"] == "text" and len(nt) < 300:
                    cap_pat = TABLE_CAPTION if elem["element_type"] == "table" else FIG_CAPTION
                    if cap_pat.search(nt):
                        elem["caption"] = nt
                        add(make_edge(neighbor["element_id"], elem["element_id"], "caption_of", 0.9))

    # 5. Same-page cross-type window edges
    by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for e in elements:
        by_page[int(e.get("page_idx", 0))].append(e)
    for page_elems in by_page.values():
        page_elems.sort(key=lambda x: x.get("position_idx", 0))
        for idx_a, a in enumerate(page_elems):
            for idx_b in range(max(0, idx_a - same_page_window), min(len(page_elems), idx_a + same_page_window + 1)):
                b = page_elems[idx_b]
                if a["element_id"] == b["element_id"]:
                    continue
                if a["element_type"] == b["element_type"]:
                    continue
                dist = abs(idx_a - idx_b)
                if dist == 0 or dist > same_page_window:
                    continue
                w = max(0.1, 0.6 - dist * 0.08)
                add(make_edge(a["element_id"], b["element_id"], "same_page_cross_type", w, position_distance=dist))

    # 6. Section containment edges (section → elements that follow until next section)
    current_section: str | None = None
    for e in elements:
        if e["element_type"] == "section":
            current_section = e["element_id"]
        elif current_section and e["element_type"] != "section":
            add(make_edge(current_section, e["element_id"], "section_contains", 0.5))

    return edges


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_doc(doc_id: str, mineru_root: Path, context_window: int, same_page_window: int) -> dict[str, Any] | None:
    doc_dir = mineru_root / doc_id
    mode_dir = resolve_mineru_dir(doc_id, mineru_root)
    if not mode_dir:
        return None

    content_items = load_content_list(mode_dir, doc_id)
    if not content_items:
        return None

    struct_elements = load_structure(doc_dir)
    # Inject mode_dir into struct elements for image path resolution
    for se in struct_elements:
        if isinstance(se, dict):
            se["_mode_dir"] = str(mode_dir)

    elements = build_elements(doc_id, content_items, struct_elements, mode_dir)
    if not elements:
        return None

    add_context(elements, context_window)
    add_referring_paragraphs(elements)
    for e in elements:
        e["quality_score"] = round(score_quality(e), 4)

    edges = build_edges(elements, same_page_window)

    return {
        "doc_id": doc_id,
        "num_elements": len(elements),
        "num_edges": len(edges),
        "mode": mode_dir.parent.name if mode_dir.parent != mode_dir else mode_dir.name,
        "elements": {e["element_id"]: e for e in elements},
        "edges": edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MinerU-only graph v1")
    parser.add_argument("--mineru-dir", default=str(DEFAULT_MINERU_DIR))
    parser.add_argument("--doc-ids-file", default=str(DEFAULT_DOC_IDS_FILE))
    parser.add_argument("--doc-ids", nargs="*", default=None)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--context-window", type=int, default=3)
    parser.add_argument("--same-page-window", type=int, default=5)
    args = parser.parse_args()

    mineru_root = Path(args.mineru_dir)
    if args.doc_ids:
        doc_ids = args.doc_ids
    else:
        doc_ids = load_doc_ids(Path(args.doc_ids_file))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / f"data/05_eval/mineru_only_graph_v1_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    docs: dict[str, Any] = {}
    all_edges: list[dict[str, Any]] = []
    skipped = 0
    type_counts: Counter[str] = Counter()
    edge_counts: Counter[str] = Counter()

    for doc_id in doc_ids:
        result = process_doc(doc_id, mineru_root, args.context_window, args.same_page_window)
        if not result:
            skipped += 1
            print(f"  [SKIP] {doc_id}")
            continue
        docs[doc_id] = result
        for e in result["elements"].values():
            type_counts[e["element_type"]] += 1
        for edge in result["edges"]:
            edge["doc_id"] = doc_id
            edge_counts[edge["edge_type"]] += 1
        all_edges.extend(result["edges"])
        print(f"  [{doc_id}] elements={result['num_elements']} edges={result['num_edges']} mode={result['mode']}")

    summary = {
        "builder": "mineru_only_graph_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "docs_processed": len(docs),
        "docs_skipped": skipped,
        "total_elements": sum(type_counts.values()),
        "element_type_counts": dict(type_counts),
        "total_edges": len(all_edges),
        "edge_type_counts": dict(edge_counts),
    }

    # Write outputs
    graph = {"metadata": {"builder": "mineru_only_graph_v1", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, "documents": docs}
    (out_dir / "mineru_elements_v1.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "mineru_edges_v1.jsonl").open("w", encoding="utf-8") as f:
        for item in all_edges:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Report
    lines = [
        "# MinerU-only Graph v1",
        "",
        f"- docs processed: **{len(docs)}** / skipped: {skipped}",
        f"- total elements: **{summary['total_elements']}** `{dict(type_counts)}`",
        f"- total edges: **{summary['total_edges']}** `{dict(edge_counts)}`",
        "",
        "## Edge types",
    ]
    for etype, cnt in edge_counts.most_common():
        lines.append(f"- `{etype}`: {cnt}")
    lines.extend([
        "",
        "## Notes",
        "- Pure MinerU data: content_list.json (reading order + text) merged with structure.json (LaTeX metadata)",
        "- Text paragraphs are first-class elements (~100/doc instead of ~1 in v0)",
        "- Referencing LaTeX graph design: regex_reference, co_reference, caption_of, section_contains",
    ])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Symlink latest
    latest = ROOT / "data/05_eval/mineru_only_graph_v1_latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_dir.resolve())
    except OSError:
        pass

    print(f"\n[ok] wrote {out_dir / 'report.md'}")
    print(f"  elements: {out_dir / 'mineru_elements_v1.json'}")
    print(f"  edges: {out_dir / 'mineru_edges_v1.jsonl'}")


if __name__ == "__main__":
    main()
