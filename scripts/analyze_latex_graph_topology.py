#!/usr/bin/env python3
"""Analyze LaTeX reference graph topology and hub-centric multi-hop seeds.

Algorithm v2 – key improvements over v1:
  1. Backbone edges: paragraphs sorted by line_no within doc → sequential
     para[i]→para[i+1] "backbone" edges represent natural reading order.
  2. position_idx from MinerU (true reading-order proxy, replaces all-zero page_idx).
  3. Adjacent backbone bridge detection: consecutive backbone paragraphs each
     referencing a different modality form a "bridge span" candidate.
  4. Cross-document citation edges from citation_graph.json (123 edges) allow
     multi-hop paths that traverse at most 1 citation boundary.
  5. Improved label matching: number+type AND caption Jaccard AND fallback label suffix.
  6. Traffic-hub scoring: modality diversity × backbone depth × cross-doc reach.

Outputs:
  1) Topology report  (density + hub metrics + position-gap variance)
  2) Hub list
  3) Hub-started multi-hop candidate paths
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple


LABEL_TO_MODALITY = {
    "figure": "figure",
    "table": "table",
    "equation": "equation",
    "formula": "equation",
}
ELEMENT_MODALITIES = {"figure", "table", "equation"}
MINERU_ELEMENT_TYPES = {"figure", "table", "formula"}

ARCHIVE_TIME = datetime.now(timezone.utc).isoformat()

# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Node:
    node_id: str
    doc_id: str
    node_type: str          # section | subsection | subsubsection | paragraph | figure | table | equation
    label: str
    mapped_element_id: Optional[str] = None
    page_idx: Optional[int] = None
    position_idx: Optional[int] = None   # MinerU reading-order index (better proxy)
    line_no: Optional[int] = None        # LaTeX source line (backbone ordering)
    line_no_end: Optional[int] = None    # range end (sections)
    section_level: Optional[int] = None
    section_title: Optional[str] = None
    source_file: Optional[str] = None
    paragraph_order: Optional[int] = None  # sequential index within doc backbone
    text_snippet: Optional[str] = None     # first 200 chars of paragraph text (for hub scoring)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "doc_id": self.doc_id,
            "node_type": self.node_type,
            "label": self.label,
            "mapped_element_id": self.mapped_element_id,
            "page_idx": self.page_idx,
            "position_idx": self.position_idx,
            "line_no": self.line_no,
            "line_no_end": self.line_no_end,
            "section_level": self.section_level,
            "section_title": self.section_title,
            "source_file": self.source_file,
            "paragraph_order": self.paragraph_order,
            "text_snippet": self.text_snippet,
        }


@dataclass
class Edge:
    source_id: str
    target_id: str
    doc_id: str
    edge_type: str  # paragraph_ref | element_ref | backbone | cross_doc_cite | section_contains_*

    def key(self) -> Tuple[str, str, str]:
        return (self.source_id, self.target_id, self.edge_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "doc_id": self.doc_id,
            "edge_type": self.edge_type,
        }


# ─── Utilities ────────────────────────────────────────────────────────────────

def normalize_label_type(label_type: str) -> Optional[str]:
    if not label_type:
        return None
    return LABEL_TO_MODALITY.get(str(label_type).lower())


def parse_number(text: str) -> Optional[int]:
    """Extract trailing or first numeric value from a string."""
    if not text:
        return None
    m = re.search(r"(\d+)\s*$", text)
    if not m:
        m = re.search(r"(\d+)", text)
    if not m:
        return None
    return int(m.group(1))


def tokenize(text: str) -> Set[str]:
    return {t for t in re.findall(r"[a-zA-Z]{3,}", (text or "").lower())}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def variance(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def summarize_distribution(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "variance": 0.0,
                "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0}
    arr = sorted(values)
    p50 = arr[int(round(0.50 * (len(arr) - 1)))]
    p90 = arr[int(round(0.90 * (len(arr) - 1)))]
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 4),
        "variance": round(variance(values), 4),
        "min": round(arr[0], 4),
        "max": round(arr[-1], 4),
        "p50": round(p50, 4),
        "p90": round(p90, 4),
    }


def _find_para_for_line(
    file_path: str,
    line_no: int,
    para_meta: List[Dict[str, Any]],
) -> Optional[str]:
    """Return the paragraph p_key (``'<file>:<start>'``) whose line range
    contains ``(file_path, line_no)``, preferring the tightest span.

    Returns ``None`` when no paragraph covers the ref site (e.g. the
    paragraph metadata is from an older graph without this field).
    """
    best_key: Optional[str] = None
    best_span: float = float("inf")
    for para in para_meta:
        pfile = str(para.get("file_path") or "")
        # Accept if files match or if either side is unknown (empty string)
        if pfile and file_path and pfile != file_path:
            continue
        start = int(para.get("line_no_start") or 0)
        end = int(para.get("line_no_end") or start)
        if start <= line_no <= end:
            span = end - start
            if span < best_span:
                best_span = span
                best_key = f"{pfile}:{start}"
    return best_key


# ─── MinerU index ─────────────────────────────────────────────────────────────

_CL_TYPE_TO_ETYPE = {"image": "figure", "table": "table", "equation": "formula"}
_ETYPE_TO_CL_TYPE = {"figure": "image", "table": "table", "formula": "equation"}


def build_real_page_index(mm_data: Dict[str, Any], mineru_output_dir: str) -> Dict[str, int]:
    """Extract true page_idx for each element by matching against content_list.json.

    MinerU's multimodal_elements.json stores page_idx=0 for all elements (parser bug).
    content_list.json has correct page_idx per item. We match using sequential type ordering:
    the N-th figure/table/formula in content_list corresponds to element_id *_figure_N etc.

    Returns:
        {element_id: real_page_idx}  — coverage ~95% where content_list.json exists
    """
    base = Path(mineru_output_dir)
    page_map: Dict[str, int] = {}

    for doc_id, doc in (mm_data.get("documents", {}) or {}).items():
        elements = doc.get("elements", {}) or {}
        cl_path = base / doc_id / doc_id / "hybrid_auto" / f"{doc_id}_content_list.json"
        if not cl_path.exists():
            continue
        try:
            cl: List[Dict[str, Any]] = json.loads(cl_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Group content_list items by type (image/table/equation), preserving doc order
        cl_by_type: Dict[str, List[Dict[str, Any]]] = {"image": [], "table": [], "equation": []}
        for item in cl:
            t = str(item.get("type", ""))
            if t in cl_by_type:
                cl_by_type[t].append(item)

        # Group our element_ids by type, sorted by position_idx (sequential)
        elems_by_etype: Dict[str, List[Tuple[int, str]]] = {
            "figure": [], "table": [], "formula": []
        }
        for eid, elem in elements.items():
            etype = str(elem.get("element_type", "")).lower()
            if etype not in elems_by_etype:
                continue
            pos = elem.get("position_idx") or 0
            # Bug fix: position_idx is 0 for all elements (parser bug documented in CLAUDE.md).
            # Fall back to the figure/table/formula sequential number as a reliable
            # document-order proxy when position_idx is missing or zero.
            num = elem.get("number")
            sort_key = pos if pos > 0 else (int(num) if num is not None else 99999)
            elems_by_etype[etype].append((sort_key, eid))
        for etype in elems_by_etype:
            elems_by_etype[etype].sort()

        # Sequential N-th match
        for etype, sorted_elems in elems_by_etype.items():
            cl_type = _ETYPE_TO_CL_TYPE[etype]
            cl_items = cl_by_type[cl_type]
            for i, (_, eid) in enumerate(sorted_elems):
                if i < len(cl_items):
                    page = cl_items[i].get("page_idx")
                    if isinstance(page, int):
                        page_map[eid] = page

    return page_map


def build_multimodal_index(mm_data: Dict[str, Any], mineru_output_dir: str = "") -> Dict[str, Any]:
    """Build lookup structures from MinerU element data.

    Returns:
        by_doc:             {doc_id: {"by_number": {type: {n: eid}}, "by_caption": {type: [(eid, tok)]}}}
        page_by_element:    {eid: page_idx}      — real page from content_list.json (94.8% coverage)
        position_by_element: {eid: position_idx} — reading-order sequential index
        elements_by_doc:    {doc_id: [eid, ...]} sorted by position_idx
    """
    by_doc: Dict[str, Dict[str, Any]] = {}
    page_by_element: Dict[str, int] = {}
    position_by_element: Dict[str, int] = {}
    elements_by_doc: Dict[str, List[str]] = defaultdict(list)

    # Load real page_idx from content_list.json (overrides the all-zero page_idx in mm_data)
    real_page_map: Dict[str, int] = {}
    if mineru_output_dir:
        real_page_map = build_real_page_index(mm_data, mineru_output_dir)

    for doc_id, doc in (mm_data.get("documents", {}) or {}).items():
        idx_num: Dict[str, Dict[int, str]] = defaultdict(dict)
        idx_caption: Dict[str, List[Tuple[str, Set[str]]]] = defaultdict(list)
        elements = doc.get("elements", {}) or {}

        elem_list: List[Tuple[int, str]] = []  # (position_idx, eid)
        for eid, elem in elements.items():
            etype = str(elem.get("element_type", "")).lower()
            if etype not in MINERU_ELEMENT_TYPES:
                continue

            # Real page from content_list (primary), fallback to mm_data page_idx
            real_page = real_page_map.get(eid)
            if real_page is not None:
                page_by_element[eid] = real_page
            else:
                page = elem.get("page_idx")
                if isinstance(page, int):
                    page_by_element[eid] = page

            pos = elem.get("position_idx")
            if isinstance(pos, int):
                position_by_element[eid] = pos
                elem_list.append((pos, eid))
            elif eid in page_by_element:
                elem_list.append((page_by_element[eid] * 1000, eid))

            num = elem.get("number")
            if isinstance(num, int):
                idx_num[etype][num] = eid
            id_num = parse_number(eid.rsplit("_", 1)[-1])
            if id_num is not None and id_num not in idx_num[etype]:
                idx_num[etype][id_num] = eid

            caption_tokens = tokenize(str(elem.get("caption", "")))
            if caption_tokens:
                idx_caption[etype].append((eid, caption_tokens))

        by_doc[doc_id] = {"by_number": idx_num, "by_caption": idx_caption}
        elem_list.sort()
        elements_by_doc[doc_id] = [eid for _, eid in elem_list]

    return {
        "by_doc": by_doc,
        "page_by_element": page_by_element,
        "position_by_element": position_by_element,
        "elements_by_doc": dict(elements_by_doc),
        "real_page_coverage": len(real_page_map),
    }


def map_label_to_element(
    doc_id: str,
    label_key: str,
    label_info: Dict[str, Any],
    mm_index: Dict[str, Any],
) -> Optional[str]:
    """Map a LaTeX label to a MinerU element_id.

    Strategy (in order):
    1. number extracted from label_key matched against type+number index
    2. number from label_key suffix (e.g. "fig:arch" → try suffix parts)
    3. caption Jaccard >= 0.25 (relaxed from 0.35)
    """
    doc_idx = mm_index["by_doc"].get(doc_id)
    if not doc_idx:
        return None

    ltype = normalize_label_type(str(label_info.get("label_type", "")))
    if ltype not in ELEMENT_MODALITIES:
        return None
    mm_type = "formula" if ltype == "equation" else ltype

    # 1. number from label key
    number = parse_number(label_key)
    if number is not None:
        eid = doc_idx["by_number"].get(mm_type, {}).get(number)
        if eid:
            return eid
        # Try each part of the label key
        for part in re.split(r"[_\-:]+", label_key):
            n = parse_number(part)
            if n is not None:
                eid = doc_idx["by_number"].get(mm_type, {}).get(n)
                if eid:
                    return eid

    # 2. caption Jaccard (relaxed threshold)
    caption_tokens = tokenize(str(label_info.get("caption", "")))
    if caption_tokens:
        best_id = None
        best_score = 0.0
        for eid, cap_tokens in doc_idx["by_caption"].get(mm_type, []):
            s = jaccard(caption_tokens, cap_tokens)
            if s > best_score:
                best_score = s
                best_id = eid
        if best_score >= 0.25:
            return best_id

    return None


# ─── Graph construction ───────────────────────────────────────────────────────

def build_topology_graph(
    latex_data: Dict[str, Any],
    mm_index: Dict[str, Any],
) -> Tuple[Dict[str, Node], List[Edge], Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Dict[str, int]]]:
    """Build intra-document reference topology with backbone edges.

    Backbone edges connect consecutive paragraph nodes (sorted by line_no)
    representing natural reading order — mentor's core insight.
    """
    nodes: Dict[str, Node] = {}
    edges: List[Edge] = []
    out_adj: Dict[str, Set[str]] = defaultdict(set)
    in_adj: Dict[str, Set[str]] = defaultdict(set)
    doc_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "nodes": 0, "edges": 0, "paragraph_nodes": 0,
        "element_nodes": 0, "backbone_edges": 0,
    })
    edge_seen: Set[Tuple[str, str, str]] = set()
    docs = latex_data.get("documents", {}) or {}
    position_by_element = mm_index["position_by_element"]

    def add_edge(e: Edge) -> None:
        k = e.key()
        if k in edge_seen:
            return
        edge_seen.add(k)
        edges.append(e)
        out_adj[e.source_id].add(e.target_id)
        in_adj[e.target_id].add(e.source_id)

    for doc_id, doc in docs.items():
        label_node_map: Dict[str, str] = {}
        labels = doc.get("labels", {}) or {}
        sections = doc.get("metadata", {}).get("sections", []) or []

        # ── Section / subsection nodes ──────────────────────────────────────
        section_nodes: List[Node] = []
        for i, sec in enumerate(sections, start=1):
            command = str(sec.get("command", "section")).lower()
            node_type = command if command in {"section", "subsection", "subsubsection"} else "section"
            source_file = sec.get("file_path")
            source_name = Path(str(source_file)).name if source_file else "unknown"
            start = sec.get("line_no_start")
            end = sec.get("line_no_end")
            start_line = int(start) if isinstance(start, int) else None
            end_line = int(end) if isinstance(end, int) else None
            title_str = str(sec.get("title", "") or "")
            title_short = title_str[:40].rstrip()
            sec_node = Node(
                node_id=f"{doc_id}::sec::{i:04d}",
                doc_id=doc_id,
                node_type=node_type,
                # Include the section title so label is human-readable and
                # contributes to core_module keyword matching in hub scoring.
                label=f"{title_short} [{source_name}:{start_line or 0}]",
                line_no=start_line,
                line_no_end=end_line,
                section_level=sec.get("level"),
                section_title=title_str,
                source_file=str(source_file) if source_file else None,
            )
            nodes[sec_node.node_id] = sec_node
            section_nodes.append(sec_node)

        # ── Element nodes (figure / table / formula) ──────────────────────
        for label_key, info in labels.items():
            modal = normalize_label_type(str(info.get("label_type", "")))
            if modal not in ELEMENT_MODALITIES:
                continue
            node_id = f"{doc_id}::el::{label_key}"
            mapped_eid = map_label_to_element(doc_id, label_key, info, mm_index)
            raw_line = info.get("line_no")
            elem_line_no = int(raw_line) if isinstance(raw_line, int) else None
            pos_idx = position_by_element.get(mapped_eid) if mapped_eid else None
            # Use real page_idx from content_list.json (94.8% coverage)
            real_page = mm_index["page_by_element"].get(mapped_eid) if mapped_eid else None
            nodes[node_id] = Node(
                node_id=node_id,
                doc_id=doc_id,
                node_type=modal,
                label=label_key,
                mapped_element_id=mapped_eid,
                page_idx=real_page,
                position_idx=pos_idx,
                line_no=elem_line_no,
                source_file=info.get("file_path"),
            )
            label_node_map[label_key] = node_id

        # ── Paragraph nodes ────────────────────────────────────────────────
        # Preferred path: use paragraph blocks extracted by the LaTeX parser
        # (one node per blank-line-separated paragraph, not one per \ref site).
        # Fallback: legacy ref-site granularity for graphs built without the
        # new `paragraphs` metadata field.
        paragraph_node_map: Dict[str, str] = {}  # p_key → node_id
        para_meta: List[Dict[str, Any]] = doc.get("metadata", {}).get("paragraphs", []) or []

        if para_meta:
            # Build one node per extracted paragraph block
            for para_data in para_meta:
                pfile = str(para_data.get("file_path") or "unknown")
                start = int(para_data.get("line_no_start") or 0)
                end = int(para_data.get("line_no_end") or start)
                snippet = str(para_data.get("text_snippet") or "")
                p_key = f"{pfile}:{start}"
                if p_key not in paragraph_node_map:
                    pnode_id = f"{doc_id}::p::{len(paragraph_node_map)+1:05d}"
                    paragraph_node_map[p_key] = pnode_id
                    nodes[pnode_id] = Node(
                        node_id=pnode_id,
                        doc_id=doc_id,
                        node_type="paragraph",
                        label=f"{Path(pfile).name}:{start}",
                        line_no=start,
                        line_no_end=end,
                        source_file=pfile,
                        text_snippet=snippet if snippet else None,
                    )

            # Assign each \ref{} call-site to its containing paragraph node
            refs = doc.get("refs", []) or []
            for ref in refs:
                tgt = str(ref.get("target_key", ""))
                tgt_node = label_node_map.get(tgt)
                if not tgt_node:
                    continue
                file_path = str(ref.get("file_path", "") or "unknown")
                line_no_raw = ref.get("line_no", 0)
                line_no = int(line_no_raw) if isinstance(line_no_raw, int) else 0
                p_key = _find_para_for_line(file_path, line_no, para_meta)
                if p_key and p_key in paragraph_node_map:
                    src = paragraph_node_map[p_key]
                    add_edge(Edge(source_id=src, target_id=tgt_node,
                                  doc_id=doc_id, edge_type="paragraph_ref"))
                else:
                    # Ref site not covered by any paragraph block — create a
                    # minimal ad-hoc node so we don't silently drop the edge.
                    fallback_key = f"{file_path}:{line_no}"
                    if fallback_key not in paragraph_node_map:
                        pnode_id = f"{doc_id}::p::{len(paragraph_node_map)+1:05d}"
                        paragraph_node_map[fallback_key] = pnode_id
                        nodes[pnode_id] = Node(
                            node_id=pnode_id,
                            doc_id=doc_id,
                            node_type="paragraph",
                            label=f"{Path(file_path).name}:{line_no}",
                            line_no=line_no if line_no else None,
                            source_file=file_path,
                        )
                    src = paragraph_node_map[fallback_key]
                    add_edge(Edge(source_id=src, target_id=tgt_node,
                                  doc_id=doc_id, edge_type="paragraph_ref"))
        else:
            # Legacy fallback: one paragraph node per (file, line_no) ref site
            refs = doc.get("refs", []) or []
            for ref in refs:
                tgt = str(ref.get("target_key", ""))
                tgt_node = label_node_map.get(tgt)
                if not tgt_node:
                    continue
                file_path = str(ref.get("file_path", "") or "unknown")
                line_no_raw = ref.get("line_no", 0)
                line_no = int(line_no_raw) if isinstance(line_no_raw, int) else 0
                p_key = f"{file_path}:{line_no}"
                if p_key not in paragraph_node_map:
                    pnode_id = f"{doc_id}::p::{len(paragraph_node_map)+1:05d}"
                    paragraph_node_map[p_key] = pnode_id
                    nodes[pnode_id] = Node(
                        node_id=pnode_id,
                        doc_id=doc_id,
                        node_type="paragraph",
                        label=f"{Path(file_path).name}:{line_no}",
                        line_no=line_no if line_no else None,
                        source_file=file_path,
                    )
                src = paragraph_node_map[p_key]
                add_edge(Edge(source_id=src, target_id=tgt_node,
                              doc_id=doc_id, edge_type="paragraph_ref"))

        # ── Element→element co-reference edges ────────────────────────────
        for e in (doc.get("edges", []) or []):
            src_label = str(e.get("source_label", ""))
            tgt_label = str(e.get("target_label", ""))
            src_modal = normalize_label_type(str(e.get("source_type", "")))
            tgt_modal = normalize_label_type(str(e.get("target_type", "")))
            if src_modal not in ELEMENT_MODALITIES or tgt_modal not in ELEMENT_MODALITIES:
                continue
            src_node = label_node_map.get(src_label)
            tgt_node = label_node_map.get(tgt_label)
            if not src_node or not tgt_node:
                continue
            add_edge(Edge(source_id=src_node, target_id=tgt_node, doc_id=doc_id, edge_type="element_ref"))

        # ── BACKBONE EDGES: sequential paragraph order ─────────────────────
        # Sort paragraphs by line_no → they form the natural reading order.
        # Connect consecutive paragraphs with backbone edges.
        doc_paragraphs = sorted(
            [n for n in nodes.values() if n.doc_id == doc_id and n.node_type == "paragraph"],
            key=lambda n: (n.line_no or 0, n.node_id),
        )
        for order_idx, para_node in enumerate(doc_paragraphs):
            para_node.paragraph_order = order_idx
        bb_count = 0
        for i in range(len(doc_paragraphs) - 1):
            p_cur = doc_paragraphs[i]
            p_nxt = doc_paragraphs[i + 1]
            add_edge(Edge(
                source_id=p_cur.node_id,
                target_id=p_nxt.node_id,
                doc_id=doc_id,
                edge_type="backbone",
            ))
            bb_count += 1

        # ── Section containment edges ──────────────────────────────────────
        def find_section_for(node: Node) -> Optional[str]:
            if node.line_no is None:
                return None
            best: Optional[Node] = None
            for sec in section_nodes:
                if sec.line_no is None or sec.line_no_end is None:
                    continue
                same_file = (not node.source_file or not sec.source_file or node.source_file == sec.source_file)
                if not same_file:
                    continue
                if sec.line_no <= node.line_no <= sec.line_no_end:
                    if best is None:
                        best = sec
                    else:
                        best_span = (best.line_no_end or best.line_no) - (best.line_no or 0)
                        cur_span = (sec.line_no_end or sec.line_no) - (sec.line_no or 0)
                        if cur_span <= best_span:
                            best = sec
            return best.node_id if best else None

        for para in doc_paragraphs:
            sec_id = find_section_for(para)
            if sec_id:
                add_edge(Edge(source_id=sec_id, target_id=para.node_id, doc_id=doc_id,
                              edge_type="section_contains_paragraph"))

        for nid, node in nodes.items():
            if node.doc_id != doc_id or node.node_type not in ELEMENT_MODALITIES:
                continue
            sec_id = find_section_for(node)
            if sec_id:
                add_edge(Edge(source_id=sec_id, target_id=nid, doc_id=doc_id,
                              edge_type="section_contains_element"))

        # ── Per-doc stats ──────────────────────────────────────────────────
        doc_nodes = [n for n in nodes.values() if n.doc_id == doc_id]
        doc_stats[doc_id]["nodes"] = len(doc_nodes)
        doc_stats[doc_id]["paragraph_nodes"] = sum(1 for n in doc_nodes if n.node_type == "paragraph")
        doc_stats[doc_id]["element_nodes"] = sum(1 for n in doc_nodes if n.node_type in ELEMENT_MODALITIES)
        doc_stats[doc_id]["section_nodes"] = sum(1 for n in doc_nodes if n.node_type in {"section", "subsection", "subsubsection"})
        doc_stats[doc_id]["edges"] = sum(1 for e in edges if e.doc_id == doc_id)
        doc_stats[doc_id]["backbone_edges"] = bb_count

    for node_id in nodes:
        out_adj.setdefault(node_id, set())
        in_adj.setdefault(node_id, set())

    return nodes, edges, out_adj, in_adj, doc_stats


def build_cross_doc_edges(
    citation_graph: Dict[str, Any],
    nodes: Dict[str, Node],
    out_adj: Dict[str, Set[str]],
    in_adj: Dict[str, Set[str]],
) -> Tuple[List[Edge], Set[Tuple[str, str]]]:
    """Add cross-document citation edges between high-hub element/paragraph nodes.

    For each citation edge (src_paper → tgt_paper) in citation_graph:
    - Find the top element node (by in_degree) in tgt_paper
    - Find the top paragraph node (by out_degree) in src_paper
    - Connect them with a "cross_doc_cite" edge

    Returns:
        cross_edges: list of new Edge objects
        cite_pairs:  set of (src_doc_id, tgt_doc_id) tuples for path filtering
    """
    cg_edges = citation_graph.get("edges", []) or []

    # Group nodes by doc
    nodes_by_doc: Dict[str, List[Node]] = defaultdict(list)
    for n in nodes.values():
        nodes_by_doc[n.doc_id].append(n)

    cross_edges: List[Edge] = []
    cite_pairs: Set[Tuple[str, str]] = set()
    edge_seen: Set[Tuple[str, str, str]] = set()

    def add_edge(e: Edge) -> None:
        k = e.key()
        if k in edge_seen:
            return
        edge_seen.add(k)
        cross_edges.append(e)
        out_adj[e.source_id].add(e.target_id)
        in_adj[e.target_id].add(e.source_id)

    for ce in cg_edges:
        src_doc = str(ce.get("source", ""))
        tgt_doc = str(ce.get("target", ""))
        cite_pairs.add((src_doc, tgt_doc))

        src_nodes = nodes_by_doc.get(src_doc, [])
        tgt_nodes = nodes_by_doc.get(tgt_doc, [])
        if not src_nodes or not tgt_nodes:
            continue

        # Best source: paragraph in src_doc with highest out_degree to elements
        src_para = sorted(
            [n for n in src_nodes if n.node_type == "paragraph"],
            key=lambda n: len(out_adj.get(n.node_id, ())),
            reverse=True,
        )
        # Best target: element node in tgt_doc with highest in_degree
        tgt_elem = sorted(
            [n for n in tgt_nodes if n.node_type in ELEMENT_MODALITIES],
            key=lambda n: len(in_adj.get(n.node_id, ())),
            reverse=True,
        )
        if not src_para or not tgt_elem:
            continue

        # Connect top-2 src paragraphs to top-2 tgt elements
        for sp in src_para[:2]:
            for te in tgt_elem[:2]:
                add_edge(Edge(
                    source_id=sp.node_id,
                    target_id=te.node_id,
                    doc_id=f"{src_doc}->{tgt_doc}",
                    edge_type="cross_doc_cite",
                ))

    return cross_edges, cite_pairs


# ─── Hub detection ────────────────────────────────────────────────────────────

def pagerank(
    node_ids: List[str],
    out_adj: Dict[str, Set[str]],
    in_adj: Dict[str, Set[str]],
    damping: float = 0.85,
    max_iter: int = 40,
    tol: float = 1e-9,
) -> Dict[str, float]:
    n = len(node_ids)
    if n == 0:
        return {}
    rank = {nid: 1.0 / n for nid in node_ids}
    dangling = [nid for nid in node_ids if not out_adj.get(nid)]
    for _ in range(max_iter):
        new_rank: Dict[str, float] = {}
        dangle_term = damping * sum(rank[d] for d in dangling) / n
        diff = 0.0
        for nid in node_ids:
            r = (1.0 - damping) / n + dangle_term
            for src in in_adj.get(nid, ()):
                out_deg = len(out_adj.get(src, ()))
                if out_deg > 0:
                    r += damping * rank[src] / out_deg
            new_rank[nid] = r
            diff += abs(r - rank[nid])
        rank = new_rank
        if diff < tol:
            break
    return rank


def compute_hubs(
    nodes: Dict[str, Node],
    out_adj: Dict[str, Set[str]],
    in_adj: Dict[str, Set[str]],
    top_k: int,
    degree_norm: float = 12.0,
) -> List[Dict[str, Any]]:
    """Compute hub scores with bridge/connectivity/core-module signals.

    ``degree_norm`` is used to normalise ``edge_connectivity``; pass the 90th
    percentile of per-node total degree so the score is adaptive to each graph
    rather than relying on a hardcoded constant.
    """
    node_ids = sorted(nodes.keys())
    pr = pagerank(node_ids, out_adj, in_adj)
    hubs: List[Dict[str, Any]] = []
    for nid in node_ids:
        out_deg = len(out_adj.get(nid, ()))
        in_deg = len(in_adj.get(nid, ()))
        total = in_deg + out_deg
        if total == 0:
            continue
        node = nodes[nid]

        # Bridge score: rewarded for connecting multiple modalities as a paragraph
        out_modalities: Set[str] = set()
        out_to_elements = 0
        cross_type_edges = 0
        for nb in out_adj.get(nid, ()):
            if nodes[nb].node_type in ELEMENT_MODALITIES:
                out_modalities.add(nodes[nb].node_type)
                out_to_elements += 1
        for nb in out_adj.get(nid, ()) | in_adj.get(nid, ()):
            if nodes[nb].node_type != node.node_type:
                cross_type_edges += 1

        bridge_role = 1.0 if len(out_modalities) >= 2 else (0.5 if len(out_modalities) == 1 else 0.0)
        dn = max(degree_norm, 4.0)  # guard against degenerate sparse graphs
        edge_connectivity = min(1.0, (total / dn) + (cross_type_edges / dn))

        # Include paragraph text_snippet so keyword-based core_module scoring
        # works for paragraph nodes (not just section/figure nodes whose title
        # and label are already descriptive).
        core_text = " ".join([
            node.section_title or "",
            node.label or "",
            node.text_snippet or "",
        ]).lower()
        core_score = 0.0
        for pat, w in (
            (r"\bintroduction\b", 1.0),
            (r"\b(main\s+result|key\s+result|experiment|ablation)\b", 0.9),
            (r"\b(method|approach|framework|architecture|model)\b", 0.8),
            (r"\b(conclusion|discussion)\b", 0.6),
            (r"\b(related\s+work|background)\b", 0.3),
        ):
            if re.search(pat, core_text):
                core_score = w
                break
        if node.node_type == "figure":
            if re.search(r"\b(architecture|framework|overview|pipeline|model\s+structure)\b", core_text):
                core_score = max(core_score, 1.0)
            elif re.search(r"\b(result|performance|comparison|ablation)\b", core_text):
                core_score = max(core_score, 0.8)

        bridge_score = bridge_role * 100
        connectivity_score = edge_connectivity * 100
        core_module_score = core_score * 100
        penalty = 20.0 if (in_deg > out_deg * 2 and len(out_modalities) <= 1) else 0.0
        hub_score = (0.40 * bridge_score + 0.35 * connectivity_score +
                     0.25 * core_module_score + 20.0 * pr.get(nid, 0.0) - penalty)

        # Hub category
        # A node is a "bridge" when it structurally connects ≥2 distinct element
        # modalities (figure/table/formula).  Both paragraph nodes (via \ref{})
        # and section nodes (via section_contains_element edges) can serve this
        # bridging role.  Section nodes that contain elements from ≥2 modalities
        # should therefore be promoted to "bridge" alongside paragraphs.
        if (node.node_type in {"paragraph", "section", "subsection", "subsubsection"}
                and len(out_modalities) >= 2):
            hub_category = "bridge"
        elif node.node_type in ELEMENT_MODALITIES and in_deg > out_deg:
            hub_category = "authority"
        else:
            hub_category = "mixed"

        balance = min(in_deg, out_deg) / max(1, total)
        hubs.append({
            "node_id": nid,
            "doc_id": node.doc_id,
            "node_type": node.node_type,
            "label": node.label,
            "hub_category": hub_category,
            "mapped_element_id": node.mapped_element_id,
            "page_idx": node.page_idx,
            "position_idx": node.position_idx,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "out_to_elements": out_to_elements,
            "out_modalities": sorted(out_modalities),
            "degree_total": total,
            "degree_balance": round(balance, 4),
            "pagerank": round(pr.get(nid, 0.0), 8),
            "bridge_score": round(bridge_score, 2),
            "connectivity_score": round(connectivity_score, 2),
            "core_module_score": round(core_module_score, 2),
            "penalty": round(penalty, 2),
            "hub_score": round(hub_score, 6),
        })
    # Bridge-first sort: bridge hubs ranked before authority sinks
    hubs.sort(
        key=lambda x: (
            x["hub_category"] == "bridge",   # bridge first
            x["bridge_score"],
            x["hub_score"],
        ),
        reverse=True,
    )
    return hubs[:top_k] if top_k > 0 else hubs


def compute_bridge_hubs(
    nodes: Dict[str, Node],
    out_adj: Dict[str, Set[str]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Nodes co-referencing ≥2 distinct element modalities (true bridge hubs).

    Includes paragraph nodes (via \\ref{}) and section/subsection/subsubsection
    nodes (via section_contains_element edges).  Section-level bridges capture
    the structural importance of sections that contain multiple modality types,
    which paragraphs may miss when elements are spread across sub-sections.

    Scored by:
      bridge_score = modality_count * 10 + out_degree
    """
    _BRIDGE_NODE_TYPES = {"paragraph", "section", "subsection", "subsubsection"}
    bridges: List[Dict[str, Any]] = []
    for nid, node in nodes.items():
        if node.node_type not in _BRIDGE_NODE_TYPES:
            continue
        neighbors = out_adj.get(nid, set())
        modalities = {nodes[nb].node_type for nb in neighbors if nodes[nb].node_type in ELEMENT_MODALITIES}
        if len(modalities) < 2:
            continue
        bridges.append({
            "node_id": nid,
            "doc_id": node.doc_id,
            "node_type": node.node_type,
            "label": node.label,
            "section_title": node.section_title,
            "line_no": node.line_no,
            "paragraph_order": node.paragraph_order,
            "out_degree": len(neighbors),
            "unique_modalities_referred": len(modalities),
            "modalities": sorted(modalities),
            "bridge_score": len(modalities) * 10 + len(neighbors),
        })
    bridges.sort(key=lambda x: (x["unique_modalities_referred"], x["out_degree"]), reverse=True)
    return bridges[:top_k]


def compute_adjacent_backbone_bridges(
    nodes: Dict[str, Node],
    out_adj: Dict[str, Set[str]],
) -> List[Dict[str, Any]]:
    """Detect 'adjacent paragraph bridges': two backbone-consecutive paragraphs
    where para_i references modality A and para_j references modality B (A ≠ B).

    This is the mentor's idea: "前一段引用1张图片，后一段引用1张表格 → 让他俩成为1个桥接片段"

    Returns a list of candidate dicts ready to be used as multi-hop seeds.
    """
    adj_bridges: List[Dict[str, Any]] = []

    # Group paragraphs by doc, sorted by paragraph_order
    doc_para_map: Dict[str, List[Node]] = defaultdict(list)
    for n in nodes.values():
        if n.node_type == "paragraph":
            doc_para_map[n.doc_id].append(n)
    for paras in doc_para_map.values():
        paras.sort(key=lambda n: (n.paragraph_order or 0, n.line_no or 0, n.node_id))

    for doc_id, paras in doc_para_map.items():
        for i in range(len(paras) - 1):
            p_i = paras[i]
            p_j = paras[i + 1]

            # Make sure there's actually a backbone edge between them
            if p_j.node_id not in out_adj.get(p_i.node_id, set()):
                continue

            # Modalities referenced by each paragraph
            mods_i = {nodes[nb].node_type for nb in out_adj.get(p_i.node_id, set())
                      if nodes[nb].node_type in ELEMENT_MODALITIES}
            mods_j = {nodes[nb].node_type for nb in out_adj.get(p_j.node_id, set())
                      if nodes[nb].node_type in ELEMENT_MODALITIES}

            # Both must reference at least one element, and together cover ≥2 modalities
            cross_mods = mods_i | mods_j
            if len(mods_i) == 0 or len(mods_j) == 0:
                continue
            if cross_mods == mods_i and mods_j.issubset(mods_i):
                continue  # para_j only references same modalities as para_i
            if len(cross_mods) < 2:
                continue

            # Collect element nodes from both paragraphs
            elems_i = [nb for nb in out_adj.get(p_i.node_id, set())
                       if nodes[nb].node_type in ELEMENT_MODALITIES]
            elems_j = [nb for nb in out_adj.get(p_j.node_id, set())
                       if nodes[nb].node_type in ELEMENT_MODALITIES]

            # Line number span
            line_nos = [n for n in (p_i.line_no, p_j.line_no) if n is not None]
            for eid in elems_i + elems_j:
                ln = nodes[eid].line_no
                if ln is not None:
                    line_nos.append(ln)
            line_no_span = (max(line_nos) - min(line_nos)) if len(line_nos) >= 2 else None

            # Position span (MinerU reading order)
            pos_vals = [nodes[eid].position_idx for eid in elems_i + elems_j
                        if nodes[eid].position_idx is not None]
            position_span = (max(pos_vals) - min(pos_vals)) if len(pos_vals) >= 2 else None
            position_variance = round(variance([float(x) for x in pos_vals]), 4) if len(pos_vals) >= 2 else None

            adj_bridges.append({
                "bridge_type": "adjacent_backbone",
                "doc_id": doc_id,
                "para_i_id": p_i.node_id,
                "para_i_label": p_i.label,
                "para_i_order": p_i.paragraph_order,
                "para_j_id": p_j.node_id,
                "para_j_label": p_j.label,
                "para_j_order": p_j.paragraph_order,
                "modalities_i": sorted(mods_i),
                "modalities_j": sorted(mods_j),
                "cross_modalities": sorted(cross_mods),
                "element_ids_i": elems_i,
                "element_ids_j": elems_j,
                "line_no_span": line_no_span,
                "position_span": position_span,
                "position_variance": position_variance,
                # Build a 4-node path: elem_i -> para_i -> para_j -> elem_j
                "path_node_ids": (elems_i[:1] + [p_i.node_id, p_j.node_id] + elems_j[:1]),
            })

    adj_bridges.sort(key=lambda x: (len(x["cross_modalities"]), -(x["position_span"] or 0)), reverse=True)
    return adj_bridges


# ─── Path enumeration ─────────────────────────────────────────────────────────

def edge_density(node_count: int, edge_count: int) -> float:
    if node_count <= 1:
        return 0.0
    return edge_count / (node_count * (node_count - 1))


def gather_density_report(
    nodes: Dict[str, Node],
    edges: List[Edge],
    doc_stats: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    by_node_type = Counter(node.node_type for node in nodes.values())
    by_edge_type = Counter(e.edge_type for e in edges)
    total_nodes = len(nodes)
    total_edges = len(edges)
    per_doc: Dict[str, Any] = {}
    for doc_id, st in doc_stats.items():
        per_doc[doc_id] = {
            **st,
            "density": round(edge_density(st["nodes"], st["edges"]), 8),
        }
    return {
        "global": {
            "nodes": total_nodes,
            "edges": total_edges,
            "density": round(edge_density(total_nodes, total_edges), 8),
            "node_type_distribution": dict(by_node_type),
            "edge_type_distribution": dict(by_edge_type),
        },
        "per_doc": per_doc,
    }


def build_undirected_adj(out_adj: Dict[str, Set[str]], in_adj: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    ud: Dict[str, Set[str]] = defaultdict(set)
    for src, tgts in out_adj.items():
        for tgt in tgts:
            ud[src].add(tgt)
            ud[tgt].add(src)
    for tgt, srcs in in_adj.items():
        for src in srcs:
            ud[src].add(tgt)
            ud[tgt].add(src)
    return ud


def enumerate_candidates_from_bridge_hubs(
    nodes: Dict[str, Node],
    bridge_hubs: List[Dict[str, Any]],
    out_adj: Dict[str, Set[str]],
    in_adj: Dict[str, Set[str]],
    cite_pairs: Set[Tuple[str, str]],
    max_candidates: int,
) -> List[Dict[str, Any]]:
    """Targeted 2-3 hop enumeration from bridge hub paragraphs.

    For each bridge hub paragraph (directly references >= 2 modality types):
      1. 2-hop intra-doc: [elem_A, hub_para, elem_B] for each cross-modal pair
      2. 3-hop via backbone neighbor: [elem_A, hub_para, P_adj, elem_B]
      3. Cross-doc via citation: [elem_A_src, hub_para, elem_B_tgt]

    Per-combo cap: MAX_PER_COMBO=5 per (doc_id, modality_frozenset, is_cross_doc).
    """
    candidates: List[Dict[str, Any]] = []
    seen_signatures: Set[Tuple[str, ...]] = set()
    seen_struct_keys: Set[FrozenSet[str]] = set()   # structural dedup by element label set
    per_combo_count: Dict[Tuple[str, ...], int] = defaultdict(int)
    MAX_PER_COMBO = 5

    def path_signature(path: List[str]) -> Tuple[str, ...]:
        fwd = tuple(path)
        rev = tuple(reversed(path))
        return fwd if fwd <= rev else rev

    def sort_by_position(elem_ids: List[str]) -> List[str]:
        """Sort element IDs: real page_idx first, then position_idx, then others.
        This prioritises pairs where physical distance (page_span) can be computed.
        """
        return sorted(elem_ids, key=lambda nid: (
            nodes[nid].page_idx is None,       # prefer real page_idx (from content_list.json)
            nodes[nid].position_idx is None,   # then position_idx
            nid,
        ))

    def try_add_candidate(
        path: List[str],
        hub_nid: str,
        is_cross_doc: bool,
    ) -> bool:
        """Validate and add a candidate. Returns True if added."""
        sig = path_signature(path)
        if sig in seen_signatures:
            return False
        # Verify all node IDs exist
        for nid in path:
            if nid not in nodes:
                return False
        types = [nodes[n].node_type for n in path]
        modal_types = sorted({t for t in types if t in ELEMENT_MODALITIES})
        if len(modal_types) < 2:
            return False

        primary_doc = nodes[hub_nid].doc_id
        combo_key = (primary_doc,) + tuple(modal_types) + (str(is_cross_doc),)
        if per_combo_count[combo_key] >= MAX_PER_COMBO:
            return False

        # Structural dedup: same element label set = semantically identical path
        elem_labels = frozenset(
            nodes[nid].label for nid in path if nodes[nid].node_type in ELEMENT_MODALITIES
        )
        if elem_labels in seen_struct_keys:
            return False

        seen_signatures.add(sig)
        seen_struct_keys.add(elem_labels)
        per_combo_count[combo_key] += 1

        # Real page span (from content_list.json, primary distance metric).
        # Cross-doc paths are excluded: pages from different papers are independent
        # numbering schemes so the difference is physically meaningless.
        if is_cross_doc:
            page_span = None
            page_variance = None
        else:
            # Use only the two terminal element nodes; intermediate paragraph hubs
            # have no page_idx (LaTeX-only nodes) so the filter is redundant but
            # explicit for clarity.
            terminal_nids = [path[0], path[-1]]
            page_vals = [nodes[n].page_idx for n in terminal_nids
                         if isinstance(nodes[n].page_idx, int)]
            page_span = (max(page_vals) - min(page_vals)) if len(page_vals) >= 2 else None
            page_variance = round(variance([float(x) for x in page_vals]), 4) if len(page_vals) >= 2 else None

        # Position span (MinerU reading order, secondary)
        pos_vals = [nodes[n].position_idx for n in path if isinstance(nodes[n].position_idx, int)]
        position_span = (max(pos_vals) - min(pos_vals)) if len(pos_vals) >= 2 else None

        # Line-number span (LaTeX source proxy, 100% coverage)
        line_nos = [nodes[n].line_no for n in path if isinstance(nodes[n].line_no, int)]
        line_no_span = (max(line_nos) - min(line_nos)) if len(line_nos) >= 2 else None

        short_seed, long_seed = build_query_seeds(nodes, path)
        candidates.append({
            "doc_id": primary_doc,
            "hub_node_id": hub_nid,
            "hub_label": nodes[hub_nid].label,
            "hub_type": nodes[hub_nid].node_type,
            "hop_distance": len(path) - 1,
            "path_node_ids": path,
            "path_node_types": types,
            "modalities_in_path": modal_types,
            "is_cross_doc": is_cross_doc,
            "page_span": page_span,
            "page_variance": page_variance,
            "line_no_span": line_no_span,
            "position_span": position_span,
            "short_query_seed": short_seed,
            "long_query_seed": long_seed,
        })
        return True

    for hub in bridge_hubs:
        hub_nid = hub["node_id"]
        hub_doc = nodes[hub_nid].doc_id

        # Collect intra-doc element neighbors of this hub, grouped by modality
        # Sort each group: elements with position_idx first (improves distance coverage)
        hub_elem_neighbors: Dict[str, List[str]] = defaultdict(list)
        for nb in out_adj.get(hub_nid, set()):
            if nodes[nb].doc_id != hub_doc:
                continue  # skip cross-doc elements for intra-doc strategies
            ntype = nodes[nb].node_type
            if ntype in ELEMENT_MODALITIES:
                hub_elem_neighbors[ntype].append(nb)
        for mod in hub_elem_neighbors:
            hub_elem_neighbors[mod] = sort_by_position(hub_elem_neighbors[mod])

        modality_list = sorted(hub_elem_neighbors.keys())

        # ── Strategy 1: 2-hop intra-doc [elem_A, hub, elem_B] ────────────
        for i, mod_a in enumerate(modality_list):
            for mod_b in modality_list[i + 1:]:
                for ea in hub_elem_neighbors[mod_a]:
                    for eb in hub_elem_neighbors[mod_b]:
                        if ea == eb:
                            continue
                        path = [ea, hub_nid, eb]
                        try_add_candidate(path, hub_nid, is_cross_doc=False)
                        if len(candidates) >= max_candidates:
                            break
                    if len(candidates) >= max_candidates:
                        break
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break

        if len(candidates) >= max_candidates:
            break

        # ── Strategy 2: 3-hop via backbone neighbor [elem_A, hub, P_adj, elem_B]
        # Find backbone-adjacent paragraphs (connected via backbone edge in either direction)
        backbone_neighbors: List[str] = []
        for nb in out_adj.get(hub_nid, set()):
            if nb in nodes and nodes[nb].node_type == "paragraph" and nodes[nb].doc_id == hub_doc:
                backbone_neighbors.append(nb)
        for nb in in_adj.get(hub_nid, set()):
            if nb in nodes and nodes[nb].node_type == "paragraph" and nodes[nb].doc_id == hub_doc:
                if nb not in backbone_neighbors:
                    backbone_neighbors.append(nb)

        for p_adj in backbone_neighbors:
            adj_elem_neighbors: Dict[str, List[str]] = defaultdict(list)
            for nb in out_adj.get(p_adj, set()):
                ntype = nodes[nb].node_type
                if ntype in ELEMENT_MODALITIES:
                    adj_elem_neighbors[ntype].append(nb)
            for mod in adj_elem_neighbors:
                adj_elem_neighbors[mod] = sort_by_position(adj_elem_neighbors[mod])

            # Cross-modal pairs: elem_A from hub, elem_B from p_adj
            for mod_a in modality_list:
                for mod_b, adj_elems in adj_elem_neighbors.items():
                    if mod_a == mod_b:
                        continue
                    for ea in hub_elem_neighbors[mod_a]:
                        for eb in adj_elems:
                            if ea == eb:
                                continue
                            path = [ea, hub_nid, p_adj, eb]
                            try_add_candidate(path, hub_nid, is_cross_doc=False)
                            if len(candidates) >= max_candidates:
                                break
                        if len(candidates) >= max_candidates:
                            break
                    if len(candidates) >= max_candidates:
                        break
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break

        if len(candidates) >= max_candidates:
            break

        # ── Strategy 3: Cross-doc via citation [elem_A_src, hub, elem_B_tgt]
        # Hub paragraph may have cross_doc_cite edges to elements in other docs
        for nb in out_adj.get(hub_nid, set()):
            if nb not in nodes:
                continue
            nb_node = nodes[nb]
            if nb_node.doc_id == hub_doc:
                continue
            if nb_node.node_type not in ELEMENT_MODALITIES:
                continue
            # nb is a cross-doc element; pair it with hub's intra-doc elements
            nb_mod = nb_node.node_type
            for mod_a, elems_a in hub_elem_neighbors.items():
                if mod_a == nb_mod:
                    continue
                for ea in elems_a:
                    path = [ea, hub_nid, nb]
                    try_add_candidate(path, hub_nid, is_cross_doc=True)
                    if len(candidates) >= max_candidates:
                        break
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break

        if len(candidates) >= max_candidates:
            break

    # ── Strategy 4: Section-bridged paths [elem_A, sec_node, elem_B] ────────
    # Section nodes have section_contains_element edges to element nodes within
    # their span.  Any section that directly contains ≥2 distinct modalities
    # acts as an implicit hub connecting those elements.
    section_node_types = {"section", "subsection", "subsubsection"}
    # Collect all section nodes grouped by doc
    section_nodes_by_doc: Dict[str, List[str]] = defaultdict(list)
    for nid, node in nodes.items():
        if node.node_type in section_node_types:
            section_nodes_by_doc[node.doc_id].append(nid)

    for sec_nid in [nid for nids in section_nodes_by_doc.values() for nid in nids]:
        if len(candidates) >= max_candidates:
            break
        sec_node = nodes[sec_nid]
        sec_doc = sec_node.doc_id

        # Collect element children of this section node (section_contains_element edges)
        sec_elem_neighbors: Dict[str, List[str]] = defaultdict(list)
        for nb in out_adj.get(sec_nid, set()):
            nb_node = nodes.get(nb)
            if nb_node and nb_node.doc_id == sec_doc and nb_node.node_type in ELEMENT_MODALITIES:
                sec_elem_neighbors[nb_node.node_type].append(nb)

        if len(sec_elem_neighbors) < 2:
            continue  # section must cover ≥2 modalities to be a useful bridge

        for mod in sec_elem_neighbors:
            sec_elem_neighbors[mod] = sort_by_position(sec_elem_neighbors[mod])

        modality_list = sorted(sec_elem_neighbors.keys())
        for i, mod_a in enumerate(modality_list):
            for mod_b in modality_list[i + 1:]:
                for ea in sec_elem_neighbors[mod_a]:
                    for eb in sec_elem_neighbors[mod_b]:
                        if ea == eb:
                            continue
                        path = [ea, sec_nid, eb]
                        try_add_candidate(path, sec_nid, is_cross_doc=False)
                        if len(candidates) >= max_candidates:
                            break
                    if len(candidates) >= max_candidates:
                        break
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break

    candidates.sort(
        key=lambda x: (
            int(x["is_cross_doc"]),
            x["hop_distance"],
            len(x["modalities_in_path"]),
            len(x["path_node_ids"]),
        ),
        reverse=True,
    )
    return candidates[:max_candidates]


def convert_adj_bridges_to_candidates(
    adj_bridges: List[Dict[str, Any]],
    nodes: Dict[str, Node],
) -> List[Dict[str, Any]]:
    """Convert adjacent backbone bridge entries into candidate format."""
    candidates: List[Dict[str, Any]] = []
    seen_signatures: Set[Tuple[str, ...]] = set()
    per_combo_count: Dict[Tuple[str, ...], int] = defaultdict(int)
    MAX_PER_COMBO = 5

    for ab in adj_bridges:
        doc_id = ab["doc_id"]
        elems_i = ab.get("element_ids_i", [])
        elems_j = ab.get("element_ids_j", [])
        para_i = ab["para_i_id"]
        para_j = ab["para_j_id"]
        mods_i = set(ab.get("modalities_i", []))
        mods_j = set(ab.get("modalities_j", []))

        # Generate cross-modal pairs from the two paragraphs
        for ei in elems_i:
            if ei not in nodes:
                continue
            mod_ei = nodes[ei].node_type
            for ej in elems_j:
                if ej not in nodes:
                    continue
                mod_ej = nodes[ej].node_type
                if mod_ei == mod_ej:
                    continue
                path = [ei, para_i, para_j, ej]
                sig_fwd = tuple(path)
                sig_rev = tuple(reversed(path))
                sig = sig_fwd if sig_fwd <= sig_rev else sig_rev
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)

                modal_types = sorted({mod_ei, mod_ej})
                combo_key = (doc_id,) + tuple(modal_types) + ("False",)
                if per_combo_count[combo_key] >= MAX_PER_COMBO:
                    continue
                per_combo_count[combo_key] += 1

                types = [nodes[n].node_type for n in path]
                page_vals = [nodes[n].page_idx for n in path if isinstance(nodes[n].page_idx, int)]
                page_span = (max(page_vals) - min(page_vals)) if len(page_vals) >= 2 else None
                page_variance = round(variance([float(x) for x in page_vals]), 4) if len(page_vals) >= 2 else None
                pos_vals = [nodes[n].position_idx for n in path if isinstance(nodes[n].position_idx, int)]
                position_span = (max(pos_vals) - min(pos_vals)) if len(pos_vals) >= 2 else None
                line_nos = [nodes[n].line_no for n in path if isinstance(nodes[n].line_no, int)]
                line_no_span = (max(line_nos) - min(line_nos)) if len(line_nos) >= 2 else None

                short_seed, long_seed = build_query_seeds(nodes, path)
                candidates.append({
                    "doc_id": doc_id,
                    "hub_node_id": para_i,
                    "hub_label": nodes[para_i].label,
                    "hub_type": "paragraph",
                    "hop_distance": len(path) - 1,
                    "path_node_ids": path,
                    "path_node_types": types,
                    "modalities_in_path": modal_types,
                    "is_cross_doc": False,
                    "page_span": page_span,
                    "page_variance": page_variance,
                    "line_no_span": line_no_span,
                    "position_span": position_span,
                    "short_query_seed": short_seed,
                    "long_query_seed": long_seed,
                    "source": "adjacent_backbone_bridge",
                })
    return candidates


_SEED_TYPES = ["WHY", "WHAT_IF", "MISMATCH", "CONDITION"]


def build_query_seeds(nodes: Dict[str, Node], path: List[str]) -> Tuple[str, str]:
    """Generate a (short, long) query seed pair with diverse sentence structures.

    Seed types rotate by path hash to ensure variety across candidates:
      WHY:       causal/explanatory framing
      WHAT_IF:   counterfactual/predictive framing
      MISMATCH:  apparent contradiction framing
      CONDITION: conditional dependency framing
    """
    start = nodes[path[0]]
    end = nodes[path[-1]]
    modalities = [nodes[n].node_type for n in path if nodes[n].node_type in ELEMENT_MODALITIES]
    modal_phrase = " and ".join(sorted(set(modalities))) if modalities else "cross-modal evidence"
    is_cross = start.doc_id != end.doc_id

    # Rotate seed type based on path hash for structural diversity
    seed_idx = hash(tuple(path)) % len(_SEED_TYPES)
    seed_type = _SEED_TYPES[seed_idx]

    a_label = start.label
    b_label = end.label
    a_doc = start.doc_id
    b_doc = end.doc_id

    if is_cross:
        if seed_type == "WHY":
            short_q = f"Why does {a_label} ({a_doc}) align with {b_label} ({b_doc})?"
            long_q = (
                f"What mechanism explains why results shown in {a_label} of {a_doc} "
                f"are consistent with the {modal_phrase} evidence in {b_label} of {b_doc}, "
                f"given that the two papers cite each other?"
            )
        elif seed_type == "WHAT_IF":
            short_q = f"If {a_label} changed, how would {b_label} in {b_doc} be affected?"
            long_q = (
                f"If the findings in {a_label} ({a_doc}) were reversed, what would that imply "
                f"for the {modal_phrase} conclusions drawn in {b_label} ({b_doc})?"
            )
        elif seed_type == "MISMATCH":
            short_q = f"What reconciles {a_label} ({a_doc}) with {b_label} ({b_doc})?"
            long_q = (
                f"Despite {a_doc} and {b_doc} citing each other, {a_label} and {b_label} "
                f"appear to reach different conclusions about {modal_phrase} — "
                f"what explains this apparent discrepancy?"
            )
        else:  # CONDITION
            short_q = f"Under what conditions does {a_label} predict {b_label} ({b_doc})?"
            long_q = (
                f"Under what experimental or theoretical conditions would the {modal_phrase} "
                f"pattern in {a_label} ({a_doc}) generalize to or predict the outcome "
                f"shown in {b_label} ({b_doc})?"
            )
    else:
        if seed_type == "WHY":
            short_q = f"Why does {a_label} show a different trend than {b_label}?"
            long_q = (
                f"What underlying mechanism causes the {modal_phrase} relationship between "
                f"{a_label} and {b_label} — specifically, why does one appear to contradict "
                f"or depend on the other?"
            )
        elif seed_type == "WHAT_IF":
            short_q = f"If {a_label} were modified, what would {b_label} predict?"
            long_q = (
                f"If the conditions captured in {a_label} changed substantially, "
                f"what does the {modal_phrase} evidence in {b_label} suggest the outcome "
                f"would be, and what intermediate steps link the two?"
            )
        elif seed_type == "MISMATCH":
            short_q = f"What explains the mismatch between {a_label} and {b_label}?"
            long_q = (
                f"The {modal_phrase} evidence in {a_label} seems inconsistent with "
                f"what {b_label} shows — what context or assumption bridges this apparent "
                f"contradiction in the paper's argument?"
            )
        else:  # CONDITION
            short_q = f"When does {a_label} determine the outcome in {b_label}?"
            long_q = (
                f"Under what specific conditions or parameter regimes does the pattern "
                f"demonstrated in {a_label} govern the {modal_phrase} result shown in {b_label}?"
            )

    return short_q, long_q


# ─── Position-distance statistics ─────────────────────────────────────────────

def compute_position_distance_stats(
    mm_index: Dict[str, Any],
    cross_modal_pairs: Dict[str, Any],
    hub_candidates: List[Dict[str, Any]],
    adj_bridges: List[Dict[str, Any]],
    nodes: Dict[str, Node],
) -> Dict[str, Any]:
    """Compute reading-order position distance statistics (replaces page_idx-based stats).

    Uses position_idx from MinerU (true physical ordering proxy).
    """
    position_by_element = mm_index["position_by_element"]

    # ── Cross-modal pair gaps ──────────────────────────────────────────────
    pair_gaps: List[float] = []
    gaps_by_pair_type: Dict[str, List[float]] = defaultdict(list)
    gaps_by_doc: Dict[str, List[float]] = defaultdict(list)

    for p in (cross_modal_pairs.get("pairs", []) or []):
        doc_id = str(p.get("doc_id", ""))
        ptype = str(p.get("pair_type", "unknown"))
        a = str(p.get("element_a_id", ""))
        b = str(p.get("element_b_id", ""))
        if not a or not b:
            continue
        pa = position_by_element.get(a)
        pb = position_by_element.get(b)
        if pa is None or pb is None:
            continue
        gap = float(abs(int(pa) - int(pb)))
        pair_gaps.append(gap)
        gaps_by_pair_type[ptype].append(gap)
        gaps_by_doc[doc_id].append(gap)

    # ── Hub candidate chain position variance ─────────────────────────────
    chain_pos_vars: List[float] = []
    for cand in hub_candidates:
        path = cand.get("path_node_ids", []) or []
        positions: List[int] = []
        for nid in path:
            n = nodes.get(nid)
            if not n:
                continue
            pos = n.position_idx
            if isinstance(pos, int):
                positions.append(pos)
        if len(positions) < 2:
            continue
        chain_pos_vars.append(variance([float(x) for x in positions]))

    # ── Adjacent bridge position spans ────────────────────────────────────
    adj_spans: List[float] = []
    for ab in adj_bridges:
        ps = ab.get("position_span")
        if ps is not None:
            adj_spans.append(float(ps))

    return {
        "pair_gap_global": summarize_distribution(pair_gaps),
        "pair_gap_by_type": {k: summarize_distribution(v) for k, v in sorted(gaps_by_pair_type.items())},
        "pair_gap_by_doc_top20": {
            doc_id: summarize_distribution(vals)
            for doc_id, vals in sorted(
                gaps_by_doc.items(),
                key=lambda x: summarize_distribution(x[1])["count"],
                reverse=True,
            )[:20]
        },
        "chain_position_variance": summarize_distribution(chain_pos_vars),
        "adjacent_bridge_position_span": summarize_distribution(adj_spans),
    }


# ─── I/O ──────────────────────────────────────────────────────────────────────

def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze LaTeX topology (v2) with backbone + cross-doc edges.")
    ap.add_argument("--latex-graph", default="data/latex_reference_graph.json")
    ap.add_argument("--elements", default="data/multimodal_elements.json")
    ap.add_argument("--cross-modal-pairs", default="data/latex_cross_modal_pairs.json")
    ap.add_argument("--citation-graph", default="data/citation_graph.json")
    ap.add_argument("--mineru-output", default="data/mineru_output",
                    help="MinerU output directory containing per-doc content_list.json files")
    ap.add_argument("--output-report", default="data/latex_graph_topology_report.json")
    ap.add_argument("--output-hubs", default="data/latex_graph_hubs.json")
    ap.add_argument("--output-candidates", default="data/latex_hub_multihop_candidates.json")
    ap.add_argument("--top-k-hubs", type=int, default=60)
    ap.add_argument("--min-hops", type=int, default=2)
    ap.add_argument("--max-hops", type=int, default=5)
    ap.add_argument("--max-candidates", type=int, default=500)
    ap.add_argument(
        "--single-doc-only",
        action="store_true",
        default=False,
        help="Exclude cross-document candidates (keep only intra-document paths).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    latex_path = Path(args.latex_graph)
    elements_path = Path(args.elements)
    pairs_path = Path(args.cross_modal_pairs)
    citation_path = Path(args.citation_graph)

    for p in (latex_path, elements_path, pairs_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    latex_data = json.loads(latex_path.read_text(encoding="utf-8"))
    mm_data = json.loads(elements_path.read_text(encoding="utf-8"))
    cross_modal_pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
    citation_graph: Dict[str, Any] = {}
    if citation_path.exists():
        citation_graph = json.loads(citation_path.read_text(encoding="utf-8"))
    else:
        print(f"[WARN] Citation graph not found at {citation_path}, skipping cross-doc edges.")

    # ── Phase 1: MinerU index (with real page_idx from content_list.json) ─
    mm_index = build_multimodal_index(mm_data, mineru_output_dir=args.mineru_output)

    # ── Phase 2: Intra-doc topology + backbone edges ───────────────────────
    nodes, edges, out_adj, in_adj, doc_stats = build_topology_graph(latex_data, mm_index)

    # ── Phase 3: Cross-doc citation edges ─────────────────────────────────
    cross_edges, cite_pairs = build_cross_doc_edges(citation_graph, nodes, out_adj, in_adj)
    edges.extend(cross_edges)

    # ── Phase 4: Hub detection ─────────────────────────────────────────────
    density = gather_density_report(nodes, edges, doc_stats)

    # Adaptive degree normalisation: use the 90th-percentile total degree so
    # edge_connectivity is calibrated to this graph's actual density rather
    # than a hardcoded constant.  Floor at 4 to avoid division issues.
    all_degrees = sorted(
        len(out_adj.get(nid, ())) + len(in_adj.get(nid, ())) for nid in nodes
    )
    if all_degrees:
        p90_idx = max(0, int(0.90 * (len(all_degrees) - 1)))
        degree_norm = float(max(all_degrees[p90_idx], 4))
    else:
        degree_norm = 12.0

    all_hubs = compute_hubs(nodes, out_adj, in_adj, top_k=0, degree_norm=degree_norm)
    hubs = all_hubs[:args.top_k_hubs]
    traffic_hubs = sorted(
        [h for h in all_hubs if h["in_degree"] > 0 and h["out_degree"] > 0],
        key=lambda x: (min(x["in_degree"], x["out_degree"]), x["degree_total"], x["hub_score"]),
        reverse=True,
    )[:args.top_k_hubs]
    bridge_hubs = compute_bridge_hubs(nodes, out_adj, top_k=args.top_k_hubs)

    # ── Phase 5: Adjacent backbone bridges ────────────────────────────────
    adj_bridges = compute_adjacent_backbone_bridges(nodes, out_adj)

    # ── Phase 6: Multi-hop candidate paths (targeted enumeration) ───────
    hub_candidates = enumerate_candidates_from_bridge_hubs(
        nodes=nodes,
        bridge_hubs=bridge_hubs,
        out_adj=out_adj,
        in_adj=in_adj,
        cite_pairs=cite_pairs,
        max_candidates=max(1, args.max_candidates),
    )

    # Convert adjacent backbone bridges directly to candidates
    adj_bridge_candidates = convert_adj_bridges_to_candidates(adj_bridges, nodes)

    # Merge: hub candidates first, then adj_bridge candidates (deduplicate by path signature)
    seen_sigs: Set[Tuple[str, ...]] = set()
    candidates: List[Dict[str, Any]] = []
    all_merged = hub_candidates + adj_bridge_candidates
    if args.single_doc_only:
        all_merged = [c for c in all_merged if not c.get("is_cross_doc", False)]
    for c in all_merged:
        path = c["path_node_ids"]
        fwd = tuple(path)
        rev = tuple(reversed(path))
        sig = fwd if fwd <= rev else rev
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            candidates.append(c)
    candidates = candidates[:max(1, args.max_candidates)]

    # ── Phase 7: Position distance statistics ─────────────────────────────
    pos_stats = compute_position_distance_stats(
        mm_index=mm_index,
        cross_modal_pairs=cross_modal_pairs,
        hub_candidates=candidates,
        adj_bridges=adj_bridges,
        nodes=nodes,
    )

    # ── Label mapping quality ──────────────────────────────────────────────
    element_nodes = [n for n in nodes.values() if n.node_type in ELEMENT_MODALITIES]
    mapped_count = sum(1 for n in element_nodes if n.mapped_element_id is not None)
    mapping_rate = round(mapped_count / max(1, len(element_nodes)), 4)

    # ── Position coverage for candidates ──────────────────────────────────
    with_page_span = sum(1 for c in candidates if c.get("page_span") is not None)
    with_pos_span = sum(1 for c in candidates if c.get("position_span") is not None)
    with_line_span = sum(1 for c in candidates if c.get("line_no_span") is not None)
    with_any_distance = sum(1 for c in candidates
                            if any(c.get(k) is not None for k in ("page_span", "position_span", "line_no_span")))
    real_page_cov = mm_index.get("real_page_coverage", 0)
    pos_coverage = {
        "total_candidates": len(candidates),
        "real_page_idx_elements_mapped": real_page_cov,
        "with_page_span": with_page_span,
        "with_position_span": with_pos_span,
        "with_line_no_span": with_line_span,
        "with_any_distance": with_any_distance,
        "coverage_page_pct": round(with_page_span / max(1, len(candidates)), 4),
        "coverage_position_pct": round(with_pos_span / max(1, len(candidates)), 4),
        "coverage_line_no_pct": round(with_line_span / max(1, len(candidates)), 4),
        "note": (
            "page_idx now from content_list.json (real pages, ~95% element coverage). "
            "position_idx is MinerU reading-order index. line_no_span is LaTeX source proxy (100% coverage)."
        ),
    }

    # ── Hub category breakdown ─────────────────────────────────────────────
    hub_cats = Counter(h.get("hub_category", "unknown") for h in hubs)

    # ── Assemble outputs ──────────────────────────────────────────────────
    report = {
        "metadata": {
            "generated_at": ARCHIVE_TIME,
            "algorithm_version": "v2",
            "inputs": {
                "latex_graph": str(latex_path),
                "elements": str(elements_path),
                "cross_modal_pairs": str(pairs_path),
                "citation_graph": str(citation_path),
            },
            "params": {
                "top_k_hubs": args.top_k_hubs,
                "min_hops": args.min_hops,
                "max_hops": args.max_hops,
                "max_candidates": args.max_candidates,
            },
        },
        "density": density,
        "label_mapping": {
            "element_nodes_total": len(element_nodes),
            "mapped_to_mineru": mapped_count,
            "mapping_rate": mapping_rate,
        },
        "cross_doc": {
            "citation_pairs": len(cite_pairs),
            "cross_doc_edges_added": len(cross_edges),
        },
        "hubs_summary": {
            "count": len(hubs),
            "hub_category_breakdown": dict(hub_cats),
            "note_scoring": (
                "hub_score = 0.40*bridge_score + 0.35*connectivity_score + "
                "0.25*core_module_score + 20*pagerank - penalty. "
                "bridge_score rewards multi-modality bridge behavior; "
                "connectivity_score captures degree + cross-type neighbors; "
                "core_module_score uses regex keywords from section/title labels. "
                "Penalty suppresses authority sinks (high in-degree with weak bridging)."
            ),
            "top10": hubs[:10],
            "traffic_hubs_count": len(traffic_hubs),
            "traffic_top10": traffic_hubs[:10],
            "bridge_hubs_count": len(bridge_hubs),
            "bridge_top10": bridge_hubs[:10],
        },
        "adjacent_bridges": {
            "count": len(adj_bridges),
            "top20": adj_bridges[:20],
        },
        "hub_multihop_summary": {
            "candidate_count": len(candidates),
            "cross_doc_candidates": sum(1 for c in candidates if c.get("is_cross_doc")),
            "top20": candidates[:20],
        },
        "physical_position_distance": pos_stats,
        "position_coverage": pos_coverage,
    }

    hubs_payload = {
        "metadata": {"generated_at": ARCHIVE_TIME, "algorithm_version": "v2", "top_k": args.top_k_hubs},
        "hubs": hubs,
        "traffic_hubs": traffic_hubs,
        "bridge_hubs": bridge_hubs,
        "adjacent_backbone_bridges": adj_bridges,
    }
    candidates_payload = {
        "metadata": {
            "generated_at": ARCHIVE_TIME,
            "algorithm_version": "v2",
            "min_hops": args.min_hops,
            "max_hops": args.max_hops,
            "max_candidates": args.max_candidates,
        },
        "candidates": candidates,
    }

    write_json(Path(args.output_report), report)
    write_json(Path(args.output_hubs), hubs_payload)
    write_json(Path(args.output_candidates), candidates_payload)

    print("=" * 72)
    print("LaTeX topology analysis v2 complete")
    print("=" * 72)
    print(f"Nodes:              {density['global']['nodes']}")
    print(f"Edges:              {density['global']['edges']}")
    print(f"  backbone:         {density['global']['edge_type_distribution'].get('backbone', 0)}")
    print(f"  paragraph_ref:    {density['global']['edge_type_distribution'].get('paragraph_ref', 0)}")
    print(f"  cross_doc_cite:   {density['global']['edge_type_distribution'].get('cross_doc_cite', 0)}")
    print(f"Label mapping:      {mapped_count}/{len(element_nodes)} ({mapping_rate:.1%})")
    print(f"Bridge hubs:        {len(bridge_hubs)}")
    print(f"Adjacent bridges:   {len(adj_bridges)}")
    print(f"Candidates total:   {len(candidates)}")
    print(f"  cross-doc:        {sum(1 for c in candidates if c.get('is_cross_doc'))}")
    print(f"Page coverage:      {with_page_span}/{len(candidates)} ({pos_coverage['coverage_page_pct']:.1%}) real page (content_list.json)")
    print(f"                    {with_pos_span}/{len(candidates)} ({pos_coverage['coverage_position_pct']:.1%}) position_idx (MinerU)")
    print(f"                    {with_line_span}/{len(candidates)} ({pos_coverage['coverage_line_no_pct']:.1%}) line_no_span (LaTeX, 100%)")
    print(f"Hub categories:     bridge={hub_cats.get('bridge',0)} authority={hub_cats.get('authority',0)} mixed={hub_cats.get('mixed',0)}")
    print(f"Report:             {args.output_report}")
    print(f"Hubs:               {args.output_hubs}")
    print(f"Candidates:         {args.output_candidates}")
    print("=" * 72)


if __name__ == "__main__":
    main()
