"""Shared data models used across graph-building and evaluation scripts.

Extracted from analyze_latex_graph_topology.py (Node, Edge) and
run_phase0_eval_ab.py (Chunk) to allow reuse without copy-paste.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


# ── Graph nodes ──────────────────────────────────────────────────────────────

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


# ── Graph edges ──────────────────────────────────────────────────────────────

@dataclass
class Edge:
    source_id: str
    target_id: str
    doc_id: str
    edge_type: str  # paragraph_ref | element_ref | backbone | cross_doc_cite | section_contains_*
    weight: float = 1.0  # edge strength: reference count, semantic relevance, etc.

    def key(self) -> Tuple[str, str, str]:
        return (self.source_id, self.target_id, self.edge_type)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "doc_id": self.doc_id,
            "edge_type": self.edge_type,
        }
        if self.weight != 1.0:
            d["weight"] = round(self.weight, 4)
        return d


# ── Retrieval chunks ─────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    caption: str = ""
    content: str = ""
    context: str = ""
    enriched_title: str = ""
    enriched_content: str = ""
