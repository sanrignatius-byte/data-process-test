"""Shared data models used across scripts.

Node / Edge     — graph topology (analyze_latex_graph_topology, run_phase0_eval_ab)
Chunk           — retrieval unit  (run_phase0_eval_ab, eval_cpool_keyword_boost_graph)
StandardQuery   — unified query schema for L1/L2/L3 (training pipeline)
Triplet         — contrastive learning triplet
EvidenceSpan    — single piece of evidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    """A node in the document reference graph."""

    node_id: str
    doc_id: str
    node_type: str  # "figure", "table", "equation", "paragraph", "section", ...
    label: str = ""
    mapped_element_id: Optional[str] = None
    page_idx: Optional[int] = None
    position_idx: Optional[int] = None
    line_no: Optional[int] = None
    line_no_end: Optional[int] = None
    source_file: Optional[str] = None
    text_snippet: Optional[str] = None
    section_level: Optional[int] = None
    section_title: Optional[str] = None
    paragraph_order: Optional[int] = None


@dataclass
class Edge:
    """A directed edge in the document reference graph."""

    source_id: str
    target_id: str
    doc_id: str = ""
    edge_type: str = ""  # "paragraph_ref", "backbone", "element_ref", ...
    weight: float = 1.0

    def key(self) -> tuple:
        """Deduplication key: (source_id, target_id, edge_type)."""
        return (self.source_id, self.target_id, self.edge_type)


@dataclass
class Chunk:
    """A retrieval unit (one per multimodal element)."""

    chunk_id: str
    doc_id: str
    text: str
    caption: str = ""
    content: str = ""
    context: str = ""
    enriched_title: str = ""
    enriched_content: str = ""
