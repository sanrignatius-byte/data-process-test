"""Unified heterogeneous graph over document and element nodes.

This module merges:
1) Intra-document LaTeX structure edges
2) Cross-document citation edges
3) MinerU element metadata (when available)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class NodeType(Enum):
    DOCUMENT = "document"
    ELEMENT = "element"


class EdgeType(Enum):
    CITATION = "citation"
    INTRA_REF = "intra_ref"
    CONTAINMENT = "containment"
    HAS_ELEMENT = "has_element"
    ALIGNMENT = "alignment"


@dataclass
class Attribution:
    name: str
    prior_confidence: float


ATTRIBUTIONS: Dict[str, Attribution] = {
    "enclosing_env": Attribution("enclosing_env", 0.95),
    "section_fallback": Attribution("section_fallback", 0.65),
    "containment": Attribution("containment", 0.80),
    "citation": Attribution("citation", 0.90),
    "alignment": Attribution("alignment", 0.85),
    "derived": Attribution("derived", 0.70),
}


@dataclass
class GraphNode:
    node_id: str
    node_type: NodeType
    doc_id: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    src: str
    tgt: str
    edge_type: EdgeType
    confidence: float
    attribution: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._adj_forward: Dict[str, List[GraphEdge]] = {}

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)
        self._adj_forward.setdefault(edge.src, []).append(edge)

    def _iter_out_edges(self, node_id: str) -> List[GraphEdge]:
        """Return outgoing edges for traversal."""
        return self._adj_forward.get(node_id, [])

    @staticmethod
    def _edge_confidence(attribution: str, extra_conf: Optional[float] = None) -> float:
        base = ATTRIBUTIONS.get(attribution, ATTRIBUTIONS["derived"]).prior_confidence
        if extra_conf is None:
            return base
        return max(0.0, min(1.0, 0.5 * base + 0.5 * extra_conf))

    @staticmethod
    def compute_path_score(edge_confidences: List[float], decay_factor: float = 0.85) -> float:
        """Markov-style joint probability with multiplicative length decay."""
        if not edge_confidences:
            return 0.0
        eps = 1e-5
        joint_prob = math.prod(max(c, eps) for c in edge_confidences)
        decay = decay_factor ** max(0, len(edge_confidences) - 1)
        return round(joint_prob * decay, 6)

    @staticmethod
    def _caption_tokens(text: str) -> Set[str]:
        if not text:
            return set()
        t = text.lower()
        t = ''.join(ch if ch.isalnum() else ' ' for ch in t)
        return {w for w in t.split() if len(w) >= 3}

    @classmethod
    def _caption_jaccard(cls, a: str, b: str) -> float:
        ta = cls._caption_tokens(a)
        tb = cls._caption_tokens(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def load_intra_doc_edges(self, latex_reference_graph_path: str) -> None:
        data = json.loads(Path(latex_reference_graph_path).read_text(encoding="utf-8"))
        docs = data.get("documents", {})
        for doc_id, doc in docs.items():
            doc_node_id = f"doc:{doc_id}"
            self.add_node(GraphNode(doc_node_id, NodeType.DOCUMENT, doc_id, payload={"meta": doc.get("metadata", {})}))

            labels = doc.get("labels", {})
            for label_key, info in labels.items():
                elem_node_id = f"elem:{doc_id}:{label_key}"
                self.add_node(GraphNode(elem_node_id, NodeType.ELEMENT, doc_id, payload=info))
                self.add_edge(GraphEdge(
                    src=doc_node_id,
                    tgt=elem_node_id,
                    edge_type=EdgeType.HAS_ELEMENT,
                    confidence=1.0,
                    attribution="derived",
                    metadata={},
                ))
                # Add reverse navigation edge to enable element->doc->citation traversal.
                self.add_edge(GraphEdge(
                    src=elem_node_id,
                    tgt=doc_node_id,
                    edge_type=EdgeType.HAS_ELEMENT,
                    confidence=1.0,
                    attribution="derived",
                    metadata={"reverse": True},
                ))

            for e in doc.get("edges", []):
                src = f"elem:{doc_id}:{e['source_label']}"
                tgt = f"elem:{doc_id}:{e['target_label']}"
                attribution = e.get("attribution") or ("containment" if e.get("ref_text") == "[containment]" else "enclosing_env")
                conf = self._edge_confidence(attribution)
                if src in self.nodes and tgt in self.nodes:
                    self.add_edge(GraphEdge(
                        src=src,
                        tgt=tgt,
                        edge_type=EdgeType.CONTAINMENT if attribution == "containment" else EdgeType.INTRA_REF,
                        confidence=conf,
                        attribution=attribution,
                        metadata={
                            "occurrence_count": e.get("occurrence_count", 1),
                            "source_type": e.get("source_type"),
                            "target_type": e.get("target_type"),
                        },
                    ))

    def load_citation_edges(self, citation_graph_path: str) -> None:
        data = json.loads(Path(citation_graph_path).read_text(encoding="utf-8"))
        for e in data.get("edges", []):
            src_doc = e["source"]
            tgt_doc = e["target"]
            src = f"doc:{src_doc}"
            tgt = f"doc:{tgt_doc}"
            if src not in self.nodes:
                self.add_node(GraphNode(src, NodeType.DOCUMENT, src_doc, payload={}))
            if tgt not in self.nodes:
                self.add_node(GraphNode(tgt, NodeType.DOCUMENT, tgt_doc, payload={}))
            conf = self._edge_confidence("citation", e.get("confidence", 0.0))
            self.add_edge(GraphEdge(
                src=src,
                tgt=tgt,
                edge_type=EdgeType.CITATION,
                confidence=conf,
                attribution="citation",
                metadata={
                    "match_method": e.get("match_method"),
                    "match_margin": e.get("match_margin"),
                    "is_ambiguous": e.get("is_ambiguous", False),
                },
            ))

    def load_mineru_elements_and_align(
        self,
        multimodal_elements_path: str,
        min_caption_jaccard: float = 0.35,
    ) -> None:
        """Load MinerU elements and align them to LaTeX element nodes by caption similarity."""
        data = json.loads(Path(multimodal_elements_path).read_text(encoding="utf-8"))
        docs = data.get("documents", {})

        # index latex element nodes by doc
        latex_by_doc: Dict[str, List[GraphNode]] = {}
        for n in self.nodes.values():
            if n.node_type == NodeType.ELEMENT and n.node_id.startswith("elem:"):
                latex_by_doc.setdefault(n.doc_id, []).append(n)

        align_count = 0
        for doc_id, doc in docs.items():
            mm_elements = doc.get("elements", {})
            if not isinstance(mm_elements, dict):
                continue
            doc_node_id = f"doc:{doc_id}"
            if doc_node_id not in self.nodes:
                self.add_node(GraphNode(doc_node_id, NodeType.DOCUMENT, doc_id, payload={}))

            for mm_id, mm in mm_elements.items():
                mm_node_id = f"mm:{doc_id}:{mm_id}"
                self.add_node(GraphNode(mm_node_id, NodeType.ELEMENT, doc_id, payload=mm))
                self.add_edge(GraphEdge(
                    src=doc_node_id,
                    tgt=mm_node_id,
                    edge_type=EdgeType.HAS_ELEMENT,
                    confidence=1.0,
                    attribution="derived",
                    metadata={"source": "mineru"},
                ))
                self.add_edge(GraphEdge(
                    src=mm_node_id,
                    tgt=doc_node_id,
                    edge_type=EdgeType.HAS_ELEMENT,
                    confidence=1.0,
                    attribution="derived",
                    metadata={"source": "mineru", "reverse": True},
                ))

                mm_caption = (mm.get("caption") or "")
                if not mm_caption:
                    continue

                best_node: Optional[GraphNode] = None
                best_sim = 0.0
                for latex_node in latex_by_doc.get(doc_id, []):
                    lp = latex_node.payload or {}
                    latex_caption = lp.get("caption") or ""
                    sim = self._caption_jaccard(mm_caption, latex_caption)
                    if sim > best_sim:
                        best_sim = sim
                        best_node = latex_node

                if best_node is not None and best_sim >= min_caption_jaccard:
                    conf = self._edge_confidence("alignment", best_sim)
                    self.add_edge(GraphEdge(
                        src=best_node.node_id,
                        tgt=mm_node_id,
                        edge_type=EdgeType.ALIGNMENT,
                        confidence=conf,
                        attribution="alignment",
                        metadata={"caption_jaccard": round(best_sim, 3)},
                    ))
                    self.add_edge(GraphEdge(
                        src=mm_node_id,
                        tgt=best_node.node_id,
                        edge_type=EdgeType.ALIGNMENT,
                        confidence=conf,
                        attribution="alignment",
                        metadata={"caption_jaccard": round(best_sim, 3), "reverse": True},
                    ))
                    align_count += 1

        print(f"Loaded MinerU elements and created {align_count} alignment links")

    def find_cross_doc_element_paths(
        self,
        max_hops: int = 4,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Find docA:elem -> ... -> docB:elem paths across citation edges."""
        results: List[Dict[str, Any]] = []
        element_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.ELEMENT]

        for start in element_nodes:
            stack: List[Tuple[str, List[str], List[float], Set[str]]] = [
                (start.node_id, [start.node_id], [], {start.node_id})
            ]
            while stack:
                current, path_nodes, confs, visited = stack.pop()
                if len(path_nodes) - 1 > max_hops:
                    continue

                current_node = self.nodes[current]
                if (
                    current_node.node_type == NodeType.ELEMENT
                    and current_node.doc_id != start.doc_id
                    and len(path_nodes) >= 4
                ):
                    score = self.compute_path_score(confs)
                    if score >= min_score:
                        results.append({
                            "start": start.node_id,
                            "end": current,
                            "path_nodes": path_nodes,
                            "score": score,
                            "hops": len(path_nodes) - 1,
                        })

                for edge in self._iter_out_edges(current):
                    nxt = edge.tgt
                    if nxt in visited:
                        continue
                    src_degree = len(self._iter_out_edges(current))
                    degree_penalty = 1.0 / math.log(max(2, src_degree + 1))
                    effective_conf = max(1e-5, edge.confidence * degree_penalty)
                    next_visited = set(visited)
                    next_visited.add(nxt)
                    stack.append((nxt, path_nodes + [nxt], confs + [effective_conf], next_visited))

        # De-duplicate by (start, end, path)
        uniq = {}
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            key = (r["start"], r["end"], tuple(r["path_nodes"]))
            uniq[key] = r
        return list(uniq.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [
                {
                    "node_id": n.node_id,
                    "node_type": n.node_type.value,
                    "doc_id": n.doc_id,
                    "payload": n.payload,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "src": e.src,
                    "tgt": e.tgt,
                    "edge_type": e.edge_type.value,
                    "confidence": e.confidence,
                    "attribution": e.attribution,
                    "metadata": e.metadata,
                }
                for e in self.edges
            ],
        }
