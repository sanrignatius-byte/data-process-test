"""
Unified Two-Layer Graph for Cross-Document Multi-Hop Reasoning

Combines two levels of structure into a single traversable graph:

  Layer 1 (Doc-level):   citation edges between papers
  Layer 2 (Element-level): intra-document edges between figures/tables/formulas/sections

A cross-document multi-hop path looks like:

  docA:fig_3 --(intra ref)--> docA:eq_1 --(containment)--> docA
       |
       docA --(citation)--> docB
       |
       docB --(containment)--> docB:tab_2 --(intra ref)--> docB:fig_1

Key design decisions:
  - Nodes are heterogeneous: DocNode (paper) or ElementNode (figure/table/formula/section)
  - Edges carry attribution, confidence, and occurrence metadata
  - Confidence propagation uses geometric mean + log-length penalty (not raw product)
  - Hub suppression is applied at both levels (G1 pattern from cross-modal links)

Usage:
  graph = UnifiedGraph()
  graph.load_from_data(
      latex_ref_graph="data/latex_reference_graph.json",
      citation_graph="data/citation_graph.json",
      multimodal_elements="data/multimodal_elements.json",
  )
  paths = graph.find_cross_doc_paths(max_hops=4)
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeType(Enum):
    """Types of nodes in the unified graph."""
    DOCUMENT = "document"
    FIGURE = "figure"
    TABLE = "table"
    FORMULA = "formula"
    SECTION = "section"


class EdgeType(Enum):
    """Types of edges in the unified graph."""
    # Layer 1: between documents
    CITATION = "citation"          # docA cites docB (from .bbl matching)

    # Layer 2: within documents
    INTRA_REF = "intra_ref"        # element references another (from \ref{})
    CONTAINMENT = "containment"    # section contains element
    CO_REFERENCE = "co_reference"  # two elements co-referenced in same paragraph
    PROXIMITY = "proximity"        # elements nearby in document (from MinerU)

    # Cross-layer: connects doc to its elements
    HAS_ELEMENT = "has_element"    # document → element (structural)
    ALIGNMENT = "alignment"        # LaTeX element ↔ MinerU element entity resolution


class Attribution(Enum):
    """How an edge was discovered — controls confidence weighting."""
    ENCLOSING_ENV = "enclosing_env"        # \ref{} inside labeled environment [high]
    SECTION_FALLBACK = "section_fallback"  # nearest preceding section          [medium]
    CONTAINMENT = "containment"            # section → element structural       [high]
    CITATION_EXACT = "citation_exact"      # arXiv ID or exact title match      [high]
    CITATION_FUZZY = "citation_fuzzy"      # fuzzy title match                  [medium]
    CITATION_AMBIGUOUS = "citation_ambiguous"  # fuzzy match with low margin    [low]
    MINERU_PROXIMITY = "mineru_proximity"  # MinerU position-based              [low]
    MINERU_DIRECT = "mineru_direct"        # MinerU direct reference            [medium-high]
    LATEX_BRIDGE = "latex_bridge"          # LaTeX \ref{} co-citation bridge    [high]
    STRUCTURAL = "structural"             # doc → element ownership             [certain]
    LATEX_MINERU_ALIGNMENT = "latex_mineru_alignment"  # caption/entity match    [high]

    @property
    def base_confidence(self) -> float:
        """Default confidence weight for this attribution type."""
        return {
            Attribution.ENCLOSING_ENV: 0.95,
            Attribution.SECTION_FALLBACK: 0.65,
            Attribution.CONTAINMENT: 0.90,
            Attribution.CITATION_EXACT: 0.98,
            Attribution.CITATION_FUZZY: 0.80,
            Attribution.CITATION_AMBIGUOUS: 0.50,
            Attribution.MINERU_PROXIMITY: 0.55,
            Attribution.MINERU_DIRECT: 0.85,
            Attribution.LATEX_BRIDGE: 0.90,
            Attribution.STRUCTURAL: 1.00,
            Attribution.LATEX_MINERU_ALIGNMENT: 0.92,
        }[self]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class UnifiedNode:
    """A node in the unified graph (either a document or an element)."""
    node_id: str              # e.g. "1908.09635" or "1908.09635_figure_3"
    node_type: NodeType
    doc_id: str               # which document this belongs to
    label: str                # human-readable label
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_document(self) -> bool:
        return self.node_type == NodeType.DOCUMENT

    @property
    def is_element(self) -> bool:
        return not self.is_document

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "doc_id": self.doc_id,
            "label": self.label,
            "metadata": self.metadata,
        }


@dataclass
class UnifiedEdge:
    """
    A directed edge in the unified graph with confidence and attribution.

    The confidence field controls path scoring. Attribution tells you
    how the edge was discovered, which determines its reliability tier.
    """
    source_id: str
    target_id: str
    edge_type: EdgeType
    attribution: Attribution
    confidence: float            # 0.0 – 1.0, used for path scoring
    occurrence_count: int = 1    # how many times this edge was observed
    context: str = ""            # surrounding text evidence
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_cross_document(self) -> bool:
        return self.edge_type == EdgeType.CITATION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "attribution": self.attribution.value,
            "confidence": round(self.confidence, 4),
            "occurrence_count": self.occurrence_count,
            "context": self.context[:300],
            "metadata": self.metadata,
        }


@dataclass
class ScoredPath:
    """A path through the unified graph with confidence score."""
    node_ids: List[str]
    edge_indices: List[int]     # indices into UnifiedGraph.edges
    path_score: float           # geometric mean confidence with length penalty
    docs_involved: Set[str]
    modalities_involved: Set[str]
    hop_attributions: List[str]  # attribution at each hop

    @property
    def num_hops(self) -> int:
        return len(self.edge_indices)

    @property
    def is_cross_document(self) -> bool:
        return len(self.docs_involved) >= 2

    @property
    def is_cross_modal(self) -> bool:
        return len(self.modalities_involved) >= 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_ids": self.node_ids,
            "num_hops": self.num_hops,
            "path_score": round(self.path_score, 4),
            "docs_involved": sorted(self.docs_involved),
            "modalities_involved": sorted(self.modalities_involved),
            "hop_attributions": self.hop_attributions,
            "is_cross_document": self.is_cross_document,
            "is_cross_modal": self.is_cross_modal,
        }


# ---------------------------------------------------------------------------
# Unified Graph
# ---------------------------------------------------------------------------

class UnifiedGraph:
    """
    Two-layer heterogeneous graph combining:
      - Doc-level citation edges (Layer 1)
      - Element-level intra-doc edges (Layer 2)

    Paths that traverse both layers naturally produce cross-document
    multi-hop reasoning chains:

      docA:elem_i -> (intra) -> docA -> (citation) -> docB -> (intra) -> docB:elem_j
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, UnifiedNode] = {}
        self.edges: List[UnifiedEdge] = []

        # Adjacency: node_id -> [(edge_index, neighbor_id)]
        self._adj_forward: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        self._adj_reverse: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

        # Quick lookups
        self._doc_nodes: Set[str] = set()
        self._element_nodes: Dict[str, Set[str]] = defaultdict(set)  # doc_id -> element node_ids

    # -------------------------------------------------------------------
    # Graph construction
    # -------------------------------------------------------------------

    def add_node(self, node: UnifiedNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node
        if node.is_document:
            self._doc_nodes.add(node.node_id)
        else:
            self._element_nodes[node.doc_id].add(node.node_id)

    def add_edge(self, edge: UnifiedEdge) -> int:
        """Add an edge and return its index."""
        idx = len(self.edges)
        self.edges.append(edge)
        self._adj_forward[edge.source_id].append((idx, edge.target_id))
        self._adj_reverse[edge.target_id].append((idx, edge.source_id))
        return idx

    # -------------------------------------------------------------------
    # Data loading — wire together existing data sources
    # -------------------------------------------------------------------

    def load_citation_edges(self, citation_data: Dict[str, Any]) -> int:
        """
        Load Layer 1 (doc-level) edges from citation_graph.json.

        Returns number of edges added.
        """
        added = 0
        edges_data = citation_data.get("edges", [])

        for e in edges_data:
            src_id = e["source"]
            tgt_id = e["target"]

            # Ensure document nodes exist
            for doc_id in (src_id, tgt_id):
                if doc_id not in self.nodes:
                    self.add_node(UnifiedNode(
                        node_id=doc_id,
                        node_type=NodeType.DOCUMENT,
                        doc_id=doc_id,
                        label=doc_id,
                        metadata={"title": e.get("bib_title", "")},
                    ))

            # Determine attribution based on match quality
            method = e.get("match_method", "title_fuzzy")
            is_ambiguous = e.get("is_ambiguous", False)
            if method in ("arxiv_id_explicit", "arxiv_id_bare"):
                attr = Attribution.CITATION_EXACT
            elif is_ambiguous:
                attr = Attribution.CITATION_AMBIGUOUS
            else:
                attr = Attribution.CITATION_FUZZY

            conf = e.get("confidence", 0.75)
            # Downgrade ambiguous matches
            if is_ambiguous:
                conf = min(conf, 0.50)

            self.add_edge(UnifiedEdge(
                source_id=src_id,
                target_id=tgt_id,
                edge_type=EdgeType.CITATION,
                attribution=attr,
                confidence=conf,
                occurrence_count=e.get("cite_count", 1),
                context=e.get("contexts", [""])[0][:300] if e.get("contexts") else "",
                metadata={
                    "match_method": method,
                    "match_margin": e.get("match_margin", 1.0),
                    "cite_keys": e.get("cite_keys", []),
                },
            ))
            added += 1

        return added

    def load_intra_doc_edges(
        self,
        latex_ref_data: Dict[str, Any],
        element_data: Optional[Dict[str, Any]] = None,
        enable_alignment: bool = True,
        alignment_threshold: float = 0.55,
    ) -> int:
        """
        Load Layer 2 (element-level) edges from latex_reference_graph.json
        and optionally multimodal_elements.json.

        Returns number of edges added.
        """
        added = 0
        documents = latex_ref_data.get("documents", {})

        for doc_id, doc in documents.items():
            # Ensure document node exists
            if doc_id not in self.nodes:
                title = doc.get("metadata", {}).get("title", doc_id)
                self.add_node(UnifiedNode(
                    node_id=doc_id,
                    node_type=NodeType.DOCUMENT,
                    doc_id=doc_id,
                    label=title,
                ))

            # Create element nodes from labels
            labels = doc.get("labels", {})
            for label_key, label_info in labels.items():
                label_type = label_info.get("label_type", "unknown")
                node_type = self._label_type_to_node_type(label_type)
                if node_type is None:
                    continue

                elem_id = f"{doc_id}:{label_key}"
                if elem_id not in self.nodes:
                    self.add_node(UnifiedNode(
                        node_id=elem_id,
                        node_type=node_type,
                        doc_id=doc_id,
                        label=label_key,
                        metadata={
                            "latex_label": label_key,
                            "label_type": label_type,
                            "number": label_info.get("number"),
                        },
                    ))

                # has_element edge: doc → element
                self.add_edge(UnifiedEdge(
                    source_id=doc_id,
                    target_id=elem_id,
                    edge_type=EdgeType.HAS_ELEMENT,
                    attribution=Attribution.STRUCTURAL,
                    confidence=1.0,
                ))
                added += 1

            # Create intra-doc edges from LaTeX ref edges
            for edge_data in doc.get("edges", []):
                src_label = edge_data.get("source_label", "")
                tgt_label = edge_data.get("target_label", "")
                src_id = f"{doc_id}:{src_label}"
                tgt_id = f"{doc_id}:{tgt_label}"

                if src_id not in self.nodes or tgt_id not in self.nodes:
                    continue

                # Map attribution string to enum
                attr_str = edge_data.get("attribution", "enclosing_env")
                attr_map = {
                    "enclosing_env": Attribution.ENCLOSING_ENV,
                    "section_fallback": Attribution.SECTION_FALLBACK,
                    "containment": Attribution.CONTAINMENT,
                }
                attr = attr_map.get(attr_str, Attribution.SECTION_FALLBACK)

                ref_text = edge_data.get("ref_text", "")
                edge_type = EdgeType.CONTAINMENT if ref_text == "[containment]" else EdgeType.INTRA_REF

                self.add_edge(UnifiedEdge(
                    source_id=src_id,
                    target_id=tgt_id,
                    edge_type=edge_type,
                    attribution=attr,
                    confidence=attr.base_confidence,
                    occurrence_count=edge_data.get("occurrence_count", 1),
                    context=edge_data.get("context", "")[:300],
                ))
                added += 1

        # Also load MinerU-based element edges if provided
        if element_data:
            added += self._load_mineru_elements(element_data)
            if enable_alignment:
                added += self._align_latex_and_mineru_elements(
                    threshold=alignment_threshold,
                )

        return added

    def _load_mineru_elements(self, element_data: Dict[str, Any]) -> int:
        """Load element nodes and edges from multimodal_elements.json."""
        added = 0
        for doc_id, doc_info in element_data.items():
            if not isinstance(doc_info, dict):
                continue

            elements = doc_info.get("elements", {})
            edges_list = doc_info.get("edges", [])

            # Create element nodes
            for elem_id, elem in elements.items():
                if elem_id in self.nodes:
                    continue
                elem_type = elem.get("element_type", "text")
                node_type = {
                    "figure": NodeType.FIGURE,
                    "table": NodeType.TABLE,
                    "formula": NodeType.FORMULA,
                    "section": NodeType.SECTION,
                }.get(elem_type)
                if node_type is None:
                    continue

                self.add_node(UnifiedNode(
                    node_id=elem_id,
                    node_type=node_type,
                    doc_id=doc_id,
                    label=elem.get("label", elem_id),
                    metadata={
                        "caption": elem.get("caption", ""),
                        "image_path": elem.get("image_path"),
                        "number": elem.get("number"),
                    },
                ))

            # Create edges
            for edge in edges_list:
                src = edge.get("source_id", "")
                tgt = edge.get("target_id", "")
                if src not in self.nodes or tgt not in self.nodes:
                    continue

                self.add_edge(UnifiedEdge(
                    source_id=src,
                    target_id=tgt,
                    edge_type=EdgeType.CO_REFERENCE,
                    attribution=Attribution.MINERU_DIRECT,
                    confidence=Attribution.MINERU_DIRECT.base_confidence,
                    context=edge.get("context_snippet", "")[:300],
                ))
                added += 1

        return added

    def _align_latex_and_mineru_elements(self, threshold: float = 0.55) -> int:
        """
        Align LaTeX logical elements with MinerU physical elements.

        Creates bidirectional ALIGNMENT edges between best caption matches
        (same document + same modality) so downstream paths can bridge
        topology-only nodes and image-backed nodes.
        """
        added = 0

        for doc_id, element_ids in self._element_nodes.items():
            latex_ids = [
                nid for nid in element_ids
                if self.nodes[nid].metadata.get("latex_label")
            ]
            mineru_ids = [
                nid for nid in element_ids
                if self.nodes[nid].metadata.get("image_path") or self.nodes[nid].metadata.get("caption")
            ]
            if not latex_ids or not mineru_ids:
                continue

            used_mineru: Set[str] = set()
            for latex_id in latex_ids:
                latex_node = self.nodes[latex_id]
                best_mineru_id: Optional[str] = None
                best_score = 0.0

                for mineru_id in mineru_ids:
                    if mineru_id in used_mineru:
                        continue
                    mineru_node = self.nodes[mineru_id]
                    if mineru_node.node_type != latex_node.node_type:
                        continue

                    sim = self._element_similarity(latex_node, mineru_node)
                    if sim > best_score:
                        best_score = sim
                        best_mineru_id = mineru_id

                if not best_mineru_id or best_score < threshold:
                    continue

                used_mineru.add(best_mineru_id)
                conf = min(1.0, 0.75 + 0.25 * best_score)
                metadata = {"alignment_score": round(best_score, 4)}

                self.add_edge(UnifiedEdge(
                    source_id=latex_id,
                    target_id=best_mineru_id,
                    edge_type=EdgeType.ALIGNMENT,
                    attribution=Attribution.LATEX_MINERU_ALIGNMENT,
                    confidence=conf,
                    metadata=metadata,
                ))
                self.add_edge(UnifiedEdge(
                    source_id=best_mineru_id,
                    target_id=latex_id,
                    edge_type=EdgeType.ALIGNMENT,
                    attribution=Attribution.LATEX_MINERU_ALIGNMENT,
                    confidence=conf,
                    metadata=metadata,
                ))
                added += 2

        return added

    @staticmethod
    def _normalize_caption_text(text: str) -> str:
        t = (text or "").lower().strip()
        t = re.sub(r"[^\w\s]", " ", t)
        t = re.sub(r"\s+", " ", t)
        return t

    def _element_similarity(self, latex_node: UnifiedNode, mineru_node: UnifiedNode) -> float:
        """Caption + number based similarity for entity resolution."""
        latex_caption = self._normalize_caption_text(latex_node.metadata.get("caption", ""))
        mineru_caption = self._normalize_caption_text(mineru_node.metadata.get("caption", ""))

        caption_sim = self._token_jaccard(latex_caption, mineru_caption)

        n1 = str(latex_node.metadata.get("number", "") or "").strip()
        n2 = str(mineru_node.metadata.get("number", "") or "").strip()
        number_match = 1.0 if n1 and n2 and n1 == n2 else 0.0

        if caption_sim == 0.0 and number_match == 0.0:
            return 0.0
        return 0.8 * caption_sim + 0.2 * number_match

    @staticmethod
    def _token_jaccard(t1: str, t2: str) -> float:
        w1 = set(t1.split())
        w2 = set(t2.split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)

    @staticmethod
    def _label_type_to_node_type(label_type: str) -> Optional[NodeType]:
        """Convert LaTeX label type to unified NodeType."""
        return {
            "figure": NodeType.FIGURE,
            "table": NodeType.TABLE,
            "equation": NodeType.FORMULA,
            "section": NodeType.SECTION,
            "algorithm": None,  # skip for now
            "theorem": None,
            "unknown": None,
        }.get(label_type)

    # -------------------------------------------------------------------
    # B3: Confidence propagation
    # -------------------------------------------------------------------

    @staticmethod
    def compute_path_score(
        edge_confidences: List[float],
        decay_factor: float = 0.85,
    ) -> float:
        """
        B3: Markovian decay score using joint probability and hop decay.

        Formula:
            score = (∏ confidence_i) * (decay_factor ** (n - 1))

        Args:
            edge_confidences: confidence values for each hop
            decay_factor: multiplicative per-hop decay factor

        Returns:
            Path score in [0, 1] range (clamped)
        """
        if not edge_confidences:
            return 0.0

        joint_prob = math.prod(max(c, 1e-5) for c in edge_confidences)
        score = joint_prob * (decay_factor ** (len(edge_confidences) - 1))
        return max(0.0, min(1.0, score))

    # -------------------------------------------------------------------
    # Path finding
    # -------------------------------------------------------------------

    def find_cross_doc_element_paths(
        self,
        max_hops: int = 5,
        min_score: float = 0.3,
        require_cross_doc: bool = True,
        require_cross_modal: bool = True,
        max_paths: int = 500,
    ) -> List[ScoredPath]:
        """
        Find scored paths connecting elements across documents.

        A valid cross-doc path must:
        1. Start and end at element nodes (not document nodes)
        2. Traverse at least one citation edge
        3. Optionally cross modalities (figure→table, etc.)

        Args:
            max_hops: maximum number of edges in path
            min_score: minimum path score to keep
            require_cross_doc: require path to span 2+ documents
            require_cross_modal: require path to span 2+ modalities
            max_paths: maximum number of paths to return

        Returns:
            List of ScoredPath sorted by score descending
        """
        results: List[ScoredPath] = []

        # Start from all element nodes
        element_starts = [
            nid for nid, n in self.nodes.items()
            if n.is_element
        ]

        for start_id in element_starts:
            self._dfs_paths(
                current=start_id,
                path_nodes=[start_id],
                path_edges=[],
                visited={start_id},
                max_hops=max_hops,
                min_score=min_score,
                require_cross_doc=require_cross_doc,
                require_cross_modal=require_cross_modal,
                results=results,
                max_paths=max_paths,
            )
            if len(results) >= max_paths:
                break

        results.sort(key=lambda p: -p.path_score)
        return results[:max_paths]

    def _dfs_paths(
        self,
        current: str,
        path_nodes: List[str],
        path_edges: List[int],
        visited: Set[str],
        max_hops: int,
        min_score: float,
        require_cross_doc: bool,
        require_cross_modal: bool,
        results: List[ScoredPath],
        max_paths: int,
    ) -> None:
        """DFS to find valid paths."""
        if len(results) >= max_paths:
            return

        # Check if current path is a valid result
        if len(path_edges) >= 2:
            end_node = self.nodes.get(current)
            if end_node and end_node.is_element:
                # Compute score
                confidences = self._get_effective_confidences(path_edges)
                score = self.compute_path_score(confidences)

                if score >= min_score:
                    docs = {self.nodes[nid].doc_id for nid in path_nodes if nid in self.nodes}
                    modalities = {
                        self.nodes[nid].node_type.value
                        for nid in path_nodes
                        if nid in self.nodes and self.nodes[nid].is_element
                    }

                    valid = True
                    if require_cross_doc and len(docs) < 2:
                        valid = False
                    if require_cross_modal and len(modalities) < 2:
                        valid = False

                    if valid:
                        results.append(ScoredPath(
                            node_ids=list(path_nodes),
                            edge_indices=list(path_edges),
                            path_score=score,
                            docs_involved=docs,
                            modalities_involved=modalities,
                            hop_attributions=[
                                self.edges[ei].attribution.value for ei in path_edges
                            ],
                        ))

        # Continue searching if we haven't hit max hops
        if len(path_edges) >= max_hops:
            return

        # Explore neighbors (both forward and backward for undirected traversal)
        neighbors: List[Tuple[int, str]] = []
        neighbors.extend(self._adj_forward.get(current, []))
        neighbors.extend(self._adj_reverse.get(current, []))

        for edge_idx, neighbor_id in neighbors:
            if neighbor_id in visited:
                continue

            # Early pruning: check if adding this edge could still reach min_score
            current_confs = self._get_effective_confidences(path_edges)
            current_confs.append(self._effective_edge_confidence(current, edge_idx))
            optimistic_score = self.compute_path_score(current_confs)
            if optimistic_score < min_score * 0.5:  # generous early prune
                continue

            visited.add(neighbor_id)
            path_nodes.append(neighbor_id)
            path_edges.append(edge_idx)

            self._dfs_paths(
                current=neighbor_id,
                path_nodes=path_nodes,
                path_edges=path_edges,
                visited=visited,
                max_hops=max_hops,
                min_score=min_score,
                require_cross_doc=require_cross_doc,
                require_cross_modal=require_cross_modal,
                results=results,
                max_paths=max_paths,
            )

            path_nodes.pop()
            path_edges.pop()
            visited.discard(neighbor_id)

    def _effective_edge_confidence(self, current_node: str, edge_idx: int) -> float:
        """Apply soft degree penalty to damp hub-driven traversals."""
        edge = self.edges[edge_idx]
        out_degree = len(self._adj_forward.get(current_node, []))
        degree_penalty = 1.0 / math.log(max(2, out_degree + 1))
        return max(1e-5, min(1.0, edge.confidence * degree_penalty))

    def _get_effective_confidences(self, edge_indices: List[int]) -> List[float]:
        """Compute effective confidences along a path with dynamic degree penalties."""
        if not edge_indices:
            return []

        confs: List[float] = []
        for i, edge_idx in enumerate(edge_indices):
            if i == 0:
                src = self.edges[edge_idx].source_id
            else:
                src = self.edges[edge_indices[i - 1]].target_id
            confs.append(self._effective_edge_confidence(src, edge_idx))
        return confs

    # -------------------------------------------------------------------
    # B4: Hub suppression (generalized from G1/G2)
    # -------------------------------------------------------------------

    def get_node_degrees(self) -> Dict[str, Dict[str, int]]:
        """Get in/out degree for each node."""
        degrees: Dict[str, Dict[str, int]] = {}
        for nid in self.nodes:
            degrees[nid] = {
                "out": len(self._adj_forward.get(nid, [])),
                "in": len(self._adj_reverse.get(nid, [])),
                "total": (len(self._adj_forward.get(nid, []))
                          + len(self._adj_reverse.get(nid, []))),
            }
        return degrees

    def identify_hubs(
        self,
        max_degree: int = 20,
        node_type: Optional[NodeType] = None,
    ) -> List[str]:
        """
        Identify hub nodes that may need suppression.

        Args:
            max_degree: degree threshold above which a node is a hub
            node_type: only check nodes of this type (None = all)

        Returns:
            List of hub node IDs sorted by degree descending
        """
        degrees = self.get_node_degrees()
        hubs = []
        for nid, d in degrees.items():
            if node_type and self.nodes[nid].node_type != node_type:
                continue
            if d["total"] > max_degree:
                hubs.append((nid, d["total"]))
        hubs.sort(key=lambda x: -x[1])
        return [h[0] for h in hubs]

    # -------------------------------------------------------------------
    # Statistics & export
    # -------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Compute graph statistics."""
        node_type_counts: Dict[str, int] = defaultdict(int)
        for n in self.nodes.values():
            node_type_counts[n.node_type.value] += 1

        edge_type_counts: Dict[str, int] = defaultdict(int)
        attr_counts: Dict[str, int] = defaultdict(int)
        for e in self.edges:
            edge_type_counts[e.edge_type.value] += 1
            attr_counts[e.attribution.value] += 1

        num_docs = len(self._doc_nodes)
        docs_with_elements = sum(
            1 for elems in self._element_nodes.values() if elems
        )

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "node_type_distribution": dict(node_type_counts),
            "edge_type_distribution": dict(edge_type_counts),
            "attribution_distribution": dict(attr_counts),
            "num_documents": num_docs,
            "docs_with_elements": docs_with_elements,
            "avg_elements_per_doc": (
                sum(len(e) for e in self._element_nodes.values()) / max(1, docs_with_elements)
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export full graph as serializable dict."""
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "statistics": self.get_statistics(),
        }

    def export_paths_for_query_generation(
        self,
        paths: List[ScoredPath],
    ) -> List[Dict[str, Any]]:
        """
        Export scored paths in a format ready for query generation scripts.

        Each path becomes a candidate pair/chain with all necessary context
        for prompt construction.
        """
        candidates = []
        for path in paths:
            nodes_info = []
            for nid in path.node_ids:
                node = self.nodes.get(nid)
                if node:
                    nodes_info.append(node.to_dict())

            edges_info = []
            for ei in path.edge_indices:
                edge = self.edges[ei]
                edges_info.append(edge.to_dict())

            candidates.append({
                "path": path.to_dict(),
                "nodes": nodes_info,
                "edges": edges_info,
                "start_element": path.node_ids[0],
                "end_element": path.node_ids[-1],
                "score": path.path_score,
            })

        return candidates
