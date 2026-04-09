"""Multi-hop chain discovery within a single document.

Given the intra-document adjacency graph (built from edges in
``multimodal_elements.json``), this module discovers long element
chains that represent multi-hop reasoning paths.

Key concepts:

- A **chain** is a simple path through the element graph — no node
  is visited twice.
- Chains are scored by length, modality diversity, and cross-modal
  transitions.
- Unlike the old fixed 2-hop / 3-hop approach, the algorithm explores
  chains up to the full graph diameter.

Typical usage::

    finder = ChainFinder.from_doc(doc_dict)
    chains = finder.find_chains(min_length=3)
    top5 = finder.find_longest(top_k=5)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class ChainResult:
    """A discovered multi-hop chain within one document.

    Attributes
    ----------
    score : float
        Composite quality score (higher = better).  Used as the default
        sort key so ``sorted(chains)`` gives worst-first.
    path : tuple[str, ...]
        Ordered element IDs along the chain (length = hop_count + 1).
    doc_id : str
        Document containing all elements.
    hop_count : int
        Number of hops (edges traversed) = ``len(path) - 1``.
    modality_sequence : tuple[str, ...]
        Element types along the path, e.g. ``("figure", "table", "formula")``.
    cross_modal_transitions : int
        How many consecutive pairs in the chain differ in modality.
    unique_modalities : int
        Number of distinct modality types in the chain.
    """

    score: float
    path: Tuple[str, ...] = field(compare=False)
    doc_id: str = field(compare=False)
    hop_count: int = field(compare=False)
    modality_sequence: Tuple[str, ...] = field(compare=False)
    cross_modal_transitions: int = field(compare=False)
    unique_modalities: int = field(compare=False)

    @property
    def endpoint_pair_type(self) -> str:
        """Sorted pair type of the two endpoints."""
        return "+".join(sorted([self.modality_sequence[0],
                                self.modality_sequence[-1]]))

    @property
    def endpoints(self) -> Tuple[str, str]:
        """(first_element_id, last_element_id)."""
        return (self.path[0], self.path[-1])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_chain(
    path: List[str],
    types: List[str],
) -> float:
    """Score a chain for quality.

    Criteria (all additive, normalised to roughly 0–1):
    - Length bonus: ``min(len(path) / 8, 1.0) * 0.4``
    - Cross-modal transitions: each transition adds 0.1 (max 0.4)
    - Modality diversity: ``unique / 3 * 0.2``  (3 = fig/tbl/formula)
    """
    length_score = min(len(path) / 8.0, 1.0) * 0.4
    transitions = sum(1 for a, b in zip(types, types[1:]) if a != b)
    trans_score = min(transitions * 0.1, 0.4)
    unique = len(set(types))
    diversity_score = (unique / 3.0) * 0.2
    return round(length_score + trans_score + diversity_score, 4)


# ---------------------------------------------------------------------------
# ChainFinder
# ---------------------------------------------------------------------------

class ChainFinder:
    """Discover multi-hop chains in a single document's element graph.

    Parameters
    ----------
    doc_id : str
        Document identifier.
    elements : dict[str, dict]
        ``{element_id: element_dict}`` — only modal elements
        (figure / table / formula).
    adj : dict[str, set[str]]
        Undirected adjacency built from intra-doc edges.
    """

    MODAL_TYPES = frozenset({"figure", "table", "formula"})

    # Default maximum chain length (nodes) to prevent exponential blowup
    # on dense graphs.  Can be overridden per call.
    DEFAULT_MAX_LENGTH = 12

    # Maximum number of chains to collect before stopping DFS early.
    MAX_RESULTS = 2000

    def __init__(
        self,
        doc_id: str,
        elements: Dict[str, Dict[str, Any]],
        adj: Dict[str, Set[str]],
    ) -> None:
        self.doc_id = doc_id
        self._elements = elements
        self._adj = adj

    # -- Factories ---------------------------------------------------------

    @classmethod
    def from_doc(cls, doc: Dict[str, Any]) -> "ChainFinder":
        """Build a ChainFinder from a single document dict.

        Parameters
        ----------
        doc : dict
            One entry from ``multimodal_elements.json["documents"]``.
        """
        doc_id = doc.get("doc_id", "")
        raw_elements = doc.get("elements", {})
        elements: Dict[str, Dict[str, Any]] = {}
        for eid, el in raw_elements.items():
            etype = el.get("element_type", el.get("type", ""))
            if etype in cls.MODAL_TYPES:
                elements[eid] = el

        edges = doc.get("edges", [])
        adj: Dict[str, Set[str]] = defaultdict(set)
        for edge in edges:
            src = edge.get("source_id", "")
            tgt = edge.get("target_id", "")
            if src in elements and tgt in elements:
                adj[src].add(tgt)
                adj[tgt].add(src)

        return cls(doc_id, elements, dict(adj))

    @classmethod
    def from_file(
        cls, path: str, doc_id: str,
    ) -> "ChainFinder":
        """Load a specific document from multimodal_elements.json."""
        import json
        from pathlib import Path as P
        data = json.loads(P(path).read_text(encoding="utf-8"))
        doc = data.get("documents", {}).get(doc_id, {})
        doc.setdefault("doc_id", doc_id)
        return cls.from_doc(doc)

    # -- Public API --------------------------------------------------------

    def find_chains(
        self,
        min_length: int = 3,
        max_length: Optional[int] = None,
        cross_modal_only: bool = False,
    ) -> List[ChainResult]:
        """Enumerate all maximal simple paths (chains) in the element graph.

        A path is **maximal** if it cannot be extended in either direction
        without revisiting a node.

        Parameters
        ----------
        min_length : int
            Minimum number of nodes in the chain (not edges).
            ``min_length=3`` means ≥2 hops.
        max_length : int or None
            If set, stop growing chains beyond this many nodes.
            Defaults to ``DEFAULT_MAX_LENGTH`` (12) to prevent
            exponential blowup on dense graphs.
        cross_modal_only : bool
            If True, only emit chains whose endpoints are different modalities.

        Returns
        -------
        list[ChainResult]
            Sorted by score descending (best first).
        """
        if max_length is None:
            max_length = self.DEFAULT_MAX_LENGTH

        results: List[ChainResult] = []
        seen_paths: Set[FrozenSet[str]] = set()

        for start_node in sorted(self._elements.keys()):
            self._dfs_chains(
                start_node,
                [start_node],
                {start_node},
                min_length,
                max_length,
                cross_modal_only,
                results,
                seen_paths,
            )

        results.sort(key=lambda c: -c.score)
        return results

    def find_longest(
        self,
        top_k: int = 10,
        cross_modal_only: bool = False,
    ) -> List[ChainResult]:
        """Return the *top_k* longest chains.

        This is a convenience wrapper that finds all maximal chains,
        then returns the longest ones.  For very dense graphs you may
        want to use :meth:`find_chains` with ``max_length`` instead.
        """
        all_chains = self.find_chains(
            min_length=2,
            cross_modal_only=cross_modal_only,
        )
        # Sort by hop_count desc, then score desc
        all_chains.sort(key=lambda c: (-c.hop_count, -c.score))
        return all_chains[:top_k]

    def find_endpoint_pairs(
        self,
        min_hops: int = 2,
        max_hops: Optional[int] = None,
        cross_modal_only: bool = True,
    ) -> List[ChainResult]:
        """Find chains and deduplicate by endpoint pair.

        When multiple chains connect the same two endpoints, keep only
        the highest-scoring one.

        Parameters
        ----------
        min_hops : int
            Minimum hop count (edges, not nodes).
        max_hops : int or None
            Maximum hop count.
        cross_modal_only : bool
            If True, only keep chains whose endpoints differ in modality.
        """
        chains = self.find_chains(
            min_length=min_hops + 1,
            max_length=(max_hops + 1) if max_hops else None,
            cross_modal_only=cross_modal_only,
        )
        best: Dict[FrozenSet[str], ChainResult] = {}
        for c in chains:
            key = frozenset(c.endpoints)
            if key not in best or c.score > best[key].score:
                best[key] = c
        result = sorted(best.values(), key=lambda c: -c.score)
        return result

    # -- Statistics --------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics about the document graph."""
        n_elem = len(self._elements)
        n_edges = sum(len(nbs) for nbs in self._adj.values()) // 2
        types = [el.get("element_type", "") for el in self._elements.values()]
        type_dist = {}
        for t in types:
            type_dist[t] = type_dist.get(t, 0) + 1
        return {
            "doc_id": self.doc_id,
            "elements": n_elem,
            "edges": n_edges,
            "type_distribution": type_dist,
            "connected_components": self._count_components(),
        }

    # -- Internal ----------------------------------------------------------

    def _dfs_chains(
        self,
        node: str,
        path: List[str],
        visited: Set[str],
        min_length: int,
        max_length: Optional[int],
        cross_modal_only: bool,
        results: List[ChainResult],
        seen_paths: Set[FrozenSet[str]],
    ) -> None:
        """Depth-first enumeration of maximal simple paths."""
        # Early termination if we've collected enough chains
        if len(results) >= self.MAX_RESULTS:
            return

        neighbors = self._adj.get(node, set())
        extendable = [nb for nb in neighbors if nb not in visited]
        is_maximal = len(extendable) == 0
        at_max = max_length is not None and len(path) >= max_length

        if is_maximal or at_max:
            # Emit this path if long enough
            if len(path) >= min_length:
                self._maybe_emit(path, cross_modal_only, results, seen_paths)
            return

        for nb in sorted(extendable):
            if len(results) >= self.MAX_RESULTS:
                return
            visited.add(nb)
            path.append(nb)
            self._dfs_chains(
                nb, path, visited, min_length, max_length,
                cross_modal_only, results, seen_paths,
            )
            path.pop()
            visited.discard(nb)

    def _maybe_emit(
        self,
        path: List[str],
        cross_modal_only: bool,
        results: List[ChainResult],
        seen_paths: Set[FrozenSet[str]],
    ) -> None:
        """Emit a chain result if it passes filters and dedup."""
        types = [
            self._elements[eid].get("element_type", "")
            for eid in path
        ]
        if cross_modal_only and types[0] == types[-1]:
            return

        # Canonical dedup: use frozenset of the full path (order-agnostic)
        # to avoid emitting A→B→C and C→B→A as separate chains.
        canon = frozenset(path)
        if canon in seen_paths:
            return
        seen_paths.add(canon)

        transitions = sum(1 for a, b in zip(types, types[1:]) if a != b)
        unique_m = len(set(types))

        results.append(ChainResult(
            score=_score_chain(path, types),
            path=tuple(path),
            doc_id=self.doc_id,
            hop_count=len(path) - 1,
            modality_sequence=tuple(types),
            cross_modal_transitions=transitions,
            unique_modalities=unique_m,
        ))

    def _count_components(self) -> int:
        """Count connected components in the element graph."""
        visited: Set[str] = set()
        components = 0
        for node in self._elements:
            if node not in visited:
                components += 1
                stack = [node]
                while stack:
                    n = stack.pop()
                    if n in visited:
                        continue
                    visited.add(n)
                    stack.extend(self._adj.get(n, set()))
        return components
