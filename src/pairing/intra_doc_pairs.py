"""Intra-document element pair selector.

Generates pairs of multimodal elements within a single document using
four complementary strategies:

- **direct**: Elements linked by a direct cross-reference edge (hop=1).
- **2hop**: Elements reachable via a 2-hop path through a shared
  intermediate element (e.g. figure→table→formula).
- **section**: Elements co-located in the same document section.
- **chain**: Multi-hop chains discovered via DFS traversal — no fixed
  hop limit.  Emits pairs at chain endpoints with the full path.

All strategies enforce a strict same-document boundary check.
Cross-document pairs are never produced.

Typical usage::

    selector = IntraDocPairSelector.from_file(
        "data/01_graphs/multimodal_elements.json"
    )
    pairs = selector.select(strategy="all", max_per_doc=10)
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from src.pairing.pair_schema import CandidatePair, ElementDetail
from src.pairing.chain_finder import ChainFinder
from src.pairing.context_dedup import dedup_context

# Element types eligible for pairing (non-text modalities).
MODAL_TYPES = frozenset({"figure", "table", "formula"})

# Truncation limits for element detail fields (match enrich_hub_candidates.py).
MAX_CONTENT_LENGTH = 2000
MAX_CONTEXT_LENGTH = 300
MAX_CAPTION_DISPLAY = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _element_detail(el: Dict[str, Any]) -> ElementDetail:
    """Build an ElementDetail from a raw element dict."""
    raw_before = (el.get("context_before", "") or "")[:MAX_CONTEXT_LENGTH]
    raw_after = (el.get("context_after", "") or "")[:MAX_CONTEXT_LENGTH]
    cleaned_before, cleaned_after = dedup_context(raw_before, raw_after)
    return ElementDetail(
        element_id=el.get("element_id", ""),
        element_type=el.get("element_type", ""),
        caption=el.get("caption", "") or "",
        content=(el.get("content", "") or "")[:MAX_CONTENT_LENGTH],
        image_path=el.get("image_path", "") or "",
        context_before=cleaned_before,
        context_after=cleaned_after,
        enriched_title=el.get("enriched_title", "") or "",
        enriched_content=el.get("enriched_content", "") or "",
        enriched_metadata=el.get("enriched_metadata", {}) or {},
    )


def _make_pair_type(type_a: str, type_b: str) -> str:
    return "+".join(sorted([type_a, type_b]))


def _build_hub_summary(el_a: Dict[str, Any], el_b: Dict[str, Any],
                       edge_text: str = "") -> str:
    """Build a brief textual summary describing both elements."""
    cap_a = (el_a.get("caption", "") or "")[:MAX_CAPTION_DISPLAY]
    cap_b = (el_b.get("caption", "") or "")[:MAX_CAPTION_DISPLAY]
    parts = []
    if cap_a:
        parts.append(f"[{el_a.get('element_type','').upper()} A] {cap_a[:MAX_CAPTION_DISPLAY]}")
    if cap_b:
        parts.append(f"[{el_b.get('element_type','').upper()} B] {cap_b[:MAX_CAPTION_DISPLAY]}")
    if edge_text:
        parts.append(f"[BRIDGE] {edge_text[:MAX_CAPTION_DISPLAY]}")
    return " | ".join(parts) if parts else ""


def _doc_id_from_element_id(eid: str) -> str:
    """Extract doc_id from MinerU element_id like '1306.5204_figure_1'."""
    for sep in ("_figure_", "_table_", "_formula_"):
        if sep in eid:
            return eid.split(sep)[0]
    return ""


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

class IntraDocPairSelector:
    """Select intra-document multimodal element pairs.

    Parameters
    ----------
    documents : dict
        ``multimodal_elements.json["documents"]`` — keyed by doc_id.
    """

    def __init__(self, documents: Dict[str, Dict[str, Any]]) -> None:
        self._docs = documents
        # Pre-build per-doc element and edge indices.
        self._elements: Dict[str, Dict[str, Dict[str, Any]]] = {}  # doc_id → eid → el
        self._edges: Dict[str, List[Dict[str, Any]]] = {}          # doc_id → edges
        self._sections: Dict[str, Dict[str, List[str]]] = {}       # doc_id → section_label → eids

        for doc_id, doc in documents.items():
            raw_elements = doc.get("elements", {})
            elems: Dict[str, Dict[str, Any]] = {}
            for eid, el in raw_elements.items():
                etype = el.get("element_type", el.get("type", ""))
                if etype in MODAL_TYPES:
                    el_copy = dict(el)
                    el_copy.setdefault("element_id", eid)
                    el_copy.setdefault("element_type", etype)
                    el_copy.setdefault("doc_id", doc_id)
                    elems[eid] = el_copy
            self._elements[doc_id] = elems
            self._edges[doc_id] = doc.get("edges", [])

    # -- Factory -----------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> "IntraDocPairSelector":
        """Load multimodal_elements.json and construct a selector."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data.get("documents", {}))

    # -- Public API --------------------------------------------------------

    def select(
        self,
        strategy: str = "all",
        max_per_doc: int = 20,
        pair_types: Optional[Set[str]] = None,
        min_quality: float = 0.0,
        min_chain_hops: int = 2,
    ) -> List[CandidatePair]:
        """Select pairs across all documents.

        Parameters
        ----------
        strategy
            ``"direct"`` | ``"2hop"`` | ``"section"`` | ``"chain"`` | ``"all"``
        max_per_doc
            Maximum pairs per document (applied per strategy, then merged).
        pair_types
            If given, only keep pairs whose pair_type is in this set.
            E.g. ``{"figure+table", "figure+formula"}``.
        min_quality
            Minimum quality score (0-1) to include a pair.
        min_chain_hops
            Minimum number of hops for chain strategy (default 2).
        """
        all_pairs: List[CandidatePair] = []
        for doc_id in sorted(self._docs.keys()):
            doc_pairs = self._select_for_doc(
                doc_id, strategy, max_per_doc, min_chain_hops,
            )
            all_pairs.extend(doc_pairs)

        # Post-filter
        if pair_types:
            all_pairs = [p for p in all_pairs if p.pair_type in pair_types]
        if min_quality > 0:
            all_pairs = [p for p in all_pairs if p.quality_score >= min_quality]

        return all_pairs

    # -- Per-doc selection -------------------------------------------------

    def _select_for_doc(
        self,
        doc_id: str,
        strategy: str,
        max_per_doc: int,
        min_chain_hops: int = 2,
    ) -> List[CandidatePair]:
        elems = self._elements.get(doc_id, {})
        if len(elems) < 2:
            return []

        pairs: List[CandidatePair] = []
        seen: Set[FrozenSet[str]] = set()  # dedup by element pair

        def _add(pair: CandidatePair) -> bool:
            key = frozenset([pair.element_a_id, pair.element_b_id])
            if key in seen:
                return False
            seen.add(key)
            pairs.append(pair)
            return True

        counter = 0

        def _next_id() -> str:
            nonlocal counter
            counter += 1
            return f"{doc_id}_pair_{counter}"

        if strategy in ("direct", "all"):
            for p in self._direct_pairs(doc_id, elems, _next_id):
                _add(p)

        if strategy in ("2hop", "all"):
            for p in self._two_hop_pairs(doc_id, elems, _next_id):
                _add(p)

        if strategy in ("section", "all"):
            for p in self._section_pairs(doc_id, elems, _next_id):
                _add(p)

        if strategy in ("chain", "all"):
            for p in self._chain_pairs(doc_id, elems, _next_id, min_chain_hops):
                _add(p)

        # Sort by quality descending, then cap
        pairs.sort(key=lambda p: -p.quality_score)
        return pairs[:max_per_doc]

    # -- Strategy A: direct reference edges --------------------------------

    def _direct_pairs(
        self,
        doc_id: str,
        elems: Dict[str, Dict[str, Any]],
        next_id,
    ) -> List[CandidatePair]:
        """Pairs connected by a direct cross-reference edge (hop=1)."""
        edges = self._edges.get(doc_id, [])
        pairs: List[CandidatePair] = []
        seen: Set[FrozenSet[str]] = set()

        for edge in edges:
            src = edge.get("source_id", "")
            tgt = edge.get("target_id", "")
            if src not in elems or tgt not in elems:
                continue
            el_a, el_b = elems[src], elems[tgt]
            ta = el_a.get("element_type", "")
            tb = el_b.get("element_type", "")
            if ta == tb:
                continue  # same modality → not cross-modal

            key = frozenset([src, tgt])
            if key in seen:
                continue
            seen.add(key)

            ref_text = edge.get("ref_text", "")
            ctx = edge.get("context_snippet", "")
            edge_ctx = [{
                "source": src, "target": tgt,
                "ref_text": ref_text, "context_snippet": ctx,
            }] if ref_text else []

            pairs.append(CandidatePair(
                pair_id=next_id(),
                doc_id=doc_id,
                element_a_id=src,
                element_b_id=tgt,
                element_a_type=ta,
                element_b_type=tb,
                pair_type=_make_pair_type(ta, tb),
                hop_distance=1,
                path=[src, tgt],
                quality_score=1.0,  # direct reference = highest quality
                element_a=_element_detail(el_a),
                element_b=_element_detail(el_b),
                node_group=[_element_detail(el_a), _element_detail(el_b)],
                edge_contexts=edge_ctx,
                hub_semantic_summary=_build_hub_summary(el_a, el_b, ref_text),
                strategy="direct",
                hub_metadata={"is_cross_doc": False, "strategy": "direct"},
            ))

        return pairs

    # -- Strategy B: 2-hop via shared neighbor -----------------------------

    def _two_hop_pairs(
        self,
        doc_id: str,
        elems: Dict[str, Dict[str, Any]],
        next_id,
    ) -> List[CandidatePair]:
        """Pairs reachable in 2 hops via a shared intermediate element.

        A → mid → B where A and B are different modalities and mid is any
        element (same or different modality) in the same document.
        """
        edges = self._edges.get(doc_id, [])

        # Build adjacency: eid → set of neighbor eids (undirected, intra-doc only)
        adj: Dict[str, Set[str]] = defaultdict(set)
        edge_text: Dict[Tuple[str, str], str] = {}
        for edge in edges:
            src = edge.get("source_id", "")
            tgt = edge.get("target_id", "")
            if src in elems and tgt in elems:
                adj[src].add(tgt)
                adj[tgt].add(src)
                key = (min(src, tgt), max(src, tgt))
                if key not in edge_text:
                    edge_text[key] = edge.get("ref_text", "")

        pairs: List[CandidatePair] = []
        seen: Set[FrozenSet[str]] = set()

        for mid_id in elems:
            neighbors = adj.get(mid_id, set())
            if len(neighbors) < 2:
                continue
            neighbor_list = sorted(neighbors)
            for i, ea_id in enumerate(neighbor_list):
                for eb_id in neighbor_list[i + 1:]:
                    ta = elems[ea_id].get("element_type", "")
                    tb = elems[eb_id].get("element_type", "")
                    if ta == tb:
                        continue  # need cross-modal

                    key = frozenset([ea_id, eb_id])
                    if key in seen:
                        continue
                    seen.add(key)

                    el_a = elems[ea_id]
                    el_b = elems[eb_id]

                    # Collect edge texts
                    ref1 = edge_text.get((min(ea_id, mid_id), max(ea_id, mid_id)), "")
                    ref2 = edge_text.get((min(mid_id, eb_id), max(mid_id, eb_id)), "")
                    bridge = f"{ref1} ... {ref2}".strip(" .")

                    pairs.append(CandidatePair(
                        pair_id=next_id(),
                        doc_id=doc_id,
                        element_a_id=ea_id,
                        element_b_id=eb_id,
                        element_a_type=ta,
                        element_b_type=tb,
                        pair_type=_make_pair_type(ta, tb),
                        hop_distance=2,
                        path=[ea_id, mid_id, eb_id],
                        quality_score=0.7,  # indirect = slightly lower
                        element_a=_element_detail(el_a),
                        element_b=_element_detail(el_b),
                        node_group=[_element_detail(el_a), _element_detail(el_b)],
                        edge_contexts=[],
                        hub_semantic_summary=_build_hub_summary(el_a, el_b, bridge),
                        strategy="2hop",
                        hub_metadata={"is_cross_doc": False, "strategy": "2hop",
                                      "bridge_element_id": mid_id},
                    ))

        return pairs

    # -- Strategy C: co-section pairs --------------------------------------

    def _section_pairs(
        self,
        doc_id: str,
        elems: Dict[str, Dict[str, Any]],
        next_id,
    ) -> List[CandidatePair]:
        """Pairs of different-modality elements that share a document section.

        Uses context_before / context_after to detect implicit section
        co-location by looking for shared section heading patterns.
        Also uses multimodal_pairs from the source data when available.
        """
        doc = self._docs.get(doc_id, {})
        existing_pairs = doc.get("multimodal_pairs", [])

        # Use existing multimodal_pairs that are 2/3-hop (already validated)
        pairs: List[CandidatePair] = []
        seen: Set[FrozenSet[str]] = set()

        for mp in existing_pairs:
            ea_id = mp.get("element_a_id", "")
            eb_id = mp.get("element_b_id", "")
            if ea_id not in elems or eb_id not in elems:
                continue
            ta = elems[ea_id].get("element_type", "")
            tb = elems[eb_id].get("element_type", "")
            if ta == tb:
                continue

            key = frozenset([ea_id, eb_id])
            if key in seen:
                continue
            seen.add(key)

            el_a = elems[ea_id]
            el_b = elems[eb_id]
            hop = mp.get("hop_distance", 2)
            rel = mp.get("relationship", "")
            q = 0.8 if rel == "direct_reference" else (0.6 if hop <= 2 else 0.4)

            pairs.append(CandidatePair(
                pair_id=next_id(),
                doc_id=doc_id,
                element_a_id=ea_id,
                element_b_id=eb_id,
                element_a_type=ta,
                element_b_type=tb,
                pair_type=_make_pair_type(ta, tb),
                hop_distance=hop,
                path=mp.get("path", [ea_id, eb_id]),
                quality_score=q,
                element_a=_element_detail(el_a),
                element_b=_element_detail(el_b),
                node_group=[_element_detail(el_a), _element_detail(el_b)],
                edge_contexts=[],
                hub_semantic_summary=_build_hub_summary(el_a, el_b),
                strategy="section",
                hub_metadata={"is_cross_doc": False, "strategy": "section",
                              "relationship": rel},
            ))

        return pairs

    # -- Strategy D: multi-hop chain discovery -----------------------------

    def _chain_pairs(
        self,
        doc_id: str,
        elems: Dict[str, Dict[str, Any]],
        next_id,
        min_hops: int = 2,
    ) -> List[CandidatePair]:
        """Pairs connected by multi-hop chains (variable length).

        Uses :class:`ChainFinder` to discover all maximal simple paths
        in the element graph, then emits pairs at chain endpoints with
        the full path and intermediate nodes in ``node_group``.
        """
        doc = self._docs.get(doc_id, {})
        finder = ChainFinder.from_doc(doc)
        chains = finder.find_endpoint_pairs(
            min_hops=min_hops,
            cross_modal_only=True,
        )

        pairs: List[CandidatePair] = []
        for chain in chains:
            ea_id, eb_id = chain.endpoints
            if ea_id not in elems or eb_id not in elems:
                continue
            el_a = elems[ea_id]
            el_b = elems[eb_id]
            ta = el_a.get("element_type", "")
            tb = el_b.get("element_type", "")

            # Build node_group with ALL elements in the chain path
            node_group = []
            for eid in chain.path:
                if eid in elems:
                    node_group.append(_element_detail(elems[eid]))

            # Collect edge contexts along the path
            edge_ctxs: List[Dict[str, Any]] = []
            edges = self._edges.get(doc_id, [])
            edge_lookup: Dict[FrozenSet[str], Dict[str, Any]] = {}
            for edge in edges:
                src = edge.get("source_id", "")
                tgt = edge.get("target_id", "")
                edge_lookup[frozenset([src, tgt])] = edge

            for i in range(len(chain.path) - 1):
                key = frozenset([chain.path[i], chain.path[i + 1]])
                if key in edge_lookup:
                    e = edge_lookup[key]
                    edge_ctxs.append({
                        "source": chain.path[i],
                        "target": chain.path[i + 1],
                        "ref_text": e.get("ref_text", ""),
                        "context_snippet": e.get("context_snippet", ""),
                    })

            summary = _build_hub_summary(el_a, el_b)
            modality_str = "→".join(chain.modality_sequence)

            pairs.append(CandidatePair(
                pair_id=next_id(),
                doc_id=doc_id,
                element_a_id=ea_id,
                element_b_id=eb_id,
                element_a_type=ta,
                element_b_type=tb,
                pair_type=_make_pair_type(ta, tb),
                hop_distance=chain.hop_count,
                path=list(chain.path),
                quality_score=chain.score,
                element_a=_element_detail(el_a),
                element_b=_element_detail(el_b),
                node_group=node_group,
                edge_contexts=edge_ctxs,
                hub_semantic_summary=f"{summary} | Chain: {modality_str}",
                strategy="chain",
                hub_metadata={
                    "is_cross_doc": False,
                    "strategy": "chain",
                    "chain_hops": chain.hop_count,
                    "modality_sequence": list(chain.modality_sequence),
                    "cross_modal_transitions": chain.cross_modal_transitions,
                    "unique_modalities": chain.unique_modalities,
                },
            ))

        return pairs

    # -- Statistics --------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics about the loaded documents."""
        n_docs = len(self._docs)
        n_elements = sum(len(e) for e in self._elements.values())
        n_edges = sum(len(e) for e in self._edges.values())
        type_counts: Dict[str, int] = defaultdict(int)
        for doc_elems in self._elements.values():
            for el in doc_elems.values():
                type_counts[el.get("element_type", "")] += 1
        return {
            "documents": n_docs,
            "elements": n_elements,
            "edges": n_edges,
            "type_distribution": dict(type_counts),
            "docs_with_multi_modal": sum(
                1 for doc_elems in self._elements.values()
                if len(set(el.get("element_type", "") for el in doc_elems.values())) >= 2
            ),
        }
