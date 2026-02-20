#!/usr/bin/env python3
"""
build_latex_cross_modal_links.py  —  Step 0 v3.2

Enrich MinerU cross-modal element pairs with LaTeX co-citation evidence.

Architecture principle
----------------------
MinerU is the PRIMARY source: all element data (image_path, caption, content,
context_before/after) comes from multimodal_elements.json.

LaTeX is the ENRICHMENT/REFERENCE layer: it provides `latex_bridge` —
the author-written sentence that *explains why* two elements are related,
extracted from the \ref{} co-citation context in the .tex source.

Three discovery strategies (in descending precision)
------------------------------------------------------
1. Direct cross-modal edges:
       fig:roc --[ref]--> eq:fairness
   LatexRefEdge where source_type ≠ target_type and both are element types.
   Edge.context is the bridge text.

2. Section-mediated co-citation:
       sec:experiments --[ref]--> fig:roc
       sec:experiments --[ref]--> tab:results
   When a section references ≥2 elements of different modalities, those
   elements are implicitly co-cited.  Bridge text = the union of ref contexts
   from the same section.

3. Paragraph co-citation (from raw RefInstance list):
       context("...Figure 3 and Equation (1)...") shared by two refs
   Refs whose context strings share high token overlap are in the same
   paragraph.  If they target different label types → cross-modal pair.

Outputs
-------
  data/latex_cross_modal_pairs.json   — enriched pairs (see schema below)
  data/latex_cross_modal_report.json  — statistics

Output schema is backward-compatible with multihop_l1_candidates.json
(used by generate_multihop_l1_queries.py) plus an optional `latex_bridge`
field.  Downstream code should treat `latex_bridge` as best-effort: it may
be absent if LaTeX matching failed for that pair.

Usage
-----
  python scripts/build_latex_cross_modal_links.py
  python scripts/build_latex_cross_modal_links.py \\
      --elements data/multimodal_elements.json \\
      --latex-graph data/latex_reference_graph.json \\
      --output data/latex_cross_modal_pairs.json \\
      --min-match-conf 0.25
"""

import argparse
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LaTeX label types that map to MinerU element types
LABEL_TO_ELEMENT: Dict[str, str] = {
    "figure":    "figure",
    "table":     "table",
    "equation":  "formula",
    "algorithm": "formula",   # treat algorithm blocks like formulas
}

# Cross-modal pairs we care about (order-independent)
CROSS_MODAL_SETS: Set[frozenset] = {
    frozenset(["figure", "table"]),
    frozenset(["figure", "formula"]),
    frozenset(["table",  "formula"]),
}

# Section-like label types that mediate co-citations but are not elements
SECTION_TYPES = {"section", "appendix"}

# Jaccard threshold for two ref contexts to be considered "same paragraph"
PARA_CONTEXT_JACCARD = 0.45

# Jaccard threshold for LaTeX label caption → MinerU element caption matching
DEFAULT_CAPTION_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _clean_latex(text: str) -> str:
    """Strip common LaTeX commands for text comparison."""
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)   # \cmd{x} → x
    text = re.sub(r'\\[a-zA-Z]+\*?',          ' ',   text)   # \cmd → space
    text = re.sub(r'[${}\\~]',                ' ',   text)
    return re.sub(r'\s+', ' ', text).strip()


def _tokenize(text: str) -> Set[str]:
    """Return content word tokens (length ≥ 3)."""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return {t for t in text.split() if len(t) >= 3}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _normalize_pair_type(ta: str, tb: str) -> str:
    """Canonical 'typeA+typeB' string (alphabetical)."""
    return "+".join(sorted([ta, tb]))


# ---------------------------------------------------------------------------
# Label → MinerU element matching
# ---------------------------------------------------------------------------

def _match_label_to_element(
    label_key:    str,
    label_type:   str,        # LabelType.value string from LaTeX graph
    label_caption: str,
    elements:     dict,       # element_id → element dict (MinerU)
    threshold:    float = DEFAULT_CAPTION_THRESHOLD,
) -> Optional[Tuple[str, float]]:
    """
    Return (element_id, confidence) or None.

    Strategy 1 — number extraction from label key:
        fig3 / fig:3 / fig_3 → figure with number == 3  (conf 0.95)
    Strategy 2 — caption Jaccard:
        clean LaTeX from both captions, compute token Jaccard  (conf = score)
    """
    target_type = LABEL_TO_ELEMENT.get(label_type)
    if target_type is None:
        return None

    candidates = {
        eid: el for eid, el in elements.items()
        if el.get("element_type") == target_type
    }
    if not candidates:
        return None

    # Strategy 1: numeric suffix in label key
    num_match = re.search(r'(\d+)', label_key)
    if num_match:
        num = int(num_match.group(1))
        for eid, el in candidates.items():
            el_num = el.get("number")
            if el_num is not None and int(el_num) == num:
                return (eid, 0.95)

    # Strategy 2: caption text Jaccard
    label_tokens = _tokenize(_clean_latex(label_caption or ""))
    if not label_tokens:
        return None

    best_eid, best_score = None, 0.0
    for eid, el in candidates.items():
        el_tokens = _tokenize(_clean_latex(el.get("caption", "") or ""))
        score = _jaccard(label_tokens, el_tokens)
        if score > best_score:
            best_score, best_eid = score, eid

    if best_score >= threshold:
        return (best_eid, best_score)
    return None


# ---------------------------------------------------------------------------
# Pair construction helpers
# ---------------------------------------------------------------------------

def _make_pair(
    pair_id:    str,
    doc_id:     str,
    elem_a_id:  str,
    elem_b_id:  str,
    elem_a:     dict,
    elem_b:     dict,
    bridge_text: str,
    label_a:    str,
    label_b:    str,
    conf_a:     float,
    conf_b:     float,
    strategy:   str,
    quality_score: float,
) -> dict:
    """Build a pair dict in the multihop_l1_candidates.json schema."""
    type_a = elem_a["element_type"]
    type_b = elem_b["element_type"]
    return {
        "pair_id":             pair_id,
        "doc_id":              doc_id,
        "element_a_id":        elem_a_id,
        "element_b_id":        elem_b_id,
        "element_a_type":      type_a,
        "element_b_type":      type_b,
        "pair_type":           _normalize_pair_type(type_a, type_b),
        "hop_distance":        1,
        "path":                [elem_a_id, elem_b_id],
        "quality_score":       round(quality_score, 3),
        "overlap_with_existing_l1": False,
        "element_a": {
            "element_id":    elem_a_id,
            "element_type":  type_a,
            "caption":       (elem_a.get("caption")       or "")[:500],
            "content":       (elem_a.get("content")       or "")[:1000],
            "image_path":    elem_a.get("image_path"),
            "context_before":(elem_a.get("context_before") or "")[:500],
            "context_after": (elem_a.get("context_after")  or "")[:500],
        },
        "element_b": {
            "element_id":    elem_b_id,
            "element_type":  type_b,
            "caption":       (elem_b.get("caption")       or "")[:500],
            "content":       (elem_b.get("content")       or "")[:1000],
            "image_path":    elem_b.get("image_path"),
            "context_before":(elem_b.get("context_before") or "")[:500],
            "context_after": (elem_b.get("context_after")  or "")[:500],
        },
        # MinerU-style edge_contexts field (for compatibility)
        "edge_contexts": [{
            "source":          elem_a_id,
            "target":          elem_b_id,
            "ref_text":        f"\\ref{{{label_b}}}",
            "context_snippet": bridge_text[:300],
        }],
        # LaTeX enrichment (the key addition)
        "latex_bridge": {
            "bridge_text":    bridge_text,
            "label_a":        label_a,
            "label_b":        label_b,
            "match_conf_a":   round(conf_a, 3),
            "match_conf_b":   round(conf_b, 3),
            "strategy":       strategy,   # "direct" | "section" | "paragraph"
        },
    }


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build(args) -> None:
    print(f"[1/4] Loading MinerU elements   → {args.elements}")
    with open(args.elements, encoding="utf-8") as f:
        mm_data = json.load(f)
    mm_docs = mm_data.get("documents", {})

    print(f"[2/4] Loading LaTeX ref graph   → {args.latex_graph}")
    with open(args.latex_graph, encoding="utf-8") as f:
        latex_data = json.load(f)
    latex_docs = latex_data.get("documents", {})

    stats = {
        "docs_with_latex": 0,
        "label_match_attempts": 0,
        "label_match_success": 0,
        "by_strategy": defaultdict(int),
        "by_pair_type": defaultdict(int),
        "by_conf_bucket": defaultdict(int),  # low/med/high
    }

    output_pairs: List[dict] = []

    for doc_id, mm_doc in mm_docs.items():
        if doc_id not in latex_docs:
            continue
        stats["docs_with_latex"] += 1

        elements = mm_doc.get("elements", {})
        if not elements:
            continue

        latex_doc    = latex_docs[doc_id]
        latex_labels = latex_doc.get("labels", {})   # key → LabelInfo dict
        latex_edges  = latex_doc.get("edges",  [])   # list of LatexRefEdge dicts
        latex_refs   = latex_doc.get("refs",   [])   # list of RefInstance dicts

        # ------------------------------------------------------------------
        # Step A: build label → MinerU element mapping for this document
        # ------------------------------------------------------------------
        label_to_elem: Dict[str, Tuple[str, float]] = {}  # label_key → (element_id, conf)
        for lkey, linfo in latex_labels.items():
            ltype    = linfo.get("label_type", "unknown")
            lcaption = linfo.get("caption", "") or ""
            stats["label_match_attempts"] += 1
            result = _match_label_to_element(
                lkey, ltype, lcaption, elements, args.min_match_conf
            )
            if result:
                label_to_elem[lkey] = result
                stats["label_match_success"] += 1

        # Track which element pairs we've already added (order-independent)
        seen: Set[Tuple[str, str]] = set()

        def _register_pair(pair: dict) -> bool:
            key = tuple(sorted([pair["element_a_id"], pair["element_b_id"]]))
            if key in seen:
                return False
            if pair["element_a_id"] == pair["element_b_id"]:
                return False
            seen.add(key)
            output_pairs.append(pair)
            stats["by_strategy"][pair["latex_bridge"]["strategy"]] += 1
            stats["by_pair_type"][pair["pair_type"]] += 1
            conf = min(pair["latex_bridge"]["match_conf_a"],
                       pair["latex_bridge"]["match_conf_b"])
            bucket = "high" if conf >= 0.7 else ("med" if conf >= 0.4 else "low")
            stats["by_conf_bucket"][bucket] += 1
            return True

        pair_counter = [len(output_pairs)]  # mutable counter for pair_id

        def _next_id() -> str:
            pair_counter[0] += 1
            return f"{doc_id}_xl_{pair_counter[0]:04d}"

        # ------------------------------------------------------------------
        # Strategy 1: Direct cross-modal edges
        #   source_type ∈ {figure,table,equation,...}
        #   target_type ∈ {figure,table,equation,...}
        #   source_type ≠ target_type (cross-modal)
        # ------------------------------------------------------------------
        for edge in latex_edges:
            src_ltype = edge.get("source_type", "")
            tgt_ltype = edge.get("target_type", "")
            src_etype = LABEL_TO_ELEMENT.get(src_ltype)
            tgt_etype = LABEL_TO_ELEMENT.get(tgt_ltype)

            if not src_etype or not tgt_etype:
                continue
            if frozenset([src_etype, tgt_etype]) not in CROSS_MODAL_SETS:
                continue

            src_label = edge.get("source_label", "")
            tgt_label = edge.get("target_label", "")
            bridge    = edge.get("context", "")

            src_match = label_to_elem.get(src_label)
            tgt_match = label_to_elem.get(tgt_label)
            if not src_match or not tgt_match:
                continue

            src_eid, src_conf = src_match
            tgt_eid, tgt_conf = tgt_match

            _register_pair(_make_pair(
                _next_id(), doc_id,
                src_eid, tgt_eid,
                elements[src_eid], elements[tgt_eid],
                bridge, src_label, tgt_label,
                src_conf, tgt_conf,
                strategy="direct",
                quality_score=min(src_conf, tgt_conf),
            ))

        # ------------------------------------------------------------------
        # Strategy 2: Section-mediated co-citation
        #   A section node has edges to fig:X AND tab:Y (or eq:Z etc.)
        #   → the two element targets are implicitly co-cited
        # ------------------------------------------------------------------
        # Collect: section_label → {etype: [(elem_id, conf, bridge_text)]}
        sec_targets: Dict[str, Dict[str, List[Tuple[str, float, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for edge in latex_edges:
            src_ltype = edge.get("source_type", "")
            tgt_ltype = edge.get("target_type", "")
            if src_ltype not in SECTION_TYPES:
                continue
            tgt_etype = LABEL_TO_ELEMENT.get(tgt_ltype)
            if not tgt_etype:
                continue
            tgt_label = edge.get("target_label", "")
            tgt_match = label_to_elem.get(tgt_label)
            if not tgt_match:
                continue
            tgt_eid, tgt_conf = tgt_match
            bridge = edge.get("context", "")
            sec_targets[edge["source_label"]][tgt_etype].append(
                (tgt_eid, tgt_conf, bridge)
            )

        for sec_label, by_type in sec_targets.items():
            etypes = list(by_type.keys())
            for ta, tb in combinations(etypes, 2):
                if frozenset([ta, tb]) not in CROSS_MODAL_SETS:
                    continue
                for (eid_a, conf_a, bridge_a) in by_type[ta]:
                    for (eid_b, conf_b, bridge_b) in by_type[tb]:
                        # Merge the two context snippets as bridge evidence
                        bridge = " … ".join(
                            filter(None, [bridge_a, bridge_b])
                        )
                        _register_pair(_make_pair(
                            _next_id(), doc_id,
                            eid_a, eid_b,
                            elements[eid_a], elements[eid_b],
                            bridge, f"(via {sec_label})", f"(via {sec_label})",
                            conf_a, conf_b,
                            strategy="section",
                            quality_score=min(conf_a, conf_b) * 0.8,  # slight penalty
                        ))

        # ------------------------------------------------------------------
        # Strategy 3: Paragraph-level co-citation
        #   Two RefInstance objects whose context strings share high token
        #   overlap are in the same paragraph.  If they target different
        #   element types → cross-modal pair.
        # ------------------------------------------------------------------
        # Group refs by context fingerprint
        resolvable_refs: List[Tuple[str, str, str, float]] = []  # (target_key, etype, context, conf)
        for ref in latex_refs:
            tkey  = ref.get("target_key", "")
            rtype = ref.get("ref_type", "")
            if rtype == "cite":
                continue   # skip bibliography citations
            match = label_to_elem.get(tkey)
            if not match:
                continue
            eid, conf = match
            etype = elements[eid]["element_type"]
            context = ref.get("context", "")
            resolvable_refs.append((tkey, etype, context, conf, eid))

        # Cluster by context Jaccard
        clustered: List[List[int]] = []   # list of groups (indices into resolvable_refs)
        assigned  = [False] * len(resolvable_refs)

        for i, (_, _, ctx_i, _, _) in enumerate(resolvable_refs):
            if assigned[i]:
                continue
            group = [i]
            assigned[i] = True
            tok_i = _tokenize(ctx_i)
            for j in range(i + 1, len(resolvable_refs)):
                if assigned[j]:
                    continue
                tok_j = _tokenize(resolvable_refs[j][2])
                if _jaccard(tok_i, tok_j) >= PARA_CONTEXT_JACCARD:
                    group.append(j)
                    assigned[j] = True
            clustered.append(group)

        for group in clustered:
            if len(group) < 2:
                continue
            # Collect by element type within this paragraph cluster
            by_etype: Dict[str, List[Tuple[str, float, str]]] = defaultdict(list)
            for idx in group:
                tkey, etype, ctx, conf, eid = resolvable_refs[idx]
                by_etype[etype].append((eid, conf, ctx))

            etypes = list(by_etype.keys())
            for ta, tb in combinations(etypes, 2):
                if frozenset([ta, tb]) not in CROSS_MODAL_SETS:
                    continue
                # Take the highest-confidence match from each type
                best_a = max(by_etype[ta], key=lambda x: x[1])
                best_b = max(by_etype[tb], key=lambda x: x[1])
                eid_a, conf_a, ctx_a = best_a
                eid_b, conf_b, ctx_b = best_b

                # Bridge text = the shared context (use longer one)
                bridge = ctx_a if len(ctx_a) >= len(ctx_b) else ctx_b

                _register_pair(_make_pair(
                    _next_id(), doc_id,
                    eid_a, eid_b,
                    elements[eid_a], elements[eid_b],
                    bridge, f"(para-ref:{ta})", f"(para-ref:{tb})",
                    conf_a, conf_b,
                    strategy="paragraph",
                    quality_score=min(conf_a, conf_b) * 0.65,  # lower precision
                ))

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    by_pair_type   = dict(stats["by_pair_type"])
    by_strategy    = dict(stats["by_strategy"])
    by_conf        = dict(stats["by_conf_bucket"])
    label_rate     = (
        stats["label_match_success"] / stats["label_match_attempts"]
        if stats["label_match_attempts"] else 0.0
    )

    output = {
        "metadata": {
            "source_elements":  str(args.elements),
            "source_latex":     str(args.latex_graph),
            "min_match_conf":   args.min_match_conf,
            "para_jaccard":     PARA_CONTEXT_JACCARD,
            "description": (
                "Cross-modal pairs discovered via LaTeX \\ref{} co-citation. "
                "Primary element data is from MinerU (multimodal_elements.json). "
                "latex_bridge provides author-written evidence text."
            ),
        },
        "summary": {
            "total_pairs":          len(output_pairs),
            "docs_with_latex":      stats["docs_with_latex"],
            "label_match_rate":     round(label_rate, 3),
            "label_match_success":  stats["label_match_success"],
            "label_match_attempts": stats["label_match_attempts"],
            "by_strategy":          by_strategy,
            "by_pair_type":         by_pair_type,
            "by_conf_bucket":       by_conf,
        },
        "pairs": output_pairs,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    report_path = args.output.replace(".json", "_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(output["summary"] | {"metadata": output["metadata"]}, f, indent=2)

    # Pretty summary
    print(f"\n{'='*55}")
    print(f"  Total pairs        : {len(output_pairs)}")
    print(f"  Docs with LaTeX    : {stats['docs_with_latex']}")
    print(f"  Label match rate   : {label_rate:.1%}  "
          f"({stats['label_match_success']}/{stats['label_match_attempts']})")
    print(f"  By strategy        : {by_strategy}")
    print(f"  By pair type       : {by_pair_type}")
    print(f"  By conf bucket     : {by_conf}")
    print(f"  Output → {args.output}")
    print(f"  Report → {report_path}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich cross-modal element pairs with LaTeX \ref{} co-citation evidence."
    )
    parser.add_argument(
        "--elements", default="data/multimodal_elements.json",
        help="MinerU multimodal elements JSON (primary source)"
    )
    parser.add_argument(
        "--latex-graph", default="data/latex_reference_graph.json",
        help="LaTeX reference graph JSON (enrichment source)"
    )
    parser.add_argument(
        "--output", default="data/latex_cross_modal_pairs.json",
        help="Output path for enriched cross-modal pairs"
    )
    parser.add_argument(
        "--min-match-conf", type=float, default=DEFAULT_CAPTION_THRESHOLD,
        help=f"Min caption Jaccard to accept a label→element match (default {DEFAULT_CAPTION_THRESHOLD})"
    )
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
