#!/usr/bin/env python3
"""
build_latex_cross_modal_links.py  —  Step 0 v3.2  (v2: precision-first rebuild)

Enrich MinerU cross-modal element pairs with LaTeX co-citation evidence.

Architecture principle
----------------------
MinerU is the PRIMARY source: all element data (image_path, caption, content,
context_before/after) comes from multimodal_elements.json.

LaTeX is the ENRICHMENT/REFERENCE layer: it provides `latex_bridge` —
the author-written sentence that *explains why* two elements are related,
extracted from the \\ref{} co-citation context in the .tex source.

Three discovery strategies (precision-first, descending order)
--------------------------------------------------------------
1. Direct cross-modal edges  [highest precision]:
       fig:roc --[ref]--> eq:fairness
   LatexRefEdge where source_type ≠ target_type and both are element types.
   Edge.context is the bridge text.

2. Proximity co-citation  [replaced section-level]:
       Two \ref{} calls in the SAME passage within CHAR_PROXIMITY_LIMIT characters
   of each other → cross-modal pair if they target different element types.
   This replaces the old "section-mediated" strategy which was too coarse
   (entire section = false positive factory).

3. Paragraph co-citation (from raw RefInstance list):
       context("...Figure 3 and Equation (1)...") shared by two refs
   Refs whose context strings share high token overlap are in the same
   paragraph.  If they target different label types → cross-modal pair.

Changes from v1
---------------
- F1: Section-level strategy REMOVED.  Replaced by char-distance proximity
  (CHAR_PROXIMITY_LIMIT = 1000 chars ≈ 1-2 natural paragraphs).  Immune to
  author line-wrapping style differences.
- F2: Caption matching now strips leading "Figure N:", "Table N:", "Eq. (N):"
  prefixes before computing Jaccard, preventing index-offset mismatches
  (e.g. MinerU "Table 4" vs LaTeX "Table 3") from inflating similarity.
  Numeric-suffix match now requires caption Jaccard ≥ 0.35 as second gate
  (unless caption is absent, in which case number-only match stands at
  reduced confidence 0.70).
- F3: quality_score uses exponential char-distance decay:
      score = min(conf_a, conf_b) × exp(-char_dist / DECAY_CONST)
  making score a continuous, meaningful confidence measure rather than a
  step-function with a single penalty multiplier.

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
      --min-match-conf 0.35
"""

import argparse
import json
import math
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

# Section-like label types that are NOT elements (kept only for filtering)
SECTION_TYPES = {"section", "appendix"}

# F1: Maximum character distance between two \ref{} calls to be considered
# "proximity co-citation" (≈ 1-2 natural paragraphs regardless of line wrapping)
CHAR_PROXIMITY_LIMIT = 1000

# F3: Exponential decay constant for char-distance quality scoring.
# At distance=0   → multiplier ≈ 1.0
# At distance=500 → multiplier ≈ 0.37  (e^-1)
# At distance=1000→ multiplier ≈ 0.14  (e^-2)
DECAY_CONST = 500.0

# Jaccard threshold for two ref contexts to be considered "same paragraph"
PARA_CONTEXT_JACCARD = 0.30   # loosened from 0.45 to recover near-neighbor co-cites

# F2: Jaccard threshold for *description text* (after prefix strip) in caption matching
CAPTION_DESC_THRESHOLD = 0.40  # raised from 0.25 — harder gate prevents offset mismatches

# If numeric suffix matches but caption is absent → accept at reduced confidence
NUMERIC_ONLY_CONF = 0.70       # down from 0.95; signals we couldn't verify via text

# G1: Hub de-duplication — max pairs any single element can participate in.
# Elements referenced N times across doc create O(N²) pairs; we keep only top-K
# by quality_score.  Applied as post-processing on the full pair list.
HUB_MAX_PAIRS_PER_ELEMENT = 3

# G2: Co-reference hard gate — for "proximity" strategy, the bridge text must
# explicitly reference BOTH elements via \\ref{} or the pairing is penalized.
# Pairs where bridge contains only ONE side get quality_score halved;
# pairs where bridge contains NEITHER side are dropped.
REQUIRE_COREF_IN_BRIDGE = True

# Regex to strip "Figure 3:", "Table 3:", "Fig. 3", "Eq. (3):", etc. from captions
# F2 core: must strip before Jaccard to avoid offset matching
_PREFIX_RE = re.compile(
    r"""^
    \s*
    (?:fig(?:ure)?|table|tab|eq(?:uation)?|alg(?:orithm)?)   # type word
    [\s.]*                                                      # optional punct
    [\(\[]?                                                     # optional open bracket
    \d+                                                         # number
    [\)\]]?                                                     # optional close bracket
    \s*[:\-–]?\s*                                               # optional colon/dash
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _clean_latex(text: str) -> str:
    """Strip common LaTeX commands for text comparison."""
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)   # \cmd{x} → x
    text = re.sub(r'\\[a-zA-Z]+\*?',          ' ',   text)   # \cmd → space
    text = re.sub(r'[${}\\~]',                ' ',   text)
    return re.sub(r'\s+', ' ', text).strip()


def _strip_prefix(caption: str) -> str:
    """F2: Remove leading 'Figure N:' / 'Table N:' / 'Eq. (N):' prefix."""
    return _PREFIX_RE.sub('', _clean_latex(caption)).strip()


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
# F3: Quality score with exponential distance decay
# ---------------------------------------------------------------------------

def _quality_score(conf_a: float, conf_b: float, char_dist: int) -> float:
    """
    F3: Continuous quality score combining match confidence and spatial proximity.

    score = min(conf_a, conf_b) × exp(-char_dist / DECAY_CONST)

    char_dist=0   (direct edge, no gap)  → no penalty
    char_dist=500 → ×0.37
    char_dist=1000→ ×0.14  (proximity boundary)
    """
    decay = math.exp(-char_dist / DECAY_CONST)
    return min(conf_a, conf_b) * decay


# ---------------------------------------------------------------------------
# G2: Co-reference bridge quality check
# ---------------------------------------------------------------------------

def _count_refs_in_bridge(bridge_text: str, label_a: str, label_b: str) -> int:
    """
    Count how many of {label_a, label_b} appear as \\ref{...} in bridge_text.

    Checks for:
      \\ref{label}  \\eqref{label}  \\autoref{label}  \\cref{label}
      \\hyperref[label]{...}  and bare label substrings as fallback.

    Returns 0, 1, or 2.
    """
    ref_pattern = re.compile(
        r'\\(?:ref|eqref|autoref|cref|hyperref)\s*[\[{]([^\]{}]+)[\]}]'
    )
    found_labels = set(ref_pattern.findall(bridge_text))
    count = 0
    if label_a in found_labels or label_a in bridge_text:
        count += 1
    if label_b in found_labels or label_b in bridge_text:
        count += 1
    return count


def _apply_coref_quality(
    quality_score: float,
    bridge_text: str,
    label_a: str,
    label_b: str,
    strategy: str,
) -> Optional[float]:
    """
    G2: Apply co-reference gate for proximity/paragraph pairs.

    - Both refs found  → no penalty, return quality_score as-is
    - One ref found    → halve quality_score (partial evidence)
    - Neither found    → return None (pair should be dropped)

    Direct edges already guarantee the ref is present (by construction),
    so this check only applies to proximity and paragraph strategies.
    """
    if strategy == "direct":
        return quality_score   # no gate needed

    n = _count_refs_in_bridge(bridge_text, label_a, label_b)
    if n >= 2:
        return quality_score
    if n == 1:
        return quality_score * 0.5
    # n == 0: bridge has no explicit reference to either element
    return None


# ---------------------------------------------------------------------------
# Label → MinerU element matching  (F2 applied here)
# ---------------------------------------------------------------------------

def _parse_number(text: str) -> Optional[int]:
    """Extract the first integer from a string like 'Equation 1', 'Eq. (3)'."""
    m = re.search(r'(\d+)', text or "")
    return int(m.group(1)) if m else None


def _match_label_to_element(
    label_key:     str,
    label_type:    str,        # LabelType.value string from LaTeX graph
    label_caption: str,
    elements:      dict,       # element_id → element dict (MinerU)
    threshold:     float = CAPTION_DESC_THRESHOLD,
) -> Optional[Tuple[str, float]]:
    """
    Return (element_id, confidence) or None.

    F2 changes:
    - Strip "Figure N:" / "Table N:" / "Eq. (N):" prefixes from BOTH sides
      before computing Jaccard, so index-offset mismatches don't inflate scores.
    - Numeric-suffix match now REQUIRES caption Jaccard ≥ 0.35 as second gate
      (unless the LaTeX caption is completely absent, where it falls back to
      NUMERIC_ONLY_CONF = 0.70 — lower confidence, no text verification).
    - Caption Jaccard threshold raised from 0.25 → CAPTION_DESC_THRESHOLD (0.40).
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

    # F2: strip prefixes from LaTeX caption once
    label_desc_tokens = _tokenize(_strip_prefix(label_caption or ""))

    # Strategy 1: numeric suffix in label key → match element number
    label_num = _parse_number(label_key)
    if label_num is not None:
        for eid, el in candidates.items():
            el_num = el.get("number")
            if el_num is None:
                el_num = _parse_number(el.get("label", ""))

            if el_num is not None and int(el_num) == label_num:
                # F2: double-gate — verify with caption text if available
                if not label_desc_tokens:
                    # No caption to verify → accept but at reduced confidence
                    return (eid, NUMERIC_ONLY_CONF)

                # F2: strip prefix from MinerU caption too, then compare
                el_desc_tokens = _tokenize(_strip_prefix(el.get("caption", "") or ""))
                if not el_desc_tokens:
                    return (eid, NUMERIC_ONLY_CONF)

                cap_jaccard = _jaccard(label_desc_tokens, el_desc_tokens)
                if cap_jaccard >= 0.35:
                    # Both number and description agree → high confidence
                    conf = min(0.95, 0.70 + cap_jaccard * 0.25)
                    return (eid, conf)
                # Number matches but description does NOT agree → likely offset error; skip
                # (do NOT return here — fall through to caption-only strategy)

    # Strategy 2: caption description Jaccard only (works for renamed labels)
    if not label_desc_tokens:
        return None

    best_eid, best_score = None, 0.0
    for eid, el in candidates.items():
        el_desc_tokens = _tokenize(_strip_prefix(el.get("caption", "") or ""))
        score = _jaccard(label_desc_tokens, el_desc_tokens)
        if score > best_score:
            best_score, best_eid = score, eid

    if best_score >= threshold:
        return (best_eid, best_score)
    return None


def _ordered_position_match(
    unmatched_labels: List[Tuple[str, dict]],   # (label_key, label_info)
    unmatched_elems:  List[Tuple[str, dict]],   # (element_id, element_dict)
) -> List[Tuple[str, str, float]]:
    """
    Fallback: match equation labels to formula elements by document order.

    Sort LaTeX labels by line_no, MinerU elements by (page_idx, position_idx),
    then match 1:1.  Only used when counts are reasonably close.

    Returns list of (label_key, element_id, confidence).
    """
    if not unmatched_labels or not unmatched_elems:
        return []

    n_labels = len(unmatched_labels)
    n_elems  = len(unmatched_elems)

    # Only attempt when counts are within 3× of each other
    ratio = max(n_labels, n_elems) / max(min(n_labels, n_elems), 1)
    if ratio > 3.0:
        return []

    # Sort by position
    sorted_labels = sorted(unmatched_labels, key=lambda x: x[1].get("line_no", 0))
    sorted_elems  = sorted(unmatched_elems,  key=lambda x: (
        x[1].get("page_idx", 0), x[1].get("position_idx", 0)
    ))

    # Confidence scales with how close the counts are
    base_conf = max(0.35, 0.60 - 0.10 * (ratio - 1))

    results = []
    for i in range(min(n_labels, n_elems)):
        lkey = sorted_labels[i][0]
        eid  = sorted_elems[i][0]
        results.append((lkey, eid, base_conf))

    return results


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
    char_dist:  int = 0,
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
            "strategy":       strategy,   # "direct" | "proximity" | "paragraph"
            "char_dist":      char_dist,  # F3: stored for audit / curriculum learning
        },
    }


# ---------------------------------------------------------------------------
# F1: Char-distance proximity discovery from LaTeX ref instances
# ---------------------------------------------------------------------------

def _find_proximity_pairs(
    latex_refs:   List[dict],           # raw RefInstance list from latex graph
    label_to_elem: Dict[str, Tuple[str, float]],  # label_key → (elem_id, conf)
    elements:     dict,                 # element_id → element dict
    latex_labels: dict,                 # label_key → LabelInfo dict (for char_pos)
) -> List[Tuple[str, str, float, float, float, str]]:
    """
    F1: Discover cross-modal pairs by char-distance proximity between \\ref{} calls.

    For each pair of ref instances that:
      (a) both resolve to known MinerU elements
      (b) target different element types forming a valid cross-modal pair
      (c) their char positions in the merged LaTeX content are ≤ CHAR_PROXIMITY_LIMIT apart

    Returns list of (elem_a_id, elem_b_id, conf_a, conf_b, char_dist, bridge_text).

    Uses `char_pos` field from RefInstance if present; falls back to line_no×80
    as a rough estimate (80 chars/line average).
    """
    # Build list of resolvable refs with position info
    resolved: List[Tuple[int, str, str, float, str]] = []
    # (char_pos, target_key, elem_id, conf, context)
    for ref in latex_refs:
        tkey  = ref.get("target_key", "")
        rtype = ref.get("ref_type", "")
        if rtype == "cite":
            continue
        match = label_to_elem.get(tkey)
        if not match:
            continue
        eid, conf = match
        etype = elements[eid]["element_type"]
        if etype not in {"figure", "table", "formula"}:
            continue

        # Estimate char position: prefer explicit char_pos, else line_no × 80
        char_pos = ref.get("char_pos")
        if char_pos is None:
            char_pos = ref.get("line_no", 0) * 80

        context = ref.get("context", "")
        resolved.append((int(char_pos), tkey, eid, conf, context))

    # Sort by char position
    resolved.sort(key=lambda x: x[0])

    results: List[Tuple[str, str, float, float, float, str]] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    n = len(resolved)
    for i in range(n):
        pos_i, tkey_i, eid_i, conf_i, ctx_i = resolved[i]
        etype_i = elements[eid_i]["element_type"]

        for j in range(i + 1, n):
            pos_j, tkey_j, eid_j, conf_j, ctx_j = resolved[j]

            char_dist = pos_j - pos_i
            if char_dist > CHAR_PROXIMITY_LIMIT:
                break   # list is sorted; no point looking further

            etype_j = elements[eid_j]["element_type"]

            # Must be cross-modal
            if frozenset([etype_i, etype_j]) not in CROSS_MODAL_SETS:
                continue
            # Must be different elements
            if eid_i == eid_j:
                continue

            pair_key = tuple(sorted([eid_i, eid_j]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Bridge text: use the longer of the two contexts
            bridge = ctx_i if len(ctx_i) >= len(ctx_j) else ctx_j

            results.append((eid_i, eid_j, conf_i, conf_j, float(char_dist), bridge))

    return results


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
        "ordered_match_rescued": 0,
        "by_strategy": defaultdict(int),
        "by_pair_type": defaultdict(int),
        "by_conf_bucket": defaultdict(int),  # low/med/high
        "by_coref_gate": defaultdict(int),   # G2: both_found/penalized/dropped
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
        # F2: prefix-stripped Jaccard + numeric double-gate applied inside
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

        # ------------------------------------------------------------------
        # Step A.2: Ordered position fallback for unmatched equation labels
        # ------------------------------------------------------------------
        matched_eids = {eid for eid, _ in label_to_elem.values()}
        eq_types_in_latex = {"equation", "algorithm"}

        unmatched_eq_labels = [
            (lkey, linfo) for lkey, linfo in latex_labels.items()
            if lkey not in label_to_elem
            and linfo.get("label_type") in eq_types_in_latex
        ]
        unmatched_formulas = [
            (eid, el) for eid, el in elements.items()
            if el.get("element_type") == "formula"
            and eid not in matched_eids
        ]

        if unmatched_eq_labels and unmatched_formulas:
            for lkey, eid, conf in _ordered_position_match(
                unmatched_eq_labels, unmatched_formulas
            ):
                label_to_elem[lkey] = (eid, conf)
                stats["label_match_success"] += 1
                stats["ordered_match_rescued"] += 1

        # Track which element pairs we've already added (order-independent)
        seen: Set[Tuple[str, str]] = set()

        def _register_pair(pair: dict) -> bool:
            key = tuple(sorted([pair["element_a_id"], pair["element_b_id"]]))
            if key in seen:
                return False
            if pair["element_a_id"] == pair["element_b_id"]:
                return False

            # G2: Co-reference hard gate for proximity / paragraph strategies
            if REQUIRE_COREF_IN_BRIDGE:
                bridge = pair["latex_bridge"]
                new_qs = _apply_coref_quality(
                    pair["quality_score"],
                    bridge["bridge_text"],
                    bridge["label_a"],
                    bridge["label_b"],
                    bridge["strategy"],
                )
                if new_qs is None:
                    # Neither element referenced in bridge → drop
                    stats["by_coref_gate"]["dropped"] += 1
                    return False
                elif new_qs < pair["quality_score"]:
                    # Only one ref found → penalize
                    stats["by_coref_gate"]["penalized"] += 1
                    pair = {**pair, "quality_score": round(new_qs, 3)}
                    pair["latex_bridge"] = {
                        **pair["latex_bridge"], "coref_penalty": True
                    }
                else:
                    stats["by_coref_gate"]["both_found"] += 1

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
        # Strategy 1: Direct cross-modal edges  [highest precision]
        #   source_type ∈ {figure,table,equation,...}
        #   target_type ∈ {figure,table,equation,...}
        #   source_type ≠ target_type (cross-modal)
        #   char_dist = 0 (explicit ref in text), F3 decay = 1.0
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

            # F3: direct edge → char_dist=0, no decay penalty
            _register_pair(_make_pair(
                _next_id(), doc_id,
                src_eid, tgt_eid,
                elements[src_eid], elements[tgt_eid],
                bridge, src_label, tgt_label,
                src_conf, tgt_conf,
                strategy="direct",
                quality_score=_quality_score(src_conf, tgt_conf, 0),
                char_dist=0,
            ))

        # ------------------------------------------------------------------
        # Strategy 2 (F1): Proximity co-citation
        #   Replaces section-mediated strategy.
        #   Two \ref{} within CHAR_PROXIMITY_LIMIT chars of each other
        #   targeting different element modalities → cross-modal pair.
        #   F3 decay applied: score decays with char distance.
        # ------------------------------------------------------------------
        for eid_a, eid_b, conf_a, conf_b, char_dist, bridge in _find_proximity_pairs(
            latex_refs, label_to_elem, elements, latex_labels
        ):
            # Determine labels from label_to_elem reverse map (best effort)
            def _eid_to_label(eid: str) -> str:
                for lk, (le, _) in label_to_elem.items():
                    if le == eid:
                        return lk
                return eid  # fallback: use element id

            la = _eid_to_label(eid_a)
            lb = _eid_to_label(eid_b)

            _register_pair(_make_pair(
                _next_id(), doc_id,
                eid_a, eid_b,
                elements[eid_a], elements[eid_b],
                bridge, la, lb,
                conf_a, conf_b,
                strategy="proximity",
                quality_score=_quality_score(conf_a, conf_b, int(char_dist)),
                char_dist=int(char_dist),
            ))

        # ------------------------------------------------------------------
        # Strategy 3: Paragraph-level co-citation (from raw RefInstance list)
        #   Refs whose context strings share high token overlap are in the same
        #   paragraph.  Jaccard threshold loosened to 0.30 to recover more
        #   near-neighbor co-citations that proximity strategy might miss.
        # ------------------------------------------------------------------
        resolvable_refs: List[Tuple[str, str, str, float, str]] = []
        for ref in latex_refs:
            tkey  = ref.get("target_key", "")
            rtype = ref.get("ref_type", "")
            if rtype == "cite":
                continue
            match = label_to_elem.get(tkey)
            if not match:
                continue
            eid, conf = match
            etype = elements[eid]["element_type"]
            context = ref.get("context", "")
            resolvable_refs.append((tkey, etype, context, conf, eid))

        # Cluster by context Jaccard (≥ PARA_CONTEXT_JACCARD = 0.30)
        clustered: List[List[int]] = []
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
            by_etype: Dict[str, List[Tuple[str, float, str]]] = defaultdict(list)
            for idx in group:
                tkey, etype, ctx, conf, eid = resolvable_refs[idx]
                by_etype[etype].append((eid, conf, ctx))

            etypes = list(by_etype.keys())
            for ta, tb in combinations(etypes, 2):
                if frozenset([ta, tb]) not in CROSS_MODAL_SETS:
                    continue
                best_a = max(by_etype[ta], key=lambda x: x[1])
                best_b = max(by_etype[tb], key=lambda x: x[1])
                eid_a, conf_a, ctx_a = best_a
                eid_b, conf_b, ctx_b = best_b

                bridge = ctx_a if len(ctx_a) >= len(ctx_b) else ctx_b

                # F3: paragraph = context overlap, treat as very short distance (≈100)
                # to reflect "same sentence / adjacent sentence" proximity
                _register_pair(_make_pair(
                    _next_id(), doc_id,
                    eid_a, eid_b,
                    elements[eid_a], elements[eid_b],
                    bridge, f"(para-ref:{ta})", f"(para-ref:{tb})",
                    conf_a, conf_b,
                    strategy="paragraph",
                    quality_score=_quality_score(conf_a, conf_b, 100),
                    char_dist=100,
                ))

    # -----------------------------------------------------------------------
    # G1: Hub de-duplication — post-processing pass
    # -----------------------------------------------------------------------
    # Count how many pairs each element already participates in.
    # Sort all pairs by quality_score descending; greedily accept pairs
    # only if both elements have not yet hit HUB_MAX_PAIRS_PER_ELEMENT.
    output_pairs.sort(key=lambda p: p["quality_score"], reverse=True)
    elem_pair_count: Dict[str, int] = defaultdict(int)
    hub_filtered: List[dict] = []
    hub_dropped = 0
    for pair in output_pairs:
        ea = pair["element_a_id"]
        eb = pair["element_b_id"]
        if (elem_pair_count[ea] < HUB_MAX_PAIRS_PER_ELEMENT and
                elem_pair_count[eb] < HUB_MAX_PAIRS_PER_ELEMENT):
            hub_filtered.append(pair)
            elem_pair_count[ea] += 1
            elem_pair_count[eb] += 1
        else:
            hub_dropped += 1
    output_pairs = hub_filtered

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    by_pair_type   = dict(stats["by_pair_type"])
    by_strategy    = dict(stats["by_strategy"])
    by_conf        = dict(stats["by_conf_bucket"])
    by_coref       = dict(stats["by_coref_gate"])
    label_rate     = (
        stats["label_match_success"] / stats["label_match_attempts"]
        if stats["label_match_attempts"] else 0.0
    )

    output = {
        "metadata": {
            "source_elements":       str(args.elements),
            "source_latex":          str(args.latex_graph),
            "min_match_conf":        args.min_match_conf,
            "para_jaccard":          PARA_CONTEXT_JACCARD,
            "char_proximity_limit":  CHAR_PROXIMITY_LIMIT,
            "decay_const":           DECAY_CONST,
            "caption_desc_threshold": CAPTION_DESC_THRESHOLD,
            "version":               "v3-precision",
            "hub_max_pairs_per_element": HUB_MAX_PAIRS_PER_ELEMENT,
            "require_coref_in_bridge":   REQUIRE_COREF_IN_BRIDGE,
            "description": (
                "Cross-modal pairs discovered via LaTeX \\ref{} co-citation. "
                "v2: (F1) section-level replaced by char-distance proximity "
                f"(≤{CHAR_PROXIMITY_LIMIT} chars); "
                "(F2) caption prefix-stripped before Jaccard + numeric double-gate; "
                "(F3) quality_score = min(conf_a,conf_b) × exp(-dist/500). "
                "v3: (G1) hub de-dup: max 3 pairs/element by quality_score; "
                "(G2) coref gate: proximity bridge must contain both \\ref{} → "
                "drop if 0 refs, halve score if 1 ref."
            ),
        },
        "summary": {
            "total_pairs":            len(output_pairs),
            "docs_with_latex":        stats["docs_with_latex"],
            "label_match_rate":       round(label_rate, 3),
            "label_match_success":    stats["label_match_success"],
            "label_match_attempts":   stats["label_match_attempts"],
            "ordered_match_rescued":  stats["ordered_match_rescued"],
            "by_strategy":            by_strategy,
            "by_pair_type":           by_pair_type,
            "by_conf_bucket":         by_conf,
            "g1_hub_dropped":         hub_dropped,
            "g2_coref_gate":          by_coref,
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
    print(f"\n{'='*60}")
    print(f"  Total pairs         : {len(output_pairs)}")
    print(f"  Docs with LaTeX     : {stats['docs_with_latex']}")
    print(f"  Label match rate    : {label_rate:.1%}  "
          f"({stats['label_match_success']}/{stats['label_match_attempts']})")
    print(f"  Eq ordered rescue   : {stats['ordered_match_rescued']}")
    print(f"  By strategy         : {by_strategy}")
    print(f"  By pair type        : {by_pair_type}")
    print(f"  By conf bucket      : {by_conf}")
    print(f"  G1 hub dropped      : {hub_dropped}  (max {HUB_MAX_PAIRS_PER_ELEMENT} pairs/element)")
    print(f"  G2 coref gate       : {by_coref}")
    print(f"  Output → {args.output}")
    print(f"  Report → {report_path}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich cross-modal element pairs with LaTeX \\ref{} co-citation evidence. "
            "v2: precision-first (F1 proximity, F2 caption double-gate, F3 exp decay)."
        )
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
        "--min-match-conf", type=float, default=CAPTION_DESC_THRESHOLD,
        help=f"Min caption Jaccard for label→element match (default {CAPTION_DESC_THRESHOLD})"
    )
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
