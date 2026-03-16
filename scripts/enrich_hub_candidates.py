#!/usr/bin/env python3
"""Enrich hub multi-hop candidates with MinerU element details.

Converts latex_hub_multihop_candidates.json (topology output) into the same
format as multihop_l1_candidates.json so it can be fed directly to
generate_multihop_l1_queries.py.

The key task: map LaTeX node IDs (e.g. "1306.5204::el::fig:histograms")
back to MinerU element IDs (e.g. "1306.5204_figure_2") and attach full
element details (caption, image_path, context_before/after).

Supports optional MoDora-style enriched elements (--enriched-elements) to
inject enriched_title / enriched_content / enriched_metadata and generate
hub_semantic_summary fields for each candidate pair.

Usage:
    python scripts/enrich_hub_candidates.py \
        --hub-candidates data/latex_hub_multihop_candidates.json \
        --elements data/multimodal_elements.json \
        --latex-graph data/latex_reference_graph.json \
        --output data/hub_candidates_enriched.json \
        [--enriched-elements data/multimodal_elements_enriched.json] \
        [--limit 50]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ─── Label type normalization (from analyze_latex_graph_topology.py) ──────────

_LABEL_TYPE_MAP = {
    "figure": "figure",
    "fig": "figure",
    "subfigure": "figure",
    "table": "table",
    "tab": "table",
    "equation": "equation",
    "eq": "equation",
    "align": "equation",
    "eqnarray": "equation",
    "formula": "equation",
}

ELEMENT_MODALITIES = {"figure", "table", "equation"}
MINERU_MODAL_MAP = {"figure": "figure", "table": "table", "equation": "formula"}


def normalize_label_type(raw: str) -> str:
    return _LABEL_TYPE_MAP.get(raw.lower().strip(), raw.lower().strip())


def tokenize(text: str) -> Set[str]:
    return set(re.findall(r"\w+", text.lower()))


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def parse_number(s: str) -> Optional[int]:
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


# ─── Build MinerU index ──────────────────────────────────────────────────────

def build_mm_index(mm_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build lookup indices from multimodal_elements.json."""
    by_doc: Dict[str, Dict] = {}
    all_elements: Dict[str, Dict] = {}  # element_id → element dict

    for doc_key, doc in mm_data["documents"].items():
        doc_id = doc["doc_id"]
        by_number: Dict[str, Dict[int, str]] = defaultdict(dict)  # type → {number: eid}
        by_caption: Dict[str, List[Tuple[str, Set[str]]]] = defaultdict(list)

        elements = doc.get("elements", {})
        if isinstance(elements, dict):
            elem_list = elements.values()
        else:
            elem_list = elements

        for el in elem_list:
            eid = el["element_id"]
            etype = el["element_type"]
            all_elements[eid] = el

            num = el.get("number")
            if num is not None:
                num_int = _safe_int(num)
                if num_int != 99999:
                    by_number[etype][num_int] = eid

            cap = el.get("caption", "") or ""
            cap_tokens = tokenize(cap)
            if cap_tokens:
                by_caption[etype].append((eid, cap_tokens))

        by_doc[doc_id] = {
            "by_number": dict(by_number),
            "by_caption": dict(by_caption),
        }

    return {"by_doc": by_doc, "all_elements": all_elements}


def map_label_to_element(
    doc_id: str,
    label_key: str,
    label_info: Dict[str, Any],
    mm_index: Dict[str, Any],
) -> Optional[str]:
    """Map a LaTeX label to a MinerU element_id (same logic as topology script)."""
    doc_idx = mm_index["by_doc"].get(doc_id)
    if not doc_idx:
        return None

    ltype = normalize_label_type(str(label_info.get("label_type", "")))
    if ltype not in ELEMENT_MODALITIES:
        return None
    mm_type = MINERU_MODAL_MAP.get(ltype, ltype)

    # 1. number from label key
    number = parse_number(label_key)
    if number is not None:
        eid = doc_idx["by_number"].get(mm_type, {}).get(number)
        if eid:
            return eid
        for part in re.split(r"[_\-:]+", label_key):
            n = parse_number(part)
            if n is not None:
                eid = doc_idx["by_number"].get(mm_type, {}).get(n)
                if eid:
                    return eid

    # 2. caption Jaccard
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


# ─── Build node→element mapping ─────────────────────────────────────────────

def _safe_int(val: Any) -> int:
    """Convert a value to int for sorting; non-numeric strings get 99999."""
    if val is None:
        return 99999
    try:
        return int(val)
    except (ValueError, TypeError):
        return 99999


def build_sequential_mapping(
    doc_id: str,
    labels: Dict[str, Any],
    mm_elements: Dict[str, Any],
    exclude_node_ids: Optional[Set[str]] = None,
    exclude_eids: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """Sequential order matching: sort labels by line_no, elements by number/position.

    For each modality type, if we have N LaTeX labels and M MinerU elements,
    match them 1:1 in order (up to min(N,M)).

    Pre-excludes already-mapped node_ids and element_ids so that the zip
    alignment only operates on genuinely unmatched items.
    """
    result: Dict[str, str] = {}
    _excl_nodes = exclude_node_ids or set()
    _excl_eids = exclude_eids or set()

    for latex_type, mm_type in [("figure", "figure"), ("table", "table"),
                                 ("equation", "formula")]:
        # Collect LaTeX labels of this type, sorted by line_no
        # Skip labels already mapped in Phase 1/2
        type_labels = []
        for lk, info in labels.items():
            lt = normalize_label_type(str(info.get("label_type", "")))
            if lt == latex_type:
                node_id = f"{doc_id}::el::{lk}"
                if node_id in _excl_nodes:
                    continue
                line = _safe_int(info.get("line_no"))
                type_labels.append((line, lk))
        type_labels.sort()

        # Collect MinerU elements of this type, sorted by number then position
        # Skip elements already claimed in Phase 1/2
        type_elements = []
        for eid, el in mm_elements.items():
            if el["element_type"] == mm_type:
                if eid in _excl_eids:
                    continue
                num = _safe_int(el.get("number"))
                pos = _safe_int(el.get("position_idx"))
                type_elements.append((num, pos, eid))
        type_elements.sort()

        # Match 1:1
        for i in range(min(len(type_labels), len(type_elements))):
            _, label_key = type_labels[i]
            _, _, eid = type_elements[i]
            node_id = f"{doc_id}::el::{label_key}"
            result[node_id] = eid

    return result


def build_node_element_map(
    latex_data: Dict[str, Any],
    mm_index: Dict[str, Any],
    mm_data: Dict[str, Any],
) -> Dict[str, str]:
    """Map all LaTeX element node IDs to MinerU element IDs.

    Strategy (in priority order):
    1. Number match (from label key)
    2. Caption Jaccard >= 0.25
    3. Sequential order match (sort by line_no vs number/position)
    """
    node_to_element: Dict[str, str] = {}

    for doc_key, doc in latex_data.items():
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("doc_id", doc_key)
        labels = doc.get("labels", {})

        # Phase 1+2: standard mapping (number + caption)
        for label_key, info in labels.items():
            modal = normalize_label_type(str(info.get("label_type", "")))
            if modal not in ELEMENT_MODALITIES:
                continue
            node_id = f"{doc_id}::el::{label_key}"
            mapped_eid = map_label_to_element(doc_id, label_key, info, mm_index)
            if mapped_eid:
                node_to_element[node_id] = mapped_eid

        # Phase 3: sequential matching for remaining unmapped labels
        # Pre-exclude already-mapped node_ids and element_ids so that
        # zip alignment only operates on genuinely unmatched items.
        mm_doc = mm_data["documents"].get(doc_id)
        if mm_doc:
            mm_elements = mm_doc.get("elements", {})
            if isinstance(mm_elements, dict):
                used_eids = set(node_to_element.values())
                mapped_nodes = set(node_to_element.keys())
                seq_map = build_sequential_mapping(
                    doc_id, labels, mm_elements,
                    exclude_node_ids=mapped_nodes,
                    exclude_eids=used_eids,
                )
                for node_id, eid in seq_map.items():
                    node_to_element[node_id] = eid

    return node_to_element


# ─── Extract edge contexts from LaTeX graph ──────────────────────────────────

def build_edge_context_index(
    latex_data: Dict[str, Any],
) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    """Build index of edge contexts from LaTeX reference graph."""
    edge_ctx: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)

    for doc_key, doc in latex_data.items():
        if not isinstance(doc, dict):
            continue
        doc_id = doc.get("doc_id", doc_key)
        for edge in doc.get("edges", []):
            src_label = edge.get("source_label", edge.get("source", ""))
            tgt_label = edge.get("target_label", edge.get("target", ""))
            src_nid = f"{doc_id}::el::{src_label}"
            tgt_nid = f"{doc_id}::el::{tgt_label}"
            ctx = {
                "source": src_nid,
                "target": tgt_nid,
                "ref_text": edge.get("ref_text", ""),
                "context_snippet": (edge.get("context", "") or "")[:300],
            }
            edge_ctx[(src_nid, tgt_nid)].append(ctx)
            edge_ctx[(tgt_nid, src_nid)].append(ctx)

    return dict(edge_ctx)


# ─── Main enrichment ────────────────────────────────────────────────────────

def _first_n_words(text: str, n: int) -> str:
    """Return the first n words of text, preserving a trailing sentence boundary if close."""
    words = text.split()
    if len(words) <= n:
        return text
    excerpt = " ".join(words[:n])
    # Try to end at a sentence boundary within ±5 words
    for i in range(min(n + 5, len(words)), max(n - 5, 1), -1):
        candidate = " ".join(words[:i])
        if candidate.rstrip().endswith((".", "?", "!")):
            return candidate
    return excerpt


def build_hub_semantic_summary(
    el_a: Dict[str, Any],
    el_b: Dict[str, Any],
    edge_contexts: List[Dict[str, str]],
) -> str:
    """Build a compressed hub semantic summary (~50-80 words).

    Inspired by MoDora's bottom-up cascade summarization: combines enriched
    content from both elements plus edge context.  The result is compressed
    from the raw concatenation to a dense 50-80 word bridge description so
    it fits within prompt context budgets without dominating the input.

    Compression strategy (rule-based, no extra LLM call):
      - Take first ~20 words from each element's enriched description.
      - Take first ~15 words from the strongest bridge context snippet.
      - Include up to 5 shared keywords.
    Total budget: ~55-65 words plus tags.
    """
    parts: List[str] = []

    def element_excerpt(el: Dict[str, Any], label: str) -> str:
        title = (el.get("enriched_title") or "").strip()
        content = (el.get("enriched_content") or "").strip()
        # Prefer enriched fields; fall back to caption
        if not title and not content:
            content = (el.get("caption") or "").strip()
        combined = f"{title}: {content}" if (title and content) else (title or content)
        return _first_n_words(combined, 20) if combined else ""

    # Element A
    exc_a = element_excerpt(el_a, "A")
    if exc_a:
        a_type = el_a.get("element_type", "element").upper()
        parts.append(f"[{a_type} A] {exc_a}")

    # Element B
    exc_b = element_excerpt(el_b, "B")
    if exc_b:
        b_type = el_b.get("element_type", "element").upper()
        parts.append(f"[{b_type} B] {exc_b}")

    # Best bridge context snippet (first non-empty, up to 15 words)
    for ectx in edge_contexts[:3]:
        snippet = ectx.get("context_snippet", "").strip()
        if snippet:
            parts.append(f"[BRIDGE] {_first_n_words(snippet, 15)}")
            break  # one bridge snippet is enough for compression target

    # Keywords: merged from both elements, deduped, max 5
    keywords_a = (el_a.get("enriched_metadata") or {}).get("keywords", [])
    keywords_b = (el_b.get("enriched_metadata") or {}).get("keywords", [])
    all_keywords = list(dict.fromkeys(keywords_a + keywords_b))[:5]
    if all_keywords:
        parts.append(f"[KEYWORDS] {', '.join(all_keywords)}")

    summary = " | ".join(parts) if parts else ""

    # Final word-count guard: hard-cap at 80 words to stay within budget
    words = summary.split()
    if len(words) > 80:
        summary = " ".join(words[:80])

    return summary


def _build_hub_quality_scores(hub_data: Dict[str, Any]) -> Dict[str, float]:
    """Build per-hub quality scores from topology features.

    Instead of a constant 0.8 for all hubs, compute a continuous score
    from bridge_score, pagerank, and out_to_elements — fields already
    present in the hubs list produced by analyze_latex_graph_topology.py.

    Score = w1 * norm(bridge_score) + w2 * norm(pagerank) + w3 * norm(out_to_elements)

    The resulting score is in [0, 1] with meaningful variance across hubs.
    """
    hubs = hub_data.get("hubs", [])
    if not hubs:
        # Fallback: also check if candidates carry hub metadata
        return {}

    bridge_scores = [float(h.get("bridge_score", 0)) for h in hubs]
    pageranks = [float(h.get("pagerank", 0)) for h in hubs]
    out_to_elems = [float(h.get("out_to_elements", 0)) for h in hubs]

    def _norm(vals):
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        if rng < 1e-9:
            return [0.5] * len(vals)
        return [(v - lo) / rng for v in vals]

    n_bs = _norm(bridge_scores)
    n_pr = _norm(pageranks)
    n_oe = _norm(out_to_elems)

    # Weights: bridge_score dominates (captures multi-modality coverage),
    # pagerank adds structural centrality, out_to_elements adds reach.
    w1, w2, w3 = 0.5, 0.25, 0.25

    hub_scores: Dict[str, float] = {}
    for i, h in enumerate(hubs):
        node_id = h.get("node_id", "")
        score = w1 * n_bs[i] + w2 * n_pr[i] + w3 * n_oe[i]
        # Ensure minimum floor of 0.1 so all hubs get some signal
        hub_scores[node_id] = max(0.1, round(score, 4))

    return hub_scores


def enrich_candidates(
    hub_data: Dict[str, Any],
    mm_index: Dict[str, Any],
    node_to_element: Dict[str, str],
    edge_ctx_index: Dict[Tuple[str, str], List[Dict[str, str]]],
    limit: int = 0,
    enriched_elements: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    """Convert hub candidates to generation-ready format."""
    candidates = hub_data["candidates"]
    if limit > 0:
        candidates = candidates[:limit]

    # Build per-hub quality scores from topology features (replaces constant 0.8)
    hub_quality = _build_hub_quality_scores(hub_data)

    pairs: List[Dict[str, Any]] = []
    skipped_no_mapping = 0
    skipped_no_element = 0
    pair_counter: Dict[str, int] = defaultdict(int)

    for cand in candidates:
        # Find the two element endpoints (non-paragraph nodes)
        path_ids = cand["path_node_ids"]
        path_types = cand["path_node_types"]
        if len(path_ids) != len(path_types):
            print(f"  WARNING: path length mismatch in {cand.get('candidate_id', '?')}: "
                  f"ids={len(path_ids)} types={len(path_types)}, skipping")
            skipped_no_mapping += 1
            continue
        endpoints = []
        for nid, ntype in zip(path_ids, path_types):
            if ntype in {"figure", "table", "equation", "formula"}:
                endpoints.append((nid, ntype))

        if len(endpoints) < 2:
            skipped_no_mapping += 1
            continue

        # Take first and last element in path as A and B (backward compat)
        nid_a, ntype_a = endpoints[0]
        nid_b, ntype_b = endpoints[-1]

        # Map to MinerU element IDs
        eid_a = node_to_element.get(nid_a)
        eid_b = node_to_element.get(nid_b)
        if not eid_a or not eid_b:
            skipped_no_mapping += 1
            continue

        # Get element details
        el_a = mm_index["all_elements"].get(eid_a)
        el_b = mm_index["all_elements"].get(eid_b)
        if not el_a or not el_b:
            skipped_no_element += 1
            continue

        # Normalize modality types (equation→formula for MinerU compat)
        mm_type_a = "formula" if ntype_a in ("equation", "formula") else ntype_a
        mm_type_b = "formula" if ntype_b in ("equation", "formula") else ntype_b
        pair_type = "+".join(sorted([mm_type_a, mm_type_b]))

        # Generate pair_id
        doc_id = cand["doc_id"]
        pair_counter[doc_id] += 1
        pair_id = f"{doc_id}_hub_pair_{pair_counter[doc_id]}"

        # Build element dicts (same format as multihop_l1_candidates)
        def make_element_dict(el: Dict) -> Dict[str, Any]:
            d = {
                "element_id": el["element_id"],
                "element_type": el["element_type"],
                "caption": el.get("caption", "") or "",
                "content": el.get("content", "") or "",
                "image_path": el.get("image_path", "") or "",
                "context_before": (el.get("context_before", "") or "")[:300],
                "context_after": (el.get("context_after", "") or "")[:300],
            }
            # Attach MoDora-style enriched fields if available
            eid = el["element_id"]
            if enriched_elements and eid in enriched_elements:
                enr = enriched_elements[eid]
                d["enriched_title"] = enr.get("enriched_title", "")
                d["enriched_metadata"] = enr.get("enriched_metadata", {})
                d["enriched_content"] = enr.get("enriched_content", "")
            return d

        # Collect edge contexts along the path
        path_nids = cand["path_node_ids"]
        edge_contexts = []
        for i in range(len(path_nids) - 1):
            key = (path_nids[i], path_nids[i + 1])
            if key in edge_ctx_index:
                for ctx in edge_ctx_index[key]:
                    # Replace node IDs with element IDs in context
                    edge_contexts.append({
                        "source": node_to_element.get(ctx["source"], ctx["source"]),
                        "target": node_to_element.get(ctx["target"], ctx["target"]),
                        "ref_text": ctx["ref_text"],
                        "context_snippet": ctx["context_snippet"],
                    })

        # Map path node IDs to element IDs where possible
        mapped_path = []
        for nid in path_nids:
            mapped = node_to_element.get(nid)
            mapped_path.append(mapped if mapped else nid)

        elem_a_dict = make_element_dict(el_a)
        elem_b_dict = make_element_dict(el_b)

        # B3: Build node_group covering ALL distinct element endpoints in the path
        # (not just the first and last).  Supports 1-3 element groups as discussed
        # with mentor.  element_a / element_b are kept for backward compatibility.
        node_group: List[Dict[str, Any]] = []
        seen_group_eids: Set[str] = set()
        for ep_nid, ep_ntype in endpoints:
            ep_eid = node_to_element.get(ep_nid)
            if not ep_eid or ep_eid in seen_group_eids:
                continue
            ep_el = mm_index["all_elements"].get(ep_eid)
            if not ep_el:
                continue
            seen_group_eids.add(ep_eid)
            node_group.append(make_element_dict(ep_el))

        # Build hub semantic summary from enriched descriptions
        hub_summary = build_hub_semantic_summary(
            elem_a_dict, elem_b_dict, edge_contexts,
        )

        pair_dict = {
            "pair_id": pair_id,
            "doc_id": doc_id,
            "element_a_id": eid_a,
            "element_b_id": eid_b,
            "element_a_type": mm_type_a,
            "element_b_type": mm_type_b,
            "pair_type": pair_type,
            "hop_distance": cand["hop_distance"],
            "path": mapped_path,
            "quality_score": hub_quality.get(cand.get("hub_node_id", ""), 0.5),
            "overlap_with_existing_l1": False,
            "element_a": elem_a_dict,
            "element_b": elem_b_dict,
            # B3: node_group lists all distinct element endpoints in the path
            # (1-3 elements).  Supports real-user style templates that can
            # reference a variable number of evidence nodes.
            "node_group": node_group,
            "edge_contexts": edge_contexts,
            # Hub semantic summary (MoDora cascade aggregation, compressed ~50-80 words)
            "hub_semantic_summary": hub_summary,
            # Preserve hub metadata for analysis
            "hub_metadata": {
                "hub_node_id": cand["hub_node_id"],
                "hub_label": cand["hub_label"],
                "is_cross_doc": cand["is_cross_doc"],
                "page_span": cand.get("page_span"),
                "line_no_span": cand.get("line_no_span"),
                "short_query_seed": cand.get("short_query_seed", ""),
                "long_query_seed": cand.get("long_query_seed", ""),
            },
        }
        pairs.append(pair_dict)

    # ── Build adjacent bridge element mappings (MinerU IDs) ──
    # These elements aren't in the top-60 hub candidate pairs but appear in
    # adjacent_backbone_bridges. Including them expands hub coverage for eval.
    adj_bridge_elements: Dict[str, float] = {}  # eid → quality_score
    adj_bridges = hub_data.get("adjacent_backbone_bridges", [])
    if adj_bridges:
        adj_bridge_adjacency: List[Dict[str, Any]] = []
        for ab in adj_bridges:
            mapped_i = []
            mapped_j = []
            for nid in (ab.get("element_ids_i") or []):
                eid = node_to_element.get(str(nid))
                if eid:
                    mapped_i.append(eid)
                    adj_bridge_elements[eid] = max(adj_bridge_elements.get(eid, 0), 0.4)
            for nid in (ab.get("element_ids_j") or []):
                eid = node_to_element.get(str(nid))
                if eid:
                    mapped_j.append(eid)
                    adj_bridge_elements[eid] = max(adj_bridge_elements.get(eid, 0), 0.4)
            if mapped_i and mapped_j:
                adj_bridge_adjacency.append({
                    "doc_id": ab.get("doc_id", ""),
                    "elements_i": mapped_i,
                    "elements_j": mapped_j,
                })

    # Summary statistics
    by_type: Dict[str, int] = defaultdict(int)
    by_hop: Dict[str, int] = defaultdict(int)
    docs_covered: Set[str] = set()
    cross_doc_count = 0

    for p in pairs:
        by_type[p["pair_type"]] += 1
        by_hop[str(p["hop_distance"])] += 1
        docs_covered.add(p["doc_id"])
        if p["hub_metadata"]["is_cross_doc"]:
            cross_doc_count += 1

    result = {
        "metadata": {
            "source": "latex_hub_multihop_candidates (enriched)",
            "hub_candidates_file": str(hub_data.get("metadata", {}).get("generated_at", "")),
            "algorithm_version": hub_data.get("metadata", {}).get("algorithm_version", ""),
            "max_pairs": len(pairs),
            "enrichment_stats": {
                "input_candidates": len(candidates),
                "output_pairs": len(pairs),
                "skipped_no_mapping": skipped_no_mapping,
                "skipped_no_element": skipped_no_element,
                "mapping_rate": round(len(pairs) / len(candidates), 4) if candidates else 0,
            },
        },
        "summary": {
            "total_selected": len(pairs),
            "by_type": dict(by_type),
            "by_hop": dict(by_hop),
            "docs_covered": len(docs_covered),
            "cross_doc": cross_doc_count,
            "intra_doc": len(pairs) - cross_doc_count,
            "adjacent_bridge_elements_count": len(adj_bridge_elements),
        },
        "pairs": pairs,
        # Additional elements from adjacent_backbone_bridges mapped to MinerU IDs.
        # These expand hub coverage beyond the top-60 bridge hubs.
        "adjacent_bridge_elements": {eid: score for eid, score in adj_bridge_elements.items()},
        "adjacent_bridge_adjacency": adj_bridge_adjacency if adj_bridges else [],
    }

    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--hub-candidates",
        default="data/latex_hub_multihop_candidates.json",
        help="Input hub candidates from topology analysis",
    )
    ap.add_argument(
        "--elements",
        default="data/multimodal_elements.json",
        help="MinerU multimodal elements",
    )
    ap.add_argument(
        "--latex-graph",
        default="data/latex_reference_graph.json",
        help="LaTeX reference graph (for label info and edge contexts)",
    )
    ap.add_argument(
        "--output",
        default="data/hub_candidates_enriched.json",
        help="Output enriched candidates",
    )
    ap.add_argument(
        "--enriched-elements",
        default=None,
        help="Optional MoDora-enriched elements file (multimodal_elements_enriched.json)",
    )
    ap.add_argument(
        "--hubs",
        default="data/latex_graph_hubs.json",
        help="Hub topology data (for quality_score computation from bridge_score/pagerank)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Limit candidates (0=all)")
    args = ap.parse_args()

    print("Loading data...")
    with open(args.hub_candidates, encoding="utf-8") as f:
        hub_data = json.load(f)
    print(f"  Hub candidates: {len(hub_data['candidates'])}")

    with open(args.elements, encoding="utf-8") as f:
        mm_data = json.load(f)
    print(f"  Documents: {len(mm_data['documents'])}")

    with open(args.latex_graph, encoding="utf-8") as f:
        latex_raw = json.load(f)
    # Handle both flat dict and nested {"documents": {...}} format
    if "documents" in latex_raw and isinstance(latex_raw["documents"], dict):
        latex_data = latex_raw["documents"]
    else:
        latex_data = latex_raw
    latex_docs = {k: v for k, v in latex_data.items() if isinstance(v, dict) and "labels" in v}
    print(f"  LaTeX docs with labels: {len(latex_docs)}")

    # Load enriched elements if provided
    enriched_elements: Optional[Dict[str, Dict]] = None
    if args.enriched_elements:
        print(f"  Loading enriched elements from {args.enriched_elements}...")
        with open(args.enriched_elements, encoding="utf-8") as f:
            enriched_data = json.load(f)
        enriched_elements = {}
        for doc in enriched_data["documents"].values():
            elements = doc.get("elements", {})
            if isinstance(elements, dict):
                for eid, el in elements.items():
                    if "enriched_title" in el:
                        enriched_elements[eid] = el
        print(f"  Enriched elements available: {len(enriched_elements)}")

    # Load hubs topology data for quality_score computation
    hubs_path = Path(args.hubs)
    if hubs_path.exists():
        with open(hubs_path, encoding="utf-8") as f:
            hubs_topo = json.load(f)
        print(f"  Hubs topology: {len(hubs_topo.get('hubs', []))} hubs loaded")
        # Merge hubs into hub_data so enrich_candidates can access them
        hub_data["hubs"] = hubs_topo.get("hubs", [])
        hub_data["adjacent_backbone_bridges"] = hubs_topo.get("adjacent_backbone_bridges", [])
    else:
        print(f"  WARNING: hubs file {hubs_path} not found, quality_score will use default 0.5")

    print("\nBuilding indices...")
    mm_index = build_mm_index(mm_data)
    print(f"  Total elements indexed: {len(mm_index['all_elements'])}")

    node_to_element = build_node_element_map(latex_docs, mm_index, mm_data)
    print(f"  Node→element mappings: {len(node_to_element)}")

    edge_ctx_index = build_edge_context_index(latex_docs)
    print(f"  Edge context pairs: {len(edge_ctx_index)}")

    print("\nEnriching candidates...")
    result = enrich_candidates(
        hub_data, mm_index, node_to_element, edge_ctx_index, args.limit,
        enriched_elements=enriched_elements,
    )

    summary = result["summary"]
    stats = result["metadata"]["enrichment_stats"]
    print(f"\n=== Enrichment Results ===")
    print(f"  Input:  {stats['input_candidates']} candidates")
    print(f"  Output: {stats['output_pairs']} pairs ({stats['mapping_rate']*100:.1f}% mapping rate)")
    print(f"  Skipped (no mapping): {stats['skipped_no_mapping']}")
    print(f"  Skipped (no element): {stats['skipped_no_element']}")
    print(f"\n  By type: {dict(summary['by_type'])}")
    print(f"  By hop:  {dict(summary['by_hop'])}")
    print(f"  Docs covered: {summary['docs_covered']}")
    print(f"  Cross-doc: {summary['cross_doc']} / Intra-doc: {summary['intra_doc']}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
