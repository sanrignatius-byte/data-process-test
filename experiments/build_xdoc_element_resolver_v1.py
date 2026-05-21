#!/usr/bin/env python3
"""Cross-doc element resolver v1 with explicit target numbered refs + fanout penalty.

v0 baseline: experiments/build_xdoc_element_resolver_v0.py
Key additions over v0:
  - Target-side explicit Figure/Table/Eq N resolution (anchored / unanchored tiers)
  - Source nearest-position fanout penalty
  - Target title aliases from latex_reference_graph_v2 for anchoring
  - Extended hub_metadata with resolution method, ref text, fanout info
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Iterator, List, Tuple, Set, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EDGES = ROOT / "data/04_xdoc_citation/predicted_xdoc_edges_chunks_filtered.jsonl"
DEFAULT_ELEMENTS = ROOT / "data/01_graphs/multimodal_elements_v2.json"
DEFAULT_CHUNKS = ROOT / "data/01_graphs/chunk_virtual_nodes_v2.json"
DEFAULT_REF_GRAPH = ROOT / "data/01_graphs/latex_reference_graph_v2.json"
DEFAULT_OUT_ROOT = ROOT / "data/05_eval"

MODAL_TYPES = {"figure", "table", "formula"}
TYPE_WORDS = {
    "figure": {"figure", "fig", "plot", "chart", "diagram", "image"},
    "table": {"table", "tabular", "row", "column", "metric", "result"},
    "formula": {"equation", "eq", "formula", "loss", "objective", "constraint"},
}
# Numbered ref extraction: (element_type, regex)
REF_PATTERNS = [
    ("figure", re.compile(r"\b(?:fig(?:ure)?\.?)\s*(\d+)", re.I)),
    ("table", re.compile(r"\btable\s*(\d+)", re.I)),
    ("formula", re.compile(r"\b(?:eq(?:uation|n)?\.?)\s*\(?(\d+)\)?", re.I)),
]
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "can", "could", "do", "does", "each", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "may", "more", "most", "of", "on",
    "or", "our", "paper", "papers", "proposed", "show", "shown", "shows",
    "such", "that", "the", "their", "these", "this", "to", "using", "via",
    "we", "which", "with", "without", "work", "works",
}

# Fanout penalty tiers per design
FANOUT_PENALTY_TABLE = [
    (2, 1.0),
    (5, 0.75),
    (10, 0.55),
    (float("inf"), 0.35),
]

# Numbered ref window chars on each side
REF_WINDOW_CHARS = 120


def target_score_bucket(score: float) -> str:
    """Bucket a target score for both raw attempts and selected final pairs."""
    if score >= 0.90:
        return ">=0.90 (explicit anchored)"
    if score >= 0.70:
        return "0.70-0.90 (explicit unanchored)"
    if score >= 0.20:
        return ">=0.20"
    if score >= 0.12:
        return "0.12-0.20"
    return "0.07-0.12"


def target_anchor_reason(pair: dict[str, Any]) -> str:
    """Return the anchor reason used by a selected pair, or the fallback method."""
    meta = pair.get("hub_metadata") or {}
    detail = meta.get("target_resolution_detail") or {}
    reason = detail.get("anchor_reason")
    if reason:
        return str(reason)
    return str(meta.get("target_resolution_method") or "unknown")


# ── file I/O ────────────────────────────────────────────────────────────────

def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def compact_text(value: Any, limit: int = 500) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.split())[:limit]


def tokens(text: str) -> set[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", (text or "").lower())
    return {t for t in toks if t not in STOPWORDS and not t.isdigit()}


# ── title aliases ───────────────────────────────────────────────────────────

def load_title_aliases(ref_graph_path: Path) -> dict[str, set[str]]:
    """Return doc_id -> normalized title tokens for target anchoring."""
    if not ref_graph_path.exists():
        return {}
    graph = read_json(ref_graph_path)
    docs = graph.get("documents") or {}
    aliases: dict[str, set[str]] = {}
    for doc_id, doc in docs.items():
        title = (doc.get("metadata") or {}).get("title") or ""
        if title and title != "N/A":
            # Tokenize: keep words >= 4 chars, lowercase, drop stopwords
            title_tokens = set()
            for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", title.lower()):
                if t not in STOPWORDS:
                    title_tokens.add(t)
            if title_tokens:
                aliases[doc_id] = title_tokens
    return aliases


# ── numbered ref extraction ─────────────────────────────────────────────────

def extract_numbered_refs_with_windows(
    text: str, window_chars: int = REF_WINDOW_CHARS
) -> list[dict[str, Any]]:
    """Return type, number, span, and local text windows for Figure/Table/Eq refs."""
    refs: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for etype, pat in REF_PATTERNS:
        for match in pat.finditer(text or ""):
            number = int(match.group(1))
            start, end = match.start(), match.end()
            key = (etype, number, start)
            if key in seen:
                continue
            seen.add(key)
            window_before = text[max(0, start - window_chars):start].strip()
            window_after = text[end:end + window_chars].strip()
            refs.append({
                "element_type": etype,
                "number": number,
                "span": [start, end],
                "ref_text": text[start:end],
                "window_before": window_before,
                "window_after": window_after,
            })
    return refs


def is_target_anchored_ref(
    ref: dict[str, Any],
    edge: dict[str, Any],
    target_aliases: set[str],
    fanout: int,
    all_refs_in_chunk: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Decide whether a numbered ref likely points to edge.target_doc.

    Returns (is_anchored, reason).
    """
    # Condition 1: ref window contains target title words
    if target_aliases:
        window_text = (ref["window_before"] + " " + ref["window_after"]).lower()
        window_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", window_text))
        title_overlap = target_aliases & window_tokens
        if len(title_overlap) >= 2:
            return True, "title_words_in_window"

    # Condition 2: edge features have title_match >= 0.2
    features = edge.get("features") or {}
    title_match = float(features.get("title_match") or 0.0)
    if title_match >= 0.2:
        return True, "title_match_ge_0.2"

    # Condition 3: single ref in chunk + high C18 probability
    probability = float(edge.get("probability") or 0.0)
    if len(all_refs_in_chunk) == 1 and probability >= 0.95:
        return True, "single_ref_high_prob"

    # Condition 4: low citation fanout
    if fanout <= 2:
        return True, "low_fanout"

    return False, "unanchored"


# ── fanout computation ──────────────────────────────────────────────────────

def compute_citation_fanout(edges: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Count filtered citation edges per source chunk_id."""
    fanout: Counter = Counter()
    for edge in edges:
        chunk_id = edge.get("chunk_id") or ""
        if chunk_id:
            fanout[chunk_id] += 1
    return dict(fanout)


def source_fanout_penalty(fanout: int) -> float:
    """Return source nearest-position penalty for a given fanout."""
    for threshold, penalty in FANOUT_PENALTY_TABLE:
        if fanout <= threshold:
            return penalty
    return 0.35


# ── element index ───────────────────────────────────────────────────────────

def element_detail(elem: dict[str, Any]) -> dict[str, Any]:
    return {
        "element_id": elem.get("element_id", ""),
        "doc_id": elem.get("doc_id", ""),
        "element_type": elem.get("element_type", ""),
        "caption": elem.get("caption", "") or "",
        "content": compact_text(elem.get("content", ""), 2000),
        "image_path": elem.get("image_path", "") or "",
        "context_before": compact_text(elem.get("context_before", ""), 300),
        "context_after": compact_text(elem.get("context_after", ""), 300),
        "enriched_title": elem.get("enriched_title", "") or "",
        "enriched_content": elem.get("enriched_content", "") or "",
        "enriched_metadata": elem.get("enriched_metadata", {}) or {},
        "label": elem.get("label", "") or "",
        "number": elem.get("number"),
        "position_idx": elem.get("position_idx"),
        "page_idx": elem.get("page_idx"),
    }


def element_text(elem: dict[str, Any]) -> str:
    return " ".join(
        compact_text(elem.get(k), 800)
        for k in ("label", "caption", "content", "context_before", "context_after")
        if elem.get(k)
    )


def build_element_index(elements_path: Path) -> dict[str, Any]:
    raw = read_json(elements_path)
    doc_index: dict[str, Any] = {}
    for doc_id, doc in (raw.get("documents") or {}).items():
        elements: list[dict[str, Any]] = []
        by_number: dict[tuple[str, int], dict[str, Any]] = {}
        max_pos = 0
        for elem_id, elem in (doc.get("elements") or {}).items():
            etype = elem.get("element_type")
            if etype not in MODAL_TYPES:
                continue
            row = dict(elem)
            row.setdefault("element_id", elem_id)
            row.setdefault("doc_id", doc_id)
            row["_tokens"] = tokens(element_text(row))
            elements.append(row)
            pos = row.get("position_idx")
            if isinstance(pos, int):
                max_pos = max(max_pos, pos)
            number = row.get("number")
            if isinstance(number, int):
                by_number[(etype, number)] = row
        doc_index[doc_id] = {
            "elements": elements,
            "by_number": by_number,
            "max_position_idx": max_pos,
        }
    return doc_index


# ── chunk position ──────────────────────────────────────────────────────────

def chunk_position(edge: dict[str, Any], chunks: dict[str, Any]) -> float:
    features = edge.get("features") or {}
    pos = features.get("position")
    if isinstance(pos, (int, float)):
        return max(0.0, min(1.0, float(pos)))
    source_doc = edge.get("source_doc")
    chunk_id = edge.get("chunk_id")
    nodes = ((chunks.get(source_doc) or {}).get("nodes") or {})
    if chunk_id in nodes:
        total = max(len(nodes), 1)
        return float(nodes[chunk_id].get("chunk_idx", 0)) / total
    return 0.5


# ── source element resolution (same as v0) ──────────────────────────────────

def explicit_source_elements(
    doc_info: dict[str, Any],
    refs: list[dict[str, int]],
) -> list[tuple[dict[str, Any], str, float]]:
    found: list[tuple[dict[str, Any], str, float]] = []
    for ref in refs:
        elem = doc_info["by_number"].get((ref["element_type"], ref["number"]))
        if elem:
            found.append((elem, "source_explicit_ref", 1.0))
    return found


def nearest_source_elements(
    doc_info: dict[str, Any],
    position: float,
    limit: int,
    fanout_penalty: float = 1.0,
) -> list[tuple[dict[str, Any], str, float]]:
    max_pos = max(int(doc_info.get("max_position_idx") or 0), 1)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for elem in doc_info["elements"]:
        pos = elem.get("position_idx")
        if not isinstance(pos, int):
            continue
        norm = pos / max_pos
        dist = abs(norm - position)
        score = max(0.0, 1.0 - dist / 0.22) * fanout_penalty
        ranked.append((score, elem))
    ranked.sort(key=lambda item: (-item[0], item[1].get("element_id", "")))
    return [(elem, "source_nearest_position", score) for score, elem in ranked[:limit] if score > 0.0]


def choose_source_elements(
    edge: dict[str, Any],
    doc_info: dict[str, Any],
    chunks: dict[str, Any],
    limit: int,
    fanout: int = 1,
) -> list[tuple[dict[str, Any], str, float]]:
    refs = extract_refs(edge.get("chunk_text", ""))
    explicit = explicit_source_elements(doc_info, refs)
    if explicit:
        return explicit[:limit]
    penalty = source_fanout_penalty(fanout)
    return nearest_source_elements(doc_info, chunk_position(edge, chunks), limit, penalty)


def extract_refs(text: str) -> list[dict[str, int]]:
    refs: list[dict[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for etype, pat in REF_PATTERNS:
        for match in pat.finditer(text or ""):
            number = int(match.group(1))
            key = (etype, number)
            if key in seen:
                continue
            seen.add(key)
            refs.append({"element_type": etype, "number": number})
    return refs


# ── target element resolution (v1: explicit number first) ───────────────────

def target_score_lexical(
    elem: dict[str, Any],
    query_tokens: set[str],
    mentioned_types: set[str],
) -> tuple[float, dict[str, Any]]:
    """Lexical overlap score (same as v0 target_score)."""
    elem_tokens = elem.get("_tokens") or set()
    if not elem_tokens or not query_tokens:
        return 0.0, {"overlap": 0, "overlap_terms": []}
    overlap = query_tokens & elem_tokens
    cosine_like = len(overlap) / math.sqrt(len(query_tokens) * len(elem_tokens))
    coverage = len(overlap) / max(1, min(len(query_tokens), len(elem_tokens)))
    type_bonus = 0.025 if elem.get("element_type") in mentioned_types else 0.0
    score = 0.75 * cosine_like + 0.25 * coverage + type_bonus
    return score, {
        "overlap": len(overlap),
        "overlap_terms": sorted(overlap)[:12],
        "cosine_like": round(cosine_like, 6),
        "coverage": round(coverage, 6),
        "type_bonus": type_bonus,
    }


def rank_target_elements_v1(
    edge: dict[str, Any],
    doc_info: dict[str, Any],
    target_aliases: set[str],
    fanout: int,
    limit: int,
    min_score: float,
    min_overlap_terms: int,
) -> list[tuple[dict[str, Any], str, float, dict[str, Any]]]:
    """v1 target resolver: explicit-number first, then lexical fallback."""
    text = " ".join([edge.get("section_title", "") or "", edge.get("chunk_text", "") or ""])
    all_refs = extract_numbered_refs_with_windows(text)

    results: list[tuple[dict[str, Any], str, float, dict[str, Any]]] = []
    seen_elem_ids: set[str] = set()

    # Tier 1: explicit numbered refs that resolve to target doc elements
    for ref in all_refs:
        elem = doc_info["by_number"].get((ref["element_type"], ref["number"]))
        if not elem:
            continue
        eid = elem["element_id"]
        if eid in seen_elem_ids:
            continue
        seen_elem_ids.add(eid)
        is_anchored, anchor_reason = is_target_anchored_ref(
            ref, edge, target_aliases, fanout, all_refs,
        )
        method = "target_explicit_number_anchored" if is_anchored else "target_explicit_number_unanchored"
        score = 1.0 if is_anchored else 0.70
        detail = {
            "ref_text": ref["ref_text"],
            "ref_window_before": ref["window_before"][:200],
            "ref_window_after": ref["window_after"][:200],
            "element_type": ref["element_type"],
            "number": ref["number"],
            "anchor_reason": anchor_reason,
            "anchor_is_anchored": is_anchored,
        }
        results.append((elem, method, score, detail))

    # Sort explicit results: anchored first, then unanchored
    results.sort(key=lambda x: (-x[2], x[0].get("element_id", "")))

    # Tier 2: lexical overlap fallback
    query_tokens = tokens(text)
    mentioned_types = {
        etype for etype, words in TYPE_WORDS.items()
        if any(word in query_tokens for word in words)
    }
    lexical_results: list[tuple[dict[str, Any], str, float, dict[str, Any]]] = []
    for elem in doc_info["elements"]:
        eid = elem["element_id"]
        if eid in seen_elem_ids:
            continue
        score, lex_detail = target_score_lexical(elem, query_tokens, mentioned_types)
        if score >= min_score and int(lex_detail.get("overlap") or 0) >= min_overlap_terms:
            lexical_results.append((elem, "target_caption_overlap", score, lex_detail))
    lexical_results.sort(key=lambda x: (-x[2], x[0].get("element_id", "")))

    results.extend(lexical_results)
    return results[:limit]


# ── pair construction ───────────────────────────────────────────────────────

def make_pair(
    edge: dict[str, Any],
    source_elem: dict[str, Any],
    source_method: str,
    source_score: float,
    target_elem: dict[str, Any],
    target_method: str,
    target_score_value: float,
    target_detail: dict[str, Any],
    index: int,
    fanout: int,
    source_fanout_penalty_value: float,
) -> dict[str, Any]:
    source_doc = edge["source_doc"]
    target_doc = edge["target_doc"]
    citation_probability = float(edge.get("probability") or 0.0)
    quality = 0.45 * citation_probability + 0.25 * source_score + 0.30 * min(target_score_value / 0.35, 1.0)
    pair_type = "+".join(sorted([source_elem["element_type"], target_elem["element_type"]]))
    chunk_id = edge.get("chunk_id", "")
    return {
        "pair_id": f"xdoc_resolver_v1_{index:06d}",
        "doc_id": f"{source_doc}__{target_doc}",
        "source_doc": source_doc,
        "target_doc": target_doc,
        "element_a_id": source_elem["element_id"],
        "element_b_id": target_elem["element_id"],
        "element_a_type": source_elem["element_type"],
        "element_b_type": target_elem["element_type"],
        "pair_type": pair_type,
        "hop_distance": 3,
        "path": [source_elem["element_id"], chunk_id, target_doc, target_elem["element_id"]],
        "quality_score": round(quality, 6),
        "element_a": element_detail(source_elem),
        "element_b": element_detail(target_elem),
        "node_group": [element_detail(source_elem), element_detail(target_elem)],
        "edge_contexts": [{
            "source": chunk_id,
            "target": target_doc,
            "edge_type": "cross_doc_citation_chunk",
            "context_snippet": compact_text(edge.get("chunk_text", ""), 700),
            "section_title": edge.get("section_title", ""),
            "probability": citation_probability,
            "features": edge.get("features", {}),
        }],
        "hub_semantic_summary": (
            f"[SOURCE {source_elem['element_type'].upper()}] "
            f"{compact_text(source_elem.get('caption') or source_elem.get('content'), 180)} | "
            f"[CITATION BRIDGE] {compact_text(edge.get('chunk_text'), 220)} | "
            f"[TARGET {target_elem['element_type'].upper()}] "
            f"{compact_text(target_elem.get('caption') or target_elem.get('content'), 180)}"
        ),
        "strategy": "xdoc_citation_element_resolver_v1",
        "hub_metadata": {
            "is_cross_doc": True,
            "strategy": "xdoc_citation_element_resolver_v1",
            "resolver_version": "v1",
            "source_doc": source_doc,
            "target_doc": target_doc,
            "chunk_id": chunk_id,
            "citation_probability": citation_probability,
            "source_resolution_method": source_method,
            "source_resolution_score": round(source_score, 6),
            "target_resolution_method": target_method,
            "target_resolution_score": round(target_score_value, 6),
            "target_resolution_detail": target_detail,
            "citation_fanout": fanout,
            "source_fanout_penalty": round(source_fanout_penalty_value, 6),
            "citation_filter_metadata": edge.get("filter_metadata", {}),
        },
    }


# ── report ──────────────────────────────────────────────────────────────────

def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Cross-Doc Element Resolver v1",
        "",
        f"- input citation edges: **{summary['input_edges']}**",
        f"- edges with resolved source+target elements: **{summary['edges_with_pairs']}**",
        f"- candidate pairs: **{summary['total_pairs']}**",
        f"- docs covered: source **{summary['source_docs_covered']}**, "
        f"target **{summary['target_docs_covered']}**",
        f"- cross-modal pairs: **{summary['cross_modal_pairs']}**",
        f"- pair types: `{summary['pair_type_counts']}`",
        f"- source methods: `{summary['source_method_counts']}`",
        f"- target methods: `{summary['target_method_counts']}`",
        f"- raw target score buckets: `{summary['target_score_buckets']}`",
        f"- post-filter target score buckets: `{summary.get('post_filter_target_score_buckets', {})}`",
        f"- post-filter anchor reasons: `{summary.get('post_filter_anchor_reasons', {})}`",
        f"- fanout buckets: `{summary.get('fanout_buckets', {})}`",
        "",
        "## Method comparison vs v0",
        "",
        f"| metric | v0 | v1 | delta |",
        f"|---|---|---|---|",
    ]
    v0_cmp = summary.get("v0_comparison") or {}
    for metric, vals in v0_cmp.items():
        v0_val = vals.get("v0", 0)
        v1_val = vals.get("v1", 0)
        delta = v1_val - v0_val
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        lines.append(f"| {metric} | {v0_val} | {v1_val} | {delta_str} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "v1 adds target-side explicit Figure/Table/Eq N resolution with anchored/unanchored",
        "tiers and source nearest-position fanout penalty. Pairs are candidate generators",
        "for manual/LLM judging, not final ground truth.",
        "",
    ])
    return "\n".join(lines)


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build v1 xdoc element pairs with explicit target refs + fanout penalty.")
    parser.add_argument("--edges", default=str(DEFAULT_EDGES))
    parser.add_argument("--elements", default=str(DEFAULT_ELEMENTS))
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    parser.add_argument("--ref-graph", default=str(DEFAULT_REF_GRAPH))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--stamp", default="")
    parser.add_argument("--max-edges", type=int, default=0)
    parser.add_argument("--max-source-elements", type=int, default=2)
    parser.add_argument("--max-target-elements", type=int, default=2)
    parser.add_argument("--min-target-score", type=float, default=0.12)
    parser.add_argument("--min-overlap-terms", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=5000)
    parser.add_argument("--max-pairs-per-source-chunk", type=int, default=25)
    parser.add_argument("--allow-same-type", action="store_true")
    args = parser.parse_args()

    stamp = args.stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / f"xdoc_element_resolver_v1_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    doc_index = build_element_index(Path(args.elements))
    chunks = (read_json(Path(args.chunks)).get("documents") or {})
    title_aliases = load_title_aliases(Path(args.ref_graph))
    print(f"Loaded title aliases for {len(title_aliases)} docs")

    # Precompute citation fanout across all filtered edges
    all_edges = list(iter_jsonl(Path(args.edges)))
    if args.max_edges and args.max_edges < len(all_edges):
        all_edges = all_edges[:args.max_edges]
    fanout_map = compute_citation_fanout(all_edges)
    print(f"Computed fanout for {len(fanout_map)} unique source chunks")

    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    stats: Counter = Counter()
    source_docs: set[str] = set()
    target_docs: set[str] = set()
    target_score_buckets: Counter = Counter()
    fanout_buckets: Counter = Counter()

    for edge_idx, edge in enumerate(all_edges, start=1):
        stats["input_edges"] += 1
        source_doc = edge.get("source_doc")
        target_doc = edge.get("target_doc")
        if source_doc not in doc_index or target_doc not in doc_index:
            stats["missing_doc_index"] += 1
            continue

        chunk_id = edge.get("chunk_id", "")
        fanout = fanout_map.get(chunk_id, 1)

        # Fanout bucket for reporting
        if fanout <= 2:
            fanout_buckets["1-2"] += 1
        elif fanout <= 5:
            fanout_buckets["3-5"] += 1
        elif fanout <= 10:
            fanout_buckets["6-10"] += 1
        else:
            fanout_buckets[">10"] += 1

        source_candidates = choose_source_elements(
            edge, doc_index[source_doc], chunks, args.max_source_elements, fanout,
        )
        target_aliases_set = title_aliases.get(target_doc, set())
        target_candidates = rank_target_elements_v1(
            edge, doc_index[target_doc], target_aliases_set, fanout,
            args.max_target_elements, args.min_target_score, args.min_overlap_terms,
        )
        if not source_candidates:
            stats["no_source_element"] += 1
            continue
        if not target_candidates:
            stats["no_target_element"] += 1
            continue
        stats["edges_with_pairs"] += 1
        source_docs.add(source_doc)
        target_docs.add(target_doc)

        source_fanout_penalty_value = source_fanout_penalty(fanout)

        for source_elem, source_method, source_score in source_candidates:
            for target_elem, target_method, target_score_value, target_detail in target_candidates:
                if source_elem["element_id"] == target_elem["element_id"]:
                    continue
                if not args.allow_same_type and source_elem["element_type"] == target_elem["element_type"]:
                    stats["drop_same_type"] += 1
                    continue
                pair = make_pair(
                    edge, source_elem, source_method, source_score,
                    target_elem, target_method, target_score_value, target_detail,
                    len(dedup) + 1, fanout, source_fanout_penalty_value,
                )
                key = (pair["element_a_id"], pair["element_b_id"], chunk_id)
                old = dedup.get(key)
                if old is None or pair["quality_score"] > old["quality_score"]:
                    dedup[key] = pair
                stats[f"source_method:{source_method}"] += 1
                stats[f"target_method:{target_method}"] += 1
                stats[f"pair_type:{pair['pair_type']}"] += 1
                target_score_buckets[target_score_bucket(target_score_value)] += 1

    sorted_pairs = sorted(dedup.values(), key=lambda p: (-p["quality_score"], p["pair_id"]))
    pairs: list[dict[str, Any]] = []
    by_source_chunk: Counter = Counter()
    for pair in sorted_pairs:
        chunk_id = pair["hub_metadata"]["chunk_id"]
        if (
            args.max_pairs_per_source_chunk > 0
            and by_source_chunk[chunk_id] >= args.max_pairs_per_source_chunk
        ):
            stats["drop_source_chunk_cap"] += 1
            continue
        by_source_chunk[chunk_id] += 1
        pairs.append(pair)
    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]
    for i, pair in enumerate(pairs, start=1):
        pair["pair_id"] = f"xdoc_resolver_v1_{i:06d}"

    pair_type_counts = Counter(p["pair_type"] for p in pairs)
    source_method_counts = Counter(p["hub_metadata"]["source_resolution_method"] for p in pairs)
    target_method_counts = Counter(p["hub_metadata"]["target_resolution_method"] for p in pairs)
    post_filter_target_score_buckets = Counter(
        target_score_bucket(float(p["hub_metadata"]["target_resolution_score"]))
        for p in pairs
    )
    post_filter_anchor_reasons = Counter(target_anchor_reason(p) for p in pairs)
    post_filter_anchor_reasons_by_method: dict[str, dict[str, int]] = {}
    for method in sorted(target_method_counts):
        post_filter_anchor_reasons_by_method[method] = dict(Counter(
            target_anchor_reason(p)
            for p in pairs
            if p["hub_metadata"]["target_resolution_method"] == method
        ))
    pair_fanout_buckets: Counter = Counter()
    for p in pairs:
        f = p["hub_metadata"]["citation_fanout"]
        if f <= 2:
            pair_fanout_buckets["1-2"] += 1
        elif f <= 5:
            pair_fanout_buckets["3-5"] += 1
        elif f <= 10:
            pair_fanout_buckets["6-10"] += 1
        else:
            pair_fanout_buckets[">10"] += 1

    # Build v0 comparison
    v0_summary_path = Path(args.out_root) / "xdoc_element_resolver_v0_latest" / "summary.json"
    v0_cmp = {}
    if v0_summary_path.exists():
        v0 = read_json(v0_summary_path)
        v0_cmp = {
            "explicit_target_pairs": {
                "v0": 0,
                "v1": target_method_counts.get("target_explicit_number_anchored", 0)
                + target_method_counts.get("target_explicit_number_unanchored", 0),
            },
            "explicit_source_pairs": {
                "v0": v0.get("source_method_counts", {}).get("source_explicit_ref", 0),
                "v1": source_method_counts.get("source_explicit_ref", 0),
            },
            "total_pairs": {
                "v0": v0.get("total_pairs", 0),
                "v1": len(pairs),
            },
        }

    summary = {
        "builder": "build_xdoc_element_resolver_v1.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_edges_path": str(Path(args.edges)),
        "elements_path": str(Path(args.elements)),
        "chunks_path": str(Path(args.chunks)),
        "input_edges": stats["input_edges"],
        "edges_with_pairs": stats["edges_with_pairs"],
        "total_pairs": len(pairs),
        "cross_modal_pairs": sum(1 for p in pairs if p["element_a_type"] != p["element_b_type"]),
        "source_docs_covered": len({p["source_doc"] for p in pairs}),
        "target_docs_covered": len({p["target_doc"] for p in pairs}),
        "pair_type_counts": dict(pair_type_counts),
        "source_method_counts": dict(source_method_counts),
        "target_method_counts": dict(target_method_counts),
        "target_score_buckets_scope": "raw candidate attempts before dedup/source-chunk cap/top-k",
        "target_score_buckets": dict(target_score_buckets),
        "post_filter_target_score_buckets": dict(post_filter_target_score_buckets),
        "post_filter_anchor_reasons": dict(post_filter_anchor_reasons),
        "post_filter_anchor_reasons_by_method": post_filter_anchor_reasons_by_method,
        "fanout_buckets": dict(pair_fanout_buckets),
        "v0_comparison": v0_cmp,
        "raw_stats": dict(stats),
        "params": {
            "max_edges": args.max_edges,
            "max_source_elements": args.max_source_elements,
            "max_target_elements": args.max_target_elements,
            "min_target_score": args.min_target_score,
            "min_overlap_terms": args.min_overlap_terms,
            "max_pairs": args.max_pairs,
            "max_pairs_per_source_chunk": args.max_pairs_per_source_chunk,
            "allow_same_type": args.allow_same_type,
            "fanout_penalty_table": FANOUT_PENALTY_TABLE,
            "title_aliases_docs": len(title_aliases),
        },
    }

    # Write output
    pairs_json = {
        "metadata": {
            "source": "build_xdoc_element_resolver_v1.py",
            "generated_at": summary["created_at"],
            "summary": summary,
        },
        "summary": {
            "total_selected": len(pairs),
            "cross_doc": len(pairs),
            "intra_doc": 0,
            "by_type": dict(pair_type_counts),
            "docs_covered": len({p["source_doc"] for p in pairs} | {p["target_doc"] for p in pairs}),
        },
        "pairs": pairs,
        "adjacent_bridge_elements": {},
        "adjacent_bridge_adjacency": [],
    }

    (out_dir / "cross_doc_pairs_v1.json").write_text(
        json.dumps(pairs_json, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(out_dir / "cross_doc_pairs_v1.jsonl", iter(pairs))
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(build_report(summary), encoding="utf-8")

    latest = Path(args.out_root) / "xdoc_element_resolver_v1_latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_dir, target_is_directory=True)

    print(f"Output: {out_dir}")
    print(f"Pairs: {len(pairs)}")
    print(f"Edges with pairs: {summary['edges_with_pairs']} / {summary['input_edges']}")
    print(f"Pair types: {dict(pair_type_counts)}")
    print(f"Source methods: {dict(source_method_counts)}")
    print(f"Target methods: {dict(target_method_counts)}")
    print(f"Fanout buckets: {dict(pair_fanout_buckets)}")
    if v0_cmp:
        print(f"v0 comparison: {json.dumps(v0_cmp, indent=2)}")
    print(f"Latest: {latest}")


if __name__ == "__main__":
    main()
