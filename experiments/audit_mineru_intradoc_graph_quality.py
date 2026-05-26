#!/usr/bin/env python3
"""Audit intra-document MinerU soft edges against hard local relations.

The question this answers is deliberately narrower than the cross-document
audit: for paragraph -> figure/table links inside one paper, how do VL/text
signals compare with the existing deterministic graph edges?

We treat MinerU regex_reference paragraph -> visual edges as a high-precision
silver anchor, then compare:
  - same_page_cross_type paragraph <-> visual candidates
  - CLIP text_describes_figure candidates
  - their union with caption/context/enriched-text support scores
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPOLOGY = ROOT / "data/05_eval/mineru_topology_graph_v1_latest/mineru_topology_graph_v1.json"
DEFAULT_VL_EDGES = ROOT / "data/05_eval/mineru_vl_edges_v1_latest/mineru_vl_edges_v1.jsonl"
DEFAULT_ENRICHED = ROOT / "data/02_enriched/multimodal_elements_enriched.json"

VISUAL_TYPES = {"figure", "table"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compact(value: Any, limit: int = 2000) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.split())[:limit]


def clean_for_match(value: Any, limit: int = 5000) -> str:
    text = compact(value, limit)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[$_{}^|=+*/<>\\[\\](),;:]", " ", text)
    generic = {
        "figure", "fig", "table", "image", "images", "page", "shown",
        "using", "with", "from", "this", "that", "their", "there",
        "which", "where", "have", "has", "were", "are", "the", "and",
        "td", "tr", "tbody", "thead", "html", "cell", "row", "column",
        "columns",
    }
    toks = []
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower()):
        if tok in generic:
            continue
        if tok.replace("-", "").isdigit():
            continue
        toks.append(tok)
    return " ".join(toks)


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    vals = sorted(values)

    def q(p: float) -> float:
        idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * p)))
        return round(vals[idx], 6)

    return {
        "min": round(vals[0], 6),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": round(vals[-1], 6),
        "mean": round(sum(vals) / len(vals), 6),
    }


def pct(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def load_enriched(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    obj = read_json(path)
    out: dict[str, dict[str, Any]] = {}
    for doc in (obj.get("documents") or {}).values():
        for element_id, elem in (doc.get("elements") or {}).items():
            if isinstance(elem, dict):
                out[str(element_id)] = elem
    return out


def keywords_text(enriched: dict[str, Any]) -> str:
    meta = enriched.get("enriched_metadata")
    if not isinstance(meta, dict):
        return ""
    kws = meta.get("keywords")
    if isinstance(kws, list):
        return " ".join(str(k) for k in kws)
    return ""


def enriched_kind(enriched: dict[str, Any]) -> str:
    meta = enriched.get("enriched_metadata")
    if not isinstance(meta, dict):
        return ""
    return str(meta.get("figure_type") or meta.get("visual_type") or "")


def node_enriched(node: dict[str, Any], enriched_by_element: dict[str, dict[str, Any]]) -> dict[str, Any]:
    element_id = str(node.get("element_id") or node.get("mapped_element_id") or "")
    return enriched_by_element.get(element_id, {})


def node_caption(node: dict[str, Any], enriched_by_element: dict[str, dict[str, Any]]) -> str:
    meta = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    enriched = node_enriched(node, enriched_by_element)
    return clean_for_match(" ".join([
        str(meta.get("caption") or ""),
        str(node.get("label") or ""),
        str(enriched.get("caption") or ""),
    ]), 1800)


def node_context(node: dict[str, Any], enriched_by_element: dict[str, dict[str, Any]]) -> str:
    meta = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    enriched = node_enriched(node, enriched_by_element)
    return clean_for_match(" ".join([
        str(meta.get("content_preview") or ""),
        str(meta.get("context_before") or ""),
        str(meta.get("context_after") or ""),
        str(enriched.get("content") or ""),
        str(enriched.get("context_before") or ""),
        str(enriched.get("context_after") or ""),
    ]), 3200)


def node_enriched_text(node: dict[str, Any], enriched_by_element: dict[str, dict[str, Any]]) -> str:
    enriched = node_enriched(node, enriched_by_element)
    return clean_for_match(" ".join([
        str(enriched.get("enriched_title") or ""),
        str(enriched.get("enriched_content") or ""),
        keywords_text(enriched),
    ]), 3200)


def paragraph_text(node: dict[str, Any]) -> str:
    return clean_for_match(" ".join([
        str(node.get("text_snippet") or ""),
        str(node.get("label") or ""),
    ]), 1800)


def pair_cosine(left: list[str], right: list[str]) -> list[float]:
    if not left:
        return []
    if not any(t.strip() for t in left + right):
        return [0.0 for _ in left]
    try:
        mat = TfidfVectorizer(
            max_features=8192,
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            norm="l2",
        ).fit_transform(left + right)
    except ValueError:
        return [0.0 for _ in left]
    n = len(left)
    scores = mat[:n].multiply(mat[n:]).sum(axis=1)
    return [float(x) for x in np.asarray(scores).reshape(-1)]


def canonical_para_visual_pair(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> tuple[str, str] | None:
    src = nodes.get(edge.get("source_id"), {})
    tgt = nodes.get(edge.get("target_id"), {})
    if src.get("node_type") == "paragraph" and tgt.get("node_type") in VISUAL_TYPES:
        return str(edge["source_id"]), str(edge["target_id"])
    if tgt.get("node_type") == "paragraph" and src.get("node_type") in VISUAL_TYPES:
        return str(edge["target_id"]), str(edge["source_id"])
    return None


def build_pair_sets(
    topology_edges: list[dict[str, Any]],
    vl_edges: list[dict[str, Any]],
    nodes: dict[str, dict[str, Any]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], dict[tuple[str, str], dict[str, Any]]]:
    regex: set[tuple[str, str]] = set()
    same_page: set[tuple[str, str]] = set()
    clip: dict[tuple[str, str], dict[str, Any]] = {}

    for edge in topology_edges:
        if edge.get("edge_type") == "element_ref":
            meta = edge.get("metadata") if isinstance(edge.get("metadata"), dict) else {}
            if meta.get("original_edge_type") == "regex_reference":
                pair = canonical_para_visual_pair(edge, nodes)
                if pair:
                    regex.add(pair)
        elif edge.get("edge_type") == "same_page_cross_type":
            pair = canonical_para_visual_pair(edge, nodes)
            if pair:
                same_page.add(pair)

    for edge in vl_edges:
        if edge.get("edge_type") != "text_describes_figure":
            continue
        pair = canonical_para_visual_pair(edge, nodes)
        if not pair:
            continue
        old = clip.get(pair)
        if old is None or float(edge.get("weight") or 0.0) > float(old.get("weight") or 0.0):
            clip[pair] = edge
    return regex, same_page, clip


def target_map(pairs: set[tuple[str, str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for para, visual in pairs:
        out[visual].add(para)
    return out


def set_metrics(
    name: str,
    pairs: set[tuple[str, str]],
    regex_pairs: set[tuple[str, str]],
    nodes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    regex_targets = target_map(regex_pairs)
    candidate_targets = target_map(pairs)
    exact_hits = pairs & regex_pairs
    visual_targets = {visual for _, visual in pairs}
    target_hits = set(regex_targets) & visual_targets
    return {
        "name": name,
        "pairs": len(pairs),
        "exact_regex_hits": len(exact_hits),
        "exact_precision_vs_regex": pct(len(exact_hits), len(pairs)),
        "exact_recall_vs_regex": pct(len(exact_hits), len(regex_pairs)),
        "visual_targets": len(visual_targets),
        "visual_targets_with_regex": len(regex_targets),
        "target_level_hits_vs_regex": len(target_hits),
        "target_level_recall_vs_regex": pct(len(target_hits), len(regex_targets)),
        "doc_count": len({nodes[p].get("doc_id") for p, _ in pairs if p in nodes}),
    }


def rank_metrics(
    rows: list[dict[str, Any]],
    regex_pairs: set[tuple[str, str]],
    key: str,
    top_ns: tuple[int, ...] = (100, 250, 500, 1000),
) -> dict[str, Any]:
    regex_targets = target_map(regex_pairs)
    sorted_rows = sorted(rows, key=lambda row: (row.get(key, 0.0), row.get("clip_score", 0.0)), reverse=True)
    out: dict[str, Any] = {}
    for n in top_ns:
        subset = sorted_rows[:n]
        exact = sum(1 for row in subset if row["pair"] in regex_pairs)
        target = sum(1 for row in subset if row["visual_id"] in regex_targets)
        out[f"top{n}"] = {
            "n": len(subset),
            "exact_hits": exact,
            "exact_precision": pct(exact, len(subset)),
            "target_has_regex": target,
            "target_precision": pct(target, len(subset)),
        }
    return out


def per_visual_topk(
    rows: list[dict[str, Any]],
    regex_pairs: set[tuple[str, str]],
    key: str,
    k_values: tuple[int, ...] = (1, 3),
) -> dict[str, Any]:
    regex_targets = target_map(regex_pairs)
    by_visual: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["visual_id"] in regex_targets:
            by_visual[row["visual_id"]].append(row)

    out: dict[str, Any] = {
        "visual_targets_with_regex": len(regex_targets),
        "visual_targets_with_candidates": len(by_visual),
    }
    for k in k_values:
        hit = 0
        for visual_id, visual_rows in by_visual.items():
            ranked = sorted(visual_rows, key=lambda row: (row.get(key, 0.0), row.get("clip_score", 0.0)), reverse=True)
            if any(row["pair"] in regex_pairs for row in ranked[:k]):
                hit += 1
        out[f"top{k}_exact_target_hit"] = hit
        out[f"top{k}_exact_target_hit_rate_all_regex_targets"] = pct(hit, len(regex_targets))
        out[f"top{k}_exact_target_hit_rate_candidate_targets"] = pct(hit, len(by_visual))
    return out


def make_rows(
    pairs: set[tuple[str, str]],
    clip_edges: dict[tuple[str, str], dict[str, Any]],
    same_page_pairs: set[tuple[str, str]],
    regex_pairs: set[tuple[str, str]],
    nodes: dict[str, dict[str, Any]],
    enriched_by_element: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    para_texts: list[str] = []
    visual_captions: list[str] = []
    visual_contexts: list[str] = []
    visual_enriched: list[str] = []
    visual_all: list[str] = []

    for para_id, visual_id in sorted(pairs):
        para = nodes[para_id]
        visual = nodes[visual_id]
        p_text = paragraph_text(para)
        cap = node_caption(visual, enriched_by_element)
        ctx = node_context(visual, enriched_by_element)
        enr = node_enriched_text(visual, enriched_by_element)
        para_texts.append(p_text)
        visual_captions.append(cap)
        visual_contexts.append(ctx)
        visual_enriched.append(enr)
        visual_all.append(" ".join([cap, ctx, enr]))
        clip_edge = clip_edges.get((para_id, visual_id), {})
        meta = clip_edge.get("metadata") if isinstance(clip_edge.get("metadata"), dict) else {}
        enriched = node_enriched(visual, enriched_by_element)
        rows.append({
            "pair": (para_id, visual_id),
            "paragraph_id": para_id,
            "visual_id": visual_id,
            "doc_id": para.get("doc_id"),
            "visual_type": visual.get("node_type"),
            "in_clip": (para_id, visual_id) in clip_edges,
            "in_same_page": (para_id, visual_id) in same_page_pairs,
            "is_regex_exact": (para_id, visual_id) in regex_pairs,
            "clip_score": float(clip_edge.get("weight") or 0.0),
            "clip_rank_for_figure": meta.get("rank_for_figure"),
            "visual_enriched_kind": enriched_kind(enriched),
            "paragraph_preview": compact(para.get("text_snippet"), 220),
            "visual_label": compact(visual.get("label"), 160),
        })

    cap_scores = pair_cosine(para_texts, visual_captions)
    ctx_scores = pair_cosine(para_texts, visual_contexts)
    enr_scores = pair_cosine(para_texts, visual_enriched)
    all_scores = pair_cosine(para_texts, visual_all)

    for row, cap, ctx, enr, all_s in zip(rows, cap_scores, ctx_scores, enr_scores, all_scores):
        row["caption_sim"] = round(cap, 6)
        row["context_sim"] = round(ctx, 6)
        row["enriched_sim"] = round(enr, 6)
        row["all_text_sim"] = round(all_s, 6)
        text_support = max(cap, ctx, enr, all_s)
        same_page_bonus = 0.20 if row["in_same_page"] else 0.0
        combined = (
            0.42 * row["clip_score"]
            + 0.22 * cap
            + 0.16 * ctx
            + 0.12 * enr
            + 0.08 * all_s
            + same_page_bonus
        )
        row["text_support"] = round(text_support, 6)
        row["combined_score"] = round(combined, 6)
    return rows


def threshold_table(rows: list[dict[str, Any]], regex_pairs: set[tuple[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for threshold in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20):
        subset = [row for row in rows if row["text_support"] >= threshold]
        exact = sum(1 for row in subset if row["pair"] in regex_pairs)
        out.append({
            "text_support_ge": threshold,
            "pairs": len(subset),
            "exact_hits": exact,
            "exact_precision": pct(exact, len(subset)),
            "exact_recall": pct(exact, len(regex_pairs)),
        })
    return out


def audit(topology_path: Path, vl_edges_path: Path, enriched_path: Path) -> dict[str, Any]:
    topology = read_json(topology_path)
    nodes = {node["node_id"]: node for node in topology.get("nodes", [])}
    topology_edges = topology.get("edges", [])
    vl_edges = iter_jsonl(vl_edges_path)
    enriched_by_element = load_enriched(enriched_path)

    regex_pairs, same_page_pairs, clip_edges = build_pair_sets(topology_edges, vl_edges, nodes)
    clip_pairs = set(clip_edges)
    union_pairs = same_page_pairs | clip_pairs

    rows_by_set = {
        "same_page": make_rows(same_page_pairs, clip_edges, same_page_pairs, regex_pairs, nodes, enriched_by_element),
        "clip": make_rows(clip_pairs, clip_edges, same_page_pairs, regex_pairs, nodes, enriched_by_element),
        "union": make_rows(union_pairs, clip_edges, same_page_pairs, regex_pairs, nodes, enriched_by_element),
    }

    edge_counts = Counter(edge.get("edge_type") for edge in topology_edges)
    visual_nodes = [node for node in nodes.values() if node.get("node_type") in VISUAL_TYPES]
    regex_targets = target_map(regex_pairs)
    clip_targets = target_map(clip_pairs)
    same_page_targets = target_map(same_page_pairs)
    union_targets = target_map(union_pairs)

    result = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "topology": str(topology_path),
            "vl_edges": str(vl_edges_path),
            "enriched": str(enriched_path),
        },
        "graph_counts": {
            "nodes": len(nodes),
            "visual_nodes": len(visual_nodes),
            "topology_edge_counts": dict(edge_counts),
            "regex_text_visual_silver_pairs": len(regex_pairs),
            "same_page_para_visual_pairs": len(same_page_pairs),
            "clip_text_visual_pairs": len(clip_pairs),
            "union_pairs": len(union_pairs),
            "visual_targets_with_regex": len(regex_targets),
            "visual_targets_with_same_page": len(same_page_targets),
            "visual_targets_with_clip": len(clip_targets),
            "visual_targets_with_union": len(union_targets),
        },
        "set_metrics": {
            "regex_silver": set_metrics("regex_silver", regex_pairs, regex_pairs, nodes),
            "same_page": set_metrics("same_page", same_page_pairs, regex_pairs, nodes),
            "clip": set_metrics("clip", clip_pairs, regex_pairs, nodes),
            "union": set_metrics("union", union_pairs, regex_pairs, nodes),
        },
        "score_quantiles": {
            name: {
                "clip_score": quantiles([row["clip_score"] for row in rows if row["clip_score"] > 0]),
                "caption_sim": quantiles([row["caption_sim"] for row in rows]),
                "context_sim": quantiles([row["context_sim"] for row in rows]),
                "enriched_sim": quantiles([row["enriched_sim"] for row in rows]),
                "all_text_sim": quantiles([row["all_text_sim"] for row in rows]),
                "text_support": quantiles([row["text_support"] for row in rows]),
                "combined_score": quantiles([row["combined_score"] for row in rows]),
            }
            for name, rows in rows_by_set.items()
        },
        "rank_metrics": {
            name: {
                "by_clip_score": rank_metrics(rows, regex_pairs, "clip_score"),
                "by_text_support": rank_metrics(rows, regex_pairs, "text_support"),
                "by_combined_score": rank_metrics(rows, regex_pairs, "combined_score"),
            }
            for name, rows in rows_by_set.items()
        },
        "per_visual_ranking": {
            name: {
                "by_clip_score": per_visual_topk(rows, regex_pairs, "clip_score"),
                "by_text_support": per_visual_topk(rows, regex_pairs, "text_support"),
                "by_combined_score": per_visual_topk(rows, regex_pairs, "combined_score"),
            }
            for name, rows in rows_by_set.items()
        },
        "text_support_thresholds": {
            name: threshold_table(rows, regex_pairs)
            for name, rows in rows_by_set.items()
        },
        "clip_hit_rank_distribution": dict(sorted(Counter(
            row["clip_rank_for_figure"]
            for row in rows_by_set["clip"]
            if row["is_regex_exact"] and isinstance(row["clip_rank_for_figure"], int)
        ).items())),
        "top_union_by_combined": sorted(rows_by_set["union"], key=lambda row: row["combined_score"], reverse=True)[:30],
        "top_clip_by_text_support": sorted(rows_by_set["clip"], key=lambda row: row["text_support"], reverse=True)[:30],
    }
    return result


def write_report(out_dir: Path, result: dict[str, Any]) -> None:
    counts = result["graph_counts"]
    sets = result["set_metrics"]
    per_visual = result["per_visual_ranking"]
    thresholds = result["text_support_thresholds"]
    ranks = result["rank_metrics"]

    lines = [
        "# MinerU Intra-doc Graph Quality Audit",
        "",
        "## Summary",
        "",
        "| Slice | Pairs | Exact hits vs regex | Precision | Recall | Target-level recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("regex_silver", "same_page", "clip", "union"):
        row = sets[name]
        lines.append(
            f"| {name} | {row['pairs']} | {row['exact_regex_hits']} | "
            f"{row['exact_precision_vs_regex']:.1%} | {row['exact_recall_vs_regex']:.1%} | "
            f"{row['target_level_recall_vs_regex']:.1%} |"
        )
    lines.extend([
        "",
        "## Counts",
        "",
        f"- visual nodes: **{counts['visual_nodes']}**",
        f"- regex text→visual silver pairs: **{counts['regex_text_visual_silver_pairs']}**",
        f"- same-page paragraph↔visual pairs: **{counts['same_page_para_visual_pairs']}**",
        f"- CLIP text→visual pairs: **{counts['clip_text_visual_pairs']}**",
        f"- union candidates: **{counts['union_pairs']}**",
        f"- visual targets covered by regex / same-page / CLIP / union: "
        f"**{counts['visual_targets_with_regex']} / {counts['visual_targets_with_same_page']} / "
        f"{counts['visual_targets_with_clip']} / {counts['visual_targets_with_union']}**",
        "",
        "## Per-visual Top-k Exact Hits",
        "",
        "| Candidate set | Rank score | targets with candidates | top1 hit rate | top3 hit rate |",
        "|---|---|---:|---:|---:|",
    ])
    for name in ("same_page", "clip", "union"):
        for key, label in (("by_clip_score", "clip"), ("by_text_support", "text"), ("by_combined_score", "combined")):
            row = per_visual[name][key]
            lines.append(
                f"| {name} | {label} | {row['visual_targets_with_candidates']} | "
                f"{row['top1_exact_target_hit_rate_candidate_targets']:.1%} | "
                f"{row['top3_exact_target_hit_rate_candidate_targets']:.1%} |"
            )
    lines.extend([
        "",
        "## Top Precision Diagnostics",
        "",
        "| Candidate set | Rank score | top100 exact precision | top500 exact precision | top100 target precision |",
        "|---|---|---:|---:|---:|",
    ])
    for name in ("same_page", "clip", "union"):
        for key, label in (("by_clip_score", "clip"), ("by_text_support", "text"), ("by_combined_score", "combined")):
            row100 = ranks[name][key]["top100"]
            row500 = ranks[name][key]["top500"]
            lines.append(
                f"| {name} | {label} | {row100['exact_precision']:.1%} | "
                f"{row500['exact_precision']:.1%} | {row100['target_precision']:.1%} |"
            )
    lines.extend([
        "",
        "## Text-support Thresholds",
        "",
        "| Candidate set | threshold | pairs | exact precision | exact recall |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in ("same_page", "clip", "union"):
        for row in thresholds[name]:
            lines.append(
                f"| {name} | >= {row['text_support_ge']} | {row['pairs']} | "
                f"{row['exact_precision']:.1%} | {row['exact_recall']:.1%} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "1. `regex_reference` remains the high-precision intra-doc anchor. It is the only relation here that directly encodes explicit textual references.",
        "2. `same_page_cross_type` is a useful but noisy locality candidate layer. It has much better exact overlap with regex anchors than raw CLIP text→visual, but it is positional rather than semantic.",
        "3. CLIP text→visual gives broad visual coverage and often finds a semantically nearby paragraph, but exact reference matching is weak. It should not replace regex anchors.",
        "4. Caption/context/enriched matching is useful as a validation/rerank signal, especially over the union candidate pool; however, exact regex recovery is still limited because many true referring paragraphs mention only a figure number and not the caption semantics.",
        "",
        "## Top Union Candidates by Combined Score",
        "",
    ])
    for i, row in enumerate(result["top_union_by_combined"][:12], 1):
        lines.extend([
            f"### {i}. {row['paragraph_id']} → {row['visual_id']}",
            f"- exact_regex={row['is_regex_exact']} same_page={row['in_same_page']} clip={row['in_clip']} "
            f"combined={row['combined_score']} text={row['text_support']} clip_score={row['clip_score']}",
            f"- paragraph: {row['paragraph_preview']}",
            f"- visual: {row['visual_label']}",
            "",
        ])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit intra-doc MinerU graph quality")
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY))
    parser.add_argument("--vl-edges", default=str(DEFAULT_VL_EDGES))
    parser.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / f"data/05_eval/mineru_intradoc_quality_audit_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    result = audit(Path(args.topology), Path(args.vl_edges), Path(args.enriched))
    (out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir, result)

    latest = ROOT / "data/05_eval/mineru_intradoc_quality_audit_latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_dir.resolve())
    except OSError:
        pass

    print(f"[ok] wrote {out_dir / 'report.md'}")
    print(json.dumps(result["set_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
