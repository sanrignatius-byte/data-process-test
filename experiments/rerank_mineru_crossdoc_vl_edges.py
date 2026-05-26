#!/usr/bin/env python3
"""Rerank MinerU cross-document VL edges with caption/context/enriched text.

CLIP visual similarity is a good recall signal, but high-score pairs often match
layout rather than semantics.  This script keeps the CLIP candidates and adds
textual evidence scores from:
  - caption / label
  - local context and content preview
  - MoDora-style enriched_title / enriched_content / keywords
Then it writes a reranked edge list plus diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPOLOGY = ROOT / "data/05_eval/mineru_topology_graph_v1_latest/mineru_topology_graph_v1.json"
DEFAULT_VL_EDGES = ROOT / "data/05_eval/mineru_vl_edges_v1_latest/mineru_vl_edges_v1.jsonl"
DEFAULT_ENRICHED = ROOT / "data/02_enriched/multimodal_elements_enriched.json"


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
    toks = []
    generic = {
        "figure", "fig", "table", "image", "images", "page", "td", "tr",
        "tbody", "thead", "html", "cell", "row", "column", "columns",
    }
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower()):
        if tok in generic:
            continue
        if tok.replace("-", "").isdigit():
            continue
        toks.append(tok)
    return " ".join(toks)


_PLACEHOLDER_TOKENS = (
    "placeholder", "marker", "icon", "logo", "bullet", "checkmark", "tick",
    "arrow", "spacer", "divider", "separator", "decoration",
)


def is_generic_caption(text: str) -> bool:
    """A caption carries no semantics worth reranking on.

    Two cases:
      1. Bare figure/table numbering, e.g. "Figure 9", "Table 2a".
      2. Very short caption whose only words are layout/placeholder tokens
         (small marker/icon images that CLIP still scores highly on layout).
    """
    compacted = compact(text, 80)
    if re.fullmatch(r"(?:figure|fig\.?|table)\s*\d+[a-z]?[.:]?", compacted, flags=re.I):
        return True
    words = re.findall(r"[A-Za-z]{3,}", compacted.lower())
    if not words:
        return True
    content_words = [
        w for w in words
        if w not in {"figure", "fig", "table", "image", "the", "and", "for", "with"}
        and w not in _PLACEHOLDER_TOKENS
    ]
    if not content_words and any(w in _PLACEHOLDER_TOKENS for w in words):
        return True
    return False


def load_enriched(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    obj = read_json(path)
    enriched: dict[str, dict[str, Any]] = {}
    for doc in (obj.get("documents") or {}).values():
        for element_id, elem in (doc.get("elements") or {}).items():
            if isinstance(elem, dict):
                enriched[element_id] = elem
    return enriched


def keywords_text(enriched: dict[str, Any]) -> str:
    meta = enriched.get("enriched_metadata")
    if not isinstance(meta, dict):
        return ""
    kws = meta.get("keywords")
    if isinstance(kws, list):
        return " ".join(str(k) for k in kws)
    return ""


def node_fields(node: dict[str, Any], enriched_by_element: dict[str, dict[str, Any]]) -> dict[str, str]:
    meta = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    element_id = str(node.get("element_id") or node.get("mapped_element_id") or "")
    enriched = enriched_by_element.get(element_id, {})
    caption = compact(" ".join([
        str(meta.get("caption") or ""),
        str(node.get("label") or ""),
        str(enriched.get("caption") or ""),
    ]), 1200)
    context = compact(" ".join([
        str(meta.get("content_preview") or ""),
        str(meta.get("context_before") or ""),
        str(meta.get("context_after") or ""),
        str(enriched.get("content") or ""),
        str(enriched.get("context_before") or ""),
        str(enriched.get("context_after") or ""),
    ]), 3000)
    enriched_text = compact(" ".join([
        str(enriched.get("enriched_title") or ""),
        str(enriched.get("enriched_content") or ""),
        keywords_text(enriched),
    ]), 2500)
    all_text = compact(" ".join([caption, context, enriched_text]), 5000)
    return {
        "caption": clean_for_match(caption, 1200),
        "context": clean_for_match(context, 3000),
        "enriched": clean_for_match(enriched_text, 2500),
        "all": clean_for_match(all_text, 5000),
        "has_enriched": str(bool(enriched_text)),
        "caption_raw": caption,
        "enriched_raw": enriched_text,
    }


def cosine_scores(texts: list[str], pairs: list[tuple[int, int]]) -> list[float]:
    if not texts:
        return []
    if not any(t.strip() for t in texts):
        return [0.0 for _ in pairs]
    try:
        matrix = TfidfVectorizer(
            max_features=4096,
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            norm="l2",
        ).fit_transform(texts)
    except ValueError:
        return [0.0 for _ in pairs]
    out: list[float] = []
    for i, j in pairs:
        out.append(float(cosine_similarity(matrix[i], matrix[j])[0, 0]))
    return out


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    vals = sorted(values)
    def q(p: float) -> float:
        return round(vals[min(len(vals) - 1, max(0, round((len(vals) - 1) * p)))], 6)
    return {
        "min": round(vals[0], 6),
        "p10": q(0.10),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": round(vals[-1], 6),
        "mean": round(sum(vals) / len(vals), 6),
    }


def tier(row: dict[str, Any], min_text: float, min_score: float) -> str:
    if row["combined_score"] >= min_score and row["all_text_sim"] >= min_text:
        return "strong_text_supported"
    if row["combined_score"] >= min_score and row["enriched_sim"] >= min_text:
        return "strong_enriched_supported"
    if row["visual_score"] >= 0.90 and row["all_text_sim"] < min_text:
        return "visual_only_risky"
    if row["all_text_sim"] >= min_text:
        return "text_supported_candidate"
    return "weak_text_support"


def rerank(
    topology_path: Path,
    vl_edges_path: Path,
    enriched_path: Path,
    min_text_sim: float,
    min_combined_score: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    topology = read_json(topology_path)
    nodes = {node["node_id"]: node for node in topology.get("nodes", [])}
    enriched = load_enriched(enriched_path)
    vl_edges = [
        e for e in iter_jsonl(vl_edges_path)
        if e.get("edge_type") == "cross_doc_visual_sim"
        and e.get("source_id") in nodes
        and e.get("target_id") in nodes
    ]

    node_ids = sorted({e["source_id"] for e in vl_edges} | {e["target_id"] for e in vl_edges})
    idx = {nid: i for i, nid in enumerate(node_ids)}
    fields_by_node = {nid: node_fields(nodes[nid], enriched) for nid in node_ids}
    pairs = [(idx[e["source_id"]], idx[e["target_id"]]) for e in vl_edges]

    caption_scores = cosine_scores([fields_by_node[nid]["caption"] for nid in node_ids], pairs)
    context_scores = cosine_scores([fields_by_node[nid]["context"] for nid in node_ids], pairs)
    enriched_scores = cosine_scores([fields_by_node[nid]["enriched"] for nid in node_ids], pairs)
    all_scores = cosine_scores([fields_by_node[nid]["all"] for nid in node_ids], pairs)

    rows: list[dict[str, Any]] = []
    for edge, cap_s, ctx_s, enr_s, all_s in zip(vl_edges, caption_scores, context_scores, enriched_scores, all_scores):
        src_id = edge["source_id"]
        tgt_id = edge["target_id"]
        src_fields = fields_by_node[src_id]
        tgt_fields = fields_by_node[tgt_id]
        visual = float(edge.get("weight") or 0.0)
        generic_both = is_generic_caption(src_fields["caption_raw"]) and is_generic_caption(tgt_fields["caption_raw"])
        text_support = max(cap_s, ctx_s, enr_s, all_s)
        penalty = 0.0
        if generic_both:
            penalty += 0.08
        if text_support < min_text_sim:
            penalty += 0.10
        combined = (
            0.50 * visual
            + 0.15 * cap_s
            + 0.15 * ctx_s
            + 0.15 * enr_s
            + 0.05 * all_s
            - penalty
        )
        row = {
            "source_id": src_id,
            "target_id": tgt_id,
            "doc_id": edge.get("doc_id"),
            "edge_type": "cross_doc_visual_text_rerank",
            "weight": round(combined, 6),
            "combined_score": round(combined, 6),
            "visual_score": round(visual, 6),
            "caption_sim": round(cap_s, 6),
            "context_sim": round(ctx_s, 6),
            "enriched_sim": round(enr_s, 6),
            "all_text_sim": round(all_s, 6),
            "text_support": round(text_support, 6),
            "generic_caption_both": generic_both,
            "source_has_enriched": src_fields["has_enriched"] == "True",
            "target_has_enriched": tgt_fields["has_enriched"] == "True",
            "support_tier": "",
            "metadata": {
                "original_edge_type": edge.get("edge_type"),
                "original_weight": visual,
                "source_element_id": (nodes[src_id].get("element_id") or nodes[src_id].get("mapped_element_id")),
                "target_element_id": (nodes[tgt_id].get("element_id") or nodes[tgt_id].get("mapped_element_id")),
                "source_caption": compact(src_fields["caption_raw"], 280),
                "target_caption": compact(tgt_fields["caption_raw"], 280),
                "source_enriched_preview": compact(src_fields["enriched_raw"], 320),
                "target_enriched_preview": compact(tgt_fields["enriched_raw"], 320),
            },
        }
        row["support_tier"] = tier(row, min_text_sim, min_combined_score)
        rows.append(row)

    rows.sort(key=lambda r: (r["weight"], r["visual_score"]), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank_after_rerank"] = rank

    summary = {
        "builder": "mineru_crossdoc_visual_text_rerank_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_topology": str(topology_path),
        "source_vl_edges": str(vl_edges_path),
        "source_enriched": str(enriched_path),
        "edge_count": len(rows),
        "unique_nodes": len(node_ids),
        "enriched_element_count": len(enriched),
        "nodes_with_enriched_text": sum(1 for nid in node_ids if fields_by_node[nid]["has_enriched"] == "True"),
        "thresholds": {
            "min_text_sim": min_text_sim,
            "min_combined_score": min_combined_score,
        },
        "score_quantiles": {
            "visual": quantiles([r["visual_score"] for r in rows]),
            "caption": quantiles([r["caption_sim"] for r in rows]),
            "context": quantiles([r["context_sim"] for r in rows]),
            "enriched": quantiles([r["enriched_sim"] for r in rows]),
            "all_text": quantiles([r["all_text_sim"] for r in rows]),
            "combined": quantiles([r["weight"] for r in rows]),
        },
        "tier_counts": dict(Counter(r["support_tier"] for r in rows)),
        "generic_caption_both": sum(1 for r in rows if r["generic_caption_both"]),
        "generic_caption_both_top100": sum(1 for r in rows[:100] if r["generic_caption_both"]),
        "top20": rows[:20],
    }
    return rows, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    q = summary["score_quantiles"]
    lines = [
        "# Cross-doc VL Text Rerank",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| edges | {summary['edge_count']} |",
        f"| nodes with enriched text | {summary['nodes_with_enriched_text']} / {summary['unique_nodes']} |",
        f"| visual p50 / p90 | {q['visual'].get('p50')} / {q['visual'].get('p90')} |",
        f"| caption sim p50 / p90 | {q['caption'].get('p50')} / {q['caption'].get('p90')} |",
        f"| context sim p50 / p90 | {q['context'].get('p50')} / {q['context'].get('p90')} |",
        f"| enriched sim p50 / p90 | {q['enriched'].get('p50')} / {q['enriched'].get('p90')} |",
        f"| all-text sim p50 / p90 | {q['all_text'].get('p50')} / {q['all_text'].get('p90')} |",
        f"| combined p50 / p90 | {q['combined'].get('p50')} / {q['combined'].get('p90')} |",
        f"| generic-caption-both top100 | {summary['generic_caption_both_top100']} |",
        "",
        "## Tier Counts",
        "",
    ]
    for name, count in sorted(summary["tier_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{name}`: {count}")
    lines.extend([
        "",
        "## Top Reranked Edges",
        "",
    ])
    for i, row in enumerate(summary["top20"][:12], 1):
        meta = row["metadata"]
        lines.extend([
            f"### {i}. {row['source_id']} → {row['target_id']}",
            f"- combined={row['weight']} visual={row['visual_score']} caption={row['caption_sim']} context={row['context_sim']} enriched={row['enriched_sim']} all={row['all_text_sim']}",
            f"- tier: `{row['support_tier']}` generic_both={row['generic_caption_both']}",
            f"- source caption: {meta['source_caption']}",
            f"- target caption: {meta['target_caption']}",
            f"- source enriched: {meta['source_enriched_preview']}",
            f"- target enriched: {meta['target_enriched_preview']}",
            "",
        ])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_latest(out_dir: Path) -> None:
    latest = ROOT / "data/05_eval/mineru_crossdoc_text_rerank_v1_latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_dir.resolve())
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank MinerU cross-doc VL edges with text evidence")
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY))
    parser.add_argument("--vl-edges", default=str(DEFAULT_VL_EDGES))
    parser.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--min-text-sim", type=float, default=0.03)
    parser.add_argument("--min-combined-score", type=float, default=0.45)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / f"data/05_eval/mineru_crossdoc_text_rerank_v1_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, summary = rerank(
        Path(args.topology),
        Path(args.vl_edges),
        Path(args.enriched),
        args.min_text_sim,
        args.min_combined_score,
    )
    write_jsonl(out_dir / "mineru_crossdoc_text_rerank_edges_v1.jsonl", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir, summary)
    update_latest(out_dir)

    print(f"[ok] wrote {out_dir / 'report.md'}")
    print(f"edges={len(rows)} tiers={summary['tier_counts']}")


if __name__ == "__main__":
    main()
