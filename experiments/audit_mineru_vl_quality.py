#!/usr/bin/env python3
"""Audit MinerU CLIP/VL edge quality against local structural baselines.

The strict LaTeX-v2 graph and the old_53 MinerU/CLIP run currently have no
document overlap, so this script reports:
  1. Cross-document visual-edge quality diagnostics and readable samples.
  2. Same-document text→visual CLIP edges against MinerU regex references as a
     silver label.
  3. Structural comparison with the LaTeX-v2 intra-document graph.
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

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPOLOGY = ROOT / "data/05_eval/mineru_topology_graph_v1_latest/mineru_topology_graph_v1.json"
DEFAULT_VL_EDGES = ROOT / "data/05_eval/mineru_vl_edges_v1_latest/mineru_vl_edges_v1.jsonl"
DEFAULT_CROSSDOC = ROOT / "data/05_eval/mineru_crossdoc_bridges_v1_latest/summary.json"
DEFAULT_HUB = ROOT / "data/05_eval/mineru_hub_candidates_v1_latest/summary.json"
DEFAULT_LATEX_REPORT = ROOT / "data/01_graphs/latex_graph_topology_report_v2.json"
DEFAULT_LATEX_REF = ROOT / "data/01_graphs/latex_reference_graph_v2.json"
DEFAULT_OLD53 = ROOT / "data/doc_lists/old_53_docs.txt"

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


def compact(value: Any, limit: int = 220) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.split())[:limit]


def tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", text.lower())
        if t not in {
            "figure", "fig", "table", "image", "images", "page", "shown",
            "using", "with", "from", "this", "that", "their", "there",
            "which", "where", "have", "has", "were", "are", "the", "and",
        }
    }


def jaccard(a: str, b: str) -> float:
    ta = tokens(a)
    tb = tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    vals = sorted(values)
    def q(p: float) -> float:
        idx = min(len(vals) - 1, max(0, round((len(vals) - 1) * p)))
        return round(vals[idx], 6)
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


def pct(n: int, d: int) -> float:
    return round(n / d, 4) if d else 0.0


def node_caption(node: dict[str, Any]) -> str:
    meta = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    return compact(meta.get("caption") or node.get("label") or meta.get("content_preview") or "", 320)


def node_context(node: dict[str, Any]) -> str:
    meta = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    parts = [
        node.get("text_snippet") or "",
        meta.get("content_preview") or "",
        meta.get("context_before") or "",
        meta.get("context_after") or "",
    ]
    return compact(" ".join(str(p) for p in parts if p), 600)


def is_generic_caption(caption: str) -> bool:
    text = compact(caption, 80)
    return bool(re.fullmatch(r"(?:figure|fig\.?|table)\s*\d+[a-z]?", text.strip(), flags=re.I))


def image_info(node: dict[str, Any]) -> dict[str, Any]:
    meta = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    path = meta.get("image_path") or ""
    out = {"path": path, "exists": False}
    if not path:
        return out
    p = Path(path)
    out["exists"] = p.exists()
    if not p.exists():
        return out
    out["bytes"] = p.stat().st_size
    try:
        with Image.open(p) as im:
            out["width"], out["height"] = im.size
    except Exception as exc:  # noqa: BLE001 - audit only
        out["error"] = repr(exc)
    return out


def edge_sample(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    src = nodes.get(edge["source_id"], {})
    tgt = nodes.get(edge["target_id"], {})
    src_cap = node_caption(src)
    tgt_cap = node_caption(tgt)
    return {
        "edge_type": edge.get("edge_type"),
        "weight": edge.get("weight"),
        "source_id": edge.get("source_id"),
        "target_id": edge.get("target_id"),
        "source_doc": src.get("doc_id"),
        "target_doc": tgt.get("doc_id"),
        "source_type": src.get("node_type"),
        "target_type": tgt.get("node_type"),
        "source_caption": src_cap,
        "target_caption": tgt_cap,
        "caption_jaccard": round(jaccard(src_cap, tgt_cap), 4),
        "source_generic_caption": is_generic_caption(src_cap),
        "target_generic_caption": is_generic_caption(tgt_cap),
        "source_image": image_info(src),
        "target_image": image_info(tgt),
    }


def build_silver_regex_pairs(topology_edges: list[dict[str, Any]], nodes: dict[str, dict[str, Any]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for edge in topology_edges:
        if edge.get("edge_type") != "element_ref":
            continue
        meta = edge.get("metadata") if isinstance(edge.get("metadata"), dict) else {}
        if meta.get("original_edge_type") != "regex_reference":
            continue
        src = nodes.get(edge.get("source_id"), {})
        tgt = nodes.get(edge.get("target_id"), {})
        if src.get("node_type") == "paragraph" and tgt.get("node_type") in VISUAL_TYPES:
            pairs.add((edge["source_id"], edge["target_id"]))
    return pairs


def canonical_para_visual_pair(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> tuple[str, str] | None:
    src = nodes.get(edge.get("source_id"), {})
    tgt = nodes.get(edge.get("target_id"), {})
    if src.get("node_type") == "paragraph" and tgt.get("node_type") in VISUAL_TYPES:
        return edge["source_id"], edge["target_id"]
    if tgt.get("node_type") == "paragraph" and src.get("node_type") in VISUAL_TYPES:
        return edge["target_id"], edge["source_id"]
    return None


def audit(topology_path: Path, vl_edges_path: Path, latex_report_path: Path, latex_ref_path: Path, old53_path: Path) -> dict[str, Any]:
    topology = read_json(topology_path)
    nodes = {node["node_id"]: node for node in topology.get("nodes", [])}
    topology_edges = topology.get("edges", [])
    vl_edges = iter_jsonl(vl_edges_path)

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in vl_edges:
        by_type[str(edge.get("edge_type"))].append(edge)

    text_edges = by_type.get("text_describes_figure", [])
    cross_edges = by_type.get("cross_doc_visual_sim", [])
    same_visual_edges = by_type.get("visual_similarity", [])
    formula_edges = by_type.get("formula_similarity", [])

    silver = build_silver_regex_pairs(topology_edges, nodes)
    clip_pairs = {(e["source_id"], e["target_id"]) for e in text_edges}
    same_page_pairs = {
        pair
        for edge in topology_edges
        if edge.get("edge_type") == "same_page_cross_type"
        for pair in [canonical_para_visual_pair(edge, nodes)]
        if pair is not None
    }
    same_page_total = sum(1 for edge in topology_edges if edge.get("edge_type") == "same_page_cross_type")
    exact_hits = clip_pairs & silver
    same_page_hits = same_page_pairs & silver
    silver_targets: dict[str, set[str]] = defaultdict(set)
    clip_targets: dict[str, set[str]] = defaultdict(set)
    for para, vis in silver:
        silver_targets[vis].add(para)
    for para, vis in clip_pairs:
        clip_targets[vis].add(para)
    target_overlap = {
        vis for vis, paras in silver_targets.items()
        if paras & clip_targets.get(vis, set())
    }

    rank_hits: Counter[int] = Counter()
    for edge in text_edges:
        if (edge["source_id"], edge["target_id"]) in silver:
            rank = (edge.get("metadata") or {}).get("rank_for_figure")
            if isinstance(rank, int):
                rank_hits[rank] += 1

    visual_nodes = [n for n in nodes.values() if n.get("node_type") in VISUAL_TYPES]
    described_visuals = {e["target_id"] for e in text_edges}
    visual_with_silver = set(silver_targets)

    cross_scores = [float(e.get("weight") or 0.0) for e in cross_edges]
    cross_by_source: Counter[str] = Counter(e["source_id"] for e in cross_edges)
    cross_doc_pairs: Counter[str] = Counter(e.get("doc_id", "") for e in cross_edges)
    generic_both = 0
    high_generic_zero_text = []
    for edge in cross_edges:
        src = nodes.get(edge["source_id"], {})
        tgt = nodes.get(edge["target_id"], {})
        src_cap, tgt_cap = node_caption(src), node_caption(tgt)
        if is_generic_caption(src_cap) and is_generic_caption(tgt_cap):
            generic_both += 1
        if float(edge.get("weight") or 0.0) >= 0.80 and jaccard(src_cap, tgt_cap) == 0.0:
            high_generic_zero_text.append(edge)

    cross_sorted = sorted(cross_edges, key=lambda e: float(e.get("weight") or 0.0), reverse=True)
    def generic_rate(subset: list[dict[str, Any]]) -> dict[str, Any]:
        both = 0
        one = 0
        for edge in subset:
            src = nodes.get(edge["source_id"], {})
            tgt = nodes.get(edge["target_id"], {})
            src_gen = is_generic_caption(node_caption(src))
            tgt_gen = is_generic_caption(node_caption(tgt))
            both += int(src_gen and tgt_gen)
            one += int(src_gen or tgt_gen)
        return {
            "n": len(subset),
            "generic_both": both,
            "generic_both_rate": pct(both, len(subset)),
            "generic_one_or_more": one,
            "generic_one_or_more_rate": pct(one, len(subset)),
        }
    undirected_cross_pairs = {
        tuple(sorted([edge["source_id"], edge["target_id"]]))
        for edge in cross_edges
    }
    high_samples = [edge_sample(e, nodes) for e in cross_sorted[:12]]
    mid_start = max(0, len(cross_sorted) // 2 - 6)
    mid_samples = [edge_sample(e, nodes) for e in cross_sorted[mid_start:mid_start + 12]]
    risk_samples = [edge_sample(e, nodes) for e in high_generic_zero_text[:12]]
    same_visual_samples = [
        edge_sample(e, nodes)
        for e in sorted(same_visual_edges, key=lambda x: float(x.get("weight") or 0.0), reverse=True)[:12]
    ]

    latex_report = read_json(latex_report_path) if latex_report_path.exists() else {}
    latex_ref_docs = set()
    if latex_ref_path.exists():
        latex_ref = read_json(latex_ref_path)
        latex_ref_docs = set((latex_ref.get("documents") or {}).keys())
    old_docs = [
        line.strip()
        for line in old53_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    overlap_docs = sorted(set(old_docs) & latex_ref_docs)

    result = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "topology": str(topology_path),
            "vl_edges": str(vl_edges_path),
            "latex_report": str(latex_report_path),
            "latex_reference_graph": str(latex_ref_path),
        },
        "cross_doc_visual_quality": {
            "edge_count": len(cross_edges),
            "unique_source_visuals": len(cross_by_source),
            "avg_edges_per_source": round(len(cross_edges) / max(1, len(cross_by_source)), 3),
            "score_quantiles": quantiles(cross_scores),
            "generic_caption_both_count": generic_both,
            "generic_caption_both_rate": pct(generic_both, len(cross_edges)),
            "generic_rate_top100": generic_rate(cross_sorted[:100]),
            "generic_rate_top500": generic_rate(cross_sorted[:500]),
            "undirected_unique_pairs": len(undirected_cross_pairs),
            "directed_to_undirected_ratio": round(len(cross_edges) / max(1, len(undirected_cross_pairs)), 3),
            "top_doc_pairs": cross_doc_pairs.most_common(20),
            "top_high_score_samples": high_samples,
            "median_score_samples": mid_samples,
            "risk_samples_high_score_zero_caption_overlap": risk_samples,
        },
        "intra_doc_text_visual_vs_regex": {
            "clip_text_visual_edges": len(text_edges),
            "silver_regex_text_visual_edges": len(silver),
            "same_page_cross_type_total_edges": same_page_total,
            "same_page_para_visual_edges": len(same_page_pairs),
            "exact_clip_hits_on_regex": len(exact_hits),
            "clip_exact_recall_vs_regex": pct(len(exact_hits), len(silver)),
            "clip_exact_precision_vs_regex": pct(len(exact_hits), len(clip_pairs)),
            "same_page_exact_hits_on_regex": len(same_page_hits),
            "same_page_exact_recall_vs_regex": pct(len(same_page_hits), len(silver)),
            "same_page_exact_precision_vs_regex": pct(len(same_page_hits), len(same_page_pairs)),
            "visual_targets_with_regex": len(visual_with_silver),
            "visual_targets_regex_hit_by_clip": len(target_overlap),
            "clip_target_level_recall_vs_regex": pct(len(target_overlap), len(visual_with_silver)),
            "clip_hit_rank_distribution": dict(sorted(rank_hits.items())),
            "visual_nodes_total": len(visual_nodes),
            "visual_nodes_with_clip_text_edge": len(described_visuals),
            "visual_nodes_with_clip_text_edge_rate": pct(len(described_visuals), len(visual_nodes)),
            "text_edge_score_quantiles": quantiles([float(e.get("weight") or 0.0) for e in text_edges]),
        },
        "same_doc_visual_similarity": {
            "edge_count": len(same_visual_edges),
            "score_quantiles": quantiles([float(e.get("weight") or 0.0) for e in same_visual_edges]),
            "top_samples": same_visual_samples,
        },
        "formula_similarity": {
            "edge_count": len(formula_edges),
            "score_quantiles": quantiles([float(e.get("weight") or 0.0) for e in formula_edges]),
        },
        "latex_structural_baseline": {
            "old53_docs": len(old_docs),
            "latex_reference_docs": len(latex_ref_docs),
            "old53_latex_doc_overlap": len(overlap_docs),
            "overlap_docs": overlap_docs[:20],
            "latex_mapping": latex_report.get("label_mapping", {}),
            "latex_density_global": (latex_report.get("density") or {}).get("global", {}),
            "latex_hub_multihop_summary": {
                "candidate_count": ((latex_report.get("hub_multihop_summary") or {}).get("candidate_count")),
                "cross_doc_candidates": ((latex_report.get("hub_multihop_summary") or {}).get("cross_doc_candidates")),
            },
            "latex_position_coverage": latex_report.get("position_coverage", {}),
        },
    }
    return result


def write_report(out_dir: Path, result: dict[str, Any]) -> None:
    cross = result["cross_doc_visual_quality"]
    intra = result["intra_doc_text_visual_vs_regex"]
    latex = result["latex_structural_baseline"]
    same_vis = result["same_doc_visual_similarity"]
    formula = result["formula_similarity"]

    lines = [
        "# MinerU CLIP/VL Edge Quality Audit",
        "",
        "## Raw Data Table",
        "",
        "| Slice | Metric | Value |",
        "|---|---:|---:|",
        f"| Cross-doc CLIP | edges | {cross['edge_count']} |",
        f"| Cross-doc CLIP | unique source visuals | {cross['unique_source_visuals']} |",
        f"| Cross-doc CLIP | avg edges/source | {cross['avg_edges_per_source']} |",
        f"| Cross-doc CLIP | score p50 / p90 | {cross['score_quantiles'].get('p50')} / {cross['score_quantiles'].get('p90')} |",
        f"| Cross-doc CLIP | generic-caption-both rate | {cross['generic_caption_both_rate']:.1%} |",
        f"| Cross-doc CLIP | top100 generic-caption-both rate | {cross['generic_rate_top100']['generic_both_rate']:.1%} |",
        f"| Cross-doc CLIP | undirected unique pairs | {cross['undirected_unique_pairs']} |",
        f"| Text→visual CLIP | edges | {intra['clip_text_visual_edges']} |",
        f"| Text→visual silver | regex text→visual edges | {intra['silver_regex_text_visual_edges']} |",
        f"| Text→visual CLIP | exact recall vs regex | {intra['clip_exact_recall_vs_regex']:.1%} |",
        f"| Text→visual CLIP | target-level recall vs regex | {intra['clip_target_level_recall_vs_regex']:.1%} |",
        f"| Same-page baseline | all cross-type edges | {intra['same_page_cross_type_total_edges']} |",
        f"| Same-page baseline | paragraph↔visual unique pairs | {intra['same_page_para_visual_edges']} |",
        f"| Same-page baseline | precision vs regex | {intra['same_page_exact_precision_vs_regex']:.1%} |",
        f"| Same-doc visual CLIP | edges | {same_vis['edge_count']} |",
        f"| Formula CLIP text | edges | {formula['edge_count']} |",
        f"| LaTeX-v2 graph | docs | {latex['latex_reference_docs']} |",
        f"| LaTeX-v2 graph | old_53 overlap | {latex['old53_latex_doc_overlap']} |",
        f"| LaTeX-v2 graph | mapped element rate | {latex.get('latex_mapping', {}).get('mapping_rate')} |",
        "",
        "## Key Findings",
        "",
        "1. **Cross-doc CLIP 边是有用的新能力，但不是可直接当 gold 的边。** 它产生了 "
        f"{cross['edge_count']} 条跨文档视觉边，分数中位数 {cross['score_quantiles'].get('p50')}，"
        f"p90 {cross['score_quantiles'].get('p90')}。高分样本多是视觉版式/曲线/网格相似，适合做 candidate recall layer；"
        f"但 top100 里 generic-caption-both rate 达 {cross['generic_rate_top100']['generic_both_rate']:.1%}，"
        "需要后续用 caption/context/LLM rerank 做精排。",
        "2. **CLIP 适合替换全量 `same_page_cross_type` 的噪声层，但不是 regex 引用的替代品。** "
        f"全量 same-page 跨类型边是 {intra['same_page_cross_type_total_edges']} 条；CLIP text→visual 是 "
        f"{intra['clip_text_visual_edges']} 条，量级上完成了计划里的稀疏化。若只看 paragraph↔visual 子集，"
        f"same-page 有 {intra['same_page_para_visual_edges']} 条，和 CLIP 接近。",
        "3. **用显式 regex 引用作 silver label 时，same-page 的 exact 命中更高，CLIP 的价值在语义覆盖。** "
        f"CLIP exact recall {intra['clip_exact_recall_vs_regex']:.1%}，target-level recall "
        f"{intra['clip_target_level_recall_vs_regex']:.1%}；same-page exact recall "
        f"{intra['same_page_exact_recall_vs_regex']:.1%}。这说明 CLIP 经常找到同一图附近/语义相近段落，"
        "但未必是含有显式 “Figure N” 的那一句；后续应把 regex 作为高精锚点、CLIP 作为补召回/重排信号。",
        "4. **和 LaTeX 图无法做严格同文档对照。** 当前 old_53 与 `latex_reference_graph_v2` 的文档交集是 "
        f"{latex['old53_latex_doc_overlap']}。所以这份报告只做结构级对照；若要严谨比较，需要在同一批 doc 上同时跑 LaTeX+MinerU 和 CLIP。",
        "5. **LaTeX 图仍然是高精度 intra-doc 结构基线，CLIP 是召回/跨文档补边。** LaTeX-v2 显式引用和 source line 信息强，"
        f"mapping rate {latex.get('latex_mapping', {}).get('mapping_rate')}；CLIP 则补上 LaTeX 图没有的 visual cross-doc edges。",
        "",
        "## Suggested Next Experiments",
        "",
        "1. 在同一文档集合上重跑二者：选 20 篇 LaTeX-v2 中 MinerU image assets 完整的论文，跑 `build_mineru_vl_edges.py`，做 exact overlap 和人工 Top-K 审计。",
        "2. 给 `cross_doc_visual_sim` 加二阶段过滤：保留 CLIP top-k，再用 caption/context TF-IDF 或 sentence-transformer 要求文本相似度超过低阈值，专门压掉 generic plot/layout 假阳性。",
        "3. 对 `text_describes_figure` 做 sentence-level rerank：CLIP 先找段落，regex/句子切分再找真正解释句，避免候选只停在段落级。",
        "",
        "## Cross-doc High-score Samples",
        "",
    ]
    for i, sample in enumerate(cross["top_high_score_samples"][:8], start=1):
        lines.extend([
            f"### Sample {i}: {sample['weight']}  {sample['source_doc']} → {sample['target_doc']}",
            f"- source: `{sample['source_id']}` {sample['source_type']} | {sample['source_caption']}",
            f"- target: `{sample['target_id']}` {sample['target_type']} | {sample['target_caption']}",
            f"- caption_jaccard: `{sample['caption_jaccard']}` generic: `{sample['source_generic_caption']}/{sample['target_generic_caption']}`",
            "",
        ])

    lines.extend([
        "## Risk Samples",
        "",
        "High CLIP score but zero caption-token overlap. These are not automatically wrong, but they need rerank/manual inspection.",
        "",
    ])
    for i, sample in enumerate(cross["risk_samples_high_score_zero_caption_overlap"][:8], start=1):
        lines.extend([
            f"### Risk {i}: {sample['weight']}  {sample['source_doc']} → {sample['target_doc']}",
            f"- source: `{sample['source_id']}` | {sample['source_caption']}",
            f"- target: `{sample['target_id']}` | {sample['target_caption']}",
            "",
        ])

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MinerU CLIP/VL edge quality")
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY))
    parser.add_argument("--vl-edges", default=str(DEFAULT_VL_EDGES))
    parser.add_argument("--latex-report", default=str(DEFAULT_LATEX_REPORT))
    parser.add_argument("--latex-reference", default=str(DEFAULT_LATEX_REF))
    parser.add_argument("--old53", default=str(DEFAULT_OLD53))
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / f"data/05_eval/mineru_vl_quality_audit_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    result = audit(
        Path(args.topology),
        Path(args.vl_edges),
        Path(args.latex_report),
        Path(args.latex_reference),
        Path(args.old53),
    )
    (out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir, result)
    latest = ROOT / "data/05_eval/mineru_vl_quality_audit_latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_dir.resolve())
    except OSError:
        pass
    print(f"[ok] wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
