#!/usr/bin/env python3
"""Build cross-document and bridge-text artifacts for the MinerU graph."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V1_DIR = ROOT / "data/05_eval/mineru_only_graph_v1_latest"
DEFAULT_TOPOLOGY_DIR = ROOT / "data/05_eval/mineru_topology_graph_v1_latest"
DEFAULT_VL_DIR = ROOT / "data/05_eval/mineru_vl_edges_v1_latest"

FIG_REF = re.compile(r"\b(?:Figure|Fig\.?)\s*(\d+)\b", re.IGNORECASE)
TABLE_REF = re.compile(r"\bTable\s*(\d+)\b", re.IGNORECASE)
EQ_REF = re.compile(r"(?:\b(?:Equation|Eq(?:n)?\.?)\s*\(?(\d+)\)?|\((\d+)\))", re.IGNORECASE)
VERB_LIKE = {
    "show", "shows", "shown", "present", "presents", "illustrate", "illustrates",
    "compare", "compares", "summarize", "summarizes", "report", "reports",
    "demonstrate", "demonstrates", "indicate", "indicates", "observe", "observes",
    "describe", "describes", "define", "defines", "use", "uses",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compact_text(value: Any, limit: int = 500) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.split())[:limit]


def split_sentences(text: str) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
    return [p.strip() for p in parts if len(p.strip()) >= 20]


def bridge_quality(sentence: str) -> dict[str, Any]:
    tokens = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", sentence)
    if not tokens:
        return {"score": 0.0, "verb_density": 0.0, "specificity": 0.0}
    verbs = sum(1 for tok in tokens if tok.lower() in VERB_LIKE or tok.lower().endswith(("ed", "ing")))
    numbers = sum(1 for tok in tokens if re.search(r"\d", tok))
    acronyms = sum(1 for tok in tokens if len(tok) > 1 and tok.isupper())
    verb_density = verbs / len(tokens)
    specificity = min(1.0, (numbers + acronyms) / 6.0)
    length_score = min(1.0, len(sentence) / 220.0)
    score = 0.45 * min(1.0, verb_density * 10) + 0.35 * specificity + 0.20 * length_score
    return {
        "score": round(score, 6),
        "verb_density": round(verb_density, 6),
        "specificity": round(specificity, 6),
        "length": len(sentence),
    }


def load_inputs(topology_path: Path, v1_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, Any]]:
    topo = read_json(topology_path)
    nodes = {str(node["node_id"]): node for node in topo.get("nodes", [])}
    element_to_node = {str(k): str(v) for k, v in (topo.get("element_to_node") or {}).items()}
    v1 = read_json(v1_path)
    return nodes, element_to_node, v1


def index_v1_elements(v1: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[int, str]]]]:
    elements_by_id: dict[str, dict[str, Any]] = {}
    by_number: dict[str, dict[str, dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
    for doc_id, doc in (v1.get("documents") or {}).items():
        for element_id, elem in (doc.get("elements") or {}).items():
            elem = dict(elem)
            elem["element_id"] = element_id
            elem["doc_id"] = doc_id
            elements_by_id[element_id] = elem
            etype = str(elem.get("element_type") or "")
            number = elem.get("number")
            if isinstance(number, int) and etype in {"figure", "table", "formula"}:
                by_number[doc_id][etype][number] = element_id
    return elements_by_id, by_number


def find_ref_sentence(text: str, pattern: re.Pattern[str], number: int) -> str:
    for sentence in split_sentences(text):
        for match in pattern.finditer(sentence):
            groups = [g for g in match.groups() if g]
            if groups and int(groups[0]) == number:
                return sentence
    return ""


def build_sentence_bridges(
    v1: dict[str, Any],
    element_to_node: dict[str, str],
    by_number: dict[str, dict[str, dict[int, str]]],
) -> list[dict[str, Any]]:
    bridges: list[dict[str, Any]] = []
    specs = [
        ("figure", FIG_REF),
        ("table", TABLE_REF),
        ("formula", EQ_REF),
    ]
    seen: set[tuple[str, str, str]] = set()
    for doc_id, doc in (v1.get("documents") or {}).items():
        for para_element_id, elem in (doc.get("elements") or {}).items():
            if elem.get("element_type") != "text":
                continue
            text = str(elem.get("content") or "")
            if not text:
                continue
            para_node = element_to_node.get(para_element_id, para_element_id)
            for etype, pattern in specs:
                for match in pattern.finditer(text):
                    groups = [g for g in match.groups() if g]
                    if not groups:
                        continue
                    try:
                        number = int(groups[0])
                    except ValueError:
                        continue
                    target_element_id = by_number.get(doc_id, {}).get(etype, {}).get(number)
                    if not target_element_id:
                        continue
                    target_node = element_to_node.get(target_element_id, target_element_id)
                    sentence = find_ref_sentence(text, pattern, number) or compact_text(text, 500)
                    key = (para_node, target_node, sentence)
                    if key in seen:
                        continue
                    seen.add(key)
                    quality = bridge_quality(sentence)
                    bridges.append({
                        "paragraph_node_id": para_node,
                        "target_node_id": target_node,
                        "doc_id": doc_id,
                        "target_type": etype,
                        "target_number": number,
                        "sentence": sentence,
                        "quality": quality,
                        "paragraph_element_id": para_element_id,
                        "target_element_id": target_element_id,
                    })
    bridges.sort(key=lambda row: row["quality"]["score"], reverse=True)
    return bridges


def load_rerank_tiers(rerank_path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    """Map (source_id, target_id) -> {tier, combined_score, ...} from the rerank pass."""
    if not rerank_path or not rerank_path.exists():
        return {}
    tiers: dict[tuple[str, str], dict[str, Any]] = {}
    for row in iter_jsonl(rerank_path):
        key = (row.get("source_id"), row.get("target_id"))
        tiers[key] = {
            "support_tier": row.get("support_tier"),
            "combined_score": row.get("combined_score"),
            "text_support": row.get("text_support"),
            "generic_caption_both": row.get("generic_caption_both"),
        }
    return tiers


def load_visual_crossdoc_edges(
    vl_edges_path: Path,
    rerank_tiers: dict[tuple[str, str], dict[str, Any]] | None = None,
    drop_tiers: frozenset[str] = frozenset({"visual_only_risky"}),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    crossdoc: list[dict[str, Any]] = []
    alignments: list[dict[str, Any]] = []
    all_text_describes: list[dict[str, Any]] = []
    rerank_tiers = rerank_tiers or {}
    dropped: dict[str, int] = {}
    for edge in iter_jsonl(vl_edges_path):
        etype = edge.get("edge_type")
        if etype == "cross_doc_visual_sim":
            tier_info = rerank_tiers.get((edge.get("source_id"), edge.get("target_id")))
            if tier_info:
                tier_name = tier_info.get("support_tier")
                if tier_name in drop_tiers:
                    dropped[tier_name] = dropped.get(tier_name, 0) + 1
                    continue
                # promote tier + reranked confidence onto the edge so the
                # downstream graph carries a confidence signal, not raw CLIP.
                meta = edge.setdefault("metadata", {})
                meta["support_tier"] = tier_name
                meta["rerank_combined_score"] = tier_info.get("combined_score")
                meta["text_support"] = tier_info.get("text_support")
                meta["generic_caption_both"] = tier_info.get("generic_caption_both")
            crossdoc.append(edge)
        elif etype == "text_describes_figure":
            all_text_describes.append(edge)
            alignments.append({
                "paragraph_node_id": edge.get("source_id"),
                "figure_node_id": edge.get("target_id"),
                "doc_id": edge.get("doc_id"),
                "score": edge.get("weight"),
                "similarity": (edge.get("metadata") or {}).get("similarity"),
                "paragraph_preview": (edge.get("metadata") or {}).get("paragraph_preview", ""),
                "caption": (edge.get("metadata") or {}).get("caption", ""),
            })
    return crossdoc, alignments, all_text_describes, dropped


def add_edge(
    edges: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float,
    doc_id: str,
    metadata: dict[str, Any],
) -> None:
    key = (source_id, target_id, edge_type)
    if source_id == target_id or key in seen:
        return
    seen.add(key)
    edges.append({
        "source_id": source_id,
        "target_id": target_id,
        "doc_id": doc_id,
        "edge_type": edge_type,
        "weight": round(float(weight), 6),
        "metadata": metadata,
    })


def build_section_crossdoc_edges(
    nodes: dict[str, dict[str, Any]],
    threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    sections = [
        n for n in nodes.values()
        if n.get("node_type") == "section" and compact_text(n.get("section_title") or n.get("label"), 200)
    ]
    if len(sections) < 2:
        return []
    texts = [compact_text(n.get("section_title") or n.get("label"), 300) for n in sections]
    matrix = TfidfVectorizer(max_features=2048, ngram_range=(1, 2), stop_words="english").fit_transform(texts)
    sim = cosine_similarity(matrix)
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for i, src in enumerate(sections):
        scores = []
        for j, tgt in enumerate(sections):
            if i == j or src.get("doc_id") == tgt.get("doc_id"):
                continue
            score = float(sim[i, j])
            if score >= threshold:
                scores.append((score, j))
        scores.sort(reverse=True)
        for score, j in scores[:top_k]:
            tgt = sections[j]
            add_edge(
                edges,
                seen,
                src["node_id"],
                tgt["node_id"],
                "cross_doc_section_sim",
                score,
                f"{src.get('doc_id')}->{tgt.get('doc_id')}",
                {
                    "similarity": round(score, 6),
                    "source_title": compact_text(src.get("section_title") or src.get("label"), 200),
                    "target_title": compact_text(tgt.get("section_title") or tgt.get("label"), 200),
                },
            )
    return edges


def build_paragraph_crossdoc_edges(
    nodes: dict[str, dict[str, Any]],
    threshold: float,
    top_k_per_doc_pair: int,
    max_paragraphs_per_doc: int,
) -> list[dict[str, Any]]:
    paragraphs_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes.values():
        if node.get("node_type") != "paragraph":
            continue
        text = compact_text(node.get("text_snippet"), 500)
        if len(text) >= 80:
            paragraphs_by_doc[str(node.get("doc_id"))].append(node)
    selected: list[dict[str, Any]] = []
    for doc_id, paras in paragraphs_by_doc.items():
        paras.sort(key=lambda n: len(str(n.get("text_snippet") or "")), reverse=True)
        selected.extend(paras[:max_paragraphs_per_doc])
    if len(selected) < 2:
        return []
    texts = [compact_text(n.get("text_snippet"), 500) for n in selected]
    matrix = TfidfVectorizer(max_features=4096, ngram_range=(1, 2), stop_words="english").fit_transform(texts)
    by_doc_idx: dict[str, list[int]] = defaultdict(list)
    for idx, node in enumerate(selected):
        by_doc_idx[str(node.get("doc_id"))].append(idx)

    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    doc_ids = sorted(by_doc_idx)
    for a_pos, doc_a in enumerate(doc_ids):
        for doc_b in doc_ids[a_pos + 1:]:
            idx_a = by_doc_idx[doc_a]
            idx_b = by_doc_idx[doc_b]
            sim = cosine_similarity(matrix[idx_a], matrix[idx_b])
            flat = []
            for ia, src_idx in enumerate(idx_a):
                for ib, tgt_idx in enumerate(idx_b):
                    score = float(sim[ia, ib])
                    if score >= threshold:
                        flat.append((score, src_idx, tgt_idx))
            flat.sort(reverse=True)
            for score, src_idx, tgt_idx in flat[:top_k_per_doc_pair]:
                src = selected[src_idx]
                tgt = selected[tgt_idx]
                add_edge(
                    edges,
                    seen,
                    src["node_id"],
                    tgt["node_id"],
                    "cross_doc_semantic",
                    score,
                    f"{doc_a}->{doc_b}",
                    {
                        "similarity": round(score, 6),
                        "source_preview": compact_text(src.get("text_snippet"), 180),
                        "target_preview": compact_text(tgt.get("text_snippet"), 180),
                    },
                )
                add_edge(
                    edges,
                    seen,
                    tgt["node_id"],
                    src["node_id"],
                    "cross_doc_semantic",
                    score,
                    f"{doc_b}->{doc_a}",
                    {
                        "similarity": round(score, 6),
                        "source_preview": compact_text(tgt.get("text_snippet"), 180),
                        "target_preview": compact_text(src.get("text_snippet"), 180),
                    },
                )
    return edges


def find_orphan_figures(
    nodes: dict[str, dict[str, Any]],
    text_describes_edges: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    described: set[str] = set()
    for edge in text_describes_edges:
        if float(edge.get("weight") or 0.0) >= threshold:
            described.add(str(edge.get("target_id")))
    orphans: list[dict[str, Any]] = []
    for node in nodes.values():
        if node.get("node_type") not in {"figure", "table"}:
            continue
        if node["node_id"] not in described:
            orphans.append({
                "node_id": node["node_id"],
                "doc_id": node.get("doc_id"),
                "node_type": node.get("node_type"),
                "label": node.get("label"),
                "page_idx": node.get("page_idx"),
                "caption": (node.get("metadata") or {}).get("caption", ""),
            })
    return orphans


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MinerU Cross-Doc Bridges v1",
        "",
        f"- cross-doc edges: **{summary['total_crossdoc_edges']}** `{summary['crossdoc_edge_type_counts']}`",
        f"- sentence bridges: **{summary['sentence_bridge_count']}**",
        f"- VL alignments: **{summary['vl_alignment_count']}**",
        f"- orphan visual nodes: **{summary['orphan_visual_count']}**",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_latest_symlink(out_dir: Path) -> None:
    latest = ROOT / "data/05_eval/mineru_crossdoc_bridges_v1_latest"
    try:
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        if not latest.exists():
            latest.symlink_to(out_dir.resolve())
    except OSError as exc:
        print(f"[warn] could not update latest symlink: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MinerU cross-doc bridge artifacts v1")
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY_DIR / "mineru_topology_graph_v1.json"))
    parser.add_argument("--elements", default=str(DEFAULT_V1_DIR / "mineru_elements_v1.json"))
    parser.add_argument("--vl-edges", default=str(DEFAULT_VL_DIR / "mineru_vl_edges_v1.jsonl"))
    parser.add_argument(
        "--rerank-edges",
        default=str(ROOT / "data/05_eval/mineru_crossdoc_text_rerank_v1_latest/mineru_crossdoc_text_rerank_edges_v1.jsonl"),
        help="Reranked cross-doc edges with support_tier; used to filter risky edges. Set '' to skip.",
    )
    parser.add_argument(
        "--keep-risky",
        action="store_true",
        help="Keep visual_only_risky edges (default: drop them; opt-in for offline review).",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--section-threshold", type=float, default=0.30)
    parser.add_argument("--section-top-k", type=int, default=3)
    parser.add_argument("--paragraph-threshold", type=float, default=0.60)
    parser.add_argument("--paragraph-top-k-per-doc-pair", type=int, default=1)
    parser.add_argument("--max-paragraphs-per-doc", type=int, default=40)
    parser.add_argument("--orphan-threshold", type=float, default=0.20)
    args = parser.parse_args()

    topology_path = Path(args.topology)
    elements_path = Path(args.elements)
    vl_edges_path = Path(args.vl_edges)
    if not topology_path.exists():
        raise FileNotFoundError(topology_path)
    if not elements_path.exists():
        raise FileNotFoundError(elements_path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / f"data/05_eval/mineru_crossdoc_bridges_v1_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes, element_to_node, v1 = load_inputs(topology_path, elements_path)
    _, by_number = index_v1_elements(v1)
    sentence_bridges = build_sentence_bridges(v1, element_to_node, by_number)
    rerank_path = Path(args.rerank_edges) if args.rerank_edges else None
    rerank_tiers = load_rerank_tiers(rerank_path)
    drop_tiers = frozenset() if args.keep_risky else frozenset({"visual_only_risky"})
    visual_crossdoc, vl_alignments, text_describes_edges, dropped_tiers = load_visual_crossdoc_edges(
        vl_edges_path, rerank_tiers, drop_tiers
    )
    section_edges = build_section_crossdoc_edges(nodes, args.section_threshold, args.section_top_k)
    paragraph_edges = build_paragraph_crossdoc_edges(
        nodes,
        args.paragraph_threshold,
        args.paragraph_top_k_per_doc_pair,
        args.max_paragraphs_per_doc,
    )
    orphan_figures = find_orphan_figures(nodes, text_describes_edges, args.orphan_threshold)

    crossdoc_edges = visual_crossdoc + section_edges + paragraph_edges
    bridge_payload = {
        "metadata": {
            "builder": "mineru_crossdoc_bridges_v1",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_topology": str(topology_path),
            "source_elements": str(elements_path),
            "source_vl_edges": str(vl_edges_path),
        },
        "sentence_bridges": sentence_bridges,
        "vl_alignments": vl_alignments,
        "orphan_visual_nodes": orphan_figures,
    }
    summary = {
        "builder": "mineru_crossdoc_bridges_v1",
        "created_at": bridge_payload["metadata"]["created_at"],
        "total_crossdoc_edges": len(crossdoc_edges),
        "crossdoc_edge_type_counts": dict(Counter(e.get("edge_type") for e in crossdoc_edges)),
        "sentence_bridge_count": len(sentence_bridges),
        "sentence_target_type_counts": dict(Counter(b.get("target_type") for b in sentence_bridges)),
        "vl_alignment_count": len(vl_alignments),
        "orphan_visual_count": len(orphan_figures),
        "rerank_applied": bool(rerank_tiers),
        "rerank_source": str(rerank_path) if rerank_path else "",
        "dropped_by_tier": dropped_tiers,
        "crossdoc_tier_counts": dict(Counter(
            (e.get("metadata") or {}).get("support_tier") for e in visual_crossdoc
            if e.get("edge_type") == "cross_doc_visual_sim"
        )),
        "thresholds": {
            "section_threshold": args.section_threshold,
            "paragraph_threshold": args.paragraph_threshold,
            "orphan_threshold": args.orphan_threshold,
        },
    }

    write_jsonl(out_dir / "mineru_crossdoc_edges_v1.jsonl", crossdoc_edges)
    (out_dir / "mineru_bridge_texts_v1.json").write_text(json.dumps(bridge_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir, summary)
    update_latest_symlink(out_dir)

    print(f"[ok] wrote {out_dir / 'mineru_crossdoc_edges_v1.jsonl'}")
    print(f"crossdoc_edges={len(crossdoc_edges)} sentence_bridges={len(sentence_bridges)} orphan_visual={len(orphan_figures)}")


if __name__ == "__main__":
    main()
