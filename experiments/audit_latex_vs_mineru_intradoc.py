#!/usr/bin/env python3
"""Intra-doc A/B: MinerU reference edges vs LaTeX \\ref hard edges.

Calibration baseline (NOT a go/no-go).  On documents that we have parsed BOTH
ways, this quantifies:

  1. Figure/table extraction recall  - of the figures/tables LaTeX defines,
     how many does MinerU recover (matched by caption text)?
  2. Reference recall                - of the figures/tables LaTeX *references*
     in body text (\\ref hard edges), how many does MinerU's regex_reference
     layer also link to text?
  3. MinerU-only referenced elements - elements MinerU links to text that LaTeX
     never \\ref's (implicit references, or MinerU noise).

LaTeX side  : src/parsers/latex_reference_extractor.py over the raw .tex.
MinerU side : data/05_eval/mineru_only_graph_v1_latest/mineru_edges_v1.jsonl
              (regex_reference edges) + the figure/table nodes in the topology.

Matching LaTeX labels (fig:foo, defined with a \\caption) to MinerU numbered
elements (doc_figure_N, captioned "Figure N: ...") is done per-document by
caption-token similarity, since LaTeX labels carry no compiled figure number.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.parsers.latex_reference_extractor import LaTeXReferenceExtractor  # noqa: E402

LATEX_EXTRACT_DIR = ROOT / "data/01_graphs/latex_sections_rebuild_2026-03-24/extracted"
TOPOLOGY = ROOT / "data/05_eval/mineru_topology_graph_v1_latest/mineru_topology_graph_v1.json"
MINERU_EDGES = ROOT / "data/05_eval/mineru_only_graph_v1_latest/mineru_edges_v1.jsonl"
OUT_DIR = ROOT / f"data/05_eval/latex_vs_mineru_intradoc_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

# elements LaTeX \ref can point at that MinerU also represents as visual/formula nodes
VISUAL_TYPES = {"figure", "table"}

_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?")
_FIG_PREFIX = re.compile(r"^\s*(?:figure|fig\.?|table|tab\.?)\s*\d+[a-z]?\s*[:.\-]?\s*", re.I)
_NONWORD = re.compile(r"[^a-z0-9 ]+")
_STOP = {
    "the", "a", "an", "of", "for", "and", "to", "in", "on", "with", "is", "are",
    "we", "our", "this", "that", "as", "by", "from", "at", "shows", "show",
    "figure", "fig", "table", "tab", "left", "right", "top", "bottom",
}


def caption_tokens(text: str) -> set[str]:
    text = text or ""
    text = _LATEX_CMD.sub(" ", text)
    text = _FIG_PREFIX.sub("", text)
    text = text.lower()
    text = _NONWORD.sub(" ", text)
    toks = {t for t in text.split() if len(t) >= 3 and t not in _STOP and not t.isdigit()}
    return toks


def token_sim(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    # asymmetric containment-friendly: inter / smaller side, capped by jaccard floor
    return inter / min(len(a), len(b))


def load_mineru(topology_path: Path, edges_path: Path):
    topo = json.loads(topology_path.read_text())
    figs_by_doc: dict[str, list[dict]] = defaultdict(list)
    for n in topo["nodes"]:
        if n.get("node_type") in VISUAL_TYPES:
            figs_by_doc[n["doc_id"]].append({
                "element_id": n.get("element_id") or n.get("mapped_element_id"),
                "node_type": n["node_type"],
                "caption": n.get("text_snippet") or n.get("label") or "",
                "tokens": caption_tokens(n.get("text_snippet") or n.get("label") or ""),
            })
    # regex_reference: text -> figure/table/formula  (MinerU's intra-doc hard edge)
    referenced: dict[str, set[str]] = defaultdict(set)  # doc -> {element_id targeted by >=1 text ref}
    ref_edge_count: dict[str, int] = defaultdict(int)
    for line in edges_path.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("edge_type") != "regex_reference":
            continue
        tgt = e.get("target_id", "")
        doc = e.get("doc_id")
        # only count refs to figures/tables
        if any(f"_{vt}_" in tgt for vt in VISUAL_TYPES):
            referenced[doc].add(tgt)
            ref_edge_count[doc] += 1
    return figs_by_doc, referenced, ref_edge_count


def latex_graph(doc_id: str):
    extract_dir = LATEX_EXTRACT_DIR / doc_id
    if not extract_dir.is_dir():
        return None
    extractor = LaTeXReferenceExtractor()
    try:
        g = extractor.extract(doc_id, extract_dir)
    except Exception as exc:  # parsing is best-effort
        return {"error": str(exc)}
    return g.to_dict()


def match_latex_to_mineru(latex_labels: dict, mineru_figs: list[dict], min_sim: float):
    """Greedy best caption match LaTeX figure/table label -> MinerU element."""
    lat = [
        (k, v) for k, v in latex_labels.items()
        if v.get("label_type") in VISUAL_TYPES and (v.get("caption") or "").strip()
    ]
    pairs = []
    for k, v in lat:
        lt = caption_tokens(v.get("caption", ""))
        best, best_s = None, 0.0
        for mf in mineru_figs:
            if mf["node_type"] != v["label_type"]:
                continue
            s = token_sim(lt, mf["tokens"])
            if s > best_s:
                best, best_s = mf, s
        pairs.append({
            "latex_label": k,
            "label_type": v["label_type"],
            "latex_caption": (v.get("caption") or "")[:160],
            "matched_element": best["element_id"] if best and best_s >= min_sim else None,
            "match_sim": round(best_s, 3),
        })
    return pairs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-match-sim", type=float, default=0.34,
                    help="caption-token similarity floor to call a LaTeX label and a MinerU element the same figure")
    args = ap.parse_args()

    figs_by_doc, mineru_referenced, mineru_ref_count = load_mineru(TOPOLOGY, MINERU_EDGES)
    mineru_docs = set(figs_by_doc)
    latex_docs = {p.name for p in LATEX_EXTRACT_DIR.iterdir() if p.is_dir()}
    overlap = sorted(mineru_docs & latex_docs)

    per_doc = []
    agg = Counter()
    for doc in overlap:
        lg = latex_graph(doc)
        if not lg or lg.get("error") or not lg.get("labels"):
            per_doc.append({"doc_id": doc, "skipped": lg.get("error", "no_latex_labels") if lg else "no_latex"})
            continue
        labels = lg["labels"]
        # LaTeX-referenced labels (incoming \ref hard edges) of type figure/table
        latex_referenced_labels = set()
        for e in lg["edges"]:
            if e["target_type"] in VISUAL_TYPES:
                latex_referenced_labels.add(e["target_label"])

        pairs = match_latex_to_mineru(labels, figs_by_doc[doc], args.min_match_sim)
        matched = [p for p in pairs if p["matched_element"]]
        n_latex_visual = sum(1 for v in labels.values() if v.get("label_type") in VISUAL_TYPES and (v.get("caption") or "").strip())

        # reference recall: of LaTeX-referenced figures (matched to MinerU),
        # how many does MinerU's regex layer also reference?
        mref = mineru_referenced.get(doc, set())
        ref_matched = [p for p in matched if p["latex_label"] in latex_referenced_labels]
        ref_recovered = [p for p in ref_matched if p["matched_element"] in mref]

        # MinerU-only: MinerU-referenced elements not matched to any LaTeX-referenced label
        matched_elems_for_latex_refs = {p["matched_element"] for p in ref_matched}
        mineru_only = [m for m in mref if m not in matched_elems_for_latex_refs]

        rec = {
            "doc_id": doc,
            "latex_visual_labels": n_latex_visual,
            "mineru_visual_nodes": len(figs_by_doc[doc]),
            "matched_figures": len(matched),
            "extraction_recall": round(len(matched) / n_latex_visual, 3) if n_latex_visual else None,
            "latex_referenced_figures": len(latex_referenced_labels),
            "latex_referenced_matched": len(ref_matched),
            "mineru_recovered_refs": len(ref_recovered),
            "reference_recall": round(len(ref_recovered) / len(ref_matched), 3) if ref_matched else None,
            "mineru_ref_edge_count": mineru_ref_count.get(doc, 0),
            "mineru_only_referenced": len(mineru_only),
        }
        per_doc.append(rec)
        agg["latex_visual_labels"] += n_latex_visual
        agg["matched_figures"] += len(matched)
        agg["latex_referenced_matched"] += len(ref_matched)
        agg["mineru_recovered_refs"] += len(ref_recovered)
        agg["mineru_only_referenced"] += len(mineru_only)

    summary = {
        "builder": "latex_vs_mineru_intradoc_ab",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_match_sim": args.min_match_sim,
        "overlap_docs": len(overlap),
        "docs_with_latex_labels": sum(1 for r in per_doc if "skipped" not in r),
        "corpus_extraction_recall": round(agg["matched_figures"] / agg["latex_visual_labels"], 4) if agg["latex_visual_labels"] else None,
        "corpus_reference_recall": round(agg["mineru_recovered_refs"] / agg["latex_referenced_matched"], 4) if agg["latex_referenced_matched"] else None,
        "totals": dict(agg),
        "per_doc": per_doc,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_report(OUT_DIR, summary)
    latest = ROOT / "data/05_eval/latex_vs_mineru_intradoc_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(OUT_DIR.resolve())

    print(f"[ok] {OUT_DIR/'report.md'}")
    print(f"overlap_docs={summary['overlap_docs']} with_latex={summary['docs_with_latex_labels']}")
    print(f"extraction_recall={summary['corpus_extraction_recall']} reference_recall={summary['corpus_reference_recall']}")


def write_report(out_dir: Path, s: dict):
    lines = [
        "# Intra-doc A/B: MinerU vs LaTeX \\ref hard edges",
        "",
        "Calibration baseline on documents parsed BOTH ways (LaTeX `.tex` + MinerU).",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| overlap docs | {s['overlap_docs']} |",
        f"| docs with LaTeX labels | {s['docs_with_latex_labels']} |",
        f"| LaTeX visual labels (fig+table) | {s['totals'].get('latex_visual_labels', 0)} |",
        f"| matched to MinerU element | {s['totals'].get('matched_figures', 0)} |",
        f"| **corpus extraction recall** | **{s['corpus_extraction_recall']}** |",
        f"| LaTeX-referenced & matched | {s['totals'].get('latex_referenced_matched', 0)} |",
        f"| MinerU recovered those refs | {s['totals'].get('mineru_recovered_refs', 0)} |",
        f"| **corpus reference recall** | **{s['corpus_reference_recall']}** |",
        f"| MinerU-only referenced elements | {s['totals'].get('mineru_only_referenced', 0)} |",
        "",
        "## Interpretation",
        "",
        "- **Extraction recall**: fraction of LaTeX-defined figures/tables that MinerU also extracted (caption-matched). Below 1.0 means MinerU missed or merged some figures, or captions diverged too much to match.",
        "- **Reference recall**: of the figures LaTeX explicitly `\\ref`s in body text, how many does MinerU's `regex_reference` layer also link to text. This is the direct 'can MinerU recover the LaTeX hard edge' number.",
        "- **MinerU-only referenced**: elements MinerU links to text but LaTeX never `\\ref`s. Mixed bag: implicit references MinerU's regex catches ('the figure above'), plus genuine MinerU noise.",
        "",
        "## Per-doc detail (docs with LaTeX labels)",
        "",
        "| doc | LaTeX vis | matched | ext.rec | LaTeX-ref'd | MinerU recov | ref.rec | MinerU-only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in s["per_doc"]:
        if "skipped" in r:
            continue
        lines.append(
            f"| {r['doc_id']} | {r['latex_visual_labels']} | {r['matched_figures']} | "
            f"{r['extraction_recall']} | {r['latex_referenced_matched']} | "
            f"{r['mineru_recovered_refs']} | {r['reference_recall']} | {r['mineru_only_referenced']} |"
        )
    skipped = [r for r in s["per_doc"] if "skipped" in r]
    if skipped:
        lines += ["", f"## Skipped ({len(skipped)})", ""]
        lines += [f"- `{r['doc_id']}`: {r['skipped']}" for r in skipped]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
