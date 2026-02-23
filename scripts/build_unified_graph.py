#!/usr/bin/env python3
"""Build unified heterogeneous graph and cross-doc paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.linkers.unified_graph import UnifiedGraph


def main() -> None:
    ap = argparse.ArgumentParser(description="Build unified graph from LaTeX + citation outputs")
    ap.add_argument("--latex-graph", default="data/latex_reference_graph.json")
    ap.add_argument("--citation-graph", default="data/citation_graph.json")
    ap.add_argument("--output", default="data/unified_graph.json")
    ap.add_argument("--paths-output", default="data/unified_graph_paths.json")
    ap.add_argument("--multimodal-elements", default="", help="Optional MinerU multimodal_elements.json for node alignment")
    ap.add_argument("--min-align-jaccard", type=float, default=0.35, help="Min caption Jaccard for LaTeX<->MinerU alignment")
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--min-score", type=float, default=0.3)
    args = ap.parse_args()

    latex_path = Path(args.latex_graph)
    citation_path = Path(args.citation_graph)
    if not latex_path.exists():
        raise FileNotFoundError(f"LaTeX graph not found: {latex_path}")
    if not citation_path.exists():
        raise FileNotFoundError(f"Citation graph not found: {citation_path}")

    g = UnifiedGraph()
    g.load_intra_doc_edges(str(latex_path))
    g.load_citation_edges(str(citation_path))
    if args.multimodal_elements:
        mm_path = Path(args.multimodal_elements)
        if not mm_path.exists():
            raise FileNotFoundError(f"Multimodal elements not found: {mm_path}")
        g.load_mineru_elements_and_align(
            str(mm_path),
            min_caption_jaccard=args.min_align_jaccard,
        )

    paths = g.find_cross_doc_element_paths(max_hops=args.max_hops, min_score=args.min_score)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(g.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    p_out = Path(args.paths_output)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(json.dumps({"paths": paths}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Unified graph saved: {out}")
    print(f"  nodes={len(g.nodes)} edges={len(g.edges)}")
    print(f"Paths saved: {p_out}")
    print(f"  paths={len(paths)} (max_hops={args.max_hops}, min_score={args.min_score})")


if __name__ == "__main__":
    main()
