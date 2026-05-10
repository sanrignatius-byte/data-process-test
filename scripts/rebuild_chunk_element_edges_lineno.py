#!/usr/bin/env python3
"""rebuild_chunk_element_edges_lineno.py
==========================================
B1 Phase 2: Replace position_idx-based chunk_contains_element edges with
LaTeX line_no-based deterministic edges.

Pipeline:
  1. Match each mineru paragraph to a `.tex` source line_no via text overlap
  2. For each chunk in `paragraph_chunks_n400_v2.json`:
     - line_range = [min(para line_no), max(para line_no)]
     - Re-inject element_ids: each element with `latex_line_no` ∈ line_range
       is assigned to that chunk
  3. Output `paragraph_chunks_n400_v2_lineno.json` (same schema, replaced
     element_ids + new chunk_line_start / chunk_line_end fields)
  4. Audit report comparing new vs old element_ids per chunk

Usage:
  python scripts/rebuild_chunk_element_edges_lineno.py \\
    --chunks data/01_graphs/paragraph_chunks_n400_v2.json \\
    --paragraphs data/01_graphs/chunk_virtual_nodes_v2.json \\
    --element-lineno data/01_graphs/element_latex_lineno_map.json \\
    --latex-source-root data/00_raw/latex_sources_delivery/extracted \\
    --output data/01_graphs/paragraph_chunks_n400_v2_lineno.json \\
    --audit data/01_graphs/chunk_element_edges_audit.json
"""
from __future__ import annotations
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def normalize_tex_for_match(s: str) -> str:
    s = re.sub(r"%.*$", "", s, flags=re.MULTILINE)
    s = re.sub(r"\\[a-zA-Z]+\*?\{([^{}]*?)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[\\{}\[\]\$]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_main_tex(doc_dir: Path) -> Path | None:
    if not doc_dir.exists():
        return None
    candidates = sorted(doc_dir.rglob("*.tex"), key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0] if candidates else None


def find_para_line(para_text: str, norm_lines: list[str]) -> int | None:
    """Return 1-indexed line_no of the first .tex line containing
    a leading word-window from para_text. Tries 8-word, then 4-word window."""
    words = para_text.split()
    if len(words) < 3:
        return None
    for w in (8, 4):
        if len(words) < w:
            continue
        target = " ".join(words[:w]).lower().strip()
        if len(target) < 12:
            continue
        for i, line in enumerate(norm_lines):
            if line and target in line.lower():
                return i + 1
    return None


def load_paragraph_lineno(
    paragraphs_doc: dict,
    tex_path: Path | None,
) -> dict[str, int]:
    """Returns {paragraph_id: line_no} for paragraphs matched to .tex."""
    if not tex_path or not tex_path.exists():
        return {}
    tex = tex_path.read_text(errors="replace")
    norm_lines = [normalize_tex_for_match(l) for l in tex.splitlines()]
    out: dict[str, int] = {}
    last_line = 0
    for pid, p in paragraphs_doc.items():
        text = p.get("text", "") or ""
        ln = find_para_line(text, norm_lines)
        # Monotonic guard: prefer matches >= last matched line (avoid jumping back)
        if ln is not None and ln >= last_line:
            out[pid] = ln
            last_line = ln
        elif ln is not None and ln < last_line - 50:
            # Reject big backward jumps (likely false match in references / appendix)
            pass
        elif ln is not None:
            out[pid] = ln
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/01_graphs/paragraph_chunks_n400_v2.json")
    parser.add_argument("--paragraphs", default="data/01_graphs/chunk_virtual_nodes_v2.json")
    parser.add_argument("--element-lineno", default="data/01_graphs/element_latex_lineno_map.json")
    parser.add_argument("--latex-source-root", default="data/00_raw/latex_sources_delivery/extracted")
    parser.add_argument("--output", default="data/01_graphs/paragraph_chunks_n400_v2_lineno.json")
    parser.add_argument("--audit", default="data/01_graphs/chunk_element_edges_audit.json")
    parser.add_argument("--restrict-docs", help="comma-sep doc_ids to limit work (default = all)")
    args = parser.parse_args()

    chunks_data = json.load(open(args.chunks))
    para_data = json.load(open(args.paragraphs))
    elem_lineno = json.load(open(args.element_lineno))
    src_root = Path(args.latex_source_root)

    docs = chunks_data["documents"]
    para_docs = para_data["documents"]
    target_ids = (set(args.restrict_docs.split(",")) if args.restrict_docs else set(docs.keys()))

    audit = {
        "docs_processed": 0,
        "docs_skipped_no_tex": 0,
        "paragraphs_matched": 0,
        "paragraphs_total": 0,
        "chunks_with_line_range": 0,
        "chunks_total": 0,
        "elements_assigned": 0,
        "elements_unmatched": 0,
        "old_vs_new_overlap": defaultdict(int),  # 'kept', 'added', 'removed'
        "per_doc": {},
    }

    for doc_id in list(docs.keys()):
        if doc_id not in target_ids:
            continue
        doc_chunks = docs[doc_id]
        para_doc = para_docs.get(doc_id, {})
        paragraphs = para_doc.get("paragraph_nodes", {})
        audit["paragraphs_total"] += len(paragraphs)

        tex_path = find_main_tex(src_root / doc_id)
        if not tex_path:
            audit["docs_skipped_no_tex"] += 1
            continue

        para_line = load_paragraph_lineno(paragraphs, tex_path)
        audit["paragraphs_matched"] += len(para_line)
        audit["docs_processed"] += 1

        # Map para_idx -> line_no via paragraph_id lookup
        idx_to_line: dict[int, int] = {}
        for pid, ln in para_line.items():
            p = paragraphs.get(pid, {})
            idx_str = p.get("para_idx")
            try:
                idx_to_line[int(idx_str)] = ln
            except (TypeError, ValueError):
                pass

        # Pre-build elements-by-doc with line_no
        # element_lineno keys are element_ids; filter to this doc
        doc_elem_lines: list[tuple[str, int]] = []
        for eid, info in elem_lineno.items():
            if eid.startswith(doc_id + "_") and info.get("latex_line_no") is not None:
                doc_elem_lines.append((eid, info["latex_line_no"]))

        # Iterate chunks: compute line_range, assign elements
        per_doc_stats = {
            "chunks": 0,
            "chunks_with_range": 0,
            "elem_old_total": 0,
            "elem_new_total": 0,
            "elem_kept": 0,
            "elem_added": 0,
            "elem_removed": 0,
        }

        chunks_dict = doc_chunks.get("nodes", {})
        # chunks include section nodes too — filter by chunk_id pattern
        for nid, node in chunks_dict.items():
            if "chunk_id" not in node:
                continue
            per_doc_stats["chunks"] += 1
            audit["chunks_total"] += 1
            para_idxs = node.get("paragraph_indices", [])
            line_nos = [idx_to_line[i] for i in para_idxs if i in idx_to_line]
            old_eids = set(node.get("element_ids", []) or [])
            per_doc_stats["elem_old_total"] += len(old_eids)
            if not line_nos:
                # No paragraph line_no available — keep old element_ids (cannot infer)
                node["chunk_line_start"] = None
                node["chunk_line_end"] = None
                node["element_ids_lineno"] = []
                continue
            line_start, line_end = min(line_nos), max(line_nos)
            node["chunk_line_start"] = line_start
            node["chunk_line_end"] = line_end
            audit["chunks_with_line_range"] += 1
            per_doc_stats["chunks_with_range"] += 1
            # Inject elements whose latex_line_no falls in [line_start, line_end+padding]
            # padding +30 lines to absorb figure/equation env that span beyond first \begin
            new_eids = set()
            for eid, ln in doc_elem_lines:
                if line_start - 5 <= ln <= line_end + 30:
                    new_eids.add(eid)
            audit["elements_assigned"] += len(new_eids)
            per_doc_stats["elem_new_total"] += len(new_eids)
            kept = old_eids & new_eids
            added = new_eids - old_eids
            removed = old_eids - new_eids
            per_doc_stats["elem_kept"] += len(kept)
            per_doc_stats["elem_added"] += len(added)
            per_doc_stats["elem_removed"] += len(removed)
            audit["old_vs_new_overlap"]["kept"] += len(kept)
            audit["old_vs_new_overlap"]["added"] += len(added)
            audit["old_vs_new_overlap"]["removed"] += len(removed)
            # Replace element_ids with new line_no-based set
            node["element_ids_old_position_idx"] = list(old_eids)  # preserve audit
            node["element_ids"] = sorted(new_eids)
            node["element_ids_lineno"] = sorted(new_eids)

        audit["per_doc"][doc_id] = per_doc_stats

    # Save outputs
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    chunks_data["metadata"] = {**chunks_data.get("metadata", {}),
                               "rebuilt_with_lineno_2026_05_10": True,
                               "source_chunks": args.chunks,
                               "source_element_lineno": args.element_lineno}
    Path(args.output).write_text(json.dumps(chunks_data, ensure_ascii=False))
    audit["old_vs_new_overlap"] = dict(audit["old_vs_new_overlap"])
    Path(args.audit).write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    pm = audit["paragraphs_matched"]
    pt = audit["paragraphs_total"]
    cr = audit["chunks_with_line_range"]
    ct = audit["chunks_total"]
    ov = audit["old_vs_new_overlap"]
    print(f"Wrote {args.output} + {args.audit}")
    print(f"  Docs processed: {audit['docs_processed']}/{len(target_ids)} (skipped no_tex: {audit['docs_skipped_no_tex']})")
    print(f"  Paragraphs matched: {pm}/{pt} = {pm/pt*100 if pt else 0:.1f}%")
    print(f"  Chunks with line_range: {cr}/{ct} = {cr/ct*100 if ct else 0:.1f}%")
    print(f"  Element edges: kept={ov.get('kept',0)} added={ov.get('added',0)} removed={ov.get('removed',0)}")


if __name__ == "__main__":
    main()
