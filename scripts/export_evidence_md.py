#!/usr/bin/env python3
"""Export evidence-pinning MD files for each query.

For each query in the input JSONL, generates a Markdown file that:
  - Shows the query, answer, reasoning chain
  - For each evidence element: caption, content, context, image (if any)
  - Lists required_evidence_spans, visual_anchors, text_evidence

Usage:
    python scripts/export_evidence_md.py \
        --queries data/03_queries/m2_diverse_v1_hub_kb_pass.jsonl \
        --elements data/01_graphs/multimodal_elements.json \
        --output-dir data/06_evidence_export/m2_diverse_v1 \
        --summary
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SUMMARY_QUERY_MAX_LEN = 80

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.image_utils import resolve_image_path  # noqa: E402

# Markers used to extract the MinerU-relative suffix from any image_path format.
_MINERU_MARKERS = ("/data/00_raw/mineru_output/", "/data/mineru_output/")

# The canonical local directory (relative to project root) where MinerU images live.
_LOCAL_MINERU_BASE = Path("data") / "00_raw" / "mineru_output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_relative(abs_path: Path, md_dir: Path) -> str:
    """Make a path relative to the MD file's directory for embedding.

    Always returns forward-slash paths so ``<img src="...">`` works on all
    platforms (Windows ``os.path.relpath`` returns backslash paths).
    """
    try:
        return os.path.relpath(abs_path, md_dir).replace("\\", "/")
    except ValueError:
        return str(abs_path).replace("\\", "/")


def _html_table_to_md(html: str) -> str:
    """Best-effort convert simple <table> HTML to Markdown table."""
    if not html or "<table" not in html.lower():
        return html
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE)
    if not rows:
        return html
    md_rows: List[str] = []
    for i, row in enumerate(rows):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL | re.IGNORECASE)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        md_rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            md_rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(md_rows)


def _truncate(text: str, max_len: int = 1000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n\n... (truncated)"


def _extract_mineru_suffix(raw_path: str) -> Optional[str]:
    """Extract the MinerU-relative suffix from any image_path format.

    Handles all known path formats:
      - /projects/_hdd/myyyx1/.../data/mineru_output/SUFFIX
      - /projects/myyyx1/.../data/mineru_output/SUFFIX
      - data/mineru_output/SUFFIX
      - data/00_raw/mineru_output/SUFFIX
      - Windows backslash variants of the above

    Returns the SUFFIX part (e.g. ``1711.07076/1711.07076/hybrid_auto/images/foo.jpg``),
    or ``None`` if the path doesn't match any known pattern.
    """
    if not raw_path:
        return None
    normed = raw_path.replace("\\", "/")
    for marker in _MINERU_MARKERS:
        idx = normed.find(marker)
        if idx >= 0:
            return normed[idx + len(marker):]
    # Also handle relative paths without leading /data/
    for prefix in ("data/00_raw/mineru_output/", "data/mineru_output/"):
        if normed.startswith(prefix):
            return normed[len(prefix):]
    return None


def _resolve_image_for_md(
    raw_path: Optional[str],
    output_dir: Path,
) -> Optional[str]:
    """Resolve an image_path into a relative ``src`` suitable for ``<img>`` in MD.

    Two-stage approach:
      1. Try :func:`resolve_image_path` — if the file exists locally, compute
         a verified relative path.
      2. **Fallback**: extract the MinerU suffix and build a deterministic
         relative path ``../../00_raw/mineru_output/SUFFIX`` that will work
         when the MD is viewed on the user's local machine (even if the file
         doesn't exist on the current machine).

    Returns a forward-slash relative path string, or ``None`` if the raw_path
    cannot be mapped at all.
    """
    if not raw_path:
        return None

    # Stage 1: verified resolve (file exists on disk)
    resolved = resolve_image_path(raw_path)
    if resolved:
        return _make_relative(resolved, output_dir)

    # Stage 2: deterministic fallback from suffix
    suffix = _extract_mineru_suffix(raw_path)
    if suffix:
        expected_local = PROJECT_ROOT / _LOCAL_MINERU_BASE / suffix
        return _make_relative(expected_local, output_dir)

    return None


# ---------------------------------------------------------------------------
# Element index
# ---------------------------------------------------------------------------

def load_element_index(elements_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load multimodal_elements.json into a flat {element_id: element_dict}."""
    with open(elements_path, encoding="utf-8") as f:
        data = json.load(f)

    index: Dict[str, Dict[str, Any]] = {}
    docs = data.get("documents", {})
    for _doc_id, doc in docs.items():
        elements = doc.get("elements", {})
        if isinstance(elements, dict):
            for eid, edata in elements.items():
                index[eid] = edata
        elif isinstance(elements, list):
            for edata in elements:
                eid = edata.get("element_id", "")
                if eid:
                    index[eid] = edata
    return index


# ---------------------------------------------------------------------------
# MD generation
# ---------------------------------------------------------------------------

def generate_evidence_md(
    query: Dict[str, Any],
    element_index: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> tuple:
    """Generate one MD file for a single query.

    Returns ``(md_path, images_embedded, images_missing)`` counts.
    """
    qid = query["query_id"]
    md_path = output_dir / f"{qid}.md"
    images_embedded = 0
    images_missing = 0

    lines: List[str] = []

    # ── Header ──
    lines.append(f"# Evidence Report: {qid}\n")
    lines.append(f"**Difficulty Level**: {query.get('difficulty_level', '?')}  ")
    lines.append(f"**Difficulty Label**: {query.get('difficulty_label', '?')}  ")
    lines.append(f"**Pair Type**: {query.get('pair_type', '?')}  ")
    lines.append(f"**Doc ID**: {query.get('doc_id', '?')}  ")
    lines.append(f"**Pair ID**: {query.get('pair_id', '?')}  ")
    lines.append(f"**Query Style**: {query.get('query_style', '?')}  ")
    lines.append("")

    # ── Query & Answer ──
    lines.append("## Query\n")
    lines.append(f"> {query.get('query', '')}\n")

    lines.append("## Answer\n")
    lines.append(query.get("answer", "") + "\n")

    # ── Reasoning ──
    chain = query.get("reasoning_chain", "")
    if chain:
        lines.append("## Reasoning Chain\n")
        lines.append(chain + "\n")

    steps = query.get("reasoning_steps", [])
    if steps:
        lines.append("## Reasoning Steps\n")
        for i, step in enumerate(steps, 1):
            if isinstance(step, dict):
                lines.append(f"{i}. **{step.get('step', '')}**: {step.get('evidence', '')}")
            else:
                lines.append(f"{i}. {step}")
        lines.append("")

    # ── Evidence Elements ──
    element_ids = query.get("element_ids", [])
    lines.append(f"## Evidence Elements ({len(element_ids)})\n")

    for eid in element_ids:
        elem = element_index.get(eid, {})
        etype = elem.get("element_type", "unknown")
        lines.append(f"### {eid} (`{etype}`)\n")

        # Caption
        caption = elem.get("caption", "")
        if caption:
            lines.append(f"**Caption**: {caption}\n")

        # Image — deterministic path builder (works even if file absent on this machine)
        img_path_raw = elem.get("image_path")
        img_rel = _resolve_image_for_md(img_path_raw, output_dir)
        if img_rel:
            lines.append(f'<img src="{html.escape(img_rel, quote=True)}" width="600">\n')
            images_embedded += 1
        elif img_path_raw:
            lines.append(f"*Image path (could not resolve)*: `{img_path_raw}`\n")
            images_missing += 1

        # Content (table → MD, formula → $$, else plain text)
        content = elem.get("content", "")
        if content:
            if "<table" in content.lower():
                lines.append("**Content (table)**:\n")
                lines.append(_html_table_to_md(content) + "\n")
            elif content.startswith("$$") or content.startswith("\\"):
                # Avoid double-wrapping if content already has $$ delimiters
                stripped = content.strip()
                if stripped.startswith("$$") and stripped.endswith("$$"):
                    lines.append(f"**Content (formula)**:\n\n{stripped}\n")
                else:
                    lines.append(f"**Content (formula)**:\n\n$$\n{content}\n$$\n")
            else:
                lines.append(f"**Content**:\n\n{_truncate(content)}\n")

        # Context before / after (collapsed)
        ctx_before = elem.get("context_before", "")
        ctx_after = elem.get("context_after", "")
        if ctx_before:
            lines.append("<details>")
            lines.append("<summary>Context Before</summary>\n")
            lines.append(_truncate(ctx_before, 800) + "\n")
            lines.append("</details>\n")
        if ctx_after:
            lines.append("<details>")
            lines.append("<summary>Context After</summary>\n")
            lines.append(_truncate(ctx_after, 800) + "\n")
            lines.append("</details>\n")

        lines.append("---\n")

    # ── Required Evidence Spans ──
    spans = query.get("required_evidence_spans", [])
    if spans:
        lines.append("## Required Evidence Spans\n")
        lines.append("| Element ID | Span | Evidence Type |")
        lines.append("| --- | --- | --- |")
        for s in spans:
            lines.append(
                f"| `{s.get('element_id', '')}` "
                f"| {s.get('span', '')} "
                f"| {s.get('evidence_type', '')} |"
            )
        lines.append("")

    # ── Visual Anchors ──
    anchors = query.get("visual_anchors", [])
    if anchors:
        lines.append("## Visual Anchors\n")
        lines.append("| Element ID | Anchor |")
        lines.append("| --- | --- |")
        for a in anchors:
            lines.append(f"| `{a.get('element_id', '')}` | {a.get('anchor', '')} |")
        lines.append("")

    # ── Text Evidence ──
    text_ev = query.get("text_evidence", "")
    if text_ev:
        lines.append("## Text Evidence\n")
        lines.append(f"> {text_ev}\n")

    # ── Images from query ──
    image_paths = query.get("image_paths", [])
    if image_paths:
        lines.append("## Query-level Image Paths\n")
        for ip in image_paths:
            img_rel = _resolve_image_for_md(ip, output_dir)
            if img_rel:
                lines.append(f'<img src="{html.escape(img_rel, quote=True)}" width="600">\n')
                images_embedded += 1
            else:
                lines.append(f"- `{ip}` *(could not resolve)*\n")
                images_missing += 1
        lines.append("")

    # ── QC Info ──
    lines.append("## QC Metadata\n")
    lines.append(f"- **QC Pass**: {query.get('qc_pass', '?')}")
    qc_issues = query.get("qc_issues", [])
    if qc_issues:
        lines.append(f"- **QC Issues**: {', '.join(str(i) for i in qc_issues)}")
    else:
        lines.append("- **QC Issues**: none")
    lines.append(f"- **Quality Tier**: {query.get('quality_tier', '?')}")
    lines.append(f"- **Bridge Quality**: {query.get('bridge_quality', '?')}")
    lines.append("")

    # Write
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, images_embedded, images_missing


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export evidence-pinning MD files for queries",
    )
    parser.add_argument(
        "--queries", type=Path, required=True,
        help="Input JSONL file with queries",
    )
    parser.add_argument(
        "--elements", type=Path,
        default=PROJECT_ROOT / "data" / "01_graphs" / "multimodal_elements.json",
        help="multimodal_elements.json path",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data" / "06_evidence_export" / "m2_diverse_v1",
        help="Output directory for MD files",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Also generate a summary index.md",
    )
    args = parser.parse_args()

    # Load elements
    print(f"Loading elements from {args.elements} ...")
    element_index = load_element_index(args.elements)
    print(f"  → {len(element_index)} elements indexed")

    # Load queries
    queries: List[Dict[str, Any]] = []
    with open(args.queries, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    print(f"Loaded {len(queries)} queries from {args.queries}")

    # Create output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate MD for each query
    generated: List[Path] = []
    total_embedded = 0
    total_missing = 0
    for q in queries:
        md_path, n_embedded, n_missing = generate_evidence_md(
            q, element_index, args.output_dir,
        )
        generated.append(md_path)
        total_embedded += n_embedded
        total_missing += n_missing
        status = f"  ✓ {md_path.name}"
        if n_embedded:
            status += f"  ({n_embedded} images)"
        print(status)

    # Summary index
    if args.summary:
        idx_lines = [
            "# Evidence Export Index\n",
            f"Total: {len(generated)} queries\n",
            "| # | Query ID | Pair Type | Query | Elements |",
            "| --- | --- | --- | --- | --- |",
        ]
        for i, q in enumerate(queries, 1):
            qid = q["query_id"]
            query_text = q.get("query", "")[:SUMMARY_QUERY_MAX_LEN]
            eids = ", ".join(q.get("element_ids", []))
            idx_lines.append(
                f"| {i} | [{qid}]({qid}.md) "
                f"| {q.get('pair_type', '')} "
                f"| {query_text} "
                f"| {eids} |"
            )
        idx_path = args.output_dir / "index.md"
        idx_path.write_text("\n".join(idx_lines), encoding="utf-8")
        print(f"\n  ✓ index.md")

    print(f"\nDone! {len(generated)} evidence MD files exported to {args.output_dir}")
    print(f"  Images embedded: {total_embedded}")
    if total_missing:
        print(f"  Images unresolved: {total_missing}")


if __name__ == "__main__":
    main()
