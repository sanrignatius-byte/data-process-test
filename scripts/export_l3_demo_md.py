#!/usr/bin/env python3
"""
将 L3 demo evidence 文件夹（由 package_l3_demo_evidence.py 生成）导出为单个 Markdown 文件。

每条 query 一个章节，包含：
  - Query / Answer / Reasoning Steps
  - Element A/B 截图（图片引用）或 LaTeX（formula）
  - Element A/B 上下文
  - 桥接段落

无额外依赖。

用法:
  python scripts/export_l3_demo_pdf.py
  python scripts/export_l3_demo_pdf.py --evidence-dir data/m2/l3_demo_evidence --output data/m2/l3_demo_evidence.md
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def render_query(folder: pathlib.Path, idx: int, use_relative: pathlib.Path) -> str:
    """Render a single query folder to Markdown."""
    info_path = folder / "query_info.json"
    if not info_path.exists():
        return ""

    info = json.loads(info_path.read_text(encoding="utf-8"))
    lines = []

    # ── title ──
    qid = info.get("query_id", "?")
    lines.append(f"## Query {idx}: `{qid}`")
    lines.append("")

    # ── metadata table ──
    pair_type = info.get("pair_type", "")
    is_cross = info.get("is_cross_doc", False)
    hop = info.get("hop_distance", "")
    depth = info.get("reasoning_depth", "")
    eids = info.get("element_ids", [])
    path = info.get("path", [])

    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Pair Type | {pair_type} |")
    lines.append(f"| Cross-doc | {is_cross} |")
    lines.append(f"| Hop Distance | {hop} |")
    lines.append(f"| Reasoning Depth | {depth} |")
    lines.append(f"| Elements | `{'` , `'.join(eids)}` |")
    lines.append(f"| Path | {'` → `'.join(path)} |")
    lines.append("")

    # ── query & answer ──
    lines.append("### Query")
    lines.append("")
    lines.append(f"> {info.get('query', '')}")
    lines.append("")

    lines.append("### Answer")
    lines.append("")
    lines.append(info.get("answer", ""))
    lines.append("")

    # ── reasoning steps ──
    steps = info.get("reasoning_steps", [])
    if steps:
        lines.append("### Reasoning Steps")
        lines.append("")
        for step in steps:
            sid = step.get("step_id", "?")
            role = step.get("reasoning_role", "")
            etype = step.get("evidence_type", "")
            deps = step.get("depends_on_steps", [])
            span = step.get("evidence_span", "")
            claim = step.get("produces_claim", "")

            lines.append(f"**Step {sid}** [{role}] (evidence: {etype}, depends: {deps})")
            lines.append("")
            if span:
                lines.append(f"- **Evidence:** {span}")
            if claim:
                lines.append(f"- **Claim:** {claim}")
            lines.append("")

    # ── elements ──
    for tag in ["a", "b"]:
        img_files = sorted(folder.glob(f"element_{tag}_*.*"))
        img_files = [f for f in img_files
                     if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp")
                     and "NOT_FOUND" not in f.name]
        formula_files = sorted(folder.glob(f"element_{tag}_formula.md"))
        ctx_file = folder / f"element_{tag}_context.md"

        if not img_files and not formula_files and not ctx_file.exists():
            continue

        lines.append(f"### Element {tag.upper()}")
        lines.append("")

        # images
        for img in img_files:
            rel = img.relative_to(use_relative)
            # Use forward slashes for markdown compatibility
            rel_str = str(rel).replace("\\", "/")
            lines.append(f"![{img.stem}]({rel_str})")
            lines.append("")

        # formula
        for fm in formula_files:
            text = fm.read_text(encoding="utf-8", errors="ignore")
            lines.append("```latex")
            lines.append(text.strip())
            lines.append("```")
            lines.append("")

        # context
        if ctx_file.exists():
            ctx = ctx_file.read_text(encoding="utf-8", errors="ignore")
            if len(ctx) > 3000:
                ctx = ctx[:3000] + "\n... (truncated)"
            lines.append("<details>")
            lines.append(f"<summary>Element {tag.upper()} Context</summary>")
            lines.append("")
            lines.append(ctx.strip())
            lines.append("")
            lines.append("</details>")
            lines.append("")

    # ── bridge paragraphs ──
    bridge_path = folder / "bridge_paragraphs.md"
    if bridge_path.exists():
        bridge = bridge_path.read_text(encoding="utf-8", errors="ignore").strip()
        if bridge:
            lines.append("### Bridge Paragraphs")
            lines.append("")
            if len(bridge) > 4000:
                bridge = bridge[:4000] + "\n... (truncated)"
            lines.append(bridge)
            lines.append("")

    # ── evidence spans ──
    spans = info.get("required_evidence_spans", [])
    if spans:
        lines.append("### Required Evidence Spans")
        lines.append("")
        for i, span in enumerate(spans, 1):
            lines.append(f"{i}. {span}")
        lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Export L3 demo evidence to Markdown")
    parser.add_argument(
        "--evidence-dir",
        type=pathlib.Path,
        default=ROOT / "data/m2/l3_demo_evidence",
        help="Evidence folder (output of package_l3_demo_evidence.py)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "data/m2/l3_demo_evidence.md",
        help="Output Markdown path",
    )
    args = parser.parse_args()

    evidence_dir: pathlib.Path = args.evidence_dir
    output_path: pathlib.Path = args.output

    if not evidence_dir.exists():
        print(f"ERROR: evidence dir not found: {evidence_dir}")
        print("Run package_l3_demo_evidence.py first.")
        sys.exit(1)

    folders = sorted([d for d in evidence_dir.iterdir() if d.is_dir()])
    if not folders:
        print(f"ERROR: no query folders found in {evidence_dir}")
        sys.exit(1)

    print(f"Evidence dir: {evidence_dir}")
    print(f"Query folders: {len(folders)}")
    print(f"Output: {output_path}")
    print()

    # Image paths will be relative to the output file's parent
    rel_base = output_path.parent

    parts = []

    # Cover
    parts.append("# L3 Demo Evidence Report")
    parts.append("")
    parts.append(f"**{len(folders)} Reasoning-Chain Queries** | Multi-hop | Multi-modal | Cross-document")
    parts.append("")
    parts.append("---")
    parts.append("")

    for idx, folder in enumerate(folders, 1):
        print(f"  [{idx:02d}] {folder.name}")
        parts.append(render_query(folder, idx, rel_base))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"\nDone! Markdown saved to: {output_path}")


if __name__ == "__main__":
    main()
