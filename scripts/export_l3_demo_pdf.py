#!/usr/bin/env python3
"""
将 L3 demo evidence 文件夹（由 package_l3_demo_evidence.py 生成）导出为 PDF。

每条 query 占独立页面，包含：
  - Query / Answer / Reasoning Steps
  - Element A/B 截图（figure/table）或 LaTeX（formula）
  - Element A/B 上下文
  - 桥接段落

依赖:
  pip install fpdf2 Pillow

用法:
  python scripts/export_l3_demo_pdf.py
  python scripts/export_l3_demo_pdf.py --evidence-dir data/m2/l3_demo_evidence --output data/m2/l3_demo_evidence.pdf
"""
import argparse
import json
import pathlib
import sys
import textwrap
from typing import List, Optional

try:
    from fpdf import FPDF
except ImportError:
    print("ERROR: fpdf2 is required. Install with:  pip install fpdf2")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment,misc]
    print("WARNING: Pillow not found. Image sizing may be suboptimal. Install with:  pip install Pillow")

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ─────────────────────────── helpers ───────────────────────────

def wrap_text(text: str, width: int = 90) -> str:
    """Wrap long text for PDF output."""
    lines = text.split("\n")
    wrapped: List[str] = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(line)
        else:
            wrapped.extend(textwrap.wrap(line, width=width))
    return "\n".join(wrapped)


def get_image_dimensions(img_path: pathlib.Path, max_w: float, max_h: float):
    """Return (w, h) that fits within max_w x max_h, preserving aspect ratio."""
    if Image is not None:
        with Image.open(img_path) as im:
            iw, ih = im.size
    else:
        # fallback: assume 4:3
        iw, ih = 400, 300

    ratio = min(max_w / iw, max_h / ih, 1.0)
    return iw * ratio, ih * ratio


class DemoPDF(FPDF):
    """Custom PDF with header/footer."""

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "L3 Demo Evidence Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ── convenience writers ──

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 30, 120)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def sub_title(self, title: str):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text: str, size: int = 9):
        self.set_font("Helvetica", "", size)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 4.5, wrap_text(text, width=100))
        self.ln(2)

    def kv_line(self, key: str, value: str):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(0, 0, 0)
        self.cell(35, 5, f"{key}:")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, value, new_x="LMARGIN", new_y="NEXT")

    def add_image_safe(self, img_path: pathlib.Path, max_w: float = 170, max_h: float = 90):
        """Add image with auto-sizing. Page-break if needed."""
        if not img_path.exists():
            return
        w, h = get_image_dimensions(img_path, max_w, max_h)
        if self.get_y() + h + 10 > self.h - self.b_margin:
            self.add_page()
        self.image(str(img_path), w=w, h=h)
        self.ln(3)


# ─────────────────────────── per-query renderer ───────────────────────────

def render_query(pdf: DemoPDF, folder: pathlib.Path, idx: int):
    """Render a single query's evidence to the PDF."""
    info_path = folder / "query_info.json"
    if not info_path.exists():
        return

    info = json.loads(info_path.read_text(encoding="utf-8"))

    # ── page break for each query ──
    pdf.add_page()

    # ── title ──
    pdf.section_title(f"Query {idx}: {info.get('query_id', '?')}")

    # ── metadata ──
    pdf.kv_line("Pair Type", str(info.get("pair_type", "")))
    pdf.kv_line("Cross-doc", str(info.get("is_cross_doc", False)))
    pdf.kv_line("Hop Distance", str(info.get("hop_distance", "")))
    pdf.kv_line("Reasoning Depth", str(info.get("reasoning_depth", "")))
    eids = info.get("element_ids", [])
    pdf.kv_line("Elements", " , ".join(eids))
    path_str = " -> ".join(info.get("path", []))
    pdf.kv_line("Path", path_str if len(path_str) < 120 else path_str[:117] + "...")
    pdf.ln(3)

    # ── query & answer ──
    pdf.sub_title("Query")
    pdf.body_text(info.get("query", ""))

    pdf.sub_title("Answer")
    pdf.body_text(info.get("answer", ""))

    # ── reasoning steps ──
    steps = info.get("reasoning_steps", [])
    if steps:
        pdf.sub_title("Reasoning Steps")
        for step in steps:
            sid = step.get("step_id", "?")
            role = step.get("reasoning_role", "")
            etype = step.get("evidence_type", "")
            claim = step.get("produces_claim", "")
            span = step.get("evidence_span", "")
            deps = step.get("depends_on_steps", [])

            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(30, 30, 120)
            pdf.cell(0, 5, f"Step {sid}  [{role}]  (evidence: {etype}, depends: {deps})",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(0, 0, 0)
            if span:
                pdf.multi_cell(0, 4, f"  Evidence: {wrap_text(span, 95)}")
            if claim:
                pdf.multi_cell(0, 4, f"  Claim: {wrap_text(claim, 95)}")
            pdf.ln(2)

    # ── element images & context ──
    for tag in ["a", "b"]:
        # image files: element_a_figure.png, element_a_table.jpg, element_a_formula.md, etc.
        img_files = sorted(folder.glob(f"element_{tag}_*.*"))
        img_files = [f for f in img_files if f.suffix != ".md" and "NOT_FOUND" not in f.name]
        formula_files = [f for f in sorted(folder.glob(f"element_{tag}_formula.md"))]
        ctx_file = folder / f"element_{tag}_context.md"

        if not img_files and not formula_files and not ctx_file.exists():
            continue

        pdf.sub_title(f"Element {tag.upper()}")

        # show image
        for img in img_files:
            pdf.add_image_safe(img)

        # show formula markdown
        for fm in formula_files:
            text = fm.read_text(encoding="utf-8", errors="ignore")
            pdf.set_font("Courier", "", 8)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 4, wrap_text(text, 100))
            pdf.ln(2)

        # show context
        if ctx_file.exists():
            ctx = ctx_file.read_text(encoding="utf-8", errors="ignore")
            # truncate very long context
            if len(ctx) > 2000:
                ctx = ctx[:2000] + "\n... (truncated)"
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(0, 4, wrap_text(ctx, 100))
            pdf.ln(2)

    # ── bridge paragraphs ──
    bridge_path = folder / "bridge_paragraphs.md"
    if bridge_path.exists():
        bridge = bridge_path.read_text(encoding="utf-8", errors="ignore")
        if bridge.strip():
            pdf.sub_title("Bridge Paragraphs")
            if len(bridge) > 3000:
                bridge = bridge[:3000] + "\n... (truncated)"
            pdf.body_text(bridge, size=8)

    # ── evidence spans ──
    spans = info.get("required_evidence_spans", [])
    if spans:
        pdf.sub_title("Required Evidence Spans")
        for i, span in enumerate(spans, 1):
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 4, f"{i}. {wrap_text(str(span), 95)}")
        pdf.ln(2)


# ─────────────────────────── main ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export L3 demo evidence to PDF")
    parser.add_argument(
        "--evidence-dir",
        type=pathlib.Path,
        default=ROOT / "data/m2/l3_demo_evidence",
        help="Evidence folder (output of package_l3_demo_evidence.py)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=ROOT / "data/m2/l3_demo_evidence.pdf",
        help="Output PDF path",
    )
    args = parser.parse_args()

    evidence_dir: pathlib.Path = args.evidence_dir
    output_path: pathlib.Path = args.output

    if not evidence_dir.exists():
        print(f"ERROR: evidence dir not found: {evidence_dir}")
        print("Run package_l3_demo_evidence.py first.")
        sys.exit(1)

    # Collect query folders (sorted by name)
    folders = sorted([d for d in evidence_dir.iterdir() if d.is_dir()])
    if not folders:
        print(f"ERROR: no query folders found in {evidence_dir}")
        sys.exit(1)

    print(f"Evidence dir: {evidence_dir}")
    print(f"Query folders: {len(folders)}")
    print(f"Output PDF: {output_path}")
    print()

    # ── Build PDF ──
    pdf = DemoPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.alias_nb_pages()

    # Cover page
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(30, 30, 120)
    pdf.cell(0, 15, "L3 Demo Evidence Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, f"{len(folders)} Reasoning-Chain Queries", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.cell(0, 10, "Multi-hop | Multi-modal | Cross-document", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, "Generated by export_l3_demo_pdf.py", align="C", new_x="LMARGIN", new_y="NEXT")

    # Render each query
    for idx, folder in enumerate(folders, 1):
        print(f"  [{idx:02d}] {folder.name}")
        render_query(pdf, folder, idx)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    print(f"\nDone! PDF saved to: {output_path}")


if __name__ == "__main__":
    main()
