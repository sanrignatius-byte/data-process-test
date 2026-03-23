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
import os
import pathlib
import platform
import sys
import textwrap
from typing import List, Optional, Tuple

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

# ─────────────────────────── Unicode font discovery ───────────────────────────

# Font family name used throughout (set after registration)
FONT_SANS = "unis"   # will be overridden to TTF family name if available
FONT_MONO = "unim"


def _find_ttf() -> Tuple[Optional[pathlib.Path], Optional[pathlib.Path]]:
    """Find a Unicode-capable TTF on the system. Returns (sans, mono) or (None, None)."""
    candidates_sans = []
    candidates_mono = []
    system = platform.system()

    if system == "Windows":
        windir = pathlib.Path(os.environ.get("WINDIR", r"C:\Windows"))
        fonts = windir / "Fonts"
        # Prefer: Arial (widely available), then DejaVu, then Segoe UI
        candidates_sans = [
            fonts / "arial.ttf",
            fonts / "DejaVuSans.ttf",
            fonts / "segoeui.ttf",
            fonts / "calibri.ttf",
        ]
        candidates_mono = [
            fonts / "consola.ttf",
            fonts / "cour.ttf",
            fonts / "DejaVuSansMono.ttf",
        ]
    elif system == "Darwin":
        candidates_sans = [
            pathlib.Path("/System/Library/Fonts/Helvetica.ttc"),
            pathlib.Path("/Library/Fonts/Arial.ttf"),
            pathlib.Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        ]
        candidates_mono = [
            pathlib.Path("/System/Library/Fonts/Courier.ttc"),
            pathlib.Path("/Library/Fonts/Courier New.ttf"),
        ]
    else:  # Linux
        candidates_sans = [
            pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            pathlib.Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
            pathlib.Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        ]
        candidates_mono = [
            pathlib.Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
            pathlib.Path("/usr/share/fonts/TTF/DejaVuSansMono.ttf"),
            pathlib.Path("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"),
        ]

    sans = next((p for p in candidates_sans if p.exists()), None)
    mono = next((p for p in candidates_mono if p.exists()), None)
    return sans, mono


def _register_fonts(pdf: "FPDF") -> Tuple[str, str]:
    """Register Unicode TTF fonts on the PDF instance. Returns (sans_name, mono_name)."""
    global FONT_SANS, FONT_MONO
    sans_path, mono_path = _find_ttf()

    if sans_path:
        pdf.add_font(FONT_SANS, "", str(sans_path))
        # Try bold variant (arial bold = arialbd.ttf on Windows)
        bold_path = sans_path.with_name(
            sans_path.stem + "bd" + sans_path.suffix  # arialbd.ttf
        )
        if not bold_path.exists():
            bold_path = sans_path.with_name(
                sans_path.stem + "-Bold" + sans_path.suffix
            )
        if bold_path.exists():
            pdf.add_font(FONT_SANS, "B", str(bold_path), )
        else:
            pdf.add_font(FONT_SANS, "B", str(sans_path), )

        # Italic variant
        italic_path = sans_path.with_name(
            sans_path.stem + "i" + sans_path.suffix  # ariali.ttf
        )
        if not italic_path.exists():
            italic_path = sans_path.with_name(
                sans_path.stem + "-Italic" + sans_path.suffix
            )
        if italic_path.exists():
            pdf.add_font(FONT_SANS, "I", str(italic_path), )
        else:
            pdf.add_font(FONT_SANS, "I", str(sans_path), )
    else:
        FONT_SANS = "Helvetica"

    if mono_path:
        pdf.add_font(FONT_MONO, "", str(mono_path), )
    else:
        FONT_MONO = "Courier"

    return FONT_SANS, FONT_MONO


# ─────────────────────────── helpers ───────────────────────────

def sanitize_latin1(text: str) -> str:
    """Replace common Unicode chars with ASCII equivalents (fallback when no TTF)."""
    replacements = {
        "\u2013": "-", "\u2014": "--", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " ",
        "\u2192": "->", "\u2190": "<-", "\u2264": "<=", "\u2265": ">=",
        "\u00d7": "x", "\u2022": "*", "\u00b1": "+/-", "\u03b1": "alpha",
        "\u03b2": "beta", "\u03b3": "gamma", "\u03b4": "delta",
        "\u0394": "Delta", "\u03c3": "sigma", "\u03bc": "mu",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Final safety: drop any remaining non-latin1
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _strip_unsupported(text: str) -> str:
    """Remove/replace characters outside typical Western TTF coverage (CJK, math symbols)."""
    replacements = {
        "\u2013": "-", "\u2014": "--", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " ",
        "\u2192": "->", "\u2190": "<-", "\u2264": "<=", "\u2265": ">=",
        "\u00d7": "x", "\u2022": "*", "\u00b1": "+/-",
        "\u2208": " in ", "\u2209": " not in ", "\u2282": " subset ",
        "\u2286": " subseteq ", "\u221e": "inf", "\u2248": "~=",
        "\u2260": "!=", "\u2227": " and ", "\u2228": " or ",
        "\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma",
        "\u03b4": "delta", "\u0394": "Delta", "\u03c3": "sigma",
        "\u03bc": "mu", "\u03bb": "lambda", "\u03c0": "pi",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Strip any remaining non-BMP or CJK characters
    cleaned = []
    for ch in text:
        cp = ord(ch)
        # Keep ASCII, Latin Extended, common punctuation
        if cp < 0x2000 or (0x2000 <= cp <= 0x206F):  # general punctuation ok
            cleaned.append(ch)
        elif 0x2100 <= cp <= 0x214F:  # letterlike symbols
            cleaned.append(ch)
        elif cp > 0x024F and cp < 0xFB00:
            # CJK, Arabic, Devanagari, etc. — replace with ?
            cleaned.append("?")
        else:
            cleaned.append(ch)
    return "".join(cleaned)


def safe_text(text: str) -> str:
    """Sanitize text for PDF rendering, handling both TTF and built-in fonts."""
    if FONT_SANS == "Helvetica":
        return sanitize_latin1(text)
    return _strip_unsupported(text)


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
    """Custom PDF with header/footer and Unicode support."""

    def header(self):
        self.set_font(FONT_SANS, "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "L3 Demo Evidence Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font(FONT_SANS, "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ── convenience writers ──

    def _reset_x(self):
        """Reset cursor to left margin to prevent width exhaustion."""
        self.set_x(self.l_margin)

    def section_title(self, title: str):
        self._reset_x()
        self.set_font(FONT_SANS, "B", 12)
        self.set_text_color(30, 30, 120)
        self.multi_cell(0, 8, safe_text(title))
        self.ln(1)

    def sub_title(self, title: str):
        self._reset_x()
        self.set_font(FONT_SANS, "B", 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 7, safe_text(title))
        self.ln(1)

    def body_text(self, text: str, size: int = 9):
        self._reset_x()
        self.set_font(FONT_SANS, "", size)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 4.5, safe_text(wrap_text(text, width=100)))
        self.ln(2)

    def kv_line(self, key: str, value: str):
        self._reset_x()
        self.set_font(FONT_SANS, "B", 9)
        self.set_text_color(0, 0, 0)
        if len(value) > 120:
            value = value[:117] + "..."
        self.multi_cell(0, 5, safe_text(f"{key}: {value}"))

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

            pdf._reset_x()
            pdf.set_font(FONT_SANS, "B", 9)
            pdf.set_text_color(30, 30, 120)
            pdf.multi_cell(0, 5, safe_text(f"Step {sid}  [{role}]  (evidence: {etype}, depends: {deps})"))
            pdf._reset_x()
            pdf.set_font(FONT_SANS, "", 8)
            pdf.set_text_color(0, 0, 0)
            if span:
                pdf._reset_x()
                pdf.multi_cell(0, 4, safe_text(wrap_text(f"Evidence: {span}", 95)))
            if claim:
                pdf._reset_x()
                pdf.multi_cell(0, 4, safe_text(wrap_text(f"Claim: {claim}", 95)))
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
            pdf._reset_x()
            pdf.set_font(FONT_MONO, "", 8)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 4, safe_text(wrap_text(text, 100)))
            pdf.ln(2)

        # show context
        if ctx_file.exists():
            ctx = ctx_file.read_text(encoding="utf-8", errors="ignore")
            if len(ctx) > 2000:
                ctx = ctx[:2000] + "\n... (truncated)"
            pdf._reset_x()
            pdf.set_font(FONT_SANS, "I", 8)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(0, 4, safe_text(wrap_text(ctx, 100)))
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
            pdf._reset_x()
            pdf.set_font(FONT_SANS, "", 8)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 4, safe_text(f"{i}. {wrap_text(str(span), 95)}"))
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

    # Register Unicode fonts
    sans, mono = _register_fonts(pdf)
    print(f"Fonts: sans={sans} ({FONT_SANS}), mono={mono} ({FONT_MONO})")

    # Cover page
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font(FONT_SANS, "B", 24)
    pdf.set_text_color(30, 30, 120)
    pdf.cell(0, 15, "L3 Demo Evidence Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font(FONT_SANS, "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, f"{len(folders)} Reasoning-Chain Queries", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.cell(0, 10, "Multi-hop | Multi-modal | Cross-document", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font(FONT_SANS, "", 10)
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
