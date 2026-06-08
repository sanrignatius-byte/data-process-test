#!/usr/bin/env python3
"""Build MinerU graph from realworld documents (pure markdown, no LaTeX).

Parses MinerU markdown output to extract:
  - Sections (h1-h6 headers)
  - Text paragraphs
  - Figures (![](images/...) with captions)
  - Tables (<table>...</table>)

Builds graph edges:
  - section_contains (section → child elements)
  - next_element / prev_element (reading order)
  - table/figure_reference (text → table/figure mention)
  - co_reference (two elements referenced in same paragraph)

Output: unified multimodal_elements-style JSON for downstream query generation.
"""

import argparse, json, re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MINERU_DIR = ROOT / "data/00_raw/realworld_mineru_output"
DEFAULT_OUT = ROOT / "data/01_graphs/realworld_multimodal_elements.json"

# ── Markdown parsing ────────────────────────────────────────────────

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
IMAGE_RE = re.compile(r"!\[(.*?)\]\(((?:https?:)?//[^\)]+|[^\)]+)\)")
# Filter decorative UI images (logos, icons, spacers, data URIs, tiny SVGs)
DECORATIVE_IMAGE_RE = re.compile(
    r'(?:logo|icon|opacity0|close_|toRight|toLeft|arrow|spacer|placeholder|1px)',
    re.IGNORECASE,
)
BAD_IMAGE_EXT = {'.svg'}  # skip tiny vector icons
TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL)
# Formula patterns. Display ($$...$$) is taken as-is; inline ($...$) must contain
# at least one LaTeX command (`\`) to be considered a real formula — guards against
# 10-K currency symbols like "$ 158,104" that the inline pattern would otherwise eat.
DISPLAY_FORMULA_RE = re.compile(r"\$\$([\s\S]+?)\$\$")
INLINE_FORMULA_RE = re.compile(r"(?<!\$)\$([^\$\n]{2,}?)\$(?!\$)")
LATEX_CMD_RE = re.compile(r"\\[A-Za-z]+")

HTML_TAG = re.compile(r"<[^>]+>")
MULTI_SPACE = re.compile(r"\s+")
MULTI_NL = re.compile(r"\n{3,}")

# Reference patterns
FIG_NUM_PAT = r"[\d]+(?:[.\-][\d]+)?"  # "4", "4-20", "3.17"
FIG_REF = re.compile(rf"\b(?:Figure|Fig\.?)\s*({FIG_NUM_PAT})", re.IGNORECASE)
TABLE_REF = re.compile(rf"\bTable\s*({FIG_NUM_PAT})", re.IGNORECASE)
SECTION_REF = re.compile(r"\b(?:Section|Sec\.?|§)\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
# Combined Table/Figure mention pattern for TOC-paragraph detection.
# Accepts both numeric IDs ("Table 4-20") AND appendix letter-prefixed IDs
# ("Table E-1", "Table A.3") which are common in long specs (USB-C, RFCs).
ANY_REF_RE = re.compile(
    r"\b(?:Table|Figure|Fig\.?)\s+(?:[A-Z][.\-])?\d+(?:[.\-]\d+)?",
    re.IGNORECASE,
)
# Cross-section / forward / backward reference intent words. A long-range edge
# (text → table beyond TABLE_REF_MAX_DIST) requires the source paragraph to
# contain at least one of these — i.e. the author explicitly signaled they were
# pointing somewhere far away.
INTENT_RE = re.compile(
    r"\b(?:see|shown|presented|illustrated|described|listed|summari[sz]ed|"
    r"refer(?:red|s|ring)?\s+to|previously|earlier|later|above|below|"
    r"preceding|following|in\s+the\s+(?:previous|next|prior|earlier|later))\b",
    re.IGNORECASE,
)

# Table-anchor patterns (used when caption-based "Table N" linking fails)
TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
TD_RE = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.DOTALL | re.IGNORECASE)
NUMERIC_CELL_RE = re.compile(r"^[\s\$€£¥]*[\(\-]?[\d,]+(?:\.\d+)?[\)%]?\s*$")
DISTINCTIVE_NUM_RE = re.compile(r"\$\s*[\d,]{4,}(?:\.\d+)?|\b[\d,]{4,}(?:\.\d+)?\s*%|\b\d{1,3}(?:,\d{3}){1,}(?:\.\d+)?\b")
ANCHOR_STOPWORDS = {
    "total", "other", "net", "gross", "items", "notes", "page", "the", "and",
    "year", "years", "period", "balance", "amount", "change", "yes", "no", "n/a",
}
# Reject row labels that look like calendar dates / fiscal headers (universal in 10-Ks).
DATE_LIKE_RE = re.compile(
    r"^(?:(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\s*\d*[\s,.]*|"
    r"as of [\w\s,]+\d{4}[\s,.]*|year ended[\s\S]*|fiscal year[\s\S]*|q[1-4][\s\S]*)$",
    re.IGNORECASE,
)


def clean_html(text: str) -> str:
    return MULTI_SPACE.sub(" ", HTML_TAG.sub("", text)).strip()


def extract_table_anchors(html: str, pre_text: str = "") -> dict:
    """Pull anchor signals from a MinerU HTML table — caption-free attribution.

    Returns:
        first_cell:      de facto title (top-left non-empty non-numeric cell)
        header_cells:    first row cells (column schema / time axis)
        row_labels:      column-0 non-numeric cells (financial line items, etc.)
        numeric_anchors: distinctive currency / comma-formatted values
        pre_caption:     last sentence of the text block preceding the table
    """
    rows = TR_RE.findall(html or "")
    grid: list[list[str]] = []
    for tr in rows:
        cells = [clean_html(c) for c in TD_RE.findall(tr)]
        grid.append(cells)

    first_cell = ""
    if grid:
        for c in grid[0]:
            if c and not NUMERIC_CELL_RE.match(c):
                first_cell = c
                break

    header_cells = grid[0] if grid else []

    row_labels: list[str] = []
    for r in grid[1:]:
        if not r:
            continue
        c = r[0].strip()
        if not c or NUMERIC_CELL_RE.match(c):
            continue
        if len(c) < 4 or c.lower() in ANCHOR_STOPWORDS:
            continue
        row_labels.append(c)

    numeric_anchors: list[str] = []
    for r in grid:
        for c in r:
            for m in DISTINCTIVE_NUM_RE.finditer(c):
                tok = m.group(0).strip()
                tok = MULTI_SPACE.sub(" ", tok)
                if tok and tok not in numeric_anchors:
                    numeric_anchors.append(tok)

    pre_caption = ""
    if pre_text:
        tail = pre_text.strip()
        if tail.endswith(":"):
            pre_caption = tail.rsplit(".", 1)[-1].strip(": ").strip()
        else:
            parts = re.split(r"(?<=[\.!?])\s+", tail)
            if parts:
                pre_caption = parts[-1].strip()

    return {
        "first_cell": first_cell,
        "header_cells": header_cells,
        "row_labels": row_labels,
        "numeric_anchors": numeric_anchors,
        "pre_caption": pre_caption,
    }


def parse_markdown(md_text: str, doc_id: str, images_dir: Path) -> dict:
    """Parse markdown into elements and edges."""
    elements: list[dict] = []
    
    # Split into blocks: headers, tables, images, paragraphs
    # Strategy: find all special blocks first, then remaining text is paragraphs
    
    # 1. Find all images with captions
    img_positions = []
    for m in IMAGE_RE.finditer(md_text):
        img_path = m.group(2)
        # Find caption: text after image until next blank line or header
        end = md_text.find("\n\n", m.end())
        if end == -1:
            end = len(md_text)
        cap_block = md_text[m.end():end]
        # Extract first meaningful line as caption
        caption = ""
        for line in cap_block.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("!["):
                caption = line
                break
        img_positions.append((m.start(), m.end(), "figure", img_path, caption, m.group(1)))

    # Filter out decorative UI images
    img_positions = [
        ip for ip in img_positions
        if not DECORATIVE_IMAGE_RE.search(ip[3])  # skip logo/icon/opacity0
        and not any(ip[3].lower().endswith(ext) for ext in BAD_IMAGE_EXT)  # skip SVGs
        and not ip[3].startswith("data:")  # skip data URIs
    ]
    
    # 2. Find all tables (with pre-table text snippet for anchor extraction)
    table_positions = []
    for m in TABLE_RE.finditer(md_text):
        table_html = m.group(0)
        caption = ""
        pre_window_start = max(0, m.start() - 500)
        pre_text = md_text[pre_window_start:m.start()]
        anchors = extract_table_anchors(table_html, pre_text)
        table_positions.append((m.start(), m.end(), "table", table_html, caption, anchors))
    
    # 3. Find all sections (headers)
    header_positions = []
    for m in HEADER_RE.finditer(md_text):
        level = len(m.group(1))
        title = m.group(2).strip()
        header_positions.append((m.start(), m.end(), "section", level, title))

    # 3b. Find formulas (outside any table — tables contain currency `$` symbols
    # in 10-K filings that the inline pattern would falsely match as formulas).
    table_ranges = [(p[0], p[1]) for p in table_positions]

    def _in_table(pos: int) -> bool:
        for s, e in table_ranges:
            if s <= pos < e:
                return True
        return False

    formula_positions = []
    for m in DISPLAY_FORMULA_RE.finditer(md_text):
        if _in_table(m.start()):
            continue
        body = m.group(1).strip()
        if not body:
            continue
        formula_positions.append((m.start(), m.end(), "formula", body, "display"))
    for m in INLINE_FORMULA_RE.finditer(md_text):
        if _in_table(m.start()):
            continue
        body = m.group(1).strip()
        # Require at least one LaTeX command (`\foo`) to qualify as a formula.
        # This is what separates "$\mathbb{S}$" / "$\geq 12$" from currency "$ 158,104".
        if not LATEX_CMD_RE.search(body):
            continue
        formula_positions.append((m.start(), m.end(), "formula", body, "inline"))

    # 4. Merge all positions and sort
    all_blocks = []
    for pos in img_positions:
        all_blocks.append(("figure", pos[0], pos[1], {"img_path": pos[3], "caption": pos[4], "alt": pos[5]}))
    for pos in table_positions:
        all_blocks.append(("table", pos[0], pos[1], {"html": pos[3], "caption": pos[4], "anchors": pos[5]}))
    for pos in header_positions:
        all_blocks.append(("section", pos[0], pos[1], {"level": pos[3], "title": pos[4]}))
    for pos in formula_positions:
        all_blocks.append(("formula", pos[0], pos[1], {"body": pos[3], "form": pos[4]}))
    
    all_blocks.sort(key=lambda x: x[1])  # sort by start position
    
    # 5. Extract text paragraphs between blocks
    cursor = 0
    element_idx = 0
    
    for block_type, start, end, meta in all_blocks:
        # Text before this block
        if cursor < start:
            text_block = md_text[cursor:start].strip()
            if text_block:
                # Split into paragraphs
                paras = MULTI_NL.split(text_block)
                for para in paras:
                    para = para.strip()
                    if para and len(para) > 10:
                        elements.append({
                            "element_id": f"{doc_id}_text_{element_idx}",
                            "doc_id": doc_id,
                            "element_type": "text",
                            "number": None,
                            "label": "",
                            "content": para,
                            "image_path": "",
                            "position_idx": element_idx,
                            "source": "mineru.markdown",
                        })
                        element_idx += 1
        
        # The block itself
        if block_type == "section":
            elements.append({
                "element_id": f"{doc_id}_section_{element_idx}",
                "doc_id": doc_id,
                "element_type": "section",
                "number": None,
                "label": meta["title"],
                "content": meta["title"],
                "image_path": "",
                "position_idx": element_idx,
                "source": "mineru.markdown",
                "metadata": {"header_level": meta["level"]},
            })
        elif block_type == "figure":
            img_path = meta["img_path"]
            # Remote URLs: keep as-is (even if images_dir exists)
            if img_path.startswith(("http://", "https://", "//")):
                full_img = img_path if img_path.startswith("http") else f"https:{img_path}"
            elif images_dir and (images_dir / Path(img_path).name).exists():
                full_img = str(images_dir / Path(img_path).name)
            else:
                full_img = img_path
            elements.append({
                "element_id": f"{doc_id}_figure_{element_idx}",
                "doc_id": doc_id,
                "element_type": "figure",
                "number": None,
                "label": meta.get("alt", ""),
                "caption": meta["caption"],
                "content": meta["caption"],
                "image_path": full_img,
                "position_idx": element_idx,
                "source": "mineru.markdown",
            })
        elif block_type == "table":
            table_text = clean_html(meta["html"])
            anchors = meta.get("anchors", {})
            de_facto_caption = anchors.get("pre_caption") or anchors.get("first_cell") or ""
            elements.append({
                "element_id": f"{doc_id}_table_{element_idx}",
                "doc_id": doc_id,
                "element_type": "table",
                "number": None,
                "label": anchors.get("first_cell", ""),
                "caption": meta["caption"] or de_facto_caption,
                "content": table_text,
                "image_path": "",
                "position_idx": element_idx,
                "source": "mineru.markdown",
                "anchors": anchors,
            })
        elif block_type == "formula":
            body = meta["body"]
            elements.append({
                "element_id": f"{doc_id}_formula_{element_idx}",
                "doc_id": doc_id,
                "element_type": "formula",
                "number": None,
                "label": "",
                "caption": "",
                "content": body,
                "image_path": "",
                "position_idx": element_idx,
                "source": "mineru.markdown",
                "metadata": {"form": meta.get("form", "inline")},
            })
        
        element_idx += 1
        cursor = end
    
    # Remaining text after last block
    if cursor < len(md_text):
        text_block = md_text[cursor:].strip()
        if text_block:
            paras = MULTI_NL.split(text_block)
            for para in paras:
                para = para.strip()
                if para and len(para) > 10:
                    elements.append({
                        "element_id": f"{doc_id}_text_{element_idx}",
                        "doc_id": doc_id,
                        "element_type": "text",
                        "number": None,
                        "label": "",
                        "content": para,
                        "image_path": "",
                        "position_idx": element_idx,
                        "source": "mineru.markdown",
                    })
                    element_idx += 1
    
    return elements


def build_edges(elements: list[dict]) -> list[dict]:
    """Build graph edges between elements."""
    edges = []
    seen = set()
    elements_by_id = {e["element_id"]: e for e in elements}
    
    def add(src, tgt, etype, weight=0.5, **meta):
        if src == tgt:
            return
        key = (src, tgt, etype)
        if key in seen:
            return
        seen.add(key)
        edges.append({"source_id": src, "target_id": tgt, "edge_type": etype, 
                      "weight": weight, "metadata": meta})
    
    # 1. Reading order (sequential)
    for left, right in zip(elements, elements[1:]):
        add(left["element_id"], right["element_id"], "next_element", 0.3)
        add(right["element_id"], left["element_id"], "prev_element", 0.3)
    
    # 2. Section containment (section → elements until next section)
    current_section = None
    for elem in elements:
        if elem["element_type"] == "section":
            current_section = elem["element_id"]
        elif current_section:
            add(current_section, elem["element_id"], "section_contains", 0.4)
    
    # 3. Figure/table number extraction and reference matching
    # Use full number strings (e.g. "4-20", "3.17") as keys, not just ints
    fig_nums: dict[str, str] = {}
    tbl_nums: dict[str, str] = {}
    for elem in elements:
        if elem["element_type"] == "figure":
            m = re.search(rf"Figure\s*({FIG_NUM_PAT})", elem.get("caption", ""), re.IGNORECASE)
            if not m:
                m = re.search(r"图\s*(\d+)", elem.get("caption", ""))
            if m:
                fig_nums[m.group(1)] = elem["element_id"]
        elif elem["element_type"] == "table":
            m = re.search(rf"Table\s*({FIG_NUM_PAT})", elem.get("caption", ""), re.IGNORECASE)
            if not m:
                m = re.search(r"表\s*(\d+)", elem.get("caption", ""))
            if m:
                tbl_nums[m.group(1)] = elem["element_id"]
    
    # 4. Reference edges (text → figure/table). The "Table N" / "Figure N"
    # regex branch is the original mechanism — kept verbatim for documents
    # that DO use academic-style numbering (datasheets, RFCs). The caption-free
    # anchor branch (4b) handles 10-K / legal / regulatory docs that don't.
    #
    # Both branches share a paragraph-level TOC filter: any source paragraph
    # mentioning ≥3 distinct "Table N" / "Figure N" tokens is almost certainly
    # a table-of-contents / index block, not a real reference, and would
    # otherwise generate hundreds of false long-range edges.

    def _norm(s: str) -> str:
        return MULTI_SPACE.sub(" ", s.lower().strip())

    def _named_entity(raw: str) -> bool:
        # At least TWO capitalized "real" words of ≥6 chars in the original casing.
        # Single-cap-word phrases like "Shares of common stock outstanding" are
        # business-jargon boilerplate that appears throughout 10-K cover pages;
        # genuine named entities like "Consolidated Statements of Operations"
        # carry multiple capitalized terms.
        n = 0
        for w in raw.split():
            stem = w.strip(",.()&:;'\"")
            if len(stem) >= 6 and stem[:1].isupper():
                n += 1
                if n >= 2:
                    return True
        return False

    for src in elements:
        if src["element_type"] != "text":
            continue
        text = src.get("content", "")
        if len(ANY_REF_RE.findall(text)) >= 3:
            continue  # D1: TOC paragraph — skip as reference source
        for m in FIG_REF.finditer(text):
            num = m.group(1)
            tgt = fig_nums.get(num)
            if tgt:
                add(src["element_id"], tgt, "figure_reference", 0.7, ref_text=m.group(0))
        for m in TABLE_REF.finditer(text):
            num = m.group(1)
            tgt = tbl_nums.get(num)
            if tgt:
                add(src["element_id"], tgt, "table_reference", 0.7, ref_text=m.group(0))

    # 4b. Caption-free table reference (for non-academic docs without "Table N").
    # Two indexes for phrases:
    #   phrase_short : permissive criteria, short-range only (dist ≤ TABLE_REF_MAX_DIST)
    #   phrase_long  : strict criteria, long-range allowed when source has intent words
    # Row-label and numeric anchors remain short-range only (too fine-grained
    # to safely cross sections — long-range explodes).
    phrase_short: dict[str, list[tuple[str, int]]] = defaultdict(list)
    phrase_long: dict[str, list[tuple[str, int]]] = defaultdict(list)
    rowlabel_to_tables: dict[str, list[tuple[str, int]]] = defaultdict(list)
    numeric_to_tables: dict[str, list[tuple[str, int]]] = defaultdict(list)

    for elem in elements:
        if elem["element_type"] != "table":
            continue
        tid = elem["element_id"]
        tpos = elem.get("position_idx", -1)
        a = elem.get("anchors", {}) or {}
        for raw in (a.get("first_cell", ""), a.get("pre_caption", "")):
            if not raw:
                continue
            p = _norm(raw)
            # D2: date / fiscal-header phrases are boilerplate, drop everywhere.
            if DATE_LIKE_RE.match(p):
                continue
            # Short-range candidate: ≥3 words AND ≥18 chars
            if len(p) >= 18 and len(p.split()) >= 3:
                phrase_short[p].append((tid, tpos))
            # D4: long-range candidate adds named-entity requirement on raw form
            # plus stricter length floor (≥4 words / ≥25 chars).
            if (
                len(p) >= 25
                and len(p.split()) >= 4
                and _named_entity(raw)
            ):
                phrase_long[p].append((tid, tpos))
        for rl in a.get("row_labels", [])[:20]:
            r = _norm(rl)
            if (
                len(r) >= 8 and len(r.split()) >= 2
                and r not in ANCHOR_STOPWORDS
                and not DATE_LIKE_RE.match(r)
            ):
                rowlabel_to_tables[r].append((tid, tpos))
        for n in a.get("numeric_anchors", [])[:30]:
            numeric_to_tables[_norm(n)].append((tid, tpos))

    # Doc-wide uniqueness gates
    unique_phrase_short = {k: v for k, v in phrase_short.items() if 1 <= len(v) <= 4}
    unique_phrase_long = {k: v for k, v in phrase_long.items() if len(v) == 1}
    unique_rowlabels = {k: v for k, v in rowlabel_to_tables.items() if 1 <= len(v) <= 3}
    unique_numeric = {k: v for k, v in numeric_to_tables.items() if 1 <= len(v) <= 2}

    TABLE_REF_MAX_DIST = 80
    MAX_TABLE_EDGES_PER_TEXT = 4
    for src in elements:
        if src["element_type"] != "text":
            continue
        body = src.get("content", "")
        body_lc = _norm(body)
        if len(body_lc) < 30:
            continue
        # D1: TOC filter — same threshold as the regex branch above.
        if len(ANY_REF_RE.findall(body)) >= 3:
            continue
        src_pos = src.get("position_idx", -1)
        has_intent = bool(INTENT_RE.search(body))
        added_targets: set[str] = set()

        def _maybe_add(tgt: str, tgt_pos: int, subtype: str, snippet: str,
                       weight: float = 0.6, long_range_ok: bool = False):
            if tgt in added_targets or len(added_targets) >= MAX_TABLE_EDGES_PER_TEXT:
                return
            if src_pos >= 0 and tgt_pos >= 0:
                dist = abs(src_pos - tgt_pos)
                if dist > TABLE_REF_MAX_DIST and not long_range_ok:
                    return
            added_targets.add(tgt)
            add(src["element_id"], tgt, "table_reference", weight,
                ref_text=snippet[:120], match=subtype)

        # Short-range branches (distance ≤ 80)
        for phrase, entries in unique_phrase_short.items():
            if phrase in body_lc:
                for tid, tpos in entries:
                    _maybe_add(tid, tpos, "anchor_phrase", phrase)
        for label, entries in unique_rowlabels.items():
            if label in body_lc:
                for tid, tpos in entries:
                    _maybe_add(tid, tpos, "row_label", label)
        for num, entries in unique_numeric.items():
            if num in body_lc:
                for tid, tpos in entries:
                    _maybe_add(tid, tpos, "numeric_anchor", num)

        # D3: long-range strong-anchor branch — requires explicit intent word
        # in the source paragraph. Higher weight (0.7) than short-range anchors.
        if has_intent:
            for phrase, entries in unique_phrase_long.items():
                if phrase in body_lc:
                    for tid, tpos in entries:
                        _maybe_add(tid, tpos, "anchor_phrase_long", phrase,
                                   weight=0.7, long_range_ok=True)
    
    # 5. Co-reference with distance cap (avoid long-doc cartesian explosion)
    # Only co-reference elements within CO_REF_MAX_DIST positions
    CO_REF_MAX_DIST = 50
    # Also track which section each element belongs to — only co-ref within same section
    elem_section: dict[str, str] = {}
    current_sec = None
    for elem in elements:
        if elem["element_type"] == "section":
            current_sec = elem["element_id"]
        elem_section[elem["element_id"]] = current_sec or ""
    
    for src in elements:
        if src["element_type"] != "text":
            continue
        text = src.get("content", "")
        refs_found = []
        for m in FIG_REF.finditer(text):
            num = m.group(1)
            tgt = fig_nums.get(num)
            if tgt and tgt not in refs_found:
                refs_found.append(tgt)
        for m in TABLE_REF.finditer(text):
            num = m.group(1)
            tgt = tbl_nums.get(num)
            if tgt and tgt not in refs_found:
                refs_found.append(tgt)
        
        if len(refs_found) >= 10:
            continue  # skip TOC/index paragraphs that list too many refs
        
        for i in range(len(refs_found)):
            for j in range(i + 1, len(refs_found)):
                ei = elements_by_id.get(refs_found[i], {})
                ej = elements_by_id.get(refs_found[j], {})
                pi = ei.get("position_idx", -1)
                pj = ej.get("position_idx", -1)
                sec_i = elem_section.get(refs_found[i], "")
                sec_j = elem_section.get(refs_found[j], "")
                
                # Distance + section gate
                if abs(pi - pj) > CO_REF_MAX_DIST:
                    continue
                if sec_i and sec_j and sec_i != sec_j:
                    continue  # different sections → skip
                
                add(refs_found[i], refs_found[j], "co_reference", 0.5,
                    paragraph_id=src["element_id"])
                add(refs_found[j], refs_found[i], "co_reference", 0.5,
                    paragraph_id=src["element_id"])
    
    return edges


# ── Quality scoring ──────────────────────────────────────────────────

def score_quality(elem: dict) -> float:
    score = 0.0
    if elem.get("caption"):
        score += 0.3
    if elem.get("content") and len(elem["content"]) > 50:
        score += 0.2
    if elem.get("image_path"):
        score += 0.15
    if elem.get("label"):
        score += 0.1
    etype = elem.get("element_type", "")
    if etype == "table":
        score += 0.15  # tables have structured info
    elif etype == "section":
        score += 0.1
    return min(1.0, score)


# ── Main ─────────────────────────────────────────────────────────────

def process_doc(doc_dir: Path) -> dict | None:
    doc_id = doc_dir.name
    md_file = doc_dir / f"{doc_id}.md"
    if not md_file.exists():
        return None
    
    md_text = md_file.read_text(encoding="utf-8", errors="replace")
    images_dir = doc_dir / "images"
    
    elements = parse_markdown(md_text, doc_id, images_dir)
    if not elements:
        return None
    
    edges = build_edges(elements)
    
    # Quality scoring
    for e in elements:
        e["quality_score"] = round(score_quality(e), 4)
    
    return {
        "doc_id": doc_id,
        "num_elements": len(elements),
        "num_edges": len(edges),
        "elements": {e["element_id"]: e for e in elements},
        "edges": edges,
    }


def main():
    parser = argparse.ArgumentParser(description="Build graph from realworld MinerU markdown")
    parser.add_argument("--mineru-dir", default=str(MINERU_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    
    mineru_root = Path(args.mineru_dir)
    doc_dirs = sorted(d for d in mineru_root.iterdir() if d.is_dir())
    
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    
    docs = {}
    all_edges = []
    type_counts = Counter()
    edge_counts = Counter()
    skipped = 0
    
    for doc_dir in doc_dirs:
        result = process_doc(doc_dir)
        if not result:
            skipped += 1
            print(f"  [SKIP] {doc_dir.name}")
            continue
        
        docs[doc_dir.name] = result
        for e in result["elements"].values():
            type_counts[e["element_type"]] += 1
        for edge in result["edges"]:
            edge["doc_id"] = doc_dir.name
            edge_counts[edge["edge_type"]] += 1
        all_edges.extend(result["edges"])
        
        print(f"  ✅ {doc_dir.name}: {result['num_elements']} elements, {result['num_edges']} edges")
    
    # Build unified output
    all_elements = {}
    for doc_id, doc in docs.items():
        all_elements.update(doc["elements"])
    
    summary = {
        "builder": "realworld_markdown_graph",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "docs_processed": len(docs),
        "docs_skipped": skipped,
        "total_elements": len(all_elements),
        "element_type_counts": dict(type_counts),
        "total_edges": len(all_edges),
        "edge_type_counts": dict(edge_counts),
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        **summary,
        "documents": {did: {"elements": doc["elements"]} for did, doc in docs.items()},
        "edges": all_edges,
    }
    
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    
    print(f"\n{'='*60}")
    print(f"Output: {output_path}")
    print(f"Docs:   {summary['docs_processed']} processed, {summary['docs_skipped']} skipped")
    print(f"Elements: {summary['total_elements']}")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    print(f"Edges: {summary['total_edges']}")
    for t, c in sorted(edge_counts.items()):
        print(f"  {t}: {c}")
    
    # Save summary separately
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
