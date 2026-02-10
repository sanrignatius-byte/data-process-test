"""
Figure-Text Associator

Builds (figure, caption, context) associations from MinerU output.
Prefers structured content_list JSON; falls back to markdown parsing.
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Tuple
from enum import Enum


class FigureType(Enum):
    """Classification of figure content."""
    ARCHITECTURE = "architecture"    # Model/system architecture diagrams
    PLOT = "plot"                    # Experimental result plots/charts
    TABLE_IMG = "table_img"         # Tables rendered as images
    DIAGRAM = "diagram"             # Flowcharts, process diagrams
    EXAMPLE = "example"             # Example inputs/outputs, samples
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class FigureTextPair:
    """A figure associated with its textual context."""
    doc_id: str
    figure_id: str                          # e.g., "1104.3913_fig_1"
    figure_number: Optional[int]            # Logical figure number (Figure 1, 2, ...)
    image_path: str                         # Absolute path to image file
    image_filename: str                     # e.g., "1104.3913_page0_fig0.jpg"
    caption: str                            # "Figure 1: Tag cloud of top terms..."
    context_before: str                     # Text paragraph(s) before the figure
    context_after: str                      # Text paragraph(s) after the figure
    referring_paragraphs: List[str]         # All paragraphs that mention this figure
    figure_type: FigureType = FigureType.UNKNOWN
    sub_figures: List[str] = field(default_factory=list)  # Sub-caption labels: ["(a) Firehose", "(b) Streaming API"]
    quality_score: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def get_full_context(self) -> str:
        """Get combined context for query generation."""
        parts = []
        if self.caption:
            parts.append(f"Caption: {self.caption}")
        if self.context_before:
            parts.append(f"Before: {self.context_before}")
        if self.context_after:
            parts.append(f"After: {self.context_after}")
        if self.referring_paragraphs:
            parts.append(f"References: {' '.join(self.referring_paragraphs[:3])}")
        return "\n\n".join(parts)

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "figure_id": self.figure_id,
            "figure_number": self.figure_number,
            "image_path": self.image_path,
            "image_filename": self.image_filename,
            "caption": self.caption,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "referring_paragraphs": self.referring_paragraphs,
            "figure_type": self.figure_type.value,
            "sub_figures": self.sub_figures,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }


# Regex patterns
IMG_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
FIG_CAPTION_PATTERN = re.compile(
    r'(?:Figure|Fig\.?)\s*(\d+)\s*[:.]\s*(.*)',
    re.IGNORECASE
)
FIG_REF_PATTERN = re.compile(
    r'(?:Figure|Fig\.?)\s*(\d+)',
    re.IGNORECASE
)
TABLE_CAPTION_PATTERN = re.compile(
    r'Table\s*(\d+)\s*[:.]\s*(.*)',
    re.IGNORECASE
)
TABLE_REF_PATTERN = re.compile(
    r'Table\s*(\d+)',
    re.IGNORECASE
)
SUBFIG_PATTERN = re.compile(
    r'^\s*\(([a-z])\)\s*(.*)',
    re.IGNORECASE
)


class FigureTextAssociator:
    """
    Parses MinerU .md output to associate figures with their text context.

    Uses the inline image references in markdown which preserve the
    original document reading order.
    """

    def __init__(self, mineru_output_dir: str, context_window: int = 3):
        """
        Args:
            mineru_output_dir: Root directory of MinerU output (contains doc_id/ subdirs)
            context_window: Number of paragraphs before/after figure to capture
        """
        self.base_dir = Path(mineru_output_dir)
        self.context_window = context_window

    def process_all_documents(self) -> Dict[str, List[FigureTextPair]]:
        """Process all documents and return figure-text associations."""
        results = {}
        doc_dirs = sorted([
            d for d in self.base_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

        for doc_dir in doc_dirs:
            doc_id = doc_dir.name
            pairs = self.process_document(doc_id)
            if pairs:
                results[doc_id] = pairs

        return results

    def process_document(self, doc_id: str) -> List[FigureTextPair]:
        """Process a single document and return figure-text pairs."""
        doc_dir = self.base_dir / doc_id

        # Markdown is still useful for referring-paragraph lookup fallback.
        md_path = self._find_markdown(doc_dir, doc_id)
        blocks: List[Dict[str, Any]] = []
        images_dir: Optional[Path] = None
        if md_path:
            md_content = md_path.read_text(encoding='utf-8', errors='ignore')
            images_dir = md_path.parent / "images"
            blocks = self._parse_markdown_blocks(md_content)
        md_image_paths = self._collect_markdown_image_paths(blocks, images_dir) if blocks and images_dir else []

        # Prefer structured JSON extraction first (higher caption coverage).
        pairs = self._build_associations_from_content_list(doc_dir, doc_id, md_image_paths)
        if not pairs and blocks and images_dir:
            pairs = self._build_associations(blocks, doc_id, images_dir)
        if not pairs:
            return []

        # Find referring paragraphs for each figure
        all_paragraphs = [b for b in blocks if b["type"] == "text"]
        if all_paragraphs:
            for pair in pairs:
                if pair.figure_number is not None:
                    refs = self._find_referring_paragraphs(all_paragraphs, pair.figure_number)
                    if refs:
                        pair.referring_paragraphs = self._merge_unique_texts(
                            pair.referring_paragraphs,
                            refs,
                        )

        # Score and classify
        for pair in pairs:
            pair.quality_score = self._compute_quality(pair)
            pair.figure_type = self._classify_figure(pair)

        return pairs

    def _collect_markdown_image_paths(
        self,
        blocks: List[Dict[str, Any]],
        images_dir: Path
    ) -> List[str]:
        """Collect resolved image paths from markdown in reading order."""
        paths: List[str] = []
        for block in blocks:
            if block.get("type") != "image":
                continue
            img_filename = Path(block.get("content", "")).name
            img_path = images_dir / img_filename
            if not img_path.exists():
                img_path = self._resolve_image_path(images_dir, block.get("content", ""))
            if img_path and img_path.exists():
                paths.append(str(img_path))
        return paths

    def _find_markdown(self, doc_dir: Path, doc_id: str) -> Optional[Path]:
        """Find the main MinerU markdown file for a document."""
        # Pattern: {doc_id}/{doc_id}/hybrid_auto/{doc_id}.md
        candidates = [
            doc_dir / doc_id / "hybrid_auto" / f"{doc_id}.md",
            doc_dir / doc_id / "auto" / f"{doc_id}.md",
        ]
        # Also search recursively
        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Fallback: find any .md file that's not formulas.md
        md_files = list(doc_dir.rglob("*.md"))
        md_files = [f for f in md_files if f.name != "formulas.md"]
        if md_files:
            return md_files[0]

        return None

    def _find_content_list(self, doc_dir: Path, doc_id: str) -> Optional[Path]:
        """Find preferred structured content list JSON for a document."""
        candidates = [
            doc_dir / doc_id / "hybrid_auto" / f"{doc_id}_content_list.json",
            doc_dir / doc_id / "hybrid_auto" / "content_list.json",
            doc_dir / doc_id / "auto" / f"{doc_id}_content_list.json",
            doc_dir / doc_id / "auto" / "content_list.json",
            doc_dir / doc_id / "hybrid_auto" / f"{doc_id}_content_list_v2.json",
            doc_dir / doc_id / "hybrid_auto" / "content_list_v2.json",
            doc_dir / doc_id / "auto" / f"{doc_id}_content_list_v2.json",
            doc_dir / doc_id / "auto" / "content_list_v2.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        fallback = list(doc_dir.rglob("*content_list*.json"))
        if fallback:
            # Prefer non-v2 when both exist.
            fallback.sort(key=lambda p: ("v2" in p.name, len(str(p))))
            return fallback[0]
        return None

    def _extract_rich_text(self, content: Any) -> str:
        """Extract plain text from nested MinerU JSON objects."""
        if content is None:
            return ""
        if isinstance(content, str):
            return re.sub(r"\s+", " ", content).strip()
        if isinstance(content, list):
            parts = [self._extract_rich_text(item) for item in content]
            return " ".join(p for p in parts if p).strip()
        if isinstance(content, dict):
            keys = [
                "text", "content",
                "paragraph_content", "title_content",
                "image_caption", "table_caption",
                "page_header_content", "page_footnote_content",
                "page_aside_text_content", "list_content",
            ]
            parts = []
            for key in keys:
                if key in content:
                    text = self._extract_rich_text(content.get(key))
                    if text:
                        parts.append(text)
            if not parts:
                for value in content.values():
                    if isinstance(value, (str, list, dict)):
                        text = self._extract_rich_text(value)
                        if text:
                            parts.append(text)
            return " ".join(parts).strip()
        return ""

    def _resolve_json_image_path(self, base_dir: Path, ref: Optional[str]) -> Optional[Path]:
        """Resolve image path from content_list entries."""
        if not ref:
            return None
        path = Path(ref)
        if path.is_absolute() and path.exists():
            return path
        candidate = base_dir / ref
        if candidate.exists():
            return candidate
        by_name = base_dir / "images" / path.name
        if by_name.exists():
            return by_name
        return None

    def _flatten_content_list_records(self, json_path: Path) -> List[Dict[str, Any]]:
        """
        Flatten content_list(_v2).json into ordered records.

        Record schema:
          - kind: text|heading|image
          - text: str (for text/heading)
          - image_path: str (for image)
          - caption: str (for image/table)
          - source_type: original MinerU type
        """
        try:
            data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return []

        records: List[Dict[str, Any]] = []
        base_dir = json_path.parent

        def append_text(kind: str, text: str, source_type: str, page_idx: int) -> None:
            clean = re.sub(r"\s+", " ", text).strip()
            if clean:
                records.append({
                    "kind": kind,
                    "text": clean,
                    "source_type": source_type,
                    "page_idx": page_idx,
                })

        def append_image(item: Dict[str, Any], source_type: str, page_idx: int) -> None:
            raw_content = item.get("content", {})
            if not isinstance(raw_content, dict):
                raw_content = {}
            image_source = raw_content.get("image_source", {}) if isinstance(raw_content.get("image_source"), dict) else {}
            image_ref = item.get("img_path") or image_source.get("path") or item.get("image_path")
            resolved = self._resolve_json_image_path(base_dir, image_ref)

            if source_type == "table":
                cap_raw = item.get("table_caption")
                if cap_raw is None:
                    cap_raw = raw_content.get("table_caption")
            else:
                cap_raw = item.get("image_caption")
                if cap_raw is None:
                    cap_raw = raw_content.get("image_caption")
            caption = self._extract_rich_text(cap_raw)

            records.append({
                "kind": "image",
                "image_path": str(resolved) if resolved else "",
                "image_filename": resolved.name if resolved else Path(image_ref or "").name,
                "image_ref": image_ref or "",
                "caption": caption,
                "source_type": source_type,
                "page_idx": page_idx,
            })

        def process_item(item: Dict[str, Any], page_idx: int) -> None:
            item_type = str(item.get("type", "")).lower()
            source_type = item_type
            if item_type in {"image", "figure", "table"}:
                image_type = "table" if item_type == "table" else "image"
                append_image(item, image_type, page_idx)
                return

            if item_type in {
                "title", "header", "page_header", "section_title"
            }:
                text = item.get("text") or self._extract_rich_text(item.get("content"))
                append_text("heading", text, source_type, page_idx)
                return

            if item_type in {
                "text", "paragraph", "list", "page_footnote", "page_aside_text", "aside_text"
            }:
                text = item.get("text") or self._extract_rich_text(item.get("content"))
                append_text("text", text, source_type, page_idx)
                return

            text = item.get("text") or self._extract_rich_text(item.get("content"))
            if len(text.strip()) >= 30:
                append_text("text", text, source_type, page_idx)

        if isinstance(data, list):
            # v2 format: list[page] where page is list[block]
            if data and isinstance(data[0], list):
                for page_idx, page in enumerate(data):
                    for item in page:
                        if isinstance(item, dict):
                            process_item(item, page_idx)
            else:
                # flat format: list[block] with page_idx
                for idx, item in enumerate(data):
                    if not isinstance(item, dict):
                        continue
                    page_idx = int(item.get("page_idx", idx))
                    process_item(item, page_idx)
        return records

    def _infer_caption_from_neighbors(self, records: List[Dict[str, Any]], idx: int) -> str:
        """Infer caption text from nearby text/heading blocks when caption is missing."""
        def pick_candidate(text: str) -> Optional[str]:
            clean = re.sub(r"\s+", " ", text).strip()
            if not clean:
                return None
            if FIG_CAPTION_PATTERN.search(clean) or TABLE_CAPTION_PATTERN.search(clean):
                return clean[:220]
            if len(clean) <= 220:
                return clean
            first_sent = re.split(r"(?<=[.!?])\s+", clean)[0]
            return first_sent[:220] if len(first_sent) >= 20 else clean[:220]

        for j in range(idx + 1, min(idx + 5, len(records))):
            rec = records[j]
            if rec.get("kind") == "image":
                break
            if rec.get("kind") in {"text", "heading"}:
                candidate = pick_candidate(rec.get("text", ""))
                if candidate:
                    return candidate

        for j in range(idx - 1, max(-1, idx - 5), -1):
            rec = records[j]
            if rec.get("kind") == "image":
                break
            if rec.get("kind") in {"text", "heading"}:
                candidate = pick_candidate(rec.get("text", ""))
                if candidate:
                    return candidate
        return ""

    def _get_context_from_records(
        self,
        records: List[Dict[str, Any]],
        idx: int,
        direction: str = "before"
    ) -> str:
        """Get context from flattened content_list records."""
        context_parts: List[str] = []
        collected = 0
        if direction == "before":
            search_range = range(idx - 1, -1, -1)
        else:
            search_range = range(idx + 1, len(records))

        for j in search_range:
            if collected >= self.context_window:
                break
            rec = records[j]
            kind = rec.get("kind")
            if kind == "image":
                break
            text = rec.get("text", "").strip()
            if kind == "heading":
                if text:
                    context_parts.append(f"[Section: {text}]")
                    collected += 1
                continue
            if kind == "text":
                if len(text) < 20:
                    continue
                if FIG_CAPTION_PATTERN.match(text) or TABLE_CAPTION_PATTERN.match(text):
                    continue
                context_parts.append(text)
                collected += 1

        if direction == "before":
            context_parts.reverse()
        return "\n\n".join(context_parts)

    def _find_referring_paragraphs_from_records(
        self,
        records: List[Dict[str, Any]],
        figure_number: int
    ) -> List[str]:
        """Find paragraphs that reference a figure/table number from flattened records."""
        refs: List[str] = []
        patterns = [
            re.compile(rf"\bFig(?:ure|\.)\s*{figure_number}\b", re.IGNORECASE),
            re.compile(rf"\bTable\s*{figure_number}\b", re.IGNORECASE),
        ]
        for rec in records:
            if rec.get("kind") != "text":
                continue
            text = rec.get("text", "")
            if any(p.search(text) for p in patterns):
                refs.append(text[:500])
        return refs

    def _merge_unique_texts(self, old: List[str], new: List[str]) -> List[str]:
        """Merge text lists while preserving order."""
        merged: List[str] = []
        seen = set()
        for item in old + new:
            key = re.sub(r"\s+", " ", item).strip()
            if key and key not in seen:
                seen.add(key)
                merged.append(item)
        return merged

    def _build_associations_from_content_list(
        self,
        doc_dir: Path,
        doc_id: str,
        md_image_paths: Optional[List[str]] = None,
    ) -> List[FigureTextPair]:
        """Build figure-text associations from structured content_list JSON."""
        content_list_path = self._find_content_list(doc_dir, doc_id)
        if not content_list_path:
            return []

        records = self._flatten_content_list_records(content_list_path)
        if not records:
            return []

        pairs: List[FigureTextPair] = []
        fig_counter = 0
        md_pos = 0
        md_image_paths = md_image_paths or []
        for idx, rec in enumerate(records):
            if rec.get("kind") != "image":
                continue

            md_fallback_path = ""
            if not rec.get("image_path") and md_pos < len(md_image_paths):
                md_fallback_path = md_image_paths[md_pos]
                md_pos += 1

            image_path = rec.get("image_path") or md_fallback_path
            if not image_path:
                continue
            img = Path(image_path)
            if not img.exists():
                continue

            caption = rec.get("caption", "").strip()
            caption_source = "content_list"
            if not caption:
                caption = self._infer_caption_from_neighbors(records, idx)
                caption_source = "neighbor_text" if caption else "none"

            context_before = self._get_context_from_records(records, idx, "before")
            context_after = self._get_context_from_records(records, idx, "after")
            if not caption and (context_before or context_after):
                # Last-resort fallback keeps sample usable while marking weak anchor.
                seed = context_before or context_after
                caption = re.split(r"(?<=[.!?])\s+", seed)[0][:220]
                caption_source = "context_fallback" if caption else "none"

            figure_number = None
            if caption:
                match = FIG_CAPTION_PATTERN.search(caption) or TABLE_CAPTION_PATTERN.search(caption)
                if match:
                    figure_number = int(match.group(1))

            refs: List[str] = []
            if figure_number is not None:
                refs = self._find_referring_paragraphs_from_records(records, figure_number)

            anchor_strength = "strong"
            if caption_source == "neighbor_text":
                anchor_strength = "medium"
            elif caption_source in {"context_fallback", "none"}:
                anchor_strength = "weak"

            fig_counter += 1
            pair = FigureTextPair(
                doc_id=doc_id,
                figure_id=f"{doc_id}_fig_{fig_counter}",
                figure_number=figure_number,
                image_path=str(img),
                image_filename=img.name,
                caption=caption,
                context_before=context_before,
                context_after=context_after,
                referring_paragraphs=refs,
                metadata={
                    "source": "content_list",
                    "source_file": content_list_path.name,
                    "source_type": rec.get("source_type", "image"),
                    "caption_source": caption_source,
                    "anchor_strength": anchor_strength,
                    "image_source": "content_list" if rec.get("image_path") else "markdown_fallback",
                },
            )
            pairs.append(pair)

        return pairs

    def _parse_markdown_blocks(self, content: str) -> List[Dict]:
        """
        Parse markdown content into a sequence of typed blocks.

        Returns list of dicts with keys: type, content, line_idx
        Types: "text", "image", "heading", "formula"
        """
        lines = content.split('\n')
        blocks = []
        current_text = []
        current_start = 0

        def flush_text():
            nonlocal current_text, current_start
            text = '\n'.join(current_text).strip()
            if text:
                blocks.append({
                    "type": "text",
                    "content": text,
                    "line_idx": current_start
                })
            current_text = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for inline image
            img_match = IMG_PATTERN.search(line)
            if img_match:
                flush_text()
                alt_text = img_match.group(1)
                img_src = img_match.group(2)

                # Collect sub-figure captions that follow
                sub_captions = []
                j = i + 1
                while j < len(lines):
                    stripped = lines[j].strip()
                    if not stripped:
                        j += 1
                        continue
                    # Check for another image immediately after (multi-panel figure)
                    if IMG_PATTERN.search(stripped):
                        break
                    # Check for sub-figure label like "(a) Description"
                    sub_match = SUBFIG_PATTERN.match(stripped)
                    if sub_match:
                        sub_captions.append(stripped)
                        j += 1
                        continue
                    # Check for figure caption
                    if FIG_CAPTION_PATTERN.match(stripped) or TABLE_CAPTION_PATTERN.match(stripped):
                        # This is the main caption, will be captured separately
                        break
                    break

                blocks.append({
                    "type": "image",
                    "content": img_src,
                    "alt": alt_text,
                    "sub_captions": sub_captions,
                    "line_idx": i
                })
                i = j
                current_start = i
                continue

            # Check for heading
            if line.startswith('#'):
                flush_text()
                blocks.append({
                    "type": "heading",
                    "content": line.lstrip('#').strip(),
                    "level": len(line) - len(line.lstrip('#')),
                    "line_idx": i
                })
                i += 1
                current_start = i
                continue

            # Check for display formula ($$...$$)
            if line.strip().startswith('$$'):
                flush_text()
                formula_lines = [line]
                if not line.strip().endswith('$$') or line.strip() == '$$':
                    j = i + 1
                    while j < len(lines):
                        formula_lines.append(lines[j])
                        if lines[j].strip().endswith('$$'):
                            break
                        j += 1
                    i = j
                blocks.append({
                    "type": "formula",
                    "content": '\n'.join(formula_lines),
                    "line_idx": current_start
                })
                i += 1
                current_start = i
                continue

            # Regular text line
            if not current_text:
                current_start = i
            current_text.append(line)
            i += 1

        flush_text()
        return blocks

    def _build_associations(
        self,
        blocks: List[Dict],
        doc_id: str,
        images_dir: Path
    ) -> List[FigureTextPair]:
        """Build figure-text associations from parsed blocks."""
        pairs = []
        fig_counter = 0

        for idx, block in enumerate(blocks):
            if block["type"] != "image":
                continue

            img_filename = Path(block["content"]).name
            img_path = images_dir / img_filename

            if not img_path.exists():
                # Try finding by pattern
                img_path = self._resolve_image_path(images_dir, block["content"])
                if not img_path:
                    continue

            # Extract caption from the block immediately after the image
            caption = ""
            figure_number = None
            caption_block_idx = self._find_caption_block(blocks, idx)
            if caption_block_idx is not None:
                cap_text = blocks[caption_block_idx]["content"]
                cap_match = FIG_CAPTION_PATTERN.search(cap_text)
                if cap_match:
                    figure_number = int(cap_match.group(1))
                    caption = cap_match.group(0)
                else:
                    tab_match = TABLE_CAPTION_PATTERN.search(cap_text)
                    if tab_match:
                        figure_number = int(tab_match.group(1))
                        caption = tab_match.group(0)

            # Get context before and after
            context_before = self._get_context(blocks, idx, direction="before")
            context_after = self._get_context(blocks, idx, direction="after")

            # Collect sub-figure captions
            sub_figs = block.get("sub_captions", [])

            fig_counter += 1
            pair = FigureTextPair(
                doc_id=doc_id,
                figure_id=f"{doc_id}_fig_{fig_counter}",
                figure_number=figure_number,
                image_path=str(img_path),
                image_filename=img_filename,
                caption=caption,
                context_before=context_before,
                context_after=context_after,
                referring_paragraphs=[],
                sub_figures=sub_figs,
            )
            pairs.append(pair)

        # Group consecutive images into multi-panel figures
        pairs = self._merge_subfigures(pairs)

        return pairs

    def _find_caption_block(self, blocks: List[Dict], img_idx: int) -> Optional[int]:
        """Find the caption block for an image (usually the next text block)."""
        # Look at the next few blocks for a caption
        for j in range(img_idx + 1, min(img_idx + 4, len(blocks))):
            if blocks[j]["type"] == "image":
                # Hit another image before finding caption — might be multi-panel
                continue
            if blocks[j]["type"] == "text":
                text = blocks[j]["content"]
                if FIG_CAPTION_PATTERN.search(text) or TABLE_CAPTION_PATTERN.search(text):
                    return j
                # If the text block is very short, it might be a caption without "Figure N:"
                if len(text) < 100:
                    return j
                break
        return None

    def _get_context(
        self,
        blocks: List[Dict],
        img_idx: int,
        direction: str = "before"
    ) -> str:
        """Get surrounding text context for a figure."""
        context_parts = []
        collected = 0

        if direction == "before":
            search_range = range(img_idx - 1, -1, -1)
        else:
            search_range = range(img_idx + 1, len(blocks))

        for j in search_range:
            if collected >= self.context_window:
                break
            block = blocks[j]
            if block["type"] == "text":
                text = block["content"].strip()
                # Skip very short lines (sub-captions, labels)
                if len(text) < 20:
                    continue
                # Skip if it's a figure caption (already captured)
                if FIG_CAPTION_PATTERN.match(text) or TABLE_CAPTION_PATTERN.match(text):
                    continue
                context_parts.append(text)
                collected += 1
            elif block["type"] == "image":
                # Stop at another figure
                break
            elif block["type"] == "heading":
                # Include heading for section context
                context_parts.append(f"[Section: {block['content']}]")
                collected += 1

        if direction == "before":
            context_parts.reverse()
        return "\n\n".join(context_parts)

    def _find_referring_paragraphs(
        self,
        all_paragraphs: List[Dict],
        figure_number: int
    ) -> List[str]:
        """Find all paragraphs that reference a given figure number."""
        refs = []
        patterns = [
            re.compile(rf'\bFig(?:ure|\.)\s*{figure_number}\b', re.IGNORECASE),
            re.compile(rf'\bTable\s*{figure_number}\b', re.IGNORECASE),
        ]
        for para in all_paragraphs:
            text = para["content"]
            for pat in patterns:
                if pat.search(text):
                    # Truncate very long paragraphs
                    if len(text) > 500:
                        # Find the sentence containing the reference
                        for sent in re.split(r'(?<=[.!?])\s+', text):
                            if pat.search(sent):
                                refs.append(sent)
                                break
                    else:
                        refs.append(text)
                    break
        return refs

    def _merge_subfigures(self, pairs: List[FigureTextPair]) -> List[FigureTextPair]:
        """
        Merge consecutive images that belong to the same logical figure.

        E.g., Figure 1 with (a), (b), (c) sub-panels → single FigureTextPair
        with sub_figures list.
        """
        if not pairs:
            return pairs

        merged = []
        i = 0
        while i < len(pairs):
            current = pairs[i]

            # Look ahead for consecutive images with same figure_number or no caption
            group = [current]
            j = i + 1
            while j < len(pairs):
                nxt = pairs[j]
                # Same logical figure number
                if (current.figure_number is not None
                        and nxt.figure_number == current.figure_number):
                    group.append(nxt)
                    j += 1
                # No caption and right after — likely sub-figure
                elif nxt.figure_number is None and not nxt.caption:
                    group.append(nxt)
                    j += 1
                else:
                    break

            if len(group) == 1:
                merged.append(current)
            else:
                # Merge: keep the first one, aggregate sub-figures
                primary = group[0]
                for sub in group[1:]:
                    primary.sub_figures.extend(sub.sub_figures)
                    if sub.image_filename:
                        primary.sub_figures.append(sub.image_filename)
                    # Keep the richest context
                    if len(sub.context_after) > len(primary.context_after):
                        primary.context_after = sub.context_after
                primary.metadata["sub_image_paths"] = [
                    g.image_path for g in group
                ]
                primary.metadata["panel_count"] = len(group)
                merged.append(primary)

            i = j

        return merged

    def _resolve_image_path(self, images_dir: Path, ref_path: str) -> Optional[Path]:
        """Try to resolve an image path reference."""
        # Direct path
        direct = images_dir / Path(ref_path).name
        if direct.exists():
            return direct

        # Try without subdirectory prefix
        name = Path(ref_path).name
        candidates = list(images_dir.glob(f"*{name}*"))
        if candidates:
            return candidates[0]

        return None

    def _classify_figure(self, pair: FigureTextPair) -> FigureType:
        """Classify figure type based on caption and context."""
        text = f"{pair.caption} {pair.context_before} {pair.context_after}".lower()

        architecture_keywords = [
            "architecture", "framework", "pipeline", "overview", "model",
            "network", "structure", "system", "module", "block diagram"
        ]
        plot_keywords = [
            "plot", "curve", "graph", "accuracy", "loss", "performance",
            "comparison", "result", "f1", "precision", "recall", "ablation",
            "training", "convergence", "epoch", "vs.", "versus"
        ]
        table_keywords = [
            "table", "comparison table", "results table", "dataset statistics"
        ]
        diagram_keywords = [
            "flowchart", "flow chart", "process", "workflow", "procedure",
            "algorithm", "step", "pipeline"
        ]
        example_keywords = [
            "example", "sample", "illustration", "case study", "instance",
            "demonstration", "visualization", "qualitative"
        ]

        scores = {
            FigureType.ARCHITECTURE: sum(1 for kw in architecture_keywords if kw in text),
            FigureType.PLOT: sum(1 for kw in plot_keywords if kw in text),
            FigureType.TABLE_IMG: sum(1 for kw in table_keywords if kw in text),
            FigureType.DIAGRAM: sum(1 for kw in diagram_keywords if kw in text),
            FigureType.EXAMPLE: sum(1 for kw in example_keywords if kw in text),
        }

        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
        return FigureType.OTHER

    def _compute_quality(self, pair: FigureTextPair) -> float:
        """Compute quality score for a figure-text pair."""
        score = 0.0

        # Has caption
        if pair.caption:
            score += 0.25

        # Has context
        if pair.context_before:
            score += 0.15
        if pair.context_after:
            score += 0.15

        # Has referring paragraphs (figure is actually discussed in text)
        if pair.referring_paragraphs:
            score += min(0.25, len(pair.referring_paragraphs) * 0.1)

        # Image file exists and is valid
        img = Path(pair.image_path)
        if img.exists():
            score += 0.1
            # Prefer larger images
            size = img.stat().st_size
            if size > 10000:  # > 10KB
                score += 0.1

        # Penalize weak caption fallback to reduce weak-anchor sampling.
        if pair.metadata.get("caption_source") in {"context_fallback", "none"}:
            score -= 0.1

        return min(1.0, score)

    def get_stats(self, results: Dict[str, List[FigureTextPair]]) -> Dict:
        """Generate statistics about the associations."""
        total_pairs = sum(len(pairs) for pairs in results.values())
        docs_with_figures = len(results)

        type_counts = {}
        quality_scores = []
        with_caption = 0
        with_refs = 0
        multi_panel = 0
        caption_source_counts: Dict[str, int] = {}
        weak_anchor = 0

        for pairs in results.values():
            for pair in pairs:
                ft = pair.figure_type.value
                type_counts[ft] = type_counts.get(ft, 0) + 1
                quality_scores.append(pair.quality_score)
                if pair.caption:
                    with_caption += 1
                if pair.referring_paragraphs:
                    with_refs += 1
                if pair.metadata.get("panel_count", 1) > 1:
                    multi_panel += 1
                source = pair.metadata.get("caption_source", "unknown")
                caption_source_counts[source] = caption_source_counts.get(source, 0) + 1
                if pair.metadata.get("anchor_strength") == "weak":
                    weak_anchor += 1

        return {
            "total_documents": docs_with_figures,
            "total_figure_text_pairs": total_pairs,
            "avg_pairs_per_doc": total_pairs / max(1, docs_with_figures),
            "type_distribution": type_counts,
            "avg_quality_score": sum(quality_scores) / max(1, len(quality_scores)),
            "with_caption": with_caption,
            "with_referring_paragraphs": with_refs,
            "multi_panel_figures": multi_panel,
            "caption_source_distribution": caption_source_counts,
            "weak_anchor_pairs": weak_anchor,
        }
