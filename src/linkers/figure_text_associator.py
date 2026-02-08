"""
Figure-Text Associator

Parses MinerU markdown output to build (figure, caption, context) associations.
Uses the inline image references in .md files which preserve reading order.
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
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

        # Find the markdown file
        md_path = self._find_markdown(doc_dir, doc_id)
        if not md_path:
            return []

        md_content = md_path.read_text(encoding='utf-8', errors='ignore')
        images_dir = md_path.parent / "images"

        # Parse the markdown into blocks
        blocks = self._parse_markdown_blocks(md_content)

        # Find image blocks and build associations
        pairs = self._build_associations(blocks, doc_id, images_dir)

        # Find referring paragraphs for each figure
        all_paragraphs = [b for b in blocks if b["type"] == "text"]
        for pair in pairs:
            if pair.figure_number is not None:
                pair.referring_paragraphs = self._find_referring_paragraphs(
                    all_paragraphs, pair.figure_number
                )

        # Score and classify
        for pair in pairs:
            pair.quality_score = self._compute_quality(pair)
            pair.figure_type = self._classify_figure(pair)

        return pairs

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

        return {
            "total_documents": docs_with_figures,
            "total_figure_text_pairs": total_pairs,
            "avg_pairs_per_doc": total_pairs / max(1, docs_with_figures),
            "type_distribution": type_counts,
            "avg_quality_score": sum(quality_scores) / max(1, len(quality_scores)),
            "with_caption": with_caption,
            "with_referring_paragraphs": with_refs,
            "multi_panel_figures": multi_panel,
        }
