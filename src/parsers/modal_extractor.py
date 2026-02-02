"""
Modal Extractor and Classifier

Extracts and classifies elements by modality (table, figure, formula, infographic, text)
with quality filtering and enhancement.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from PIL import Image
import hashlib


class ModalityType(Enum):
    """Supported modality types."""
    TABLE = "table"
    FIGURE = "figure"
    FORMULA = "formula"
    INFOGRAPHIC = "infographic"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class Passage:
    """
    A passage represents a single unit for contrastive learning.

    Contains the content, modality information, and metadata needed
    for query generation and retrieval.
    """
    passage_id: str
    doc_id: str
    page_idx: int
    modal_type: ModalityType
    content: str
    image_path: Optional[str] = None
    bbox: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0
    context: Optional[str] = None  # Surrounding text for context

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "passage_id": self.passage_id,
            "doc_id": self.doc_id,
            "page_idx": self.page_idx,
            "modal_type": self.modal_type.value,
            "content": self.content,
            "image_path": self.image_path,
            "bbox": self.bbox,
            "metadata": self.metadata,
            "quality_score": self.quality_score,
            "context": self.context
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Passage":
        """Create from dictionary."""
        return cls(
            passage_id=data["passage_id"],
            doc_id=data["doc_id"],
            page_idx=data.get("page_idx", 0),
            modal_type=ModalityType(data["modal_type"]),
            content=data["content"],
            image_path=data.get("image_path"),
            bbox=data.get("bbox"),
            metadata=data.get("metadata", {}),
            quality_score=data.get("quality_score", 1.0),
            context=data.get("context")
        )


class ModalExtractor:
    """
    Extracts and classifies document elements into passages by modality.

    Applies quality filtering and content enhancement for optimal
    contrastive learning data.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize extractor with configuration.

        Args:
            config: Modal type configurations from config.yaml
        """
        self.config = config or {}

        # Default thresholds
        self.table_config = self.config.get("table", {
            "enabled": True,
            "min_rows": 3,
            "min_cols": 2
        })
        self.figure_config = self.config.get("figure", {
            "enabled": True,
            "min_width": 100,
            "min_height": 100
        })
        self.formula_config = self.config.get("formula", {
            "enabled": True,
            "include_inline": False
        })
        self.text_config = self.config.get("text", {
            "enabled": True,
            "min_length": 50,
            "max_length": 2000
        })

    def extract_passages(
        self,
        elements: List[Any],
        doc_id: str,
        include_context: bool = True
    ) -> List[Passage]:
        """
        Extract passages from parsed elements.

        Args:
            elements: List of ParsedElement objects
            doc_id: Document identifier
            include_context: Whether to include surrounding context

        Returns:
            List of Passage objects
        """
        passages = []

        # Sort elements by page and position
        sorted_elements = sorted(
            elements,
            key=lambda e: (e.page_idx, e.bbox[1] if e.bbox else 0)
        )

        for idx, elem in enumerate(sorted_elements):
            # Classify and validate element
            modal_type = self._classify_modality(elem)
            if modal_type == ModalityType.UNKNOWN:
                continue

            # Apply quality filters
            if not self._validate_element(elem, modal_type):
                continue

            # Build context from surrounding elements
            context = None
            if include_context:
                context = self._build_context(sorted_elements, idx)

            # Generate passage ID
            content_hash = hashlib.md5(
                elem.content[:100].encode('utf-8')
            ).hexdigest()[:8]
            passage_id = f"{doc_id}_{modal_type.value}_{elem.page_idx}_{content_hash}"

            # Create passage
            passage = Passage(
                passage_id=passage_id,
                doc_id=doc_id,
                page_idx=elem.page_idx,
                modal_type=modal_type,
                content=self._clean_content(elem.content, modal_type),
                image_path=elem.image_path,
                bbox=elem.bbox,
                metadata=self._extract_metadata(elem, modal_type),
                quality_score=self._compute_quality_score(elem, modal_type),
                context=context
            )

            passages.append(passage)

        return passages

    def _classify_modality(self, element: Any) -> ModalityType:
        """Classify element into modality type."""
        elem_type = element.element_type.lower()

        if elem_type == "table":
            return ModalityType.TABLE
        elif elem_type in ["figure", "image"]:
            # Distinguish between regular figures and infographics
            if self._is_infographic(element):
                return ModalityType.INFOGRAPHIC
            return ModalityType.FIGURE
        elif elem_type in ["formula", "equation"]:
            return ModalityType.FORMULA
        elif elem_type == "text":
            return ModalityType.TEXT
        else:
            return ModalityType.UNKNOWN

    def _is_infographic(self, element: Any) -> bool:
        """
        Detect if a figure is an infographic (diagram, flowchart, etc.).

        Heuristics:
        - Contains multiple text regions
        - Has geometric shapes
        - Contains arrows or connectors
        """
        content = element.content.lower() if element.content else ""
        metadata = element.metadata or {}

        # Keywords suggesting infographic
        infographic_keywords = [
            "diagram", "flowchart", "architecture", "pipeline",
            "workflow", "process", "framework", "overview",
            "system", "structure"
        ]

        for keyword in infographic_keywords:
            if keyword in content:
                return True

        # Check caption if available
        caption = metadata.get("caption", "").lower()
        for keyword in infographic_keywords:
            if keyword in caption:
                return True

        return False

    def _validate_element(self, element: Any, modal_type: ModalityType) -> bool:
        """Validate element meets quality thresholds."""

        if modal_type == ModalityType.TABLE:
            if not self.table_config.get("enabled", True):
                return False
            # Check row count
            content = element.content
            rows = content.count('\n')
            if rows < self.table_config.get("min_rows", 3):
                return False
            # Check for actual table structure
            if '|' not in content:
                return False

        elif modal_type in [ModalityType.FIGURE, ModalityType.INFOGRAPHIC]:
            if not self.figure_config.get("enabled", True):
                return False
            # Check image dimensions if available
            if element.bbox:
                width = element.bbox[2] - element.bbox[0]
                height = element.bbox[3] - element.bbox[1]
                min_w = self.figure_config.get("min_width", 100)
                min_h = self.figure_config.get("min_height", 100)
                if width < min_w or height < min_h:
                    return False
            # Verify image file exists
            if element.image_path:
                img_path = Path(element.image_path)
                if img_path.exists():
                    try:
                        with Image.open(img_path) as img:
                            w, h = img.size
                            if w < 50 or h < 50:
                                return False
                    except Exception:
                        pass

        elif modal_type == ModalityType.FORMULA:
            if not self.formula_config.get("enabled", True):
                return False
            content = element.content.strip()
            # Skip very short formulas (likely inline)
            if not self.formula_config.get("include_inline", False):
                if len(content) < 10 or not any(c in content for c in ['\\', '$', '=']):
                    return False

        elif modal_type == ModalityType.TEXT:
            if not self.text_config.get("enabled", True):
                return False
            content = element.content.strip()
            min_len = self.text_config.get("min_length", 50)
            max_len = self.text_config.get("max_length", 2000)
            if len(content) < min_len or len(content) > max_len:
                return False
            # Skip boilerplate text
            if self._is_boilerplate(content):
                return False

        return True

    def _is_boilerplate(self, text: str) -> bool:
        """Check if text is boilerplate (headers, footers, etc.)."""
        text_lower = text.lower().strip()

        boilerplate_patterns = [
            r'^page\s+\d+',
            r'^\d+\s*$',
            r'^(abstract|introduction|conclusion|references|acknowledgment)s?\s*$',
            r'^figure\s+\d+',
            r'^table\s+\d+',
            r'©\s*\d{4}',
            r'all rights reserved',
            r'^arxiv:\d+\.\d+',
        ]

        for pattern in boilerplate_patterns:
            if re.match(pattern, text_lower):
                return True

        return False

    def _clean_content(self, content: str, modal_type: ModalityType) -> str:
        """Clean and normalize content based on modality."""
        if not content:
            return ""

        content = content.strip()

        if modal_type == ModalityType.TABLE:
            # Normalize table formatting
            lines = content.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line:
                    cleaned_lines.append(line)
            content = '\n'.join(cleaned_lines)

        elif modal_type == ModalityType.FORMULA:
            # Ensure LaTeX delimiters are present
            if not content.startswith('$') and not content.startswith('\\['):
                if '\n' in content or len(content) > 50:
                    content = f"$$\n{content}\n$$"
                else:
                    content = f"${content}$"

        elif modal_type == ModalityType.TEXT:
            # Clean excessive whitespace
            content = re.sub(r'\s+', ' ', content)
            # Remove control characters
            content = ''.join(c for c in content if c.isprintable() or c in '\n\t')

        return content

    def _extract_metadata(
        self,
        element: Any,
        modal_type: ModalityType
    ) -> Dict[str, Any]:
        """Extract relevant metadata for the element."""
        metadata = dict(element.metadata) if element.metadata else {}

        metadata["original_type"] = element.element_type

        if modal_type == ModalityType.TABLE:
            # Count rows and columns
            content = element.content
            lines = [l for l in content.split('\n') if l.strip() and '|' in l]
            metadata["rows"] = len(lines)
            if lines:
                # Count columns from first data row
                cols = len([c for c in lines[0].split('|') if c.strip()])
                metadata["cols"] = cols

        elif modal_type == ModalityType.FIGURE:
            # Add image dimensions if path exists
            if element.image_path:
                img_path = Path(element.image_path)
                if img_path.exists():
                    try:
                        with Image.open(img_path) as img:
                            metadata["width"], metadata["height"] = img.size
                            metadata["format"] = img.format
                    except Exception:
                        pass

        elif modal_type == ModalityType.FORMULA:
            # Check for specific formula types
            content = element.content.lower()
            if 'sum' in content or '\\sum' in content:
                metadata["formula_type"] = "summation"
            elif 'int' in content or '\\int' in content:
                metadata["formula_type"] = "integral"
            elif 'frac' in content or '\\frac' in content:
                metadata["formula_type"] = "fraction"
            elif 'matrix' in content or '\\begin{' in content:
                metadata["formula_type"] = "matrix"

        return metadata

    def _compute_quality_score(
        self,
        element: Any,
        modal_type: ModalityType
    ) -> float:
        """
        Compute quality score for the element (0.0 to 1.0).

        Higher scores indicate better quality passages.
        """
        score = 0.5  # Base score

        content = element.content or ""

        if modal_type == ModalityType.TABLE:
            # Prefer larger tables
            rows = content.count('\n')
            if rows >= 5:
                score += 0.2
            elif rows >= 3:
                score += 0.1
            # Prefer tables with numeric data
            if re.search(r'\d+\.?\d*', content):
                score += 0.1
            # Check for header row
            if '---' in content or '===' in content:
                score += 0.1

        elif modal_type == ModalityType.FIGURE:
            # Prefer figures with captions
            if element.metadata and element.metadata.get("caption"):
                score += 0.2
            # Check image path exists
            if element.image_path and Path(element.image_path).exists():
                score += 0.2
            # Prefer larger images
            if element.bbox:
                area = (element.bbox[2] - element.bbox[0]) * (element.bbox[3] - element.bbox[1])
                if area > 50000:
                    score += 0.1

        elif modal_type == ModalityType.FORMULA:
            # Prefer complex formulas
            if len(content) > 30:
                score += 0.2
            # Prefer formulas with multiple terms
            if content.count('+') + content.count('-') + content.count('=') > 2:
                score += 0.1
            # Check for proper LaTeX
            if '\\' in content:
                score += 0.1

        elif modal_type == ModalityType.TEXT:
            # Prefer moderate length text
            length = len(content)
            if 100 <= length <= 500:
                score += 0.2
            elif 500 < length <= 1000:
                score += 0.1
            # Prefer text with technical content
            tech_keywords = ['method', 'result', 'experiment', 'propose', 'show']
            for kw in tech_keywords:
                if kw in content.lower():
                    score += 0.05
                    break

        return min(1.0, max(0.0, score))

    def _build_context(
        self,
        elements: List[Any],
        current_idx: int,
        window: int = 2
    ) -> str:
        """Build context string from surrounding elements."""
        context_parts = []

        # Get surrounding text elements
        start_idx = max(0, current_idx - window)
        end_idx = min(len(elements), current_idx + window + 1)

        for idx in range(start_idx, end_idx):
            if idx == current_idx:
                continue
            elem = elements[idx]
            if elem.element_type == "text" and elem.content:
                # Truncate long text
                text = elem.content[:200]
                if len(elem.content) > 200:
                    text += "..."
                context_parts.append(text)

        return " ".join(context_parts) if context_parts else None

    def get_modal_distribution(self, passages: List[Passage]) -> Dict[str, int]:
        """Get distribution of modalities in passages."""
        distribution = {}
        for passage in passages:
            modal_type = passage.modal_type.value
            distribution[modal_type] = distribution.get(modal_type, 0) + 1
        return distribution

    def filter_by_modality(
        self,
        passages: List[Passage],
        modal_types: List[ModalityType]
    ) -> List[Passage]:
        """Filter passages by modality types."""
        return [p for p in passages if p.modal_type in modal_types]

    def filter_by_quality(
        self,
        passages: List[Passage],
        min_score: float = 0.5
    ) -> List[Passage]:
        """Filter passages by quality score."""
        return [p for p in passages if p.quality_score >= min_score]
