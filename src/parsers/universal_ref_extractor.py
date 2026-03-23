#!/usr/bin/env python3
"""Universal Reference Extractor — cross-modal reference detection for non-LaTeX documents.

Extracts figure/table/equation references from MinerU markdown output using
regex patterns, independent of LaTeX source availability. This enables the
document graph pipeline to work with pure PDF inputs.

Design goals:
  - Zero LLM cost (pure regex + heuristics)
  - Graceful degradation: LaTeX source provides highest precision; MinerU markdown
    provides broader coverage with slightly lower precision
  - Drop-in compatible with the same Node/Edge structures used in
    analyze_latex_graph_topology.py

Supported document formats:
  - MinerU markdown (primary): parses Figure/Table/Equation references from
    markdown text produced by MinerU PDF extraction
  - Plain text (fallback): regex extraction from any text

Reference patterns detected:
  - "Figure 3", "Fig. 3", "Fig 3a", "Figures 3-5", "Figs. 3 and 4"
  - "Table 1", "Tab. 1", "Tables 1-3"
  - "Equation 1", "Eq. 1", "Eqs. (1)-(3)", "(1)", "Eq. (1)"
  - "Section 3", "Sec. 3.1", "§3"

Example usage:
    extractor = MineruRefExtractor()
    refs = extractor.extract_refs(markdown_text, doc_id="1908.09635")
    # refs = [RefMention(target_type="figure", target_number=3, ...), ...]

Patent-relevant:
  This module demonstrates generalization beyond LaTeX — a key requirement
  for patent claims about document graph construction methods.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RefMention:
    """A detected reference mention in document text."""
    target_type: str        # "figure" | "table" | "equation" | "section"
    target_number: int      # extracted numeric identifier
    context: str            # surrounding text (±120 chars)
    line_no: int            # line number in source text
    char_offset: int        # character offset in line
    doc_id: str = ""
    confidence: float = 1.0  # 0.0-1.0, lower for ambiguous patterns


@dataclass
class ElementAnchor:
    """A detected element definition (caption/header) in document text."""
    element_type: str       # "figure" | "table" | "equation"
    element_number: int     # extracted numeric identifier
    caption: str            # full caption text
    line_no: int
    doc_id: str = ""


@dataclass
class ExtractedRefGraph:
    """Output of reference extraction: anchors + mentions + edges."""
    doc_id: str
    anchors: List[ElementAnchor]
    mentions: List[RefMention]
    # Edges: (paragraph_line_no, target_type, target_number)
    edges: List[Tuple[int, str, int]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "n_anchors": len(self.anchors),
            "n_mentions": len(self.mentions),
            "n_edges": len(self.edges),
            "anchors": [
                {"type": a.element_type, "number": a.element_number,
                 "caption": a.caption[:200], "line_no": a.line_no}
                for a in self.anchors
            ],
            "mention_summary": {
                "figure": sum(1 for m in self.mentions if m.target_type == "figure"),
                "table": sum(1 for m in self.mentions if m.target_type == "table"),
                "equation": sum(1 for m in self.mentions if m.target_type == "equation"),
                "section": sum(1 for m in self.mentions if m.target_type == "section"),
            },
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseRefExtractor(ABC):
    """Abstract base for reference extraction from any document format."""

    @abstractmethod
    def extract_refs(self, text: str, doc_id: str = "") -> ExtractedRefGraph:
        """Extract all cross-modal references from document text."""
        ...

    @abstractmethod
    def extract_anchors(self, text: str, doc_id: str = "") -> List[ElementAnchor]:
        """Extract element definitions (captions/headers)."""
        ...


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Figure references: "Figure 3", "Fig. 3", "Fig 3a", "Figures 3-5"
_RE_FIG = re.compile(
    r"(?:Fig(?:ure|\.)?s?)\s*\.?\s*(\d+)(?:\s*[a-z])?",
    re.IGNORECASE,
)

# Table references: "Table 1", "Tab. 1", "Tables 1-3"
_RE_TAB = re.compile(
    r"(?:Tab(?:le|\.)?s?)\s*\.?\s*(\d+)",
    re.IGNORECASE,
)

# Equation references: "Equation 1", "Eq. 1", "Eq. (1)", "(1)" in math context
_RE_EQ_EXPLICIT = re.compile(
    r"(?:Eq(?:uation|\.)?s?)\s*\.?\s*\(?(\d+)\)?",
    re.IGNORECASE,
)
# Bare "(1)" — only match when preceded by context suggesting equation reference
_RE_EQ_BARE = re.compile(
    r"(?:in|from|by|see|using|according\s+to|as\s+in)\s+\((\d+)\)",
    re.IGNORECASE,
)

# Section references: "Section 3", "Sec. 3.1", "§3"
_RE_SEC = re.compile(
    r"(?:(?:Sec(?:tion|\.)?)\s*\.?\s*|§\s*)(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

# Element anchors (captions)
_RE_FIG_CAPTION = re.compile(
    r"^(?:!\[.*?\].*?)?(?:Figure|Fig\.?)\s*(\d+)[.:\s](.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_RE_TAB_CAPTION = re.compile(
    r"^(?:Table|Tab\.?)\s*(\d+)[.:\s](.*)$",
    re.IGNORECASE | re.MULTILINE,
)

_REF_PATTERNS = [
    ("figure", _RE_FIG),
    ("table", _RE_TAB),
    ("equation", _RE_EQ_EXPLICIT),
    ("equation", _RE_EQ_BARE),
    ("section", _RE_SEC),
]

_ANCHOR_PATTERNS = [
    ("figure", _RE_FIG_CAPTION),
    ("table", _RE_TAB_CAPTION),
]


# ---------------------------------------------------------------------------
# MinerU Markdown extractor
# ---------------------------------------------------------------------------

class MineruRefExtractor(BaseRefExtractor):
    """Extract cross-modal references from MinerU markdown output.

    MinerU produces markdown with:
      - Figure captions as "Figure N: caption text"
      - Table headers as "Table N: caption text"
      - Inline references preserved as plain text: "as shown in Figure 3"
      - Equations may appear as LaTeX blocks or images

    This extractor identifies all reference mentions and builds edges
    from paragraphs (by line number) to referenced elements.
    """

    def __init__(self, context_chars: int = 120):
        self.context_chars = context_chars

    def extract_anchors(self, text: str, doc_id: str = "") -> List[ElementAnchor]:
        """Extract element captions/definitions."""
        anchors: List[ElementAnchor] = []
        lines = text.split("\n")
        for line_no, line in enumerate(lines, start=1):
            for etype, pattern in _ANCHOR_PATTERNS:
                for m in pattern.finditer(line):
                    try:
                        num = int(m.group(1))
                    except (ValueError, IndexError):
                        continue
                    caption = m.group(2).strip() if m.lastindex >= 2 else ""
                    anchors.append(ElementAnchor(
                        element_type=etype,
                        element_number=num,
                        caption=caption[:500],
                        line_no=line_no,
                        doc_id=doc_id,
                    ))
        return anchors

    def extract_refs(self, text: str, doc_id: str = "") -> ExtractedRefGraph:
        """Extract all reference mentions and build paragraph→element edges."""
        anchors = self.extract_anchors(text, doc_id)
        mentions: List[RefMention] = []
        edges: List[Tuple[int, str, int]] = []
        seen_edges: Set[Tuple[int, str, int]] = set()

        lines = text.split("\n")
        for line_no, line in enumerate(lines, start=1):
            for target_type, pattern in _REF_PATTERNS:
                for m in pattern.finditer(line):
                    try:
                        num_str = m.group(1)
                        # Handle section numbers like "3.1" — take the integer part
                        num = int(num_str.split(".")[0])
                    except (ValueError, IndexError):
                        continue

                    # Context extraction
                    start = max(0, m.start() - self.context_chars)
                    end = min(len(line), m.end() + self.context_chars)
                    context = line[start:end]

                    # Confidence: explicit patterns > bare patterns
                    confidence = 1.0
                    if pattern is _RE_EQ_BARE:
                        confidence = 0.7  # bare "(1)" is ambiguous

                    mentions.append(RefMention(
                        target_type=target_type,
                        target_number=num,
                        context=context,
                        line_no=line_no,
                        char_offset=m.start(),
                        doc_id=doc_id,
                        confidence=confidence,
                    ))

                    # Build edge (paragraph_line → element)
                    edge_key = (line_no, target_type, num)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append(edge_key)

        return ExtractedRefGraph(
            doc_id=doc_id,
            anchors=anchors,
            mentions=mentions,
            edges=edges,
        )


# ---------------------------------------------------------------------------
# Plain text extractor (fallback for non-MinerU documents)
# ---------------------------------------------------------------------------

class PlainTextRefExtractor(BaseRefExtractor):
    """Extract references from plain text (Word export, HTML-to-text, etc.).

    Uses the same regex patterns as MineruRefExtractor but with relaxed
    anchor detection (no markdown-specific patterns).
    """

    def __init__(self, context_chars: int = 120):
        self.context_chars = context_chars

    def extract_anchors(self, text: str, doc_id: str = "") -> List[ElementAnchor]:
        """Detect element definitions from plain text captions."""
        anchors: List[ElementAnchor] = []
        lines = text.split("\n")
        # Look for lines that start with "Figure N" or "Table N"
        cap_re = re.compile(
            r"^\s*(Figure|Table|Fig\.?|Tab\.?)\s*(\d+)[.:\s]+(.+)$",
            re.IGNORECASE,
        )
        for line_no, line in enumerate(lines, start=1):
            m = cap_re.match(line)
            if m:
                type_str = m.group(1).lower()
                if type_str.startswith("fig"):
                    etype = "figure"
                elif type_str.startswith("tab"):
                    etype = "table"
                else:
                    continue
                anchors.append(ElementAnchor(
                    element_type=etype,
                    element_number=int(m.group(2)),
                    caption=m.group(3).strip()[:500],
                    line_no=line_no,
                    doc_id=doc_id,
                ))
        return anchors

    def extract_refs(self, text: str, doc_id: str = "") -> ExtractedRefGraph:
        """Extract references using same regex patterns."""
        # Delegate to MineruRefExtractor logic (same patterns work on plain text)
        mineru = MineruRefExtractor(self.context_chars)
        result = mineru.extract_refs(text, doc_id)
        # Override anchors with plain-text detection
        result.anchors = self.extract_anchors(text, doc_id)
        return result


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_ref_extractor(doc_format: str = "mineru") -> BaseRefExtractor:
    """Create appropriate reference extractor for document format.

    Args:
        doc_format: "mineru" (default), "plain", or "latex" (not handled here,
                    use LaTeXReferenceExtractor directly)

    Returns:
        BaseRefExtractor instance
    """
    if doc_format == "mineru":
        return MineruRefExtractor()
    elif doc_format == "plain":
        return PlainTextRefExtractor()
    elif doc_format == "latex":
        raise ValueError(
            "For LaTeX documents, use src.parsers.latex_reference_extractor."
            "LaTeXReferenceExtractor directly — it provides higher precision."
        )
    else:
        raise ValueError(f"Unknown document format: {doc_format!r}")


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Extract cross-modal references from document text"
    )
    parser.add_argument("input", help="Input markdown/text file")
    parser.add_argument("--format", choices=["mineru", "plain"], default="mineru")
    parser.add_argument("--doc-id", default="unknown")
    parser.add_argument("--output", help="Output JSON file (default: stdout)")
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    extractor = create_ref_extractor(args.format)
    result = extractor.extract_refs(text, doc_id=args.doc_id)

    output = result.to_dict()
    output["mentions"] = [
        {"type": m.target_type, "number": m.target_number,
         "line_no": m.line_no, "confidence": m.confidence,
         "context": m.context[:100]}
        for m in result.mentions
    ]

    json_str = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Wrote {len(result.mentions)} mentions, {len(result.anchors)} anchors → {args.output}")
    else:
        print(json_str)
