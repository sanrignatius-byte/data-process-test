"""
Multimodal Relationship Builder (Step 0 v2)

Redesigned Step 0 that extracts ALL modality elements (figure, table, formula, section)
from MinerU output and builds a document-internal cross-reference DAG.

Key improvements over original figure_text_associator.py:
1. Treats tables, formulas, sections as first-class elements (not just figures)
2. Extracts cross-reference edges (Figure N → Table M, Table M → Eq K, etc.)
3. Builds a per-document DAG for multi-hop path discovery
4. Generates multimodal pairs (figure↔table, figure↔formula, etc.)

Output: multimodal_elements.json (all elements + DAG + pairs)
"""

import re
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import defaultdict


# ---------------------------------------------------------------------------
# Enums & Data Classes
# ---------------------------------------------------------------------------

class ElementType(Enum):
    FIGURE = "figure"
    TABLE = "table"
    FORMULA = "formula"
    SECTION = "section"
    TEXT = "text"


@dataclass
class DocumentElement:
    """A single semantic element in a document."""
    element_id: str                         # e.g. "1104.3913_figure_2"
    doc_id: str
    element_type: ElementType
    number: Optional[int]                   # Figure 1, Table 2, Eq 3
    label: str                              # "Figure 2", "Table 1", "Eq. 3"
    caption: str
    content: str                            # text / LaTeX / HTML table
    image_path: Optional[str]
    page_idx: int
    position_idx: int                       # reading-order position in flat list
    context_before: str
    context_after: str
    referring_paragraphs: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_id": self.element_id,
            "doc_id": self.doc_id,
            "element_type": self.element_type.value,
            "number": self.number,
            "label": self.label,
            "caption": self.caption,
            "content": self.content[:2000],       # cap for JSON size
            "image_path": self.image_path,
            "page_idx": self.page_idx,
            "position_idx": self.position_idx,
            "context_before": self.context_before[:1500],
            "context_after": self.context_after[:1500],
            "referring_paragraphs": self.referring_paragraphs[:10],
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }


@dataclass
class ReferenceEdge:
    """A directed cross-reference edge in the document DAG."""
    source_id: str                          # element_id of the referencer
    target_id: str                          # element_id of the referenced
    source_type: str                        # ElementType.value
    target_type: str
    ref_text: str                           # "as shown in Figure 3"
    context_snippet: str                    # broader context around ref

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "source_type": self.source_type,
            "target_type": self.target_type,
            "ref_text": self.ref_text,
            "context_snippet": self.context_snippet[:500],
        }


@dataclass
class MultimodalPair:
    """A pair of cross-modal elements connected by reference chain."""
    pair_id: str
    doc_id: str
    element_a_id: str
    element_b_id: str
    element_a_type: str
    element_b_type: str
    hop_distance: int                       # 1 = direct ref, 2 = 2-hop
    path: List[str]                         # element_ids on the path
    relationship: str                       # "direct_reference" / "2_hop" / ...
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "doc_id": self.doc_id,
            "element_a_id": self.element_a_id,
            "element_b_id": self.element_b_id,
            "element_a_type": self.element_a_type,
            "element_b_type": self.element_b_type,
            "hop_distance": self.hop_distance,
            "path": self.path,
            "relationship": self.relationship,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }


@dataclass
class DocumentDAG:
    """Internal reference DAG for a single document."""
    doc_id: str
    elements: Dict[str, DocumentElement] = field(default_factory=dict)
    edges: List[ReferenceEdge] = field(default_factory=list)

    # adjacency: target_id -> [source_ids]  (who references this element)
    _adj_forward: Dict[str, List[str]] = field(
        default_factory=lambda: defaultdict(list), repr=False
    )
    # reverse adjacency: source_id -> [target_ids]
    _adj_reverse: Dict[str, List[str]] = field(
        default_factory=lambda: defaultdict(list), repr=False
    )

    def add_edge(self, edge: ReferenceEdge) -> None:
        self.edges.append(edge)
        self._adj_forward[edge.source_id].append(edge.target_id)
        self._adj_reverse[edge.target_id].append(edge.source_id)

    def get_neighbors(self, element_id: str) -> Set[str]:
        """Get all elements directly connected (either direction)."""
        out = set(self._adj_forward.get(element_id, []))
        out |= set(self._adj_reverse.get(element_id, []))
        return out

    def find_paths(self, max_hops: int = 3) -> List[List[str]]:
        """Find all simple paths of length 2..max_hops between different-type elements."""
        paths: List[List[str]] = []
        for start_id in self.elements:
            self._dfs(start_id, [start_id], max_hops, paths)
        return paths

    def _dfs(
        self,
        current: str,
        path: List[str],
        max_hops: int,
        results: List[List[str]],
    ) -> None:
        if len(path) > max_hops + 1:
            return
        if len(path) >= 2:
            start_type = self.elements[path[0]].element_type
            end_type = self.elements[path[-1]].element_type
            # Only collect paths that cross modalities
            if start_type != end_type:
                results.append(list(path))
        for neighbor in self.get_neighbors(current):
            if neighbor not in path:  # simple path
                path.append(neighbor)
                self._dfs(neighbor, path, max_hops, results)
                path.pop()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "num_elements": len(self.elements),
            "num_edges": len(self.edges),
            "elements": {eid: e.to_dict() for eid, e in self.elements.items()},
            "edges": [e.to_dict() for e in self.edges],
        }


# ---------------------------------------------------------------------------
# Reference Patterns
# ---------------------------------------------------------------------------

# Match "Figure 1", "Fig. 2", "Fig 3a", "Figures 1 and 2"
FIG_REF = re.compile(
    r'(?:Figure|Fig\.?)\s*(\d+)(?:\s*[a-z])?',
    re.IGNORECASE,
)
# Match "Table 1", "Tables 1 and 2"
TABLE_REF = re.compile(
    r'Table\s*(\d+)',
    re.IGNORECASE,
)
# Match "Equation 1", "Eq. (1)", "Eq 1", "Eqn. 1", "(1)" in math context
EQ_REF = re.compile(
    r'(?:Equation|Eq(?:n)?\.?)\s*\(?(\d+)\)?',
    re.IGNORECASE,
)
# Match "Section 3", "Sec. 3.1", "§3"
SECTION_REF = re.compile(
    r'(?:Section|Sec\.?|§)\s*(\d+(?:\.\d+)*)',
    re.IGNORECASE,
)

# Caption patterns (for identifying element boundaries)
FIG_CAPTION = re.compile(
    r'(?:Figure|Fig\.?)\s*(\d+)\s*[:.]\s*(.*)',
    re.IGNORECASE,
)
TABLE_CAPTION = re.compile(
    r'Table\s*(\d+)\s*[:.]\s*(.*)',
    re.IGNORECASE,
)
EQ_LABEL = re.compile(
    r'\((\d+)\)\s*$',  # trailing "(1)" in display equations
)

# Markdown formula blocks
DISPLAY_FORMULA = re.compile(
    r'\$\$(.+?)\$\$',
    re.DOTALL,
)
# HTML table detection
HTML_TABLE = re.compile(
    r'<table[^>]*>.*?</table>',
    re.DOTALL | re.IGNORECASE,
)
# Markdown table: lines with | separators
MD_TABLE_ROW = re.compile(r'^\s*\|.*\|\s*$')
MD_TABLE_SEP = re.compile(r'^\s*\|[\s\-:|]+\|\s*$')

# Inline image in markdown
IMG_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

# Section heading
HEADING_PATTERN = re.compile(r'^(#{1,4})\s+(.+)', re.MULTILINE)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class MultimodalRelationshipBuilder:
    """
    Builds a multimodal DAG from MinerU output.

    Workflow:
      1. Parse content_list.json / markdown → flat record list
      2. Identify all semantic elements (figures, tables, formulas, sections)
      3. Extract cross-reference edges between elements
      4. Build DAG and find multi-hop paths
      5. Generate multimodal pairs for L1 query generation
    """

    def __init__(
        self,
        mineru_output_dir: str,
        context_window: int = 3,
        max_hops: int = 3,
    ):
        self.base_dir = Path(mineru_output_dir)
        self.context_window = context_window
        self.max_hops = max_hops

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def process_all_documents(
        self,
    ) -> Dict[str, DocumentDAG]:
        """Process all documents and return per-document DAGs."""
        results: Dict[str, DocumentDAG] = {}
        doc_dirs = sorted(
            d for d in self.base_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        )
        for doc_dir in doc_dirs:
            doc_id = doc_dir.name
            dag = self.process_document(doc_id)
            if dag and dag.elements:
                results[doc_id] = dag
        return results

    def process_document(self, doc_id: str) -> Optional[DocumentDAG]:
        """Process one document → DocumentDAG."""
        doc_dir = self.base_dir / doc_id

        # 1. Parse into flat records
        records = self._load_records(doc_dir, doc_id)
        if not records:
            return None

        dag = DocumentDAG(doc_id=doc_id)

        # 2. Extract semantic elements
        elements = self._extract_elements(records, doc_id)
        for elem in elements:
            dag.elements[elem.element_id] = elem

        if not dag.elements:
            return dag

        # 3. Build reference edges
        self._build_edges(dag, records)

        return dag

    def generate_multimodal_pairs(
        self,
        dag: DocumentDAG,
        max_pairs_per_doc: int = 50,
    ) -> List[MultimodalPair]:
        """Generate cross-modal pairs from a document DAG."""
        pairs: List[MultimodalPair] = []
        seen: Set[Tuple[str, str]] = set()

        # --- Direct (1-hop) cross-modal edges ---
        for edge in dag.edges:
            if edge.source_id not in dag.elements or edge.target_id not in dag.elements:
                continue
            ea = dag.elements[edge.source_id]
            eb = dag.elements[edge.target_id]
            if ea.element_type == eb.element_type:
                continue
            key = tuple(sorted([ea.element_id, eb.element_id]))
            if key in seen:
                continue
            seen.add(key)

            pair = MultimodalPair(
                pair_id=f"{dag.doc_id}_pair_{len(pairs)+1}",
                doc_id=dag.doc_id,
                element_a_id=ea.element_id,
                element_b_id=eb.element_id,
                element_a_type=ea.element_type.value,
                element_b_type=eb.element_type.value,
                hop_distance=1,
                path=[ea.element_id, eb.element_id],
                relationship="direct_reference",
                quality_score=self._pair_quality(ea, eb, 1),
            )
            pairs.append(pair)

        # --- Multi-hop (2+) cross-modal paths ---
        all_paths = dag.find_paths(max_hops=self.max_hops)
        for path in all_paths:
            if len(path) < 3:
                continue  # 1-hop already handled
            start = path[0]
            end = path[-1]
            key = tuple(sorted([start, end]))
            if key in seen:
                continue
            seen.add(key)

            ea = dag.elements[start]
            eb = dag.elements[end]
            hop_dist = len(path) - 1
            pair = MultimodalPair(
                pair_id=f"{dag.doc_id}_pair_{len(pairs)+1}",
                doc_id=dag.doc_id,
                element_a_id=ea.element_id,
                element_b_id=eb.element_id,
                element_a_type=ea.element_type.value,
                element_b_type=eb.element_type.value,
                hop_distance=hop_dist,
                path=path,
                relationship=f"{hop_dist}_hop",
                quality_score=self._pair_quality(ea, eb, hop_dist),
                metadata={
                    "intermediate_types": [
                        dag.elements[p].element_type.value
                        for p in path[1:-1]
                        if p in dag.elements
                    ],
                },
            )
            pairs.append(pair)

        # Sort by quality and cap
        pairs.sort(key=lambda p: -p.quality_score)
        return pairs[:max_pairs_per_doc]

    # -----------------------------------------------------------------------
    # Record Loading (from content_list or markdown)
    # -----------------------------------------------------------------------

    def _load_records(
        self, doc_dir: Path, doc_id: str
    ) -> List[Dict[str, Any]]:
        """Load flat record list from content_list.json or markdown."""
        cl_path = self._find_content_list(doc_dir, doc_id)
        md_path = self._find_markdown(doc_dir, doc_id)

        records: List[Dict[str, Any]] = []
        if cl_path:
            records = self._flatten_content_list(cl_path)
        if not records and md_path:
            records = self._parse_markdown(md_path)
        return records

    def _find_content_list(self, doc_dir: Path, doc_id: str) -> Optional[Path]:
        candidates = [
            doc_dir / doc_id / "hybrid_auto" / f"{doc_id}_content_list.json",
            doc_dir / doc_id / "hybrid_auto" / "content_list.json",
            doc_dir / doc_id / "auto" / f"{doc_id}_content_list.json",
            doc_dir / doc_id / "auto" / "content_list.json",
            doc_dir / doc_id / "hybrid_auto" / f"{doc_id}_content_list_v2.json",
            doc_dir / doc_id / "auto" / f"{doc_id}_content_list_v2.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        fallback = list(doc_dir.rglob("*content_list*.json"))
        if fallback:
            fallback.sort(key=lambda p: ("v2" in p.name, len(str(p))))
            return fallback[0]
        return None

    def _find_markdown(self, doc_dir: Path, doc_id: str) -> Optional[Path]:
        candidates = [
            doc_dir / doc_id / "hybrid_auto" / f"{doc_id}.md",
            doc_dir / doc_id / "auto" / f"{doc_id}.md",
        ]
        for c in candidates:
            if c.exists():
                return c
        md_files = [f for f in doc_dir.rglob("*.md") if f.name != "formulas.md"]
        return md_files[0] if md_files else None

    def _extract_rich_text(self, content: Any) -> str:
        """Recursively extract text from nested MinerU JSON."""
        if content is None:
            return ""
        if isinstance(content, str):
            return re.sub(r"\s+", " ", content).strip()
        if isinstance(content, list):
            return " ".join(self._extract_rich_text(i) for i in content).strip()
        if isinstance(content, dict):
            keys = [
                "text", "content", "paragraph_content", "title_content",
                "image_caption", "table_caption",
                "page_header_content", "list_content",
            ]
            parts = []
            for k in keys:
                if k in content:
                    t = self._extract_rich_text(content[k])
                    if t:
                        parts.append(t)
            if not parts:
                for v in content.values():
                    if isinstance(v, (str, list, dict)):
                        t = self._extract_rich_text(v)
                        if t:
                            parts.append(t)
            return " ".join(parts).strip()
        return ""

    def _flatten_content_list(self, json_path: Path) -> List[Dict[str, Any]]:
        """Flatten content_list JSON into ordered records."""
        try:
            data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return []

        records: List[Dict[str, Any]] = []
        base_dir = json_path.parent

        def process_item(item: Dict[str, Any], page_idx: int) -> None:
            item_type = str(item.get("type", "")).lower()

            if item_type in {"image", "figure", "table"}:
                # Extract image path
                raw_content = item.get("content", {})
                if not isinstance(raw_content, dict):
                    raw_content = {}
                image_source = raw_content.get("image_source", {})
                if not isinstance(image_source, dict):
                    image_source = {}
                img_ref = (
                    item.get("img_path")
                    or image_source.get("path")
                    or item.get("image_path")
                )
                resolved = self._resolve_image(base_dir, img_ref)

                # Extract caption
                if item_type == "table":
                    cap = item.get("table_caption") or raw_content.get("table_caption")
                else:
                    cap = item.get("image_caption") or raw_content.get("image_caption")
                caption = self._extract_rich_text(cap)

                records.append({
                    "kind": "image",
                    "source_type": item_type,
                    "image_path": str(resolved) if resolved else "",
                    "image_filename": resolved.name if resolved else "",
                    "caption": caption,
                    "page_idx": page_idx,
                })
                return

            if item_type in {"title", "header", "page_header", "section_title"}:
                text = item.get("text") or self._extract_rich_text(item.get("content"))
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    records.append({
                        "kind": "heading",
                        "text": text,
                        "source_type": item_type,
                        "page_idx": page_idx,
                    })
                return

            # Default: text/paragraph/list/etc.
            text = item.get("text") or self._extract_rich_text(item.get("content"))
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) >= 10:
                records.append({
                    "kind": "text",
                    "text": text,
                    "source_type": item_type,
                    "page_idx": page_idx,
                })

        if isinstance(data, list):
            if data and isinstance(data[0], list):
                # v2 nested by page
                for page_idx, page in enumerate(data):
                    for item in page:
                        if isinstance(item, dict):
                            process_item(item, page_idx)
            else:
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        process_item(item, int(item.get("page_idx", idx)))

        return records

    def _parse_markdown(self, md_path: Path) -> List[Dict[str, Any]]:
        """Parse markdown into flat records."""
        content = md_path.read_text(encoding="utf-8", errors="ignore")
        lines = content.split('\n')
        records: List[Dict[str, Any]] = []
        current_text: List[str] = []

        def flush_text() -> None:
            nonlocal current_text
            text = '\n'.join(current_text).strip()
            if text and len(text) >= 10:
                records.append({"kind": "text", "text": text, "page_idx": 0, "source_type": "text"})
            current_text = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Image
            img_match = IMG_PATTERN.search(line)
            if img_match:
                flush_text()
                records.append({
                    "kind": "image",
                    "source_type": "image",
                    "image_path": str(md_path.parent / "images" / Path(img_match.group(2)).name),
                    "image_filename": Path(img_match.group(2)).name,
                    "caption": img_match.group(1),
                    "page_idx": 0,
                })
                i += 1
                continue

            # Heading
            if line.startswith('#'):
                flush_text()
                level = len(line) - len(line.lstrip('#'))
                text = line.lstrip('#').strip()
                if text:
                    records.append({
                        "kind": "heading",
                        "text": text,
                        "source_type": "title",
                        "page_idx": 0,
                        "level": level,
                    })
                i += 1
                continue

            # Display formula $$...$$
            if line.strip().startswith('$$'):
                flush_text()
                formula_lines = [line]
                if not (line.strip().endswith('$$') and line.strip() != '$$'):
                    j = i + 1
                    while j < len(lines):
                        formula_lines.append(lines[j])
                        if lines[j].strip().endswith('$$'):
                            break
                        j += 1
                    i = j
                records.append({
                    "kind": "formula",
                    "text": '\n'.join(formula_lines),
                    "source_type": "formula",
                    "page_idx": 0,
                })
                i += 1
                continue

            current_text.append(line)
            i += 1

        flush_text()
        return records

    def _resolve_image(self, base_dir: Path, ref: Optional[str]) -> Optional[Path]:
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

    # -----------------------------------------------------------------------
    # Element Extraction
    # -----------------------------------------------------------------------

    def _extract_elements(
        self,
        records: List[Dict[str, Any]],
        doc_id: str,
    ) -> List[DocumentElement]:
        """Extract all semantic elements from flat records."""
        elements: List[DocumentElement] = []

        # Counters for numbering
        counters: Dict[str, int] = defaultdict(int)

        for idx, rec in enumerate(records):
            kind = rec.get("kind", "")

            if kind == "image":
                elem = self._extract_figure_or_table(rec, records, idx, doc_id, counters)
                if elem:
                    elements.append(elem)

            elif kind == "heading":
                elem = self._extract_section(rec, records, idx, doc_id, counters)
                if elem:
                    elements.append(elem)

            elif kind == "text":
                text = rec.get("text", "")
                # Check for inline/embedded tables (HTML or markdown)
                table_elems = self._extract_tables_from_text(text, rec, records, idx, doc_id, counters)
                elements.extend(table_elems)

                # Check for display formulas embedded in text blocks
                formula_elems = self._extract_formulas_from_text(text, rec, records, idx, doc_id, counters)
                elements.extend(formula_elems)

            elif kind == "formula":
                elem = self._extract_formula(rec, records, idx, doc_id, counters)
                if elem:
                    elements.append(elem)

        # Post-process: find referring paragraphs for numbered elements
        text_records = [r for r in records if r.get("kind") == "text"]
        for elem in elements:
            if elem.number is not None:
                refs = self._find_referring_paragraphs(
                    text_records, elem.element_type, elem.number
                )
                elem.referring_paragraphs = refs

        # Compute quality
        for elem in elements:
            elem.quality_score = self._compute_quality(elem)

        return elements

    def _extract_figure_or_table(
        self,
        rec: Dict[str, Any],
        records: List[Dict[str, Any]],
        idx: int,
        doc_id: str,
        counters: Dict[str, int],
    ) -> Optional[DocumentElement]:
        """Extract a figure or table-image element from an image record."""
        image_path = rec.get("image_path", "")
        caption = rec.get("caption", "")
        source_type = rec.get("source_type", "image")

        # Try to get caption from neighbors if missing
        if not caption:
            caption = self._infer_caption(records, idx)

        # Determine if this is a table or figure
        is_table = source_type == "table"
        if not is_table and caption:
            if TABLE_CAPTION.search(caption):
                is_table = True

        elem_type = ElementType.TABLE if is_table else ElementType.FIGURE
        caption_pattern = TABLE_CAPTION if is_table else FIG_CAPTION

        # Extract number from caption
        number = None
        if caption:
            m = caption_pattern.search(caption)
            if m:
                number = int(m.group(1))

        counters[elem_type.value] += 1
        element_id = f"{doc_id}_{elem_type.value}_{number or counters[elem_type.value]}"

        label = ""
        if number is not None:
            label = f"{'Table' if is_table else 'Figure'} {number}"

        return DocumentElement(
            element_id=element_id,
            doc_id=doc_id,
            element_type=elem_type,
            number=number,
            label=label,
            caption=caption,
            content=caption,  # for images, content = caption
            image_path=image_path if image_path else None,
            page_idx=rec.get("page_idx", 0),
            position_idx=idx,
            context_before=self._get_context(records, idx, "before"),
            context_after=self._get_context(records, idx, "after"),
            metadata={
                "source_type": source_type,
                "image_filename": rec.get("image_filename", ""),
            },
        )

    def _extract_section(
        self,
        rec: Dict[str, Any],
        records: List[Dict[str, Any]],
        idx: int,
        doc_id: str,
        counters: Dict[str, int],
    ) -> Optional[DocumentElement]:
        """Extract a section heading element."""
        text = rec.get("text", "").strip()
        if not text or len(text) < 3:
            return None

        # Try to extract section number
        number = None
        m = re.match(r'^(\d+(?:\.\d+)*)\s', text)
        if m:
            # Use integer part only for matching
            try:
                number = int(m.group(1).split('.')[0])
            except ValueError:
                pass

        counters["section"] += 1
        element_id = f"{doc_id}_section_{number or counters['section']}"

        return DocumentElement(
            element_id=element_id,
            doc_id=doc_id,
            element_type=ElementType.SECTION,
            number=number,
            label=f"Section {number}" if number else text[:50],
            caption=text,
            content=text,
            image_path=None,
            page_idx=rec.get("page_idx", 0),
            position_idx=idx,
            context_before="",
            context_after=self._get_context(records, idx, "after"),
            metadata={"level": rec.get("level", 1)},
        )

    def _extract_tables_from_text(
        self,
        text: str,
        rec: Dict[str, Any],
        records: List[Dict[str, Any]],
        idx: int,
        doc_id: str,
        counters: Dict[str, int],
    ) -> List[DocumentElement]:
        """Extract tables embedded in text blocks (HTML or markdown tables)."""
        elements: List[DocumentElement] = []

        # HTML tables
        for m in HTML_TABLE.finditer(text):
            table_html = m.group(0)
            # Try to find table number from surrounding text
            number = None
            cap_match = TABLE_CAPTION.search(text[:m.start()])
            if cap_match:
                number = int(cap_match.group(1))

            counters["table"] += 1
            elem_id = f"{doc_id}_table_{number or counters['table']}"

            elem = DocumentElement(
                element_id=elem_id,
                doc_id=doc_id,
                element_type=ElementType.TABLE,
                number=number,
                label=f"Table {number}" if number else f"Table (inline)",
                caption=cap_match.group(0) if cap_match else "",
                content=table_html,
                image_path=None,
                page_idx=rec.get("page_idx", 0),
                position_idx=idx,
                context_before=self._get_context(records, idx, "before"),
                context_after=self._get_context(records, idx, "after"),
                metadata={"source": "html_in_text", "row_count": table_html.count("<tr")},
            )
            elements.append(elem)

        # Markdown tables (consecutive lines with |)
        lines = text.split('\n')
        table_start = None
        table_lines: List[str] = []
        for li, line in enumerate(lines):
            if MD_TABLE_ROW.match(line):
                if table_start is None:
                    table_start = li
                table_lines.append(line)
            else:
                if table_lines and len(table_lines) >= 3:
                    # Check for separator row
                    has_sep = any(MD_TABLE_SEP.match(tl) for tl in table_lines)
                    if has_sep:
                        number = None
                        # Look for Table N in preceding text
                        preceding = '\n'.join(lines[:table_start])
                        cap_match = TABLE_CAPTION.search(preceding)
                        if cap_match:
                            number = int(cap_match.group(1))

                        counters["table"] += 1
                        elem_id = f"{doc_id}_table_{number or counters['table']}"

                        elem = DocumentElement(
                            element_id=elem_id,
                            doc_id=doc_id,
                            element_type=ElementType.TABLE,
                            number=number,
                            label=f"Table {number}" if number else "Table (markdown)",
                            caption=cap_match.group(0) if cap_match else "",
                            content='\n'.join(table_lines),
                            image_path=None,
                            page_idx=rec.get("page_idx", 0),
                            position_idx=idx,
                            context_before=self._get_context(records, idx, "before"),
                            context_after=self._get_context(records, idx, "after"),
                            metadata={"source": "markdown_table", "row_count": len(table_lines) - 1},
                        )
                        elements.append(elem)
                table_lines = []
                table_start = None

        # Flush remaining table
        if table_lines and len(table_lines) >= 3:
            has_sep = any(MD_TABLE_SEP.match(tl) for tl in table_lines)
            if has_sep:
                counters["table"] += 1
                elem = DocumentElement(
                    element_id=f"{doc_id}_table_{counters['table']}",
                    doc_id=doc_id,
                    element_type=ElementType.TABLE,
                    number=None,
                    label="Table (markdown)",
                    caption="",
                    content='\n'.join(table_lines),
                    image_path=None,
                    page_idx=rec.get("page_idx", 0),
                    position_idx=idx,
                    context_before=self._get_context(records, idx, "before"),
                    context_after=self._get_context(records, idx, "after"),
                    metadata={"source": "markdown_table", "row_count": len(table_lines) - 1},
                )
                elements.append(elem)

        return elements

    def _extract_formulas_from_text(
        self,
        text: str,
        rec: Dict[str, Any],
        records: List[Dict[str, Any]],
        idx: int,
        doc_id: str,
        counters: Dict[str, int],
    ) -> List[DocumentElement]:
        """Extract display formulas embedded in text blocks."""
        elements: List[DocumentElement] = []

        for m in DISPLAY_FORMULA.finditer(text):
            formula_content = m.group(1).strip()
            if len(formula_content) < 10:
                continue  # skip trivial inline

            # Try to extract equation number
            number = None
            eq_match = EQ_LABEL.search(formula_content)
            if eq_match:
                number = int(eq_match.group(1))

            counters["formula"] += 1
            elem_id = f"{doc_id}_formula_{number or counters['formula']}"

            elem = DocumentElement(
                element_id=elem_id,
                doc_id=doc_id,
                element_type=ElementType.FORMULA,
                number=number,
                label=f"Equation {number}" if number else f"Formula",
                caption="",
                content=f"$${formula_content}$$",
                image_path=None,
                page_idx=rec.get("page_idx", 0),
                position_idx=idx,
                context_before=self._get_context(records, idx, "before"),
                context_after=self._get_context(records, idx, "after"),
                metadata={"source": "display_math_in_text"},
            )
            elements.append(elem)

        return elements

    def _extract_formula(
        self,
        rec: Dict[str, Any],
        records: List[Dict[str, Any]],
        idx: int,
        doc_id: str,
        counters: Dict[str, int],
    ) -> Optional[DocumentElement]:
        """Extract a standalone formula record."""
        text = rec.get("text", "").strip()
        if len(text) < 10:
            return None

        number = None
        eq_match = EQ_LABEL.search(text)
        if eq_match:
            number = int(eq_match.group(1))

        counters["formula"] += 1
        elem_id = f"{doc_id}_formula_{number or counters['formula']}"

        return DocumentElement(
            element_id=elem_id,
            doc_id=doc_id,
            element_type=ElementType.FORMULA,
            number=number,
            label=f"Equation {number}" if number else "Formula",
            caption="",
            content=text,
            image_path=None,
            page_idx=rec.get("page_idx", 0),
            position_idx=idx,
            context_before=self._get_context(records, idx, "before"),
            context_after=self._get_context(records, idx, "after"),
            metadata={"source": "standalone_formula"},
        )

    # -----------------------------------------------------------------------
    # Edge Extraction (Cross-Reference DAG)
    # -----------------------------------------------------------------------

    def _build_edges(self, dag: DocumentDAG, records: List[Dict[str, Any]]) -> None:
        """Extract cross-reference edges from text into the DAG."""
        # Build lookup: (element_type, number) → element_id
        elem_lookup: Dict[Tuple[str, int], str] = {}
        for elem in dag.elements.values():
            if elem.number is not None:
                elem_lookup[(elem.element_type.value, elem.number)] = elem.element_id

        # Track seen edges to avoid duplicates
        seen_edges: Set[Tuple[str, str]] = set()

        # Reference patterns mapped to target element types
        ref_patterns: List[Tuple[re.Pattern, str]] = [
            (FIG_REF, "figure"),
            (TABLE_REF, "table"),
            (EQ_REF, "formula"),
            (SECTION_REF, "section"),
        ]

        # For each text-bearing element, scan for references to other elements
        for source_elem in dag.elements.values():
            # Scan all text associated with this element
            texts_to_scan = [
                source_elem.content,
                source_elem.context_before,
                source_elem.context_after,
                source_elem.caption,
            ]
            texts_to_scan.extend(source_elem.referring_paragraphs)

            for text in texts_to_scan:
                if not text:
                    continue
                for pattern, target_type in ref_patterns:
                    for m in pattern.finditer(text):
                        try:
                            target_num = int(m.group(1).split('.')[0])
                        except (ValueError, IndexError):
                            continue

                        target_id = elem_lookup.get((target_type, target_num))
                        if not target_id:
                            continue
                        if target_id == source_elem.element_id:
                            continue  # no self-loops

                        edge_key = (source_elem.element_id, target_id)
                        if edge_key in seen_edges:
                            continue
                        seen_edges.add(edge_key)

                        # Extract context around the reference
                        start = max(0, m.start() - 80)
                        end = min(len(text), m.end() + 80)
                        context_snippet = text[start:end]

                        edge = ReferenceEdge(
                            source_id=source_elem.element_id,
                            target_id=target_id,
                            source_type=source_elem.element_type.value,
                            target_type=target_type,
                            ref_text=m.group(0),
                            context_snippet=context_snippet,
                        )
                        dag.add_edge(edge)

        # Also scan raw text records for references between elements
        # (captures references in body text not directly attached to any element)
        for rec in records:
            if rec.get("kind") != "text":
                continue
            text = rec.get("text", "")
            if not text:
                continue

            # Find all references in this text block
            refs_found: List[Tuple[str, int, str, int, int]] = []  # (type, num, match_text, start, end)
            for pattern, target_type in ref_patterns:
                for m in pattern.finditer(text):
                    try:
                        num = int(m.group(1).split('.')[0])
                    except (ValueError, IndexError):
                        continue
                    refs_found.append((target_type, num, m.group(0), m.start(), m.end()))

            # If a single text block references 2+ different elements, add edges between them
            if len(refs_found) >= 2:
                for i in range(len(refs_found)):
                    for j in range(i + 1, len(refs_found)):
                        type_i, num_i, text_i, _, _ = refs_found[i]
                        type_j, num_j, text_j, _, _ = refs_found[j]
                        id_i = elem_lookup.get((type_i, num_i))
                        id_j = elem_lookup.get((type_j, num_j))
                        if not id_i or not id_j or id_i == id_j:
                            continue

                        # Co-reference in same paragraph → bidirectional edge
                        edge_key = (id_i, id_j)
                        if edge_key in seen_edges:
                            continue
                        seen_edges.add(edge_key)

                        edge = ReferenceEdge(
                            source_id=id_i,
                            target_id=id_j,
                            source_type=type_i,
                            target_type=type_j,
                            ref_text=f"{text_i} ... {text_j}",
                            context_snippet=text[:300],
                        )
                        dag.add_edge(edge)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_context(
        self,
        records: List[Dict[str, Any]],
        idx: int,
        direction: str,
    ) -> str:
        """Get surrounding text context."""
        parts: List[str] = []
        collected = 0
        rng = (
            range(idx - 1, -1, -1) if direction == "before"
            else range(idx + 1, len(records))
        )
        for j in rng:
            if collected >= self.context_window:
                break
            rec = records[j]
            kind = rec.get("kind", "")
            if kind == "image":
                break
            if kind == "heading":
                text = rec.get("text", "").strip()
                if text:
                    parts.append(f"[Section: {text}]")
                    collected += 1
            elif kind == "text":
                text = rec.get("text", "").strip()
                if len(text) < 20:
                    continue
                if FIG_CAPTION.match(text) or TABLE_CAPTION.match(text):
                    continue
                parts.append(text)
                collected += 1

        if direction == "before":
            parts.reverse()
        return "\n\n".join(parts)

    def _infer_caption(self, records: List[Dict[str, Any]], idx: int) -> str:
        """Infer caption from neighboring records."""
        for j in range(idx + 1, min(idx + 4, len(records))):
            rec = records[j]
            if rec.get("kind") == "image":
                break
            text = rec.get("text", "").strip()
            if text and (FIG_CAPTION.search(text) or TABLE_CAPTION.search(text)):
                return text[:300]
            if text and len(text) < 200:
                return text
        for j in range(idx - 1, max(-1, idx - 4), -1):
            rec = records[j]
            if rec.get("kind") == "image":
                break
            text = rec.get("text", "").strip()
            if text and (FIG_CAPTION.search(text) or TABLE_CAPTION.search(text)):
                return text[:300]
        return ""

    def _find_referring_paragraphs(
        self,
        text_records: List[Dict[str, Any]],
        elem_type: ElementType,
        number: int,
    ) -> List[str]:
        """Find paragraphs referencing a specific element."""
        refs: List[str] = []
        if elem_type == ElementType.FIGURE:
            pattern = re.compile(
                rf'\bFig(?:ure|\.)\s*{number}\b', re.IGNORECASE
            )
        elif elem_type == ElementType.TABLE:
            pattern = re.compile(rf'\bTable\s*{number}\b', re.IGNORECASE)
        elif elem_type == ElementType.FORMULA:
            pattern = re.compile(
                rf'(?:Eq(?:uation|n)?\.?\s*\(?{number}\)?|\({number}\))',
                re.IGNORECASE,
            )
        elif elem_type == ElementType.SECTION:
            pattern = re.compile(
                rf'(?:Section|Sec\.?)\s*{number}\b', re.IGNORECASE
            )
        else:
            return []

        for rec in text_records:
            text = rec.get("text", "")
            if pattern.search(text):
                refs.append(text[:500])
        return refs[:10]

    def _compute_quality(self, elem: DocumentElement) -> float:
        """Compute quality score for an element."""
        score = 0.0

        if elem.caption:
            score += 0.25
        if elem.context_before:
            score += 0.15
        if elem.context_after:
            score += 0.15
        if elem.referring_paragraphs:
            score += min(0.25, len(elem.referring_paragraphs) * 0.08)
        if elem.number is not None:
            score += 0.1  # numbered elements are more anchored

        # Type-specific bonuses
        if elem.element_type == ElementType.TABLE:
            row_count = elem.metadata.get("row_count", 0)
            if row_count >= 3:
                score += 0.1
        elif elem.element_type == ElementType.FIGURE:
            if elem.image_path:
                p = Path(elem.image_path)
                if p.exists() and p.stat().st_size > 10000:
                    score += 0.1
        elif elem.element_type == ElementType.FORMULA:
            if len(elem.content) > 30:
                score += 0.1

        return min(1.0, score)

    def _pair_quality(
        self,
        ea: DocumentElement,
        eb: DocumentElement,
        hop_distance: int,
    ) -> float:
        """Score a multimodal pair."""
        score = (ea.quality_score + eb.quality_score) / 2

        # Cross-modal bonus
        type_pair = frozenset([ea.element_type.value, eb.element_type.value])
        cross_modal_bonus = {
            frozenset(["figure", "table"]): 0.3,
            frozenset(["figure", "formula"]): 0.25,
            frozenset(["table", "formula"]): 0.2,
            frozenset(["figure", "section"]): 0.1,
            frozenset(["table", "section"]): 0.1,
        }
        score += cross_modal_bonus.get(type_pair, 0.05)

        # Penalize longer hops slightly
        if hop_distance > 1:
            score -= 0.05 * (hop_distance - 1)

        # Both elements numbered → stronger pair
        if ea.number is not None and eb.number is not None:
            score += 0.1

        return min(1.0, max(0.0, score))

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def compute_stats(
        self,
        dags: Dict[str, DocumentDAG],
        pairs: Dict[str, List[MultimodalPair]],
    ) -> Dict[str, Any]:
        """Compute aggregate statistics."""
        total_elements = 0
        total_edges = 0
        total_pairs = 0
        type_counts: Dict[str, int] = defaultdict(int)
        edge_type_counts: Dict[str, int] = defaultdict(int)
        pair_type_counts: Dict[str, int] = defaultdict(int)
        hop_dist_counts: Dict[int, int] = defaultdict(int)
        docs_with_elements = 0

        for doc_id, dag in dags.items():
            if dag.elements:
                docs_with_elements += 1
            total_elements += len(dag.elements)
            total_edges += len(dag.edges)
            for elem in dag.elements.values():
                type_counts[elem.element_type.value] += 1
            for edge in dag.edges:
                key = f"{edge.source_type}→{edge.target_type}"
                edge_type_counts[key] += 1

        for doc_id, doc_pairs in pairs.items():
            total_pairs += len(doc_pairs)
            for p in doc_pairs:
                # Canonical order for pair types
                key = "↔".join(sorted([p.element_a_type, p.element_b_type]))
                pair_type_counts[key] += 1
                hop_dist_counts[p.hop_distance] += 1

        return {
            "total_documents": len(dags),
            "docs_with_elements": docs_with_elements,
            "total_elements": total_elements,
            "total_edges": total_edges,
            "total_pairs": total_pairs,
            "element_type_distribution": dict(type_counts),
            "edge_type_distribution": dict(sorted(edge_type_counts.items(), key=lambda x: -x[1])),
            "pair_type_distribution": dict(sorted(pair_type_counts.items(), key=lambda x: -x[1])),
            "hop_distance_distribution": dict(sorted(hop_dist_counts.items())),
            "avg_elements_per_doc": total_elements / max(1, docs_with_elements),
            "avg_edges_per_doc": total_edges / max(1, docs_with_elements),
            "avg_pairs_per_doc": total_pairs / max(1, docs_with_elements),
        }
