"""
MinerU Parser Wrapper

Wraps the MinerU CLI tool for document parsing with GPU parallelization support.
Uses the `mineru` command-line interface (not the deprecated magic-pdf).
"""

import os
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import time
import re

from src.utils.file_utils import safe_json_dump


@dataclass
class ParsedElement:
    """Represents a single parsed element from a document."""
    element_id: str
    doc_id: str
    page_idx: int
    element_type: str  # table, figure, formula, text, infographic
    content: str
    bbox: Optional[List[float]] = None
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Represents a fully parsed document."""
    doc_id: str
    pdf_path: str
    output_path: str
    total_pages: int
    elements: List[ParsedElement]
    parse_time: float
    success: bool
    error_message: Optional[str] = None


class MinerUParser:
    """
    MinerU CLI wrapper for document parsing.

    Uses the `mineru` command (MinerU 2.0+) for PDF parsing with support for:
    - GPU acceleration
    - Multiple backend options (auto, pipeline, hybrid, vlm)
    - Parallel document processing across multiple GPUs
    """

    def __init__(
        self,
        output_dir: str,
        backend: str = "auto",
        devices: List[str] = None,
        num_workers: int = 4,
        language: str = "en",
        timeout: int = 300,
        verify_installation: bool = True
    ):
        """
        Initialize MinerU parser.

        Args:
            output_dir: Directory for parsed outputs
            backend: Parsing backend ("auto", "pipeline", "hybrid", "vlm")
            devices: List of CUDA devices (e.g., ["cuda:0", "cuda:1"])
            num_workers: Number of parallel workers
            language: Document language ("en", "ch", "multi")
            timeout: Timeout per document in seconds
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.devices = devices or ["cuda:0"]
        self.num_workers = min(num_workers, len(self.devices))
        self.language = language
        self.timeout = timeout

        # Verify mineru is available
        if verify_installation:
            self._verify_installation()

    def _verify_installation(self) -> None:
        """Verify MinerU CLI is installed and accessible."""
        try:
            result = subprocess.run(
                ["mineru", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip() or result.stderr.strip()
                print(f"MinerU version: {version}")
            else:
                # Try alternative check
                result = subprocess.run(
                    ["mineru", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode != 0:
                    raise RuntimeError("mineru command not available")
        except FileNotFoundError:
            raise RuntimeError(
                "MinerU CLI not found. Please install with: pip install mineru[all]"
            )
        except subprocess.TimeoutExpired:
            print("Warning: MinerU version check timed out")

    def _build_command(
        self,
        pdf_path: str,
        output_path: str,
        device_id: int = 0
    ) -> List[str]:
        """Build mineru CLI command."""
        cmd = [
            "mineru",
            "-p", str(pdf_path),
            "-o", str(output_path),
        ]

        # Add backend option
        if self.backend != "auto":
            cmd.extend(["-b", self.backend])

        # Add language option if not English
        if self.language != "en":
            cmd.extend(["-l", self.language])

        return cmd

    def _collect_output_files(self, output_dir: Path) -> List[Path]:
        """Collect expected MinerU output files."""
        if not output_dir.exists():
            return []

        valid_suffixes = {
            ".md",
            ".json",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".webp",
            ".svg"
        }

        return [
            p for p in output_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_suffixes
        ]

    def _extract_text_from_rich_content(self, content: Any) -> str:
        """Extract plain text from nested MinerU content structures."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(
                        self._extract_text_from_rich_content(
                            item.get("content") or item.get("text")
                        )
                    )
                else:
                    parts.append(self._extract_text_from_rich_content(item))
            return " ".join(p for p in parts if p)
        if isinstance(content, dict):
            if "text" in content and isinstance(content["text"], str):
                return content["text"]
            if "content" in content:
                return self._extract_text_from_rich_content(content["content"])
            if "title_content" in content:
                return self._extract_text_from_rich_content(content["title_content"])
            if "image_caption" in content:
                return self._extract_text_from_rich_content(content["image_caption"])
            if "table_caption" in content:
                return self._extract_text_from_rich_content(content["table_caption"])
        return ""

    def _normalize_latex(self, latex: Optional[str]) -> str:
        """Normalize LaTeX by stripping common block delimiters."""
        if not latex:
            return ""

        text = latex.strip()
        if text.startswith("$$"):
            text = text[2:]
        if text.endswith("$$"):
            text = text[:-2]
        if text.startswith("\\["):
            text = text[2:]
        if text.endswith("\\]"):
            text = text[:-2]

        return text.strip()

    def parse_single(
        self,
        pdf_path: str,
        doc_id: Optional[str] = None,
        device_id: int = 0
    ) -> ParsedDocument:
        """
        Parse a single PDF document.

        Args:
            pdf_path: Path to PDF file
            doc_id: Document identifier (defaults to filename)
            device_id: GPU device ID to use

        Returns:
            ParsedDocument with extracted elements
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return ParsedDocument(
                doc_id=doc_id or pdf_path.stem,
                pdf_path=str(pdf_path),
                output_path="",
                total_pages=0,
                elements=[],
                parse_time=0,
                success=False,
                error_message=f"PDF file not found: {pdf_path}"
            )

        doc_id = doc_id or pdf_path.stem
        doc_output_dir = self.output_dir / doc_id
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        # Set CUDA device for this process
        env = os.environ.copy()
        if device_id < len(self.devices):
            device = self.devices[device_id]
            if device.startswith("cuda:"):
                cuda_id = device.split(":")[1]
                env["CUDA_VISIBLE_DEVICES"] = cuda_id

        try:
            # Build and execute command
            cmd = self._build_command(str(pdf_path), str(doc_output_dir), device_id)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env
            )

            if result.stdout:
                (doc_output_dir / "mineru_stdout.log").write_text(result.stdout, encoding="utf-8")
            if result.stderr:
                (doc_output_dir / "mineru_stderr.log").write_text(result.stderr, encoding="utf-8")

            if result.returncode != 0:
                return ParsedDocument(
                    doc_id=doc_id,
                    pdf_path=str(pdf_path),
                    output_path=str(doc_output_dir),
                    total_pages=0,
                    elements=[],
                    parse_time=time.time() - start_time,
                    success=False,
                    error_message=f"MinerU error: {result.stderr}"
                )

            output_files = self._collect_output_files(doc_output_dir)
            if not output_files:
                return ParsedDocument(
                    doc_id=doc_id,
                    pdf_path=str(pdf_path),
                    output_path=str(doc_output_dir),
                    total_pages=0,
                    elements=[],
                    parse_time=time.time() - start_time,
                    success=False,
                    error_message="MinerU produced no output files. Check mineru_stderr.log."
                )

            # Parse output files
            elements = self._extract_elements_from_output(doc_output_dir, doc_id)
            total_pages = self._count_pages(doc_output_dir)

            return ParsedDocument(
                doc_id=doc_id,
                pdf_path=str(pdf_path),
                output_path=str(doc_output_dir),
                total_pages=total_pages,
                elements=elements,
                parse_time=time.time() - start_time,
                success=True
            )

        except subprocess.TimeoutExpired:
            return ParsedDocument(
                doc_id=doc_id,
                pdf_path=str(pdf_path),
                output_path=str(doc_output_dir),
                total_pages=0,
                elements=[],
                parse_time=self.timeout,
                success=False,
                error_message=f"Timeout after {self.timeout}s"
            )
        except Exception as e:
            return ParsedDocument(
                doc_id=doc_id,
                pdf_path=str(pdf_path),
                output_path=str(doc_output_dir),
                total_pages=0,
                elements=[],
                parse_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    def _extract_elements_from_output(
        self,
        output_dir: Path,
        doc_id: str
    ) -> List[ParsedElement]:
        """
        Extract structured elements from MinerU output.

        MinerU outputs:
        - markdown files with text content
        - images/ directory with extracted figures
        - JSON files with structure information
        """
        elements = []
        element_counter = 0

        # Find the actual output directory (MinerU creates subdirectory)
        actual_output = output_dir
        if not (output_dir / "auto").exists():
            # Check for subdirectory with PDF name
            subdirs = list(output_dir.iterdir()) if output_dir.exists() else []
            for subdir in subdirs:
                if subdir.is_dir():
                    actual_output = subdir
                    break

        # Try multiple possible output structures
        possible_md_paths = [
            actual_output / "auto" / f"{doc_id}.md",
            actual_output / f"{doc_id}.md",
            actual_output / "auto" / "content.md",
            actual_output / "content.md",
        ]

        # Also check for any .md file
        if actual_output.exists():
            md_files = list(actual_output.rglob("*.md"))
            possible_md_paths.extend(md_files)

        md_path = None
        for path in possible_md_paths:
            if isinstance(path, Path) and path.exists():
                md_path = path
                break

        if md_path:
            elements.extend(
                self._parse_markdown_content(md_path, doc_id, element_counter)
            )
            element_counter = len(elements)

        # Extract images
        image_dirs = [
            actual_output / "auto" / "images",
            actual_output / "images",
        ]

        for img_dir in image_dirs:
            if img_dir.exists():
                for img_file in img_dir.glob("*"):
                    if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif"]:
                        elements.append(ParsedElement(
                            element_id=f"{doc_id}_img_{element_counter}",
                            doc_id=doc_id,
                            page_idx=self._extract_page_from_filename(img_file.name),
                            element_type="figure",
                            content=f"[Image: {img_file.name}]",
                            image_path=str(img_file),
                            metadata={"filename": img_file.name}
                        ))
                        element_counter += 1
                break  # Only process first found image directory

        # Parse JSON structure if available
        json_paths = [
            actual_output / "auto" / "content_list.json",
            actual_output / "content_list.json",
            actual_output / "auto" / "content_list_v2.json",
            actual_output / "content_list_v2.json",
            actual_output / "auto" / "middle.json",
            actual_output / "middle.json",
        ]

        for json_path in json_paths:
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        structure = json.load(f)
                    additional = self._parse_json_structure(
                        structure,
                        doc_id,
                        element_counter,
                        base_dir=actual_output
                    )
                    # Only add elements not already captured
                    existing_types = {e.element_type for e in elements}
                    for elem in additional:
                        if elem.element_type not in existing_types or elem.element_type == "table":
                            elements.append(elem)
                            element_counter += 1
                except Exception:
                    pass
                break

        return elements

    def _parse_markdown_content(
        self,
        md_path: Path,
        doc_id: str,
        start_idx: int = 0
    ) -> List[ParsedElement]:
        """Parse markdown file to extract elements."""
        elements = []
        element_counter = start_idx

        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by sections/blocks
        current_page = 0
        lines = content.split('\n')

        current_text = []
        in_table = False
        table_content = []
        in_formula = False
        formula_content = []

        for line in lines:
            # Track page markers if present
            if line.startswith('<!-- Page'):
                page_match = re.search(r'Page\s+(\d+)', line)
                if page_match:
                    current_page = int(page_match.group(1)) - 1

            # Table detection (markdown tables start with |)
            if line.strip().startswith('|'):
                if not in_table:
                    # Save accumulated text
                    if current_text:
                        text = '\n'.join(current_text).strip()
                        if len(text) > 30:
                            elements.append(ParsedElement(
                                element_id=f"{doc_id}_text_{element_counter}",
                                doc_id=doc_id,
                                page_idx=current_page,
                                element_type="text",
                                content=text
                            ))
                            element_counter += 1
                        current_text = []
                    in_table = True
                table_content.append(line)
            elif in_table:
                # End of table
                table_text = '\n'.join(table_content).strip()
                if table_text:
                    elements.append(ParsedElement(
                        element_id=f"{doc_id}_table_{element_counter}",
                        doc_id=doc_id,
                        page_idx=current_page,
                        element_type="table",
                        content=table_text,
                        metadata={"rows": len(table_content) - 1}  # Minus header separator
                    ))
                    element_counter += 1
                table_content = []
                in_table = False

            # Formula detection (LaTeX blocks)
            if line.strip().startswith('$$') or line.strip().startswith('\\['):
                if not in_formula:
                    in_formula = True
                    formula_content = [line]
                else:
                    formula_content.append(line)
                    formula_text = '\n'.join(formula_content).strip()
                    normalized = self._normalize_latex(formula_text)
                    elements.append(ParsedElement(
                        element_id=f"{doc_id}_formula_{element_counter}",
                        doc_id=doc_id,
                        page_idx=current_page,
                        element_type="formula",
                        content=normalized or formula_text,
                        metadata={"latex": normalized or formula_text}
                    ))
                    element_counter += 1
                    in_formula = False
                    formula_content = []
                continue

            if in_formula:
                formula_content.append(line)
                if line.strip().endswith('$$') or line.strip().endswith('\\]'):
                    formula_text = '\n'.join(formula_content).strip()
                    normalized = self._normalize_latex(formula_text)
                    elements.append(ParsedElement(
                        element_id=f"{doc_id}_formula_{element_counter}",
                        doc_id=doc_id,
                        page_idx=current_page,
                        element_type="formula",
                        content=normalized or formula_text,
                        metadata={"latex": normalized or formula_text}
                    ))
                    element_counter += 1
                    in_formula = False
                    formula_content = []
                continue

            # Image references
            img_match = re.search(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            if img_match:
                caption = img_match.group(1)
                img_ref = img_match.group(2)
                elements.append(ParsedElement(
                    element_id=f"{doc_id}_fig_{element_counter}",
                    doc_id=doc_id,
                    page_idx=current_page,
                    element_type="figure",
                    content=caption or f"[Image: {img_ref}]",
                    image_path=img_ref,
                    metadata={"caption": caption, "reference": img_ref}
                ))
                element_counter += 1
                continue

            # Regular text
            if not in_table and not in_formula:
                current_text.append(line)

        # Final accumulated text
        if current_text:
            text = '\n'.join(current_text).strip()
            if len(text) > 30:
                elements.append(ParsedElement(
                    element_id=f"{doc_id}_text_{element_counter}",
                    doc_id=doc_id,
                    page_idx=current_page,
                    element_type="text",
                    content=text
                ))

        return elements

    def _parse_json_structure(
        self,
        structure: Any,
        doc_id: str,
        start_idx: int = 0,
        base_dir: Optional[Path] = None
    ) -> List[ParsedElement]:
        """Parse MinerU JSON structure output."""
        elements = []
        element_counter = start_idx

        def normalize_path(path_value: Optional[str]) -> Optional[str]:
            if not path_value:
                return None
            path = Path(path_value)
            if path.is_absolute():
                return str(path)
            if base_dir:
                return str(base_dir / path_value)
            return path_value

        def process_item(item: Dict, page_idx: int = 0):
            nonlocal element_counter

            item_type = item.get("type", "").lower()
            raw_content = item.get("text", "") or item.get("content", "")
            content = self._extract_text_from_rich_content(raw_content)
            bbox = item.get("bbox")

            if item_type in ["table", "表格"]:
                html = None
                caption = None
                image_path = item.get("img_path")
                if isinstance(raw_content, dict):
                    html = raw_content.get("html")
                    caption = self._extract_text_from_rich_content(raw_content.get("table_caption"))
                    image_path = image_path or (raw_content.get("image_source") or {}).get("path")
                if not html:
                    html = item.get("html")
                if html:
                    content = html
                elements.append(ParsedElement(
                    element_id=f"{doc_id}_table_{element_counter}",
                    doc_id=doc_id,
                    page_idx=page_idx,
                    element_type="table",
                    content=content,
                    bbox=bbox,
                    image_path=normalize_path(image_path),
                    metadata={
                        "html": html,
                        "rows": item.get("rows") or (raw_content.get("rows") if isinstance(raw_content, dict) else None),
                        "cols": item.get("cols") or (raw_content.get("cols") if isinstance(raw_content, dict) else None),
                        "caption": caption
                    }
                ))
                element_counter += 1

            elif item_type in ["image", "figure", "图片", "图"]:
                image_path = item.get("img_path")
                caption = None
                if isinstance(raw_content, dict):
                    image_path = image_path or (raw_content.get("image_source") or {}).get("path")
                    caption = self._extract_text_from_rich_content(raw_content.get("image_caption"))
                elements.append(ParsedElement(
                    element_id=f"{doc_id}_fig_{element_counter}",
                    doc_id=doc_id,
                    page_idx=page_idx,
                    element_type="figure",
                    content=caption or content,
                    bbox=bbox,
                    image_path=normalize_path(image_path),
                    metadata={"caption": caption or item.get("caption")}
                ))
                element_counter += 1

            elif item_type in ["equation", "formula", "公式"]:
                normalized = self._normalize_latex(item.get("latex", content))
                elements.append(ParsedElement(
                    element_id=f"{doc_id}_formula_{element_counter}",
                    doc_id=doc_id,
                    page_idx=page_idx,
                    element_type="formula",
                    content=normalized or content,
                    bbox=bbox,
                    metadata={"latex": normalized or content}
                ))
                element_counter += 1
            elif item_type in [
                "text",
                "title",
                "header",
                "footer",
                "page_footnote",
                "aside_text",
                "list",
                "paragraph"
            ]:
                if content:
                    elements.append(ParsedElement(
                        element_id=f"{doc_id}_text_{element_counter}",
                        doc_id=doc_id,
                        page_idx=page_idx,
                        element_type="text",
                        content=content,
                        bbox=bbox
                    ))
                    element_counter += 1

        # Handle different JSON structures
        if isinstance(structure, list):
            if structure and isinstance(structure[0], list):
                for page_idx, page in enumerate(structure):
                    for item in page:
                        if isinstance(item, dict):
                            process_item(item, page_idx)
            else:
                for idx, item in enumerate(structure):
                    if isinstance(item, dict):
                        page_idx = item.get("page_idx", item.get("page", idx))
                        process_item(item, page_idx)
        elif isinstance(structure, dict):
            # Handle pages structure
            pages = structure.get("pages", structure.get("content", []))
            for page_idx, page in enumerate(pages):
                if isinstance(page, dict):
                    blocks = page.get("blocks", page.get("elements", [page]))
                    for block in blocks:
                        process_item(block, page_idx)

        return elements

    def _count_pages(self, output_dir: Path) -> int:
        """Count total pages from output."""
        # Try to find page count from various sources
        for json_file in output_dir.rglob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if "total_pages" in data:
                        return data["total_pages"]
                    if "pages" in data:
                        return len(data["pages"])
            except Exception:
                continue
        return 0

    def _extract_page_from_filename(self, filename: str) -> int:
        """Extract page number from filename like 'doc_p5_img.png'."""
        match = re.search(r'_p(\d+)_', filename)
        if match:
            return int(match.group(1))
        match = re.search(r'page[_-]?(\d+)', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0

    def parse_batch(
        self,
        pdf_paths: List[str],
        progress_callback: callable = None
    ) -> List[ParsedDocument]:
        """
        Parse multiple PDFs in parallel using multiple GPUs.

        Args:
            pdf_paths: List of PDF file paths
            progress_callback: Optional callback for progress updates

        Returns:
            List of ParsedDocument results
        """
        results = []

        # Distribute work across GPUs
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            for idx, pdf_path in enumerate(pdf_paths):
                device_id = idx % len(self.devices)
                future = executor.submit(
                    self.parse_single,
                    pdf_path,
                    None,
                    device_id
                )
                futures[future] = pdf_path

            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if progress_callback:
                        progress_callback(len(results), len(pdf_paths), result)
                except Exception as e:
                    results.append(ParsedDocument(
                        doc_id=Path(pdf_path).stem,
                        pdf_path=pdf_path,
                        output_path="",
                        total_pages=0,
                        elements=[],
                        parse_time=0,
                        success=False,
                        error_message=str(e)
                    ))

        return results

    def save_structure(self, parsed_doc: ParsedDocument) -> Optional[Path]:
        """
        Save parsed document structure for reuse by other stages.

        Generates:
        - structure.json: serialized elements
        - formulas.md: LaTeX blocks (if formulas exist)
        - formulas.jsonl: structured LaTeX entries (if formulas exist)
        """
        if not parsed_doc.success:
            return None

        doc_dir = Path(parsed_doc.output_path) if parsed_doc.output_path else (self.output_dir / parsed_doc.doc_id)
        doc_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "doc_id": parsed_doc.doc_id,
            "pdf_path": parsed_doc.pdf_path,
            "output_path": parsed_doc.output_path,
            "total_pages": parsed_doc.total_pages,
            "elements": [
                {
                    "element_id": e.element_id,
                    "page_idx": e.page_idx,
                    "type": e.element_type,
                    "content": e.content,
                    "bbox": e.bbox,
                    "image_path": e.image_path,
                    "metadata": e.metadata
                }
                for e in parsed_doc.elements
            ]
        }

        structure_path = doc_dir / "structure.json"
        safe_json_dump(data, structure_path)

        self._save_formula_blocks(doc_dir, parsed_doc.elements)

        return structure_path

    def _save_formula_blocks(self, doc_dir: Path, elements: List[ParsedElement]) -> None:
        """Save formula elements as standalone LaTeX blocks."""
        formulas = []

        for elem in elements:
            if elem.element_type not in ["formula", "equation"]:
                continue

            latex = elem.metadata.get("latex") if elem.metadata else None
            latex = self._normalize_latex(latex or elem.content)
            if not latex:
                continue

            formulas.append({
                "doc_id": elem.doc_id,
                "page_idx": elem.page_idx,
                "element_id": elem.element_id,
                "latex": latex
            })

        if not formulas:
            return

        md_lines = []
        for item in formulas:
            md_lines.append("$$")
            md_lines.append(item["latex"])
            md_lines.append("$$")
            md_lines.append("")

        (doc_dir / "formulas.md").write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

        jsonl_path = doc_dir / "formulas.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in formulas:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def get_parser_stats(self) -> Dict:
        """Get parser statistics."""
        output_dirs = list(self.output_dir.iterdir())
        return {
            "total_parsed": len([d for d in output_dirs if d.is_dir()]),
            "output_dir": str(self.output_dir),
            "backend": self.backend,
            "devices": self.devices
        }
