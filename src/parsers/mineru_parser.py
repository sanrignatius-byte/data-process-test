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
        timeout: int = 300
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
        # Support both old format (content_list.json, middle.json) and new format (structure.json)
        json_paths = [
            # MinerU 新版格式 (优先)
            actual_output / "auto" / "structure.json",
            actual_output / "structure.json",
            # MinerU 旧版格式
            actual_output / "auto" / "content_list.json",
            actual_output / "content_list.json",
            actual_output / "auto" / "middle.json",
            actual_output / "middle.json",
            # model.json 也可能包含结构信息
            actual_output / "auto" / "model.json",
            actual_output / "model.json",
        ]

        for json_path in json_paths:
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        structure = json.load(f)

                    # 根据文件名选择解析方法
                    if "structure" in json_path.name:
                        additional = self._parse_structure_json(structure, doc_id, element_counter, actual_output)
                    else:
                        additional = self._parse_json_structure(structure, doc_id, element_counter)

                    # Only add elements not already captured
                    existing_ids = {e.element_id for e in elements}
                    for elem in additional:
                        if elem.element_id not in existing_ids:
                            elements.append(elem)
                            element_counter += 1
                except Exception as e:
                    print(f"Warning: Failed to parse {json_path}: {e}")
                break

        # 解析 formula.md 文件 (MinerU 新版单独的公式文件)
        formula_md_paths = [
            actual_output / "auto" / "formula.md",
            actual_output / "formula.md",
        ]
        for formula_path in formula_md_paths:
            if formula_path.exists():
                try:
                    formula_elements = self._parse_formula_md(formula_path, doc_id, element_counter)
                    existing_formulas = {e.content for e in elements if e.element_type == "formula"}
                    for elem in formula_elements:
                        if elem.content not in existing_formulas:
                            elements.append(elem)
                            element_counter += 1
                except Exception as e:
                    print(f"Warning: Failed to parse formula.md: {e}")
                break

        return elements

    def _parse_structure_json(
        self,
        structure: Any,
        doc_id: str,
        start_idx: int = 0,
        output_dir: Optional[Path] = None
    ) -> List[ParsedElement]:
        """
        Parse MinerU 新版 structure.json 格式.

        新版格式通常包含:
        - pdf_info: PDF元信息
        - pages: 按页面组织的内容块列表
        每个块包含: type, bbox, text/content, img_path 等
        """
        elements = []
        element_counter = start_idx

        def get_content(block: Dict) -> str:
            """从块中提取内容"""
            return block.get("text", "") or block.get("content", "") or block.get("latex", "") or ""

        def process_block(block: Dict, page_idx: int = 0):
            nonlocal element_counter

            block_type = block.get("type", "").lower()
            category = block.get("category", "").lower()

            # 统一类型判断
            effective_type = block_type or category

            content = get_content(block)
            bbox = block.get("bbox") or block.get("poly")

            # 处理表格
            if effective_type in ["table", "表格"] or block.get("is_table"):
                html_content = block.get("html", "")
                latex_content = block.get("latex", "")
                elements.append(ParsedElement(
                    element_id=f"{doc_id}_table_{element_counter}",
                    doc_id=doc_id,
                    page_idx=page_idx,
                    element_type="table",
                    content=content or html_content or latex_content,
                    bbox=bbox,
                    metadata={
                        "html": html_content,
                        "latex": latex_content,
                        "rows": block.get("rows"),
                        "cols": block.get("cols"),
                        "table_body": block.get("table_body")
                    }
                ))
                element_counter += 1

            # 处理图片/图表
            elif effective_type in ["image", "figure", "图片", "图", "image_body"]:
                img_path = block.get("img_path") or block.get("image_path")
                if img_path and output_dir:
                    # 转换为绝对路径
                    if not Path(img_path).is_absolute():
                        img_path = str(output_dir / img_path)

                elements.append(ParsedElement(
                    element_id=f"{doc_id}_fig_{element_counter}",
                    doc_id=doc_id,
                    page_idx=page_idx,
                    element_type="figure",
                    content=content or f"[Image: {img_path}]",
                    bbox=bbox,
                    image_path=img_path,
                    metadata={
                        "caption": block.get("caption"),
                        "img_caption": block.get("img_caption"),
                    }
                ))
                element_counter += 1

            # 处理公式 (行间公式)
            elif effective_type in ["equation", "formula", "公式", "interline_equation"]:
                latex = block.get("latex", "") or content
                elements.append(ParsedElement(
                    element_id=f"{doc_id}_formula_{element_counter}",
                    doc_id=doc_id,
                    page_idx=page_idx,
                    element_type="formula",
                    content=latex,
                    bbox=bbox,
                    metadata={
                        "latex": latex,
                        "equation_type": "block"
                    }
                ))
                element_counter += 1

            # 处理文本块
            elif effective_type in ["text", "paragraph", "文本", "text_block"] and content:
                if len(content.strip()) > 30:
                    elements.append(ParsedElement(
                        element_id=f"{doc_id}_text_{element_counter}",
                        doc_id=doc_id,
                        page_idx=page_idx,
                        element_type="text",
                        content=content,
                        bbox=bbox
                    ))
                    element_counter += 1

            # 处理标题 (也作为文本)
            elif effective_type in ["title", "heading", "标题"] and content:
                elements.append(ParsedElement(
                    element_id=f"{doc_id}_text_{element_counter}",
                    doc_id=doc_id,
                    page_idx=page_idx,
                    element_type="text",
                    content=content,
                    bbox=bbox,
                    metadata={"is_title": True}
                ))
                element_counter += 1

        # 处理不同的JSON结构
        if isinstance(structure, dict):
            # 检查是否有 pages 结构 (新版常见)
            if "pages" in structure:
                for page_idx, page in enumerate(structure["pages"]):
                    blocks = page.get("blocks", []) or page.get("items", []) or page.get("elements", [])
                    for block in blocks:
                        process_block(block, page_idx)

            # 检查是否有 pdf_info (另一种新版结构)
            elif "pdf_info" in structure:
                pdf_info = structure["pdf_info"]
                for page_idx, page in enumerate(pdf_info):
                    if isinstance(page, dict):
                        # 处理各种类型的预排序块
                        for key in ["preproc_blocks", "blocks", "layout_dets"]:
                            if key in page:
                                for block in page[key]:
                                    process_block(block, page_idx)

            # 直接是块列表
            elif "blocks" in structure:
                for block in structure["blocks"]:
                    page_idx = block.get("page_idx", block.get("page", 0))
                    process_block(block, page_idx)

            # 其他顶层键可能直接包含元素
            else:
                for key, value in structure.items():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                page_idx = item.get("page_idx", item.get("page", 0))
                                process_block(item, page_idx)

        elif isinstance(structure, list):
            for idx, item in enumerate(structure):
                if isinstance(item, dict):
                    page_idx = item.get("page_idx", item.get("page", 0))
                    process_block(item, page_idx)

        return elements

    def _parse_formula_md(
        self,
        formula_path: Path,
        doc_id: str,
        start_idx: int = 0
    ) -> List[ParsedElement]:
        """
        解析 MinerU 新版生成的 formula.md 文件.

        该文件通常包含文档中所有提取的公式，格式为 LaTeX。
        """
        elements = []
        element_counter = start_idx

        with open(formula_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取所有公式块
        # 模式1: $$ ... $$
        block_formulas = re.findall(r'\$\$(.*?)\$\$', content, re.DOTALL)

        # 模式2: \[ ... \]
        bracket_formulas = re.findall(r'\\\[(.*?)\\\]', content, re.DOTALL)

        all_formulas = block_formulas + bracket_formulas

        for formula in all_formulas:
            formula = formula.strip()
            if formula and len(formula) > 5:  # 过滤太短的公式
                elements.append(ParsedElement(
                    element_id=f"{doc_id}_formula_{element_counter}",
                    doc_id=doc_id,
                    page_idx=0,  # formula.md 通常不包含页码信息
                    element_type="formula",
                    content=f"$${formula}$$",
                    metadata={
                        "latex": formula,
                        "source": "formula.md"
                    }
                ))
                element_counter += 1

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
                    elements.append(ParsedElement(
                        element_id=f"{doc_id}_formula_{element_counter}",
                        doc_id=doc_id,
                        page_idx=current_page,
                        element_type="formula",
                        content=formula_text,
                        metadata={"latex": formula_text}
                    ))
                    element_counter += 1
                    in_formula = False
                    formula_content = []
                continue

            if in_formula:
                formula_content.append(line)
                if line.strip().endswith('$$') or line.strip().endswith('\\]'):
                    formula_text = '\n'.join(formula_content).strip()
                    elements.append(ParsedElement(
                        element_id=f"{doc_id}_formula_{element_counter}",
                        doc_id=doc_id,
                        page_idx=current_page,
                        element_type="formula",
                        content=formula_text,
                        metadata={"latex": formula_text}
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
        start_idx: int = 0
    ) -> List[ParsedElement]:
        """Parse MinerU JSON structure output."""
        elements = []
        element_counter = start_idx

        def process_item(item: Dict, page_idx: int = 0):
            nonlocal element_counter

            item_type = item.get("type", "").lower()
            content = item.get("text", "") or item.get("content", "")
            bbox = item.get("bbox")

            if item_type in ["table", "表格"]:
                elements.append(ParsedElement(
                    element_id=f"{doc_id}_table_{element_counter}",
                    doc_id=doc_id,
                    page_idx=page_idx,
                    element_type="table",
                    content=content,
                    bbox=bbox,
                    metadata={
                        "html": item.get("html"),
                        "rows": item.get("rows"),
                        "cols": item.get("cols")
                    }
                ))
                element_counter += 1

            elif item_type in ["image", "figure", "图片", "图"]:
                elements.append(ParsedElement(
                    element_id=f"{doc_id}_fig_{element_counter}",
                    doc_id=doc_id,
                    page_idx=page_idx,
                    element_type="figure",
                    content=content,
                    bbox=bbox,
                    image_path=item.get("img_path"),
                    metadata={"caption": item.get("caption")}
                ))
                element_counter += 1

            elif item_type in ["equation", "formula", "公式"]:
                elements.append(ParsedElement(
                    element_id=f"{doc_id}_formula_{element_counter}",
                    doc_id=doc_id,
                    page_idx=page_idx,
                    element_type="formula",
                    content=content,
                    bbox=bbox,
                    metadata={"latex": item.get("latex", content)}
                ))
                element_counter += 1

        # Handle different JSON structures
        if isinstance(structure, list):
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

    def get_parser_stats(self) -> Dict:
        """Get parser statistics."""
        output_dirs = list(self.output_dir.iterdir())
        return {
            "total_parsed": len([d for d in output_dirs if d.is_dir()]),
            "output_dir": str(self.output_dir),
            "backend": self.backend,
            "devices": self.devices
        }
