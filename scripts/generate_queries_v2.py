#!/usr/bin/env python3
"""
Generate contrastive learning triplets using content_list_v2.json format.
Output format: <query, positive_passage, negative_passage>

Supports: table (image), figure (image), formula (from md), text
"""

import argparse
import json
import os
import sys
import base64
import hashlib
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

import anthropic

# Import existing sampler
from src.samplers.negative_sampler import HardNegativeSampler, ContrastiveTriplet
from src.parsers.modal_extractor import Passage, ModalityType


def extract_text_from_content(content_obj) -> str:
    """Recursively extract text from content_list_v2 content object."""
    if isinstance(content_obj, str):
        return content_obj
    if isinstance(content_obj, dict):
        if "content" in content_obj:
            return extract_text_from_content(content_obj["content"])
        texts = []
        for key in ["paragraph_content", "title_content", "table_caption", "image_caption"]:
            if key in content_obj:
                for item in content_obj[key]:
                    texts.append(extract_text_from_content(item))
        return " ".join(texts)
    if isinstance(content_obj, list):
        return " ".join(extract_text_from_content(item) for item in content_obj)
    return ""


def extract_formulas_from_md(md_file: Path, doc_id: str) -> List[Passage]:
    """Extract block formulas ($$...$$) from markdown file with surrounding context."""
    passages = []

    if not md_file.exists():
        return passages

    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into lines for context extraction
    lines = content.split('\n')

    # Find all block equations ($$...$$)
    pattern = r'\$\$([\s\S]*?)\$\$'
    matches = list(re.finditer(pattern, content))

    for idx, match in enumerate(matches):
        formula_content = match.group(1).strip()

        # Skip very short formulas
        if len(formula_content) < 10:
            continue

        # Find line number of formula for context
        start_pos = match.start()
        line_num = content[:start_pos].count('\n')

        # Extract context: 2 lines before and 2 lines after
        context_start = max(0, line_num - 2)
        context_end = min(len(lines), line_num + 5)  # +5 to include formula lines
        context_lines = lines[context_start:context_end]

        # Clean context (remove empty lines and the formula itself)
        context_text = []
        for line in context_lines:
            line = line.strip()
            if line and not line.startswith('$$') and line != '$$':
                # Remove inline math for cleaner context
                clean_line = re.sub(r'\$[^$]+\$', '[math]', line)
                if clean_line and len(clean_line) > 10:
                    context_text.append(clean_line)

        context = ' '.join(context_text[:3])  # Take up to 3 context lines

        content_hash = hashlib.md5(formula_content[:50].encode()).hexdigest()[:8]

        passage = Passage(
            passage_id=f"{doc_id}_formula_{idx}_{content_hash}",
            doc_id=doc_id,
            page_idx=0,
            modal_type=ModalityType.FORMULA,
            content=formula_content,
            context=context if context else None,
            metadata={"latex": formula_content, "context": context}
        )
        passages.append(passage)

    return passages


def load_passages_from_content_list_v2(doc_dir: Path, doc_id: str) -> List[Passage]:
    """Load passages from content_list_v2.json format + formulas from md."""
    passages = []

    # Find content_list_v2.json
    content_files = list(doc_dir.rglob("*content_list_v2.json"))
    if not content_files:
        return passages

    content_file = content_files[0]
    base_dir = content_file.parent

    with open(content_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # data is a list of pages, each page is a list of elements
    for page_idx, page_elements in enumerate(data):
        for elem_idx, elem in enumerate(page_elements):
            elem_type = elem.get("type", "")
            content_obj = elem.get("content", {})
            bbox = elem.get("bbox")

            # Skip page numbers, headers, footers
            if elem_type in ["page_number", "page_aside_text", "page_header", "page_footer"]:
                continue

            passage = None

            if elem_type == "table":
                # Use image for tables
                image_source = content_obj.get("image_source", {})
                image_path = image_source.get("path", "")
                caption_text = extract_text_from_content(content_obj.get("table_caption", []))

                if image_path:
                    full_image_path = str(base_dir / image_path)
                    content_hash = hashlib.md5(image_path.encode()).hexdigest()[:8]

                    passage = Passage(
                        passage_id=f"{doc_id}_table_{page_idx}_{content_hash}",
                        doc_id=doc_id,
                        page_idx=page_idx,
                        modal_type=ModalityType.TABLE,
                        content=caption_text or f"Table on page {page_idx + 1}",
                        image_path=full_image_path,
                        bbox=bbox,
                        metadata={"caption": caption_text}
                    )

            elif elem_type == "image":
                # Figures
                image_source = content_obj.get("image_source", {})
                image_path = image_source.get("path", "")
                caption_text = extract_text_from_content(content_obj.get("image_caption", []))

                if image_path:
                    full_image_path = str(base_dir / image_path)
                    content_hash = hashlib.md5(image_path.encode()).hexdigest()[:8]

                    passage = Passage(
                        passage_id=f"{doc_id}_figure_{page_idx}_{content_hash}",
                        doc_id=doc_id,
                        page_idx=page_idx,
                        modal_type=ModalityType.FIGURE,
                        content=caption_text or f"Figure on page {page_idx + 1}",
                        image_path=full_image_path,
                        bbox=bbox,
                        metadata={"caption": caption_text}
                    )

            elif elem_type in ["paragraph", "text"]:
                text = extract_text_from_content(content_obj)
                # Filter short text
                if len(text.strip()) >= 100:
                    content_hash = hashlib.md5(text[:100].encode()).hexdigest()[:8]
                    passage = Passage(
                        passage_id=f"{doc_id}_text_{page_idx}_{content_hash}",
                        doc_id=doc_id,
                        page_idx=page_idx,
                        modal_type=ModalityType.TEXT,
                        content=text.strip(),
                        bbox=bbox
                    )

            if passage:
                passages.append(passage)

    # Extract formulas from md file
    md_files = list(doc_dir.rglob("*.md"))
    for md_file in md_files:
        formula_passages = extract_formulas_from_md(md_file, doc_id)
        passages.extend(formula_passages)

    return passages


def encode_image_base64(image_path: str) -> Optional[str]:
    """Encode image to base64."""
    try:
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def get_image_media_type(image_path: str) -> str:
    """Get media type from image path."""
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return media_types.get(ext, "image/jpeg")


@dataclass
class GeneratedQuery:
    """Generated query compatible with HardNegativeSampler."""
    query_id: str
    query_text: str
    query_type: str
    target_modality: str
    passage_id: str
    difficulty: float = 0.5


def generate_queries_for_passage(
    client: anthropic.Anthropic,
    passage: Passage,
    num_queries: int = 3
) -> List[GeneratedQuery]:
    """Generate queries for a single passage using Claude."""

    queries = []
    modal_type = passage.modal_type.value if hasattr(passage.modal_type, 'value') else passage.modal_type

    # Build prompt based on modality
    if modal_type == "table":
        prompt = f"""You are analyzing a table from a research paper.

Table caption: {passage.content}

Based on this table image, generate {num_queries} diverse questions that require reading the table to answer.

Requirements:
1. Include different types: factual lookup, comparative, computational
2. Questions should be specific and answerable from the table
3. Questions should be in English

Output as JSON: {{"questions": [{{"text": "...", "type": "factual|comparative|computational"}}]}}
Only output valid JSON."""

    elif modal_type == "figure":
        prompt = f"""You are analyzing a figure from a research paper.

Figure caption: {passage.content}

Based on this figure image, generate {num_queries} diverse questions.

Requirements:
1. Include types: descriptive, identification, interpretive
2. Questions should require understanding the figure
3. Questions should be in English

Output as JSON: {{"questions": [{{"text": "...", "type": "descriptive|identification|interpretive"}}]}}
Only output valid JSON."""

    elif modal_type == "formula":
        context_section = ""
        if passage.context:
            context_section = f"\nContext (surrounding text):\n{passage.context}\n"

        prompt = f"""You are analyzing a mathematical formula from a research paper.

Formula (LaTeX):
{passage.content}
{context_section}
Generate {num_queries} diverse questions about this formula.

Requirements:
1. Include types: semantic (what it calculates), variable (what variables mean), application (how to use)
2. Questions should test understanding of the formula
3. Use the context to make questions more specific if available
4. Questions should be in English

Output as JSON: {{"questions": [{{"text": "...", "type": "semantic|variable|application"}}]}}
Only output valid JSON."""

    else:  # text
        prompt = f"""You are analyzing a text passage from a research paper.

Text: {passage.content[:1500]}

Generate {num_queries} diverse questions that require reading this passage to answer.

Requirements:
1. Include types: factual, conceptual, inferential
2. Questions should not be answerable without the text
3. Questions should be in English

Output as JSON: {{"questions": [{{"text": "...", "type": "factual|conceptual|inferential"}}]}}
Only output valid JSON."""

    try:
        # Build message content
        content = []

        # Add image for tables and figures
        if passage.image_path and modal_type in ["table", "figure"]:
            image_data = encode_image_base64(passage.image_path)
            if image_data:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": get_image_media_type(passage.image_path),
                        "data": image_data
                    }
                })

        content.append({"type": "text", "text": prompt})

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": content}]
        )

        response_text = response.content[0].text.strip()

        # Parse JSON response
        if response_text.startswith("```"):
            match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
            if match:
                response_text = match.group(1).strip()

        data = json.loads(response_text)

        for idx, q in enumerate(data.get("questions", [])):
            query_text = q.get("text", "")
            query_type = q.get("type", "factual")

            if query_text:
                query_hash = hashlib.md5(query_text.encode()).hexdigest()[:8]
                queries.append(GeneratedQuery(
                    query_id=f"{passage.passage_id}_q{idx}_{query_hash}",
                    query_text=query_text,
                    query_type=query_type,
                    target_modality=modal_type,
                    passage_id=passage.passage_id,
                    difficulty=0.5
                ))

    except Exception as e:
        print(f"Error generating queries for {passage.passage_id}: {e}")

    return queries


def main():
    parser = argparse.ArgumentParser(description="Generate contrastive triplets using content_list_v2 format")
    parser.add_argument("--input", type=str, default="./data/mineru_output",
                        help="Input directory with parsed documents")
    parser.add_argument("--output", type=str, default="./data/queries_output",
                        help="Output directory")
    parser.add_argument("--num-docs", type=int, default=5,
                        help="Number of documents to process")
    parser.add_argument("--queries-per-element", type=int, default=3,
                        help="Queries per element")
    parser.add_argument("--max-passages-per-doc", type=int, default=10,
                        help="Max passages to process per document")
    parser.add_argument("--num-negatives", type=int, default=3,
                        help="Number of negatives per query")
    parser.add_argument("--negative-strategy", type=str, default="modal_mixed",
                        choices=["random", "modal_same", "modal_mixed", "semantic_hard"],
                        help="Negative sampling strategy")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Anthropic client
    client = anthropic.Anthropic()

    # Initialize negative sampler (using existing code)
    sampler = HardNegativeSampler(
        num_negatives=args.num_negatives,
        strategy=args.negative_strategy,
        distribution={
            "hard_same_modal": 0.6,
            "cross_modal": 0.3,
            "random": 0.1
        }
    )

    # Find document directories
    doc_dirs = [d for d in input_dir.iterdir() if d.is_dir()][:args.num_docs]

    print(f"Processing {len(doc_dirs)} documents...")

    all_triplets = []
    all_passages = []  # Global pool for negative sampling
    query_data = {}  # passage_id -> list of queries

    # First pass: load all passages
    print("\n=== Loading passages ===")
    for doc_dir in tqdm(doc_dirs, desc="Loading documents"):
        doc_id = doc_dir.name
        passages = load_passages_from_content_list_v2(doc_dir, doc_id)
        all_passages.extend(passages)

        # Show distribution
        dist = {}
        for p in passages:
            mt = p.modal_type.value if hasattr(p.modal_type, 'value') else p.modal_type
            dist[mt] = dist.get(mt, 0) + 1
        print(f"  {doc_id}: {len(passages)} passages - {dist}")

    print(f"\nTotal passages: {len(all_passages)}")
    passages_by_id = {p.passage_id: p for p in all_passages}

    # Second pass: generate queries (balanced sampling across modalities)
    print("\n=== Generating queries ===")
    for doc_dir in tqdm(doc_dirs, desc="Documents"):
        doc_id = doc_dir.name
        doc_passages = [p for p in all_passages if p.doc_id == doc_id]

        # Group passages by modality
        by_modal = {}
        for p in doc_passages:
            mt = p.modal_type.value if hasattr(p.modal_type, 'value') else p.modal_type
            if mt not in by_modal:
                by_modal[mt] = []
            by_modal[mt].append(p)

        # Balanced sampling: take from each modality
        passages_to_process = []
        per_modal = max(2, args.max_passages_per_doc // max(len(by_modal), 1))

        for modal_type in ["table", "figure", "formula", "text"]:
            modal_passages = by_modal.get(modal_type, [])
            passages_to_process.extend(modal_passages[:per_modal])

        # If still under limit, add more from largest groups
        remaining = args.max_passages_per_doc - len(passages_to_process)
        if remaining > 0:
            all_remaining = [p for p in doc_passages if p not in passages_to_process]
            passages_to_process.extend(all_remaining[:remaining])

        print(f"\n  {doc_id}: Processing {len(passages_to_process)} passages")
        proc_dist = {}
        for p in passages_to_process:
            mt = p.modal_type.value if hasattr(p.modal_type, 'value') else p.modal_type
            proc_dist[mt] = proc_dist.get(mt, 0) + 1
        print(f"    Selected: {proc_dist}")

        for passage in tqdm(passages_to_process, desc=f"  {doc_id}", leave=False):
            queries = generate_queries_for_passage(client, passage, args.queries_per_element)

            if queries:
                query_data[passage.passage_id] = queries

    total_queries = sum(len(qs) for qs in query_data.values())
    print(f"\nTotal queries generated: {total_queries}")

    # Third pass: construct triplets using HardNegativeSampler
    print("\n=== Constructing triplets ===")
    triplets = sampler.construct_triplets(query_data, all_passages, passages_by_id)

    print(f"Total triplets: {len(triplets)}")

    # Save results
    output_file = output_dir / "triplets_v2.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for triplet in triplets:
            f.write(json.dumps(triplet.to_training_format(), ensure_ascii=False) + "\n")

    print(f"\n生成完成!")
    print(f"Saved to: {output_file}")

    # Print statistics
    modal_stats = {}
    for t in triplets:
        mt = t.positive.get("modal_type", "unknown")
        modal_stats[mt] = modal_stats.get(mt, 0) + 1
    print(f"\nTriplets by modality: {modal_stats}")

    # Negative type distribution
    neg_stats = {}
    for t in triplets:
        for neg in t.negatives:
            nt = neg.get("negative_type", "unknown")
            neg_stats[nt] = neg_stats.get(nt, 0) + 1
    print(f"Negative types: {neg_stats}")


if __name__ == "__main__":
    main()
