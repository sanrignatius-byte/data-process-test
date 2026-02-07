#!/usr/bin/env python3
"""
M4 Cross-Document Query Generation Script

This script demonstrates how to use the enhanced M4QueryGenerator
with CrossDocumentLinker for generating multi-hop, multi-modal,
multi-document, and multi-turn queries.

Usage:
    # Basic usage with existing MinerU output
    python scripts/generate_m4_queries.py --input data/mineru_output --output data/m4_queries

    # With specific documents
    python scripts/generate_m4_queries.py --input data/mineru_output --doc-ids 2401.00001 2401.00002

    # Dry run (no LLM calls, just show entity/link statistics)
    python scripts/generate_m4_queries.py --input data/mineru_output --dry-run
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.linkers import CrossDocumentLinker, EvidenceChain
from src.generators import create_m4_generator, M4Query, M4QueryGenerator
from src.parsers.modal_extractor import ModalExtractor, Passage, ModalityType
from src.utils.file_utils import safe_json_load, safe_json_dump, write_jsonl


def load_passages_from_mineru_output(
    mineru_dir: Path,
    doc_ids: Optional[List[str]] = None,
    max_docs: int = 10
) -> List[Passage]:
    """
    Load passages from MinerU output directory.

    Args:
        mineru_dir: Path to MinerU output directory
        doc_ids: Specific document IDs to load (loads all if None)
        max_docs: Maximum number of documents to load

    Returns:
        List of Passage objects
    """
    extractor = ModalExtractor()
    all_passages = []

    # Find document directories
    doc_dirs = []
    for d in mineru_dir.iterdir():
        if not d.is_dir():
            continue
        if doc_ids and d.name not in doc_ids:
            continue
        doc_dirs.append(d)

    doc_dirs = doc_dirs[:max_docs]
    print(f"Loading passages from {len(doc_dirs)} documents...")

    for doc_dir in doc_dirs:
        doc_id = doc_dir.name

        # Try to load structure.json first
        structure_file = doc_dir / "structure.json"
        if structure_file.exists():
            data = safe_json_load(structure_file)
            if data and "elements" in data:
                # Convert to Passage objects
                for elem in data["elements"]:
                    modal_type_str = elem.get("type", "text")
                    try:
                        modal_type = ModalityType(modal_type_str)
                    except ValueError:
                        modal_type = ModalityType.TEXT

                    passage = Passage(
                        passage_id=elem.get("element_id", f"{doc_id}_{len(all_passages)}"),
                        doc_id=doc_id,
                        page_idx=elem.get("page_idx", 0),
                        modal_type=modal_type,
                        content=elem.get("content", ""),
                        image_path=elem.get("image_path"),
                        bbox=elem.get("bbox"),
                        quality_score=0.7,
                        context=elem.get("context")
                    )
                    all_passages.append(passage)
                continue

        # Fallback: try to load from markdown + content_list
        md_files = list((doc_dir / "auto").glob("*.md")) if (doc_dir / "auto").exists() else []
        content_list = doc_dir / "auto" / "content_list_v2.json"

        if content_list.exists():
            data = safe_json_load(content_list)
            if data:
                for idx, item in enumerate(data if isinstance(data, list) else []):
                    content = ""
                    modal_type = ModalityType.TEXT

                    if isinstance(item, dict):
                        content = item.get("text", item.get("content", ""))
                        item_type = item.get("type", "text")
                        if "table" in item_type.lower():
                            modal_type = ModalityType.TABLE
                        elif "figure" in item_type.lower() or "image" in item_type.lower():
                            modal_type = ModalityType.FIGURE
                        elif "formula" in item_type.lower() or "equation" in item_type.lower():
                            modal_type = ModalityType.FORMULA

                    if content and len(content) > 50:
                        passage = Passage(
                            passage_id=f"{doc_id}_elem_{idx}",
                            doc_id=doc_id,
                            page_idx=item.get("page_idx", 0) if isinstance(item, dict) else 0,
                            modal_type=modal_type,
                            content=content,
                            quality_score=0.6
                        )
                        all_passages.append(passage)

    print(f"Loaded {len(all_passages)} passages from {len(doc_dirs)} documents")
    return all_passages


def run_m4_generation(
    passages: List[Passage],
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-20250514",
    num_queries: int = 10,
    require_full_m4: bool = True,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Run M4 query generation pipeline.

    Args:
        passages: List of passages from multiple documents
        provider: LLM provider
        model: Model identifier
        num_queries: Target number of queries
        require_full_m4: Require all four M4 dimensions
        dry_run: If True, only compute statistics without LLM calls

    Returns:
        Dict with queries and statistics
    """
    # Step 1: Initialize linker and extract entities
    print("\n" + "="*60)
    print("Step 1: Entity Extraction and Cross-Document Linking")
    print("="*60)

    linker = CrossDocumentLinker(
        similarity_threshold=0.7,
        min_entity_frequency=3  # 只保留出现>=3次的实体，减少计算量
    )

    # Group passages by document
    passages_by_doc = {}
    for p in passages:
        if p.doc_id not in passages_by_doc:
            passages_by_doc[p.doc_id] = []
        passages_by_doc[p.doc_id].append(p)

    print(f"Documents: {len(passages_by_doc)}")
    print(f"Total passages: {len(passages)}")

    # Extract entities for each document
    for doc_id, doc_passages in passages_by_doc.items():
        entities = linker.build_document_entities(doc_passages, doc_id)
        print(f"  {doc_id}: {len(entities)} entities extracted")

    # Find cross-document links
    links = linker.find_cross_document_links()
    print(f"\nCross-document links found: {len(links)}")

    # Show some example links
    for link in links[:5]:
        print(f"  {link.entity_a.canonical_name} ({link.entity_a.doc_id}) "
              f"<-> {link.entity_b.canonical_name} ({link.entity_b.doc_id}) "
              f"[{link.link_type}, conf={link.confidence:.2f}]")

    # Get statistics
    stats = linker.get_entity_statistics()
    print(f"\nEntity Statistics:")
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Entities by type: {stats['entities_by_type']}")
    print(f"  Cross-doc links: {stats['total_cross_doc_links']}")

    if dry_run:
        print("\n[Dry run mode - skipping LLM calls]")
        return {
            "queries": [],
            "statistics": stats,
            "linkable_groups": len(linker.find_linkable_passage_groups(
                passages,
                require_multi_doc=require_full_m4,
                require_multi_modal=require_full_m4
            ))
        }

    # Step 2: Find linkable passage groups
    print("\n" + "="*60)
    print("Step 2: Finding Linkable Passage Groups")
    print("="*60)

    groups = linker.find_linkable_passage_groups(
        passages,
        require_multi_doc=require_full_m4,
        require_multi_modal=require_full_m4
    )
    print(f"Found {len(groups)} linkable passage groups")

    if not groups:
        print("No linkable groups found. Try:")
        print("  - Adding more documents")
        print("  - Relaxing require_full_m4=False")
        return {"queries": [], "statistics": stats}

    # Step 3: Generate M4 queries
    print("\n" + "="*60)
    print("Step 3: Generating M4 Queries")
    print("="*60)

    generator = create_m4_generator(provider=provider, model=model)
    generator.linker = linker

    queries = generator.generate_queries_for_passages(
        passages,
        num_queries=num_queries,
        require_full_m4=require_full_m4
    )

    print(f"Generated {len(queries)} valid M4 queries")

    # Step 4: Summarize results
    m4_satisfied = sum(1 for q in queries if q.satisfies_m4())
    print(f"\nQueries satisfying full M4: {m4_satisfied}/{len(queries)}")

    for i, q in enumerate(queries[:3]):
        print(f"\nQuery {i+1}:")
        print(f"  Turns: {q.turns}")
        print(f"  Answer: {q.answer[:100]}...")
        print(f"  Multi-hop: {q.is_multi_hop}, Multi-modal: {q.is_multi_modal}")
        print(f"  Multi-doc: {q.is_multi_doc}, Multi-turn: {q.is_multi_turn}")

    return {
        "queries": queries,
        "statistics": stats,
        "linkable_groups": len(groups)
    }


def save_queries(
    queries: List[M4Query],
    output_path: Path,
    statistics: Dict[str, Any]
) -> None:
    """Save generated queries to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    query_data = []
    for q in queries:
        query_data.append({
            "query_id": q.query_id,
            "query_type": q.query_type.value,
            "turns": q.turns,
            "answer": q.answer,
            "difficulty": q.difficulty,
            "evidence_chain": {
                "chain_id": q.evidence_chain.chain_id,
                "nodes": [
                    {
                        "node_id": n.node_id,
                        "passage_id": n.passage_id,
                        "doc_id": n.doc_id,
                        "modal_type": n.modal_type,
                        "content_snippet": n.content_snippet[:200]
                    }
                    for n in q.evidence_chain.nodes
                ],
                "reasoning_steps": q.evidence_chain.reasoning_steps,
                "modalities": list(q.evidence_chain.modalities_involved),
                "docs": list(q.evidence_chain.docs_involved)
            },
            "metadata": q.metadata,
            "validation": {
                "is_multi_hop": q.is_multi_hop,
                "is_multi_modal": q.is_multi_modal,
                "is_multi_doc": q.is_multi_doc,
                "is_multi_turn": q.is_multi_turn,
                "satisfies_full_m4": q.satisfies_m4()
            }
        })

    # Save as JSONL
    jsonl_path = output_path.with_suffix(".jsonl")
    write_jsonl(query_data, jsonl_path)
    print(f"Saved {len(query_data)} queries to {jsonl_path}")

    # Save statistics
    stats_path = output_path.parent / "m4_generation_stats.json"
    stats_data = {
        **statistics,
        "total_queries": len(queries),
        "full_m4_queries": sum(1 for q in queries if q.satisfies_m4()),
        "generated_at": datetime.now().isoformat()
    }
    safe_json_dump(stats_data, stats_path)
    print(f"Saved statistics to {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate M4 cross-document queries from MinerU output"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/mineru_output"),
        help="Path to MinerU output directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/m4_queries/queries"),
        help="Output path for generated queries"
    )
    parser.add_argument(
        "--doc-ids",
        nargs="+",
        help="Specific document IDs to process"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=10,
        help="Maximum number of documents to load"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=10,
        help="Target number of queries to generate"
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider"
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model identifier"
    )
    parser.add_argument(
        "--relaxed",
        action="store_true",
        help="Relax M4 requirements (don't require all four dimensions)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute entity/link statistics, no LLM calls"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input directory {args.input} does not exist")
        sys.exit(1)

    # Load passages
    passages = load_passages_from_mineru_output(
        args.input,
        doc_ids=args.doc_ids,
        max_docs=args.max_docs
    )

    if len(passages) < 10:
        print(f"Warning: Only {len(passages)} passages loaded. Consider adding more documents.")

    # Check document diversity
    doc_ids = set(p.doc_id for p in passages)
    if len(doc_ids) < 2:
        print("Error: Need passages from at least 2 documents for cross-document queries")
        sys.exit(1)

    # Run generation
    results = run_m4_generation(
        passages,
        provider=args.provider,
        model=args.model,
        num_queries=args.num_queries,
        require_full_m4=not args.relaxed,
        dry_run=args.dry_run
    )

    # Save results
    if results["queries"]:
        save_queries(results["queries"], args.output, results["statistics"])
    else:
        print("\nNo queries generated. Statistics saved for analysis.")
        if not args.dry_run:
            stats_path = args.output.parent / "m4_generation_stats.json"
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            safe_json_dump(results["statistics"], stats_path)


if __name__ == "__main__":
    main()
