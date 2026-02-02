#!/usr/bin/env python3
"""
Script to regenerate queries only (for existing parsed documents).

Usage:
    python scripts/generate_queries_only.py --input ./data/mineru_output --output ./data/contrastive_data
"""

import argparse
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.file_utils import safe_json_load, write_jsonl
from src.parsers.modal_extractor import ModalExtractor, Passage
from src.generators.query_generator import MultimodalQueryGenerator
from src.samplers.negative_sampler import HardNegativeSampler
from tqdm import tqdm


def load_passages_from_parsed(parsed_dir: str) -> list:
    """Load passages from parsed document outputs."""
    parsed_path = Path(parsed_dir)
    passages = []

    extractor = ModalExtractor()

    for doc_dir in tqdm(list(parsed_path.iterdir()), desc="Loading parsed documents"):
        if not doc_dir.is_dir():
            continue

        structure_file = doc_dir / "structure.json"
        if not structure_file.exists():
            continue

        data = safe_json_load(structure_file)
        if not data:
            continue

        # Create pseudo-elements for extraction
        class PseudoElement:
            def __init__(self, elem_data, doc_id):
                self.element_id = elem_data.get("element_id", "")
                self.doc_id = doc_id
                self.page_idx = elem_data.get("page_idx", 0)
                self.element_type = elem_data.get("type", "text")
                self.content = elem_data.get("content", "")
                self.bbox = elem_data.get("bbox")
                self.image_path = elem_data.get("image_path")
                self.metadata = elem_data.get("metadata", {})

        elements = [
            PseudoElement(e, data.get("doc_id", doc_dir.name))
            for e in data.get("elements", [])
        ]

        doc_passages = extractor.extract_passages(
            elements=elements,
            doc_id=data.get("doc_id", doc_dir.name)
        )
        passages.extend(doc_passages)

    return passages


def main():
    parser = argparse.ArgumentParser(description="Regenerate queries for existing parsed documents")

    parser.add_argument(
        "--input",
        type=str,
        default="./data/mineru_output",
        help="Directory with parsed documents"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./data/contrastive_data",
        help="Output directory for dataset"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Configuration file"
    )

    parser.add_argument(
        "--queries-per-element",
        type=int,
        default=3,
        help="Number of queries per element"
    )

    args = parser.parse_args()

    # Load config
    config = Config(args.config)

    # Load passages
    print(f"Loading passages from {args.input}")
    passages = load_passages_from_parsed(args.input)
    print(f"Loaded {len(passages)} passages")

    if not passages:
        print("No passages found!")
        return

    # Initialize components
    qg_config = config.query_generation
    query_gen = MultimodalQueryGenerator(
        provider=qg_config.provider,
        model=qg_config.model,
        temperature=qg_config.temperature,
        max_tokens=qg_config.max_tokens,
        rate_limit=qg_config.rate_limit
    )

    neg_config = config.negative_sampling
    sampler = HardNegativeSampler(
        num_negatives=neg_config.num_negatives,
        strategy=neg_config.strategy,
        distribution=neg_config.distribution
    )

    # Generate queries
    print("Generating queries...")
    query_data = {}
    batch_size = qg_config.batch_size

    for i in tqdm(range(0, len(passages), batch_size), desc="Query generation"):
        batch = passages[i:i + batch_size]
        batch_results = query_gen.generate_batch(batch, num_queries=args.queries_per_element)
        query_data.update(batch_results)

    total_queries = sum(len(qs) for qs in query_data.values())
    print(f"Generated {total_queries} queries")

    # Construct triplets
    print("Constructing triplets...")
    passages_by_id = {p.passage_id: p for p in passages}
    triplets = sampler.construct_triplets(query_data, passages, passages_by_id)
    print(f"Created {len(triplets)} triplets")

    # Save output
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    train_split = int(len(triplets) * 0.9)
    train_triplets = triplets[:train_split]
    val_triplets = triplets[train_split:]

    write_jsonl(
        [t.to_training_format() for t in train_triplets],
        output_path / "contrastive_dataset_train.jsonl"
    )
    write_jsonl(
        [t.to_training_format() for t in val_triplets],
        output_path / "contrastive_dataset_val.jsonl"
    )

    print(f"\nDataset saved to {output_path}")
    print(f"Train: {len(train_triplets)} triplets")
    print(f"Validation: {len(val_triplets)} triplets")


if __name__ == "__main__":
    main()
