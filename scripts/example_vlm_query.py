#!/usr/bin/env python3
"""
Example: Using VLM Query Generator with Qwen3-VL

This script demonstrates how to use the multimodal query generator
to create queries for different content types (tables, figures, formulas).

Usage:
    # First, start vLLM server (in another terminal):
    bash scripts/start_vllm_server.sh

    # Then run this example:
    python scripts/example_vlm_query.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generators.vlm_query_generator import (
    MultimodalQueryGenerator,
    VLMGeneratedQuery,
    QueryModalityType
)


def example_table_query():
    """Example: Generate queries for a table."""
    print("\n" + "="*60)
    print("Example 1: TABLE Query Generation")
    print("="*60)

    # Initialize generator (assumes vLLM server is running)
    generator = MultimodalQueryGenerator(
        backend="qwen_vllm",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        api_base="http://localhost:8000/v1"
    )

    # Example table content (markdown format)
    table_content = """
| Model | BLEU | ROUGE-L | F1 Score |
|-------|------|---------|----------|
| Baseline | 32.4 | 45.2 | 0.67 |
| Our Method | 38.7 | 52.1 | 0.75 |
| GPT-4 | 41.2 | 55.8 | 0.79 |
"""

    # Generate queries with explicit input
    queries = generator.generate_with_image(
        image_path=None,  # No image, text-only
        text_content=table_content,
        modal_type="table",
        num_queries=4,
        passage_id="example_table_1"
    )

    print(f"\nGenerated {len(queries)} queries:")
    for q in queries:
        print(f"\n  Query: {q.query_text}")
        print(f"  Type: {q.query_type}")
        print(f"  Requires Image: {q.requires_image}")
        print(f"  Difficulty: {q.difficulty:.2f}")

    return queries


def example_figure_query():
    """Example: Generate queries for a figure (with image)."""
    print("\n" + "="*60)
    print("Example 2: FIGURE Query Generation (with image)")
    print("="*60)

    generator = MultimodalQueryGenerator(
        backend="qwen_vllm",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        api_base="http://localhost:8000/v1"
    )

    # If you have an actual figure image, use its path
    # For demo, we'll use caption only
    figure_caption = """
Figure 3: Comparison of model performance across different datasets.
The blue line shows our method, red shows baseline, green shows GPT-4.
X-axis represents training epochs, Y-axis shows accuracy (%).
"""

    queries = generator.generate_with_image(
        image_path=None,  # Replace with actual path: "/path/to/figure.png"
        text_content=figure_caption,
        modal_type="figure",
        num_queries=4,
        passage_id="example_figure_1"
    )

    print(f"\nGenerated {len(queries)} queries:")
    for q in queries:
        print(f"\n  Query: {q.query_text}")
        print(f"  Type: {q.query_type}")
        print(f"  Visual Grounding: {q.visual_grounding}")
        print(f"  Difficulty: {q.difficulty:.2f}")

    return queries


def example_formula_query():
    """Example: Generate queries for a formula."""
    print("\n" + "="*60)
    print("Example 3: FORMULA Query Generation")
    print("="*60)

    generator = MultimodalQueryGenerator(
        backend="qwen_vllm",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        api_base="http://localhost:8000/v1"
    )

    # LaTeX formula
    formula_content = r"""
$$
\mathcal{L}_{contrastive} = -\log \frac{\exp(sim(q, p^+) / \tau)}{\sum_{i=1}^{N} \exp(sim(q, p_i) / \tau)}
$$

This is the InfoNCE loss function for contrastive learning, where:
- q is the query embedding
- p^+ is the positive passage embedding
- p_i includes both positive and negative passages
- tau is the temperature parameter
"""

    queries = generator.generate_with_image(
        image_path=None,
        text_content=formula_content,
        modal_type="formula",
        num_queries=3,
        passage_id="example_formula_1"
    )

    print(f"\nGenerated {len(queries)} queries:")
    for q in queries:
        print(f"\n  Query: {q.query_text}")
        print(f"  Type: {q.query_type}")
        print(f"  Difficulty: {q.difficulty:.2f}")

    return queries


def example_cross_modal_query():
    """Example: Generate cross-modal queries."""
    print("\n" + "="*60)
    print("Example 4: CROSS-MODAL Query Generation")
    print("="*60)

    generator = MultimodalQueryGenerator(
        backend="qwen_vllm",
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        api_base="http://localhost:8000/v1"
    )

    # Create pseudo-passages for different modalities
    class PseudoPassage:
        def __init__(self, modal_type, content, passage_id, doc_id, image_path=None):
            self.modal_type = modal_type
            self.content = content
            self.passage_id = passage_id
            self.doc_id = doc_id
            self.image_path = image_path
            self.context = None

    passages = [
        PseudoPassage(
            modal_type="table",
            content="| Method | Accuracy |\n|--------|----------|\n| Ours | 89.5% |",
            passage_id="table_1",
            doc_id="doc_1"
        ),
        PseudoPassage(
            modal_type="figure",
            content="Figure shows accuracy curve reaching 89.5% at epoch 50",
            passage_id="figure_1",
            doc_id="doc_1"
        ),
        PseudoPassage(
            modal_type="text",
            content="Our method achieves state-of-the-art performance on the benchmark...",
            passage_id="text_1",
            doc_id="doc_1"
        )
    ]

    queries = generator.generate_cross_modal_queries(passages, num_queries=2)

    print(f"\nGenerated {len(queries)} cross-modal queries:")
    for q in queries:
        print(f"\n  Query: {q.query_text}")
        print(f"  Query Modality: {q.query_modality.value}")
        print(f"  Difficulty: {q.difficulty:.2f}")

    return queries


def demo_contrastive_triplet():
    """Show how queries become contrastive learning triplets."""
    print("\n" + "="*60)
    print("Demo: Contrastive Learning Triplet Structure")
    print("="*60)

    triplet_example = {
        "query": "What is the F1 score achieved by 'Our Method' in the table?",
        "query_type": "visual_reading",
        "query_modality": "multimodal_grounded",
        "requires_image": True,

        "positive": {
            "content": "| Model | F1 Score |\n|-------|----------|\n| Our Method | 0.75 |",
            "image_path": "/path/to/table_image.png",
            "modal_type": "table",
            "metadata": {"rows": 3, "cols": 2}
        },

        "negatives": [
            {
                "content": "| Model | Accuracy |\n|-------|----------|\n| Baseline | 0.82 |",
                "modal_type": "table",
                "negative_type": "hard_same_modal",
                "reason": "Same table structure but different data"
            },
            {
                "content": "Figure showing F1 score trends over time",
                "modal_type": "figure",
                "negative_type": "cross_modal",
                "reason": "Related topic but wrong modality"
            },
            {
                "content": "Random unrelated passage...",
                "modal_type": "text",
                "negative_type": "random",
                "reason": "Easy negative for stable training"
            }
        ],

        "difficulty_score": 0.72
    }

    print("\nContrastive Triplet Example:")
    print(f"\n  QUERY: {triplet_example['query']}")
    print(f"  Type: {triplet_example['query_type']}")
    print(f"\n  POSITIVE (correct answer):")
    print(f"    Content: {triplet_example['positive']['content'][:50]}...")
    print(f"    Modality: {triplet_example['positive']['modal_type']}")
    print(f"\n  NEGATIVES (wrong answers):")
    for i, neg in enumerate(triplet_example['negatives'], 1):
        print(f"    {i}. [{neg['negative_type']}] {neg['content'][:40]}...")
        print(f"       Reason: {neg['reason']}")


def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("#  VLM Query Generator Examples")
    print("#  For Multimodal Contrastive Learning")
    print("#"*60)

    # Check if vLLM server is needed
    print("\n[INFO] These examples require a running vLLM server.")
    print("[INFO] Start with: bash scripts/start_vllm_server.sh")
    print("\nRunning demo (triplet structure)...")

    # Always show triplet demo (doesn't need server)
    demo_contrastive_triplet()

    # Optionally run live examples
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("\n[INFO] vLLM server detected, running live examples...")
            example_table_query()
            example_figure_query()
            example_formula_query()
            example_cross_modal_query()
        else:
            print("\n[INFO] vLLM server not responding, skipping live examples")
    except Exception:
        print("\n[INFO] vLLM server not available, skipping live examples")
        print("[INFO] To run live examples, start the vLLM server first")

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
