#!/usr/bin/env python3
"""
Test script for FigureTextAssociator.

Runs on MinerU output and prints detailed results for inspection.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.linkers.figure_text_associator import FigureTextAssociator


def main():
    mineru_dir = project_root / "data" / "mineru_output"

    print(f"Processing MinerU output from: {mineru_dir}")
    print("=" * 70)

    associator = FigureTextAssociator(str(mineru_dir), context_window=3)

    # Process all documents
    results = associator.process_all_documents()

    # Print overall stats
    stats = associator.get_stats(results)
    print("\n📊 Overall Statistics:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print()

    # Print detailed results for a few documents
    for doc_id, pairs in sorted(results.items()):
        print(f"\n{'='*70}")
        print(f"📄 Document: {doc_id} — {len(pairs)} figure-text pairs")
        print(f"{'='*70}")

        for pair in pairs:
            print(f"\n  🖼  {pair.figure_id}")
            print(f"     Figure #{pair.figure_number or '?'} | Type: {pair.figure_type.value} | Quality: {pair.quality_score:.2f}")
            print(f"     Image: {pair.image_filename}")
            if pair.caption:
                print(f"     Caption: {pair.caption[:120]}{'...' if len(pair.caption) > 120 else ''}")
            if pair.sub_figures:
                print(f"     Sub-figures: {pair.sub_figures[:5]}")
            if pair.context_before:
                print(f"     Context before: {pair.context_before[:150]}{'...' if len(pair.context_before) > 150 else ''}")
            if pair.context_after:
                print(f"     Context after: {pair.context_after[:150]}{'...' if len(pair.context_after) > 150 else ''}")
            if pair.referring_paragraphs:
                print(f"     Referenced in {len(pair.referring_paragraphs)} paragraph(s):")
                for ref in pair.referring_paragraphs[:2]:
                    print(f"       - {ref[:120]}{'...' if len(ref) > 120 else ''}")
            if pair.metadata.get("panel_count", 1) > 1:
                print(f"     Multi-panel: {pair.metadata['panel_count']} panels")

    # Summary by quality tier
    all_pairs = [p for pairs in results.values() for p in pairs]
    high_quality = [p for p in all_pairs if p.quality_score >= 0.6]
    medium_quality = [p for p in all_pairs if 0.3 <= p.quality_score < 0.6]
    low_quality = [p for p in all_pairs if p.quality_score < 0.3]

    print(f"\n{'='*70}")
    print(f"📈 Quality Summary:")
    print(f"   High (≥0.6):   {len(high_quality)} pairs — ready for query generation")
    print(f"   Medium (0.3-0.6): {len(medium_quality)} pairs — may need MLLM verification")
    print(f"   Low (<0.3):    {len(low_quality)} pairs — likely decorative/low value")
    print(f"   Total:         {len(all_pairs)} pairs across {len(results)} documents")

    # Save results to JSON
    output_path = project_root / "data" / "figure_text_pairs.json"
    output_data = {}
    for doc_id, pairs in results.items():
        output_data[doc_id] = [p.to_dict() for p in pairs]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved to: {output_path}")


if __name__ == "__main__":
    main()
