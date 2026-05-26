#!/usr/bin/env python3
"""
Extract GT cross-document citation pairs at CHUNK level.

Key difference from extract_xdoc_citation_pairs.py:
  - Uses chunk_virtual_nodes_v2.json (42K chunks, pre-built, uniform)
  - Matches LaTeX cite context against chunk text directly
  - Each chunk already has section_title, word_count, paragraph_indices
  - Chunk→element mapping from crossdoc pipeline for element-level traceability

Output: data/04_xdoc_citation/gt_citation_chunks.jsonl
  Each line: {source_doc, target_doc, chunk_id, chunk_text, section_title,
              cite_context, bib_key, bib_title, match_score, element_ids}
"""

import json
import re
import sys
import os
import argparse
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARXIV_RE = re.compile(r'(?:arxiv|arXiv)\s*[:\#]?\s*(\d{4}\.\d{4,5}(?:v\d+)?)')

def extract_arxiv_id(text: str) -> str | None:
    m = ARXIV_RE.search(text)
    if m:
        return re.sub(r'v\d+$', '', m.group(1))
    return None

def extract_arxiv_ids_from_bib(bib_entry: dict) -> list[str]:
    ids = []
    for field in ['note', 'url', 'eprint', 'title', 'journal', 'booktitle']:
        val = str(bib_entry.get(field, ''))
        aid = extract_arxiv_id(val)
        if aid:
            ids.append(aid)
    return ids

def find_best_chunk(cite_context: str, chunks: dict) -> tuple[str | None, dict | None, float]:
    """Find the chunk that best matches the cite context via text overlap."""
    context_clean = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})*', '', cite_context)
    context_clean = re.sub(r'[~\\]', ' ', context_clean)
    context_clean = ' '.join(context_clean.split())
    if len(context_clean) < 30:
        return None, None, 0.0

    best_score = 0.0
    best_id, best_chunk = None, None
    for chunk_id, chunk in chunks.items():
        score = SequenceMatcher(None, context_clean.lower(),
                                chunk['text'][:500].lower()).ratio()
        if score > best_score:
            best_score = score
            best_id = chunk_id
            best_chunk = chunk
    return best_id, best_chunk, best_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'data' / '04_xdoc_citation'))
    parser.add_argument('--min-match-score', type=float, default=0.15)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load LaTeX reference graph
    print("Loading LaTeX reference graph...")
    with open(PROJECT_ROOT / 'data' / '01_graphs' / 'latex_reference_graph_v2.json') as f:
        latex = json.load(f)
    all_docs = latex['documents']
    all_doc_ids = set(all_docs.keys())

    # Load chunk virtual nodes
    print("Loading chunk virtual nodes...")
    with open(PROJECT_ROOT / 'data' / '01_graphs' / 'chunk_virtual_nodes_v2.json') as f:
        chunk_data = json.load(f)
    chunk_docs = chunk_data['documents']

    # Load chunk→element mapping (from crossdoc_gold57 + full build)
    chunk_to_elements = defaultdict(list)
    # Try to load from crossdoc_gold57 first
    gold57_path = PROJECT_ROOT / 'data' / '01_graphs' / 'crossdoc_gold57.json'
    if gold57_path.exists():
        with open(gold57_path) as f:
            g57 = json.load(f)
            for chunk_id, elem_ids in g57.get('chunk_contains_element', {}).items():
                chunk_to_elements[chunk_id].extend(elem_ids)

    overlap = set(all_doc_ids) & set(chunk_docs.keys())
    print(f"LaTeX docs: {len(all_doc_ids)}, Chunk docs: {len(chunk_docs)}, Overlap: {len(overlap)}")

    # Extract
    gt_pairs = []
    stats = defaultdict(int)

    for doc_id in sorted(overlap):
        doc = all_docs[doc_id]
        chunks = chunk_docs[doc_id].get('nodes', {})
        if not chunks:
            continue

        bib = doc.get('bib', {})
        refs = doc.get('refs', [])
        cite_refs = [r for r in refs if r.get('ref_type') == 'cite']
        if not cite_refs:
            continue

        stats['docs_with_cites'] += 1
        stats['total_cite_refs'] += len(cite_refs)

        # Build bib_key -> arxiv_ids
        bib_arxiv = {}
        for bib_key, bib_entry in bib.items():
            if isinstance(bib_entry, dict):
                ids = extract_arxiv_ids_from_bib(bib_entry)
                if ids:
                    bib_arxiv[bib_key] = ids

        for ref in cite_refs:
            target_key = ref.get('target_key', '')
            if target_key not in bib_arxiv:
                continue

            target_ids = bib_arxiv[target_key]
            for target_id in target_ids:
                if target_id in all_doc_ids and target_id != doc_id:
                    # Find best matching chunk
                    cite_context = ref.get('context', '')
                    chunk_id, chunk, score = find_best_chunk(cite_context, chunks)

                    if chunk_id and score >= args.min_match_score:
                        bib_entry = bib.get(target_key, {})
                        gt_pairs.append({
                            'source_doc': doc_id,
                            'target_doc': target_id,
                            'chunk_id': chunk_id,
                            'chunk_text': chunk['text'][:500],
                            'section_title': chunk.get('section_title', ''),
                            'word_count': chunk.get('word_count', 0),
                            'cite_context': cite_context,
                            'line_no': ref.get('line_no', 0),
                            'bib_key': target_key,
                            'bib_title': bib_entry.get('title', '') if isinstance(bib_entry, dict) else '',
                            'match_score': round(score, 3),
                            'element_ids': chunk_to_elements.get(chunk_id, []),
                        })
                        stats['cross_doc_chunk_matches'] += 1
                    break  # one target per cite

    print(f"\nDocs with citations: {stats['docs_with_cites']}")
    print(f"Total cite refs: {stats['total_cite_refs']}")
    print(f"Cross-doc chunk-level matches: {stats['cross_doc_chunk_matches']}")

    # Deduplicate by (source_doc, target_doc, chunk_id)
    seen = set()
    unique = []
    for p in gt_pairs:
        key = (p['source_doc'], p['target_doc'], p['chunk_id'])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    print(f"Unique chunk-level pairs: {len(unique)}")

    # Save
    output_path = out_dir / 'gt_citation_chunks.jsonl'
    with open(output_path, 'w') as f:
        for p in unique:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"Saved to {output_path}")

    stats_path = out_dir / 'chunk_extraction_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(dict(stats), f, indent=2)

if __name__ == '__main__':
    main()
