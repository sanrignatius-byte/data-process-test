#!/usr/bin/env python3
"""
Phase 1a: Extract GT cross-document citation pairs from LaTeX reference graph
and align citing contexts to MinerU markdown passages.

Output: data/04_xdoc_citation/gt_citation_pairs.jsonl
  Each line: {source_doc, target_doc, cite_context, source_section,
              mineru_passage_text, passage_start_char, passage_end_char,
              bib_key, bib_title, bib_authors, match_score}
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

# --- Arxiv ID extraction ---
ARXIV_RE = re.compile(
    r'(?:arxiv|arXiv)\s*[:\#]?\s*(\d{4}\.\d{4,5}(?:v\d+)?)',
)
BARE_ARXIV_RE = re.compile(r'\b(\d{4}\.\d{4,5})\b')

def extract_arxiv_id(text: str) -> str | None:
    """Extract an arxiv ID from a bib entry field."""
    m = ARXIV_RE.search(text)
    if m:
        raw = m.group(1)
        # strip version suffix
        return re.sub(r'v\d+$', '', raw)
    return None

def extract_arxiv_ids_from_bib(bib_entry: dict) -> list[str]:
    """Try to find arxiv IDs in all bib fields."""
    ids = []
    for field in ['note', 'url', 'eprint', 'title', 'journal', 'booktitle']:
        val = str(bib_entry.get(field, ''))
        aid = extract_arxiv_id(val)
        if aid:
            ids.append(aid)
    return ids

# --- MinerU markdown reading ---
SECTION_RE = re.compile(r'^#{1,4}\s+(.+)$', re.MULTILINE)

def read_mineru_md(doc_id: str) -> str | None:
    """Read the MinerU markdown output for a document."""
    for subdir in ['auto', 'hybrid_auto']:
        path = PROJECT_ROOT / 'data' / '00_raw' / 'mineru_output' / doc_id / doc_id / subdir / f'{doc_id}.md'
        if path.exists():
            return path.read_text(encoding='utf-8')
    return None

def split_md_into_passages(md_text: str, min_chars: int = 100) -> list[dict]:
    """Split markdown into passages with section context."""
    # Split by double newline to get paragraphs
    paragraphs = re.split(r'\n\n+', md_text)
    passages = []
    current_section = 'preamble'
    char_pos = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            char_pos += 2  # account for newlines
            continue

        # Check if this is a section header
        section_match = SECTION_RE.match(para)
        if section_match and len(para) < 120:
            current_section = section_match.group(1).strip()
            char_pos += len(para) + 2
            continue

        # Skip HTML tables for passage matching
        if para.startswith('<table>') or para.startswith('<tr>'):
            char_pos += len(para) + 2
            continue

        # Clean up: remove image references
        clean = re.sub(r'!\[.*?\]\(.*?\)', '', para).strip()
        if len(clean) >= min_chars:
            passages.append({
                'text': clean,
                'section': current_section,
                'char_start': char_pos,
                'char_end': char_pos + len(para),
                'raw': para,
            })
        char_pos += len(para) + 2

    return passages

def find_best_passage(cite_context: str, passages: list[dict]) -> dict | None:
    """Find the passage that best matches the cite context via text overlap."""
    # Normalize context: remove LaTeX commands
    context_clean = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})*', '', cite_context)
    context_clean = re.sub(r'[~\\]', ' ', context_clean)
    context_clean = ' '.join(context_clean.split())

    if len(context_clean) < 30:
        return None

    best_score = 0.0
    best_passage = None
    for p in passages:
        score = SequenceMatcher(None, context_clean.lower(), p['text'].lower()).ratio()
        if score > best_score:
            best_score = score
            best_passage = p

    if best_score >= 0.15 and best_passage:
        return {**best_passage, 'match_score': round(best_score, 3)}
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'data' / '04_xdoc_citation'))
    parser.add_argument('--min-match-score', type=float, default=0.15)
    parser.add_argument('--max-pairs', type=int, default=0, help='Cap positive pairs (0=unlimited)')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load LaTeX reference graph
    latex_graph_path = PROJECT_ROOT / 'data' / '01_graphs' / 'latex_reference_graph_v2.json'
    print(f"Loading LaTeX reference graph from {latex_graph_path}")
    with open(latex_graph_path) as f:
        latex = json.load(f)

    all_docs = latex['documents']
    all_doc_ids = set(all_docs.keys())
    print(f"Total LaTeX docs: {len(all_doc_ids)}")

    # Phase 1: Extract GT cross-doc citation pairs
    gt_pairs = []  # list of dicts
    stats = {'docs_with_citations': 0, 'total_cite_refs': 0,
             'cross_doc_cites': 0, 'aligned_to_mineru': 0}

    for doc_id, doc in all_docs.items():
        bib = doc.get('bib', {})
        refs = doc.get('refs', [])
        cite_refs = [r for r in refs if r.get('ref_type') == 'cite']

        if not cite_refs:
            continue

        stats['docs_with_citations'] += 1
        stats['total_cite_refs'] += len(cite_refs)

        # Build bib_key -> arxiv_ids map
        bib_arxiv = {}
        for bib_key, bib_entry in bib.items():
            if isinstance(bib_entry, dict):
                ids = extract_arxiv_ids_from_bib(bib_entry)
                if ids:
                    bib_arxiv[bib_key] = ids

        # For each cite ref, check if it references a corpus doc
        for ref in cite_refs:
            target_key = ref.get('target_key', '')
            if target_key not in bib_arxiv:
                continue

            target_arxiv_ids = bib_arxiv[target_key]
            for target_id in target_arxiv_ids:
                if target_id in all_doc_ids and target_id != doc_id:
                    gt_pairs.append({
                        'source_doc': doc_id,
                        'target_doc': target_id,
                        'cite_context': ref.get('context', ''),
                        'line_no': ref.get('line_no', 0),
                        'bib_key': target_key,
                        'bib_title': bib.get(target_key, {}).get('title', '') if isinstance(bib.get(target_key), dict) else '',
                    })
                    stats['cross_doc_cites'] += 1
                    break  # one target per cite ref

        if args.max_pairs and stats['cross_doc_cites'] >= args.max_pairs:
            break

    print(f"Docs with citations: {stats['docs_with_citations']}")
    print(f"Total cite refs: {stats['total_cite_refs']}")
    print(f"Cross-doc citation pairs: {stats['cross_doc_cites']}")

    # Deduplicate pairs
    seen = set()
    unique_pairs = []
    for p in gt_pairs:
        key = (p['source_doc'], p['target_doc'], p['line_no'])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)
    print(f"Unique cross-doc pairs: {len(unique_pairs)}")

    # Phase 2: Align to MinerU passages
    print("\nAligning to MinerU passages...")
    aligned_pairs = []

    # Build doc_id -> passage list cache
    passage_cache = {}

    for i, pair in enumerate(unique_pairs):
        source_doc = pair['source_doc']

        if source_doc not in passage_cache:
            md_text = read_mineru_md(source_doc)
            if md_text:
                passage_cache[source_doc] = split_md_into_passages(md_text)
            else:
                passage_cache[source_doc] = []

        passages = passage_cache[source_doc]
        if not passages:
            continue

        best = find_best_passage(pair['cite_context'], passages)
        if best:
            pair['mineru_passage_text'] = best['text']
            pair['source_section'] = best['section']
            pair['passage_char_start'] = best['char_start']
            pair['passage_char_end'] = best['char_end']
            pair['match_score'] = best['match_score']
            aligned_pairs.append(pair)
            stats['aligned_to_mineru'] += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(unique_pairs)} pairs, {stats['aligned_to_mineru']} aligned")

    print(f"\nAligned to MinerU: {stats['aligned_to_mineru']} / {len(unique_pairs)}")

    # Save
    output_path = out_dir / 'gt_citation_pairs.jsonl'
    with open(output_path, 'w') as f:
        for p in aligned_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"Saved {len(aligned_pairs)} pairs to {output_path}")

    # Save stats
    stats_path = out_dir / 'extraction_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")

if __name__ == '__main__':
    main()
