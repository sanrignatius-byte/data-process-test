#!/usr/bin/env python3
"""
Phase 1b (Chunk level): Compute features for chunk-level cross-document citation pairs.

For each GT pair (source_chunk, target_doc):
  - cite_pattern_score: citation markers in chunk text
  - title_match_score: target paper title in chunk text
  - section_type: from chunk's section_title
  - position: chunk_idx / total_chunks
  - chunk_size: word_count
  - text_sim: embedding cosine similarity (chunk text vs target doc abstract)
  - element_types: what element types are in this chunk (from chunk→element mapping)

Output: data/04_xdoc_citation/features_chunk_train.npz
"""

import json
import re
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CITE_PATTERNS = [
    re.compile(r'\([^)]*\b(?:[A-Z][a-z]+\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))[^)]*\b(?:19|20)\d{2}[a-z]?[^)]*\)'),
    re.compile(r'\[\d+(?:,\s*\d+)*\]'),
    re.compile(r'[A-Z][a-z]+\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+)\s*\(\s*(?:19|20)\d{2}[a-z]?\s*\)'),
    re.compile(r'\b(?:19|20)\d{2}[a-z]?\b'),
]

SECTION_KEYWORDS = {
    'introduction': ['introduction', 'intro', 'background', 'motivation'],
    'related_work': ['related work', 'related works', 'prior work', 'previous work'],
    'method': ['method', 'approach', 'proposed', 'our model', 'architecture'],
    'experiment': ['experiment', 'evaluation', 'results', 'implementation'],
    'conclusion': ['conclusion', 'discussion', 'summary', 'future work'],
}

def compute_cite_pattern_score(text: str) -> float:
    if not text: return 0.0
    count = sum(len(p.findall(text)) for p in CITE_PATTERNS)
    return min(count / max(len(text.split()), 1) * 20, 1.0)

def compute_title_match_score(text: str, bib_title: str) -> float:
    if not bib_title or not text: return 0.0
    title_clean = re.sub(r'\{[^}]*\}', '', bib_title)
    title_clean = re.sub(r'\s+', ' ', title_clean).strip().lower()
    text_lower = text.lower()
    if len(title_clean) > 20:
        score = SequenceMatcher(None, title_clean[:100], text_lower).ratio()
        if score > 0.3: return min(score * 2, 1.0)
    title_words = set(w for w in title_clean.split() if len(w) >= 4 and w not in
                      {'this', 'that', 'with', 'from', 'using', 'based'})
    if len(title_words) >= 3:
        text_words = set(text_lower.split())
        return min(len(title_words & text_words) / len(title_words), 1.0)
    return 0.0

def classify_section(name: str) -> str:
    if not name: return 'unknown'
    for cat, kws in SECTION_KEYWORDS.items():
        for kw in kws:
            if kw in name.lower():
                return cat
    return 'body'

def load_embedding_model(model_name: str = None):
    from sentence_transformers import SentenceTransformer
    candidates = [model_name] if model_name else []
    candidates += [
        str(PROJECT_ROOT / 'models' / 'Qwen3-Embedding-4B'),
        '/projects/myyyx1/models/Qwen3-Embedding-4B',
        'all-MiniLM-L6-v2',
    ]
    for c in candidates:
        if not c: continue
        if Path(c).exists() or '/' not in c:
            print(f"Loading embedding model: {c}")
            return SentenceTransformer(c, trust_remote_code=True)
    raise FileNotFoundError(f"No embedding model found")

def compute_embeddings(texts, model, batch_size=32):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', default=str(PROJECT_ROOT / 'data' / '04_xdoc_citation' / 'gt_citation_chunks.jsonl'))
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'data' / '04_xdoc_citation'))
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2')
    parser.add_argument('--neg-ratio', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--no-embeddings', action='store_true')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GT pairs
    print(f"Loading GT chunk pairs from {args.input_file}")
    pairs = []
    with open(args.input_file) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} GT chunk pairs")
    if not pairs:
        print("ERROR: No GT pairs.")
        sys.exit(1)

    # Load chunk data for all source docs
    print("Loading chunk virtual nodes...")
    with open(PROJECT_ROOT / 'data' / '01_graphs' / 'chunk_virtual_nodes_v2.json') as f:
        chunk_data = json.load(f)
    chunk_docs = chunk_data['documents']

    # Load bib title index for target docs
    print("Loading LaTeX reference graph for target doc titles...")
    with open(PROJECT_ROOT / 'data' / '01_graphs' / 'latex_reference_graph_v2.json') as f:
        latex = json.load(f)

    source_docs = set(p['source_doc'] for p in pairs)
    target_docs = set(p['target_doc'] for p in pairs)
    print(f"Source docs: {len(source_docs)}, Target docs: {len(target_docs)}")

    # Target doc abstracts
    target_texts_map = {}
    for doc_id in target_docs:
        doc = latex['documents'].get(doc_id, {})
        meta = doc.get('metadata', {})
        target_texts_map[doc_id] = meta.get('title', '')
    # Also load from MinerU markdown for richer abstracts
    from extract_xdoc_citation_pairs import read_mineru_md
    for doc_id in target_docs:
        md = read_mineru_md(doc_id)
        if md:
            target_texts_map[doc_id] = md[:1000]

    # Chunk element types
    chunk_elem_types = defaultdict(set)
    with open(PROJECT_ROOT / 'data' / '01_graphs' / 'multimodal_elements_v2.json') as f:
        mm = json.load(f)
    for doc_id in source_docs:
        cur_chunk = None
        for elem_id, elem in mm.get('documents', {}).get(doc_id, {}).get('elements', {}).items():
            elem_type = elem.get('element_type', 'text')
            chunk_elem_types[doc_id].add(elem_type)

    # --- Compute non-embedding features ---
    print("\n=== Computing features ===")
    features_list = []
    labels = []
    pair_records = []

    # Build source_to_targets for negative mining
    source_to_targets = defaultdict(set)
    for p in pairs:
        source_to_targets[p['source_doc']].add(p['target_doc'])

    for p in pairs:
        text = p.get('chunk_text', '')
        section = p.get('section_title', '')
        word_count = p.get('word_count', 200)
        bib_title = p.get('bib_title', '')
        source_doc = p['source_doc']
        chunk_id = p.get('chunk_id', '')

        # Find chunk position
        chunks_in_doc = chunk_docs.get(source_doc, {}).get('nodes', {})
        chunk_idx = 0
        total_chunks = max(len(chunks_in_doc), 1)
        if chunk_id in chunks_in_doc:
            chunk_idx = chunks_in_doc[chunk_id].get('chunk_idx', 0)
        position = chunk_idx / total_chunks

        feats = {
            'cite_pattern': compute_cite_pattern_score(text),
            'title_match': compute_title_match_score(text, bib_title),
            'section_cat': classify_section(section),
            'position': position,
            'chunk_size': word_count / 500.0,
        }
        features_list.append(feats)
        labels.append(1)
        pair_records.append({
            'source_doc': source_doc,
            'target_doc': p['target_doc'],
            'chunk_id': chunk_id,
            'chunk_text': text[:300],
            'label': 1,
        })

    # --- Negative sampling ---
    print(f"Positive: {len(features_list)}")
    target_doc_list = sorted(target_docs)
    neg_count = 0

    for p in pairs:
        source_doc = p['source_doc']
        text = p.get('chunk_text', '')
        section = p.get('section_title', '')
        word_count = p.get('word_count', 200)

        pos_targets = source_to_targets[source_doc]
        neg_candidates = [t for t in target_doc_list if t not in pos_targets and t != source_doc]

        if not neg_candidates:
            continue

        chunks_in_doc = chunk_docs.get(source_doc, {}).get('nodes', {})
        total_chunks = max(len(chunks_in_doc), 1)
        chunk_idx = chunks_in_doc.get(p['chunk_id'], {}).get('chunk_idx', 0)

        n_needed = min(args.neg_ratio, len(neg_candidates))
        neg_sampled = np.random.choice(neg_candidates, size=n_needed, replace=False)

        for neg_target in neg_sampled:
            feats = {
                'cite_pattern': compute_cite_pattern_score(text),
                'title_match': 0.0,
                'section_cat': classify_section(section),
                'position': chunk_idx / total_chunks,
                'chunk_size': word_count / 500.0,
            }
            features_list.append(feats)
            labels.append(0)
            pair_records.append({
                'source_doc': source_doc,
                'target_doc': neg_target,
                'chunk_id': p['chunk_id'],
                'chunk_text': text[:300],
                'label': 0,
            })
            neg_count += 1

    print(f"Negative: {neg_count}")

    # --- Compute embeddings ---
    if not args.no_embeddings:
        print("\n=== Computing chunk text embeddings ===")
        model = load_embedding_model(args.embedding_model)

        chunk_texts = [p.get('chunk_text', '') for p in pair_records]
        unique_target_docs = sorted(set(p['target_doc'] for p in pair_records))
        target_texts = [target_texts_map.get(d, '') for d in unique_target_docs]

        print(f"Encoding {len(chunk_texts)} chunk texts...")
        chunk_embs = compute_embeddings(chunk_texts, model, args.batch_size)
        print(f"Encoding {len(target_texts)} target abstracts...")
        target_embs = compute_embeddings(target_texts, model, args.batch_size)
        target_emb_map = {d: e for d, e in zip(unique_target_docs, target_embs)}

        # Cosine similarity
        chunk_embs = np.array(chunk_embs)
        all_target_embs = np.array([target_emb_map[p['target_doc']] for p in pair_records])
        text_sims = (chunk_embs * all_target_embs).sum(axis=1)

        for i, sim in enumerate(text_sims):
            features_list[i]['text_sim'] = float(sim)
    else:
        for feats in features_list:
            feats['text_sim'] = 0.0

    # --- Encode to matrix ---
    section_cats = sorted(set(f['section_cat'] for f in features_list))
    section_to_idx = {c: i for i, c in enumerate(section_cats)}
    n_section = len(section_cats)

    X = np.zeros((len(features_list), 5 + n_section))
    for i, feats in enumerate(features_list):
        X[i, 0] = feats['cite_pattern']
        X[i, 1] = feats['title_match']
        X[i, 2] = feats['position']
        X[i, 3] = feats['chunk_size']
        X[i, 4] = feats.get('text_sim', 0.0)
        # One-hot section
        sec_idx = section_to_idx.get(feats['section_cat'], 0)
        X[i, 5 + sec_idx] = 1.0

    y = np.array(labels, dtype=np.int32)
    feature_names = (['cite_pattern', 'title_match', 'position', 'chunk_size_norm',
                       'text_sim'] + [f'section_{c}' for c in section_cats])

    # Save
    np.savez(out_dir / 'features_chunk_train.npz', X=X, y=y)
    with open(out_dir / 'feature_chunk_metadata.json', 'w') as f:
        json.dump({
            'feature_names': feature_names,
            'n_samples': len(features_list),
            'n_positive': int(y.sum()),
            'n_negative': int(len(y) - y.sum()),
            'section_categories': section_cats,
            'neg_ratio': args.neg_ratio,
        }, f, indent=2)
    with open(out_dir / 'pair_chunk_records.jsonl', 'w') as f:
        for rec in pair_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"\nSaved chunk features to {out_dir}/features_chunk_train.npz")
    print(f"Feature names: {feature_names}")
    print(f"Pos={int(y.sum())}, Neg={int(len(y)-y.sum())}")
    print(f"Feature matrix: {X.shape}")

if __name__ == '__main__':
    main()
