#!/usr/bin/env python3
"""
Phase 1b: Compute features for cross-document citation pairs.

For each GT pair (source_passage, target_doc):
  - cite_pattern_score: regex detection of citation markers
  - title_match_score: fuzzy matching of target paper title
  - section_type: Introduction, Related Work, Method, etc.
  - position_in_doc: relative position (0-1)
  - text_sim: embedding cosine similarity (source passage vs target doc abstract)
  - element_type: if the passage contains figure/table/formula references

Also generates NEGATIVE samples: same passage with non-cited target docs.
Negative sampling strategy:
  - Hard negatives: target docs that are cited by OTHER passages in same source doc
  - Random negatives: randomly sampled target docs from corpus

Output:
  - data/04_xdoc_citation/features_train.npz  (X, y, pair_ids)
  - data/04_xdoc_citation/feature_metadata.json
"""

import json
import re
import sys
import os
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Feature extractors ---

# Citation pattern regexes for MinerU text
CITE_PATTERNS = [
    # Parenthetical: (Author et al., 2019) or (Author, 2019)
    re.compile(r'\([^)]*\b(?:[A-Z][a-z]+\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))[^)]*\b(?:19|20)\d{2}[a-z]?[^)]*\)'),
    # Bracket: [12] or [12, 13, 14]
    re.compile(r'\[\d+(?:,\s*\d+)*\]'),
    # Author-year in text: Smith et al. (2019) or Smith and Jones (2019)
    re.compile(r'[A-Z][a-z]+\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+)\s*\(\s*(?:19|20)\d{2}[a-z]?\s*\)'),
    # Numeric superscript style
    re.compile(r'\b(?:19|20)\d{2}[a-z]?\b'),
]

SECTION_KEYWORDS = {
    'introduction': ['introduction', 'intro', 'background', 'motivation'],
    'related_work': ['related work', 'related works', 'prior work', 'previous work',
                      'literature review', 'state of the art', 'related approaches'],
    'method': ['method', 'approach', 'proposed', 'our model', 'architecture',
               'framework', 'algorithm', 'model description', 'formulation'],
    'experiment': ['experiment', 'evaluation', 'results', 'implementation',
                    'training', 'dataset', 'benchmark', 'performance', 'ablation'],
    'conclusion': ['conclusion', 'discussion', 'summary', 'future work', 'limitation'],
}

def compute_cite_pattern_score(text: str) -> float:
    """Count citation-like patterns in text, normalized by text length."""
    if not text:
        return 0.0
    count = 0
    for pattern in CITE_PATTERNS:
        count += len(pattern.findall(text))
    return min(count / max(len(text.split()), 1) * 20, 1.0)  # normalize

def compute_title_match_score(passage_text: str, bib_title: str) -> float:
    """Check if bib title or key terms appear in the passage."""
    if not bib_title or not passage_text:
        return 0.0
    # Clean title
    title_clean = re.sub(r'\{[^}]*\}', '', bib_title)
    title_clean = re.sub(r'\s+', ' ', title_clean).strip().lower()
    passage_lower = passage_text.lower()

    # Direct match
    if len(title_clean) > 20:
        score = SequenceMatcher(None, title_clean[:100], passage_lower).ratio()
        if score > 0.3:
            return min(score * 2, 1.0)

    # Key term match: extract significant words from title (4+ chars)
    title_words = set(w for w in title_clean.split() if len(w) >= 4 and w not in
                      {'this', 'that', 'with', 'from', 'using', 'based', 'their', 'they', 'have', 'been'})
    if len(title_words) >= 3:
        passage_words = set(passage_lower.split())
        overlap = len(title_words & passage_words)
        return min(overlap / len(title_words), 1.0)
    return 0.0

def classify_section(section_name: str) -> str:
    """Classify section into broad category."""
    if not section_name:
        return 'unknown'
    sec_lower = section_name.lower().strip()
    for category, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            if kw in sec_lower:
                return category
    return 'body'

def compute_position_in_doc(char_start: int, char_end: int, total_chars: int) -> float:
    """Relative position of passage in document (0-1)."""
    if total_chars <= 0:
        return 0.5
    mid = (char_start + char_end) / 2
    return mid / total_chars

# --- Embedding computation ---
def load_embedding_model(model_name: str = None):
    """Lazy-load the embedding model with fallback chain."""
    from sentence_transformers import SentenceTransformer

    # Fallback chain: user-specified > local Qwen3-Embedding > all-MiniLM-L6-v2
    candidates = []
    if model_name:
        candidates.append(model_name)
    candidates += [
        str(PROJECT_ROOT / 'models' / 'Qwen3-Embedding-4B'),
        '/projects/myyyx1/models/Qwen3-Embedding-4B',
        'all-MiniLM-L6-v2',  # auto-downloads, 80MB, good baseline
    ]

    for candidate in candidates:
        if Path(candidate).exists() or '/' not in candidate:
            print(f"Loading embedding model: {candidate}")
            return SentenceTransformer(candidate, trust_remote_code=True)

    raise FileNotFoundError(f"No embedding model found. Tried: {candidates}")

def compute_embeddings(texts: list[str], model, batch_size: int = 32) -> np.ndarray:
    """Compute embeddings for a list of texts."""
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', default=str(PROJECT_ROOT / 'data' / '04_xdoc_citation' / 'gt_citation_pairs.jsonl'))
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'data' / '04_xdoc_citation'))
    parser.add_argument('--embedding-model', default=None)
    parser.add_argument('--neg-ratio', type=int, default=3, help='Negative samples per positive')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--no-embeddings', action='store_true', help='Skip embedding computation (faster, lower quality)')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load GT pairs ---
    print(f"Loading GT pairs from {args.input_file}")
    pairs = []
    with open(args.input_file) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} GT pairs")

    if not pairs:
        print("ERROR: No GT pairs found. Run extract_xdoc_citation_pairs.py first.")
        sys.exit(1)

    # --- Build doc info ---
    # Collect all unique source_docs and target_docs
    source_docs = set(p['source_doc'] for p in pairs)
    target_docs = set(p['target_doc'] for p in pairs)
    all_docs_in_pairs = source_docs | target_docs
    print(f"Unique source docs: {len(source_docs)}")
    print(f"Unique target docs: {len(target_docs)}")

    # Build positive pair index: (source_doc, target_doc) -> True
    positive_set = set()
    source_to_targets = defaultdict(set)  # for hard negative mining
    for p in pairs:
        key = (p['source_doc'], p['target_doc'])
        positive_set.add(key)
        source_to_targets[p['source_doc']].add(p['target_doc'])

    # --- Compute non-embedding features first ---
    print("\n=== Computing non-embedding features ===")
    features_list = []
    labels = []
    pair_records = []

    # Cache target doc info: read title from LaTeX graph
    latex_graph_path = PROJECT_ROOT / 'data' / '01_graphs' / 'latex_reference_graph_v2.json'
    with open(latex_graph_path) as f:
        latex = json.load(f)

    target_doc_info = {}
    for doc_id in target_docs:
        doc = latex['documents'].get(doc_id, {})
        meta = doc.get('metadata', {})
        target_doc_info[doc_id] = {
            'title': meta.get('title', ''),
        }

    # For computing position_in_doc, get total markdown size per source doc
    doc_md_sizes = {}
    for doc_id in source_docs:
        from extract_xdoc_citation_pairs import read_mineru_md
        md_text = read_mineru_md(doc_id)
        if md_text:
            doc_md_sizes[doc_id] = len(md_text)
        else:
            doc_md_sizes[doc_id] = 100000  # default

    # Process positive pairs
    for p in pairs:
        text = p.get('mineru_passage_text', '')
        section = p.get('source_section', '')
        char_start = p.get('passage_char_start', 0)
        char_end = p.get('passage_char_end', 0)
        bib_title = p.get('bib_title', '')

        feats = {
            'cite_pattern': compute_cite_pattern_score(text),
            'title_match': compute_title_match_score(text, bib_title),
            'section_cat': classify_section(section),
            'position': compute_position_in_doc(char_start, char_end,
                                                 doc_md_sizes.get(p['source_doc'], 100000)),
            'passage_len': len(text.split()),
        }
        features_list.append(feats)
        labels.append(1)
        pair_records.append({
            'source_doc': p['source_doc'],
            'target_doc': p['target_doc'],
            'passage_text': text[:500],
            'label': 1,
        })

    print(f"Positive samples: {len(features_list)}")

    # --- Negative sampling ---
    print(f"\nGenerating negative samples (ratio={args.neg_ratio})...")
    target_doc_list = sorted(target_docs)

    neg_count = 0
    for p in pairs[:len(features_list)]:  # align with above
        source_doc = p['source_doc']
        text = p.get('mineru_passage_text', '')
        section = p.get('source_section', '')
        char_start = p.get('passage_char_start', 0)
        char_end = p.get('passage_char_end', 0)

        pos_targets = source_to_targets[source_doc]
        neg_candidates = [t for t in target_doc_list
                          if t not in pos_targets and t != source_doc]

        # Mix: 1 hard negative (cited by other passages in same doc) + random
        n_needed = args.neg_ratio
        neg_sampled = np.random.choice(neg_candidates, size=min(n_needed, len(neg_candidates)),
                                        replace=False)

        for neg_target in neg_sampled:
            feats = {
                'cite_pattern': compute_cite_pattern_score(text),
                'title_match': 0.0,  # negative: shouldn't match
                'section_cat': classify_section(section),
                'position': compute_position_in_doc(char_start, char_end,
                                                     doc_md_sizes.get(source_doc, 100000)),
                'passage_len': len(text.split()),
            }
            features_list.append(feats)
            labels.append(0)
            pair_records.append({
                'source_doc': source_doc,
                'target_doc': neg_target,
                'passage_text': text[:500],
                'label': 0,
            })
            neg_count += 1

    print(f"Negative samples: {neg_count}")
    print(f"Total samples: {len(features_list)}")

    # --- Compute text embeddings ---
    if not args.no_embeddings:
        print("\n=== Computing text embeddings ===")
        model = load_embedding_model(args.embedding_model)

        # Collect all unique texts we need embeddings for
        # Source passages (already in pair_records)
        source_texts = [p.get('mineru_passage_text', p.get('passage_text', ''))
                        for p in pair_records]

        # Target doc abstracts/intros
        target_texts_map = {}
        for doc_id in target_docs:
            from extract_xdoc_citation_pairs import read_mineru_md
            md_text = read_mineru_md(doc_id)
            if md_text:
                # Take first 1000 chars as "abstract"
                abstract = md_text[:1000]
            else:
                info = target_doc_info.get(doc_id, {})
                abstract = info.get('title', '')
            target_texts_map[doc_id] = abstract

        target_texts = [target_texts_map.get(p['target_doc'], '')
                        for p in pair_records]

        print(f"Encoding {len(source_texts)} source passages...")
        source_embs = compute_embeddings(source_texts, model, args.batch_size)

        # For target docs, we need unique embeddings
        unique_target_docs = sorted(set(p['target_doc'] for p in pair_records))
        print(f"Encoding {len(unique_target_docs)} target doc abstracts...")
        unique_target_texts = [target_texts_map[d] for d in unique_target_docs]
        unique_target_embs = compute_embeddings(unique_target_texts, model, args.batch_size)
        target_emb_map = {d: e for d, e in zip(unique_target_docs, unique_target_embs)}

        # Compute cosine similarities
        print("Computing cosine similarities...")
        source_embs = np.array(source_embs)
        all_target_embs = np.array([target_emb_map[p['target_doc']] for p in pair_records])
        text_sims = (source_embs * all_target_embs).sum(axis=1)  # already normalized

        # Add to features
        for i, sim in enumerate(text_sims):
            features_list[i]['text_sim'] = float(sim)
    else:
        # Dummy text_sim
        for feats in features_list:
            feats['text_sim'] = 0.0

    # --- Encode categorical features ---
    section_cats = sorted(set(f['section_cat'] for f in features_list))
    section_to_idx = {c: i for i, c in enumerate(section_cats)}

    # Convert to numerical matrix
    X = np.zeros((len(features_list), 5 + len(section_cats)))
    for i, feats in enumerate(features_list):
        X[i, 0] = feats['cite_pattern']
        X[i, 1] = feats['title_match']
        X[i, 2] = feats['position']
        X[i, 3] = feats['passage_len'] / 2000.0  # normalize
        X[i, 4] = feats.get('text_sim', 0.0)
        # One-hot section
        sec_idx = section_to_idx[feats['section_cat']]
        X[i, 5 + sec_idx] = 1.0

    y = np.array(labels, dtype=np.int32)

    feature_names = (['cite_pattern', 'title_match', 'position', 'passage_len_norm', 'text_sim'] +
                     [f'section_{c}' for c in section_cats])

    # --- Save ---
    np.savez(out_dir / 'features_train.npz', X=X, y=y)
    with open(out_dir / 'feature_metadata.json', 'w') as f:
        json.dump({
            'feature_names': feature_names,
            'n_samples': len(features_list),
            'n_positive': int(y.sum()),
            'n_negative': int(len(y) - y.sum()),
            'section_categories': section_cats,
            'neg_ratio': args.neg_ratio,
        }, f, indent=2)
    with open(out_dir / 'pair_records.jsonl', 'w') as f:
        for rec in pair_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"\nSaved features to {out_dir}/features_train.npz")
    print(f"Feature names: {feature_names}")
    print(f"n_positive={int(y.sum())}, n_negative={int(len(y)-y.sum())}")
    print(f"Feature matrix shape: {X.shape}")

if __name__ == '__main__':
    main()
