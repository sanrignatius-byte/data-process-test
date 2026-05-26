#!/usr/bin/env python3
"""
Phase 3: Infer cross-document citation edges on all MinerU docs.

Takes the trained XGBoost model and runs feature extraction + inference
on all documents that have MinerU output (no LaTeX needed).

Process:
  1. Load all MinerU docs, split into passages
  2. For each doc, find candidate target docs via:
     a. Citation patterns in text → try to match to corpus doc titles
     b. Text embedding similarity to other docs' abstracts
  3. Compute features for each candidate pair
  4. Score with XGBoost model
  5. Output predicted edges above threshold

Output: data/04_xdoc_citation/predicted_xdoc_edges.jsonl
"""

import json
import re
import argparse
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from extract_xdoc_citation_pairs import (read_mineru_md, split_md_into_passages,
                                          SECTION_RE)
from compute_xdoc_passage_features import (compute_cite_pattern_score,
                                            compute_title_match_score,
                                            classify_section, compute_position_in_doc)

def load_corpus_doc_ids() -> list[str]:
    """Get all doc IDs with MinerU output."""
    mineru_dir = PROJECT_ROOT / 'data' / '00_raw' / 'mineru_output'
    ids = []
    for p in sorted(mineru_dir.iterdir()):
        if p.is_dir() and re.match(r'\d{4}\.\d{4,5}', p.name):
            ids.append(p.name)
    return ids

def load_bib_title_index() -> dict[str, list[str]]:
    """Build arxiv_id -> [titles] index from LaTeX reference graph.
    Each doc in the corpus may have been cited under different title variations.
    """
    latex_graph_path = PROJECT_ROOT / 'data' / '01_graphs' / 'latex_reference_graph_v2.json'
    with open(latex_graph_path) as f:
        latex = json.load(f)

    doc_titles = defaultdict(list)
    for doc_id in load_corpus_doc_ids():
        if doc_id not in latex['documents']:
            continue
        meta = latex['documents'][doc_id].get('metadata', {})
        title = meta.get('title', '')
        if title:
            doc_titles[doc_id].append(title)

    # Also add titles from bib entries that reference this doc
    for src_doc_id, src_doc in latex['documents'].items():
        bib = src_doc.get('bib', {})
        for bib_key, bib_entry in bib.items():
            if not isinstance(bib_entry, dict):
                continue
            bib_title = bib_entry.get('title', '')
            if not bib_title:
                continue
            # Check if this bib entry's arxiv ID is a known doc
            from extract_xdoc_citation_pairs import extract_arxiv_ids_from_bib
            ids = extract_arxiv_ids_from_bib(bib_entry)
            for aid in ids:
                if aid in doc_titles:
                    if bib_title not in doc_titles[aid]:
                        doc_titles[aid].append(bib_title)

    return dict(doc_titles)

def find_candidate_pairs(passage: dict, source_doc: str,
                         all_docs: list[str],
                         doc_titles: dict[str, list[str]],
                         source_emb: np.ndarray | None,
                         target_embs: dict[str, np.ndarray] | None,
                         top_k: int = 20) -> list[dict]:
    """Find candidate target docs for a given source passage."""

    candidates = []

    # Method 1: Title matching from citation patterns
    text = passage['text']
    # If passage has citation patterns, try to match to doc titles
    cite_score = compute_cite_pattern_score(text)
    if cite_score > 0.1:
        for target_doc in all_docs:
            if target_doc == source_doc:
                continue
            titles = doc_titles.get(target_doc, [])
            best_title_match = 0.0
            for title in titles:
                match = compute_title_match_score(text, title)
                best_title_match = max(best_title_match, match)
            if best_title_match > 0.1:
                candidates.append({
                    'target_doc': target_doc,
                    'match_source': 'title_match',
                    'title_match_score': best_title_match,
                })

    # Method 2: Embedding similarity (if embeddings available)
    if source_emb is not None and target_embs is not None:
        sims = []
        for target_doc, target_emb in target_embs.items():
            if target_doc == source_doc:
                continue
            sim = float((source_emb * target_emb).sum())
            sims.append((target_doc, sim))
        sims.sort(key=lambda x: -x[1])
        for target_doc, sim in sims[:top_k]:
            if sim > 0.65:  # cosine similarity threshold
                candidates.append({
                    'target_doc': target_doc,
                    'match_source': 'embedding',
                    'text_sim': sim,
                })

    # Deduplicate by target_doc
    seen = set()
    unique = []
    for c in candidates:
        if c['target_doc'] not in seen:
            seen.add(c['target_doc'])
            unique.append(c)
    return unique

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=str(PROJECT_ROOT / 'data' / '04_xdoc_citation' / 'xgb_link_predictor.pkl'))
    parser.add_argument('--model-info', default=str(PROJECT_ROOT / 'data' / '04_xdoc_citation' / 'model_info.json'))
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'data' / '04_xdoc_citation'))
    parser.add_argument('--threshold', type=float, default=None, help='Override model threshold')
    parser.add_argument('--embedding-model', default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-docs', type=int, default=0, help='Cap number of source docs')
    parser.add_argument('--top-k-candidates', type=int, default=20)
    parser.add_argument('--no-embeddings', action='store_true', help='Skip embeddings (use only text features)')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)
    with open(args.model_info) as f:
        model_info = json.load(f)
    feature_names = model_info['feature_names']
    print(f"Features: {feature_names}")

    threshold = args.threshold if args.threshold is not None else model_info.get('optimal_threshold', 0.5)
    print(f"Decision threshold: {threshold}")

    # Load all corpus docs
    all_docs = load_corpus_doc_ids()
    print(f"Total MinerU docs: {len(all_docs)}")

    if args.max_docs and args.max_docs > 0:
        all_docs = all_docs[:args.max_docs]
        print(f"Capped to {len(all_docs)} docs")

    # Load title index
    print("Loading bib title index...")
    doc_titles = load_bib_title_index()

    # Load embedding model (if not skipped)
    source_embs_cache = {}
    target_embs_cache = {}
    model_embed = None

    if not args.no_embeddings:
        from compute_xdoc_passage_features import load_embedding_model, compute_embeddings
        model_embed = load_embedding_model(args.embedding_model)

        # Compute target doc abstract embeddings
        target_docs_for_emb = sorted(all_docs)
        target_texts = []
        for doc_id in target_docs_for_emb:
            md_text = read_mineru_md(doc_id)
            if md_text:
                target_texts.append(md_text[:1000])
            else:
                target_texts.append('')
        print(f"Encoding {len(target_texts)} target doc abstracts...")
        target_embs = compute_embeddings(target_texts, model_embed, args.batch_size)
        for doc_id, emb in zip(target_docs_for_emb, target_embs):
            target_embs_cache[doc_id] = emb
    else:
        print("Skipping embeddings (--no-embeddings).")

    # --- Pre-build target embedding matrix (for fast dot-product) ---
    target_doc_list = sorted(all_docs)
    target_emb_matrix = None
    if model_embed is not None and target_embs_cache:
        target_emb_matrix = np.stack([target_embs_cache[d] for d in target_doc_list])
        print(f"Target embedding matrix: {target_emb_matrix.shape}")

    # --- Inference loop ---
    print(f"\n=== Running inference on {len(all_docs)} source docs ===\n")
    predicted_edges = []
    stats = {'docs_processed': 0, 'passages_processed': 0,
             'candidates_evaluated': 0, 'edges_predicted': 0}

    all_section_cats = ['unknown', 'introduction', 'related_work', 'method',
                        'experiment', 'conclusion', 'body']

    for i, source_doc in enumerate(all_docs):
        md_text = read_mineru_md(source_doc)
        if not md_text:
            continue

        passages = split_md_into_passages(md_text)
        if not passages:
            continue

        stats['docs_processed'] += 1
        total_chars = len(md_text)
        n_passages = len(passages)
        stats['passages_processed'] += n_passages

        # ── Batch-encode all passages for this doc ──
        passage_embs = None
        if model_embed is not None:
            passage_texts = [p['text'] for p in passages]
            passage_embs = model_embed.encode(
                passage_texts, batch_size=args.batch_size,
                normalize_embeddings=True, show_progress_bar=False
            )

        # ── Pre-compute embedding similarities (passages × targets) ──
        emb_sims = None
        if passage_embs is not None and target_emb_matrix is not None:
            emb_sims = passage_embs @ target_emb_matrix.T  # (n_passages, n_targets)
            # For each passage, get top-K targets by embedding similarity
            top_k = min(args.top_k_candidates, len(target_doc_list))
            top_k_idx = np.argpartition(-emb_sims, top_k, axis=1)[:, :top_k]
            top_k_sims = np.take_along_axis(emb_sims, top_k_idx, axis=1)

        # ── Process each passage ──
        for p_idx, passage in enumerate(passages):
            # Pre-computed features
            cite_pattern = compute_cite_pattern_score(passage['text'])
            section_cat = classify_section(passage['section'])
            position = compute_position_in_doc(passage['char_start'],
                                                passage['char_end'], total_chars)
            passage_len = len(passage['text'].split())

            # ── Find candidates ──
            # Method 1: Title matching
            candidates = []
            if cite_pattern > 0.1:
                # Quick string match — check against all doc titles
                for t_idx, target_doc in enumerate(target_doc_list):
                    if target_doc == source_doc:
                        continue
                    titles = doc_titles.get(target_doc, [])
                    best_match = 0.0
                    for title in titles[:3]:  # max 3 title variants
                        match = compute_title_match_score(passage['text'], title)
                        if match > best_match:
                            best_match = match
                    if best_match > 0.1:
                        text_sim = float(emb_sims[p_idx, t_idx]) if emb_sims is not None else 0.0
                        candidates.append({
                            'target_doc': target_doc,
                            'title_match_score': best_match,
                            'text_sim': text_sim,
                        })

            # Method 2: Top-K by embedding similarity (if no title match found or to supplement)
            if emb_sims is not None:
                existing_targets = {c['target_doc'] for c in candidates}
                for rank in range(len(top_k_idx[p_idx])):
                    t_idx = top_k_idx[p_idx, rank]
                    target_doc = target_doc_list[t_idx]
                    sim = float(top_k_sims[p_idx, rank])
                    if target_doc == source_doc:
                        continue
                    if sim < 0.65:
                        continue
                    if target_doc in existing_targets:
                        # Update text_sim if already found via title
                        for c in candidates:
                            if c['target_doc'] == target_doc:
                                c['text_sim'] = sim
                                break
                    else:
                        best_title = 0.0
                        titles = doc_titles.get(target_doc, [])
                        for title in titles[:3]:
                            match = compute_title_match_score(passage['text'], title)
                            best_title = max(best_title, match)
                        candidates.append({
                            'target_doc': target_doc,
                            'title_match_score': best_title,
                            'text_sim': sim,
                        })
                        existing_targets.add(target_doc)

            if not candidates:
                continue

            # ── Score each candidate with XGBoost ──
            for cand in candidates:
                stats['candidates_evaluated'] += 1
                target_doc = cand['target_doc']
                title_match = cand['title_match_score']
                text_sim = cand['text_sim']

                # Build feature vector
                feat_vec = np.zeros(len(feature_names))
                for j, name in enumerate(feature_names):
                    if name == 'cite_pattern':
                        feat_vec[j] = cite_pattern
                    elif name == 'title_match':
                        feat_vec[j] = title_match
                    elif name == 'position':
                        feat_vec[j] = position
                    elif name == 'passage_len_norm':
                        feat_vec[j] = passage_len / 2000.0
                    elif name == 'text_sim':
                        feat_vec[j] = text_sim
                    elif name.startswith('section_'):
                        sec_name = name.replace('section_', '')
                        if sec_name == section_cat:
                            feat_vec[j] = 1.0

                prob = float(model.predict_proba(feat_vec.reshape(1, -1))[0, 1])

                if prob >= threshold:
                    predicted_edges.append({
                        'source_doc': source_doc,
                        'target_doc': target_doc,
                        'source_passage_start': passage['char_start'],
                        'source_passage_end': passage['char_end'],
                        'source_section': passage['section'],
                        'source_text': passage['text'][:300],
                        'probability': round(prob, 4),
                        'features': {
                            'cite_pattern': round(cite_pattern, 4),
                            'title_match': round(title_match, 4),
                            'text_sim': round(text_sim, 4),
                            'section': section_cat,
                            'position': round(position, 4),
                        },
                    })
                    stats['edges_predicted'] += 1

        if (i + 1) % 20 == 0:
            print(f"  Doc {i+1}/{len(all_docs)}: {source_doc} ({n_passages} passages), "
                  f"edges so far: {stats['edges_predicted']}")

    # Save
    output_path = out_dir / 'predicted_xdoc_edges.jsonl'
    with open(output_path, 'w') as f:
        for e in predicted_edges:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    stats_path = out_dir / 'inference_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Inference complete ===")
    print(f"Docs processed: {stats['docs_processed']}")
    print(f"Passages: {stats['passages_processed']}")
    print(f"Candidates evaluated: {stats['candidates_evaluated']}")
    print(f"Predicted edges: {stats['edges_predicted']}")
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    main()
