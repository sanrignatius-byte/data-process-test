#!/usr/bin/env python3
"""
Phase 3 (Chunk level, FAST): Infer cross-document citation edges.

Optimized: embedding-topK-first, then title-match only on top-K candidates.
This avoids O(chunks * targets) title matching and reduces to O(chunks * K).
Also uses batch XGBoost prediction instead of per-candidate calls.

Output: data/04_xdoc_citation/predicted_xdoc_edges_chunks.jsonl
"""

import json, re, sys, argparse, pickle, numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from compute_xdoc_chunk_features import (compute_cite_pattern_score,
                                          compute_title_match_score,
                                          classify_section,
                                          load_embedding_model)

def load_corpus_doc_ids():
    mineru_dir = PROJECT_ROOT / 'data' / '00_raw' / 'mineru_output'
    return sorted(p.name for p in mineru_dir.iterdir()
                  if p.is_dir() and re.match(r'\d{4}\.\d{4,5}', p.name))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default=str(PROJECT_ROOT / 'data' / '04_xdoc_citation' / 'xgb_link_predictor.pkl'))
    parser.add_argument('--model-info', default=str(PROJECT_ROOT / 'data' / '04_xdoc_citation' / 'model_info.json'))
    parser.add_argument('--output-dir', default=str(PROJECT_ROOT / 'data' / '04_xdoc_citation'))
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--embedding-model', default='all-MiniLM-L6-v2')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-docs', type=int, default=0)
    parser.add_argument('--top-k-candidates', type=int, default=15)
    parser.add_argument('--no-embeddings', action='store_true')
    parser.add_argument('--max-edges-per-doc', type=int, default=200)
    parser.add_argument('--sim-threshold', type=float, default=0.6,
                        help='Min cosine sim for embedding candidate retrieval')
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
    threshold = args.threshold or model_info.get('optimal_threshold', 0.5)
    n_feat_expected = model.n_features_in_
    print(f"Threshold={threshold:.4f}, Features={len(feature_names)}(names) vs {n_feat_expected}(model)")

    # Load chunk data
    print("Loading chunk virtual nodes...")
    with open(PROJECT_ROOT / 'data' / '01_graphs' / 'chunk_virtual_nodes_v2.json') as f:
        chunk_data = json.load(f)
    chunk_docs = chunk_data['documents']

    # Doc IDs
    all_docs = load_corpus_doc_ids()
    if args.max_docs and args.max_docs > 0:
        all_docs = all_docs[:args.max_docs]
    docs_with_chunks = [d for d in all_docs if d in chunk_docs]
    print(f"Docs: {len(all_docs)} total, {len(docs_with_chunks)} with chunks")

    # Title index
    print("Building title index...")
    with open(PROJECT_ROOT / 'data' / '01_graphs' / 'latex_reference_graph_v2.json') as f:
        latex = json.load(f)
    doc_titles = defaultdict(list)
    for doc_id in all_docs:
        d = latex['documents'].get(doc_id, {})
        title = d.get('metadata', {}).get('title', '')
        if title:
            doc_titles[doc_id].append(title)

    # Embedding model + target embeddings
    target_doc_list = sorted(all_docs)
    target_emb_matrix = None
    if not args.no_embeddings:
        model_embed = load_embedding_model(args.embedding_model)
        from compute_xdoc_chunk_features import compute_embeddings
        target_texts = []
        for doc_id in target_doc_list:
            chunks = chunk_docs.get(doc_id, {}).get('nodes', {})
            if chunks:
                target_texts.append(list(chunks.values())[0]['text'][:500])
            else:
                target_texts.append('')
        print(f"Encoding {len(target_texts)} targets...")
        target_embs = compute_embeddings(target_texts, model_embed, args.batch_size)
        target_emb_matrix = np.stack(list(target_embs))
        print(f"Target matrix: {target_emb_matrix.shape}")
    else:
        model_embed = None

    # --- Inference ---
    print(f"\n=== Inference on {len(docs_with_chunks)} docs ===\n")
    predicted_edges = []
    stats = defaultdict(int)

    for i, source_doc in enumerate(docs_with_chunks):
        chunks_dict = chunk_docs[source_doc].get('nodes', {})
        if not chunks_dict:
            continue
        stats['docs_processed'] += 1

        chunk_ids = list(chunks_dict.keys())
        chunk_texts = [chunks_dict[cid]['text'] for cid in chunk_ids]
        total_chunks = len(chunk_ids)

        # Batch encode all chunks for this doc
        chunk_embs = None
        if model_embed is not None:
            chunk_embs = model_embed.encode(chunk_texts, batch_size=args.batch_size,
                                            normalize_embeddings=True, show_progress_bar=False)

        # Matrix multiply to get all chunk-target similarities at once
        emb_sims = None
        top_k_idx = None
        if chunk_embs is not None and target_emb_matrix is not None:
            emb_sims = chunk_embs @ target_emb_matrix.T
            K = min(args.top_k_candidates, len(target_doc_list))
            top_k_idx = np.argpartition(-emb_sims, K, axis=1)[:, :K]

        doc_edges = []

        for c_idx, chunk_id in enumerate(chunk_ids):
            chunk = chunks_dict[chunk_id]
            text = chunk['text']
            section = chunk.get('section_title', '')
            word_count = chunk.get('word_count', 200)

            cite_pattern = compute_cite_pattern_score(text)
            section_cat = classify_section(section)
            position = chunk.get('chunk_idx', c_idx) / max(total_chunks, 1)

            # ── Candidates via top-K embedding + title match ──
            candidates = []
            if top_k_idx is not None:
                for rank in range(len(top_k_idx[c_idx])):
                    t_idx = top_k_idx[c_idx, rank]
                    target_doc = target_doc_list[t_idx]
                    if target_doc == source_doc:
                        continue
                    sim = float(emb_sims[c_idx, t_idx])
                    if sim < args.sim_threshold:
                        continue
                    # Title match: only for these K candidates
                    best_title = 0.0
                    for title in doc_titles.get(target_doc, [])[:3]:
                        best_title = max(best_title, compute_title_match_score(text, title))
                    candidates.append({'target_doc': target_doc, 'title_match_score': best_title, 'text_sim': sim})

            if not candidates:
                continue

            stats['chunks_processed'] += 1
            stats['candidates_evaluated'] += len(candidates)

            # ── Batch XGBoost predict ──
            n_c = len(candidates)
            feat_matrix = np.zeros((n_c, len(feature_names)))
            for j, name in enumerate(feature_names):
                if name == 'cite_pattern':
                    feat_matrix[:, j] = cite_pattern
                elif name == 'title_match':
                    feat_matrix[:, j] = [c['title_match_score'] for c in candidates]
                elif name == 'position':
                    feat_matrix[:, j] = position
                elif name == 'chunk_size_norm':
                    feat_matrix[:, j] = word_count / 500.0
                elif name == 'text_sim':
                    feat_matrix[:, j] = [c['text_sim'] for c in candidates]
                elif name.startswith('section_'):
                    if name == f'section_{section_cat}':
                        feat_matrix[:, j] = 1.0

            if feat_matrix.shape[1] < n_feat_expected:
                feat_matrix = np.pad(feat_matrix, ((0,0),(0, n_feat_expected - feat_matrix.shape[1])))
            probs = model.predict_proba(feat_matrix)[:, 1]

            for c_i, cand in enumerate(candidates):
                if probs[c_i] >= threshold:
                    doc_edges.append({
                        'source_doc': source_doc, 'target_doc': cand['target_doc'],
                        'chunk_id': chunk_id, 'section_title': section,
                        'chunk_text': text[:300], 'probability': round(float(probs[c_i]), 4),
                        'features': {'cite_pattern': round(cite_pattern, 4),
                                     'title_match': round(cand['title_match_score'], 4),
                                     'text_sim': round(cand['text_sim'], 4),
                                     'section': section_cat, 'position': round(position, 4)},
                    })

        if len(doc_edges) > args.max_edges_per_doc:
            doc_edges.sort(key=lambda x: -x['probability'])
            doc_edges = doc_edges[:args.max_edges_per_doc]
        predicted_edges.extend(doc_edges)
        stats['edges_predicted'] += len(doc_edges)

        if (i + 1) % 50 == 0:
            print(f"  Doc {i+1}/{len(docs_with_chunks)}: {source_doc} "
                  f"({total_chunks} chunks), edges: {len(doc_edges)}, total: {stats['edges_predicted']}")

    # Save
    out_path = out_dir / 'predicted_xdoc_edges_chunks.jsonl'
    with open(out_path, 'w') as f:
        for e in predicted_edges:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    print(f"\nSaved {len(predicted_edges)} edges to {out_path}")

    stats_path = out_dir / 'inference_chunk_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(dict(stats), f, indent=2)

    print(f"Docs={stats['docs_processed']}, Chunks={stats['chunks_processed']}, "
          f"Candidates={stats['candidates_evaluated']}, Edges={stats['edges_predicted']}")

if __name__ == '__main__':
    main()
