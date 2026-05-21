# exp:20260520_xdoc_citation_link_predictor

Cross-document citation link predictor trained from LaTeX ground truth, generalized to MinerU-only documents.

## Motivation
Most documents in the corpus lack LaTeX source — they only have MinerU (PDF→markdown+images) output. We need to detect cross-document citation relationships without relying on `\cite{}` markup. The idea is to learn what MinerU-side features characterize cross-document citations, using the 1,067 overlapping (LaTeX ∩ MinerU) documents as training data, then generalize to all 1,147 documents.

## Method
1. **GT extraction**: Parse LaTeX `bib + refs(ref_type:cite)` → extract cross-doc citation pairs where bib arxiv-id matches another corpus doc. Align citing context to MinerU chunks via text overlap.
2. **Feature computation**: For each (source_chunk, target_doc) pair: `cite_pattern_score` (regex), `title_match_score` (fuzzy title matching), `text_sim` (embedding cosine similarity via all-MiniLM-L6-v2), `section_type`, `position_in_doc`, `chunk_size`.
3. **Classifier**: XGBoost (n=200, max_depth=5), 5-fold GroupKFold by source doc, 3:1 neg:pos ratio.
4. **Inference**: Embedding top-K retrieval (K=15) → title match only on top-K → batch XGBoost scoring → per-doc top-200 edges.

## Pipeline variants
| Level | Units | GT pairs | AUC | F1 |
|---|---|---|---|---|
| Passage (markdown paragraphs) | ~200K | 11,047 | 0.820 | 0.628 |
| **Chunk (pre-built, ~400 words)** | ~42K | 4,028 | **0.852** | **0.746** |

Chunk level wins on both accuracy and speed — coarser units aggregate more citation context, and section info is more reliable.

## Feature importance (chunk model)
- `title_match`: 0.881 — paper title appearing in chunk text is the strongest signal
- `section_related_work`: 0.019
- `text_sim`: 0.018
- `section_experiment`: 0.014
- `cite_pattern`: 0.011

## Inference results
- **53,435 predicted cross-document citation edges** across 1,108 source × 864 target docs
- 27,349 unique doc-pairs
- 75% edges with probability ≥ 0.95
- Body (non-References) edges: 42,033; high-confidence (≥0.8) body: 34,875
- Top sections: References (21%), Introduction (16%), Related Work (10%)

## Artifacts
- GT pairs: `data/04_xdoc_citation/gt_citation_pairs.jsonl` (passage) / `gt_citation_chunks.jsonl` (chunk)
- Training data: `data/04_xdoc_citation/features_chunk_train.npz` (16,112 samples, 12 features)
- Model: `data/04_xdoc_citation/xgb_link_predictor.pkl`
- Predicted edges: `data/04_xdoc_citation/predicted_xdoc_edges_chunks.jsonl`
- Scripts: `scripts/extract_xdoc_citation_chunks.py`, `scripts/compute_xdoc_chunk_features.py`, `scripts/infer_xdoc_citation_chunks.py`, `scripts/train_xdoc_link_predictor.py`
- Slurm: `slurm_scripts/78_xdoc_citation_chunk.sh`, `slurm_scripts/79_xdoc_chunk_infer.sh`

## Key findings
1. **LaTeX citation patterns are learnable from MinerU features** — the model generalizes to documents without LaTeX source.
2. **Title matching dominates** (88% importance) — the most reliable cross-doc citation signal in MinerU text is the target paper's title appearing in the source text.
3. **Embedding similarity helps recall** (+11% recall vs no-embedding baseline) but adds noise if used as primary retrieval.
4. **Chunk level > passage level** — pre-built chunks provide better context aggregation and more reliable section labels.
5. **References-section edges are detected but trivial** — the more interesting edges are in Introduction/Related Work/Method sections where papers discuss each other's work.

## Known issues
- Acknowledgement/funding sections produce false positives (high embedding sim but not real citations)
- Title-page author lists misdetected as cross-doc links
- Pure semantic similarity (same subfield, no actual citation) can score high
- Current threshold (0.50) may need per-section calibration

## Next steps
- Filter/normalize by section type to reduce false positives
- Deduplicate at doc-pair level with aggregation
- Use predicted edges for multi-document query generation
- Integrate into cross-doc retrieval graph
