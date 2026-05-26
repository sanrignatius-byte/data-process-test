# claim:C18_latex_citation_features_transfer_to_mineru

**Claim**: Cross-document citation patterns learned from LaTeX reference graphs (bib + `\cite{}`) can be transferred to detect citation edges in pure MinerU (PDF-parsed) documents, using only features computable from MinerU output.

**Status**: supported

**Evidence**:
- Chunk-level XGBoost trained on 4,028 LaTeX→MinerU aligned citation pairs
- 5-fold GroupKFold (split by source doc): AUC 0.852, F1 0.746
- Top-50 precision: 1.0 across all folds
- `title_match` (paper title appearing in chunk text) accounts for 88.1% of feature importance
- Inference on 1,147 MinerU-only docs: 53,435 predicted edges, 75% with probability ≥ 0.95
- Body (non-References) high-confidence edges: 34,875 across 18,798 unique doc-pairs

**Scope**: Supported for papers where MinerU output preserves paper titles in reference lists and citation markers in body text. Weaker for papers with heavy OCR noise or where reference sections are poorly parsed.

**Linked**: [[exp:20260520_xdoc_citation_link_predictor]], [[C15_mineru_regex_recovers_latex_intradoc_edges]], [[C16_mineru_xdoc_visual_edges_recall_not_strong]]
