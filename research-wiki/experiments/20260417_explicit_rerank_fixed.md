---
type: experiment
node_id: exp:20260417_explicit_rerank_fixed
status: completed
verdict: supports_main_claim
created_at: 2026-04-18T00:00:00Z
updated_at: 2026-04-18T00:00:00Z
---

# One-line summary

Corrected graph rerank experiments show that explicit-only graph edges plus static prior improve precision-oriented retrieval over the dense baseline.

## Setup

- Baseline file: `data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_only_fixed/metrics_baseline.json`
- Main result file: `data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_only_fixed/metrics_graph_static_prior.json`
- Comparison files:
  - `.../metrics_graph_static_plus_neighbor.json`
  - `data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_plus_all_virtual_fixed/metrics_graph_static_prior.json`

## Key Results

- Explicit-only + static prior:
  - MRR: 0.6399
  - Recall@1: 0.2421
  - Recall@5: 0.5856
  - Recall@10: 0.6448
- Dense baseline:
  - MRR: 0.6121
  - Recall@1: 0.2336
  - Recall@5: 0.5275
  - Recall@10: 0.6195
- Explicit-only + static+neighbor:
  - MRR: 0.6017
  - Recall@1: 0.2209
  - Recall@5: 0.5941
  - Recall@10: 0.6913
- Explicit + all intra-doc virtual edges + static prior:
  - MRR: 0.6010
  - Recall@1: 0.2156
  - Recall@5: 0.5581
  - Recall@10: 0.6416

## Interpretation

- Static prior on explicit edges is the current best precision-oriented setting.
- Neighbor propagation helps deeper recall but hurts first-hit quality and MRR.
- Adding all current intra-doc virtual edges dilutes precision signal.

## Decision

Use explicit-only + static prior as the trusted graph-rerank baseline for future comparisons.

## Connections

- Tests: `idea:001`
- Supports: `claim:C1`, `claim:C2`
- Informs: `claim:C4`

