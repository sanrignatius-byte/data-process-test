---
type: experiment
node_id: exp:20260417_dense_baseline_rebuilt
status: completed
verdict: reference_baseline
created_at: 2026-04-18T00:00:00Z
updated_at: 2026-04-18T00:00:00Z
---

# One-line summary

Dense retrieval baseline on the rebuilt evaluation package serves as the trusted non-graph reference point.

## Setup

- Dataset: rebuilt package with 57 docs, 473 queries, 2809 corpus passages, 946 qrels.
- Result file: `data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_only_fixed/metrics_baseline.json`

## Key Results

- MRR: 0.6121
- Recall@1: 0.2336
- Recall@5: 0.5275
- Recall@10: 0.6195

## Interpretation

This is the baseline that corrected graph rerank results should beat. It remains valid because it does not depend on graph-layer wiring.

## Connections

- Baseline for: `exp:20260417_explicit_rerank_fixed`
- Evidence for: `claim:C1`

