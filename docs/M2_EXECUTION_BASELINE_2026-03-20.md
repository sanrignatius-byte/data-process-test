# M2 Execution Baseline — 2026-03-20

## Why this document exists

This is the first execution checkpoint for the unified five-stage plan:

1. read the current graph/retrieval code and formulas,
2. secure the existing data baseline,
3. identify what can already be packaged into M2,
4. isolate the real gap for Level 3 / experiment C,
5. keep the next long-running jobs aligned with mentor-facing deliverables.

## What is already solid enough to reuse

### 1. Retrieval enhancement baseline already exists

The strongest packaged Phase-0 retrieval report is `data/phase0_eval_report_v3_tuned.json`, which records:

- BM25: Recall@10 = 0.8467, MRR = 0.5642
- Graph full: Recall@10 = 0.8736, MRR = 0.6045
- absolute gain: +0.0269 Recall@10, +0.0403 MRR

This means **experiment B is not a cold start**. The main remaining work is to:

- repackage the existing report into the M2 experiment namespace,
- add Level-3-specific slicing once the new 3-step data is ready,
- keep the tuned report and the locked/fixed report side-by-side for reproducibility.

### 2. Level 1 and Level 2 already have reusable source pools

Current local assets already provide a conservative baseline:

- `data/l1_multihop_queries_v3.jsonl`: current Level-1-style source pool.
- `data/l1_dual_evidence_queries_v3_pass.jsonl`: QC-passed dual-evidence set.
- `data/l1_dual_evidence_triplets_v2_pass.jsonl`: triplet-form view of the same Level-2 core.
- `data/l1_dual_evidence_queries_v4_4_run1.jsonl`: a larger expansion pool that still needs selective re-QC.

So the immediate priority is **not** inventing Level 2 from scratch, but mapping these files into the M2 delivery contract.

### 3. Embedding-based virtual-edge exploration is also not a cold start

The repo already contains:

- `data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B.jsonl`
- `data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2_rerank_audit.json`

This is enough to begin the Stage-4 probe on 2-3 documents without waiting for a new end-to-end embedding pipeline.

## What is still the biggest gap

### Level 3 is the real bottleneck

There are precursor artifacts for longer chains:

- `data/l1_dual_evidence_long_chain_queries_v2_iterative.jsonl`
- `data/l2_queries_v3.jsonl`

But these are **not yet the final native 3-step benchmark** required by the current plan. The repository already documents the same issue clearly: current work is closer to “dual-evidence + pseudo-multihop” than a strict serial reasoning benchmark.

So the next engineering target should be treated as:

> build a true `m2_level3_3step.jsonl` pipeline with explicit `evidence_chain`, answer grounding, and step-deletion-oriented QC.

## Immediate execution order from here

### A. Packaging tasks that can start now

- Create the M2 namespace under `data/m2/` and `experiments/`.
- Refresh `experiments/m2_execution_manifest.json` before each execution round.
- Repackage the tuned Phase-0 retrieval report into experiment-B format.

### B. Generation tasks that need new work

- Native Level-3 3-step query generation.
- QA triangle evaluation artifact (`exp_C_qa_triangle.json`).
- Authority-hub-driven general query export (`m2_general_queries.jsonl`).

### C. Read-the-code targets that must stay fixed in explanations

When presenting the system, keep the explanation anchored to the four stable objects below:

1. node types,
2. edge types,
3. hub scoring,
4. retrieval reranking formula.

That keeps the implementation story aligned with both mentor questions and the current codebase.
