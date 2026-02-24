# Progress Summary (Since Last Report)

Date: 2026-02-24
Scope: dual-evidence L1 query pipeline -> triplet construction -> cross-doc embedding matching -> utility-aware rerank

## 0) Continuity With Previous L1 Report

Reference: `docs/L1_query_iteration_report.md`

Historical baseline (figure-text L1 track):
- 73 papers, 351 figure-text pairs input
- v3 final output: 974 queries
- QC pass: 97.2%
- validation clean rate: 84.3%

Current track in this report:
- We moved from single figure-text L1 generation to stricter **dual-evidence** pipeline
  (figure+table / figure+formula / formula+table) for M4 retrieval training.
- Therefore, absolute counts are not directly comparable with old L1 totals.
- Comparable dimension is: **quality gates became stricter, and training data now includes explicit hard negatives + cross-doc candidate control.**

Method inheritance from previous L1 work:
- prompt hard constraints and QC-first generation
- anti-meta-language filtering
- reproducible script/report artifacts

## 1) What Was Completed

1. Official L1 dual-evidence query batch is complete.
2. Triplet data construction is complete for both v1 and v2 variants.
3. Local Qwen3-Embedding-4B cross-document matching is complete.
4. Matching quality audit is complete.
5. Stage-B utility-aware rerank is implemented and executed (v2 strict + v2 balanced).

## 2) Methods Used

### A. Query Generation + QC
- Source: `data/l1_dual_evidence_queries_slurm_img150_tuned_v4_official.jsonl`
- Prompt/QC strategy:
  - pre-query `reasoning_chain`
  - entity amnesty + causal topology
  - template-collapse checks
  - anchor leakage checks (with amnesty)
  - dual-evidence overlap checks

### B. Triplet Construction
- v1: `in_doc_swap + same_type_hard`
- v2: `in_doc_swap + same_type_hard_plus`
- Added text compaction fields:
  - `text` (full)
  - `text_short` (training-friendly)
- Added image coverage checks for positive and negative bundles.

### C. Cross-doc Embedding Matching
- Model: local `Qwen3-Embedding-4B`
- Output: `data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B.jsonl`
- Audit script: `scripts/audit_mineru_crossdoc_embedding_matches.py`
- Audit dimensions:
  - score distribution (overall + top1 + rank-wise)
  - constraint validity (cross-doc, type consistency)
  - hubness concentration
  - reciprocity
  - suspicious candidate sampling

### D. Utility-aware Rerank (new)
- Script: `scripts/rerank_mineru_crossdoc_matches.py`
- Signals:
  - target hub penalty
  - target-doc popularity penalty
  - intra-list diversity penalty
  - global top1 per-target cap
- Artifacts:
  - strict: `..._v2_rerank.jsonl` (cap=8)
  - balanced: `..._v2b_cap10.jsonl` (cap=10)

## 3) Key Results

### A. Query Official Batch (v4)
Source: `data/l1_dual_evidence_queries_slurm_img150_tuned_v4_official_report.json`

- Total queries: 222
- QC pass: 173
- QC pass rate: 77.93%
- Pair type:
  - figure+table: 144
  - figure+formula: 62
  - formula+table: 16

### B. Triplets

v1 all:
- `data/l1_dual_evidence_triplets_v1_all.jsonl`: 222 triplets
- avg negatives/triplet: 2.0
- avg difficulty: 0.6248

v2 all:
- `data/l1_dual_evidence_triplets_v2_all.jsonl`: 222 triplets
- avg negatives/triplet: 2.0
- avg difficulty: 0.7288
- positive image coverage: 100%
- negative image coverage: 99.55%

Baseline retrieval stress-test (pass set, BM25):
- v1 pass (`...v1_pass_baseline_report.json`):
  - local acc@1: 0.8092
  - global acc@1: 0.5549
- v2 pass full text (`...v2_pass_baseline_text_report.json`):
  - local acc@1: 0.7514
  - global acc@1: 0.4451

Interpretation:
- v2 negatives are harder (difficulty up), so lexical baseline gets worse. This is expected if we are reducing shortcut behavior.

### C. Embedding Matching Audit (4B baseline)
Source: `data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_audit.json`

- records: 590 (top-k=20, total matches 11800)
- constraints:
  - same-doc violations: 0
  - type mismatches: 0
- top1 mean score: 0.8822
- top1 concentration (top10 targets): 0.3153
- unique top1 targets: 186
- top1 reciprocal rate: 0.7051
- suspicious candidates: 241

Interpretation:
- Retrieval is stable but hubness is strong, and top1 concentration is too high for multi-hop utility.

### D. Utility-aware Rerank Impact

Strict config (cap=8):
- source: `data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2_rerank_report.json`
- top1 mean: 0.8822 -> 0.8635
- top10 concentration: 0.3153 -> 0.1271
- unique top1 targets: 186 -> 275
- suspicious candidates: 241 -> 140

Balanced config (cap=10, recommended):
- source: `data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2b_cap10_report.json`
- top1 mean: 0.8822 -> 0.8690
- top10 concentration: 0.3153 -> 0.1305
- unique top1 targets: 186 -> 286
- top1 reciprocal: 0.7051 -> 0.8119
- suspicious candidates: 241 -> 146

Decision:
- Use `..._v2b_cap10.jsonl` as the default downstream candidate set.

## 4) Current Problems / Defects

1. Objective mismatch still exists:
- similarity score is not equal to multi-hop utility.
- We reduced hubness, but we still do not directly optimize `hop_utility`.

2. No human-labeled utility benchmark yet:
- We need at least 100-300 labeled pairs with:
  - relevance
  - hop_utility
  - redundancy
  - error_type taxonomy

3. Hubness remains at all-rank pool level:
- Top1 is improved, but all-rank concentration still shows hot targets.

4. Missing image-paths are non-zero:
- source missing rate ~12% (formula-heavy expected, but still needs data-loader policy).

5. Margin metrics after rerank need careful interpretation:
- top1 is selected by utility-aware policy, not raw-score sorting.
- raw `margin12` alone is no longer a reliable quality signal post-rerank.

## 5) Recommended Next Step (Immediate)

1. Freeze balanced rerank output as current production candidate file:
- `data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2b_cap10.jsonl`

2. Build human eval set (200 pairs) with taxonomy labels and compute:
- HopUtility@1/@5/@20
- error bucket distribution

3. Integrate reranked cross-doc candidates into triplet v3 negative mining:
- keep `in_doc_swap`
- replace/augment `same_type_hard_plus` with reranked cross-doc negatives

4. Add one context-aware reranker baseline (cross-encoder or LLM judge) for ablation:
- embedding-only vs +hub/diversity vs +context rerank
