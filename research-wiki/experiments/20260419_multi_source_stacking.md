```markdown
---
type: experiment
node_id: exp:20260419_multi_source_stacking
status: completed
verdict: explicit_only_still_dominates_without_weighting
created_at: 2026-04-19T00:00:00Z
updated_at: 2026-04-19T00:00:00Z
---

# One-line summary

Systematic ablation of multi-source edge stacking (explicit × virtual × cross_doc × summary) on Qwen3-Embedding-0.6B shows that, under the existing `merge='max'` + degree-based prior, adding any virtual or cross_doc layer to explicit ties or hurts R@1/MRR; a new per-source weight + weighted-prior mechanism is required before stacking can help.

## Setup

- Retrieval model: Qwen3-Embedding-0.6B
- Corpus: `v1_enriched` (2809 passages), 473 queries
- Script: `scripts/eval_graph_topk_rerank.py`
- Methods: `static_prior`, `static_plus_neighbor`

## Variant A — equal-weight stacking (max-merge, degree prior)

Output: `data/05_eval/dense_retrieval/stacking_06b/`

| Config | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| explicit_only (ref) | 0.2357 | 0.5507 | 0.6258 | 0.6166 |
| explicit + chunk_seq | 0.2104 | 0.5328 | 0.6237 | 0.5779 |
| explicit + same_chunk | 0.2178 | 0.5243 | 0.6237 | 0.5871 |
| explicit + virtual_all | 0.1987 | 0.5254 | 0.6247 | 0.5633 |
| explicit + crossdoc | 0.2357 | 0.5507 | 0.6258 | 0.6166 |
| crossdoc_only | 0.2389 | 0.5127 | 0.5994 | 0.6081 |

The `explicit + crossdoc == explicit_only` and `crossdoc_only == dense_baseline` identities were the diagnostic that led us to the cross_doc silent-failure bug (see `exp:20260419_cross_doc_bug_fix`).

## Variant B — per-source weighted stacking (new mechanism)

Added to `scripts/eval_graph_topk_rerank.py`:
- `--explicit-weight / --virtual-weight / --crossdoc-weight / --typed-crossdoc-weight`
- `--merge-combine {max, sum}`
- `--prior-mode {degree, weighted}` where `weighted` uses `log1p(sum-of-weights)` so per-source weights actually influence the prior.

Output: `data/05_eval/dense_retrieval/stacking_06b_weighted/`

| Config | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| explicit_only (ref) | 0.2357 | 0.5507 | 0.6258 | 0.6166 |
| ex1.0 + vir0.1 (weighted) | 0.2230 | 0.5412 | 0.6237 | 0.6016 |
| ex1.0 + vir0.2 (weighted) | 0.2146 | 0.5402 | 0.6258 | 0.5892 |
| ex5.0 + vir1.0 (weighted) | 0.2188 | 0.5455 | **0.6300** | 0.5980 |
| ex2.0 + vir0.2 + cd0.5 (sum) | 0.2252 | 0.5423 | 0.6268 | 0.6050 |

## Conclusion

1. Without the per-source weight axis, any non-explicit layer either ties explicit_only (because the layer has zero effective pid-pairs, see cross_doc bug) or dilutes precision (virtual edges).
2. Even with per-source weighting, virtual edges only approach explicit_only and never clearly exceed it on R@1/MRR.
3. Cross-doc at section granularity adds almost nothing once bug-fixed (see `exp:20260419_typed_crossdoc` for the finer element-level replacement).

## Connections

- Invalidates loose claim that stacking edges is always positive; supports `claim:C2`.
- Introduces the weighting mechanism later exploited by `exp:20260419_typed_crossdoc`.
- Addresses: `gap:G1`, `gap:G2`.
```
