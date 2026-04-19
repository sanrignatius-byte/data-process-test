---
type: claim
node_id: claim:C3
status: partially_supported_revised
created_at: 2026-04-18T00:00:00Z
updated_at: 2026-04-19T00:00:00Z
---

# Claim (revised 2026-04-19)

After the `load_cross_doc_adjacency()` bug fix, cross-document section-summary edges combined with explicit edges under `static_plus_neighbor` tie the project-high R@10 of 0.6406, but do not produce a strict R@1 / MRR uplift over explicit_only. The strong form of the original claim (cross-doc summary edges improve rerank across the board) is not supported.

## Evidence

- `exp:20260419_cross_doc_bug_fix`: fixed loader, pid_pairs 0 → 5135, doc_pairs 0 → 167.
- `exp:20260419_typed_crossdoc` under `static_plus_neighbor`:
  - `explicit + crossdoc_sec (fixed)` → R@1 0.1734, R@5 0.5423, R@10 **0.6406**, MRR 0.5273.
  - `explicit_only` reference → R@1 0.2357, R@5 0.5507, R@10 0.6258, MRR 0.6166.
- Under `static_prior`, `crossdoc_sec_only (fixed)` alone is weaker than explicit_only on all metrics.

## Revised status

- Supported: cross-doc summary edges contribute to R@10 when stacked with explicit under `static_plus_neighbor`.
- Not supported: cross-doc summary edges improve R@1 or MRR.
- Superseded at element granularity by `claim:C5`.

## Scope

- Model: Qwen3-Embedding-0.6B.
- Corpus: `v1_enriched`, 473 queries.
- Method: `static_plus_neighbor` for the R@10 tie; `static_prior` for the weaker-alone result.
- Citation-confidence variants were not re-tested post-fix; deferred until bbl coverage expands.

## Risk

- R@10 tie may be driven by neighbor propagation rather than cross-doc content; a propagation-free test on the fixed adjacency is still needed.
- With broader bbl coverage and retuned citation boost, the R@1 / MRR axes might change.

## Connections

- Tested by: `exp:20260419_typed_crossdoc` (supersedes `exp:20260418_cross_doc_summary_pending`).
- Depends on: `exp:20260419_cross_doc_bug_fix`.
- Originates from: `idea:002`.
- Superseded at element granularity by: `claim:C5`.

