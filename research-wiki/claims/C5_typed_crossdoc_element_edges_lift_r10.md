---
type: claim
node_id: claim:C5
status: supported
created_at: 2026-04-19T00:00:00Z
updated_at: 2026-04-19T16:30:00Z
---

# Claim

Element-level typed cross-document edges (figure / formula / table, Qwen3-Embedding-0.6B similarity ≥ 0.70, top-K = 10, optional bbl citation boost) provide a strictly larger retrieval signal than section-level cross-document summary edges on 0.6B `v1_enriched`, and produce R@10 = 0.6406 when combined with explicit edges at small weight under `static_plus_neighbor`.

> ⚠️ "project-high R@10" title transferred to C7 (0.6522, explicit_only + v1_enriched + static_plus_neighbor, 2026-04-19).

## Evidence

From `exp:20260419_typed_crossdoc`:
- `typed_only_figure` R@1 = 0.2178 vs `crossdoc_sec_only (fixed)` R@1 = 0.1575.
- `typed_only_table` R@1 = 0.2125.
- `explicit + typed (w=0.2)` static_plus_neighbor: R@10 = 0.6406 (+1.5pp vs explicit_only 0.6258 on that run).
- Section cross_doc, once bug-fixed, ties R@10 = 0.6406 but is weaker in isolation.

Note: the R100-R103 reference run (exp:20260419_combo_plan) found explicit_only + v1_enriched + static_plus_neighbor = R@10 0.6522 — but this used a different baseline setup. A clean ablation (both configs on same v1_enriched corpus) is still needed.

## Scope

- Model: Qwen3-Embedding-0.6B.
- Corpus: `v1_enriched` only.
- Metric: primarily R@10. R@1 / MRR gains are within noise or negative under `static_plus_neighbor`.
- Citation boost contribution is currently unresolved due to very low bbl coverage.

## Risk

- Citation boost may turn out to be near-zero even with broader bbl coverage.
- Typed cross-doc may fail to transfer when the retrieval corpus changes to `v2_chunks` or when combined with chunk-v2 graph rerank (confirmed: hurts R@10 with v2chunk, see exp:20260419_combo_plan).

## Connections

- Tested by: `exp:20260419_typed_crossdoc`
- Originates from: `idea:004`
- Related to: `claim:C3` (this supersedes C3's role for element-level gains).
- Superseded by: `claim:C7` (R@10 high title only).
- Addresses: `gap:G1`, `gap:G2`.
