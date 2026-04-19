---
type: claim
node_id: claim:C7
status: supported
created_at: 2026-04-19T16:30:00Z
updated_at: 2026-04-19T16:30:00Z
---

# Claim

`explicit_only + v1_enriched corpus + static_plus_neighbor` achieves R@10 = 0.6522, the current project-high for R@10, surpassing the prior best of 0.6406 (C5: explicit+typed_crossdoc w=0.2).

## Evidence

From `exp:20260419_combo_plan` (R100-R103 reference run, job 61463):
- `explicit_only_v1enriched_ref + static_plus_neighbor`: R@10 = **0.6522**, R@1 = 0.2008, MRR = 0.5731
- Prior best (C5): R@10 = 0.6406 (explicit + typed_crossdoc w=0.2, static_plus_neighbor)

## Scope

- Model: Qwen3-Embedding-0.6B
- Corpus: `v1_enriched` (1798 passages)
- Graph: explicit edges only (hub_pair + adjacent_bridge, 342 pids, 1324 directed edges)
- Metric: R@10. R@1 (0.2008) is below the R@1 best of 0.2505 — this is a R@10-only high.

## Interpretation

The gain over C5 does NOT mean typed_crossdoc edges are harmful in general. The R100-R103 experiment inadvertently compared explicit_only on v1_enriched against explicit+typed on v2chunk corpus — an unfair comparison. A clean ablation (both configs on v1_enriched) is still needed to determine whether typed edges genuinely help or hurt R@10.

## Risk

- If a clean v1_enriched ablation shows typed_crossdoc hurts R@10, this strengthens C7 but raises a design problem.
- R@10 = 0.6522 was not the primary target run; it needs a dedicated confirmation run to rule out randomness (deterministic embedding → deterministic result, but corpus version should be double-checked).

## Connections

- Tested by: `exp:20260419_combo_plan`
- Supersedes: C5 as "project R@10 high"
- Addresses: `gap:G1`, `gap:G2`
