---
type: experiment
node_id: exp:20260419_combo_plan
status: completed
verdict: negative
created_at: 2026-04-19T13:00:00Z
updated_at: 2026-04-19T16:30:00Z
---

# One-line summary

Combining typed cross-doc element edges with v2chunk corpus under various weight configs does NOT improve over explicit_only — and explicit_only + static_plus_neighbor on v1_enriched unexpectedly produces the new project-high R@10 = 0.6522.

## Motivation

Two current best configs were disjoint:
- REF-B: explicit_only + chunk_v2 + static_prior → R@1=0.2505, R@10=0.5391
- REF-C: explicit+typed(w=0.2) + chunk_v1 + static_plus_neighbor → R@1=0.1818, R@10=0.6406

Hypothesis: typed_crossdoc under static_prior (not neighbor-prop) with chunk_v2 graph should preserve R@1 from REF-B while adding cross-doc diversity to push R@10 toward REF-C.

## Results (Job 61463, COMPLETED 2026-04-19T04:30 UTC)

SLURM job: `32_combo_typed_chunkv2.sh` (CPU-only, ~8 sec wall time)

### static_prior mode

| config | R@1 | R@5 | R@10 | MRR |
|--------|-----|-----|------|-----|
| explicit_only_v1enriched_ref | **0.2347** | **0.5476** | 0.6216 | **0.6160** |
| explicit_typed_w01_v2chunk | 0.2336 | 0.5444 | **0.6226** | 0.6114 |
| explicit_typed_w02_v2chunk | 0.2304 | 0.5381 | 0.6195 | 0.6060 |
| explicit_typed_w02_boosted_v2chunk | 0.2304 | 0.5381 | 0.6195 | 0.6060 |
| explicit_typed_w03_v2chunk | 0.2315 | 0.5317 | 0.6099 | 0.6049 |

### static_plus_neighbor mode

| config | R@1 | R@5 | R@10 | MRR |
|--------|-----|-----|------|-----|
| explicit_only_v1enriched_ref | 0.2008 | **0.5655** | **0.6522** ← NEW HIGH | 0.5731 |
| explicit_typed_w01_v2chunk | 0.1977 | 0.5581 | 0.6512 | 0.5646 |
| explicit_typed_w02_v2chunk | 0.1818 | 0.5423 | 0.6406 | 0.5413 |
| explicit_typed_w02_boosted_v2chunk | 0.1818 | 0.5423 | 0.6406 | 0.5413 |
| explicit_typed_w03_v2chunk | 0.1786 | 0.5254 | 0.6374 | 0.5275 |

## Key Findings

1. **Success criterion NOT met**: No config achieves R@1 ≥ 0.25 AND R@10 ≥ 0.62 simultaneously.

2. **Surprise: new R@10 high = 0.6522** — `explicit_only + v1_enriched + static_plus_neighbor`, which was an incidental reference run. Previous high was 0.6406 (typed_crossdoc, C5).

3. **typed_crossdoc hurts R@10 with v2chunk corpus**: Adding typed edges monotonically lowers R@10 in static_plus_neighbor mode. This confirms C5's scope caveat — typed edge gain was measured on v1_enriched only.

4. **Root cause hypothesis**: v2chunk corpus has more passages → neighbor-prop signal gets diluted more when typed_crossdoc edges expand the graph. The extra cross-doc connectivity backfires in neighbor mode because hub-degree normalization shifts.

5. **Clean comparison still needed**: To fairly evaluate typed edges vs explicit_only at R@10, must use identical corpus (v1_enriched for both). R100-R103 inadvertently used v2chunk for the combo configs but v1_enriched for the reference.

## Updated Best Configs

| metric | best config | value |
|--------|------------|-------|
| R@1 | explicit+chunk_v2 + 0.6B rerank (prior, REF-B) | 0.2505 |
| R@10 | explicit_only + v1_enriched + static_plus_neighbor | **0.6522** (NEW) |

## Implication for Claims

- C5 (typed_crossdoc lifts R@10 to 0.6406) remains supported **on v1_enriched corpus**. The 0.6522 reference run used v1_enriched + explicit_only, not typed edges — so C5 is NOT overturned, but its claim of "project high" is now superseded.
- New claim C7 proposed: explicit_only + static_plus_neighbor on v1_enriched achieves R@10 = 0.6522, the new project high.

## Connections

- Follows: `exp:20260419_typed_crossdoc`, `exp:20260419_deliverable_420`
- Invalidates: hypothesis that explicit+typed_crossdoc beats explicit_only on v1_enriched with v2chunk corpus
- Addresses: `gap:G1`, `gap:G2`
