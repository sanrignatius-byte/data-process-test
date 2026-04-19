---
type: claim
node_id: claim:C2
status: supported
created_at: 2026-04-18T00:00:00Z
updated_at: 2026-04-18T07:00:00Z
---

# Claim

Adding the current full set of intra-document virtual edges dilutes precision-oriented rerank quality relative to explicit-only rerank.

## Evidence

- `exp:20260417_explicit_rerank_fixed` shows explicit-only + static prior beats explicit+all-virtual + static prior on MRR and Recall@1.

### Root Cause (2026-04-18)

- `static_prior` is computed as `log(degree) / log(max_degree)`, so adding `same_chunk` and `chunk_sequence` edges increases the degree of many co-passage nodes that are topically nearby but not the answer.
- This is a **ranking problem, not a coverage problem**. `explicit_only_fixed` and `all_virtual_fixed` have identical `recall@100 = 0.8636` — the gold answer is still retrieved either way.
- The degradation appears only at top ranks: `hit@1` drops 0.484 → 0.431, `median_first_hit_rank` worsens 1 → 2.
- Interpretation: virtual edges do not remove the answer from consideration; they amplify degree-based priors for non-answer nodes sharing the same local passage neighborhood, causing distractors to outrank the correct node.

## Scope

Applies to the current virtual-edge construction that includes local chunk and same-section style edges used inside `static_prior` scoring. Does NOT apply to graph-traversal retrieval (multi-hop), where these edges serve a different role.

## Why It Matters

- Sharpens the claim from "virtual edges can hurt precision" to a concrete mechanism: **degree inflation leaks structural credit to co-passage distractors through the `static_prior` term**.
- Because recall is unchanged while first-hit rank worsens, the fix should target degree-sensitive scoring or virtual-edge weighting — not broader retrieval expansion.
- Intra-doc virtual edges are not inherently useless: for multi-hop traversal (hop 2+ within a document), they remain necessary. The problem is the current scorer treats cheap local connectivity as importance evidence.

## Connections

- Supported by: `exp:20260417_explicit_rerank_fixed`
- Informs: `idea:002`

