---
type: idea
node_id: idea:006
title: "Multi-turn sessions as projections of multi-hop chains (not generated from scratch)"
stage: proposed_revised
outcome: untested
dominance: candidate
created_at: 2026-05-19T00:00:00Z
updated_at: 2026-05-19T02:40:00Z
---

# One-line thesis

Instead of generating multi-turn dialog sessions from scratch with a fresh LLM call, **derive** sessions from existing multi-hop chains where possible. Current L3 pass data does **not** contain structured `reasoning_steps[]`, so v1 must first verbalize / split the existing `reasoning_chain` into locked per-turn fields; only then should a lightweight rewriter add coreference, ellipsis, and intent shifts.

## Problem / Gap

Current `scripts/generate_multiturn_sessions.py` v2 generates sessions from scratch with persona + intent-shift prompts. This:

1. Wastes LLM tokens reconstructing reasoning that the L3 chain already proved out.
2. Loses the per-turn evidence guarantee — sessions generated end-to-end don't have a clear mapping from turn N to its grounding element.
3. Cannot reuse the existing turn-dependency QC (since there's no canonical "delete this step" operation if the session isn't structurally a chain).

## Mechanism

Input v1: an L3 pass row with endpoint elements `element_ids=[e_a, e_b]`, a `path` such as `e_a → ::p:: → e_b` or `e_a → ::p:: → ::p:: → e_b`, and a free-text `reasoning_chain`. Audit result (2026-05-19): 146/146 rows have empty `reasoning_steps[]`, path length distribution is `{3: 48, 4: 98}`, and 87/146 paths are cross-doc.

Projection rules:

- **Phase 0 verbalize**: locked-schema LLM pass extracts `turn1_answer`, `bridge_pivot`, `turn2_answer`, and evidence IDs from `reasoning_chain + path`. This is not a free-form session generator; it is a structural parser with fixed outputs.
- **Turn 1**: question grounded in endpoint `e_a`, answer = extracted `turn1_answer`.
- **Turn 2**: follow-up using **coreference / ellipsis** referencing turn 1's answer; new evidence = endpoint `e_b`; answer = extracted `turn2_answer`. Paragraph bridge text (`::p::`) is the inter-turn pivot, not an independent element turn.
- **v2 only**: 3+ turns require true multi-element paths or upstream non-empty `reasoning_steps[]`.

QC tests applied to derived sessions:

- **turn-dependency**: delete turn N's content from history, ask turn N+1 → answer should drop or become unanswerable.
- **coref_resolution_required**: turn N+1 must contain at least one referring expression (pronoun / definite NP / ellipsis) whose antecedent is in turn ≤ N.
- **retrievability_score** (reuse from `qc_real_user_query()`): each turn's answer evidence must be retrievable given the turn's user query in isolation (gold passage rank ≥ some threshold).

## Current Status

Proposed 2026-05-19, revised after red/blue-team audit. Depends on `idea:005` (chains as atomic units). Concrete reuse target: the pass L3 chains in `data/03_queries/l3_enriched_v3_rerun2_pass.jsonl` + `l3_enriched_v3_new82_rerun2_pass.jsonl`, but v1 is explicitly a **2-turn endpoint projection** with a locked verbalization step.

## Why It Matters

- Avoids full session generation from scratch; still requires a smaller locked-schema verbalization pass because existing `reasoning_steps[]` are empty.
- Inherits L3 chains' validated multi-hop structure — no risk of generating "session that looks multi-turn but isn't actually multi-hop".
- Makes turn-dependency QC tractable: delete turn N = delete hop N's evidence, which has a clear definition.
- Unblocks Phase 3 immediately for any cross-doc chain produced by Phase 2 (no new generation pipeline needed).

## Failure Modes

- Mechanical chain-to-turn projection may produce robotic dialog (every turn explicitly says "building on the previous answer..."). Need a stylistic post-pass with persona variation.
- Current data supports mostly 2-turn sessions, not the original 3-turn example; claiming 3-turn coverage without upstream `reasoning_steps[]` would be invalid.
- Retrieval evaluation inherits G7/C8/C11 style mismatch: paper-domain text-style turns may retrieve visual/formula passages poorly even if the session itself is coherent. C13 thresholds must account for this confound.
- Some chains may have nodes that are difficult to verbalize as a standalone follow-up question (e.g., a bridge formula step that's purely mathematical). May need to drop or merge such hops.
- Risk of overfitting the projection rule to current L3 prompt format — if L3 prompts change, projection may break.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
