---
type: idea
node_id: idea:006
title: "Multi-turn sessions as projections of multi-hop chains (not generated from scratch)"
stage: proposed
outcome: untested
dominance: candidate
created_at: 2026-05-19T00:00:00Z
updated_at: 2026-05-19T00:00:00Z
---

# One-line thesis

Instead of generating multi-turn dialog sessions from scratch with a fresh LLM call, **derive** sessions from existing multi-hop chains by rewriting each hop as one turn. The chain's `reasoning_steps[]` already encode the per-hop evidence and answer; the rewriter only needs to add coreference, ellipsis, and intent shifts on top.

## Problem / Gap

Current `scripts/generate_multiturn_sessions.py` v2 generates sessions from scratch with persona + intent-shift prompts. This:

1. Wastes LLM tokens reconstructing reasoning that the L3 chain already proved out.
2. Loses the per-turn evidence guarantee — sessions generated end-to-end don't have a clear mapping from turn N to its grounding element.
3. Cannot reuse the existing turn-dependency QC (since there's no canonical "delete this step" operation if the session isn't structurally a chain).

## Mechanism

Input: an L3 chain `C = [e_1, e_2, e_3, e_4]` with `reasoning_steps = [s_1, s_2, s_3]` (3 hops between 4 elements).

Projection rules:

- **Turn 1**: question grounded in `e_1`, answer = `s_1.intermediate_conclusion`.
- **Turn 2**: follow-up using **coreference / ellipsis** referencing turn 1's content; new evidence = `e_2`; answer = `s_2.intermediate_conclusion`. Bridge text (if `e_1 → e_2` is cross-doc) is the natural inter-turn pivot.
- **Turn 3**: same pattern, evidence `e_3`, answer = `s_3.intermediate_conclusion` = chain's final answer.

QC tests applied to derived sessions:

- **turn-dependency**: delete turn N's content from history, ask turn N+1 → answer should drop or become unanswerable.
- **coref_resolution_required**: turn N+1 must contain at least one referring expression (pronoun / definite NP / ellipsis) whose antecedent is in turn ≤ N.
- **retrievability_score** (reuse from `qc_real_user_query()`): each turn's answer evidence must be retrievable given the turn's user query in isolation (gold passage rank ≥ some threshold).

## Current Status

Proposed 2026-05-19. Depends on `idea:005` (chains as atomic units). Concrete reuse target: the 145 pass L3 chains in `data/03_queries/l3_enriched_v3_rerun2_pass.jsonl` + `l3_enriched_v3_new82_rerun2_pass.jsonl`.

## Why It Matters

- Avoids one LLM-generation pass per session (cost saving).
- Inherits L3 chains' validated multi-hop structure — no risk of generating "session that looks multi-turn but isn't actually multi-hop".
- Makes turn-dependency QC tractable: delete turn N = delete hop N's evidence, which has a clear definition.
- Unblocks Phase 3 immediately for any cross-doc chain produced by Phase 2 (no new generation pipeline needed).

## Failure Modes

- Mechanical chain-to-turn projection may produce robotic dialog (every turn explicitly says "building on the previous answer..."). Need a stylistic post-pass with persona variation.
- Some chains may have nodes that are difficult to verbalize as a standalone follow-up question (e.g., a bridge formula step that's purely mathematical). May need to drop or merge such hops.
- Risk of overfitting the projection rule to current L3 prompt format — if L3 prompts change, projection may break.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
