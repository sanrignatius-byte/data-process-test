---
type: idea
node_id: idea:005
title: "Cross-doc multi-hop chain as the atomic M4 unit"
stage: proposed
outcome: untested
dominance: candidate
created_at: 2026-05-19T00:00:00Z
updated_at: 2026-05-19T00:00:00Z
---

# One-line thesis

Stop treating "multi-hop", "cross-document", and "multi-modal" as three orthogonal axes to be ticked off independently. Treat the **cross-doc multi-hop chain** as the atomic M4 data unit; all three dimensions are properties of the chain's topology, not separate dimensions to be combined post-hoc.

## Problem / Gap

The current M4 roadmap (Phase 2 cross-doc → Phase 3 multi-turn → Phase 4 M4 verify) implicitly assumes the three remaining dimensions are independent. But:

- A multi-hop chain whose nodes live in 2+ documents is inherently cross-document.
- A multi-hop chain whose nodes span figure / table / formula is inherently multi-modal.
- A multi-hop chain re-narrated as a dialogue is inherently multi-turn (one hop → one turn).

Building Phase 2 and Phase 3 as separate workstreams duplicates schema work and risks producing artifacts that satisfy only one dimension each.

## Mechanism

Define the M4 atomic unit as a **chain** `C = [e_1, e_2, ..., e_n]` with:

- `cross_doc_bridge_count ≥ 1` (at least one consecutive pair spans documents)
- `modality_set = {figure, table, formula} ∩ types(C) ≠ ∅` (at least one non-text node)
- `hop_count = n - 1 ≥ 2` (true multi-hop)

From this unit, derive **three views** by projection:

| View | Form | What it tests |
|------|------|---------------|
| `atomic` | Single QA grounded in `e_n` | Baseline retrieval |
| `chain_qa` | Multi-hop QA over the full chain (current L3) | Reasoning depth |
| `session` | Multi-turn dialog, one turn per hop | Coreference + turn dependency |

Both Phase 2 (cross-doc) and Phase 3 (multi-turn) reduce to chain-level operations:

- Phase 2 = "make sure chains have cross-doc bridges" (chain-mining filter)
- Phase 3 = "project chains to dialog form" (chain-to-session rewriter)

## Current Status

Proposed 2026-05-19. Inspired by user observation that the three remaining M4 dimensions are "三位一体" — they belong to the same artifact and shouldn't be separated into independent workstreams. Builds directly on `src/pairing/` (intra-doc pairing already abstracts "chain" as a first-class object).

## Why It Matters

- Halves the schema design work (one Chain schema, not two: one for Phase 2 cross-doc bridge + one for Phase 3 multi-turn session).
- Guarantees joint coverage — every M4 sample satisfies all 4 dims by construction, no "passes Phase 2 but fails Phase 3" failure mode.
- Maps cleanly to existing assets: the 145 L3 pass chains are already candidate M4 units; only need to filter for cross-doc + project to session.
- Natural fit with `idea:004` (typed cross-doc element edges) — those edges are the bridges that make a chain cross-doc.

## Failure Modes

- If cross-doc element-level bridges are too sparse (current data has only 123 doc-level citation edges and Qwen3-embedding-based matches that need rerank validation), the atomic unit will collapse to intra-doc chains and Phase 2 won't get real coverage.
- If the chain-to-session projection introduces artifacts (e.g., turns that don't actually depend on each other), the multi-turn view becomes window dressing rather than a real test.
- Risk that "atomic unit" framing hides legitimate independent value of cross-doc-only or multi-turn-only data (e.g., for ablation studies that vary only one dimension).

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
