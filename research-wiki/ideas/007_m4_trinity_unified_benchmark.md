---
type: idea
node_id: idea:007
title: "Unified M4 trinity benchmark: one chain × three views"
stage: proposed
outcome: untested
dominance: candidate
created_at: 2026-05-19T00:00:00Z
updated_at: 2026-05-19T00:00:00Z
---

# One-line thesis

Evaluate M4 with a benchmark that holds the **underlying chain constant** and varies only the **view** (atomic QA / chain QA / multi-turn session). This isolates which dimensions of difficulty are due to multi-hop reasoning vs cross-doc retrieval vs multi-turn coreference, instead of conflating them.

## Problem / Gap

Current evaluation suite (Exp A difficulty, Exp B retrieval, Exp C QA triangle) compares L1/L2/L3 levels, which differ in **multiple ways at once**: hop count, document span, evidence count, query style. You cannot tell whether L3's lower pass rate comes from harder reasoning or harder retrieval.

The trinity framing (`idea:005`) gives a natural way to fix this: take a single chain, project it 3 ways, and compare the projections on the same retrieval / QA / judge metrics.

## Mechanism

For each chain in a fixed evaluation set of ~100 cross-doc chains:

1. **Atomic view**: ask a question grounded only in the chain's final node `e_n`. Gold passage = `e_n`'s parent chunk.
2. **Chain-QA view**: the existing L3 multi-hop query. Gold passages = all `e_i` chunks.
3. **Session view**: derived per `idea:006`. Gold passage for turn N = `e_{N+1}` chunk; turn-dependency required.

Metrics, all reported per view:

- **Retrieval**: BM25 R@10 / MRR, graph_full R@10 / MRR (reuse Phase0 eval pipeline).
- **QA**: answer-correctness via LLM judge (gpt-5.5 vs claude cross-model).
- **Reasoning difficulty**: cross-model LLM judge difficulty rating on a 5-point scale.
- **View-specific**:
  - atomic: trivially easy expected; serves as floor.
  - chain-QA: reasoning-difficulty signal.
  - session: turn-dependency rate (% turns where deleting prior turn breaks current).

## Current Status

Proposed 2026-05-19. Depends on `idea:005` (chain as atomic unit) and `idea:006` (session as projection). Reuses existing Exp A/B/C evaluation infrastructure with no schema change.

## Why It Matters

- Provides the first principled answer to "does multi-turn add difficulty, or just lengthen the query?"
- Yields a clean ablation for the M4 benchmark paper: "Our dataset has these three views; here's how they differ in difficulty, and here's the unique contribution of each."
- Makes mentor-facing positioning easier: "M4DocBench (Dong's) tests retrieval on a fixed view; ours tests retrieval × view, isolating the contribution of multi-turn."

## Failure Modes

- If atomic and session R@10 turn out the same, the multi-turn dimension may be window dressing — would need to add coref-pressure (deliberately remove explicit references) to actually test the dimension.
- 100 chains × 3 views = 300 queries; LLM-judge cost is non-trivial. Need to budget ~$10-20 per round.
- Cross-model judge disagreement could swamp the signal; need multiple judge passes and report agreement.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
