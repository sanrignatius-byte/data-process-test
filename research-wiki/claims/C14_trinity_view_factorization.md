---
type: claim
node_id: claim:C14
status: reported_phase0_required
created_at: 2026-05-19T00:00:00Z
updated_at: 2026-05-19T02:40:00Z
---

# Claim

Holding the underlying multi-hop chain constant and varying only the **view** (atomic / chain-QA / session) factorizes M4 difficulty into separable components: retrieval cost, reasoning cost, and dialog-coreference cost. Specifically: on a fixed set of 100 cross-doc chains, no single view dominates on all metrics — atomic wins R@10, chain-QA wins reasoning-difficulty rating, session wins coreference-required rate.

## Evidence

- Reported only — no experiment run yet.
- Will be validated by `exp:trinity_benchmark` (planned, per `idea:007`).
- Phase 0 dependency: `idea:006` must first validate a 2-turn endpoint-session projection or upstream generation must emit structured `reasoning_steps[]`. Otherwise the session view is not a controlled projection of the same chain.

## Scope

Applies to a controlled benchmark set of ~100 cross-doc chains; the three-view comparison is the test.

## Why It Matters

- If true: M4 paper has a clean ablation story — each dimension contributes uniquely.
- If false (one view dominates): collapse to that view as the canonical M4 unit and drop the others.
- Mentor-facing: this is the principled way to argue "we are different from M4DocBench" — same underlying chains, different evaluation views.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
