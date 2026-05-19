---
type: claim
node_id: claim:C13
status: reported
created_at: 2026-05-19T00:00:00Z
updated_at: 2026-05-19T00:00:00Z
---

# Claim

A multi-turn session derived from a multi-hop chain (per `idea:006` projection rule) **preserves** retrieval difficulty (R@10 within 5pp of the chain-QA view on the same evidence set) **and adds** turn-dependency as an additional measurable difficulty axis. The session view is not strictly easier or strictly harder than chain-QA view — it tests a different skill.

## Evidence

- Reported only — no experiment run yet.
- Will be validated by `exp:20260519_chain_to_session` (planned) and `exp:trinity_benchmark` (planned).
- Negative-evidence prior: nothing yet specifically refutes; existing multi-turn generator (`generate_multiturn_sessions.py` v2) has not been evaluated against same-chain chain-QA controls.

## Scope

Applies to chains derived from the 145 existing L3 pass queries (initial validation set) and any cross-doc chains produced by Phase 2 going forward.

## Why It Matters

- If true: session-form is a free additional benchmark axis with no quality regression.
- If false (session significantly easier): the multi-turn dimension is not actually adding evaluation signal; the multi-turn view becomes window dressing for the paper.
- If false (session significantly harder): coref / ellipsis are creating a different problem than reasoning; need to decide whether that's a feature or a confound.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
