---
type: experiment
node_id: exp:20260519_chain_to_session
status: planned
created_at: 2026-05-19T00:00:00Z
updated_at: 2026-05-19T00:00:00Z
---

# Experiment: Chain → session projection on existing L3 pass set

Validate `idea:006` by projecting the existing 145 L3 pass chains into multi-turn session form using a deterministic chain-to-turn rule (1 hop = 1 turn). Test whether turn-dependency is real and whether the projection is stylistically natural enough to use as training data.

## Design

- **Input**:
  - `data/03_queries/l3_enriched_v3_rerun2_pass.jsonl` (93 chains)
  - `data/03_queries/l3_enriched_v3_new82_rerun2_pass.jsonl` (53 chains)
  - Total: 145 chains (1 duplicate after merge)

- **Projection rule**:
  - 3-step chain → 3-turn session
  - Turn N's user query = a follow-up phrased as if continuing from turn (N-1)'s answer
  - Turn N's evidence = hop N's `evidence_spans` from the chain
  - Bridge text between hops is the natural inter-turn pivot

- **Style pass**:
  - Mechanical projection is brittle; pass through one LLM call per session to add persona / coreference variation, but **lock evidence and answer fields** so QC reproducibility is preserved.

- **QC**:
  - `turn_dependency_score`: for each turn N ≥ 2, blank out turn (N-1)'s assistant message and re-ask turn N to LLM. If LLM still answers correctly → fail this turn (no real dependency).
  - `coref_resolution_required`: each turn N ≥ 2 must contain ≥ 1 referring expression (pronoun / def NP / ellipsis) whose antecedent is in earlier turns. Detected by simple rule + LLM check.
  - `evidence_grounding`: reuse `judge_answer_grounding()` per-turn.

- **Output**:
  - `data/03_queries/l3_sessions_v1.jsonl` — one session per chain
  - `data/05_eval/l3_session_qc_report.json` — per-session per-turn QC verdicts

## Pre-registered hypothesis

- Turn-dependency rate ≥ 70% (most turns will be deletion-sensitive).
- Coref rate = 100% (rule-enforced).
- Per-turn evidence-grounding pass rate ≥ 80% (since we lock evidence from the source chain).
- Overall session-level pass (all turns pass) ≥ 50%.

If turn-dependency < 50%, the projection rule needs to inject explicit context-elision (drop entities, force pronouns).

## Status

Planned 2026-05-19. Not yet run. Blocking: chain-to-session script not yet written. Estimated effort: 2-3 days (rule + style pass + QC).

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
