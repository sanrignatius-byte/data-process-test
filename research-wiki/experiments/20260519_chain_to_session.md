---
type: experiment
node_id: exp:20260519_chain_to_session
status: planned_phase0_required
created_at: 2026-05-19T00:00:00Z
updated_at: 2026-05-19T02:40:00Z
---

# Experiment: Chain → session projection on existing L3 pass set

> **Lane rule (2026-05-19 blue-team correction)**: this is Track A / experimental-lane validation. Do **not** modify production query-generation scripts first. Prototype projection and QC under `experiments/` or `scripts/experimental_*`; only promote reusable pieces into `src/` after Phase 0/1 gates pass.

Validate `idea:006` by projecting the existing 145/146 L3 pass chains into multi-turn session form. Blue-team correction: current pass files do **not** contain structured `reasoning_steps[]` (146/146 empty) and their paths are endpoint-element chains with paragraph bridges, not 4-element chains. Therefore v1 must be a **2-turn endpoint session** plus a small LLM verbalization / step-splitting pass, not a pure deterministic 3-turn projection.

## Design

- **Input**:
  - `data/03_queries/l3_enriched_v3_rerun2_pass.jsonl` (93 chains)
  - `data/03_queries/l3_enriched_v3_new82_rerun2_pass.jsonl` (53 chains)
  - Total: 145 chains (1 duplicate after merge)
  - Observed schema (2026-05-19 audit): `reasoning_steps=[]` for all rows; `path` length distribution `{3: 48, 4: 98}`; 87/146 paths are cross-doc; `element_ids` always has 2 endpoint elements.

- **Projection rule**:
  - v1: endpoint chain → 2-turn session (`element_a` context → follow-up grounded in `element_b`)
  - Paragraph bridge nodes (`::p::`) become the natural inter-turn pivot; do not pretend they are independent element hops.
  - Because `reasoning_steps[]` is empty, first run a locked-schema verbalization pass over `reasoning_chain` + `path` to extract `turn1_answer`, `bridge_pivot`, `turn2_answer`, and per-turn evidence IDs.
  - A future v2 may support 3+ turns only if upstream L3 generation emits non-empty structured `reasoning_steps[]` or true multi-element paths.

- **Style pass**:
  - Mechanical projection is brittle; pass through one LLM call per session to add persona / coreference variation, but **lock evidence and answer fields** so QC reproducibility is preserved.
  - All company-provider calls must go through `local_api_logger` and `log_run()` per repo rules.

- **QC**:
  - `turn_dependency_score`: for each turn N ≥ 2, blank out turn (N-1)'s assistant message and re-ask turn N to LLM. If LLM still answers correctly → fail this turn (no real dependency).
  - `coref_resolution_required`: each turn N ≥ 2 must contain ≥ 1 referring expression (pronoun / def NP / ellipsis) whose antecedent is in earlier turns. Detected by simple rule + LLM check.
  - `evidence_grounding`: reuse `judge_answer_grounding()` per-turn.

- **Output**:
  - `data/03_queries/l3_sessions_v1.jsonl` — one session per chain
  - `data/05_eval/l3_session_qc_report.json` — per-session per-turn QC verdicts

## Pre-registered hypothesis

- Turn-dependency rate ≥ 70% for turn 2 (v1 has only one deletion-sensitive follow-up turn).
- Coref rate = 100% (rule-enforced).
- Per-turn evidence-grounding pass rate ≥ 80% (since we lock evidence from the source chain).
- Overall session-level pass (all turns pass) ≥ 50%.

Blue-team / red-team constraint from G7/C8/C11: M4query-style retrieval has a known style mismatch. Paper-domain text queries often fail against visual-style or formula-style passages; MODORA visual enrichment and formula caption injection both regressed retrieval. Therefore C13 should not use an unrealistically tight ±5pp preservation threshold without confidence intervals. Treat retrieval deltas larger than ±10pp as actionable; for smaller deltas report CI / bootstrap rather than declaring a win/loss.

If turn-dependency < 50%, the projection rule needs to inject explicit context-elision (drop entities, force pronouns).

## Phase 0 gates (must pass before productionization)

- Confirm merged input count and duplicate handling (observed: 146 rows before dedup, not a guaranteed 145 usable sessions).
- Write an experimental schema for the locked verbalization output; do not alter production query schemas yet.
- Run 10 sessions end-to-end under experimental outputs and manually inspect turn dependency / grounding.
- Only after Phase 0 passes, consider moving reusable parsing/QC helpers into `src/`.

## Status

Planned 2026-05-19. Phase 0 required. Blocking: current data lacks structured reasoning steps, so v1 design must use 2-turn endpoint sessions plus locked verbalization. Estimated effort: 0.5 day Phase 0 + 2-3 days for projection + style pass + QC.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
