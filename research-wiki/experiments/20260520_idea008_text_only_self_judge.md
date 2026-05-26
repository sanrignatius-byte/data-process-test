---
type: experiment
node_id: exp:20260520_idea008_text_only_self_judge
title: "idea:008 text-only Codex self-judge probe on 32 xdoc candidates"
date: 2026-05-20
status: text_only_probe_completed
lane: experimental
---

# idea:008 text-only self-judge probe

## Purpose

Run a low-cost feasibility probe on `exp:20260520_idea008_phase0_judge_pack`
before spending VLM/API budget. This judge used captions, local context, enriched
previews, and scores only. It did **not** inspect image pixels, so it is not a
replacement for the planned VLM judge.

## Input

- Source pack: `data/05_eval/idea008_phase0_judge_pack_latest/phase0_candidates.jsonl`
- Judged subset: 32 balanced candidates
- Output: `data/05_eval/idea008_phase0_self_judge_text_only_latest`

## Protocol

Verdicts:

- `strong_edge`: shared method/dataset/metric/scientific role is tight enough for a hard graph edge.
- `weak_related`: thematic or form-related; useful for recall but not promotion.
- `visual_layout_only`: CLIP/layout match with no scientific edge.
- `unrelated`: no meaningful relation.
- `insufficient`: text/context too poor to decide.

Promotion rule for a real judge:

- Promote only `strong_edge` with confidence >= 0.70.
- Keep `weak_related` as retrieval recall only.
- Reject `visual_layout_only`, `unrelated`, and `insufficient`.

## Results

Judged candidates: 32

Verdicts:

- `strong_edge`: 7
- `weak_related`: 10
- `visual_layout_only`: 8
- `unrelated`: 4
- `insufficient`: 3

Text-only strong-edge rate: 21.875% (7/32).

High-confidence promotable strong edges (`strong_edge`, confidence >= 0.70): 7/32.

Hard rejects (`visual_layout_only` + `unrelated` + `insufficient`): 15/32.

Non-promotable including `weak_related`: 25/32.

## Interpretation

The recall+judge pipeline is feasible: the candidate pool contains real strong
cross-document semantic edges, including OntoNotes/WinoBias coreference-bias tables,
Amazon Reviews domain-adaptation tables (inspected separately), causal DAGs for
fairness/path-specific counterfactual reasoning, and unsupervised fair-representation
model diagrams.

But the current CLIP/rerank edge list is not safe for direct graph promotion. The
same subset contains many layout false positives: hollow-square markers, generic
histogram-like distributions with different variables, blank/cropped panels, and
weak thematic fairness-table links.

## Decision

Proceed to a real VLM judge over all 160 Phase 0 candidates. Success should be judged
by:

1. strong-edge rate after VLM judging;
2. false-positive rate among high-CLIP/low-text candidates;
3. whether degraded-caption buckets recover strong edges that text-only judging misses.

Until then, no xdoc visual edge should be promoted into M4 chain construction solely
from CLIP/rerank scores.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
