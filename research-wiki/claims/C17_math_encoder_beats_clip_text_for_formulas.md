---
type: claim
node_id: claim:C17
status: supported
created_at: 2026-05-20T00:00:00Z
updated_at: 2026-05-20T00:00:00Z
---

# Claim

A math-aware sentence encoder (`math-similarity/Bert-MLM_arXiv-MP-class_arXiv`) is the
right backend for formula-similarity edges; CLIP's text encoder collapses formula
similarity into a narrow high band and cannot separate related from coincidental
formulas, making any top-k/threshold cut meaningless.

## Evidence

- `exp:20260520_mineru_clip_xdoc_pipeline`, `probe_formula_embedding.py`
  (`data/05_eval/probe_formula_embedding/report.json`), 200-formula sample.
- CLIP text: p50=0.921, p99=0.974, **std=0.027** (min 0.77 — everything looks similar).
- math model: p50=0.817, p99=0.977, **std=0.172**, min=0.036 — wide, usable spread.
- Wired into `build_mineru_vl_edges.py` as `--formula-backend math_similarity` (768-d,
  independent of open_clip visual/text); formula threshold auto 0.45→0.85.
- Full run: formula_similarity 4331 edges, embedding shape [876, 768].

## Scope

LaTeX-text formulas extracted by MinerU. Orthogonal to the visual/caption pipeline.
Note: the encoder is a similarity model, so high score = related formulas, not a
reference edge; a symbol-set Jaccard re-check on top edges is a sensible future guard.

## Why It Matters

Closes the May 19 `f_formula_*` line's open issue (CLIP-text formula scores were
undiscriminative). Formula edges are now meaningful similarity edges in the MinerU graph.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
