---
id: exp:20260519_mineru_only_migration
title: "MinerU-only migration after LaTeX-centric attempts"
date: 2026-05-19
status: planned_phase1_required
lane: experimental
---

# MinerU-only migration after LaTeX-centric attempts

## Motivation

Mentor direction is PDF-first: LaTeX is useful when available but cannot be the core pipeline assumption. Recent xdoc+xmodal API smoke suggests MinerU/PDF-local evidence can produce usable queries; the next step is to build the graph/query substrate from pure MinerU artifacts.

## Audit

Baseline audit script:

- `experiments/audit_mineru_only_migration.py`

Output:

- `data/05_eval/mineru_only_migration_audit_20260519T061252Z/report.md`
- `data/05_eval/mineru_only_migration_audit_20260519T061252Z/summary.json`

Key result:

- 1153 MinerU doc dirs exist, but only 80 currently expose `structure.json`.
- Current `multimodal_elements.json` covers 76 docs / 1316 elements.
- Raw structure has 2560 elements: formula 1447 / figure 1017 / text 88 / table 8.
- Existing LaTeX dependency hotspots are concentrated in topology, query generation, hub enrichment, line-number remapping, and delivery scripts.

## Working hypothesis

Pure MinerU should use `(page_idx, position_idx, image_path/bbox, caption/content, local context)` as the canonical grounding substrate. Text paragraphs must be first-class elements. LaTeX label/path/line-number signals become legacy controls only.

## Planned phases

1. Build `mineru_elements_v0.json` and `mineru_edges_v0.jsonl` from `structure.json` directly. First run complete: `data/05_eval/mineru_only_graph_v0_20260519T061721Z/`.
2. Add local edge families: reading order, same-page window, regex reference, context containment, layout-near.
3. Generate same-doc xmodal and xdoc+xmodal candidates without LaTeX labels.
4. Run small company API smoke through `api_logs_cannt_delete` + `logs/token_usage.db`.
5. Evaluate retrieval with MinerU-only edges before considering any production merge.

## Gate

- Phase 1 must explain coverage gaps: why 1073/1153 dirs lack `structure.json`, or where the missing structured outputs live.
- Phase 2 must produce source-diverse candidates with manual all-node-necessary QC.
- Phase 3 must pass parse/grounding checks before any new wiki claim.

## Phase 1 smoke result

- `experiments/build_mineru_only_graph_v0.py` parses raw `structure.json` directly.
- Output: `data/05_eval/mineru_only_graph_v0_20260519T061721Z/`.
- Scale: 80 docs / 2560 elements / 10178 local edges.
- Edge types: reading order, regex reference, same-page cross-type window.
- Blocker: only 88 text elements in raw structures; before API generation, add MinerU markdown/content-list fallback or regenerate richer MinerU output.

## Detailed plan

See `refine-logs/MINERU_ONLY_MIGRATION_PLAN_20260519.md`.

## Handoff

Full May 19 discussion handoff for the next assistant:

- `research-wiki/experiments/20260519_pdf_mineru_handoff.md`

