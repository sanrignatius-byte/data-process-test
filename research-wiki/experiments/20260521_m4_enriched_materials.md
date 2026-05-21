---
type: experiment
node_id: exp:20260521_m4_enriched_materials
status: smoke_passed
created_at: 2026-05-21T03:10:00Z
updated_at: 2026-05-21T03:10:00Z
---

# Experiment: Enriched-only M4 material construction

Experimental lane wrapper around existing Method C production prototype
(`scripts/pilot_method_c.py`). Goal: construct M4-ready material packs while
hard-filtering real multimodal elements to enriched-only content.

## Method

- Source candidates: `data/03_queries/method_c_true2_candidates_2026-04-12T050859Z.json`.
- Wrapper: `experiments/construct_m4_enriched_materials.py`.
- Core generation logic reused from `scripts/pilot_method_c.py`:
  `build_method_c_view`, `build_prompt`, `generate_qa`, `build_qc_obj`,
  `build_qc_pair`.
- Enriched-only gate:
  every real endpoint / node-group element must have non-noisy
  `enriched_title` or `enriched_content`.
  Synthetic paragraph/section bridges are allowed but marked as bridge context,
  not raw multimodal elements.
- Company API calls use `src.api.call_llm(provider="company")`, which routes
  through `local_api_logger.wrap_requests_call`.

## Result

- Output: `data/05_eval/m4_enriched_materials_company_smoke_20260521T040500Z/`.
- Latest symlink: `data/05_eval/m4_enriched_materials_latest`.
- Raw / eligible / selected: 817 / 817 / 100.
- Selected pair types: `equation+table`: 10, `figure+table`: 67,
  `equation+figure`: 23.
- Hop counts: 4-hop 52, 5-hop 48.
- Compressed bridge counts: 2 for all 100 materials.
- API smoke: 5 generated with company `gpt-5.4`; 5/5 parsed, 3/5 rule-QC pass.
  Failures: `text_evidence_over_reliance` and one `bare_deictic`.
- Tokens: 6,935 input / 3,206 output. Logged in both
  `api_logs_cannt_delete/calls/gpt-5.4/2026-05/2026-05-21.jsonl` and
  `logs/token_usage.db`.

## Files

- Materials: `data/05_eval/m4_enriched_materials_latest/m4_material_pack.jsonl`.
- Candidate subset: `data/05_eval/m4_enriched_materials_latest/m4_enriched_candidates.json`.
- Prompt batch: `data/05_eval/m4_enriched_materials_latest/prompt_batch.jsonl`.
- API smoke generations: `data/05_eval/m4_enriched_materials_latest/generated_m4_smoke.jsonl`.
- Summary/report: `data/05_eval/m4_enriched_materials_latest/summary.json`,
  `data/05_eval/m4_enriched_materials_latest/report.md`.

## Interpretation

This confirms the current Method C path can construct enriched-only M4
materials and call the company API through the required logger. It is not yet a
production M4 batch: full 100 generation should follow only after tightening
the prompt or post-processing for `text_evidence_over_reliance` and deictic
openings.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
