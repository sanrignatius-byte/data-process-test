---
type: experiment
node_id: exp:20260519_pdf_mineru_handoff
status: handoff_ready
created_at: 2026-05-19T06:25:00Z
updated_at: 2026-05-19T06:25:00Z
lane: experimental
---

# 2026-05-19 PDF/MinerU-first handoff record

This page is the compact handoff for the next assistant. It consolidates the May 19 discussion and the experimental artifacts created after the initial LaTeX-centric attempts.

---

## Executive decision

The project should pivot new graph/query work from **LaTeX-centric** assumptions to a **PDF/MinerU-first** pipeline.

LaTeX remains useful as:

- a legacy baseline/control;
- a source of historical claims and negative evidence;
- a diagnostic tool when it exists.

But new candidate generation should not depend on:

- `latex_reference_graph*.json`;
- `.tex` line numbers;
- LaTeX label → MinerU element mapping;
- BBL/citation extraction as the primary cross-doc route;
- Method-C style long LaTeX paths.

New canonical grounding unit:

```text
MinerU element = doc_id + element_type + page_idx + position_idx + content/caption/context + image_path/bbox
```

Text/paragraph must be treated as a first-class element, peer to figure/table/formula.

---

## Non-negotiable constraints

1. **Experimental lane first**: do not modify production `src/` for this line until `experiments/` outputs pass gates.
2. **Production cross-doc guard stays**: old `scripts/generate_multihop_l1_queries.py` intentionally filters cross-doc pairs; do not “fix” it casually.
3. **Company API calls must be logged** through:
   - `src.api.call_llm(provider="company")`
   - fixed call/status directory: `api_logs_cannt_delete`
   - token total DB: `logs/token_usage.db`
4. **No API key leakage**: `.env` has credentials; do not print them.

---

## What was validated today

### 1. Cross-doc input connectivity

Correct archive input:

`archive/data_legacy/embedding_probes/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2b_cap10.jsonl`

Validated facts:

- 590 source records.
- 11800 matches.
- source/target coverage in `data/01_graphs/multimodal_elements.json`: 100%.
- Direct cross-doc types are same-modality only:
  - `figure->figure`: 7660
  - `table->table`: 3260
  - `formula->formula`: 880
  - direct cross-modal: 0

### 2. XDoc + XModal construction works as a 3-node chain

Direct archive cross-doc matches are same-modality. To make them also cross-modal:

```text
A -> B = same-modality cross-document semantic bridge
C      = local cross-modal MinerU neighbor attached to A or B
```

Generated candidate design:

- `experiments/build_pdf_first_xdoc_xmodal_candidates.py`
- `experiments/xdoc_xmodal_prompt_dryrun.py`
- output: `data/05_eval/pdf_first_xdoc_xmodal_design_20260519T033730Z/`
- 12 candidates: 4 each for figure/table/formula base type.

Company API smoke:

- `experiments/xdoc_xmodal_company_api_smoke.py`
- output: `data/05_eval/xdoc_xmodal_company_api_smoke_20260519T034154Z/`
- `parsed_ok=6/6`, `failures=0`
- tokens: `7270 input / 3644 output`
- quality review: `quality_review.md`

Best examples called out in review:

- `pdf_xdoc_xmodal_0005`: figure→figure + source table.
- `pdf_xdoc_xmodal_0009`: formula→formula + source table.
- `pdf_xdoc_xmodal_0004`: table→table + source figure.

### 3. API logging has been fixed to a non-delete directory

Fixed standard call/status directory:

`api_logs_cannt_delete`

Changes made:

- `local_api_logger/logger.py`: default `APILogger()` directory changed to `api_logs_cannt_delete`.
- `local_api_logger/viewer.py`: default `LogViewer()` directory changed to `api_logs_cannt_delete`.
- company API smoke scripts explicitly use `ROOT / "api_logs_cannt_delete"`.
- `api_logs_cannt_delete/README.md` added as do-not-delete marker.
- duplicate root `local_api_logger/calls` and `local_api_logger/stats` removed to avoid split-brain logging.

Current fixed logs:

- `api_logs_cannt_delete/calls/gpt-5.4/2026-05/2026-05-19.jsonl`: 9 calls.
- `api_logs_cannt_delete/stats/gpt-5.4/trinity_xdoc_smoke_2026-05.jsonl`: 3 rows.
- `api_logs_cannt_delete/stats/gpt-5.4/xdoc_xmodal_smoke_2026-05.jsonl`: 6 rows.
- `logs/token_usage.db`: 2 smoke rows, total `9315 input / 4568 output`, `parse_failures=0`.

### 4. MinerU-only audit and graph v0 exist

Audit:

- `experiments/audit_mineru_only_migration.py`
- output: `data/05_eval/mineru_only_migration_audit_20260519T061252Z/`

Audit results:

- MinerU doc dirs: 1153.
- docs with `structure.json`: 80.
- docs with `formulas.jsonl`: 62.
- raw MinerU elements from `structure.json`: 2560.
- raw type counts: formula 1447 / figure 1017 / text 88 / table 8.
- current `multimodal_elements.json`: 76 docs / 1316 elements.
- image-like raw elements with image path: 1017 / 1025.

Graph v0:

- `experiments/build_mineru_only_graph_v0.py`
- output: `data/05_eval/mineru_only_graph_v0_20260519T061721Z/`

Graph v0 results:

- 80 docs.
- 2560 elements.
- 10178 local MinerU-only edges:
  - `next_element`: 2480
  - `prev_element`: 2480
  - `regex_reference`: 1402
  - `same_page_cross_type_window`: 3816

Critical blocker: raw `structure.json` is visual/formula-heavy and text-sparse. It only exposes 88 text elements, so query generation should not proceed until body-text fallback is added.

---

## What changed in wiki / logs

Main plan:

- `refine-logs/MINERU_ONLY_MIGRATION_PLAN_20260519.md`
- `research-wiki/experiments/20260519_mineru_only_migration.md`

API logging rule:

- `research-wiki/experiments/20260421_api_logging_compliance.md`

This handoff:

- `research-wiki/experiments/20260519_pdf_mineru_handoff.md`

---

## Recommended next task for the next assistant

Do **not** call the company API next. First make the MinerU-only graph text-rich.

Immediate coding task:

```text
Extend experiments/build_mineru_only_graph_v0.py with markdown/content_list fallback.
```

Goal:

- keep direct `structure.json` parsing for figure/formula/table/image paths;
- add body text paragraphs from MinerU markdown or `content_list.json` when available;
- create `text` elements with `page_idx` and approximate `position_idx`;
- attach visual/formula nodes to nearest text/context nodes;
- rerun graph v0 and require text coverage to rise materially above 88 nodes.

Suggested gate before another API smoke:

- ≥70% of parsed docs have text elements, or the report explains where body text is missing;
- ≥90% image path resolution for image-like nodes;
- candidate builder can produce at least 24 source-diverse same-doc xmodal / xdoc+xmodal candidates;
- manual QC on 9 candidates shows all three nodes are necessary in ≥6/9.

---

## Files to inspect first

```text
experiments/build_mineru_only_graph_v0.py
experiments/audit_mineru_only_migration.py
refine-logs/MINERU_ONLY_MIGRATION_PLAN_20260519.md
research-wiki/experiments/20260519_mineru_only_migration.md
data/05_eval/mineru_only_graph_v0_20260519T061721Z/report.md
data/05_eval/mineru_only_migration_audit_20260519T061252Z/report.md
experiments/build_pdf_first_xdoc_xmodal_candidates.py
experiments/xdoc_xmodal_company_api_smoke.py
```

---

## Do not repeat these mistakes

- Do not route future call logs into per-run `local_api_logger/`; use `api_logs_cannt_delete`.
- Do not assume `data/00_raw/...` contains the crossdoc match file; the correct file is in `archive/data_legacy/embedding_probes/`.
- Do not treat missing direct cross-modal crossdoc matches as failure; the validated construction is same-modality crossdoc bridge + local cross-modal attachment.
- Do not promote MODORA corpus-level visual enrichment; previous F2 result marked it as D5 antipattern.
- Do not use LaTeX line numbers as a new dependency for PDF-first work.
