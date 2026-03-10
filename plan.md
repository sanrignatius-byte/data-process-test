# Implementation Plan: MoDora Approach Integration + Query Style Overhaul

## Overview

Four workstreams in parallel: (A) Node granularity refinement, (B) Query style overhaul, (C) Low-quality enrichment filtering + hub summary upgrade, (D) QC overhaul. Total touches 5 files.

---

## Workstream A: Node Granularity Refinement

### A1. Paragraph splitting at section boundaries
**File**: `src/parsers/latex_reference_extractor.py` — `_extract_paragraphs()` (lines ~999-1047)

- In the main loop, before accumulating a line into the current block, check if the line matches `RE_SECTION`
- If it does, flush the current block first, then start a new block with this line
- Result: paragraphs never span across `\section{}`/`\subsection{}` boundaries
- Backward compatible: no section commands = no behavior change

### A2. Section nodes in path enumeration
**File**: `scripts/analyze_latex_graph_topology.py` — `enumerate_candidates_from_bridge_hubs()` (lines ~1031-1257)

- Add **Strategy 4**: Section-bridged paths. For each hub paragraph, find its containing section via `section_contains_paragraph` edges, then find other elements under the same section → forms `[elem_A, section_node, elem_B]` paths
- Add `--single-doc-only` CLI flag: when set, skip Strategy 3 (cross-doc via citation) entirely
- Adjust cross-doc sort priority: change `int(x["is_cross_doc"])` to `int(not x["is_cross_doc"])` to favor intra-doc candidates

### A3. Rebuild data
After code changes: re-run `build_latex_reference_graph.py` then `analyze_latex_graph_topology.py` to regenerate candidates with new paragraph boundaries and section paths.

---

## Workstream B: Query Style Overhaul

### B1. Add 5 new real-user prompt templates
**File**: `scripts/generate_multihop_l1_queries.py` — after existing PROMPT_* constants (~line 458)

New templates (English only):
1. **PROMPT_FACTUAL_LOOKUP**: "what is X" / "what does Y show" — short, direct (5-15 words)
2. **PROMPT_SUMMARY**: "summarize the approach / key results" — can reference 1-3 elements
3. **PROMPT_COMPARISON**: "which method works better on X" — naturally requires 2+ elements
4. **PROMPT_HOW_WORKS**: "how does the encoder handle X" — architecture/mechanism queries
5. **PROMPT_WHAT_IF**: "what would happen if we removed X" — counterfactual queries

Key differences from academic templates:
- System prompt: "researcher looking up information" instead of "PhD student at lab meeting"
- No mandatory observation injection
- Yes/no questions ALLOWED
- "Which component" ALLOWED
- Word count: 5-20 words (vs 8-30)
- Outputs `node_group` (1-3 element_ids) instead of exactly 2
- Outputs `query_style: "real_user"` marker

### B2. Modify `select_template()` and add `--query-style` CLI flag
**File**: `scripts/generate_multihop_l1_queries.py`

- Add `--query-style` with values `academic` (default), `real_user`, `mixed`
- `academic`: existing behavior
- `real_user`: always pick from new template set
- `mixed`: 50/50 random split between old and new

### B3. Support node groups in enrichment
**File**: `scripts/enrich_hub_candidates.py` — `enrich_candidates()` (lines ~384-396)

- Add `node_group` field to pair dict: list of 1-3 element dicts (all element nodes found in path)
- Keep `element_a`/`element_b` for backward compat (first two elements)

---

## Workstream C: Enrichment Quality + Hub Summary Upgrade (colleague's feedback)

### C1. Low-quality enrichment filter (HIGHEST PRIORITY)
**File**: `scripts/generate_multihop_l1_queries.py` — new function `filter_low_quality_enrichment()`

Before query generation, check each element's enriched fields:
- Detect noise patterns: `glyph`, `icon`, `standalone symbol`, `marker`, `no axis`, `no trend`, `decorative`, `logo`, `separator`
- If `enriched_title` or `enriched_content` matches noise patterns → fall back to original `caption`/`context_before`/`context_after`
- Also flag `figure_type=other` + purely layout keywords as low-confidence

### C2. Lightweight consistency check for figure/table
**File**: `scripts/generate_multihop_l1_queries.py` — within the same filter function

- If caption contains "Figure/Table + number + metric word" but enriched output has `figure_type=other` and keywords are pure layout terms → mark as low-confidence, fall back to original context
- Log warning with element_id for manual review

### C3. Hub summary compressed rewrite
**File**: `scripts/enrich_hub_candidates.py` — modify `build_hub_semantic_summary()`

Current behavior: concatenates endpoint enriched descriptions + edge context + keywords.
New behavior: after concatenation, add a compression step:
- Take the concatenated summary (often 200+ words)
- Use a lightweight rewrite (rule-based or LLM) to compress to 50-80 words
- Focus on: what the two endpoints share, how they connect, what reasoning the bridge enables
- If `--no-rewrite` flag is set, keep current concatenation behavior

---

## Workstream D: QC Overhaul

### D1. New `qc_real_user_query()` function
**File**: `scripts/generate_multihop_l1_queries.py` — add alongside existing `qc_multihop_query()` (~line 1036)

**Checks KEPT** (from existing):
- `meta_language`, `anchor_leakage`, `numeric_leakage`, `empty_query`, `short_answer`, `premise_answer_contradiction`
- `evidence_spans_incomplete` (adapted: require ≥1 span instead of ≥2)

**Checks REMOVED** for real-user queries:
- `yes_no_question`, `yes_no_answer` (real users ask yes/no)
- `template_shortcut`, `templated_opening`, `template_collapse` ("which component" is natural)
- `opening_repetition`, `length_mix_missing`
- `architecture_intent_missing`

**Checks SOFTENED**:
- `single_element_answer`: advisory metric only, not hard fail
- `weak_reasoning_connector`: removed entirely
- `missing_reasoning_chain`: optional for factual queries
- Word bounds: 4-25 words (vs current bounds)

**New checks ADDED**:
- `retrievability_score`: token overlap between query and target elements' captions + enriched_title + context. Warn if < 0.10
- `query_type_diversity`: per-document type distribution warning if any type > 60%

### D2. Route QC by query_style
In main loop: check `q_obj.get("query_style", "academic")` and dispatch to appropriate QC function.

### D3. Extended output schema
New fields in JSONL output:
- `query_style`: `"real_user"` | `"academic"`
- `node_group`: list of element_ids (1-3)
- `retrievability_score`: float

Existing fields (`element_ids`, `pair_type`, etc.) remain for backward compat.

---

## Token Logging (Iron Rule Compliance)

All existing `log_run()` calls remain. New code paths reuse the same call. The `purpose` string updated to include query style:
```python
purpose=f"L1 {args.query_style} query generation — {kept}/{query_idx} QC pass ..."
```

---

## Execution Order

```
C1 (enrichment filter) ← HIGHEST PRIORITY, do first
    ↓
A1 + A2 (paragraph split + section paths)  ‖  B1 + B2 (new templates + CLI)  ‖  C3 (hub rewrite)
    ↓                                           ↓
A3 (rebuild data)                           B3 (node groups in enrichment)
                                                ↓
                                            D1 + D2 + D3 (QC overhaul)
                                                ↓
                                            Integration testing
```

## Files Modified

| File | Workstream | Changes |
|------|-----------|---------|
| `src/parsers/latex_reference_extractor.py` | A1 | Paragraph split at section boundaries |
| `scripts/analyze_latex_graph_topology.py` | A2 | Section paths + `--single-doc-only` flag |
| `scripts/generate_multihop_l1_queries.py` | B1, B2, C1, C2, D1-D3 | New templates, enrichment filter, new QC, CLI flags |
| `scripts/enrich_hub_candidates.py` | B3, C3 | Node groups + hub summary rewrite |
| `src/utils/token_logger.py` | — | No changes needed (reference only) |
