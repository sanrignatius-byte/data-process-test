---
type: experiment
node_id: exp:20260519_xdoc_pairing_module
status: phase0_artifact_built
created_at: 2026-05-19T00:00:00Z
updated_at: 2026-05-21T02:00:00Z
---

# Experiment: Element-level cross-doc pairing module

> **Lane rule (2026-05-19 blue-team correction)**: this is Track A / experimental-lane work only. Do **not** modify production `src/` first. Prototype under `experiments/` or `scripts/experimental_*`, write outputs under `data/05_eval/` or `data/02_enriched/experimental/`, then promote a minimal reusable module into `src/pairing/` only after gates pass.

Prototype a cross-doc pairing module by mirroring the logic of `src/pairing/intra_doc_pairs.py` in the **experimental lane** first. Use the existing reranked Qwen3-Embedding-4B element-level cross-doc matches to produce a `cross_doc_pairs_v1.json` artifact in the same schema as `hub_candidates_enriched_v3.json`, so it can be fed to existing query generators (`generate_multihop_l1_queries.py`) without code changes downstream.

## Design

- **Input**:
  - `archive/data_legacy/embedding_probes/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2b_cap10.jsonl` (590 records, 11800 reranked matches, audit completed: `reciprocal=0.8119`, `unique_top1=286`). Earlier draft incorrectly pointed to `data/00_raw/`; the file exists in `archive/data_legacy/embedding_probes/`, not production raw data.
  - `data/01_graphs/multimodal_elements.json` for element metadata
  - `data/01_graphs/latex_reference_graph.json` for in-doc paragraph references (used by bridge filter)

- **Filters**:
  - At least one endpoint is figure / table / formula (multi-modal constraint)
  - Both endpoints are referenced by ≥1 paragraph in their parent doc (no orphan elements)
  - Reciprocal rank ≥ 0.5 (high-confidence matches only)
  - Reject pairs whose docs are identical (intra-doc safety check)

- **Output**:
  - `data/02_enriched/experimental/cross_doc_pairs_v1.json` — schema-compatible with `hub_candidates_enriched_v3.json`
  - Each pair has `pair_type`, `element_a`, `element_b`, `bridge_evidence`, `cross_doc_metadata` (similarity score, reciprocal rank)

- **Validation hooks**:
  - Reuse `qc_multihop_query()` rule QC and `run_llm_qc()` LLM QC.
  - Sanity check: feed 20 pairs to `generate_multihop_l1_queries.py --dry-run` and verify prompts render without missing fields.

## Pre-registered hypothesis

Element-level cross-doc pairs will produce queries with **lower** BM25 R@10 than intra-doc pairs (truly harder retrieval) but **same** QC pass rate (~58-64% per v4.2 baseline). If pass rate drops more than 10pp, the pair filter is too lax.

Blue-team calibration: the correct comparison is **three-way**, not only element-level vs doc-citation:

1. intra-doc baseline pairs;
2. existing paragraph-mediated cross-doc L3 pass chains (`elem → ::p:: → [::p::] → elem`, 87/146 already cross-doc);
3. new element-direct cross-doc pairs from Qwen3 reranked matches.

C12 is supported only if (3) is harder than (1) **without** QC collapse and adds value over (2), not merely over doc-level citation walks.

## Phase 0 gates (must pass before code promotion)

- Verify the archive input file exists and has 590 records.
- Produce only an experimental artifact under `data/02_enriched/experimental/`.
- Run a 20-pair dry-run through query rendering.
- Do not add `src/pairing/cross_doc_pairs.py` until experimental output passes schema, QC, and comparison gates.

## Status

Planned 2026-05-19. Phase 0 required. Blocking: experimental prototype and three-way baseline comparison not yet run. Estimated effort: 0.5 day for Phase 0 + 1-2 days for validated prototype.

Update 2026-05-19T06:25Z:

- Connectivity smoke confirmed the archive input path is correct and usable.
- Direct cross-doc matches are same-modality only: figure→figure, table→table, formula→formula.
- Cross-modal cross-doc should be composed as a 3-node chain: same-modality cross-doc bridge + local MinerU/PDF cross-modal neighbor.
- See handoff: `research-wiki/experiments/20260519_pdf_mineru_handoff.md`.

Update 2026-05-21T02:00Z:

- Built an experimental citation-backed cross-doc element resolver under `experiments/build_xdoc_element_resolver_v0.py`.
- Input is the G11-filtered C18 backbone: `data/04_xdoc_citation/predicted_xdoc_edges_chunks_filtered.jsonl` (34,447 edges).
- Output artifact: `data/05_eval/xdoc_element_resolver_v0_20260521T015847Z/` and latest symlink `data/05_eval/xdoc_element_resolver_v0_latest/`.
- Result: 24,246 citation chunks had both source and target elements resolvable; retained top 5,000 cross-modal pairs (`figure+table`: 2,898; `figure+formula`: 1,495; `formula+table`: 607).
- Prompt gate: existing query generator initially filtered all cross-doc candidates; added default-off `--allow-cross-doc-candidates` and verified 20/20 dry-run prompt rendering with no LLM calls.
- Caveat: v0 target resolution is lexical caption/content overlap, not final truth. Next gate should run manual/LLM judging on a stratified sample before promoting anything into `src/pairing/cross_doc_pairs.py`.

Update 2026-05-21T02:20Z:

- Added v1 execution design from external review feedback: `refine-logs/XDOC_ELEMENT_RESOLVER_V1_DESIGN_20260521.md`.
- Added execution tracker: `refine-logs/XDOC_ELEMENT_RESOLVER_V1_TRACKER_20260521.md`.
- v1 priorities: target-side explicit `Figure/Table/Eq N` matching, source nearest-position fanout penalty, v0-vs-v1 recovery on the known cross-doc L3 pass set, then a stratified 100-item judge pack only if recovery gates are credible.

Update 2026-05-21T03:00Z:

- **v1 executed.** Full artifact at `data/05_eval/xdoc_element_resolver_v1_latest/`.
- `experiments/build_xdoc_element_resolver_v1.py`: target explicit numbered ref resolution (anchored/unanchored tiers) + source nearest-position fanout penalty. Title aliases from `latex_reference_graph_v2.json` (1,317 docs) used for anchoring.
- v1 results on 34,447 filtered edges: 5,000 pairs.  
  Target methods: 182 `target_explicit_number_anchored`, 30 `target_explicit_number_unanchored`, 4,788 `target_caption_overlap`.  
  Source methods: 643 `source_explicit_ref`, 4,357 `source_nearest_position`.  
  Fanout buckets: 1–2: 3,307 / 3–5: 1,102 / 6–10: 387 / >10: 204.
- v0 comparison: explicit target 0→212, explicit source 494→643.
- **L3 recovery evaluation** (`experiments/evaluate_xdoc_resolver_l3_recovery.py`):  
  128 cross-doc L3 gold rows (37 unique doc pairs), element ID intersection 100%.  
  61/128 gold rows use doc pairs in the filtered C18 graph (recoverable).  
  On recoverable subset: v1 doc-pair recovery 14.8% (9/61), v0 0%. v1 covers 2/17 recoverable doc pairs, v0 covers 0.  
  Binding constraint: C18 citation graph doc coverage (~1,250 docs) only partially overlaps L3 gold (31 docs, 8 in resolver). Recovery is a positive v1 signal but not a definitive gate.
- **Stratified judge pack** (`experiments/build_xdoc_resolver_judge_pack.py`):  
  100 items at `judge_pack_100.jsonl`. Strata: 25 anchored explicit, 25 unanchored explicit, 50 overlap.  
  Types: figure+table 54, figure+formula 22, formula+table 24.  
  Rubric: strong_chain / weak_but_related / topic_only / wrong_target / wrong_source / insufficient_context.
- **Prompt gate**: 20/20 dry-run renders pass with `--allow-cross-doc-candidates`.
- **Tests**: 77 passed (6 G11 filter + 16 v1 resolver + 55 intra-doc pairing).
- **Gate status**: G1 (explicit coverage): **PASS** (212 > 0, but < 300 — coverage is genuinely low). G5 (prompt): **PASS** (20/20). G2 (L3 recovery): **PARTIAL** (v1 > v0 but absolute recovery limited by doc coverage). G3/G4 (judge precision): **PENDING** (judge pack built, not judged).

Update 2026-05-21T03:45Z:

- **v1 stats refreshed after independent review.** Rebuilt latest symlink to `data/05_eval/xdoc_element_resolver_v1_20260521T034500Z/`; prior v1 artifact remains intact.
- `summary.json` now separates raw candidate-attempt score buckets from final selected-pair buckets:
  raw `target_score_buckets` scope = before dedup/source-chunk cap/top-k; final `post_filter_target_score_buckets` sums to 5,000.
- Final selected-pair anchor reasons:
  `title_words_in_window`: 19, `title_match_ge_0.2`: 115, `low_fanout`: 24, `single_ref_high_prob`: 24, `unanchored`: 30, `target_caption_overlap`: 4,788.
  Interpretation: only `title_words_in_window` is a hard explicit target anchor; `title_match_ge_0.2` and `low_fanout/single_ref_high_prob` are exploratory soft anchors.
- **Judge pack rebuilt** by `anchor_reason`, not broad anchored/unanchored labels:
  `judge_pack_120.jsonl` with strata A/F = 19 hard-title / 21 title-match / 20 soft-fanout-or-single-ref / 20 unanchored / 20 overlap-high / 20 overlap-low.
  A has only 19 available hard-title candidates, so one slot was redistributed to B; no duplicate samples were introduced.
- **L3 recovery report made method-stratified.** At K=5,000: doc-pair recall 9/128, endpoint recall 3/128; all 3 endpoint hits are `target_caption_overlap`, with **0 explicit endpoint hits**.
  Gate interpretation tightened: G2 is only "v1 > v0 weak positive signal"; it does **not** validate the explicit target route.
- **Prompt gate**: 20/20 dry-run renders pass with no LLM calls. **Tests**: 81 passed.
- **Red lines preserved**: no LLM judge run; no promotion to `src/pairing/cross_doc_pairs.py`. G3 hard explicit precision may only be claimed for stratum A after judging; B/C remain exploratory.

## Final closure 2026-05-24

**Route formally closed.** On 2026-05-24, the 120-item stratified judge pack (`judge_pack_120.jsonl`) was fully judged via LLM-as-judge:
- **0/120 strong (0.0%)**
- 96/120 wrong_target (80%)
- Stratum A (hard-title, the highest-precision tier): 18/19 wrong_target

Citation-based cross-document element resolution is infeasible. Even when an explicit "Figure 6" reference exists in the source paper's text, the VLM judge determines the resolved target element does not match in ≥95% of cases.

The closure is recorded in the wiki log at `2026-05-24T05:15:00Z` and in `docs/CROSS_DOC_LONG_CHAIN_REPORT_20260522.md`.

**Do not re-open this route** unless (a) a fundamentally different resolution mechanism is proposed (not citation-text matching), or (b) a new corpus with dense cross-document figure/table numbering conventions becomes available.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
