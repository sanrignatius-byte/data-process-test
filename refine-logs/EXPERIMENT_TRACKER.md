# Experiment Tracker

> Query-production runs (R001-R007) from 2026-04-18 plan archived below.
> Active retrieval experiment runs start from R100.

---

## Retrieval Experiments (active)

| Run ID | Milestone | Purpose | System / Variant | Key Params | Metrics Target | Priority | Status | Notes |
|--------|-----------|---------|------------------|------------|----------------|----------|--------|-------|
| R100 | M1 | Combo: chunk-v2 graph + typed(w=0.1), static_prior | explicit+typed, chunk_v2, prior=weighted | typed_w=0.1 | R@1≥0.25, R@10≥0.62 | MUST | **READY** | slurm_scripts/32_combo_typed_chunkv2.sh |
| R101 | M1 | Combo: chunk-v2 graph + typed(w=0.2), static_prior | explicit+typed, chunk_v2, prior=weighted | typed_w=0.2 | R@1≥0.25, R@10≥0.62 | MUST | **READY** | Main hypothesis |
| R102 | M1 | Combo: chunk-v2 graph + typed(w=0.3), static_prior | explicit+typed, chunk_v2, prior=weighted | typed_w=0.3 | R@1≥0.25 | MUST | **READY** | Upper weight boundary |
| R103 | M1 | Combo: chunk-v2 graph + typed(w=0.2), static_plus_neighbor | explicit+typed, chunk_v2 | typed_w=0.2 | R@10≥0.64 | MUST | **READY** | Check neighbor metrics from R101 output |
| R104 | M2 | Paragraph merge n=400, dense baseline | build_paragraph_chunks + eval_dense_retrieval | chunk_size=400 | R@1 vs 0.2389 | MUST | **READY** | slurm_scripts/33_paragraph_merge.sh + build_paragraph_chunks.py done |
| R105 | M2 | Paragraph merge n=500, dense baseline | build_paragraph_chunks + eval_dense_retrieval | chunk_size=500 | R@1 vs 0.2389 | MUST | **READY** | In same SLURM job as R104 |
| R106 | M2 | Paragraph merge n=best, graph rerank | explicit+typed on v3 corpus | best chunk_size | R@1, R@10 | MUST | TODO | Only if R104/R105 > v1_enriched baseline |
| R107 | M3 | bbl expansion → rebuild typed_crossdoc | bbl extractor on 53 docs | coverage target ≥30 docs | cite_boost coverage | NICE | TODO | Currently 59 docs/123 edges |
| R108 | M3 | Typed_crossdoc with expanded bbl, best B3 config | explicit+typed, expanded bbl | typed_w=0.2 | R@10 delta | NICE | TODO | Depends on R107 |

---

## Modality Routing Ablation (2026-05-03, revised)

| Run ID | Milestone | Purpose | System / Variant | Key Params | Metrics Target | Priority | Status | Notes |
|--------|-----------|---------|------------------|------------|----------------|----------|--------|-------|
| R120 | M0 | Sanity: both encoders load, 10q RRF | routing ablation dry-run | k=60 | Non-random rankings | MUST | TODO | Verify env |
| R121 | M1 | Best-guess first: r_4b_fig_tab | fig/tab→4B, formula→VL, text→4B, RRF k=60 | — | R@10 > 0.45 | MUST | TODO | New best guess |
| R122 | M2 | Baseline: r_4b_all | all→4B (=split_4B_text mixed) | — | R@10 ≈ 0.4767 | MUST | TODO | Reference |
| R123 | M2 | r_vl_fig_tab (original hypothesis) | fig/tab→VL, formula/text→4B | — | Compare vs R121 | MUST | TODO | Test original assumption |
| R124 | M2 | r_vl_formula_only | fig/tab/text→4B, formula→VL | — | Minimal VL | MUST | TODO | VL仅用于formula |
| R125 | M2 | r_vl_all_nontext | all non-text→VL (=split_VL_2B_t5 mixed) | — | R@10 ≈ 0.2579 | MUST | TODO | Lower bound |
| R126 | M5 | Mixed vs split index (best routing) | same routing, split indexes | — | mixed > split | NICE | TODO | Verify C_R3 |
| R127 | M6 | RRF k sensitivity (k=30,100) | best routing | k=30,100 | ≤2pp variation | NICE | TODO | Robustness |

---

## VL Enrich-Only Comparison (2026-05-03)

| Run ID | Milestone | Purpose | System / Variant | Key Params | Metrics Target | Priority | Status | Notes |
|--------|-----------|---------|------------------|------------|----------------|----------|--------|-------|
| R130 | M0 | Build enrich_only corpus | build_enrich_only_corpus.py | --text-mode enrich_only | ≥80% enriched coverage | MUST | TODO | Verify before encoding |
| R131 | M1 | Text enrich_only eval (4B) | Qwen3-Embedding-4B on enrich_only corpus | max_length=512 | Baseline R@10 | MUST | TODO | ~10 min GPU |
| R132 | M2 | VL image eval (2B) | Qwen3-VL-Embedding-2B on raw images | img max_length=4096 | Compare vs R131 | MUST | TODO | ~15 min GPU, needs tf5 overlay |
| R133 | M3 | Enrichment quality correlation | Post-hoc from R131+R132 | Spearman corr | Negative correlation | NICE | TODO | Scatter plot for appendix |

## Cross-Document Citation Query Pipeline (2026-05-03)

| Run ID | Milestone | Purpose | System / Variant | Key Params | Metrics Target | Priority | Status | Notes |
|--------|-----------|---------|------------------|------------|----------------|----------|--------|-------|
| R140 | M0 | Verify citation context quality | Manual: read 20 random edge_contexts | — | ≥60% contain element refs | MUST | TODO | 30 min manual |
| R141 | M1 | Extract element-level citation pairs | build_cross_doc_citation_pairs.py | label match + fallback | ≥50 valid pairs | MUST | TODO | ~2 min CPU |
| R142 | M2 | Generate cross-doc queries (top-50) | generate_multihop_l1_queries.py | PROMPT_CROSS_DOC_CITATION | ≥25 QC pass | MUST | TODO | ~30 min API, ~$3-5 |
| R143 | M3 | Cross-doc retrieval eval | BM25 + dense + graph rerank | 3 configs | Graph delta > intra-doc delta | MUST | TODO | ~20 min GPU |
| R109 | M4 | C-Pool QA validation with graph rerank | 0.6B + graph rerank on 78 universal queries | — | evidence recall | NICE | BLOCKED | Needs qrels (mentor decision 4.20) |

---

## Reference Results (done)

| Run ID | System | R@1 | R@5 | R@10 | MRR | Source |
|--------|--------|-----|-----|------|-----|--------|
| REF-A | 0.6B dense v1_enriched (baseline) | 0.2389 | 0.5127 | 0.5994 | 0.6081 | exp:20260417_dense_baseline_rebuilt |
| REF-B | 0.6B + explicit_only, static_prior, chunk_v2 | 0.2505 | 0.4852 | 0.5391 | 0.6162 | exp:20260419_deliverable_420 |
| REF-C | 0.6B + explicit+typed(w=0.2), static_plus_neighbor, chunk_v1 | 0.1818 | 0.5423 | 0.6406 | 0.5413 | exp:20260419_typed_crossdoc |
| REF-D | 4B + explicit_only, static_prior | 0.2421 | 0.5856 | 0.6448 | 0.6399 | exp:20260419_deliverable_420 |

---

## Archived — Query Production Runs (from 2026-04-18 plan)

| Run ID | Milestone | Purpose | Status |
|--------|-----------|---------|--------|
| R001 | M0 | Audit production readiness | DONE |
| R002 | M1 | First M2 production sweep (academic) | TODO |
| R003 | M1 | Diversity-boosted production sweep | TODO |
| R004 | M1 | Secondary candidate sweep | TODO |
| R005 | M2 | Delivery packaging rebuild | TODO |
| R006 | M3 | Bottleneck diagnosis | TODO |
| R007 | M4 | Supply expansion (pairing module) | NICE |
