---
id: exp:20260520_mineru_clip_xdoc_pipeline
title: "MinerU CLIP cross-document visual linking pipeline — completion, A/B vs LaTeX, edge audit"
date: 2026-05-20
status: completed_baseline_locked
lane: experimental
---

# MinerU CLIP cross-document visual linking pipeline

Continuation of the May 19 MinerU-only line (`exp:20260519_mineru_only_migration`,
`exp:20260519_pdf_mineru_handoff`). Goal: finish the interrupted pipeline (formula
backend swap was in-progress), run the full 53-doc regression, and **actually audit
the resulting edges** — intra-doc and cross-doc — to judge whether pure MinerU can
build effective and strong edges, since the project must move to non-LaTeX PDFs.

## What was done

1. **Formula embedding backend swapped** (`build_mineru_vl_edges.py`):
   added `--formula-backend math_similarity` (`math-similarity/Bert-MLM_arXiv-MP-class_arXiv`,
   768-d) independent of the open_clip visual/text backend. Probe
   (`probe_formula_embedding.py`, `data/05_eval/probe_formula_embedding/`): CLIP text
   collapses formula similarity (p50=0.966, std=0.027) vs math model (p50=0.817,
   std=0.172, min=0.036). Formula threshold auto-bumped 0.45→0.85 for the new dist.
2. **Generic-caption detector strengthened** + **`visual_only_risky` filtered by default**
   (`rerank_mineru_crossdoc_vl_edges.py`, `build_mineru_crossdoc_bridges.py`). Rerank
   promoted from sidecar to build-time: every cross-doc edge now carries `support_tier`.
3. **Full 53-doc regression** via `run_mineru_pipeline_regression.sh` (VL → rerank →
   bridges → hub). Baseline locked: `experiments/MINERU_PIPELINE_BASELINE_20260520.md`.
4. **Same-doc A/B vs LaTeX** (`audit_latex_vs_mineru_intradoc.py`) and **threshold
   portability** (`audit_rerank_threshold_portability.py`).

## Key results

**Edge inventory (53 docs)**
- intra-doc (`mineru_edges_v1`): same_page_cross_type 29290, next/prev 8271×2,
  section_contains 7897, regex_reference 4926, co_reference 232.
- VL: cross_doc_visual_sim 3238, visual_similarity 2520, text_describes_figure 2703,
  formula_similarity 4331 (math backend, 768-d).
- rerank tiers (3238 xdoc): strong_text 587 / strong_enriched 64 /
  candidate 970 / weak 1431 / **visual_only_risky 186 (dropped by default)**.
  generic-caption-both @ top100 = 0 (was 72% pre-rerank).
- bridges after dropping risky: 4741 xdoc edges, 909 sentence bridges,
  2703 VL alignments, orphan visual 19. hub: 100 hubs / 500 candidates / xdoc 137.

**Same-doc A/B vs LaTeX `\ref` (52 overlapping docs)** — the headline calibration.
Prior session believed overlap=0; that was only true against the *other corpus's*
`latex_reference_graph_v2`. Raw `.tex` in `latex_sections_rebuild_2026-03-24/extracted`
overlaps 52/53.
- figure/table extraction recall **90.8%** (464/511 LaTeX visual labels matched).
- reference recall **84.0%** (326/388 LaTeX-`\ref`'d figures also linked by MinerU
  `regex_reference`). Per-doc median 1.0 / 1.0; 26/52 docs at 100% reference recall.
- 139 MinerU-only referenced elements (implicit refs + some noise).
- Output: `data/05_eval/latex_vs_mineru_intradoc_latest/`.

**Edge quality audit (opened real samples, not just counts)**
- `regex_reference`: 6/6 sampled correct (eq.(2)→formula, Figure 1→Figure 1, …).
  This is the **strong** intra-doc edge, corroborated by the 84% A/B recall.
- `same_page_cross_type`: positional recall layer, weight already decays with
  position_distance; A/B-style precision ~27%. Recall/fallback, not strong.
- cross-doc visual: **87.2% of xdoc edges have caption_sim=0**; only **5.1% have
  enriched_sim>0.15** (real text support). The `strong_text_supported` tier is
  optimistically named — its context support median is only ~0.07; it is mostly
  vis≈0.88 lifted by a sliver of context. Genuinely good xdoc edges exist
  (e.g. gender-bias-template tables across 1910.10872↔1809.02208, enr=0.27) but are
  rare and not found via caption.
- Root cause quantified: of 937 figure/table nodes, 64.2% have real captions but
  **35.8% are unusable** for text matching (20% subfig labels "(a)(b)", 10% too short,
  5.5% OCR'd HTML fragments). Even real captions rarely share tokens across documents.

**Threshold portability** (`rerank_threshold_portability_latest`): verdict `marginal`.
Score *scale* is portable (per-field median delta ≤0.041 across split halves) but
strong/weak tier boundary drifts ±0.14 with the doc set's visual-similarity
distribution. Caveat: only 1/53 docs is PDF-only (1805.03677), so this is a
split-half stability proxy, not a true LaTeX-vs-PDF test.

## Verdict

- **Intra-doc edges: effective AND strong.** `regex_reference` + structural edges
  are reliable; A/B-validated at 84% recall of LaTeX hard reference edges. Pure MinerU
  can replace the LaTeX intra-doc graph for the non-LaTeX PDF future.
- **Cross-doc edges: effective RECALL layer, not yet STRONG.** Pure MinerU+CLIP+caption
  rerank is bottlenecked by degraded captions (87% zero token overlap cross-doc); only
  ~5% of xdoc edges have real text support. Fine as candidate recall; risky as hard
  graph edges (layout false positives).

## Literature placement (research-lit, this session)

Intra-doc unsupervised linking is not new (Hessel+2019 `paper:hessel2019_multilink`,
VLM-based `paper:hwang2026_connecting_dots`); cross-doc figure-text alignment exists
(`paper:wang2026_s1mmalign`, with abstract/citation recaption enrichment, +18% CLIP).
The open niche is **robust cross-doc multimodal edge recovery under parser-degraded,
masked-reference conditions** — existing work assumes clean captions/context; our 87%
caption-zero-overlap measurement is exactly the regime they don't cover. See `gap:G10`.

## Reusable next step (not yet run)

Lift xdoc edges from recall to strong via (1) VLM direct link judgment
(Connecting-the-Dots style, bypasses caption), (2) recaption enrichment (S1-MMAlign),
(3) LLM coarse-to-fine rerank (CoRank `paper:tian2026_corank`), (4) CLIP score
de-biasing (BSAP). Validate on the existing 3238 xdoc candidates: can strong-edge
share rise from ~5%?

## Artifacts

- Baseline doc: `experiments/MINERU_PIPELINE_BASELINE_20260520.md`
- Scripts: `build_mineru_vl_edges.py`, `rerank_mineru_crossdoc_vl_edges.py`,
  `build_mineru_crossdoc_bridges.py`, `run_mineru_pipeline_regression.sh`,
  `audit_latex_vs_mineru_intradoc.py`, `audit_rerank_threshold_portability.py`,
  `probe_formula_embedding.py`
- Locked outputs (latest symlinks): `mineru_vl_edges_v1`,
  `mineru_crossdoc_text_rerank_v1`, `mineru_crossdoc_bridges_v1`,
  `mineru_hub_candidates_v1`, `latex_vs_mineru_intradoc`, `rerank_threshold_portability`.
