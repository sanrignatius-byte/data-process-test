# Experiment Plan — VL Embedding Controlled Comparison (Enrich-only vs Image)

**Problem**: Current figure/table passages contain caption + context + enriched_content. 4B text achieves strong R@10 (0.53 figure, 0.50 table mixed) using primarily text signals. The value of VL image encoding over text enrichment alone is unknown.
**Method Thesis**: When the passage contains ONLY enriched_content (no caption, no context_before/after, no raw content), VL image encoding will outperform 4B text encoding if and only if the image carries information beyond what the LLM enrichment captures.
**Date**: 2026-05-03

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C_VL1 | VL image encoding provides measurable gain over text enrichment alone on figures | VL R@10 > text R@10 by ≥ 5pp on figure subset | B1 |
| C_VL2 | Enrichment quality is the binding constraint — richer enrichment shrinks the VL advantage | Per-element correlation: enrichment_word_count vs VL-text delta | B2 |

---

## Current Context

Per-modality R@10 with FULL text (caption + context + enriched_content):

| System | figure R@10 | table R@10 |
|--------|------------:|-----------:|
| `split_4B_text` mixed | 0.5307 | 0.4985 |
| `split_VL_2B_t5` mixed | 0.4102 | 0.0236 |

These numbers don't isolate VL's image-understanding value — 4B text wins because it has caption + context text that VL doesn't get (since VL encodes raw images, not captions).

**This experiment strips the text advantage**: both encoders get the same information budget — enriched_content only for text, raw image for VL.

---

## Experiment Blocks

### Block 1: Enrich-Only vs VL Image — Main Result [MUST-RUN]

- **Claim tested**: C_VL1
- **Why this block exists**: Isolate VL's image-understanding contribution from 4B text's caption-reading advantage
- **Dataset**: M4query_v1, 473 queries, 2809 passages
  - Figure passages: 1095 (852 have image_path, 243 fallback to text)
  - Table passages: 237 (228 have image_path)
- **Compared systems**:

  | Config | figure passage text | table passage text | text passages | Encoder |
  |--------|--------------------:|-------------------:|--------------|---------|
  | `text_enrich_only` | enriched_content only | enriched_content only | enriched_content only | Qwen3-Embedding-4B |
  | `vl_image` | raw image (no text) | raw image (no text) | enriched_content only | Qwen3-VL-Embedding-2B |
  | `text_full` (ref) | caption+ctx+enriched | caption+ctx+enriched | full text | Qwen3-Embedding-4B |

- **Metrics**: R@1, R@5, R@10, R@100, MRR; per-modality figure R@10, table R@10
- **Setup details**:
  - Corpus builder: modify `build_graph_augmented_corpus.py` to accept `--text-mode enrich_only`
    - `enrich_only`: passage text = `enriched_content` field only
    - If enriched_content is empty, use caption as fallback (mark in build_report)
  - Text encoder: Qwen3-Embedding-4B, max_length=512
  - VL encoder: Qwen3-VL-Embedding-2B, text max_length=512, image max_length=4096
  - VL environment: `PYTHONPATH=/projects/myyyx1/envs/qwen3vl_tf5_overlay:$PYTHONPATH`
  - Query encoding: text queries encoded by respective encoder (text for 4B, VL for 2B)
  - Evaluation: `eval_dense_retrieval.py`, M4query_v1 qrels
- **Success criterion**: VL image R@10 > text_enrich_only R@10 by ≥ 5pp on figure subset → VL has irreducible image-understanding value
- **Failure interpretation**: If text_enrich_only ≥ VL image, enrichment already captures all retrieval-relevant visual information. VL-2B is underpowered for this task (2B vs 4B), not "image understanding doesn't matter."
- **Table / figure target**: Paper ablation table
- **Priority**: MUST-RUN
- **Est. GPU time**: ~30 min (two encodings × 473 queries × 2809 passages)

### Block 2: Enrichment Quality Correlation [NICE-TO-HAVE]

- **Claim tested**: C_VL2
- **Why this block exists**: Understand whether better enrichment closes the VL-text gap
- **Dataset**: Post-hoc from B1
- **Compared systems**: Per-element analysis — for each figure, compute `enrichment_word_count` vs `(VL_rank - text_rank)` delta
- **Metrics**: Spearman correlation between enrichment length/quality and rank delta
- **Setup details**: Post-hoc from B1 rankings + enriched_content field
- **Success criterion**: Negative correlation — longer enrichment → smaller VL advantage (confirms hypothesis)
- **Table / figure target**: Appendix scatter plot
- **Priority**: NICE-TO-HAVE

---

## Run Order

| Milestone | Goal | Runs | Est. GPU-min | Decision Gate |
|-----------|------|------|-------------|---------------|
| M0 | Build enrich_only corpus + verify passage texts | — | 0 min | ≥ 80% of figure/table passages have enriched_content |
| M1 | Run text_enrich_only (4B) | 1 run | 10 min | Baseline R@10 established |
| M2 | Run vl_image (VL-2B) | 1 run | 15 min | Compare vs M1 |
| M3 | B2 post-hoc analysis | 0 min | 0 min | Correlation computed |

**Total estimated GPU-time: ~25 min**

---

## Risks and Mitigations

- **Risk**: 243/1095 figure passages have no image_path → fallback to text encoding in VL config
  - **Mitigation**: Report per-modality R@10 separately for "has image" vs "text fallback" subsets
- **Risk**: enriched_content coverage is only 40.4% on the 1040 corpus; on M4query_v1 53-doc it should be near 100%
  - **Mitigation**: M0 sanity check confirms coverage before running
- **Risk**: VL-2B (2B params) vs text-4B (4B params) is not a fair model-size comparison
  - **Mitigation**: Frame as "VL-2B is the only available VL embedding model; the question is whether it adds value beyond text-4B+enrichment"

---

## Implementation

### New/Modified Files

| File | Action | Purpose |
|------|--------|---------|
| `scripts/build_enrich_only_corpus.py` | New | Build corpus with enriched_content only, fallback to caption |
| `scripts/eval_vl_vs_text_enrich.py` | New | Run both encoders, compare per-modality R@10 |
| `slurm_scripts/43_vl_enrich_comparison.sh` | New | Slurm job: build corpus → text eval → VL eval → compare |

### Key Implementation Notes

1. The enrich_only corpus builder should use `data/02_enriched/multimodal_elements_enriched.json` for M4query_v1 docs (~100% enriched coverage on 53 docs)
2. VL encoder setup: reuse `eval_split_modality_vl.py`'s image path resolution logic (handles `mineru_output/{id}/auto/images/` paths)
3. Per-modality metric: filter qrels by `element_type` field from `multimodal_elements.json`
4. Output: `data/05_eval/dense_retrieval/vl_enrich_comparison/eval_report.json`
