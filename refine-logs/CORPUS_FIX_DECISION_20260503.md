# Corpus Enrichment Fix — Decision Report (D5: Antipattern)

**Date**: 2026-05-03 (UTC)
**Plan**: `refine-logs/CORPUS_ENRICH_FIX_PLAN_20260503.md`
**Verdict**: **D5 — file as antipattern, do NOT promote.**
**Phase D skipped**: dense R@100 Δ < +2pp gate failed (Δ = −2.0pp).

---

## 1. Premise (recap)

The 4/17 baseline corpus `corpus_v1_enriched.jsonl` (rebuilt_20260417/augmented)
contained 695 / 1095 (63.5%) figure passages whose text was just the placeholder
`[Image: <path>.jpg]`. The hypothesis driving F2 was: enrich those figures with
descriptive text, expect a recall lift comparable to the +7pp graph-rerank step.

## 2. Phase A — root-cause diagnosis (`scripts/diagnose_corpus_enrich_mapping.py`)

Two independent bugs in `scripts/build_graph_augmented_corpus.py`:

1. **Format-mismatch bug** in `load_enriched_index`. The loader only parsed a
   nested `pair.element_a.element_id` layout. The current hub files use flat
   `element_a_id` keys → 0 entries returned. The MODORA per-element file
   (`data/02_enriched/multimodal_elements_enriched.json`, 1285 enriched
   elements, `documents.elements` layout) was never even passed in.

2. **Mutually-exclusive priority** in `build_element_text`. When MODORA enrich
   was present, the function used `title + content` and **skipped** the
   `graph_elem` branch (caption + content + context_before + context_after).
   For ~640 figures that already had rich paper-domain context, this would
   replace caption+context with visual-only description.

Diagnose buckets (1095 figures total):

| bucket | meaning | count |
|---|---|---|
| D1 | text starts `[Image:`, no MODORA entry → genuinely missing upstream | **402** |
| D2 | text starts `[Image:`, MODORA entry exists → silently dropped | **293** |
| D3 | not degraded (had graph context already) | **400** |

All 293 D2 cases match MODORA via full passage_id (no prefix stripping).

Gate: D2 ≈ 293 ✓ — single coherent rule. Proceeded to Phase B.

## 3. Phase B — patches

Three changes applied to `scripts/build_graph_augmented_corpus.py`:

* `load_enriched_index`: added second branch detecting `documents` top-level
  key, ingesting `documents[*].elements[*]` → `{enriched_title, enriched_content}`.
* `build_v1_enriched`: lookup `enrich_index[pid]` first (MODORA uses full
  passage_id as element_id), fall back to short_eid (legacy).
* `build_element_text`: changed mutually-exclusive priority to **layered
  additive** — concatenate MODORA visual + graph caption/context.

Two corpus variants built for ablation:

| variant | strategy | mean figure text len | degraded figures |
|---|---|---|---|
| OLD (rebuilt_20260417) | enrich missed → graph fallback | 128 | 695 (63.5%) |
| **fix_v1** (replace) | MODORA only, skip graph context | 405 | 398 (36.3%) |
| **fix_v2** (additive) | MODORA + graph caption/context | 683 | 398 (36.3%) |

(Residual 398 = D1 floor, no MODORA upstream coverage.)

## 4. Phase C — re-baseline on M4query_v1 (jobs 66371, 66384)

All metrics on 473 queries / 2809 passages / 946 qrels. Encoder = Qwen3-Embedding-4B.

### 4.1 Dense baseline

| metric | anchor (4/17) | fix_v1 | Δ_v1 | fix_v2 | Δ_v2 |
|---|---|---|---|---|---|
| R@1   | — | 0.1786 | — | 0.1850 | — |
| R@5   | — | 0.4154 | — | 0.4725 | — |
| R@10  | **0.6195** | 0.5106 | **−10.9pp** | **0.5888** | **−3.1pp** |
| R@100 | **0.8636** | 0.7569 | −10.7pp | 0.8436 | **−2.0pp** |
| MRR   | 0.6122 | 0.4900 | −12.2pp | 0.5173 | −9.5pp |

### 4.2 Graph rerank (best variants)

| pipeline | anchor R@10 | fix_v1 R@10 | fix_v2 R@10 |
|---|---|---|---|
| graph_explicit_only / static_prior | ~0.69 | 0.5497 | 0.6385 |
| graph_explicit_only / static_plus_neighbor | **0.6913** | 0.5888 | **0.6860** |
| graph_explicit_plus_same_chunk / static_prior | ~0.69 | 0.5433 | 0.6268 |

fix_v2 best graph-rerank (0.6860) is **−0.5pp** below the standing ceiling of
0.6913 — the additive enrichment recovers most of fix_v1's regression but does
not improve over the buggy baseline.

## 5. Decision (apply rules D1–D5 from plan)

| rule | criterion | met? |
|---|---|---|
| D1 | dense R@10 ≥ +1pp | ❌ (−3.1pp) |
| D2 | dense R@10 +0.5–1pp **and** R@100 +2pp | ❌ |
| D3 | graph rerank improves > dense alone | ❌ (also regresses) |
| D4 | dense flat **and** R@100 +2pp ⇒ trigger Phase D CE | ❌ (R@100 −2pp) |
| **D5** | regression ⇒ file as antipattern | ✅ |

**Verdict: D5.** MODORA visual enrichment is net-negative on M4query_v1.

## 6. Why does enrichment hurt? (mechanism)

Same direction as the **BGE-reranker text-bias** finding from the cross-encoder
pilot (refine-logs/CEILING_DECISION_20260503.md S3). M4query_v1 queries are
domain/text style ("RoBERTa pretraining objective", "PASCAL 2012 instance
distribution"). MODORA descriptions are **modality-faithful but
domain-detached** — they describe what the figure *looks like* rather than
what it *means in the paper*:

* **Anchor figure text** (figure_1, 4/17): *"Fig. 14 Performance of the
  'optimistic' computer vision model as a function of object properties.
  The x-axis corresponds to object properties annotated by human labelers
  for each object class (Russakovsky et al., 2014) …"* — 1085 chars,
  paper-domain language.
* **fix_v1 text**: *"Object detection precision vs man-made and natural
  object properties. Average precision increases with the annotated property
  level from None to Low for both Man-made and Natural categories …"* — 536
  chars, no paper-context anchor.
* **fix_v2 text**: visual prefix + paper context concatenated (3109 chars).
  The 600+ chars of visual prefix dilute the text-style query match, only
  partly compensated by the recovered paper context.

Both regressions confirm: **on text-style retrieval benchmarks, modality-only
visual descriptions act as semantic noise**. The same passages were already
retrievable via their paper context.

## 7. Actions taken

* **Reverted** `DEFAULT_ENRICHED_FILES` to legacy hub list (no MODORA by
  default). Canonical pipeline behaviour unchanged.
* **Kept** `load_enriched_index` `documents.elements`-format branch
  (latent-bug fix; no production effect because no caller passes the file).
* **Kept** `build_element_text` additive priority (latent bug fix; no
  production effect since `enrich_index` is empty under default args).
* Added inline comments at `DEFAULT_ENRICHED_FILES` warning future maintainers
  that MODORA enrichment is net-negative on M4query_v1; require per-benchmark
  evaluation before enabling.

## 8. Artifacts (kept)

```
scripts/diagnose_corpus_enrich_mapping.py
data/05_eval/corpus_fix_v1/                # diagnose + replace-variant baseline
data/05_eval/corpus_fix_v2/                # additive-variant baseline
slurm_scripts/46_corpus_fix_rebaseline.sh  # job 66371 (fix_v1)
slurm_scripts/47_corpus_fix_v2_rebaseline.sh  # job 66384 (fix_v2)
refine-logs/CORPUS_FIX_DECISION_20260503.md   # this file
```

## 9. Next-step recommendations

1. **Do not use MODORA visual enrichment as text-side replacement** on text-
   style benchmarks. If multimodal grounding is wanted, route the visual
   description through a separate retrieval lane (vision-language encoder)
   and combine via late fusion, not corpus-level text replacement.
2. **402 D1 figures** (no MODORA coverage at all) remain untouched — out of
   scope here; would require a fresh upstream MODORA run, which we now know
   would not improve M4query_v1 retrieval anyway.
3. **Direction worth re-considering**: per the CE pilot, RRF(dense, CE)
   already yielded R@100 +2.3pp. That remains the only direction with a
   confirmed positive signal on the ceiling. CE on the *anchor* corpus is the
   higher-EV next experiment, NOT CE on a corpus_fix variant.
