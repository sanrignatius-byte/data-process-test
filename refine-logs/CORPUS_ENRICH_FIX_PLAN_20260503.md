# Plan — Corpus Enrichment Mapping Fix (P0 after CE rerank failure)

**Date**: 2026-05-03
**Status**: Ready for execution (handed off to coder assistant)
**Owner**: fourth assistant (coder)
**Target wall time**: ~75 min (15 min diagnose + 10 min fix+rebuild + 10 min GPU re-baseline + 30 min GPU re-rerank + 10 min decision)

---

## Why This Plan Exists

The cross-encoder rerank experiment (job 66349, BGE-reranker-v2-m3 on dense top-500) produced two findings:

1. **R@10 dropped catastrophically (0.6195 → 0.4482, −17pp)** under CE alone, because BGE-reranker actively demotes figure passages whose `text` is `[Image: xxx.jpg]` — they look like garbage to a NL reranker (correctly, given the input).
2. **R@100 cracked +2.3pp (0.8636 → 0.8869) via RRF(dense, CE)** — the first time the dense recall ceiling has moved since 2026-04-17.

Direct inspection of `data/05_eval/dense_retrieval/rebuilt_20260417/augmented/corpus_v1_enriched.jsonl` against `data/02_enriched/multimodal_elements_enriched.json` revealed:

- **63.5% of figure passages (695 / 1095)** have `text = "[Image: xxx.jpg]"`
- **42.2% of those degraded figures (293 / 695)** have a complete `enriched_content` field available in the enrichment source — **the corpus build is silently dropping this enrichment via a key-mapping miss**

This is a **data bug, not an algorithm choice**. Fixing it is upstream of every algorithmic next step (F1 = Qwen3-Reranker-4B, F3 = HyDE, formula-targeted query expansion). The CE failure already proves that switching reranker without fixing the corpus is likely to fail in the same way.

The third assistant's F1 → F2 → F3 ordering should be inverted: this plan executes the new P0.

---

## Pre-confirmed Inputs (do NOT redo)

| Data point | Value | Source |
|---|---|---|
| Total figure passages in M4query_v1 corpus | 1095 / 2809 | `corpus_v1_enriched.jsonl` |
| Figure passages with `text = [Image:...]` | **695 (63.5%)** | direct grep of corpus |
| Enrichment entries in canonical source | 1285 (across 76 docs) | `multimodal_elements_enriched.json` documents.*.elements |
| Figure passages with enrichment available | 664 / 1095 (60.6%) | passage_id ↔ element_id join |
| **Degraded figures with enrichment dropped** | **293 / 695 (42.2%)** | join on stripped doc-prefix key |
| CE rerank R@10 (current best CE result) | 0.4482 alone / 0.6258 via RRF | `posthoc_fusion_metrics.json` job 66349 |
| RRF R@100 lift over dense baseline | +2.33pp (0.8636 → 0.8869) | same |

---

## Phase A — Diagnose the Mapping Miss (15 min, no GPU)

**Goal**: identify the EXACT reason 293 figure passages with available enrichment fail to receive it during corpus build.

1. Read [scripts/build_graph_augmented_corpus.py](scripts/build_graph_augmented_corpus.py) — note:
   - which enrich source file(s) the script loads (line 100 onwards, `_build_enrich_index` function)
   - the key normalization in `build_v1_enriched` (line 211: `short_eid = pid[len(doc_id) + 1:] if pid.startswith(doc_id + "_") else pid`)
2. Write a diagnostic script `scripts/diagnose_corpus_enrich_mapping.py` (≤80 LOC) that:
   - loads the canonical enrichment source(s) the build script uses (verify it's `multimodal_elements_enriched.json` or whatever else)
   - loads `corpus_v1_enriched.jsonl`
   - for each degraded figure passage, computes the candidate keys (raw pid, stripped pid, doc_id+element_id reconstructions) and checks which one — if any — hits the enrichment index
   - bucket the 695 degraded figures into:
     - **D1**: enrichment available under exact key matched by current build logic (should be 0 — those would already be enriched)
     - **D2**: enrichment available under different key format (the 293 — diagnose the format gap)
     - **D3**: enrichment genuinely missing for this element (the 402)
   - dump 5 sample (pid, expected_key, available_key) triples for D2

3. **Gate**: if D2 ≠ ~293 or the format gap isn't a single coherent rule, escalate before Phase B. Don't apply blind patches.

Output: `data/05_eval/corpus_fix_v1/diagnose_report.md` — root cause + proposed patch (one-paragraph).

---

## Phase B — Patch and Rebuild (10 min, no GPU)

1. Apply the minimum patch to [scripts/build_graph_augmented_corpus.py](scripts/build_graph_augmented_corpus.py) that fixes the D2 mapping. Likely candidates (depending on Phase A finding):
   - Add a doc-prefixed alias in `_build_enrich_index` so both `{element_id}` and `{doc_id}_{element_id}` are queryable
   - Fix the `short_eid` strip if doc_id contains underscores or has a different format than expected
   - Point to the correct enrich source if `multimodal_elements_enriched.json` is not what the build is actually loading
2. Rebuild only the M4query_v1 corpus (not the 1040-doc production corpus):

   ```bash
   python scripts/build_graph_augmented_corpus.py \
     --output-dir data/05_eval/corpus_fix_v1/ \
     --tag v1_enriched_fixed \
     [whatever flags the existing build needs]
   ```

3. Verification gate (run inline, before any eval):
   - Figure `[Image:]` rate must drop from 63.5% to **≤ 25%** (since 293/695 = 42% should now be filled, leaving ~402/1095 = 36.7% degraded — call ≤ 25% the success bar to allow for some defensive headroom)
   - Total enriched figure count must rise from 400 → ≥ 690 (≥ 60% of 1095)
   - **If verification fails, do not proceed to Phase C; revert and re-diagnose.**

Output: `data/05_eval/corpus_fix_v1/corpus_v1_enriched_fixed.jsonl` + a `verify_report.md` with before/after counts.

---

## Phase C — Re-baseline Dense + Graph Rerank Ceiling (~10 min GPU)

Slurm: `slurm_scripts/46_corpus_fix_rebaseline.sh` (new, A6000, minerU env).

Re-run the two anchors that define the current ceiling:

1. **Dense baseline**: `eval_dense_retrieval.py` on the fixed corpus, M4query_v1 qrels
2. **Graph rerank ceiling**: `eval_graph_topk_rerank.py` with the best known config (`explicit_only + static_plus_neighbor`)

Output:
- `data/05_eval/corpus_fix_v1/dense_baseline_metrics.json`
- `data/05_eval/corpus_fix_v1/graph_rerank_metrics.json`
- `data/05_eval/corpus_fix_v1/delta_table.md` — side-by-side before/after for R@1/5/10/100/MRR

---

## Phase D — Re-run BGE Rerank + RRF on Fixed Corpus (~30 min GPU)

**Conditional**: only run if Phase C shows dense R@100 moves by ≥ +2pp. If R@100 is unchanged or moves <+2pp, skip Phase D and go straight to Phase E with that signal.

If gate passes, reuse [scripts/cross_encoder_rerank.py](scripts/cross_encoder_rerank.py) and [slurm_scripts/45_ce_rerank_bge_v2m3.sh](slurm_scripts/45_ce_rerank_bge_v2m3.sh) with the new corpus path. Same fp16 + max_length=2048 + bs=64 settings (those were tuned in job 66349).

Output: `data/05_eval/corpus_fix_v1/ce_rerank_metrics_fixed_corpus.json` (full replace, RRF k∈{20, 60}, modality breakdown).

---

## Phase E — Decision Gate (10 min, write only)

Output: `refine-logs/CORPUS_FIX_DECISION_20260503.md` (≤1 page).

Read Phases C + D and apply the rules **in order, first match wins**:

```
Variables:
  d_R@100  = dense R@100 delta (fixed corpus − original)
  d_R@10   = dense R@10 delta
  g_R@10   = graph rerank ceiling delta (fixed − 0.6913)
  ce_R@10  = best CE/RRF R@10 on fixed corpus (if Phase D ran)

Decision rules:

D1. if g_R@10 ≥ +0.030:
      → Outcome: corpus fix alone broke the 0.6913 ceiling.
      → Next: lock in fixed corpus as new baseline. Re-run all open
              experiments (Plan 1, Plan 2, etc.) on the new corpus before
              any new algorithm work.

D2. elif ce_R@10 ≥ 0.72:
      → Outcome: fixed corpus + BGE rerank achieves the original target.
      → Next: tune RRF weight, then close out the rerank track.
              Move to F3 (HyDE) only if R@1 lift is also wanted.

D3. elif d_R@100 ≥ +0.030:
      → Outcome: dense ceiling moved meaningfully but R@10 still below
                 0.72. Reranker is now the bottleneck on a healthier corpus.
      → Next: F1 (Qwen3-Reranker-4B) as a single targeted experiment,
              same eval slot.

D4. elif d_R@100 < +0.010:
      → Outcome: corpus fix didn't matter at the ceiling level — the bug
                 was theoretical. CE rerank's modality bias is real on its
                 own merits, not driven by the [Image:] artifact.
      → Next: skip F1, go directly to formula-targeted query expansion or
              a different reranker family (Cohere rerank-3.5, Voyage
              rerank-2). Re-think the modality story.

D5. else (mixed signal):
      → Outcome: partial corpus lift. Investigate per-modality R@10 to
                 decide whether to fix more (table/formula passages may
                 have analogous bugs) before any new model.
      → Next: re-audit table + formula passages with same diagnostic.
```

Report must include:
- Phase B verification numbers (figure `[Image:]` rate before/after, enrichment hit rate before/after)
- Phase C delta table (5 metrics × 2 systems × {old, new, Δ})
- Phase D summary if it ran
- Matched rule (D1–D5) + the triggering numerics
- Recommended next experiment (single-line)

---

## Out-of-scope

- Don't touch tables / formulas in this plan. They may have analogous bugs but Phase E rule D5 routes there if needed.
- Don't run Plan 1 (VL enrich-only) or Plan 2 (cross-doc citation) on either corpus version.
- Don't start F1 (Qwen3-Reranker-4B) or F3 (HyDE) — Phase E gates them.
- Don't change the M4query_v1 qrels. The eval set is fixed; only the corpus moves.

---

## File Manifest

| Path | Action | Owner |
|---|---|---|
| `scripts/diagnose_corpus_enrich_mapping.py` | New, ≤80 LOC | coder |
| `scripts/build_graph_augmented_corpus.py` | Patch (minimal diff) | coder |
| `slurm_scripts/46_corpus_fix_rebaseline.sh` | New, ~30 lines | coder |
| `data/05_eval/corpus_fix_v1/diagnose_report.md` | Phase A output | coder |
| `data/05_eval/corpus_fix_v1/corpus_v1_enriched_fixed.jsonl` | Phase B rebuild | coder |
| `data/05_eval/corpus_fix_v1/verify_report.md` | Phase B verification | coder |
| `data/05_eval/corpus_fix_v1/{dense_baseline,graph_rerank}_metrics.json` | Phase C | coder |
| `data/05_eval/corpus_fix_v1/delta_table.md` | Phase C aggregate | coder |
| `data/05_eval/corpus_fix_v1/ce_rerank_metrics_fixed_corpus.json` | Phase D (conditional) | coder |
| `refine-logs/CORPUS_FIX_DECISION_20260503.md` | Phase E report (≤1 page) | coder |
| `research-wiki/experiments/20260503_corpus_enrich_fix.md` | New wiki experiment node | coder |
| `research-wiki/log.md` | Append timestamped one-line summary on completion | coder |
| `research-wiki/index.md` | Add link to new experiment node | coder |

---

## Acceptance Criteria

1. **Phase A**: `diagnose_report.md` names the exact key-format mismatch (e.g. "build script keys on `{element_id}` but enrichment source keys on `{doc_id}_{element_id}`") and identifies ≥ 290 of the 293 expected D2 cases. The remaining ~3 may be edge cases — note them, don't block.
2. **Phase B**: figure `[Image:]` rate ≤ 25% in the rebuilt corpus; total enriched figure count ≥ 690. If either fails, halt and report.
3. **Phase C**: `delta_table.md` reports R@1 / R@5 / R@10 / R@100 / MRR for both dense baseline and graph_static_plus_neighbor on the fixed corpus, with deltas vs the rebuilt_20260417 anchors (0.6195 / 0.6913 / 0.8636).
4. **Phase D** (if triggered): per-modality R@10 reported for fixed-corpus CE+RRF, comparable to the original modality breakdown.
5. **Phase E**: `CORPUS_FIX_DECISION_20260503.md` contains matched rule (D1–D5), triggering numerics, and a single-line next-experiment recommendation.
6. Wiki updated: experiment node + log.md timestamp + index.md link.

---

## Notes for the coder

- The build script's `_build_enrich_index` likely loads multiple sources by config; check ALL of them, not just `multimodal_elements_enriched.json`. The 293 might be split across several enrich files with different key conventions.
- If you find that the 1040-doc production corpus uses a different enrich source than the M4query_v1 53-doc corpus, **only fix the M4query_v1 path in this plan**. Production corpus is out of scope here.
- The rebuilt corpus must remain compatible with the existing `eval_dense_retrieval.py` and `eval_graph_topk_rerank.py` interfaces — don't change passage_id format, just the `text` field content.
- If Phase A reveals the bug is in the upstream enrichment file (some figures genuinely never got enriched, but were silently expected to), surface that as a separate issue to land in `gap_map.md` — don't try to enrich them in this plan (LLM-cost out of scope).
- Strategic context: the +2.3pp R@100 from RRF is the only ceiling-cracking signal we've ever measured. This plan tests the hypothesis that the ceiling is a corpus artifact. A negative result (D4) is also high-value information.
