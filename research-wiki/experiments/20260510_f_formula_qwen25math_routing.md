---
type: experiment
node_id: exp:20260510_f_formula_qwen25math_routing
title: "F-formula Mode B — Qwen2.5-Math-7B routing + RRF fusion"
date: 2026-05-10
status: pending
job: 68281
---

# Hypothesis

C11 the formula R@10 ≈ 0.56 ceiling is dense-encoder bound on LaTeX content. Ten configurations across graph topology, NL augmentation, surface form normalization, and reranker family swap all stay ≤ 0.56. The only untested lever is a true math-aware encoder.

If a math-specialized encoder substantially lifts the formula bucket on smoke50, C11 is fully validated and we have a positive finding to publish. If it does not lift, C11 is upgraded once more and formula retrieval is declared structurally bounded for any LLM-style dense encoder.

# Design

Mode B routing (formula-only):

- Qwen3-Embedding-4B keeps producing the baseline ranking over all 2809 corpus passages (existing `ranking_v1_enriched.jsonl`, R@10=0.6195).
- Qwen2.5-Math-7B re-encodes only the 1253 formula passages and all 473 queries (mean-pool last hidden state, L2-normalize, hidden_size=3584).
- For each query, a math-only ranking over the 1253 formula passages is produced via cosine sim.
- Reciprocal-rank fusion combines the two rankings: formula passages get two votes (Q3 and Math), non-formula passages get one vote (Q3 only). Sweep RRF k ∈ {10, 20, 60}.

Cleanest A/B isolation: math-aware signal is restricted to the formula bucket, so any figure/table regression cannot mask formula gains.

# Inputs

- Corpus: `data/05_eval/dense_retrieval/rebuilt_20260417/augmented/corpus_v1_enriched.jsonl`
- Queries: `data/03_queries/M4query_v1/queries.jsonl` (473)
- Qrels: `data/03_queries/M4query_v1/qrels.jsonl`
- Baseline ranking: `data/05_eval/dense_retrieval/rebuilt_20260417/ranking_v1_enriched.jsonl`
- Smoke50 source: `data/03_queries/M4query_smoke50/queries.jsonl`
- Encoder: HF `Qwen/Qwen2.5-Math-7B` (auto-download to HF cache)

# Outputs

`data/05_eval/dense_retrieval/qwen25math_formula_routing/`:
- `formula_embeddings.npy` (1253 × 3584)
- `query_embeddings.npy` (473 × 3584)
- `ranking_rrf_k{10,20,60}.jsonl`
- `eval_report.json` with `overall`, `formula_bucket_overall`, `smoke50`, `formula_bucket_smoke50` for each k

# Decision rules (set before run)

- HD: best `formula_bucket_smoke50.R@10` ≤ 0.56 — C11 fully validated, mark "math-aware dense encoder cannot rescue LaTeX in M4query distribution"; close F-formula path; pivot to LLM-judge / generation-side improvements.
- D2: best `formula_bucket_smoke50.R@10` ∈ (0.56, 0.62] — partial lift; report carefully, document RRF-k sensitivity.
- D1: best `formula_bucket_smoke50.R@10` > 0.62 — first ceiling break; promote to paper main result; trigger overall-corpus regression analysis (figure/table buckets must not drop).

# Implementation notes

- `scripts/eval_formula_qwen25math_routing.py` — single self-contained encoder + fusion + eval (~250 lines).
- `slurm_scripts/52_f_formula_qwen25math_routing.sh` — A6000 / 2h / 24G mem (slurm capped from 48G).
- minerU env, transformers 4.57 (Qwen2.5-Math is older arch, no overlay needed).
- Mean-pool over `attention_mask`-weighted last hidden state, then L2 normalize. No instruct chat template, no special tokens beyond tokenizer defaults.
- RRF skips passages absent from a given lane (standard rank-aggregation behavior).

# Status

- 2026-05-10T15:42Z: job 68269 started on gpu-a6000-1, died at HF model download because CephFS quota was already 4.6 GB over (300 GB cap, 304.6 GB used). Only config + tokenizer downloaded before failure.
- 2026-05-10T16:30Z: freed 15 GB by removing unrelated `models--stabilityai--stable-diffusion-xl-base-1.0` from HF cache; re-submitted as **job 68281** (PENDING). Same script / args / inputs.

# Refs

- [claim:C11](../claims/C11_formula_ceiling_is_dense_encoder_bound.md) — what this experiment closes / strengthens
- [exp:20260510_f_formula_caption](20260510_f_formula_caption.md) — caption injection HD predecessor
- [exp:20260510_f_formula_math_norm](20260510_f_formula_math_norm.md) — LaTeX normalization HD predecessor
- [exp:20260510_b1_phase2_lineno](20260510_b1_phase2_lineno.md) — graph topology HD predecessor
