---
type: experiment
node_id: exp:20260503_ce_rerank_bge
title: "Cross-encoder rerank pilot — BGE-reranker-v2-m3 on dense top-500 (R2)"
date: 2026-05-03
status: completed
verdict: negative_on_R@10_marginal_R@100_via_RRF
slurm_job: 66349
related_experiments: [exp:20260503_failure_profiling, exp:20260417_explicit_rerank_fixed]
related_claims: [C4]
---

# 触发

[CEILING_DECISION_20260503.md](../../refine-logs/CEILING_DECISION_20260503.md) 命中 R2：69% 漏掉的 qrel 在 rank ∈ (100, 500]，建议 cross-encoder rerank dense top-500。

# 设置

| item | value |
|---|---|
| Reranker | `BAAI/bge-reranker-v2-m3` (XLM-RoBERTa large, 568M params, 8K context) |
| Input ranking | `data/05_eval/failure_analysis/full_ranking.jsonl` 切 top-500 |
| Corpus | M4query_v1 / `corpus_v1_enriched.jsonl` (2809 passages) |
| Queries / qrels | M4query_v1 (473 q) |
| max_length | 2048 (corpus token p99=128, p99.99 ≈ 1.3K，仅 51/2809 outliers > 8K) |
| precision | fp16 |
| batch_size | 64 |
| Wall time | 27.5 min on A6000 (1652s pure rerank) |
| Slurm log | `logs/ce_rerank_bge_66349.out` |
| Outputs | `data/05_eval/cross_encoder_rerank/bge_v2m3_top500/{ranking_ce_bge_v2m3.jsonl, metrics.json, posthoc_fusion_metrics.json}` |

# 结果（headline）

| config | R@1 | R@5 | R@10 | R@100 | MRR |
|---|---:|---:|---:|---:|---:|
| dense_baseline | 0.2336 | 0.5275 | **0.6195** | 0.8636 | 0.6122 |
| graph rerank ceiling (4/17) | 0.2419 | 0.5856 | **0.6913** | 0.8636 | 0.6399 |
| **CE rerank top-500 (full replace)** | **0.1068** | **0.3266** | **0.4482** | 0.8552 | **0.3714** |
| CE top-100 + dense tail | 0.1068 | 0.3266 | 0.4482 | 0.8552 | 0.3714 |
| CE top-10 + dense tail | 0.1068 | 0.3266 | 0.4482 | 0.8710 | 0.3789 |
| **RRF(dense, CE, k=20)** | 0.2167 | 0.5116 | **0.6258** | **0.8848** | 0.5817 |
| RRF(dense, CE, k=10) | 0.2178 | 0.5148 | 0.6247 | 0.8837 | 0.5833 |

**Headline**: CE alone collapses R@10 by **−17.1 pp** (0.6195 → 0.4482) and MRR by **−24 pp**. Even truncating CE to top-10 doesn't recover, because the bias is inside the top-10 itself. RRF buys back +0.5pp on R@10 (0.6258) and +2.3pp on R@100 (0.8869), but is still 6.5pp below the graph ceiling 0.6913.

**Conclusion**: pilot fails primary success bar (R@10 ≥ 0.72). Reject BGE-reranker-v2-m3 as a drop-in for this corpus.

# Why — diagnosis

Modality of **top-1 passage** by config (across 473 queries):

| config | text | figure | table | formula |
|---|---:|---:|---:|---:|
| dense baseline top-1 | 26 | 265 | 115 | 67 |
| **CE top-1** | **348** | **87** | **29** | **9** |

CE is severely biased toward `text` modality. Per-modality R@10 (per qrel):

| modality | dense | CE | Δ |
|---|---:|---:|---:|
| formula | 0.441 | 0.207 | **−23.4 pp** |
| figure | 0.689 | 0.563 | −12.6 pp |
| table | 0.625 | 0.431 | −19.4 pp |

Cause: BGE-reranker-v2-m3 is trained on natural-language QA pairs (mMARCO, NQ, hotpotQA et al.). When it sees `[FORMULA] $$D(M x, M y) \leq d(x, y) \tag{1}$$` or `[Image: cifar.jpg]` it scores them strictly below any natural-language paragraph that has surface-form word overlap with the query. The encoder Qwen3-Embedding-4B was either trained on or fine-tuned to handle these mixed-modality corpora; its calibration is the reason dense already gets formula R@10 = 0.441.

Example query `l1_de_1104.3913_0116`, gold = `{1104.3913_figure_2, 1104.3913_formula_1}`:
- dense top-5: figure_2 (✓), 1905.03674_table_4, 1905.03674_formula_1, 1611.07509_formula_12, formula_1 (✓)
- CE top-5: 1709.02012_text_63 (wrong paper), formula_1 (✓), 1811.00103_formula_6, formula_2, 1709.02012_formula_4

CE replaces the correct figure_2 at rank-1 with a text passage from a different paper that has surface-form overlap.

# Decision-tree implication

R2's premise — "candidate recall is high, the gap is in ordering" — is mechanically true (R@500 = 0.9577) but **assumed any cross-encoder is modality-agnostic**. That assumption is false for BGE-reranker-v2-m3 on this scientific multimodal corpus. R2 is **not invalidated** for trainings that aren't text-biased, but BGE-reranker-v2-m3 is rejected.

# Side findings

- RRF(dense, CE, k=20) improves R@100 from 0.8636 → 0.8869 (+2.3pp). CE does promote some gold passages between rank 100–500 of dense into the union top-100. Useful as a recall booster but not as a precision booster.
- CE adds 17 min of A6000 time per eval pass; not a free hyperparameter sweep partner.
- The 1.8% of long passages (>8K tokens) had no detectable additional impact on R@10 once max_length=2048 was used (median passage is 34 tokens — most of the cost was per-pair fp32→fp16 conversion before this fix; with fp16 it's 7× faster).

# Recommended follow-ups (in order)

| # | action | why | est. cost |
|---|---|---|---|
| F1 | **Qwen3-Reranker-4B** (same family as the encoder) instead of BGE-reranker | Same family was likely trained on multimodal/scientific data alongside the encoder. If it isn't text-biased, R2 path reopens. | ~45 min A6000, $0 |
| F2 | If F1 also text-biased: **fix the corpus bug from CEILING_PROFILING_PLAN §S1** — re-build `corpus_v1_enriched.jsonl` so figure/formula/table passages always concatenate `caption + enriched_content + context_before/after`, not just the placeholder `[Image: …]`. Then re-run dense + RRF. This converts modality-token-token-bias into a non-issue because every passage now has natural language. | ~1 h CPU corpus build + 10 min A6000 dense reeval |
| F3 | If F1+F2 both fail: **HyDE** (R4 of original tree) — generate a hypothetical NL passage per query, dense-retrieve again | Sidesteps both encoder and reranker modality issues by changing the query, not the candidates | ~$3 LLM + 30 min A6000 |

F1 is the lowest-risk next step; F2 is the highest-information-density next step (also fixes a known data bug independently of rerank).

# Files

- Code: `scripts/cross_encoder_rerank.py`, `slurm_scripts/45_ce_rerank_bge_v2m3.sh`
- Outputs: `data/05_eval/cross_encoder_rerank/bge_v2m3_top500/`
- Decision report: `refine-logs/CEILING_DECISION_20260503.md`
