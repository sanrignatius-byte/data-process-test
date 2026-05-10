# Ceiling Decision — 2026-05-03

**Status**: Decided
**Trigger**: Phase B of `refine-logs/CEILING_PROFILING_PLAN_20260503.md`
**Inputs**: `data/05_eval/failure_analysis/missed_qrel_ranks.json`, `decision_tables.md` (slurm job 66324)

---

## Result

**Matched rule: R2 — cross-encoder rerank on dense top-500**

### Triggering numerics

| variable | value | threshold |
|---|---:|---:|
| `r_low` (rank ∈ (100, 500]) | **0.690** | ≥ 0.60 ✅ |
| `m_form` | 0.496 | ≥ 0.40, but… |
| `form_high` (formula misses with rank > 2000) | **0.016** | ✗ ≥ 0.50 → R1 **rejected** |
| `r_mid` (rank ∈ (500, 2000]) | 0.264 | — |
| `r_high` (rank > 2000) | 0.047 | — |

R1 is rejected because `form_high = 0.016`: 63 of 64 formula misses are still inside top-2000, and 42/64 (65.6%) are inside top-500. The encoder is **not** failing to bridge NL ↔ LaTeX semantics. R2 fires on `r_low = 0.690`.

### What this means

- 89 of 129 missed qrels (69%) sit in rank 101–500. They are already in the candidate pool of any reasonable top-500 dense retrieval; the gap is ordering, not recall.
- A modality-blind cross-encoder reranker on dense top-500 is the lowest-cost intervention that can lift R@10 above the 0.6913 ceiling, because:
  - candidate recall is high (cf. dense R@500 ≈ 0.95+ given r_low share),
  - all three target modalities (formula 65.6%, figure 73.5%, table 71.0%) are predominantly in the rerankable bucket.

---

## Recommended next experiment

**Cross-encoder rerank on dense top-500 of `ranking_v1_enriched.jsonl`.**

| item | value |
|---|---|
| Reranker (primary) | `BAAI/bge-reranker-v2-m3` (MIT, multilingual, scientific-tolerant) |
| Reranker (alt) | `Qwen/Qwen3-Reranker-4B` (same family as encoder, larger) |
| Input | dense top-500 from `data/05_eval/dense_retrieval/rebuilt_20260417/augmented` |
| Eval set | M4query_v1 (473 queries, 2809 passages) |
| Output metric | R@10 / R@100, primary signal R@10 vs ceiling 0.6913 |
| Budget — GPU | ~30 min on a single A6000 (473 queries × 500 candidates ≈ 236K pairs; bge-v2-m3 ≈ 2K pairs/s/A6000 with bf16) |
| Budget — $ | 0 (local cluster) |
| Wall-time | ~45 min including build + eval |
| Success bar | R@10 ≥ 0.72 on M4query_v1 (i.e. +3pp over 0.6913). Stretch: R@10 ≥ 0.75 |
| Failure mode | If R@10 ≤ 0.70, formula-specific weakness is the likely cause; fall back to formula-targeted query expansion as the second pilot |

### One-paragraph design

Rebuild a candidate file of dense top-500 per query (rerunning `eval_dense_retrieval.py` once with `--top-k 500`, or slicing from the already-emitted `data/05_eval/failure_analysis/full_ranking.jsonl`). For every (query, candidate_passage) pair, score with bge-reranker-v2-m3 in bf16, keep the top-100, and re-score using the same evaluator that produced 0.6913. Compare against the dense baseline (0.6195) and the graph rerank ceiling (0.6913). If R@10 ≥ 0.72, fuse the cross-encoder score with the graph static prior (linear combination, sweep weight ∈ {0.1, 0.2, 0.3}) to check whether the two signals are additive. Record per-modality R@10 to verify formula is rescued at the same rate as figure/table.

---

## Out-of-scope for this decision

- Plan 1 (VL enrich-only) and Plan 2 (cross-doc citation) remain side tracks per the profiling plan.
- Math-aware encoder / HyDE / corpus enrichment fix are **not** triggered by this data and should not be started yet.

## Sanity checks for the next assistant (post-decision audit)

### S1 — Passage length vs reranker 8K context

`corpus_v1_enriched.jsonl` (n=2809), `text` field char length, est tokens = chars / 4:

| stat | chars | est tokens |
|---|---:|---:|
| p50 | 136 | 34 |
| p90 | 514 | 128 |
| p99 | 54 153 | 13 538 |
| max | 202 031 | 50 507 |
| passages > 32K chars (~> 8K tok) | **51 / 2809 (1.8%)** | — |

**Action item for the rerank pilot**: 51 passages will exceed BGE-reranker-v2-m3's 8K context. Truncate to first 7 500 tokens before scoring (or split into windows and take max). Do not silently feed >8K — HuggingFace will truncate from the right and lose the tail of long enriched figure/table descriptions.

Queries are well-behaved: char p50=143, p90=204, max=235 (~60 tokens), nowhere near the limit.

### S2 — R@K curve from `full_ranking.jsonl` (sanity ceiling)

| K | R@K |
|---:|---:|
| 10 | 0.6195 |
| 100 | 0.8636 |
| 200 | 0.9080 |
| **500** | **0.9577** |
| 1000 | 0.9810 |
| 2000 | 0.9937 |

**Reranker theoretical ceiling**: R@500 = 0.9577. Hitting the **stretch** target R@10 ≥ 0.75 requires the cross-encoder to promote 0.75 / 0.9577 ≈ **78.3%** of in-pool gold into top-10. The primary target R@10 ≥ 0.72 requires 75.2%. Both are aggressive but achievable for a strong reranker on natural-language candidates; expect formula-heavy queries to be the slow tail.

### S3 — Strong baseline

The cross-encoder pilot must beat **two** baselines, not one:
- dense (`ranking_v1_enriched`) R@10 = 0.6195
- **graph rerank ceiling** `graph_explicit_only_fixed/metrics_graph_static_plus_neighbor.json` R@10 = **0.6913** ← this is the headline number to break

Recommended fusion experiment (already in §One-paragraph design): linearly combine cross-encoder score with `graph_static_plus_neighbor` boost, sweep weight ∈ {0.1, 0.2, 0.3}; the two signals come from disjoint mechanisms (semantic similarity vs graph prior) so they should be additive on a non-trivial slice.

## Pointers

- Ranks: `data/05_eval/failure_analysis/missed_qrel_ranks.json`
- Tables: `data/05_eval/failure_analysis/decision_tables.md`
- Full ranking dump (used for top-500 slicing): `data/05_eval/failure_analysis/full_ranking.jsonl`
- Eval report: `data/05_eval/failure_analysis/eval_report_full_rank.json`
- Slurm log: `logs/fail_full_rank_66324.out`
- Strong baseline metrics: `data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_only_fixed/metrics_graph_static_plus_neighbor.json`
