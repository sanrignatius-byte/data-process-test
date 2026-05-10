---
type: claim
node_id: claim:C10
title: "Graph rerank effect is modality-selective (figure/table only)"
status: supported
date: 2026-05-10
related_experiments: [exp:20260505_smoke50_balanced_audit, exp:20260503_failure_profiling, exp:20260503_qwen3_rerank_fusion]
related_claims: [C1, C5, C7]
---

# Statement

Graph-based rerank (`explicit_only_fixed + static_plus_neighbor`) improves R@10 selectively by element modality:
- **figure**: +10.3pp over dense (0.7179 → 0.8205)
- **table**: +8.3pp over dense (0.6111 → 0.6944)
- **formula**: 0.0pp (graph and dense both at 0.5600; ties with Qwen3-Reranker-4B too)

Graph rerank does **not** improve formula retrieval. Three independent reranker families (dense embedding, graph rerank, Qwen3-Reranker-4B) all converge on R@10 = 0.56 for formula, suggesting this is the dense-encoder ceiling for `[FORMULA] $$...$$` LaTeX passages.

# Scope of evidence

- M4query_v1 (473 queries / 946 qrels, modality dist figure 218 / formula 138 / table 117 / **text 0**)
- M4query_smoke50 (50 queries balanced 17/17/16, derived 2026-05-10)
- Both datasets are paper-domain, dual-evidence (2 qrels per query)
- M4query_v1 has zero text-evidence queries — text modality is **not** validated by this claim

# Implication for paper claims

Replace prior over-generalized statements such as:
- C1 ("explicit static prior improves rerank")
- C5 ("typed crossdoc element edges lift R@10")
- C7 ("explicit-only static_plus_neighbor R@10 high")

with modality-scoped versions:
- "Graph rerank improves figure and table retrieval (+8 to +10pp R@10) on M4query benchmarks. No significant effect on formula retrieval."

# Evidence

| Source | Result |
|---|---|
| `data/05_eval/smoke50/per_system_per_modality.md` | T1 table |
| `data/05_eval/cross_encoder_rerank/qwen3_reranker_4b_transformers_anchor_top500/fusion_report.md` | M4query_v1 modality guard |
| `exp:20260503_failure_profiling` | Missed qrels 49.6% are formula |

# Why graph fails on formula

**5/10 update — 2 hypotheses tested**:

1. ~~Hypothesis A: formula nodes have low intra-document `same_section`/`adjacent_bridge` connectivity~~ — superseded by Hypothesis C
2. ~~Hypothesis B: chunk-element edges are wrong, graph signal doesn't reach formula~~ — **falsified by B1 Phase 2** ([exp:20260510_b1_phase2_lineno](../experiments/20260510_b1_phase2_lineno.md)). Rebuilt edges via LaTeX line_no (24/49 formulas got new chunk membership), formula R@10 still 0.5600.
3. **Hypothesis C** (now supported): **formula ceiling is dense-encoder bound, not graph-topology bound**. 6 independent configs (dense, graph explicit-only, graph explicit-only+lineno, graph explicit+virtual orig, graph explicit+virtual lineno, qwen3-CE) all hit R@10 ≤ 0.5600 on formula bucket. → see [claim:C11](C11_formula_ceiling_is_dense_encoder_bound.md).

# Open questions

1. Does math-aware encoder (Qwen3-Math, Mistral-Math) lift formula R@10 above 0.56?
2. Does adding `formula_proximity` virtual edges (formulas in same equation block) lift formula graph rerank?
3. Does the 0.56 saturate include passages with degraded LaTeX (e.g. `$$ \\tag{1}$$` text-only) or only well-formatted LaTeX?

# Status

- 2026-05-10: Created from smoke50 results
- Pending validation: F-formula experiment (Qwen3-Math encoder) — not started
