---
type: claim
node_id: claim:C11
title: "Formula retrieval ceiling is dense-encoder bound — not surface form, not graph topology, not NL augmentation"
status: supported
date: 2026-05-10
updated: 2026-05-10 (F-caption + math_norm = 10 configs total, 0 breaks 0.5600)
related_experiments: [exp:20260510_b1_phase2_lineno, exp:20260505_smoke50_balanced_audit, exp:20260510_f_formula_caption, exp:20260510_f_formula_math_norm]
related_claims: [C10, C8]
---

# Statement

The formula-modality retrieval R@10 ≈ 0.56 ceiling on M4query_v1 / smoke50 is determined
by the dense encoder's fundamental inability to represent mathematical semantics — not by
LaTeX surface form, graph topology, model size, or NL augmentation strategy.

# Evidence — 10 independent configurations, none breaks 0.5600 on smoke50 formula bucket

| # | Configuration | formula R@10 | Strategy class |
|---|--------------|---:|---|
| 1 | dense baseline (Qwen3-Emb-0.6B) | 0.5600 | — |
| 2 | dense (Qwen3-Emb-4B) | 0.5600 | model scale |
| 3 | graph explicit only | 0.5600 | graph topology |
| 4 | graph explicit + line_no fix | 0.5600 | edge construction |
| 5 | graph explicit + virtual orig | 0.5200 ⬇ | more graph edges |
| 6 | graph explicit + virtual line_no | 0.5200 ⬇ | edges + line_no |
| 7 | dense + F-caption injection | 0.4000 ⬇ | NL augmentation |
| 8 | graph + F-caption injection | 0.5200 ⬇ | NL + graph |
| 9 | **dense + LaTeX normalization** | **0.5600** | **surface form** |
| 10 | **graph + LaTeX normalization** | **0.5600** | **surface form + graph** |

3 strategy classes exhausted (graph topology, NL augmentation, LaTeX surface form).
One path remains: **true math-aware encoder** (model pretrained on mathematical text).

# Mechanism

Qwen3-Embedding is pretrained on natural language. Whether the input is raw LaTeX
(`\operatorname{opt}`) or normalized text (`opt`), the encoder produces near-uniform
similarity scores for formula passages because it has no internal representation of
mathematical structure. Graph rerank and NL augmentation are layering on noise.

The F-caption regression (-16pp) and math_norm null result (+0pp) together confirm
that the bottleneck is NOT in how the LaTeX is tokenized, but in what the encoder
learned during pretraining.

# Implication

To break formula 0.56, the **only viable lever** is a different encoder:

1. **Math-aware embedding model**: pretrained on arXiv LaTeX, math StackExchange, etc.
   - `jinaai/jina-embeddings-v3` (multilingual, strong on technical text)
   - Custom math-BERT fine-tuned on LaTeX corpus
2. **LLM-as-encoder**: use Qwen2.5-Math hidden states as embeddings
3. **LoRA fine-tune** Qwen3-Embedding-4B on LaTeX corpus (expensive, deferred)

# Why this matters for the paper

Together with C8 (cross-modal style injection = net negative) and C10 (graph gain =
modality-selective), this forms a clean triple result:

> "Graph rerank improves figure (+10.3pp) and table (+8.3pp) retrieval on M4query.
> On formula retrieval, all strategies — graph topology, edge construction, cross-modal
> injection, LaTeX surface normalization — fail to break the dense encoder ceiling of
> R@10 = 0.56. We attribute this to the text encoder's lack of mathematical pretraining,
> and recommend math-aware encoders for formula-heavy retrieval tasks."

# Status

- 2026-05-10 morning: Created from B1 Phase 2 + smoke50 6-config evidence
- 2026-05-10 midday: F-caption experiment → 8 configs, C8 parallel confirmed
- 2026-05-10 afternoon: LaTeX normalization → 10 configs, surface-form hypothesis killed
- Next: F-formula Phase 3 — true math-aware encoder (jina-embeddings-v3 or Qwen2.5-Math)
