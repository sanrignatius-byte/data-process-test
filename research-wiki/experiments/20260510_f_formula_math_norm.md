---
type: experiment
node_id: exp:20260510_f_formula_math_norm
title: "F-formula Phase 2 — LaTeX surface normalization test"
date: 2026-05-10
status: completed
verdict: HD_FAIL
job_id: 68131
related_claims: [C11]
---

# Goal

Test hypothesis: raw LaTeX `\operatorname`, `\mathbb`, `\stackrel` are tokenized as
nonsense subwords by Qwen3-Embedding. Normalizing to readable text should rescue formula
retrieval.

# Method

`build_math_normalized_corpus.py` — 1253 formula passages normalized:
- Font commands stripped: `\operatorname{opt}` → `opt`
- Greek letters expanded: `\alpha` → `alpha`
- Relations normalized: `\leq` → `<=`
- Structural transforms: `\frac{a}{b}` → `(a)/(b)`, `\sum_{i}` → `sum over i`
- Operators preserved: `\min`, `\max`, `\log`

# Results

| Config | figure R@10 | formula R@10 | table R@10 | overall R@10 |
|--------|------------|-------------|-----------|-------------|
| dense baseline | 0.7179 | 0.5600 | 0.6111 | 0.6400 |
| dense math_norm | 0.6923 | **0.5600** | 0.6111 | 0.6300 |
| graph baseline | 0.8205 | 0.5600 | 0.6944 | 0.7100 |
| graph math_norm | 0.8205 | **0.5600** | 0.7222 | 0.7200 |

# Verdict

**HD: FAIL.** Formula R@10 unchanged at 0.5600. Surface form is NOT the bottleneck.
The encoder simply cannot represent mathematical semantics, regardless of tokenization.

# Outputs
- Script: `scripts/build_math_normalized_corpus.py`
- Corpus: `data/05_eval/dense_retrieval/rebuilt_20260417/augmented/corpus_v1_math_norm.jsonl`
- Rankings: `data/05_eval/dense_retrieval/rebuilt_20260417/graph_math_norm/`
- Slurm: `slurm_scripts/51_f_formula_math_norm.sh`
- Decision: `refine-logs/F_FORMULA_MATH_NORM_DECISION_20260510.md`
