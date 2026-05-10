# F-formula Phase 2 — LaTeX Normalization Decision

**Job**: 68131 | **Date**: 2026-05-10 | **Wall**: ~4 min GPU

## Hypothesis

Raw-LaTeX formula passages (e.g., `\operatorname`, `\mathbb`, `\stackrel`) are tokenized
as nonsense subwords by Qwen3-Embedding's BPE tokenizer. Normalizing LaTeX markup to
readable text before encoding should rescue formula retrieval.

## Method

`build_math_normalized_corpus.py` — LaTeX symbol normalization pass:

- `\operatorname{opt}` → `opt`
- `\mathbb{E}` → `E`
- `\leq` → `<=`
- `\frac{a}{b}` → `(a) / (b)`
- `\sum_{i=1}^{n} X_i` → `sum over i=1 to n of (X_i)`
- Greek letters: `\alpha` → `alpha`, etc.
- Named operators: `\min`, `\max`, `\log`, `\exp` preserved

1253 formula passages normalized in corpus. 1556 non-formula passages unchanged.

## Results

### M4query_v1 full (473 queries)

| Config | R@10 | ΔR@10 | MRR |
|--------|------|-------|-----|
| dense baseline (raw LaTeX) | 0.6195 | — | 0.6121 |
| dense math_norm | 0.6237 | +0.4pp | 0.6174 |
| graph baseline (ceiling) | 0.6913 | — | 0.6017 |
| graph math_norm | 0.6977 | +0.6pp | 0.5931 |

### Smoke50 per-modality R@10

| Config | figure (39) | formula (25) | table (36) | all (100) |
|--------|-------------|-------------|-----------|----------|
| dense baseline | 0.7179 | **0.5600** | 0.6111 | 0.6400 |
| dense math_norm | 0.6923 | **0.5600** | 0.6111 | 0.6300 |
| graph baseline | 0.8205 | **0.5600** | 0.6944 | 0.7100 |
| graph math_norm | 0.8205 | **0.5600** | 0.7222 | 0.7200 |

## Verdict

**HD: FAIL.** LaTeX normalization has exactly zero effect on formula retrieval.
Formula R@10 = 0.5600 unchanged in all configs. Figure drops slightly (-2.6pp
dense, likely noise). Table unchanged in dense, +2.8pp in graph (noise).

## Root cause

The Bottleneck is not LaTeX surface form. Even with normalized, human-readable text
(`sum over i=1 to n of (X_i)`), Qwen3-Embedding cannot represent mathematical
semantics. The encoder was pretrained on natural language, not mathematical
discourse. Tokenizer fix, surface normalization, and NL augmentation are all
treating a symptom that doesn't exist.

## Cumulative evidence on formula R@10 = 0.56 ceiling

| # | Experiment | Formula R@10 | Strategy class |
|---|-----------|-------------|----------------|
| 1 | dense baseline (0.6B) | 0.5600 | — |
| 2 | dense (4B) | 0.5600 | model scale |
| 3 | graph explicit-only | 0.5600 | graph topology |
| 4 | graph explicit+virtual | 0.5200 | more graph edges |
| 5 | F-caption injection (dense) | 0.4000 | NL augmentation |
| 6 | F-caption injection (graph) | 0.5200 | NL augmentation + graph |
| 7 | **LaTeX normalization (dense)** | **0.5600** | **surface form** |
| 8 | **LaTeX normalization (graph)** | **0.5600** | **surface form + graph** |
| 9 | Qwen3-VL-Embedding-2B | << 0.50 | multimodal encoder |
| 10 | BGE CE reranker | 0.2400 | text CE reranker |

**10 configs. All ≤ 0.5600.** Three strategy classes exhausted. One path remains.

## Next step

F-formula Phase 3: **True math-aware encoder.** Download a model pretrained on
mathematical text (arXiv LaTeX, math StackExchange, etc.):

- Option α: `jinaai/jina-embeddings-v3` (multilingual, strong on technical text)
- Option β: Fine-tuned BERT on math corpus from HuggingFace
- Option γ: Use Qwen2.5-Math-1.5B hidden states as embeddings (LLM-as-encoder)

Success bar unchanged: formula R@10 > 0.65.
