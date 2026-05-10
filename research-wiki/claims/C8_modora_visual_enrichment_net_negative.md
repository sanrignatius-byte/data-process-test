---
type: claim
node_id: claim:C8
status: supported
created_at: 2026-05-03T16:00:00Z
updated_at: 2026-05-03T16:00:00Z
---

# Claim

Replacing or augmenting figure-passage text with **MODORA-style modality-faithful visual descriptions is net-negative** on text-style scientific QA retrieval (M4query_v1). The retrieval signal that was apparently "missing" (figure passages with `text = [Image: …]` placeholder) is in fact already carried by the build script's graph fallback (caption + context_before + context_after), and adding visual descriptions on top displaces or dilutes that paper-domain context.

## Evidence

- `exp:20260503_corpus_enrich_fix` (verdict D5):
  - **fix_v1 (visual replace)**: dense R@10 0.6195 → 0.5106 (−10.9pp), R@100 0.8636 → 0.7569 (−10.7pp)
  - **fix_v2 (visual additive: visual prefix + paper context concat)**: dense R@10 0.5888 (−3.1pp), R@100 0.8436 (−2.0pp), graph_explicit_only / static_plus_neighbor R@10 0.6860 (−0.5pp vs the 0.6913 ceiling)
  - Both variants regress; the additive variant is less bad but still net-negative
- 293 figure passages (D2 bucket) had MODORA enrichment available but were silently dropped by the build script's key-format mismatch. **Restoring them does not lift retrieval** — confirming the placeholders were not the bottleneck.
- Mechanism: MODORA descriptions are modality-faithful but **domain-detached** (e.g. "Histogram of small-valued metric"). M4query_v1 queries are paper-domain text-style ("RoBERTa pretraining objective", "PASCAL 2012 instance distribution"). The visual description does not lexically or semantically anchor to the query.
- Cross-validation with `exp:20260503_ce_rerank_bge`: BGE-reranker-v2-m3 also demoted figure/formula passages in the same direction. Two independent failure modes (corpus replace, NL reranker) converge on the same finding — **text-style benchmarks reward paper-context language over visual fidelity**.

## Scope

- Applies to text-style retrieval benchmarks where queries are written in paper-domain language (M4query_v1 is one such; most evaluator-written QA sets share this property).
- Does **not** apply to truly visual queries (e.g. "find the bar chart with three blue bars peaking at 35"). For those, vision-language late fusion is still on the table — `exp:20260502_split_modality_vl_t5_rerun` showed VL improves the figure lane in isolation (0.4112 → 0.5397).
- Does not generalize to all visual-description sources — only validated for the MODORA enrichment style (short, visually-grounded descriptions). LLM-generated paper-aware captions (different style) are an open question.

## Why It Matters

- **Falsifies the "corpus is the ceiling" hypothesis**. The 0.6913 R@10 ceiling is held by graph rerank propagation under the original corpus, not by passage text quality. Future ceiling-breaking work should avoid corpus-text replacement strategies and target either query-side rewriting (HyDE) or rerank-side signals that complement dense scores.
- **Promotes graph rerank to load-bearing status**. fix_v2 dense dropped −3.1pp R@10 but graph rerank only dropped −0.5pp — the graph absorbs corpus quality variance. This strengthens C1 (graph rerank's robustness) and reframes how dense quality interacts with graph signal.
- **Saves future LLM-cost on enrichment**: the 402 D1 figures (no MODORA coverage) need not be enriched in MODORA style, since the data shows it would not help retrieval anyway. Only differently-styled enrichment (paper-aware caption rewrite) is worth an LLM-cost pilot.

## Connections

- Supported by: `exp:20260503_corpus_enrich_fix`
- Cross-validates with: `exp:20260503_ce_rerank_bge` (same text-bias direction)
- Conflicts with: prior assumption in 2026-05-03 14:30 UTC log entry that F2 was P0 — that hypothesis is now formally falsified
- Informs: future rerank choice, query rewriting design, late-fusion architecture
- Updates `gap:G2` framing — visual edges and visual content are not the missing piece for text-style queries
