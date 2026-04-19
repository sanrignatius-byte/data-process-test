```markdown
---
type: experiment
node_id: exp:20260419_typed_crossdoc
status: completed
verdict: r10_new_high_0.6406_at_small_typed_weight
created_at: 2026-04-19T00:00:00Z
updated_at: 2026-04-19T00:00:00Z
---

# One-line summary

Typed cross-document element edges (figure / formula / table via Qwen3-Embedding-0.6B + bbl citation boost), combined with explicit edges at weight 0.2 under `static_plus_neighbor`, push R@10 to **0.6406**, the highest R@10 recorded for Qwen3-Embedding-0.6B on the `v1_enriched` corpus; the fixed section-level cross_doc layer ties the same number.

## Construction

Script: `scripts/build_typed_crossdoc_edges.py`
- Embeds figure / formula / table element bodies with Qwen3-Embedding-0.6B.
- Keeps pairs with cosine ≥ 0.70, top-10 per node, cross-document only.
- Citation boost: +0.05 if the `(doc_a, doc_b)` pair appears in `citation_graph.json` bbl edges.

Output: `data/01_graphs/typed_crossdoc_edges.json`
- 16520 edges: figure 7411, formula 8877, table 232
- 744 doc-pairs, 1616 edges with cite boost
- 1922 / 2809 = 68.4% pid coverage

## Eval setup

- Model: Qwen3-Embedding-0.6B
- Corpus: `v1_enriched` (2809 passages), 473 queries
- Script: `scripts/eval_graph_topk_rerank.py`
- New CLI used: `--graph-sources typed_crossdoc`, `--typed-crossdoc-weight`, `--typed-crossdoc-use-boost`, `--prior-mode weighted`
- Output: `data/05_eval/dense_retrieval/typed_crossdoc/`

## Results — static_prior

| Config | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| explicit_only (ref) | 0.2357 | 0.5507 | 0.6258 | 0.6166 |
| crossdoc_sec_only (fixed) | 0.1575 | 0.4820 | 0.6047 | 0.4945 |
| typed_only_figure | 0.2178 | 0.5085 | 0.5983 | 0.5849 |
| typed_only_formula | 0.1755 | 0.4471 | 0.5782 | 0.5019 |
| typed_only_table | 0.2125 | 0.5085 | 0.6036 | 0.5767 |
| typed_only_all | 0.1607 | 0.4440 | 0.5909 | 0.4911 |
| **explicit + typed (w=0.2)** | 0.2304 | 0.5381 | 0.6195 | **0.6060** |
| explicit + typed (w=0.5) | 0.2283 | 0.5254 | 0.6057 | 0.5952 |
| explicit + typed (w=1.0) | 0.2167 | 0.5148 | 0.6025 | 0.5784 |
| explicit + typed (boosted, w=1.0) | 0.2167 | 0.5148 | 0.6025 | 0.5784 |
| explicit + sec_crossdoc + typed | 0.2008 | 0.5222 | 0.6131 | 0.5612 |

## Results — static_plus_neighbor

| Config | R@1 | R@5 | **R@10** | MRR |
|---|---|---|---|---|
| explicit_only (ref) | 0.2357 | 0.5507 | 0.6258 | 0.6166 |
| **explicit + typed (w=0.2)** | 0.1818 | 0.5423 | **0.6406** | 0.5413 |
| **explicit + crossdoc_sec (fixed)** | 0.1734 | 0.5423 | **0.6406** | 0.5273 |

## Observations

1. R@10 new high 0.6406 (+1.5pp over explicit_only 0.6258).
2. Typed element edges in isolation are stronger than section cross_doc edges (figure R@1 0.2178 vs section 0.1575) — element-level granularity dominates section-level on 0.6B.
3. Typed weight is narrow: 0.2 is best; ≥ 0.5 dilutes the explicit signal (R@1 drops).
4. Citation boost currently barely matters: only 1616 / 16520 = 10% of edges carry it, and +0.05 is small relative to similarity. `citation_graph.json` covers just 59 docs / 123 edges — bbl extraction coverage is the next bottleneck.
5. Stacking fixed section cross_doc + typed cross_doc simultaneously regresses: R@1 0.2008. The two cross-doc layers carry overlapping information and the second one subtracts margin.
6. R@10 gains under `static_plus_neighbor` cost R@1 / MRR; the two methods optimize different axes.

## Connections

- Validates `claim:C3` in revised form (cross-doc edges help R@10, not R@1/MRR).
- Originates: `claim:C5` (new — typed element edges beat section edges at the same method).
- Addresses: `gap:G1`, `gap:G2`.
- Depends on: `exp:20260419_cross_doc_bug_fix`, mechanism from `exp:20260419_multi_source_stacking`.
```
