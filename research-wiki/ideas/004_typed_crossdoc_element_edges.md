```markdown
---
type: idea
node_id: idea:004
title: "Typed cross-document element edges (figure/formula/table) with citation boost"
stage: validated
outcome: r10_uplift_only
dominance: supporting
created_at: 2026-04-19T00:00:00Z
updated_at: 2026-04-19T00:00:00Z
---

# One-line thesis

Build cross-document virtual edges at element granularity for figure, formula, and table nodes using Qwen3-Embedding-0.6B, filter by similarity threshold plus top-K, and add a small citation-confidence boost for `(doc_a, doc_b)` pairs present in the bbl graph.

## Problem / Gap

Section-level cross-document summary edges are too coarse: in `exp:20260419_typed_crossdoc`, section cross_doc alone gives R@1 0.1575 while typed figure-only gives 0.2178. Multimodal scientific retrieval needs element-level cross-doc adjacency.

## Mechanism

1. For each element of type in {figure, formula, table}, embed the element body with Qwen3-Embedding-0.6B.
2. Compute cross-document cosine similarity, keep `sim ≥ 0.70`, top-10 per node.
3. If `(doc_a, doc_b)` is in `citation_graph.json` bbl edges, add +0.05 boost.
4. Use as a `typed_crossdoc` graph source in `eval_graph_topk_rerank.py` with `--typed-crossdoc-weight` and `--prior-mode weighted`.

## Current Status

Validated in `exp:20260419_typed_crossdoc`. `explicit + typed (w=0.2)` under `static_plus_neighbor` reaches R@10 = 0.6406 (new project high); under `static_prior` it is approximately matched on R@1/MRR with explicit_only (0.2304 / 0.6060 vs 0.2357 / 0.6166).

## Why It Matters

- Gives a first systematic R@10 uplift attributable to graph structure on 0.6B.
- Exercises the new per-source-weight mechanism introduced in `exp:20260419_multi_source_stacking`.

## Failure Modes

- Typed weight ≥ 0.5 dilutes explicit signal on R@1.
- Citation boost is currently near-neutral because `citation_graph.json` covers only 59 docs / 123 edges; expanding bbl coverage is required before any strong claim about citation contribution.
- Stacking typed cross-doc with the fixed section cross-doc layer regresses R@1 (overlapping information).

## Connections

- Tested by: `exp:20260419_typed_crossdoc`
- Supports: `claim:C5`
- Addresses: `gap:G2`, `gap:G1`
- Requires: `exp:20260419_cross_doc_bug_fix`, `exp:20260419_multi_source_stacking` (weighting mechanism)
```
