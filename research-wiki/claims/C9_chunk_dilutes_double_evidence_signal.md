---
type: claim
node_id: claim:C9
title: "Chunk-as-retrieval-unit dilutes signal on double-evidence queries"
status: supported
date: 2026-05-10
related_experiments: [exp:20260503_chunk_query_element_recall, exp:20260502_chunk_element_coverage, exp:20260421_chunk_as_retrieval_unit]
related_claims: [C2, C7, C10]
---

# Statement

Using chunks (n=400 / n=500 word merges) as the retrieval unit dilutes the signal on M4query_v1 double-evidence queries. Compared with element-level retrieval, chunk-level retrieval shows:

- **15pp gap** at R@10 on best lane (n500 partial-overlay): elem R@10 = 0.530 vs chunk R@10 = 0.678
- **K=1 zero-recall rate = 71%** — for 71% of queries, the top-1 chunk contains zero gold elements
- 75% of queries have evidence spanning **2 different chunks**; only 2% have both evidence in the same chunk
- enrich overlay adds +17pp R@10 but **caps at 0.530** — chunk substitution is the upper bound, not the floor

# Mechanism

M4query_v1 queries require **dual evidence** — two elements (e.g. figure+table) jointly support the answer. When chunks are the retrieval unit:
1. Each chunk averages 1.94 elements (chunk:element ≈ 1:2). Most chunks contain only one of the two evidence elements.
2. R@1 on chunks therefore caps at ~50% of element coverage — to recall both evidence pieces, the system needs at least 2 chunks (R@2+).
3. Top-1 chunk is forced to choose between 2 evidence locations → 71% lose at least one evidence element at K=1.

# Scope of evidence

- M4query_v1 (473 queries / 946 dual-evidence qrels)
- Source datasets: `chunk_corpus_n400_partial_overlay`, `chunk_corpus_n500_partial_overlay`, fair / partial overlay
- Mentor 录音 60 (5/2) explicitly asked: "recall 出来的 chunk 里平均含几个 element"

# Implication for paper claims

- **Do not claim chunks improve M4query retrieval**. Stay with element-level retrieval as the primary mode for dual-evidence query benchmarks.
- Chunks remain useful as **structural scaffold** in the graph (chunk-paragraph-element hierarchy), but not as primary retrieval units.
- Rebuilding with the upcoming chunk-element edge fix (B1 Phase 2 with LaTeX line_no, replacing position_idx fuzzy) **may** narrow the gap, but is unlikely to close it on dual-evidence queries — that's a structural property of dual-evidence query design, not a chunk-quality issue.

# Mentor 录音 60 status

| Mentor todo | Before | After this claim |
|---|---|---|
| **C2** "重新审视 chunk 是不是噪声" | ⚠️ 已坐实但未升 claim | ✅ claim:C9 |

# Open questions

1. Does claim hold on single-evidence queries? Untested — M4query_v1 is 100% dual-evidence by design.
2. Would chunk + cross-chunk graph traversal (multi-hop walk) close the 15pp gap? Partially tested by graph rerank — even with full graph, chunk corpus underperforms.
3. Is there a query class where chunk > element? Probably yes for "concept overview" queries that require coherent paragraph context, but not present in M4query_v1.

# Evidence files

- [exp:20260503_chunk_query_element_recall](../experiments/20260503_chunk_query_element_recall.md) — per-query analysis
- `data/05_eval/chunk_query_element_recall_summary.md` — cross-lane summary table
- [exp:20260502_chunk_element_coverage](../experiments/20260502_chunk_element_coverage.md) — chunk:element coverage stats

# P1 bug related

[ref:p1_chunk_element_edge_bug](../reference/p1_chunk_element_edge_bug.md): `paragraph_chunks_n400_v2.json` chunk_id 与 eval qrels chunk_id 0% 一致 — 这个 bug 不改变 C9 的结论（C9 来自 eval qrels 的实测），但修了之后 chunk graph rerank 才能可信地评估。
