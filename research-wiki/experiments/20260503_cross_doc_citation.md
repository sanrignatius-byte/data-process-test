---
type: experiment
node_id: exp:20260503_cross_doc_citation
title: "Cross-document element-level citation query pipeline"
date: 2026-05-03
status: planned
verdict: pending
related_experiments: [exp:20260421_crossdoc_gold57_validation]
related_claims: [C_XD1, C_XD2, C_XD3]
---

# 目的

从 LaTeX `\cite{}` 上下文中提取 element-level 跨文档引用对，生成真正需要双文档证据的 query，在跨文档 retrieval 上评估 graph rerank。

# 为什么之前的跨文档方法都失败了

| 方法 | 失败模式 | 根因 |
|------|---------|------|
| Citation walk | MRR 负 | Doc-level 粒度，跟 element-level evidence 不对齐 |
| Cross-doc summary | 无 uplift | Summary→summary 相似度太模糊 |
| Typed cross-doc | R@10 < explicit_only | 跨文档边引入噪声，缺 citation context 过滤 |

**缺的东西**：当 Doc A 说 "as shown in [Doc B]'s Figure 3..." 时，这是一个精确的跨文档 element 引用。需要提取这些引用、映射到 MinerU element ID、用作 query anchor。

# 五步 Pipeline

```
latex_reference_graph_v2.json
  → Step 1: 从 citation edge_context 提取 element 引用
  → Step 2: LaTeX label → MinerU element ID 映射
  → Step 3: 构建 (doc_A, elem_X) → (doc_B, elem_Y) candidate pairs
  → Step 4: 生成跨文档 query（用 citation context 作 bridge）
  → Step 5: 跨文档 retrieval eval
```

# 假设

- **C_XD1**: ≥ 50 对有效 element-level citation pairs
- **C_XD2**: 跨文档 query 的 LLM ablation pass rate ≥ 50%（删任一文档证据 → 不可答）
- **C_XD3**: Graph rerank 在跨文档 query 上的 delta > 在 intra-doc 上的 delta

# 实验块

- **B1 (MUST)**: Element-level citation pair 提取。输出 `cross_doc_citation_pairs.json`。≥ 50 pairs。
- **B2 (MUST)**: 跨文档 query 生成（top-50 pairs）。输出 `cross_doc_citation_queries.jsonl`。≥ 25 QC pass。
- **B3 (MUST)**: 跨文档 retrieval eval（BM25 vs dense vs graph rerank）。GPU ~20 min。

# 文件

- 计划：[refine-logs/EXPERIMENT_PLAN_CROSS_DOC_20260503.md](../../refine-logs/EXPERIMENT_PLAN_CROSS_DOC_20260503.md)
- 代码：待创建 `scripts/build_cross_doc_citation_pairs.py`, `scripts/generate_cross_doc_queries.py`
- 输出：`data/02_enriched/cross_doc_citation_pairs.json`, `data/03_queries/cross_doc_citation_queries.jsonl`
