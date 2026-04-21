# exp:20260421_crossdoc_gold57_validation

**Date**: 2026-04-21  
**Status**: COMPLETE（build + validate + eval 全部跑通，跨法验证为"机制成立、净效应为负"）  
**Job**: `62671` on `gpu-a6000-1`（62649 因 `sentence_transformers` API 变化挂掉，已在 `build_crossdoc_gold57.py` 里改用 `model.max_seq_length = N`）。

---

## 这轮要回答的问题

1. 对 gold-57 生成 `figure / formula / table + chunk` 级跨文档边（Qwen3-Embedding-0.6B + BBL cite-boost），**跨法本身是否合理**？
2. 把 `chunk↔chunk` 边通过 `chunk_contains_element` 投影回 element 空间，能否在 `M4query_v1` qrels 下提升检索指标？
3. 两条 lane 分清楚：**生成测无 chunk，chunk 仅参与检索**（element 留在 corpus / qrels，chunk 只作为 graph edge 的载体）。

---

## 生成侧（build_crossdoc_gold57.py）

- qrels = `data/03_queries/M4query_v1/qrels.jsonl`（权威）
- corpus = `M4query_v1/corpus.jsonl`（element only — qrels 认它）
- chunk graph = `paragraph_chunks_n400_trial57_enriched.json`（**仅用于取 chunk text 和 chunk→element 映射**，chunk 本身不出现在 corpus / qrels 里）
- citation = `citation_graph.json`（BBL +0.05 boost）
- threshold=0.70 / top-K=10 / types = figure formula table **+ chunk**

### 生成统计

| 项 | 值 |
|---|---|
| 总边数 | 19619 |
| figure | 7411 |
| formula | 8864 |
| table | 232 |
| chunk | 3112 |
| cite-boosted | 1361 |
| 连接 doc pair | 881 |
| gold 中 BBL pair 覆盖 | 72 / 85 = 84.7% |
| 低于阈值未覆盖 BBL pair | 13 |

- 投影后 `chunk_mediated` element-element 边：845（`chunk_fanout`: mean=0.27 / p95=2 / max=20，fanout 非常保守，没有爆）

### 每 query top-100 内 cross-doc 邻居对数

| adjacency | 有至少 1 对 | mean | p50 | p95 |
|---|---|---|---|---|
| element-only | 472 / 473 | 100.25 | 40 | 396 |
| element + chunk-mediated | 473 / 473 | 108.36 | 48 | 408 |

→ 覆盖面上跨法正确，密度合理，**没有数据级缺陷**。

---

## 检索评估（eval_graph_topk_rerank.py, rebuilt_20260417/ranking_qwen3_4B.jsonl 上重排 top-100）

### static_plus_neighbor

| config | R@1 | R@5 | R@10 | R@100 | MRR |
|---|---|---|---|---|---|
| baseline_nograph | 0.2336 | 0.5275 | 0.6195 | 0.8636 | 0.6121 |
| explicit_only | 0.2220 | 0.5835 | **0.6892** | 0.8636 | 0.6006 |
| crossdoc_elem_only | 0.0761 | 0.3753 | 0.5550 | 0.8636 | 0.3509 |
| crossdoc_elem_only_boosted | 0.0761 | 0.3742 | 0.5550 | 0.8636 | 0.3500 |
| crossdoc_with_chunk_projection | 0.0782 | 0.3710 | 0.5539 | 0.8636 | 0.3488 |
| crossdoc_with_chunk_projection_boosted | 0.0782 | 0.3710 | 0.5539 | 0.8636 | 0.3489 |
| explicit + crossdoc(elem, w=0.5) | 0.1808 | 0.5317 | 0.6564 | 0.8636 | 0.5233 |
| explicit + crossdoc(elem+chunk, w=0.5) | 0.1860 | 0.5264 | 0.6638 | 0.8636 | 0.5295 |
| explicit + crossdoc(elem+chunk, w=0.2) | 0.2082 | 0.5655 | 0.6755 | 0.8636 | 0.5737 |

### static_prior

| config | R@1 | R@5 | R@10 | R@100 | MRR |
|---|---|---|---|---|---|
| baseline_nograph | 0.2336 | 0.5275 | 0.6195 | 0.8636 | 0.6121 |
| explicit_only | **0.2431** | 0.5751 | 0.6427 | 0.8636 | **0.6372** |
| crossdoc_elem_only | 0.1512 | 0.4577 | 0.6047 | 0.8636 | 0.4843 |
| crossdoc_with_chunk_projection | 0.1575 | 0.4588 | 0.6089 | 0.8636 | 0.4958 |
| crossdoc_with_chunk_projection_boosted | 0.1564 | 0.4588 | 0.6099 | 0.8636 | 0.4944 |
| explicit + crossdoc(elem+chunk, w=0.5) | 0.2262 | 0.5497 | 0.6290 | 0.8636 | 0.6094 |
| explicit + crossdoc(elem+chunk, w=0.2) | 0.2347 | 0.5645 | 0.6332 | 0.8636 | 0.6228 |
| explicit + crossdoc(elem, w=0.5) | 0.2326 | 0.5507 | 0.6311 | 0.8636 | 0.6151 |

---

## 结论

1. **跨法正确**：gold-57 内 85 对 BBL cite-pair 覆盖 84.7%（13 对因阈值 0.7 过滤），doc-pair 不爆炸、chunk-fanout 保守、query-level 覆盖充分。
2. **单独用 crossdoc 作 graph source 会掉分**（R@1 从 0.2336 → 0.0761）。这是 neighbor propagation 把大量不相关 cross-doc pid 拉上来的结果，属于**信号太广 + 权重太大**。
3. **chunk-mediated 投影相较 element-only 的增量很小**（static_prior 下 R@1 `0.1512 → 0.1575`；neighbor 下几乎平）。在当前 chunk graph 稀疏 + enrich 覆盖仅 47.2% 情况下，chunk 投影没有打开新的有效 pid。
4. **和 explicit 混合时 crossdoc 仍是净负**：`explicit_only R@10=0.6892 / MRR=0.6006`（neighbor）、`0.6427 / 0.6372`（static）一律高于任何加了 crossdoc 的 combo。权重 `w=0.2 < 0.5` 一致更好 → 方向是**越少越好**。
5. **cite-boost 在当前量纲下无感**：`_boosted` vs 非 boosted 差异 < 0.002。1361/19619 cite-boost 边占比 6.9%，且 boost 只加 0.05，被淹没。

---

## 下一步

- 不要继续加权搜索：explicit_only 在 M4query_v1 上已是本地 SOTA（`R@10=0.6892`）。
- 若要救 crossdoc：缩窄到**高置信子集**（top-3 per pid + sim>0.85 + cite-only）再试，而不是全量 19619 边。
- chunk-mediated 投影看起来**机制合理但价值有限**；在 enrich 覆盖被修复前不再优化。
- `sentence_transformers` 新 API 的坑已在 `build_crossdoc_gold57.py` 里用 `model.max_seq_length = N` 绕过，其他脚本如再出现 `encode(..., max_length=)` 需要同步修。

---

## 关联产物

- `scripts/build_crossdoc_gold57.py`（gold-57 crossdoc 边构建器；支持 `--include-chunks` / `--citation-only`）
- `scripts/validate_and_project_crossdoc.py`（doc-pair / BBL / top-100 coverage 审查 + chunk→element 投影）
- `slurm_scripts/40_crossdoc_gold57_eval.sh`（end-to-end：build → validate → 9 个 rerank config）
- `data/01_graphs/crossdoc_gold57.json`（19619 边）
- `data/01_graphs/crossdoc_gold57_report.json`
- `data/01_graphs/crossdoc_gold57_projected.json`（figure + formula + table + chunk_mediated 共 17352 element-level 边）
- `data/05_eval/dense_retrieval/crossdoc_gold57_eval/`（9 个 rerank output dirs）
