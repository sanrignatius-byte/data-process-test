---
type: reference
node_id: ref:p1_chunk_element_edge_bug
title: "P1 Bug: chunk_contains_element 边与 eval qrels 不一致"
date: 2026-05-03
status: fix_ready
updated: 2026-05-03T07:00:00Z
---

# P1 Bug: Chunk-Element 边数据流不一致

## 根因

两个独立 chunk build 流程产生不兼容的 chunk→element 映射：

```
build_paragraph_chunks.py (v2 source)
  → paragraph_chunks_n400_v2.json
  → chunk_contains_element 边: element X ∈ chunk_Y (v2)

build_chunk_corpus.py --chunks paragraph_chunks_n400.json (v1 source)
  → chunk_corpus_n400_fair/qrels.jsonl
  → source_element_id: element X ∈ chunk_Z (v1)
```

虽然都用 `{doc_id}_chunk_{i}` 作为 chunk ID 格式，且都用 `position_idx ∈ paragraph_indices` 做 element→chunk 映射，但 **v1 和 v2 的 paragraph_indices 不同**（因为从不同的 paragraph nodes 源构建，合并结果不同）。

## 为什么之前没发现

1. `paragraph_chunks_n400.json` (v1) **根本没有 `element_ids` 和 `chunk_contains_element` 边**——它是用旧 pipeline 构建的，不注入 element
2. `build_chunk_corpus.py` 靠自己实时计算 element→chunk 映射（从 paragraph_indices + position_idx），所以 eval qrels 内部一致
3. Rerank 脚本 (`38_chunk_graph_rerank.sh`) 独立重建 chunk 图拿 `chunk_contains_element` 边，但 eval qrels 来自另一个 build → 图传播用错了映射

## 影响范围

| 受影响 | 不受影响 |
|--------|---------|
| `eval_chunk_graph_rerank.py` 的所有结果 | `eval_dense_retrieval.py` 的 dense-only 结果 |
| 依赖 v2 `element_ids` 的 graph propagation | `build_chunk_corpus.py` 自身计算的映射（内部一致） |
| `paragraph_chunks_n400_v2.json` edges + v1 eval qrels 的任何交叉使用 | element-level eval（不涉及 chunk） |

## 修复

`slurm_scripts/37f_chunk_corpus_eval_consistent.sh`：

1. 一次性从 `chunk_virtual_nodes_v2.json` 构建 chunk 图（`build_paragraph_chunks.py`，带 element injection）
2. 用同一个 chunk 图构建 eval corpus + qrels（`build_chunk_corpus.py`）
3. 用同一个 chunk 图做 graph rerank（`eval_chunk_graph_rerank.py`）
4. 内置 consistency check：qrels `source_element_id` vs graph `chunk_contains_element` 边一致性 ≥ 99%。**<99% 则直接 exit 1（硬 gate）**，防止在 inconsistent 数据上跑 rerank。

## 验证

提交 `37f_chunk_corpus_eval_consistent.sh` 后，检查 consistency check 输出。若 ≥ 99%，P1 修复确认。旧 fair eval 数值可能会因 chunk rebuild 轻微偏移，但方向正确。

## Related

- [exp:20260503_chunk_query_element_recall](../experiments/20260503_chunk_query_element_recall.md) — 发现此 bug 的实验
- [ref:data_architecture](data_architecture.md) — 数据架构全景
