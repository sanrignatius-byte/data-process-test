# exp:20260421_chunk_as_retrieval_unit

**Date**: 2026-04-21  
**Status**: PARTIAL COMPLETE（graph-only fair / partial-overlay / BM25 已完成；fair enriched 仍阻塞）  
**Motivation**: 4.19.md §5.3 指向的核心问题是评估设计错配，chunk 作检索单元只有在 qrels、passage 组成、图投影都对齐后才有资格被判断。

---

## 背景与修正目标

早期 paragraph-merge 线失败，不是因为 chunk 这件事天然无效，而是因为当时同时犯了两类错误：

1. **corpus 稀释**
   - element passage + chunk passage 混在一起
   - gold 在 corpus 里的密度被显著冲淡

2. **qrels 失配**
   - qrels 只有 `element_id`
   - chunk 命中后拿不到任何 credit

当前路线的修正是：
- 用 `chunk-only corpus`
- 用 `element_id -> parent_chunk_id` 做 qrels 重映射
- 把 element 语义显式注入 chunk passage
- 把 chunk 当成正式图节点，而不是临时拼出来的 text blob

---

## 当前构造口径

### chunk text

- `graph-only fair`：paragraph text + element raw visible text
- `partial overlay`：paragraph text + element raw visible text + enriched overlay

当前 rebuilt trial57 口径下：
- n400 corpus：`1963` passages
- n500 corpus：`1703` passages
- qrels 覆盖：`460 / 473` queries

### chunk 图边

当前 chunk 图要求至少包含：
- `chunk_contains_paragraph`
- `chunk_contains_element`
- `section_contains_chunk`
- `chunk_sequence`

图 rerank 用的新脚本是 `scripts/eval_chunk_graph_rerank.py`，显式区分：
- `chunk_sequence`
- `explicit_projected`（element-pair 投影到 chunk 空间）

---

## 迭代历史

### v2（element-last + max_length 512，失败）

| variant | R@1 | R@10 | R@100 | MRR |
|---------|-----|------|-------|-----|
| v1_enriched baseline | 0.2389 | 0.5994 | 0.8362 | 0.6081 |
| chunk_n400 v2 | 0.0285 | 0.2199 | 0.5603 | 0.1344 |
| chunk_n500 v2 | 0.0254 | 0.2051 | 0.5518 | 0.1275 |

根因是 truncation：element injection 被放在 passage 末尾，query 对应的 element 语义大量落在模型截断窗外。

### v3（element-first + max_length 1024）

这一轮证明 truncation 确实是大问题，但还不足以说明 chunk 已经优于 element。

---

## 最终结果

### element baseline 参考

| variant | R@1 | R@5 | R@10 | R@100 | MRR |
|---------|-----|-----|------|-------|-----|
| v1_enriched baseline | 0.2389 | 0.5127 | 0.5994 | 0.8362 | 0.6081 |

### graph-only fair（新 chunk）

| variant | R@1 | R@5 | R@10 | R@100 | MRR |
|---------|-----|-----|------|-------|-----|
| chunk_n400_fair dense | 0.0902 | 0.2587 | 0.3348 | 0.6902 | 0.2587 |
| chunk_n500_fair dense | 0.0967 | 0.2609 | 0.3609 | 0.7130 | 0.2855 |
| chunk_n400_fair bm25 | 0.0837 | 0.2554 | 0.3359 | 0.5891 | 0.2574 |
| chunk_n500_fair bm25 | 0.0826 | 0.2696 | 0.3652 | 0.6065 | 0.2696 |

**解释**：
- fair 口径下，`dense` 和 `BM25` 很接近
- `n500` consistently 略优于 `n400`
- 但 chunk-only 仍显著弱于 element baseline

### partial-overlay exploratory（新 chunk）

| variant | R@1 | R@5 | R@10 | R@100 | MRR |
|---------|-----|-----|------|-------|-----|
| chunk_n400_partial dense | 0.1500 | 0.3837 | 0.5109 | 0.8033 | 0.3975 |
| chunk_n500_partial dense | 0.1587 | 0.4109 | 0.5304 | 0.8239 | 0.4196 |
| chunk_n400_partial bm25 | 0.1924 | 0.4565 | 0.5522 | 0.8185 | 0.4831 |
| chunk_n500_partial bm25 | 0.2054 | 0.4663 | 0.5870 | 0.8283 | 0.5054 |

**解释**：
- partial overlay 对 chunk retrieval 有明确帮助
- `BM25` 在 partial overlay 下当前强于 dense
- `n500` 仍比 `n400` 更稳

### partial-overlay rerank（n400）

| config | R@1 | R@10 | MRR |
|--------|-----|------|-----|
| dense_baseline | 0.1500 | 0.5109 | 0.3975 |
| seq_only / static_prior | 0.1489 | 0.5011 | 0.3966 |
| seq_only / static+neighbor | 0.1250 | 0.4511 | 0.3474 |
| exp_only / static_prior | 0.1935 | 0.6533 | 0.4871 |
| exp_only / static+neighbor | 0.2011 | 0.6489 | 0.4992 |
| seq+exp / static_prior | 0.2109 | 0.6598 | 0.5168 |
| seq+exp / static+neighbor | 0.2261 | 0.6391 | 0.5297 |

**解释**：
- `chunk_sequence` 单独用时仍偏负作用
- `explicit_projected` 是有效信号
- `seq+exp / static_prior` 的 `R@10=0.6598` 已超过 element baseline `0.5994`
- 但最佳 `R@1=0.2261` 仍未超过 element baseline `0.2389`

---

## 当前判断

1. “`chunk-only retrieval` 已经优于 `element-level retrieval`” 这条主张当前仍然不成立。
2. chunk 的合理定位更像是：
   - 检索与下游消费的工作单元
   - 图信号的投影宿主
   - QA / evidence packaging 的载体
3. partial overlay 显示出方向性价值，但它不是 fair enriched 结论。
4. 真正有前景的故事不是“chunk 单挑赢 element”，而是：
   - `new chunk graph + overlay + explicit rerank`
   - 在保持公平性的前提下，能否把 top-k evidence 拉得更靠前

---

## 关联

- 修复问题：4.19.md §5.1、§5.2、§5.3
- 新脚本：`scripts/build_chunk_corpus.py`、`scripts/eval_chunk_graph_rerank.py`、`scripts/eval_bm25_retrieval.py`
- 图结构脚本：`scripts/build_paragraph_chunks.py`
