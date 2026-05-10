---
type: experiment
node_id: exp:20260503_chunk_query_element_recall
title: "Per-query: real elements actually recalled via top-K chunks (mentor 录音60)"
date: 2026-05-03
status: completed
verdict: chunk_dilutes_signal_confirmed
related_claims: [claim:C2, claim:C7]
related_experiments: [exp:20260502_chunk_element_coverage]
---

# 目的

直接回答 mentor 录音 60 的问题：
> "你实际把你 recalled 的唱客的 recall at 一去算个平均值算一下，实际 recall 几个 element... recall at 十呢？"

## 与已有分析的差异

| 脚本 | 算什么 | 用 mentor 视角是不是答了问题 |
|------|--------|----------------------------|
| `analyze_chunk_element_coverage.py` | corpus 全局：964 chunks × 1.94 elem/chunk | ❌ 不是 mentor 要的 per-query |
| `compute_chunk_element_stats.py` | 收 `--qrels` 但没用，仍是全局 | ❌ |
| **`analyze_chunk_query_element_recall.py`** (本次新增) | per-query: top-K 真的恢复几个 element | ✅ |

## 方法

数据源（authoritative chunk→element 映射）：用 eval-time `chunk_corpus_*/qrels.jsonl` 里的 `(passage_id=chunk_id, source_element_id)` 字段。

> ⚠️ **Side finding**：`paragraph_chunks_n400_v2.json` 的 `chunk_contains_element` 边和 eval-time qrels 的 chunk_id 对全部 57 个 M4query_v1 文档**0% 一致**。两套独立 build 流程、两套 chunk-id 命名空间。详见下文"副产品发现"。

每 query 计算：
```
top_chunks_K     = ranking[qid][:K]
hit_elements_K   = { eid | (cid, eid) ∈ relevant_pairs(qid) AND cid ∈ top_chunks_K }
elem_recall@K    = |hit_elements_K| / |distinct relevant elements|
chunk_hit_rate@K = (top_chunks_K ∩ relevant_chunks ≠ ∅)  # 与 wiki 报的 chunk R@K 对齐
```

## 结果

460 queries（M4query_v1 有 qrels 的子集），平均每 query 1.78 个相关 element。

### 跨 lane 对比（K=10）

| Lane | recalled@10 | elem R@10 | chunk hit@10 | zero@10 | full@10 |
|------|------------:|----------:|-------------:|--------:|--------:|
| n400 fair | 0.60 / 1.78 | **0.335** | 0.463 | 247 (54%) | 95 (21%) |
| n400 partial-overlay | 0.91 / 1.78 | **0.511** | 0.665 | 154 (33%) | 164 (36%) |
| n500 fair | 0.65 / 1.77 | **0.361** | 0.498 | 231 (50%) | 103 (22%) |
| n500 partial-overlay | 0.95 / 1.77 | **0.530** | 0.678 | 148 (32%) | 176 (38%) |

### n500 partial-overlay 按 K 展开（最优 lane）

| K | recalled (avg) | elem R@K | chunk hit | full% | zero% |
|---|--------------:|---------:|----------:|------:|------:|
| 1  | 0.29 | 0.159 | 0.289 | 3% | 71% |
| 2  | 0.43 | 0.239 | 0.402 | 8% | 60% |
| 5  | 0.74 | 0.411 | 0.585 | 24% | 42% |
| 10 | 0.95 | 0.530 | 0.678 | 38% | 32% |
| 15 | 1.04 | 0.579 | 0.726 | 43% | 27% |
| 20 | 1.12 | 0.627 | 0.761 | 49% | 24% |

## 结论

1. **chunk R@10 (0.68) vs element R@10 (0.53) 有 ~15pp 鸿沟**：相当一部分"chunk 命中"只命中了 2 个证据中的 1 个，另一个证据被埋在没被检索到的 chunk 里。坐实 mentor 录音 60 的怀疑：**chunk 单元在双证据 query 上稀释信号**。
2. **K=1 时 71% 的 query zero recall**：1.77 个相关 element 平均只 recall 出 0.29 个。R@1 低不仅是 chunk 排序问题，更是 chunk 单元结构问题。
3. **enrich overlay 给 +17pp**（fair → partial），证明注入 element 描述到 chunk passage 是对的方向，但天花板停在 53%。
4. **n500 略优于 n400（+2pp）**，差距小，不构成路线选择。
5. **想把 element R@10 推到 >70%**，要么修 chunk-element 边构建（消除 33% zero），要么落 mentor 提议的分离式检索（图/表/公式独立被找到）。

## 副产品发现：chunk-element 边数据流不一致

`paragraph_chunks_n400_v2.json` vs `chunk_corpus_*/qrels.jsonl` 的 chunk_id 完全是两套：
- v2 文件：`1104.3913_formula_2 ∈ chunk_1`（用 paragraph_indices 切）
- eval qrels：`1104.3913_formula_2 ∈ chunk_9`（用 build_graph_augmented_corpus 切）
- 全 57 个 M4query_v1 文档的命中率对比：**0/4+ qrels rows hit per doc**

依赖 v2 文件 `chunk_contains_element` 边的代码（rerank、graph propagation）可能在错前提上工作。优先级 P1。

## 文件

- 脚本：[scripts/analyze_chunk_query_element_recall.py](../../scripts/analyze_chunk_query_element_recall.py)
- 跨 lane 总表：[data/05_eval/chunk_query_element_recall_summary.md](../../data/05_eval/chunk_query_element_recall_summary.md)
- 各 lane JSON：`data/05_eval/chunk_corpus_n*/chunk_query_element_recall.json`

## Related

- [exp:20260502_chunk_element_coverage] — corpus 全局统计（被本实验取代）
- [exp:20260421_chunk_as_retrieval_unit] — chunk 作检索单元 fairness 主线
- [claim:C2] — intra-doc virtual edges dilute precision（机制类似）
