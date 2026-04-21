# exp:20260421_trial57_fairness_repair

**Date**: 2026-04-21  
**Status**: PARTIAL COMPLETE（fair graph-only + partial-overlay exploratory 已完成；fair enriched 仍被 company API auth / budget 阻塞）  
**Motivation**: 修正 chunk 检索线在旧试验田上的公平性问题，特别是 selective enrich、lane 混用、以及 chunk graph 结构不闭环的问题。

---

## 这轮修复真正完成了什么

1. **试验田与生产线彻底分离**
   - `57 gold docs` 只用于闭集 retrieval-eval claim
   - `1040 production docs` 继续承担 query / evidence 生产
   - 不再允许用 production partial enrich 为 `57-doc` retrieval 结论背书

2. **fair / partial 两条 lane 被严格拆开**
   - `graph-only fair lane`：完整 graph elements + raw visible text
   - `enriched fair lane`：完整 graph elements + full trial57 enriched overlay
   - `partial-overlay exploratory lane`：允许只用当前旧 overlay，但不能作为正式 fair claim

3. **公平性缺口被量化并加了 guard**
   - enrichable elements：`2000`
   - already enriched：`1021`
   - overall coverage：`51.0%`
   - gold qrel 中可 enrich 元素覆盖：`232 / 242 = 95.9%`
   - 仍缺 `979` 个 enrichable elements，其中 `formula = 928`
   - `build_chunk_corpus.py` 现已支持 `--min-enriched-coverage`，旧 partial overlay 不再能伪装成 fair enriched eval

4. **工具链已经补齐**
   - `scripts/build_trial57_enrich_subset.py`
   - `scripts/merge_enriched_overlays.py`
   - `scripts/eval_bm25_retrieval.py`
   - `slurm_scripts/37b_trial57_backfill_enrich.sh`
   - `slurm_scripts/37bb_trial57_rebuild_chunk_graphs.sh`
   - `slurm_scripts/37c_chunk_corpus_eval_enriched_fair.sh`
   - `slurm_scripts/37d_chunk_corpus_eval_partial_overlay.sh`
   - `slurm_scripts/37e_chunk_bm25_eval_newchunk.sh`
   - `slurm_scripts/38b_chunk_graph_rerank_enriched_fair.sh`
   - `slurm_scripts/38c_chunk_graph_rerank_partial_overlay.sh`

---

## Job 最终状态

### graph-only fair rerun

- `62476` — `37_chunk_corpus_eval.sh` — `COMPLETED`
- `62477` — `38_chunk_graph_rerank.sh` — `COMPLETED`

### trial57 enriched fair wave

- `62482` — `37b_trial57_backfill_enrich.sh` — `COMPLETED`，但 backfill 业务失败
- `62494` — `37bb_trial57_rebuild_chunk_graphs.sh` — `COMPLETED`
- `62495` — `37c_chunk_corpus_eval_enriched_fair.sh` — `FAILED`
- `62496` — `38b_chunk_graph_rerank_enriched_fair.sh` — `CANCELLED`

### partial-overlay exploratory wave

- `62553` — `37d_chunk_corpus_eval_partial_overlay.sh` — `COMPLETED`
- `62554` — `38c_chunk_graph_rerank_partial_overlay.sh` — `COMPLETED`

---

## fair enriched 为什么没跑起来

1. `62482` 的 company API 调用确实经过了：

```text
enrich_elements_modora.py
  -> src.api.call_llm(provider="company")
  -> local_api_logger
  -> api_logs
```

2. `.env` 中当前 endpoint 为：
   - `COMPANY_API_URL=https://az.gptplus5.com/v1/chat/completions`

3. 实际运行返回持续 `401 Unauthorized`
   - backfill 实际新增 enrich 数量为 `0`
   - post-merge 目标覆盖率仍只有 `47.2%`

4. 因此 `62495` 被 `--min-enriched-coverage 0.95` 正常拦下  
   这不是坏事，说明 fair guard 已经生效。

---

## 结果

### graph-only fair（新 chunk / dense）

| variant | R@1 | R@5 | R@10 | R@100 | MRR |
|---------|-----|-----|------|-------|-----|
| chunk_n400_fair dense | 0.0902 | 0.2587 | 0.3348 | 0.6902 | 0.2587 |
| chunk_n500_fair dense | 0.0967 | 0.2609 | 0.3609 | 0.7130 | 0.2855 |

### graph-only fair（新 chunk / BM25）

| variant | R@1 | R@5 | R@10 | R@100 | MRR |
|---------|-----|-----|------|-------|-----|
| chunk_n400_fair bm25 | 0.0837 | 0.2554 | 0.3359 | 0.5891 | 0.2574 |
| chunk_n500_fair bm25 | 0.0826 | 0.2696 | 0.3652 | 0.6065 | 0.2696 |

### partial-overlay exploratory（新 chunk / dense）

| variant | R@1 | R@5 | R@10 | R@100 | MRR |
|---------|-----|-----|------|-------|-----|
| chunk_n400_partial dense | 0.1500 | 0.3837 | 0.5109 | 0.8033 | 0.3975 |
| chunk_n500_partial dense | 0.1587 | 0.4109 | 0.5304 | 0.8239 | 0.4196 |

### partial-overlay exploratory（新 chunk / BM25）

| variant | R@1 | R@5 | R@10 | R@100 | MRR |
|---------|-----|-----|------|-------|-----|
| chunk_n400_partial bm25 | 0.1924 | 0.4565 | 0.5522 | 0.8185 | 0.4831 |
| chunk_n500_partial bm25 | 0.2054 | 0.4663 | 0.5870 | 0.8283 | 0.5054 |

### partial-overlay rerank（n400）

| config | R@1 | R@10 | MRR |
|--------|-----|------|-----|
| dense_baseline | 0.1500 | 0.5109 | 0.3975 |
| exp_only / static_prior | 0.1935 | 0.6533 | 0.4871 |
| exp_only / static+neighbor | 0.2011 | 0.6489 | 0.4992 |
| seq+exp / static_prior | 0.2109 | 0.6598 | 0.5168 |
| seq+exp / static+neighbor | 0.2261 | 0.6391 | 0.5297 |

---

## 当前结论

1. `57 gold docs` 与 `1040 production` 的 lane 分离已经完成，partial enrich 不会再污染 fair 结论。
2. fair graph-only 口径下，`new chunk dense` 与 `new chunk bm25` 非常接近，且都明显弱于 element baseline。
3. partial overlay 对 chunk retrieval 有明确正向提升；`n500` consistently 优于 `n400`。
4. partial overlay 下，`BM25` 当前强于 dense，说明 enriched overlay 注入后 lexical 线索被明显放大。
5. 图 rerank 在 partial-overlay n400 上有效，最佳 `R@10=0.6598` 已超过 element baseline `0.5994`；但最佳 `R@1=0.2261` 仍略低于 element baseline `0.2389`。
6. fair enriched lane 仍未完成，阻塞点不是构图，而是 company API key / URL 当前不可用。

---

## 合规要求

本轮新增的 backfill lane 使用 company API 时，必须保留：

```text
enrich_elements_modora.py
  -> src.api.call_llm(provider="company")
  -> local_api_logger
  -> api_logs
```

项目内 `token_logger` 可作为补充审计，但不能替代 `api_logs`。
