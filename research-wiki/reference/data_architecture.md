---
type: reference
node_id: ref:data_architecture
title: "DPT 数据架构：实验轨 vs 生产轨"
date: 2026-05-03
status: living
---

# DPT 数据架构

## 两轨全景

```
                     ┌─── Track A（实验轨）───┐        ┌─── Track B（生产轨）───┐
源数据               │  87 PDF / 56 LaTeX     │        │  1425 LaTeX / 1152 mineru│
                     │  data/00_raw/{raw_pdfs, │        │  data/00_raw/{latex_    │
                     │    latex_sources_delivery}│       │    sources_batch2,       │
                     │                         │        │    mineru_output}         │
                     └──────────┬──────────────┘        └──────────┬──────────────┘
                                │                                  │
图结构               ┌──────────┴──────────────┐        ┌──────────┴──────────────┐
                     │ reference_graph.json     │        │ reference_graph_v2.json  │
                     │   (6.7M, 53 docs)        │        │   (274M, 1425 docs)      │
                     │ multimodal_elements.json  │        │ multimodal_elements_v2   │
                     │   (4.7M, ~1798 elems)     │        │   .json (93M, 27209 elems)│
                     │ hub_candidates v1-v4      │        │ hub_candidates_v2_*      │
                     │ paragraph_chunks_n400.json│        │ paragraph_chunks_n400_v2 │
                     └──────────┬──────────────┘        └──────────┬──────────────┘
                                │                                  │
Enrich              ┌──────────┴──────────────┐        ┌──────────┴──────────────┐
                     │ enriched.json (全量)     │        │ production_full.json     │
                     │ hub_candidates_enriched  │        │   (88M, 6707/27209)      │
                     │   v1-v4                  │        │ gap227_enriched.json     │
                     │ trial57_backfill         │        │ hub_shortchain_subset     │
                     └──────────┬──────────────┘        └──────────┬──────────────┘
                                │                                  │
Query                ┌──────────┴──────────────┐        ┌──────────┴──────────────┐
                     │ M4query_v1/              │        │ (空 — 等 enrich 完成)    │
                     │   queries.jsonl (473条)  │        │ full_doc_v2_prod_1040/   │
                     │   qrels.jsonl (有标注)   │        │   (空目录，预留)         │
                     │   corpus.jsonl           │        │                          │
                     └──────────┬──────────────┘        └──────────────────────────┘
                                │
Eval                 ┌──────────┴──────────────────────────────────┐
                     │ data/05_eval/dense_retrieval/  (40+ 实验组)  │
                     │ data/05_eval/chunk_corpus_n*/  (6 个 chunk lane) │
                     │ data/05_eval/split_modality*/  (分离式检索)   │
                     │ data/05_eval/cpool_retrieval/  (C-Pool 78q)  │
                     └─────────────────────────────────────────────┘
```

## Track A — 实验轨（53-doc closed eval）

**用途**：检索方法论验证、消融实验、图信号分析、claims 支持/证伪

**数据流**：
```
00_raw/{raw_pdfs, latex_sources_delivery}
  → 01_graphs/{reference_graph, multimodal_elements, hub_candidates, paragraph_chunks}
  → 02_enriched/{multimodal_elements_enriched, hub_candidates_enriched_v1-v4}
  → 03_queries/M4query_v1/{queries, qrels, corpus}
  → 05_eval/{dense_retrieval, chunk_corpus_n*, split_modality}
```

**关键文件（按用途）**：

| 用途 | 文件 | 说明 |
|------|------|------|
| Reference graph | `data/01_graphs/latex_reference_graph.json` | 53-doc 引用图 |
| Elements | `data/01_graphs/multimodal_elements.json` | ~1798 elements |
| | `data/03_queries/M4query_v1/graphs/multimodal_elements.json` | M4query_v1 绑定的副本 |
| Chunks v1 | `data/01_graphs/paragraph_chunks_n400.json` | 旧 chunk build，eval 在用 |
| Chunks v2 | `data/01_graphs/paragraph_chunks_n400_v2.json` | 新 chunk build（从 chunk_virtual_nodes_v2.json） |
| Enrich | `data/02_enriched/multimodal_elements_enriched.json` | 53-doc 全量 enrich |
| Hub pairs | `data/02_enriched/hub_candidates_enriched_v4.json` | 最新 hub pair enrich |
| Queries | `data/03_queries/M4query_v1/queries.jsonl` | 473 queries |
| Qrels | `data/03_queries/M4query_v1/qrels.jsonl` | element-level qrels（权威） |
| Corpus | `data/03_queries/M4query_v1/corpus.jsonl` | element-level corpus（1798 passages） |

**Eval 目录说明**：

| 目录 | 构建方式 | 检索单元 | 状态 |
|------|---------|---------|------|
| `dense_retrieval/augmented/` | `build_graph_augmented_corpus.py` | element/chunk/summary | 基准 |
| `dense_retrieval/augmented_v2/` | 同上 + sec_context/chunk_hint | element | 扩展 |
| `dense_retrieval/rebuilt_20260417/` | 同上（修正版） | element | 47-config 消融 |
| `dense_retrieval/split_modality/` | `eval_split_modality.py` | text/non-text split | 进行中 |
| `chunk_corpus_n400_fair/` | `build_chunk_corpus.py --chunks n400.json` | chunk | fair baseline |
| `chunk_corpus_n400_partial_overlay/` | `build_chunk_corpus.py --chunks n400_trial57_enriched.json` | chunk + partial enrich | exploratory |
| 其他 chunk_corpus_* | 同上，不同 chunk_size / overlay | chunk | — |

## Track B — 生产轨（1040-doc 量产）

**用途**：大规模 SFT 数据生产（最终交付物）

**数据流**：
```
00_raw/{latex_sources_batch2, mineru_output}
  → 01_graphs/{reference_graph_v2, multimodal_elements_v2, chunk_virtual_nodes_v2, hub_scores_v2}
  → 02_enriched/{production_full, production_partial, gap227, hub_shortchain}
  → 03_queries/full_doc_v2_prod_1040/ (待产出)
  → 04_triplets/ (待产出)
```

**关键文件（按用途）**：

| 用途 | 文件 | 大小 | 说明 |
|------|------|------|------|
| Reference graph | `data/01_graphs/latex_reference_graph_v2.json` | 274M | 1425-doc 全量引用图 |
| Elements | `data/01_graphs/multimodal_elements_v2.json` | 93M | 27209 elements（1040 docs） |
| Chunk VNodes | `data/01_graphs/chunk_virtual_nodes_v2.json` | — | 段落节点（build_paragraph_chunks.py 的输入） |
| Hub scores | `data/01_graphs/hub_scores_v2.json` | 126M | 全量 hub score |
| Hub pairs | `data/01_graphs/hub_candidates_v2_top25.json` | 11M | top 25% hub pair (1156 pairs) |
| Hub combined | `data/02_enriched/hub_candidates_v2_combined.json` | 12M | gap227 + hub_shortchain 合并 |
| Enrich (full) | `data/02_enriched/multimodal_elements_v2_production_full.json` | 88M | 全量元素 + 已有 enrich |
| Enrich (partial) | `data/02_enriched/multimodal_elements_v2_production_partial.json` | — | 813 docs / 6707 enriched |
| Gap227 enrich | `data/02_enriched/multimodal_elements_v2_gap227_enriched.json` | — | gap227 4301 elements |
| Doc lists | `data/doc_lists/{gap227_doc_ids, old_53_docs}.txt` | — | 文档子集清单 |

**Enrich 状态**：

| 组件 | 目标 | 已完成 | 比例 | 说明 |
|------|------|--------|------|------|
| production_full | 27209 | 10988 | **40.4%** | 全量 enrich 覆盖 |
| production_partial | 27209 | 6705 | 24.6% | 813 docs 子集 |
| gap227 | 4301 | 4283 | **99.6%** | 接近完成 |
| hub_shortchain | 759 | 0 | 0% | API 403 阻塞（Job 61529） |
| 其余元素 | ~15442 | 0 | 0% | 策略决定暂不 enrich |

## 跨轨共享

部分数据在 A/B 之间共享：

| 数据 | 共享方式 |
|------|---------|
| `data/00_raw/mineru_output/` | 两轨共用（B 的 1040 docs 包含 A 的 53 docs） |
| `data/00_raw/latex_sources_batch2/` | B 专用（1425 docs，包含 A 的 53） |
| Enrich model/env | 共用 `minerU` conda env + company API key |
| Eval scripts | `eval_dense_retrieval.py`、`eval_graph_topk_rerank.py` 等可跨轨复用 |

## 关键边界规则

1. **Track A 的实验不应依赖 Track B 的 partial enrich 文件**（会造成 unfair comparison）
2. **Track B 的 corpus 构建应等 enrich 完成后一次性进行**（避免分批次的不一致）
3. **Chunk build 一致性问题（P1 bug）**：eval qrels 和 chunk_contains_element 边必须来自同一次 chunk build
4. **API logging**：所有 company API 调用必须经过 `local_api_logger → api_logs/`

## 已知数据质量问题

| 问题 | 严重度 | 影响范围 | 状态 |
|------|--------|---------|------|
| Chunk-element 边不一致 | P1 | Track A chunk rerank 全部结果 | 待修复 |
| mineru→latex 模糊匹配丢失率 | P1 | element 定位精度 | 待核查（B2） |
| API key 403 | P0 | Track B hub_shortchain enrich 阻塞 | 待解决 |
| `compute_chunk_element_stats.py` 忽略 --qrels | P3 | 无实际影响 | 已知 |
