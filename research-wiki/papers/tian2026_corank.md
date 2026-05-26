---
type: paper
node_id: paper:tian2026_corank
title: "CoRank: LLM-Based Compact Reranking with Document Features for Scientific Retrieval"
authors: ["Runchu Tian", "Xueqiang Xu", "Bowen Jin", "SeongKu Kang", "Jiawei Han"]
year: 2026
venue: WSDM
external_ids:
  arxiv: "2505.13757"
tags: [reranking, LLM, scientific-retrieval, document-features, coarse-to-fine]
relevance: related
origin_skill: research-lit
created_at: 2026-05-20T00:00:00Z
updated_at: 2026-05-20T00:00:00Z
---

# One-line thesis

Training-free LLM reranking that first ranks on compact semantic features
(categories, sections, keywords) then fine-reranks top candidates on full text.

## Problem / Gap

Full-document LLM reranking is expensive and noisy; scientific retrieval needs
structure-aware, efficient reranking.

## Method

Three stages: (i) offline-extract high-level features (categories/keywords) from
unstructured docs; (ii) coarse rerank on compact representations, keep a top subset;
(iii) fine rerank that subset on full documents.

## Key Results

5 academic retrieval datasets, average nDCG@10 50.6→55.5; beats RankVicuna/RankZephyr/
ChatGPT-rerank. Code stated "to be released" (not found yet).

## Reusable Ingredients

- **Coarse-to-fine LLM rerank with document features** is exactly the pattern to lift
  our cross-doc edges from recall to strong: CLIP top-k recall → LLM rerank using
  `figure_type`/keywords/local context as document features (`gap:G10`).

## Limitations / Failure Modes

Text retrieval, not multimodal/figure linking; would need adaptation to figure pairs.

## Relevance to This Project

Method template for the semantic-rerank step we proposed but have not built.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
