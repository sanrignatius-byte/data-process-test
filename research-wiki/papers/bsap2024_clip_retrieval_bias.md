---
type: paper
node_id: paper:bsap2024_clip_retrieval_bias
title: "Balanced Similarity with Auxiliary Prompts: Towards Alleviating Text-to-Image Retrieval Bias for CLIP in Zero-shot Learning"
authors: ["(BSAP authors)"]
year: 2024
venue: arXiv
external_ids:
  arxiv: "2402.18400"
tags: [CLIP, retrieval-bias, similarity-calibration, zero-shot, failure-mode]
relevance: related
origin_skill: research-lit
created_at: 2026-05-20T00:00:00Z
updated_at: 2026-05-20T00:00:00Z
---

# One-line thesis

CLIP's retrieval bias comes from an imbalanced range of similarity scores; balancing
the similarity distribution (auxiliary prompts) mitigates it.

## Problem / Gap

Zero-shot CLIP retrieval is biased because raw cosine scores occupy an uneven,
compressed range — high scores do not mean high relevance.

## Method

Balanced Similarity with Auxiliary Prompts (BSAP): calibrate/rebalance the similarity
distribution so scores are comparable across queries.

## Key Results

Reduces text-to-image retrieval bias in zero-shot settings.

## Reusable Ingredients

- Diagnosis matches our observation that layout-similar figures get uniformly high CLIP
  scores (`claim:C16`). Score de-biasing/calibration is a cheap mitigation for our
  cross-doc visual false positives.

## Limitations / Failure Modes

General zero-shot retrieval, not scientific figures specifically.

## Relevance to This Project

Explains *why* `cross_doc_visual_sim` p90≈0.92 with poor precision; offers a calibration
fix orthogonal to the semantic rerank.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
