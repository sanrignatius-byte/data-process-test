---
type: paper
node_id: paper:wang2026_s1mmalign
title: "S1-MMAlign: A Large-Scale, Multi-Disciplinary Dataset for Scientific Figure-Text Understanding"
authors: ["He Wang", "Longteng Guo", "Pengkang Huo", "Xuanxu Lin", "Yichen Yuan", "Jie Jiang", "Jing Liu"]
year: 2026
venue: arXiv
external_ids:
  arxiv: "2601.00264"
tags: [figure-text, cross-doc, dataset, recaption, CLIP, scientific-documents]
relevance: core
origin_skill: research-lit
created_at: 2026-05-20T00:00:00Z
updated_at: 2026-05-20T00:00:00Z
---

# One-line thesis

15.5M scientific image-text pairs from 2.5M papers, with figures recaptioned from
abstract + citation context to fix weak/generic captions, and explicit cross-document
figure-text alignment.

## Problem / Gap

Raw scientific captions are weak/generic, hurting figure-text alignment; existing data
is intra-doc and small.

## Method

Semantic-enhancement pipeline: a multimodal LLM recaptions each figure by synthesizing
the paper abstract + citation contexts, turning a bare caption into a context-rich one.
Reports CLIP image-text alignment +18.21% from this enrichment. Explicitly supports
cross-document figure matching.

## Key Results

15.5M pairs / 2.5M open-access papers; CLIP alignment +18.21% after recaption.
Dataset on HuggingFace `ScienceOne-AI/S1-MMAlign` (CC BY-NC 4.0).

## Reusable Ingredients

- **Recaption enrichment** = our `enriched` rerank direction, externally validated.
  Could upgrade our static enriched fields into active VLM recaption.
- A ready cross-doc figure-text dataset for training/calibration/eval.

## Limitations / Failure Modes

Assumes clean, recoverable text context (abstract, citation context, parsable captions).
Does NOT cover the parser-degraded regime where captions are bare numbers / OCR
fragments — exactly our measured 35.8% unusable-caption case (`claim:C16`, `gap:G10`).

## Relevance to This Project

Closest competitor: cross-doc figure-text + caption enrichment. Our differentiator is
robustness under masked/degraded reference signals, which they assume away.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
