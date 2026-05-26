---
type: paper
node_id: paper:hessel2019_multilink
title: "Unsupervised Discovery of Multimodal Links in Multi-image, Multi-sentence Documents"
authors: ["Jack Hessel", "Lillian Lee", "David Mimno"]
year: 2019
venue: EMNLP
external_ids:
  arxiv: "1904.07826"
  acl: "D19-1210"
tags: [multimodal-linking, unsupervised, figure-text, document-structure, intra-doc]
relevance: core
origin_skill: research-lit
created_at: 2026-05-20T00:00:00Z
updated_at: 2026-05-20T00:00:00Z
---

# One-line thesis

Document-level image/sentence co-occurrence is enough weak supervision to predict
element-level image↔sentence links at test time, with no per-pair annotation.

## Problem / Gap

Images and text co-occur in documents but explicit element-to-element links are absent;
manual annotation is prohibitive at scale.

## Method

Train on a structured objective that only knows whether a *bag* of images and a *bag* of
sentences co-occur in the same document; the learned similarity then predicts specific
sentence↔image links within a held-out document. Document co-occurrence is the training
signal; layout/proximity is an implicit prior.

## Key Results

Outperforms unsupervised baselines at recovering intra-document image-sentence
correspondences across web document collections.

## Reusable Ingredients

- **Open source**: https://github.com/jmhessel/multi-retrieval (upgraded to TF2).
- The core idea we are NOT yet using: treat *document co-occurrence as weak supervision*
  to train/calibrate a linker, instead of pure zero-shot CLIP. Directly applicable to
  lifting our cross-doc edges beyond zero-shot similarity.

## Limitations / Failure Modes

Intra-document only; assumes images and their describing sentences live in the same doc.
Pre-CLIP visual features.

## Relevance to This Project

Academic ancestor of our `same_page_cross_type` + `text_describes_figure` intra-doc soft
links. Its weak-supervision-from-co-occurrence trick is a candidate for turning our
zero-shot CLIP recall into a trained linker (relevant to `claim:C16`, `gap:G10`).

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
