---
type: idea
node_id: idea:008
title: "Robust cross-doc multimodal edge recovery under parser-degraded captions"
stage: proposed
outcome: null
created_at: 2026-05-20T00:00:00Z
updated_at: 2026-05-20T00:00:00Z
---

# Idea

Turn pure-MinerU cross-document figure/table edges from a recall layer into strong
edges **specifically in the regime where captions are degraded/masked by parsing** —
the case existing work assumes away. Keep CLIP as cheap recall; replace caption
token-matching with a caption-independent semantic rerank.

## Why now

`exp:20260520_mineru_clip_xdoc_pipeline` measured the bottleneck: 87% of xdoc edges have
zero caption token overlap, 35.8% of captions are unusable, only ~5% of edges have real
text support (`claim:C16`, `gap:G10`). The intra-doc side is already strong
(`claim:C15`), so cross-doc is the binding constraint for the PDF-first multi-hop graph.

## Approach (reuse existing methods)

1. VLM direct link judgment on CLIP top-k candidates (bypasses captions) —
   `paper:hwang2026_connecting_dots`.
2. Recaption enrichment from abstract + local context before similarity —
   `paper:wang2026_s1mmalign` (+18% CLIP reported).
3. LLM coarse-to-fine rerank using `figure_type`/keywords/context as document features —
   `paper:tian2026_corank`.
4. CLIP similarity de-biasing to kill layout false positives —
   `paper:bsap2024_clip_retrieval_bias`.
5. (Optional, from `paper:hessel2019_multilink`) use document co-occurrence as weak
   supervision to train a linker instead of staying zero-shot.

## Success criterion

On the existing 3238 xdoc candidates, raise the strong-edge share from ~5% without
inflating false positives (manual spot-check + held-out check), and show the gain is
larger on degraded-caption edges than on clean-caption ones (the niche claim).

## Target gaps

- `gap:G10`

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
