---
type: paper
node_id: paper:hwang2026_connecting_dots
title: "Connecting the Dots: Surfacing Structure in Documents through AI-Generated Cross-Modal Links"
authors: ["Alyssa Hwang", "Hita Kambhamettu", "Yue Yang", "Ajay Patel", "Joseph Chee Chang", "Andrew Head"]
year: 2026
venue: arXiv
external_ids:
  arxiv: "2602.16895"
tags: [multimodal-linking, VLM, figure-text, document-structure, intra-doc, reading-interface]
relevance: core
origin_skill: research-lit
created_at: 2026-05-20T00:00:00Z
updated_at: 2026-05-20T00:00:00Z
---

# One-line thesis

Use VLMs to auto-generate fine-grained cross-modal links (figure points ↔ highlighted
phrases) inside a document, surfacing structure even where no explicit reference exists.

## Problem / Gap

Dense documents scatter related details across text, figures, tables; explicit
references are often missing, hurting comprehension/navigation.

## Method

A VLM identifies entities in figures ("figure points") and matching text passages
("highlighted phrases"), consolidates them in a reference panel, and powers a research-
paper reading interface (navigate by clicking figures in a visual index). No manual
annotation; the VLM judges relatedness directly from pixels + text.

## Key Results

User studies on comprehension/navigation; this is a systems/HCI + VLM-linking paper
rather than a retrieval-metric paper.

## Reusable Ingredients

- **VLM-direct link judgment** bypasses caption text entirely — the exact move that
  could fix our 87% caption-zero-overlap bottleneck (`claim:C16`).
- No public code found (reading-interface paper).

## Limitations / Failure Modes

Intra-document only — no cross-document linking. VLM cost per link.

## Relevance to This Project

SOTA of the intra-doc soft-link line we approximate with CLIP+TF-IDF. Motivates
replacing caption token-matching with VLM judgment in our rerank (`gap:G10`).

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
