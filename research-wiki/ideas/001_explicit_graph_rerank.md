---
type: idea
node_id: idea:001
title: "Explicit bridge-edge rerank with hub-aware static prior"
stage: active
outcome: positive
dominance: dominant
created_at: 2026-04-18T00:00:00Z
updated_at: 2026-04-18T00:00:00Z
---

# One-line thesis

Use explicit document-structure bridge edges and a hub-aware static prior to rerank dense retrieval results so that true evidence surfaces earlier.

## Problem / Gap

Dense retrieval alone does not rank multimodal evidence strongly enough on the rebuilt scientific-document benchmark.

## Mechanism

- Use explicit bridge edges from enriched hub candidates.
- Rerank top-k dense results with graph-aware static prior.
- Keep the method simple and precision-oriented.

## Current Status

This is the current main method thesis and the best-supported precision-oriented result.

## Why It Matters

It directly maps to the latest requirement: show retrieval improvement that is easy to explain as business value.

## Failure Modes

- Wrong graph layer selection invalidates conclusions.
- Neighbor propagation can over-spread relevance and hurt first-hit quality.

## Connections

- Tested by: `exp:20260417_explicit_rerank_fixed`
- Supports: `claim:C1`, `claim:C4`
- Addresses: `gap:G1`

