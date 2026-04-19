---
type: idea
node_id: idea:002
title: "Cross-document summary similarity edges with citation boost"
stage: pivoted
outcome: retrieval_uplift_closed_crossdoc_anchor_active
dominance: supporting
created_at: 2026-04-18T00:00:00Z
updated_at: 2026-04-19T00:00:00Z
---

# One-line thesis

Use LLM section summaries as semantic anchors to build cross-document similarity edges; summary-as-direct-retrieval is closed, but summary nodes remain the primary scaffold for cross-document traversal experiments.

## Problem / Gap

Current virtual edges are mostly intra-document and do not help cross-document retrieval enough.

## Mechanism

- Use LLM section summaries as semantic anchors.
- Build summary-summary edges with Qwen3 embedding similarity.
- Apply citation-based confidence boost only after similarity thresholding.

## Current Status

**Summary-as-retrieval-uplift closed (C6)**: All ablations across section/chunk scope and weight configs showed no strict R@1/MRR gain over explicit_only baseline.

**Cross-doc direction still active**: Summary nodes are the canonical text representation for cross-document bridge experiments. `idea:004` (typed element edges) extends this direction at finer granularity and currently provides the only confirmed R@10 cross-doc uplift. Future cross-doc experiments (paragraph-level, entity-based) may still route through summary-level scaffolding.

## Why It Matters

Summary nodes are the only available text-rich representation for section-level cross-document connectivity. Even if they don't directly lift retrieval metrics, they are needed as the structural backbone for any cross-doc traversal that goes beyond element similarity.

## Failure Modes

- Semantic edges may inject too much noise and damage R@1 or MRR (confirmed for direct retrieval use).
- Citation boost may let weak semantic matches slip in if applied incorrectly.

## Connections

- Tested by: `exp:20260418_cross_doc_summary_pending`, `exp:20260419_summary_line_closed`
- Cross-doc direction continued in: `idea:004`
- Supports: `claim:C3` (partially), structural scaffold for future cross-doc ideas
- Addresses: `gap:G2`
