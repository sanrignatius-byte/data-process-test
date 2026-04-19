---
type: idea
node_id: idea:003
title: "Method C long-chain query synthesis"
stage: deferred
outcome: partial
dominance: supporting
created_at: 2026-04-18T00:00:00Z
updated_at: 2026-04-18T00:00:00Z
---

# One-line thesis

Generate harder long-chain multimodal queries through bridge-chain compression and QC so the project can synthesize richer training data.

## Problem / Gap

The project needs complex, graph-grounded synthetic data, but not every generation pipeline should be elevated to the main contribution.

## Mechanism

- Discover long evidence paths.
- Compress bridge chains into controllable query-generation prompts.
- QC outputs for grounding and hallucination.

## Current Status

Useful for data generation, but currently not the dominant story. Latest requirements favor retrieval uplift, QA uplift, and practical data delivery first.

## Why It Matters

This remains relevant for the SFT-data deliverable and future agent-facing data pipelines.

## Failure Modes

- QC cost is high.
- Long-chain complexity can distract from the simpler and stronger retrieval story.

## Connections

- Supports: `claim:C4`
- Addresses: `gap:G3`

