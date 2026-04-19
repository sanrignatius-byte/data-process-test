```markdown
---
type: claim
node_id: claim:C6
status: supported
created_at: 2026-04-19T00:00:00Z
updated_at: 2026-04-19T00:00:00Z
---

# Claim

On Qwen3-Embedding-0.6B with the `v1_enriched` corpus (473 queries), summary virtual nodes as a retrieval edge mechanism do not produce a strict positive delta on both R@1 and MRR relative to the corrected `explicit_only` static-prior baseline, across section- and chunk-scoped variants and across the full tested grid of child_alpha, max_children, and weight values.

## Evidence

From `exp:20260419_summary_line_closed` (SLURM 61417):
- Best summary configuration: `explicit_plus_chunk_w010_t10_a1 / summary_plus_static` → R@1 = 0.2400, MRR = 0.6352.
- Reference `explicit_only` static_prior → R@1 = 0.2421, MRR = 0.6399.
- No tested combination strictly exceeds the reference on both metrics.

## Implication

- **Summary-as-retrieval-signal** is closed under this model/corpus: do not claim R@1/MRR uplift from summary edges.
- **Summary nodes themselves are NOT closed.** They serve two active purposes:
  1. Cross-doc structural scaffold (connecting documents via section-level summaries).
  2. Embedding input for downstream tasks (embedding quality and coverage depend on summary content).
- Do not conflate "retrieval uplift line closed" with "summary nodes abandoned."

## Scope

This claim is limited to:
- Model: Qwen3-Embedding-0.6B.
- Corpus: `v1_enriched`.
- Metric gate: strict positive on both R@1 and MRR.
- Summary granularities: section and chunk.

## Risk

- A much stronger embedding model (e.g., full 4B or larger) could change the picture; retest if models change.
- A different method axis (e.g., neighbor propagation with tuned decay) could still extract summary signal, but is out of the tested grid.

## Connections

- Supported by: `exp:20260419_summary_line_closed`
- Redirects priority to: `idea:004`, `claim:C5`
- Addresses: `gap:G4` (partially — narrows the retrieval side of the graph-value story).
```
