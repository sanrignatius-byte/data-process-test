```markdown
---
type: experiment
node_id: exp:20260419_summary_line_closed
status: completed
verdict: invalidates_idea_of_summary_virtual_node_uplift
created_at: 2026-04-19T00:00:00Z
updated_at: 2026-04-19T00:00:00Z
---

# One-line summary

Fine-grained summary-signal ablation (section × chunk × child_alpha × max_children × weights, SLURM 61417) yields no configuration that strictly beats `explicit_only` on both R@1 and MRR; the summary-as-virtual-node line is formally closed for this project.

## Setup

- Model: Qwen3-Embedding-0.6B
- Corpus: `v1_enriched`, 473 queries
- Scripts: `slurm_scripts/25_summary_signal.sh`, `26_summary_chunk.sh`, `27_summary_finegrained.sh`
- Methods: `summary_plus_static`, `static_prior` (fallback)

## Best configuration

`explicit_plus_chunk_w010_t10_a1 / summary_plus_static`:
- R@1 = 0.2400
- R@5 = 0.5856
- MRR = 0.6352

Reference `explicit_only` static_prior baseline:
- R@1 = 0.2421
- MRR = 0.6399

## Conclusion

Across all tested scopes (section, chunk), child_alpha values, max_children caps, and weight schedules, summary signal cannot produce a strict positive delta on R@1 and MRR simultaneously. The best configuration is within noise of `explicit_only` and slightly below it.

This invalidates, for this project and this corpus, the idea that summary nodes are a standalone retrieval uplift. Summary nodes may still be useful as metadata for QA or evidence traceability, but not as an edge-generating mechanism in retrieval rerank.

## Implication for the 4.16 mentor plan

Mentor's 4.16 plan listed `summary` as the priority virtual-node type. With this negative result, the priority shifts to element-level typed cross-doc edges (see `claim:C5`, `exp:20260419_typed_crossdoc`).

## Connections

- Invalidates the retrieval-uplift framing of summary virtual nodes.
- Originates: `claim:C6` (new — summary virtual nodes do not improve 0.6B retrieval).
- Redirects priority to: `idea:004`, `claim:C5`.
```
