---
type: experiment
node_id: exp:20260418_graph_source_audit
status: completed
verdict: invalidates_old_graph_claims
created_at: 2026-04-18T00:00:00Z
updated_at: 2026-04-18T00:00:00Z
---

# One-line summary

Audit of graph-source handling found that earlier graph rerank runs silently defaulted to virtual-only graph layers, which invalidates old explicit-graph conclusions.

## Audit Trigger

The biggest issue reported on 2026-04-18 was that previous graph experiments were run on the wrong graph base and effectively missed the intended explicit bridge-edge layer.

## Evidence

- Old report: `data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_only/report.json`
  - `hub_pair`: 192
  - `adjacent_bridge`: 868
- Fixed report: `data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_only_fixed/report.json`
  - `hub_pair`: 456
  - `adjacent_bridge`: 868
- Script correction: `scripts/eval_graph_topk_rerank.py` now requires explicit `--graph-sources` instead of silently defaulting to the wrong layer.

## Interpretation

Old graph-rerank conclusions should be treated as provisional or invalid. The corrected `graph_explicit_only_fixed` family becomes the new starting point.

## Connections

- Invalidates prior evidence attached to: `idea:001`
- Resets baseline for: `exp:20260418_cross_doc_summary_pending`
- Supports: `claim:C4`

