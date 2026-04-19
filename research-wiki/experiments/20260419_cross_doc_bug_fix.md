```markdown
---
type: experiment
node_id: exp:20260419_cross_doc_bug_fix
status: completed
verdict: prior_cross_doc_results_were_silently_null
created_at: 2026-04-19T00:00:00Z
updated_at: 2026-04-19T00:00:00Z
---

# One-line summary

`load_cross_doc_adjacency()` was producing zero pid-pairs because it tried to parse `(doc, section_title)` from node IDs like `1104.3913_secsummary_1104.3913_secsum_1`; it now uses edge metadata `source_doc` / `source_section` hints first, turning cross_doc from a silent no-op into 5135 pid-pairs / 167 doc-pairs.

## Diagnostic

In `exp:20260419_multi_source_stacking`:
- `explicit + crossdoc` gave exactly the same metrics as `explicit_only`
- `crossdoc_only` gave exactly the same metrics as dense baseline

Both identities indicated cross_doc contributed an empty adjacency at the pid layer.

## Fix

`scripts/eval_graph_topk_rerank.py :: load_cross_doc_adjacency()` now:
1. Reads `source_doc` and `source_section` fields from edge metadata as the primary hint.
2. Falls back to the old node-ID string parser only if the metadata is missing.
3. Keeps the same downstream pid-expansion logic.

## Post-fix counts

- pid_pairs: 0 → 5135
- doc_pairs: 0 → 167

## Post-fix impact

See `exp:20260419_typed_crossdoc` — the fixed section-level cross_doc layer, combined with explicit under `static_plus_neighbor`, ties the new R@10 high of 0.6406.

## Connections

- Upgrades baseline for: `claim:C3`
- Unblocks: `exp:20260419_typed_crossdoc`
- Explains anomalies observed in `exp:20260419_multi_source_stacking`
```
