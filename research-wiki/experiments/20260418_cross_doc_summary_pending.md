---
type: experiment
node_id: exp:20260418_cross_doc_summary_pending
status: pending
verdict: unknown
created_at: 2026-04-18T00:00:00Z
updated_at: 2026-04-18T00:00:00Z
---

# One-line summary

Pending experiment to test whether cross-document summary edges with citation boost improve the corrected explicit-only rerank baseline.

## Planned Variants

- Threshold 0.70 + citation boost 0.10
- Threshold 0.80 + citation boost 0.10
- Threshold 0.70 + citation boost 0.00

## Acceptance Gate

The method only graduates if it preserves or improves R@1 and MRR relative to the corrected explicit-only + static-prior baseline.

## Current Note

This is the correct next experiment, but it must be compared only against the corrected baseline, not the invalid old graph runs.

## Connections

- Tests: `idea:002`
- Resolves: `claim:C3`
- Addresses: `gap:G2`

