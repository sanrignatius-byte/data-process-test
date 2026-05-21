---
type: claim
node_id: claim:C12
status: reported_phase0_required
created_at: 2026-05-19T00:00:00Z
updated_at: 2026-05-19T02:40:00Z
---

# Claim

Element-level cross-document bridges (Qwen3-Embedding-4B element-to-element matches, reranked) produce harder and more useful multi-hop queries than doc-level citation-walk bridges **and** existing paragraph-mediated cross-doc L3 chains. Specifically: queries built from element-level cross-doc chains will show **lower BM25 R@10** (truly harder retrieval) **without** lowering QC pass rate or answer correctness.

## Evidence

- Reported only — no experiment run yet.
- Supporting prior: `claim:C5` (typed cross-doc element edges already lift R@10 +1.5pp) shows element edges are non-trivial signal.
- Supporting prior: existing `archive/data_legacy/embedding_probes/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2b_cap10.jsonl` (590 records, 11800 matches, audited and reranked) is the raw material.
- Red/blue-team correction: current L3 pass data already has 87/146 paragraph-mediated cross-doc chains, so C12 must compare against that baseline too; doc-level citation walks alone are an insufficient control.
- Will be validated by `exp:20260519_xdoc_pairing_module` (planned).

## Scope

Applies to cross-doc chain construction on the 86-doc CS corpus (current) and the noncs1000 corpus when ready (downloading 2026-05-19, ~700 expected delivered).

## Why It Matters

If true, the cross-doc dimension of M4 should be sourced from element-level matches, not from citation walks. This affects the entire Phase 2 design: the bridge layer becomes embedding-based, not citation-based.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
