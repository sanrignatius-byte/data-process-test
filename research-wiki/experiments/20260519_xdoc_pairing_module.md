---
type: experiment
node_id: exp:20260519_xdoc_pairing_module
status: planned
created_at: 2026-05-19T00:00:00Z
updated_at: 2026-05-19T00:00:00Z
---

# Experiment: Element-level cross-doc pairing module

Mirror `src/pairing/intra_doc_pairs.py` → `src/pairing/cross_doc_pairs.py`. Use the existing reranked Qwen3-Embedding-4B element-level cross-doc matches to produce a `cross_doc_pairs_v1.json` artifact in the same schema as `hub_candidates_enriched_v3.json`, so it can be fed to existing query generators (`generate_multihop_l1_queries.py`) without code changes downstream.

## Design

- **Input**:
  - `data/00_raw/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2b_cap10.jsonl` (590 records, 11800 reranked matches, audit completed: `reciprocal=0.8119`, `unique_top1=286`)
  - `data/01_graphs/multimodal_elements.json` for element metadata
  - `data/01_graphs/latex_reference_graph.json` for in-doc paragraph references (used by bridge filter)

- **Filters**:
  - At least one endpoint is figure / table / formula (multi-modal constraint)
  - Both endpoints are referenced by ≥1 paragraph in their parent doc (no orphan elements)
  - Reciprocal rank ≥ 0.5 (high-confidence matches only)
  - Reject pairs whose docs are identical (intra-doc safety check)

- **Output**:
  - `data/02_enriched/cross_doc_pairs_v1.json` — schema-compatible with `hub_candidates_enriched_v3.json`
  - Each pair has `pair_type`, `element_a`, `element_b`, `bridge_evidence`, `cross_doc_metadata` (similarity score, reciprocal rank)

- **Validation hooks**:
  - Reuse `qc_multihop_query()` rule QC and `run_llm_qc()` LLM QC.
  - Sanity check: feed 20 pairs to `generate_multihop_l1_queries.py --dry-run` and verify prompts render without missing fields.

## Pre-registered hypothesis

Element-level cross-doc pairs will produce queries with **lower** BM25 R@10 than intra-doc pairs (truly harder retrieval) but **same** QC pass rate (~58-64% per v4.2 baseline). If pass rate drops more than 10pp, the pair filter is too lax.

## Status

Planned 2026-05-19. Not yet run. Blocking: cross_doc_pairs.py module not yet written. Estimated effort: 1-2 days to mirror intra_doc_pairs.py.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
