# Experiment Plan — Cross-Document Element-Level Citation Query Pipeline

**Problem**: Existing cross-document edges (citation walk, cross-doc summary, typed cross-doc) all fail to improve retrieval on M4query_v1. Root cause: M4query_v1 is intra-doc only — all qrels evidence is within a single document. Cross-doc signals are noise for this eval. Need a dedicated cross-document eval with queries that genuinely require evidence from two documents.
**Method Thesis**: Build element-level citation pairs from LaTeX `\cite{}` context → map to MinerU element IDs → generate cross-document queries with citation context as bridge → evaluate on a multi-document retrieval corpus.
**Date**: 2026-05-03

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C_XD1 | Element-level citation mapping produces valid cross-document query candidates | ≥ 50 candidate pairs with matched elements on both sides | B1 |
| C_XD2 | Cross-document queries are answerable only with evidence from both docs (true cross-doc) | LLM ablation pass rate ≥ 50% (removing either doc's evidence → cannot answer) | B2 |
| C_XD3 | Graph rerank helps cross-document retrieval more than intra-document (because BM25 is weaker across docs) | Graph MRR delta on cross-doc queries > intra-doc delta | B3 |

---

## Why Previous Approaches Failed

| Approach | Failure Mode | Root Cause |
|----------|-------------|------------|
| Citation walk (Phase0) | Net negative MRR (−0.01) | Doc-level granularity: one paper cites another doesn't mean specific elements are related |
| Cross-doc summary edges | No uplift beyond 0.6913 | Summary-to-summary similarity is too vague for element-level retrieval |
| Typed cross-doc edges | 0.6406 < explicit_only 0.6913 | Cross-doc element edges introduce noise without citation context to filter relevance |
| Method C long chain | 8.5% pass rate | Single-document true multi-hop is extremely rare |

**The missing ingredient**: element-level citation context. When Doc A says "as shown in [Doc B]'s Figure 3...", that's a precise cross-document element reference. We need to extract these, map them to element IDs, and use them as query anchors.

---

## Pipeline Overview

```
latex_reference_graph_v2.json (1040 docs)
    │
    ├── cross_doc_cite edges (434 edges, doc→doc)
    │   └── edge_context: ±300 chars around each \cite{...}
    │
    ├── Step 1: Extract element references from citation context
    │   └── Regex: "Figure~\ref{...}" / "Table~\ref{...}" / "Eq.~\ref{...}"
    │       within edge_context ± expansion
    │
    ├── Step 2: Map LaTeX labels → MinerU element IDs
    │   └── Use existing label→element mapping from multimodal_elements_v2.json
    │       (49.8% label match rate → need fallback strategies)
    │
    ├── Step 3: Build candidate pairs
    │   └── (doc_A, element_X) → (doc_B, element_Y)
    │       with citation_context as bridge text
    │       Filter: both elements must have enriched_content or caption
    │       Target: 50-100 pairs
    │
    ├── Step 4: Generate cross-document queries
    │   └── Reuse generate_multihop_l1_queries.py with cross-doc prompt
    │       Key: query must reference context from doc_A, answer requires doc_B
    │
    └── Step 5: Evaluate
        └── Build retrieval corpus spanning both docs' elements
            BM25 baseline → dense → graph rerank
            Per-query ablation: remove doc_A evidence → can't answer?
```

---

## Experiment Blocks

### Block 1: Element-Level Citation Pair Extraction [MUST-RUN]

- **Claim tested**: C_XD1
- **Why this block exists**: Foundation for all downstream cross-document work. Without element-level pairs, cross-doc query generation has no anchor.
- **Input**:
  - `data/01_graphs/latex_reference_graph_v2.json` (1425 docs, 67880 edges)
  - `data/01_graphs/multimodal_elements_v2.json` (1040 docs, 27209 elements)
- **Process**:

  **Step 1a**: Collect cross-document citation edges
  - Filter `latex_reference_graph_v2.json` edges where `source_doc ≠ target_doc`
  - Extract `edge_context` field (±300 chars around `\cite{}`)
  - Expand to ±600 chars if no element reference found in ±300

  **Step 1b**: Extract element references from citation context
  - Regex patterns:
    - `Figure~?\\ref\{([^}]+)\}` or `Figure\s+(\d+)`
    - `Table~?\\ref\{([^}]+)\}` or `Table\s+(\d+)`
    - `Equation~?\\ref\{([^}]+)\}` or `Eq\.?\s*\\ref\{([^}]+)\}`
  - For each citation edge, collect: (cited_doc_id, label_in_cited_doc, element_type)
  - Also extract element references from the CITING side: what figure/table in doc_A is making the reference?

  **Step 1c**: Map LaTeX labels → MinerU element IDs
  - Primary: exact label match in `multimodal_elements_v2.json` label_index
  - Fallback 1: number-based matching (e.g., "fig:results" → doc_id + "_figure_3" where number extracted from label)
  - Fallback 2: caption Jaccard matching (threshold 0.25)
  - Record match confidence per mapping

  **Step 1d**: Filter and rank
  - Require: at least one element mapped on cited side
  - Prefer: elements mapped on BOTH citing and cited sides (for dual-anchor queries)
  - Rank by: citation context richness (word count, presence of comparison/contrast language)
  - Deduplicate: (citing_element, cited_element) frozenset

- **Output**: `data/02_enriched/cross_doc_citation_pairs.json`
  - Format: compatible with `hub_candidates_enriched` schema
  - Fields: `element_a` (citing side), `element_b` (cited side), `citation_context`, `match_confidence`, `bridge_text`
- **Metrics**: Number of valid pairs, element mapping success rate, per-doc coverage
- **Success criterion**: ≥ 50 valid pairs with matched elements on both sides
- **Priority**: MUST-RUN

### Block 2: Cross-Document Query Generation [MUST-RUN]

- **Claim tested**: C_XD2
- **Why this block exists**: Produce a clean cross-document query set for evaluation
- **Input**: `cross_doc_citation_pairs.json` from B1 (top-ranked 50-100 pairs)
- **Prompt design**:
  - Use existing `generate_multihop_l1_queries.py` with a new `PROMPT_CROSS_DOC_CITATION`
  - Key requirements:
    - Query must reference specific context from doc_A (the citing paper)
    - Answer requires evidence from doc_B (the cited paper)
    - Citation context provided as bridge text: "Doc A cites Doc B's [element], saying: [citation_context]"
    - Ban: "compare X and Y" templates, yes/no questions
    - Require: specific metric/claim/mechanism in query
  - QC: standard rule QC + LLM ablation (drop doc_A evidence → can't answer? drop doc_B evidence → can't answer?)
- **Output**: `data/03_queries/cross_doc_citation_queries.jsonl`
- **Metrics**: QC pass rate, per-query dual-doc answerability
- **Success criterion**: ≥ 25 QC-pass queries (50% of 50 input pairs)
- **Priority**: MUST-RUN
- **Cost**: ~$3-5 (company API, gpt-5.4)

### Block 3: Cross-Document Retrieval Evaluation [MUST-RUN]

- **Claim tested**: C_XD3
- **Why this block exists**: Measure whether graph rerank helps cross-document retrieval more than intra-document
- **Dataset**: Cross-doc queries from B2 + retrieval corpus spanning both cited and citing docs
- **Compared systems**:
  1. BM25 baseline
  2. Dense retrieval (Qwen3-Embedding-4B)
  3. Graph rerank: explicit_only + static_plus_neighbor (best config from Phase0)
- **Metrics**: R@10, MRR; per-query "both docs hit?" metric (% queries where top-10 includes evidence from BOTH documents)
- **Setup details**:
  - Corpus: element-level passages from all docs involved in cross-doc queries
  - Use `enriched_content` where available
  - Graph: `latex_reference_graph_v2.json` restricted to involved docs
- **Success criterion**: Graph MRR delta on cross-doc > intra-doc delta (from Phase0)
- **Table / figure target**: Paper cross-document section
- **Priority**: MUST-RUN
- **Est. GPU time**: ~20 min

---

## Run Order

| Milestone | Goal | Runs | Est. Time | Decision Gate |
|-----------|------|------|-----------|---------------|
| M0 | Verify citation edge data quality | Read 20 random citation contexts | 30 min manual | ≥ 60% contain element references |
| M1 | Run B1: extract element-level pairs | 1 script run | 2 min CPU | ≥ 50 valid pairs |
| M2 | Run B2: generate queries (top-50 pairs) | 1 batch | 30 min API | ≥ 25 QC pass |
| M3 | Run B3: retrieval eval | 3 eval runs | 20 min GPU | Graph delta measured |

**Total estimated time: ~1 hour human + 20 min GPU + 30 min API**

---

## Risks and Mitigations

- **Risk**: Element reference extraction from citation context yields few matches (label match rate only 49.8%)
  - **Mitigation**: Accept single-sided matches (only cited side needs element mapping). Fall back to section-level mapping if element-level fails.
- **Risk**: Cross-document queries are actually answerable from a single doc (QC fails)
  - **Mitigation**: LLM ablation QC explicitly tests single-doc answerability. Tighten prompt if pass rate < 30%.
- **Risk**: 50 pairs is too few for statistical significance
  - **Mitigation**: Frame as pilot/proof-of-concept. If method works, scale to all citation edges.

---

## Implementation

### New/Modified Files

| File | Action | Purpose |
|------|--------|---------|
| `scripts/build_cross_doc_citation_pairs.py` | New | B1: extract element-level citation pairs from reference graph |
| `scripts/generate_cross_doc_queries.py` | New or modify existing | B2: cross-document query generation with citation context |
| `scripts/eval_cross_doc_retrieval.py` | New or modify `run_phase0_eval_ab.py` | B3: cross-document retrieval evaluation |

### Key Implementation Notes

1. **Reuse `analyze_latex_graph_topology.py`'s label mapping**: The `_ELEMENT_TO_LABELS` mapping already handles LaTeX label → MinerU element ID conversion
2. **Citation context**: Use the `edge_contexts` field from `latex_reference_graph_v2.json` edges. If empty, use `load_reference_graph_bridge_texts()` to resolve at runtime.
3. **Cross-doc prompt**: Model after `PROMPT_3STEP_REASONING_CHAIN` but replace "bridge paragraph" with "citation context". The citing paper's text IS the bridge.
4. **Evaluation corpus**: Include elements from both cited and citing documents. Use `multimodal_elements_v2.json` enriched fields.
5. **Output format**: Compatible with existing eval scripts (`eval_dense_retrieval.py`, `eval_graph_topk_rerank.py`)
