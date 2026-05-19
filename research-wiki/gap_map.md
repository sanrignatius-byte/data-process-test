# Gap Map

- `gap:G1` **[Track A]**
  Summary: Dense retrieval on multimodal academic documents still struggles to surface the right evidence early enough. Root cause: 899/1798 passages are raw LaTeX formulas, 842/1798 are image-only passages — both have weak embedding representations. R@100 ceiling is ~20% without VLM captions.
  Priority: high
  Status: partially addressed (graph rerank improves top-k precision/recall; split-modality retrieval experiment running (Job 66036) to test per-modality indexes)

- `gap:G2` **[Track A]**
  Summary: Virtual edges are mostly intra-document. Typed cross-doc element edges (C5) partially solve cross-doc retrieval (R@10 +1.5pp), but only at element granularity; section-level and paragraph-level cross-doc connectivity is still weak.
  Priority: high
  Status: partially addressed (C5 validated; para merge + bbl expansion in R104–R108 pending)

- `gap:G3` **[Track B]**
  Summary: Need 500+ deliverable SFT queries with qrels, positives, negatives, and corpus entries packaged for training use.
  Priority: high
  Status: active — 556 pass queries in inventory; delivery packaging (P001) pending

- `gap:G4` **[Track A]**
  Summary: Graph value still lacks direct QA validation. Retrieval-side gains are proven (C1, C5), but no end-to-end QA experiment has been run.
  Priority: medium
  Status: unresolved (blocked on C-Pool 78q qrels decision — mentor 4.20)

- `gap:G5` **[Track A/B]**
  Summary: The repository mixes parsing, graphing, generation, QC, and evaluation in one place, increasing maintenance burden and context cost.
  Priority: low
  Status: acknowledged

- `gap:G6` **[Track A]**
  Summary: Chunk as retrieval unit has two structural disadvantages vs element: (1) 75% of dual-evidence queries need 2+ distinct chunks, so R@1 inherently covers only one element; (2) chunk corpus is sparser (964 vs 1798), reducing dense recall ceiling. Chunk may be better positioned as a downstream consumption unit rather than primary retrieval unit.
  Priority: medium
  Status: quantified (2026-05-02 chunk→element analysis); split-modality retrieval experiment running (Job 66036) to test whether per-modality indexes can close the gap.

- `gap:G7` **[Track A]**
  Summary: Modality-style mismatch — M4query_v1 queries are paper-domain text-style ("RoBERTa pretraining objective"), but figure/table/formula passages either (a) use placeholder text `[Image:]` mostly recovered via graph context, or (b) under MODORA enrichment carry visual-only descriptions that are domain-detached. Both BGE-CE rerank (`exp:20260503_ce_rerank_bge`) and corpus-side MODORA injection (`exp:20260503_corpus_enrich_fix`, claim C8) regress dense R@10 on this mismatch. The 0.6913 R@10 ceiling is held by graph rerank propagating from text-paragraph anchors; non-text passages are essentially passive (only −0.5pp shift when corpus quality changes ±10pp).
  Priority: high
  Status: identified 2026-05-03; next moves are non-text-biased reranker family swap, late-fusion VL lane, or HyDE query rewriting (paper-style query → match paper-style passage). Corpus-text replacement strategies are antipattern (C8).

- `gap:G8` **[Track A — M4 trinity]**
  Summary: No element-level cross-doc bridge layer exists as a chain-ready artifact. We have 123 doc-level citation edges (`citation_graph.json`) and 11800 reranked Qwen3-Embedding-4B element-to-element matches (`mineru_crossdoc_embedding_matches_..._v2b_cap10.jsonl`), but no pairing module that turns the latter into the `hub_candidates_enriched_v3.json` schema that downstream query generators consume. Without this, multi-hop chains stay intra-document and Phase 2 (multi-doc) of M4 is structurally blocked.
  Priority: high
  Status: addressed-in-design (idea:005, exp:20260519_xdoc_pairing_module). Concrete next step: mirror `src/pairing/intra_doc_pairs.py` → `cross_doc_pairs.py`.

- `gap:G9` **[Track A — M4 trinity]**
  Summary: Multi-turn session generator (`generate_multiturn_sessions.py` v2) generates sessions from scratch with persona + intent-shift prompts. It does not consume existing L3 chains — meaning every session is a fresh LLM call duplicating reasoning the chain already proved. Also: no canonical "delete turn N" QC operation, because sessions aren't structurally tied to a chain's hops.
  Priority: high
  Status: addressed-in-design (idea:006, exp:20260519_chain_to_session). Concrete next step: write chain-to-session projection rule that uses L3 chain's `reasoning_steps[]` as the structural skeleton.
