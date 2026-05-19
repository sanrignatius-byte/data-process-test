# Research Wiki Query Pack

_Auto-generated. Do not edit._

## Open Gaps
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
  Summary: Graph value still la
## Recent Relationships (26 total)
  exp:20260417_explicit_rerank_fixed --supports--> claim:C2
  exp:20260418_graph_source_audit --supports--> claim:C4
  exp:20260417_explicit_rerank_fixed --supports--> idea:001
  idea:001 --supports--> claim:C1
  idea:002 --supports--> claim:C3
  idea:003 --supports--> claim:C4
  exp:20260417_dense_baseline_rebuilt --supports--> claim:C1
  claim:C3 --tested_by--> exp:20260418_cross_doc_summary_pending
  idea:005 --addresses_gap--> gap:G8
  idea:006 --addresses_gap--> gap:G9
  idea:007 --extends--> idea:005
  idea:007 --extends--> idea:006
  idea:005 --extends--> idea:004
  idea:005 --supports--> claim:C5
  claim:C12 --tested_by--> exp:20260519_xdoc_pairing_module
  claim:C13 --tested_by--> exp:20260519_chain_to_session
  idea:005 --supports--> claim:C12
  idea:006 --supports--> claim:C13
  idea:007 --supports--> claim:C14
  idea:005 --addresses_gap--> gap:G2
