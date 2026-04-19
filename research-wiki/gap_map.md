# Gap Map

- `gap:G1` **[Track A]**
  Summary: Dense retrieval on multimodal academic documents still struggles to surface the right evidence early enough. Root cause: 899/1798 passages are raw LaTeX formulas, 842/1798 are image-only passages — both have weak embedding representations. R@100 ceiling is ~20% without VLM captions.
  Priority: high
  Status: partially addressed (graph rerank improves top-k precision/recall; embedding ceiling requires VLM captions — blocked by API budget)

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
