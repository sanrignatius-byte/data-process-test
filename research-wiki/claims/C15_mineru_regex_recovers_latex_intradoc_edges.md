---
type: claim
node_id: claim:C15
status: supported
created_at: 2026-05-20T00:00:00Z
updated_at: 2026-05-20T00:00:00Z
---

# Claim

Pure-MinerU intra-document edges can replace the LaTeX `\ref` hard-reference graph.
On documents parsed both ways, MinerU's `regex_reference` layer recovers the large
majority of LaTeX explicit reference edges, so the non-LaTeX PDF pipeline does not
lose the intra-doc relational backbone.

## Evidence

- `exp:20260520_mineru_clip_xdoc_pipeline`, `audit_latex_vs_mineru_intradoc.py`.
- 52/53 MinerU docs overlap with raw `.tex` in `latex_sections_rebuild_2026-03-24`.
  (The prior "overlap=0" was only against the *other corpus's* `latex_reference_graph_v2`.)
- **Figure/table extraction recall 90.8%** (464/511 LaTeX visual labels caption-matched).
- **Reference recall 84.0%** (326/388 LaTeX-`\ref`'d figures also linked by MinerU
  `regex_reference`); per-doc median 1.0; 26/52 docs at 100%.
- Manual sample of `regex_reference`: 6/6 correct (eq.(2)→formula, Figure 1→Figure 1, …).
- Output: `data/05_eval/latex_vs_mineru_intradoc_latest/`.

## Scope

53-doc arXiv corpus (52 with `.tex`). Two low outliers (1607.06520 extraction 0.455,
1709.02012 reference 0.462) are caption-text mismatch in cross-parse alignment, not
MinerU parse failures. Holds for the figure/table reference relation; not evaluated for
prose-only relations.

## Why It Matters

Removes the main risk in the PDF-first migration: the intra-doc hard graph survives the
loss of LaTeX `\ref`/`\label`. Strong edges for the graph should come from
`regex_reference` + structural edges, not from CLIP soft edges.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
