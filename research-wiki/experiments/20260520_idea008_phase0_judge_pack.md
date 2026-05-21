---
type: experiment
node_id: exp:20260520_idea008_phase0_judge_pack
title: "idea:008 Phase 0 judge pack construction for degraded-caption xdoc edges"
date: 2026-05-20
status: constructed_not_judged
lane: experimental
---

# idea:008 Phase 0 judge pack construction

## Purpose

Construct the first auditable input pack for testing whether caption-independent
semantic judgment can promote MinerU+CLIP cross-document recall edges into strong
graph edges under parser-degraded captions (`gap:G10`, `idea:008`).

This is **not** yet a judged result. It is the stable sample/prompt artifact for the
next VLM/LLM/manual audit step.

## Method

Script:
`experiments/construct_idea008_phase0_judge_pack.py`

Inputs:

- Reranked xdoc candidates:
  `data/05_eval/mineru_crossdoc_text_rerank_v1_latest/mineru_crossdoc_text_rerank_edges_v1.jsonl`
- Topology nodes/images:
  `data/05_eval/mineru_topology_graph_v1_latest/mineru_topology_graph_v1.json`

Construction:

- Start from 3238 directed cross-doc visual candidates.
- Canonicalize A->B / B->A duplicates into 3003 undirected pairs.
- Attach topology metadata: image path, caption, local context, page/position, bbox.
- Classify caption quality and caption buckets:
  `clean_text_overlap`, `clean_caption_zero_overlap`, `degraded_one`, `degraded_both`.
- Stratified sample 160 pairs across support tiers:
  `strong_text_supported`, `strong_enriched_supported`, `text_supported_candidate`,
  `weak_text_support`, `visual_only_risky`.
- Render a caption-independent judge prompt per pair. The prompt asks for JSON verdict:
  `strong_edge`, `weak_related`, `visual_layout_only`, `unrelated`, or `insufficient`.

## Output

Latest symlink:
`data/05_eval/idea008_phase0_judge_pack_latest`

Timestamped output:
`data/05_eval/idea008_phase0_judge_pack_20260520T032040Z`

Files:

- `phase0_candidates.jsonl` — full candidate records with prompts and image paths.
- `prompt_batch.jsonl` — minimal API-ready prompt records.
- `phase0_candidates.csv` — spreadsheet-friendly audit index.
- `summary.json` — construction metadata and counts.
- `report.md` — human-readable summary and prompt samples.

## Counts

- Raw directed edges: 3238
- Deduped pairs: 3003
- Sample size: 160
- Missing image pairs: 0

Sample tier counts:

- `strong_text_supported`: 36
- `strong_enriched_supported`: 24
- `text_supported_candidate`: 36
- `weak_text_support`: 40
- `visual_only_risky`: 24

Sample caption buckets:

- `clean_text_overlap`: 38
- `clean_caption_zero_overlap`: 43
- `degraded_one`: 42
- `degraded_both`: 37

## Next Step

Run an actual judge on `prompt_batch.jsonl`, attaching both images for each candidate.
The key measurement is whether the judged `strong_edge` rate rises above the current
~5% text-supported edge share, especially in `clean_caption_zero_overlap`,
`degraded_one`, and `degraded_both` buckets, without inflating `visual_layout_only`
false positives.

If positive, only judged-strong edges should be promoted into cross-doc chain
construction. If negative, keep xdoc visual edges as recall candidates only.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
