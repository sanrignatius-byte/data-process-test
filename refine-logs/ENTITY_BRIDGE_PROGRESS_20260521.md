# Entity-Bridge Cross-Document Chain Construction — Progress 2026-05-21

## What we did

Flipped the cross-doc element linking approach from bottom-up (resolver v1: "Figure N matches Figure N") to top-down (entity-bridge: papers share research entities → elements linked through those entities).

## Key result: 0% → 40% strong_chain

| | resolver v1 | entity-bridge |
|---|---|---|
| strong_chain | 0/120 (0%) | **12/30 (40%)** |
| wrong_target | 95 | **0** |
| cost | $1.11 | $0.19 |

## Scripts created

1. **`experiments/build_entity_bridge_candidates.py`** — Build cross-doc element pairs linked by shared enriched keywords. Works on all 76 docs. IDF-weighted keyword matching, visual pattern filtering, stratified output.

2. **`experiments/judge_entity_bridge_pack.py`** — Judge entity-bridge candidates via company VLM. Entity-aware prompt that asks: "Do these two elements both specifically discuss the same research entity?"

3. **`experiments/build_entity_bridge_chains_53.py`** — Build cross-document long chains for the 53-paper (old_53) subset. Constructs 1-hop (2-paper) and 2-hop (3-paper) chains by connecting entity-bridge pairs through shared papers.

## Output artifacts

- `data/05_eval/entity_bridge_candidates_latest/judge_pack.jsonl` — 72 candidates (76 docs)
- `data/05_eval/entity_bridge_judge_20260521T113000Z/` — 30-candidate judge results (40% strong_chain)
- `data/05_eval/entity_bridge_chains_53_20260521T115104Z/` — 53 entity-bridge pairs + 115 chains (45 1-hop, 70 2-hop)

## Chain diversity (53-paper subset)

- 33/53 papers connected in chains
- Heavy concentration: causal fairness (72) + coreference/NLP (54)
- 20 papers not connected at all (keyword coverage gap)
- Hub papers dominate: 1802.08139 (30 chains), 1907.06430 (29)

## Root cause & next direction

Element-level keyword matching is too sparse (2-5 keywords per element). Only keyword-dense papers form bridges.

**Next: chunk-level bridge construction.**
Instead of matching element keywords, match paragraph/chunk text across papers:
- Each paragraph in the topology graph has `content_preview`, `context_before`, `context_after`
- TF-IDF or embedding similarity between chunks → find paragraph pairs that discuss the same concept
- The matching paragraph text IS the bridge text
- Elements referenced in those paragraphs become the chain elements

This should:
1. Cover more papers (every paper has paragraphs, not just elements)
2. Provide richer bridge text (full sentences, not just keyword lists)
3. Be judgeable (bridge text can be fed to VLM for quality assessment)

## Files to look at

- `/projects/myyyx1/data-process-test/data/05_eval/mineru_topology_graph_v1_latest/graph.json` — paragraph nodes with text
- `/projects/myyyx1/data-process-test/data/02_enriched/multimodal_elements_enriched.json` — enriched element keywords
- `/projects/myyyx1/data-process-test/experiments/build_mineru_topology_graph.py` — how topology graph is built
