# Cross-Document Long-Chain Report

Date: 2026-05-22

This report summarizes the current `data-process-test` cross-document multimodal chain work from the recent Claude Code sessions, the corrected artifacts produced during handoff, and my independent assessment.

## Scope

This report is about the M4 / cross-document evidence-chain data line under `data-process-test`. It is separate from the dissertation SVG-harness direction recorded in memory files; that dissertation track remains a different project line.

The immediate goal here is to build cross-document multimodal evidence chains from the 53-paper survey subset, so that later query generation or retrieval evaluation can use chains grounded in concrete figure/table/formula elements.

## Research Route

### Phase 1: Chunk-Bridge Baseline

The chunk-bridge route connected papers through chunk-level text/context. It produced many broad related pairs, but precision was weak.

Judged sample:

| Method | Judged | strong_chain | weak_but_related | usable |
| --- | ---: | ---: | ---: | ---: |
| Chunk-bridge v2 | 300 | 24 (8.0%) | 156 | 60.0% |

Main failure modes:

- `wrong_target`: 36
- `wrong_source`: 34
- `insufficient_context`: 25
- `topic_only`: 24

Interpretation: chunk-level evidence finds topical neighborhood, but it often does not pin the source and target elements to the same concrete scientific object. It is useful as broad recall, not as the primary source of high-confidence training chains.

Artifacts:

- `data/05_eval/chunk_bridge_judge_v2/summary.json`
- `data/05_eval/chunk_bridge_judge_v2/judgments.jsonl`

### Phase 2: Entity-Bridge Pairing

The entity-bridge route uses enriched multimodal elements and high-IDF shared entities. This gives much better precision because the bridge is a specific research concept such as `demographic parity`, `winobias`, `g-formula`, or `structural equation`, not just nearby prose.

Pair-level judged sample:

| Method | Judged | strong_chain | weak_but_related | topic_only |
| --- | ---: | ---: | ---: | ---: |
| Entity-bridge v2 | 83 | 21 (25.3%) | 53 | 9 |

Interpretation: entity-bridge is clearly better than chunk-bridge for precision. It still should not be treated as automatically strong: many pairs are real but only weakly related, especially generic visual or method terms.

Artifacts:

- `data/05_eval/entity_bridge_candidates_v2/judge_pack.jsonl`
- `data/05_eval/entity_bridge_judge_v2/summary.json`
- `data/05_eval/entity_bridge_judge_v2/judgments.jsonl`

### Phase 3: Entity-Bridge Chains

The next step connected entity-bridge pairs into cross-document chains. The natural unit is:

- 3 papers
- 2 cross-document entity bridges
- 3 or 4 unique multimodal elements

Important correction: the earlier session described this as "3 elements, 3 papers". That was sometimes false. If the middle paper uses one element for bridge A-B and another element for bridge B-C, the chain has 4 unique elements. The fixed pipeline now preserves both middle-paper elements.

## Bugs Found And Fixed

### 1. Undirected Graph Orientation Bug

The paper graph was treated as undirected, but element ids were not swapped when traversing an edge in reverse. This produced invalid records such as:

```text
doc_id = 1805.05859
element_id = 1907.06430_formula_8
```

Impact:

- Old `entity_bridge_chains_53_20260521T115104Z/chains.jsonl`: 50 bad elements across 50 chains.
- Old `cross_doc_chains_final.json`: 37 bad elements across 37 chains.
- Old `cross_doc_long_chains_clean.json`: 7 bad elements.
- Old `cross_doc_long_chains_v3.json`: 7 bad elements.

Fix:

- Added `orient_bridge_pair()` in `experiments/build_entity_bridge_chains_53.py`.
- Regenerated fixed entity chains under `data/05_eval/entity_bridge_chains_53_fixed_20260522T0910Z/`.

### 2. Missing Middle-Bridge Endpoint

The old renderer only kept the target element of the first hop as the middle-paper element. When the second hop started from a different middle-paper element, that element was silently dropped.

Fix:

- `render_chain_material()` now preserves every unique bridge endpoint.
- Bridge records now include `from_element_id` and `to_element_id`.

### 3. Finalization Was Ad Hoc

The old final set was produced by inline code in the session. It was not easily reproducible.

Fix:

- Added `experiments/finalize_cross_doc_chains.py`.
- It filters generic visual entities, keeps 2-hop chains, deduplicates by full element set, and audits doc/element consistency.

## Current Corrected Results

### Fixed Raw Entity Chains

Source:

- `data/05_eval/entity_bridge_chains_53_fixed_20260522T0910Z/chains.jsonl`
- `data/05_eval/entity_bridge_chains_53_fixed_20260522T0910Z/summary.json`

Stats:

| Metric | Value |
| --- | ---: |
| Entity-bridge pairs | 53 |
| Unique paper pairs | 31 |
| Papers in entity graph | 33 / 53 |
| Total chains | 115 |
| 1-hop chains | 45 |
| 2-hop chains | 70 |
| Doc/element mismatches | 0 |

### Final Fixed Natural Chains

Primary output:

- `data/05_eval/cross_doc_chains_final_fixed.json`
- `data/05_eval/cross_doc_chains_final_fixed.jsonl`
- `data/05_eval/cross_doc_chains_final_fixed_audit.json`

Stats:

| Metric | Value |
| --- | ---: |
| Final chains | 38 |
| Papers covered | 18 |
| Papers per chain | 3 for all 38 |
| Cross-doc bridges | 76 |
| Doc/element mismatches | 0 |
| Element length distribution | 13 chains with 3 elements; 25 chains with 4 elements |

Element types:

| Type | Count |
| --- | ---: |
| formula | 56 |
| figure | 52 |
| table | 31 |

Top shared entities:

| Entity | Count |
| --- | ---: |
| linear model | 14 |
| structural equation | 11 |
| winobias | 11 |
| coreference resolution | 11 |
| counterfactual | 8 |
| ontonotes | 7 |
| causal dag | 7 |
| outcome | 6 |
| predictor | 6 |
| adversarial training | 6 |

### Fixed Topology-Extended Long Chains

Secondary output:

- `data/05_eval/cross_doc_long_chains_v3_fixed.json`
- `data/05_eval/cross_doc_long_chains_v3_fixed.jsonl`

Stats:

| Metric | Value |
| --- | ---: |
| Chains | 49 |
| Papers covered | 21 |
| Papers per chain | 3 for all 49 |
| Elements per chain | 5 or 6 |
| Cross-doc bridges | 98 |
| Intra-doc topology bridges | 98 |
| Doc/element mismatches | 0 |

Element types:

| Type | Count |
| --- | ---: |
| formula | 95 |
| figure | 149 |
| table | 39 |

Interpretation: the topology-extended set is useful as augmented context, but it should not replace the natural entity-chain set as the primary semantic label source. The extra intra-doc elements increase length, but not necessarily cross-document semantic strength.

## Why The Count Is Lower Than Expected

The user's intuition is right: 53 papers from the same survey should be connected. The corrected result says something more precise:

They are connected at the broad topic and citation-survey level, but high-confidence multimodal evidence chains are much sparser when we require concrete shared entities between extracted figure/table/formula elements.

The main bottlenecks are:

1. Entity specificity: generic visual entities such as `overlap`, `distribution comparison`, or `point cloud` create weak links and are filtered out.
2. Multimodal element sparsity: not every paper has enriched figure/table/formula elements around the same scientific entity.
3. Strict 2-hop chain semantics: a chain needs two valid cross-document entity bridges, not just one shared topic.
4. Deduplication: many paths are reverse/permutation variants of the same element set. After preserving all bridge endpoints, 29 duplicates collapse.
5. The >=5 element requirement is not the natural semantic unit. It is a topology-augmentation requirement. The natural unit is 3 papers, 2 cross-document bridges, and 3-4 unique elements.

## Independent Assessment

My view: the corrected entity-bridge route is the right backbone, but the dataset is not yet ready to call "all strong chains".

What is strong:

- Causal/fairness formula and DAG chains: `g-formula`, `structural equation`, `counterfactual`, `path-specific fairness`, `mediation`.
- Coreference/WinoBias chains: `winobias`, `ontonotes`, `coreference resolution`, gender-bias tables.
- Some model-family chains: `adversarial training`, `predictor`, `discriminators`, `invariance`.

What is weaker:

- Embedding/clustering links can be real but often compare different aspects of the same broad representation idea.
- Generic classifier links such as random forest / logistic regression are often related but not strong unless they share the exact benchmark or fairness criterion.
- Visual-pattern links should be mostly excluded.

The corrected final set of 38 chains is a defensible high-precision candidate set, not a finished gold set. The pair-level judge result gives evidence that entity-bridge is much better than chunk-bridge, but chain-level judgment has not yet been run on `cross_doc_chains_final_fixed.json`.

## Recommended Next Step

Run a chain-level judge on the 38 fixed natural chains. The judge prompt should inspect both bridge endpoints and decide:

- `strong_chain`: both bridges are specific and the middle paper gives a coherent semantic transition.
- `weak_but_related`: the bridges are real but loose or the two hops do not form one coherent reasoning chain.
- `topic_only`: generic topic/visual overlap.
- `invalid`: extraction or element mismatch.

After that:

1. Use `strong_chain` as positive training/eval examples.
2. Use selected `weak_but_related` as hard or medium examples.
3. Keep `cross_doc_long_chains_v3_fixed.json` as extra context expansion, not as the canonical label set.
4. If higher coverage is needed, expand recall through survey taxonomy/citation context, but keep entity-specific filtering as the precision layer.

## Files Changed During Handoff

Code:

- `experiments/build_entity_bridge_chains_53.py`
  - fixed undirected pair orientation
  - preserved all bridge endpoint elements
  - added bridge-level element ids
- `experiments/build_cross_doc_long_chains.py`
  - fixed indentation in sample printer
  - pointed default entity-chain input to fixed chain directory
- `experiments/finalize_cross_doc_chains.py`
  - new reproducible finalization and audit script

Data outputs:

- `data/05_eval/entity_bridge_chains_53_fixed_20260522T0910Z/chains.jsonl`
- `data/05_eval/entity_bridge_chains_53_fixed_20260522T0910Z/summary.json`
- `data/05_eval/cross_doc_chains_final_fixed.json`
- `data/05_eval/cross_doc_chains_final_fixed.jsonl`
- `data/05_eval/cross_doc_chains_final_fixed_audit.json`
- `data/05_eval/cross_doc_long_chains_v3_fixed.json`
- `data/05_eval/cross_doc_long_chains_v3_fixed.jsonl`

## Bottom Line

The earlier conclusion "55 clean 3-element chains" should be replaced.

Corrected conclusion:

> The fixed pipeline produces 38 unique high-precision candidate chains, each spanning 3 papers with 2 cross-document entity bridges and 3-4 unique multimodal elements. A topology-extended variant produces 49 chains with 5-6 elements. The natural semantic unit is the 3-paper / 2-bridge entity chain; the >=5 element version is useful as augmented context but should not be treated as inherently more meaningful.
