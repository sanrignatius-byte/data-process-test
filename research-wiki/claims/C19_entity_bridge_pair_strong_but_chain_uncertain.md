---
type: claim
node_id: claim:C19
status: partially_supported
created_at: 2026-05-25
granularity_note: >
  This claim explicitly separates two levels of analysis:
  pair-level (two papers share a concrete entity via their multimodal elements)
  vs chain-level (three papers, two entity bridges, coherent reasoning across all three).
  The pair-level evidence is solid; chain-level is still under judge evaluation.
---

# C19: Entity-bridge pairs are strong at pair level, but chain-level quality is uncertain

## Pair-level evidence (solid)

Entity-bridge pair judge on 83 pairs from the 53-doc survey subset:
- **strong_chain: 21/83 (25.3%)**
- weak_but_related: 53/83 (63.9%)
- topic_only: 9/83 (10.8%)

This is the strongest precision signal among all cross-document pairing methods tested:
- Chunk-bridge: 8.0% strong (300 judged)
- CLIP visual xdoc: ~5% strong (3238 edges, audit-based estimate)
- Citation-based element resolver: 0% strong (120 judged, route closed)
- Linguistics section-level: 46% usable edges but 0% chain usable after cartesian projection

Entity-bridge works because the bridge is a *specific research concept* (e.g. "demographic parity", "g-formula", "structural equation"), not just nearby prose or visual similarity.

## Chain-level evidence (uncertain, under active evaluation)

From the 38 fixed entity-bridge chains (`cross_doc_chains_final_fixed.json`):
- Chains where **both** bridges come from strong entity pairs: **3/38** (from `entity_bridge_chains_improved_v1/summary.json`)
- Chain-level LLM judge on the 38 chains: **2/38 strong_chain** (preliminary, not yet a formal stratified judge pack)

The drop from 25.3% pair-strong → ~5-8% chain-strong is expected: a 3-paper chain needs *two* independent entity bridges to both be strong, and the middle paper must provide a coherent semantic transition.

## What this means for paper claims

- **Safe to claim**: "Entity-bridge pairing achieves 25.3% strong cross-document element pairs, outperforming chunk-bridge (8.0%) and CLIP-visual (~5%)."
- **NOT safe to claim**: "Entity-bridge chains are strong" — chain-level judge not yet run at scale.
- **Conditional claim** (if T2 gate passes): "X% of entity-bridge chains are strong, providing Y high-confidence cross-document reasoning chains for M4."

## Next steps

- T2: Scale entity-bridge to 1147-doc corpus, run formal chain-level judge on 120 chains.
- Update status to `supported` or `falsified` based on T2 gate.

## Connections

- `exp:20260521_m4_enriched_materials` — entity-bridge chains fed into M4 Trinity E2E
- `docs/CROSS_DOC_LONG_CHAIN_REPORT_20260522.md` — corrected chain audit
- `idea:005` — cross-doc multi-hop chain as atomic M4 unit
- `idea:007` — unified M4 trinity benchmark
