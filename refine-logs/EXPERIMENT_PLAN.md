# Experiment Plan

**Problem**: The project now needs a reliable way to mass-produce early-stage M2 dual-evidence queries for delivery and retrieval training, without waiting for more complex long-chain or virtual-edge machinery.
**Method Thesis**: Use the existing strict intra-doc pair selection plus `generate_multihop_l1_queries.py` production pipeline to batch-generate initial M2 queries now, then treat candidate supply and lightweight QC as the main scaling bottlenecks.
**Date**: 2026-04-18

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| C1 | We can already mass-produce initial M2 queries with the current repo. | A successful production sweep on the current M2 candidate pools, plus pass-only outputs and packaged delivery files. | B1, B2 |
| C2 | The limiting factor is candidate supply and QC policy, not missing batch infrastructure. | Existing scripts already support resume, shuffling, batching, pass-only subsets, and packaging; remaining unused candidate counts are finite and measurable. | B1, B3 |

## Paper Storyline

- Main paper must prove:
  - The graph or pair-selection pipeline can already produce useful dual-evidence training data at scale.
  - The production path is simple enough to support delivery without relying on complex long-chain generation.
- Appendix can support:
  - Persona and style diversification.
  - Cross-doc and long-chain expansions after M2 production stabilizes.
- Experiments intentionally cut:
  - Broad virtual-edge expansion as a precondition for production.
  - Method C as the main production path.
  - QA-side full validation before initial M2 delivery exists.

## Experiment Blocks

### Block 1: Current Production Readiness Audit
- Claim tested: `C1`, `C2`
- Why this block exists: We need to confirm whether the repository already contains a production-capable M2 path or still needs engineering work first.
- Dataset / split / task:
  - `data/03_queries/M4query_v1/candidates/m2_diverse_candidates_intra_doc.json`
  - `data/03_queries/M4query_v1/candidates/hub_candidates_intra_doc.json`
  - Existing outputs in `data/03_queries/M4query_v1/`, `data/05_eval/m2/`, and `data/03_queries/delivery_v1_2026-04-13.jsonl`
- Compared systems:
  - Current M2 production path
  - Deferred alternatives: Method C, broad virtual-node expansion
- Metrics:
  - Candidate pool size
  - Existing packaged M2 / delivery counts
  - Presence of batch-resume / packaging support
- Setup details:
  - `select_intra_doc_pairs.py` for supply
  - `generate_multihop_l1_queries.py` for generation
  - `run_production_batch.py` / `slurm_scripts/12_production_sweep.sh` for orchestration
  - `build_full_delivery.py` for packaging
- Success criterion:
  - Show that the repo already supports production runs without new core code.
- Failure interpretation:
  - If production infra were missing, we would need to build orchestration first. That is not the case here.
- Table / figure target:
  - Internal planning table only
- Priority: MUST-RUN

### Block 2: Initial M2 Production Sweep
- Claim tested: `C1`
- Why this block exists: This is the first decisive production run that turns “we can probably scale” into concrete query output.
- Dataset / split / task:
  - Primary input: `data/03_queries/M4query_v1/candidates/m2_diverse_candidates_intra_doc.json` (108 pairs)
  - Secondary input: `data/03_queries/M4query_v1/candidates/hub_candidates_intra_doc.json` (96 pairs)
- Compared systems:
  - `academic`
  - `mixed + persona`
- Metrics:
  - Total queries written
  - Pass-only queries
  - Pass rate
  - Pair-type coverage
  - Doc coverage
- Setup details:
  - Script: `generate_multihop_l1_queries.py`
  - Must use `--pass-only`, `--shuffle`, `--skip-done`
  - Prefer company provider with `gpt-5.4`
- Success criterion:
  - Produce a clean new pass-only M2 batch that can be merged into the current delivery namespace.
- Failure interpretation:
  - If pass rate is poor, the next knob is QC simplification or prompt style adjustment, not a method pivot to L3 or Method C.
- Table / figure target:
  - Production summary table in internal report / delivery log
- Priority: MUST-RUN

### Block 3: Lightweight QC and Delivery Packaging Check
- Claim tested: `C2`
- Why this block exists: Production only matters if outputs can be packaged into training data with positives, negatives, qrels, and corpus entries.
- Dataset / split / task:
  - New M2 pass-only outputs
  - `data/03_queries/delivery_v1_2026-04-13.jsonl`
  - `scripts/build_full_delivery.py`
- Compared systems:
  - Current stricter QC path
  - Simplified QC path aligned with latest project requirement
- Metrics:
  - Number of queries accepted into delivery
  - Number of triplets / qrels produced
  - Token-cost pressure from QC
- Setup details:
  - Keep rule QC
  - Relax LLM QC only if it is the real bottleneck
  - Rebuild package after new M2 batch lands
- Success criterion:
  - New M2 queries are incorporated into packaged training artifacts without schema breakage.
- Failure interpretation:
  - If packaging becomes the bottleneck, the next work item is schema or delivery cleanup, not query-generation research.
- Table / figure target:
  - Delivery stats table
- Priority: MUST-RUN

### Block 4: Supply Expansion Only If Needed
- Claim tested: `C2`
- Why this block exists: The current candidate pools are finite, so if the first sweep saturates them we need the cheapest next supply source.
- Dataset / split / task:
  - `select_intra_doc_pairs.py` with pairing module
  - `run_production_batch.py --use-pairing-module`
- Compared systems:
  - Current saved candidate pools
  - Freshly regenerated strict intra-doc pools
- Metrics:
  - Additional unused pair count
  - Quality mix by pair type and hop
- Setup details:
  - Prefer `all` strategy first
  - Raise `--pairing-max-per-doc` only after confirming diversity does not collapse
- Success criterion:
  - Find additional safe M2 supply without touching more complex pipelines.
- Failure interpretation:
  - If new supply is weak, that is the point where summary-based or cross-doc expansion may become necessary.
- Table / figure target:
  - Appendix or internal tracker only
- Priority: NICE-TO-HAVE

### Block 5: Deferred Validation After M2 Production
- Claim tested: downstream relevance, not immediate production readiness
- Why this block exists: QA uplift and cross-doc summary edges still matter, but should not block current M2 scaling.
- Dataset / split / task:
  - QA validation
  - Cross-doc rerank
  - Method C supporting generation
- Compared systems:
  - Current M2 production pipeline
  - Deferred richer pipelines
- Metrics:
  - Retrieval and QA deltas
- Setup details:
  - Run only after initial M2 production is stable
- Success criterion:
  - Supporting evidence for later paper or patent framing
- Failure interpretation:
  - Does not block initial delivery
- Table / figure target:
  - Future paper / appendix
- Priority: NICE-TO-HAVE

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0 | Freeze current production thesis | Audit current candidate pools, outputs, and orchestration scripts | If production path already exists, do not build new infra | Low | Over-planning instead of running |
| M1 | Launch first M2 production sweep | `m2_academic`, `m2_mixed_persona` | If pass-only outputs are healthy, merge and package immediately | Medium API cost | QC too strict reduces usable yield |
| M2 | Merge into delivery package | Rebuild delivery artifacts | If schema holds, M2 production is officially unblocked | Low | Delivery contract drift |
| M3 | Inspect bottleneck | Compare yield vs candidate pool size | If candidate pool is the bottleneck, expand supply; otherwise tune QC | Low | Misdiagnosing QC vs supply |
| M4 | Expand supply if necessary | Pairing-module regeneration or extra strict intra-doc pools | Only after current pools are mostly exhausted | Medium | New pairs may reduce quality |

## Compute and Data Budget

- Total estimated GPU-hours:
  - Not the main constraint; generation is API-bound rather than GPU-bound.
- Data preparation needs:
  - Current saved M2 candidate pools are already available.
  - Current packaged delivery schema is already available.
- Human evaluation needs:
  - Light sample audit only, focused on whether generated queries remain dual-evidence and usable.
- Biggest bottleneck:
  - Candidate supply and QC cost, not missing batch-generation code.

## Risks and Mitigations

- Risk: We confuse “can generate many queries” with “can generate many deliverable queries”.
- Mitigation: Judge production by pass-only outputs and successful delivery packaging, not raw query count.

- Risk: We widen the story into L3, cross-doc, or Method C before stabilizing M2.
- Mitigation: Treat initial M2 as the only production gate right now.

- Risk: Candidate pools are finite and we overestimate near-term scale.
- Mitigation: Use existing saved pools first, then regenerate strict intra-doc pools through the pairing module if needed.

- Risk: LLM QC consumes too much budget.
- Mitigation: Follow the latest requirement and simplify QC once rule checks are stable.

## Final Checklist

- [x] Main production story is compact
- [x] Dominant path is simple and executable
- [x] Nice-to-have runs are separated from must-run runs
- [x] M2 production is treated as the immediate gate
- [x] Method C and broad virtual-edge work are explicitly downgraded for now

