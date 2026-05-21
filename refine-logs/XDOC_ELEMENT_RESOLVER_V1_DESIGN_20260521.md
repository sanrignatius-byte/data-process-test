# Cross-Doc Element Resolver v1 Design

**Date**: 2026-05-21  
**Project**: `/projects/myyyx1/data-process-test`  
**Owner for next execution**: next assistant / Track A experimental lane  
**Status**: design ready, not executed  

## Executive Summary

v0 proved the full pipeline is runnable:

`G11 citation filter -> xdoc element resolver -> schema-compatible pairs -> query-generator dry run`

The current v0 artifact is:

- `data/05_eval/xdoc_element_resolver_v0_20260521T015847Z/`
- `data/05_eval/xdoc_element_resolver_v0_latest/`

v0 generated 5,000 cross-modal candidate pairs from 34,447 filtered C18 citation edges, but all target resolution used lexical caption/content overlap. The next version must answer the precision question, not just produce more candidates.

The v1 goal is therefore narrow:

1. Add target-side explicit numbered element resolution: `Figure/Table/Eq N` in a citing chunk should resolve directly to the target document's numbered element when the evidence is target-anchored.
2. Penalize noisy source-side nearest-position fallback in high-fanout citation chunks.
3. Validate against the existing cross-doc L3 pass set before any manual/LLM judging.
4. Produce a stratified judge pack only after the above instrumentation is in place.

Do not promote anything to `src/pairing/cross_doc_pairs.py` in this execution. Keep v1 under `experiments/` and `data/05_eval/`.

## Current Baseline

### G11 Filter

Input:

- `data/04_xdoc_citation/predicted_xdoc_edges_chunks.jsonl`

Output:

- `data/04_xdoc_citation/predicted_xdoc_edges_chunks_filtered.jsonl`
- `data/04_xdoc_citation/predicted_xdoc_edges_chunks_filtered_stats.json`

Observed counts:

| Metric | Count |
|---|---:|
| Raw C18 chunk edges | 53,435 |
| Kept body structural citation edges | 34,447 |
| Dropped references | 11,582 |
| Dropped semantic-without-citation | 3,877 |
| Dropped author-list | 2,540 |
| Dropped noisy section | 989 |

### Resolver v0

Code:

- `experiments/build_xdoc_element_resolver_v0.py`

Output:

- `data/05_eval/xdoc_element_resolver_v0_20260521T015847Z/cross_doc_pairs_v0.json`
- `data/05_eval/xdoc_element_resolver_v0_20260521T015847Z/summary.json`
- `data/05_eval/xdoc_element_resolver_v0_20260521T015847Z/report.md`

Observed counts:

| Metric | Count |
|---|---:|
| Filtered citation edges | 34,447 |
| Citation chunks with source+target elements | 24,246 |
| Retained pairs | 5,000 |
| `figure+table` | 2,898 |
| `figure+formula` | 1,495 |
| `formula+table` | 607 |
| Source explicit refs | 494 |
| Source nearest-position fallback | 4,506 |
| Target explicit refs | 0 |
| Target caption overlap | 5,000 |

Dry-run gate:

- `generate_multihop_l1_queries.py --allow-cross-doc-candidates --dry-run --limit 20 --no-images --skip-llm-qc`
- Result: 20/20 prompts rendered.

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Blocks |
|---|---|---|---|
| C1: Target explicit numbered references improve precision. | v0's largest weakness is target lexical overlap. Explicit target-side `Figure/Table/Eq N` is the highest-precision signal available in pure MinerU text. | v1 emits a nonzero explicit-target tier; explicit-target sample precision is at least 0.75, or L3 exact target recovery improves over v0. | B1, B3, B4 |
| C2: Citation-backed resolver can recover paragraph-mediated cross-doc L3 structure better than visual-similarity edges. | This is the central viability test for M4 long chains from pure MinerU. | On the cross-doc L3 pass gold set, v1 has higher endpoint/doc-pair recovery than v0 and produces auditable miss reasons. | B2, B3 |
| Anti-claim: v1 is just producing topical near-neighbors. | A topical resolver would pass token overlap but fail target-specific L3 recovery and human judging. | Stratified judge pack separates explicit-target, overlap-high, overlap-low, and high-fanout cases; noisy strata are not allowed to dominate the final artifact. | B4 |

## Experiment Blocks

### B1: Target Explicit Numbered Resolver

- **Claim tested**: C1.
- **Why this block exists**: v0 has zero target explicit matches despite citing chunks containing patterns such as `Table 2`, `Fig. 1`, and `Eq. (3)`.
- **Input**:
  - `data/04_xdoc_citation/predicted_xdoc_edges_chunks_filtered.jsonl`
  - `data/01_graphs/multimodal_elements_v2.json`
  - `data/01_graphs/chunk_virtual_nodes_v2.json`
  - Optional title aliases from `data/01_graphs/latex_reference_graph_v2.json`
- **Implementation file**:
  - Create `experiments/build_xdoc_element_resolver_v1.py`.
  - Copy v0 as the starting point; do not mutate v0 behavior except for shared bug fixes.
- **Core design**:
  - Extract numbered refs with local windows:
    - `Figure 2`, `Fig. 2`, `Table 4`, `Eq. (3)`, `Equation 3`.
    - Store `element_type`, `number`, `span`, and `window_before/window_after`.
  - Build target element lookup:
    - `(target_doc, element_type, number) -> element`.
  - Add target resolution methods:
    - `target_explicit_number_anchored`
    - `target_explicit_number_unanchored`
    - `target_caption_overlap`
  - Treat explicit number as anchored if any of these is true:
    - the ref window contains target title words from `latex_reference_graph_v2`;
    - the edge has `features.title_match >= 0.2`;
    - the chunk mentions only one numbered ref and the C18 edge probability is `>=0.95`;
    - the chunk's citation fanout is `<=2`.
  - Treat explicit number as unanchored if the number exists in the target doc but the chunk has high fanout or no target title anchor.
  - Ranking priority:
    - anchored explicit number first;
    - unanchored explicit number second, with lower confidence;
    - lexical overlap fallback third.
- **Scoring changes**:
  - Target score:
    - anchored explicit: `1.0`
    - unanchored explicit: `0.70`
    - overlap: current lexical score, requiring `min_target_score` and `min_overlap_terms`
  - Quality score:
    - keep the same three terms, but use `target_resolution_score` from the new target method.
- **Success criterion**:
  - At least 300 retained pairs use explicit target number, or explain why coverage is genuinely lower.
  - Anchored explicit target cases have no obvious schema/path failures in a 20-case spot check.
- **Failure interpretation**:
  - If explicit target coverage is near zero, source chunks rarely name target elements explicitly; continue with lexical fallback but do not claim target-specific precision from numbered refs.
- **Output**:
  - `data/05_eval/xdoc_element_resolver_v1_<stamp>/cross_doc_pairs_v1.json`
  - `data/05_eval/xdoc_element_resolver_v1_<stamp>/cross_doc_pairs_v1.jsonl`
  - `data/05_eval/xdoc_element_resolver_v1_<stamp>/summary.json`
  - `data/05_eval/xdoc_element_resolver_v1_<stamp>/report.md`

### B2: Source Nearest-Position Fanout Penalty

- **Claim tested**: Anti-claim.
- **Why this block exists**: v0 uses `source_nearest_position` for 4,506/5,000 retained pairs. This is weak in related-work chunks that cite many papers.
- **Input**:
  - Same filtered C18 edges.
- **Implementation**:
  - Precompute `citation_fanout_by_chunk_id`: number of filtered citation edges per source chunk.
  - Add `citation_fanout` to each pair's `hub_metadata`.
  - Source score rules:
    - `source_explicit_ref`: keep `1.0`.
    - `source_nearest_position`: multiply by a fanout penalty.
  - Suggested penalty:
    - `fanout <= 2`: `1.0`
    - `3 <= fanout <= 5`: `0.75`
    - `6 <= fanout <= 10`: `0.55`
    - `fanout > 10`: `0.35`
  - Add an optional CLI flag:
    - `--max-citation-fanout 0` means no hard cutoff.
    - For the primary v1 artifact, use no hard cutoff but record fanout.
- **Success criterion**:
  - Top 5,000 are less dominated by broad survey chunks than v0.
  - `summary.json` reports fanout buckets and source method counts.
- **Failure interpretation**:
  - If useful explicit cases are over-penalized, keep fanout as metadata but do not use it in ranking.

### B3: Cross-Doc L3 Pass Recovery Evaluation

- **Claim tested**: C2.
- **Why this block exists**: Existing L3 pass rows are the closest free gold signal. They already encode paragraph-mediated cross-doc endpoint chains.
- **Gold source**:
  - Use the known audit target: 146 L3 pass rows, 87 cross-doc.
  - Expected clues from wiki:
    - `reasoning_steps=[]` for all rows;
    - path length distribution `{3: 48, 4: 98}`;
    - 87/146 paths are cross-doc;
    - `element_ids` always has two endpoint elements.
- **Implementation file**:
  - Either add evaluation functions into `experiments/build_xdoc_element_resolver_v1.py`, or create:
    - `experiments/evaluate_xdoc_resolver_l3_recovery.py`
- **Gold discovery**:
  - Add CLI args:
    - `--l3-input` can be specified multiple times.
    - `--l3-glob` defaults to:
      - `data/03_queries/**/*.jsonl`
      - `archive/data/batch_phase2a/**/*.jsonl`
      - `data/05_eval/m2/*reasoning*json*`
  - Parse only rows that have:
    - `qc_pass` true or file name includes `_pass`;
    - exactly two endpoint `element_ids`;
    - endpoint docs differ OR `path`/`graph_path` contains multiple doc ids;
    - `difficulty_label` or `query_id` suggests L3/reasoning when available.
  - Save discovered gold to:
    - `l3_crossdoc_gold.jsonl`
  - Include a `gold_discovery_report.json` with counts and sample rejected reasons.
- **Recovery metrics**:
  - `gold_count`: number of cross-doc gold rows.
  - `doc_pair_recall@K`: gold source/target doc pair appears in v1 top K.
  - `endpoint_pair_recall@K`: exact unordered endpoint pair appears in v1 top K.
  - `target_endpoint_recall@K`: for a matching doc pair/source endpoint, target element recovered.
  - `source_endpoint_recall@K`: source element recovered.
  - `method_breakdown`: recovery by `target_resolution_method`, `source_resolution_method`, and fanout bucket.
  - K values: 100, 500, 1000, 5000.
- **Comparison**:
  - Evaluate v0 artifact:
    - `data/05_eval/xdoc_element_resolver_v0_latest/cross_doc_pairs_v0.json`
  - Evaluate v1 artifact:
    - `data/05_eval/xdoc_element_resolver_v1_latest/cross_doc_pairs_v1.json`
- **Success criterion**:
  - v1 endpoint-pair recall@5000 is higher than v0.
  - v1 target endpoint recall improves in explicit-number strata.
  - The script prints concrete miss reasons instead of only aggregate failure.
- **Failure interpretation**:
  - If both v0 and v1 recover few gold pairs, the L3 gold may be from a different corpus/schema slice; record ID mismatch statistics before concluding the resolver failed.

### B4: Stratified Manual/LLM Judge Pack

- **Claim tested**: C1 and Anti-claim.
- **Why this block exists**: L3 recovery may be sparse. A stratified precision estimate is needed before production promotion.
- **Implementation**:
  - Create `experiments/build_xdoc_resolver_judge_pack.py` or add a `--write-judge-pack` option to v1 builder.
  - Sample 100 pairs, balanced by:
    - `target_resolution_method`:
      - 25 anchored explicit
      - 25 unanchored explicit
      - 25 overlap score `>=0.20`
      - 25 overlap score `0.12-0.20`
    - If a stratum has fewer examples, redistribute to the closest stricter stratum.
  - Also balance across pair types where possible:
    - `figure+table`
    - `figure+formula`
    - `formula+table`
  - Include fanout buckets:
    - low `<=2`
    - medium `3-5`
    - high `>5`
- **Judge item schema**:
  - `candidate_id`
  - `source_doc`, `target_doc`
  - `source_element_id`, `target_element_id`
  - `source_caption_or_content`
  - `target_caption_or_content`
  - `citation_bridge_text`
  - `section_title`
  - `citation_probability`
  - `source_resolution_method`
  - `target_resolution_method`
  - `target_resolution_score`
  - `citation_fanout`
  - `question_for_judge`
- **Judge rubric**:
  - `valid_target_element`: target element is specifically discussed or strongly implied by the citation bridge.
  - `valid_source_anchor`: source element is a plausible local anchor for the citing chunk.
  - `valid_chain`: source element -> citation bridge -> target element forms a coherent M4 chain.
  - Verdict:
    - `strong_chain`
    - `weak_but_related`
    - `topic_only`
    - `wrong_target`
    - `wrong_source`
    - `insufficient_context`
- **Primary precision metric**:
  - Count only `strong_chain` as pass.
  - Report pass rate by stratum.
- **Success criterion**:
  - anchored explicit target precision `>=0.75`;
  - overall top-strata precision `>=0.55`;
  - high-fanout nearest-position cases are visibly lower precision, validating the penalty.
- **Failure interpretation**:
  - If explicit-target is strong but lexical overlap is weak, promote only explicit-target v1 or use overlap only as recall pool for LLM judge.
  - If both are weak, do not promote; revisit resolver with embeddings or title-aware rerank.

### B5: Query Generator Smoke After v1

- **Claim tested**: pipeline compatibility.
- **Why this block exists**: v0 required adding `--allow-cross-doc-candidates`; v1 must continue to render.
- **Command**:

```bash
python3 scripts/generate_multihop_l1_queries.py \
  --candidates data/05_eval/xdoc_element_resolver_v1_latest/cross_doc_pairs_v1.json \
  --output data/05_eval/xdoc_element_resolver_v1_latest/dryrun_queries.jsonl \
  --limit 20 \
  --dry-run \
  --no-images \
  --skip-llm-qc \
  --allow-cross-doc-candidates
```

- **Success criterion**:
  - 20/20 prompts render.
  - No missing field exceptions.
  - Prompt preview includes citation bridge context.

## Required Code Changes

### 1. Create v1 Builder

File:

- `experiments/build_xdoc_element_resolver_v1.py`

Start from:

- `experiments/build_xdoc_element_resolver_v0.py`

Add functions:

```python
def load_title_aliases(latex_reference_graph_v2_path: Path) -> dict[str, set[str]]:
    """Return doc_id -> normalized title/token aliases."""

def extract_numbered_refs_with_windows(text: str, window_chars: int = 120) -> list[dict]:
    """Return type, number, span, and local text window for Figure/Table/Eq refs."""

def is_target_anchored_ref(ref: dict, edge: dict, target_aliases: set[str], fanout: int) -> bool:
    """Decide whether an explicit numbered ref likely points to edge.target_doc."""

def rank_target_elements_v1(edge, target_doc_info, target_aliases, fanout, ...):
    """Return explicit-number candidates first, then lexical fallback."""

def compute_citation_fanout(edges: Iterable[dict]) -> dict[str, int]:
    """Count filtered citation edges per source chunk."""

def source_fanout_penalty(fanout: int) -> float:
    """Return source nearest-position penalty."""
```

Keep v0 output schema fields and add metadata:

```json
{
  "target_resolution_method": "target_explicit_number_anchored",
  "target_ref_text": "Table 2",
  "target_ref_window": "...",
  "target_anchor_reason": "title_match>=0.2",
  "citation_fanout": 4,
  "source_fanout_penalty": 0.75
}
```

### 2. Add L3 Recovery Evaluator

Preferred file:

- `experiments/evaluate_xdoc_resolver_l3_recovery.py`

Inputs:

```bash
--pairs data/05_eval/xdoc_element_resolver_v1_latest/cross_doc_pairs_v1.json
--baseline-pairs data/05_eval/xdoc_element_resolver_v0_latest/cross_doc_pairs_v0.json
--output-dir data/05_eval/xdoc_element_resolver_v1_latest/l3_recovery
```

Outputs:

- `l3_crossdoc_gold.jsonl`
- `gold_discovery_report.json`
- `l3_recovery_report.json`
- `l3_recovery_report.md`

### 3. Add Judge Pack Builder

Preferred file:

- `experiments/build_xdoc_resolver_judge_pack.py`

Inputs:

```bash
--pairs data/05_eval/xdoc_element_resolver_v1_latest/cross_doc_pairs_v1.json
--output data/05_eval/xdoc_element_resolver_v1_latest/judge_pack_100.jsonl
--n 100
```

Outputs:

- `judge_pack_100.jsonl`
- `judge_pack_summary.json`
- optional `judge_pack_preview.md`

### 4. Tests

Add:

- `tests/test_xdoc_element_resolver_v1.py`

Minimum unit tests:

1. `target_explicit_number_anchored` resolves target `Table 2` when the target title appears near `Table 2`.
2. `target_explicit_number_unanchored` resolves but has lower score when no target anchor exists.
3. lexical fallback still works when no numbered target exists.
4. source nearest-position score decreases as citation fanout increases.
5. L3 endpoint matching treats endpoint pairs as unordered.
6. v1 pair schema keeps fields required by `generate_multihop_l1_queries.py`.

Run:

```bash
python3 -m py_compile \
  experiments/build_xdoc_element_resolver_v1.py \
  experiments/evaluate_xdoc_resolver_l3_recovery.py \
  experiments/build_xdoc_resolver_judge_pack.py

pytest -q tests/test_xdoc_citation_filter.py tests/test_xdoc_element_resolver_v1.py tests/test_intra_doc_pairing.py
```

## Execution Order

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | Preserve v0 baseline and confirm inputs | Check v0 summary, filtered edge stats | Inputs exist and counts match known baseline | minutes, CPU | Dirty worktree confusion |
| M1 | Implement v1 explicit target resolver | `build_xdoc_element_resolver_v1.py --max-edges 1000 --stamp smoke` | Nonzero explicit target matches; no schema errors | minutes, CPU | false explicit matches from source-doc refs |
| M2 | Full v1 artifact | `build_xdoc_element_resolver_v1.py --max-pairs 5000` | 5,000 pairs, method/fanout breakdown, latest symlink | minutes, CPU | low explicit coverage |
| M3 | L3 recovery evaluation | evaluator on v0 and v1 | v1 improves at least one endpoint/target recovery metric over v0 | minutes, CPU | gold rows live in another file/schema |
| M4 | Judge pack | stratified 100 pack | Pack has balanced strata and clear rubric fields | minutes, CPU | not enough explicit examples |
| M5 | Prompt dry-run | 20-pair dry-run with `--allow-cross-doc-candidates` | 20/20 prompts render | minutes, CPU | query templates still say "same document" in some text |

## Exact Command Skeleton

### M0: Baseline checks

```bash
python3 - <<'PY'
import json
from pathlib import Path
for p in [
    "data/04_xdoc_citation/predicted_xdoc_edges_chunks_filtered_stats.json",
    "data/05_eval/xdoc_element_resolver_v0_latest/summary.json",
]:
    obj = json.loads(Path(p).read_text())
    print("\\n==", p, "==")
    print(json.dumps(obj, indent=2)[:4000])
PY
```

### M1: v1 smoke

```bash
python3 experiments/build_xdoc_element_resolver_v1.py \
  --stamp smoke \
  --max-edges 1000 \
  --max-pairs 500 \
  --min-target-score 0.12 \
  --min-overlap-terms 4
```

### M2: v1 full

```bash
python3 experiments/build_xdoc_element_resolver_v1.py \
  --max-pairs 5000 \
  --min-target-score 0.12 \
  --min-overlap-terms 4 \
  --max-pairs-per-source-chunk 25
```

### M3: L3 recovery

```bash
python3 experiments/evaluate_xdoc_resolver_l3_recovery.py \
  --pairs data/05_eval/xdoc_element_resolver_v1_latest/cross_doc_pairs_v1.json \
  --baseline-pairs data/05_eval/xdoc_element_resolver_v0_latest/cross_doc_pairs_v0.json \
  --output-dir data/05_eval/xdoc_element_resolver_v1_latest/l3_recovery
```

If the evaluator cannot discover the expected gold set, run with explicit inputs after locating the 146-row file:

```bash
rg -n '"reasoning_steps": \\[\\]|"difficulty_label": "reasoning' data/03_queries archive/data data/05_eval/m2 -S
```

Then:

```bash
python3 experiments/evaluate_xdoc_resolver_l3_recovery.py \
  --l3-input path/to/l3_pass_file.jsonl \
  --pairs data/05_eval/xdoc_element_resolver_v1_latest/cross_doc_pairs_v1.json \
  --baseline-pairs data/05_eval/xdoc_element_resolver_v0_latest/cross_doc_pairs_v0.json \
  --output-dir data/05_eval/xdoc_element_resolver_v1_latest/l3_recovery
```

### M4: judge pack

```bash
python3 experiments/build_xdoc_resolver_judge_pack.py \
  --pairs data/05_eval/xdoc_element_resolver_v1_latest/cross_doc_pairs_v1.json \
  --output data/05_eval/xdoc_element_resolver_v1_latest/judge_pack_100.jsonl \
  --n 100
```

### M5: prompt dry-run

```bash
python3 scripts/generate_multihop_l1_queries.py \
  --candidates data/05_eval/xdoc_element_resolver_v1_latest/cross_doc_pairs_v1.json \
  --output data/05_eval/xdoc_element_resolver_v1_latest/dryrun_queries.jsonl \
  --limit 20 \
  --dry-run \
  --no-images \
  --skip-llm-qc \
  --allow-cross-doc-candidates
```

## Acceptance Gates

| Gate | Required Result | Action If Pass | Action If Fail |
|---|---|---|---|
| G1: explicit target coverage | Nonzero and preferably `>=300` explicit-target pairs | keep explicit method in v1 | report low coverage, keep lexical fallback only |
| G2: L3 recovery | v1 improves over v0 on endpoint or target recovery | proceed to judge pack | diagnose ID/schema mismatch, do not judge blindly |
| G3: explicit target precision | anchored explicit sample precision `>=0.75` | eligible for production candidate tier | keep as experimental only |
| G4: overall precision | top-strata precision `>=0.55` | generate LLM query batch | tighten thresholds or add embedding rerank |
| G5: prompt rendering | 20/20 dry-run prompts render | proceed to small LLM generation | fix schema/template text first |

Production promotion requires G2, G3, and G5 at minimum. If only G1 passes, keep the artifact as a recall pool.

## Known Design Caveats

1. **Numbered refs can be source-local**  
   A citing chunk may say "Figure 1" referring to its own paper, not the cited paper. This is why target explicit resolution has anchored and unanchored tiers.

2. **Title-match is imperfect**  
   `features.title_match >= 0.2` is a useful C18 signal, but not a proof that every numbered ref in the same chunk points to the target paper.

3. **L3 gold may be schema-shifted**  
   The 87 cross-doc L3 pass rows may use an older ID family. The evaluator must report ID mismatch separately from recovery failure.

4. **Some query templates still say "same document"**  
   In the dry-run output, `formula_table` currently says "from the same document." Do not interpret prompt rendering as semantic prompt perfection. If v1 passes precision gates, patch wording to be cross-doc-aware.

## Deliverables Checklist

- [ ] `experiments/build_xdoc_element_resolver_v1.py`
- [ ] `experiments/evaluate_xdoc_resolver_l3_recovery.py`
- [ ] `experiments/build_xdoc_resolver_judge_pack.py`
- [ ] `tests/test_xdoc_element_resolver_v1.py`
- [ ] `data/05_eval/xdoc_element_resolver_v1_<stamp>/summary.json`
- [ ] `data/05_eval/xdoc_element_resolver_v1_<stamp>/report.md`
- [ ] `data/05_eval/xdoc_element_resolver_v1_<stamp>/cross_doc_pairs_v1.json`
- [ ] `data/05_eval/xdoc_element_resolver_v1_<stamp>/l3_recovery/l3_recovery_report.md`
- [ ] `data/05_eval/xdoc_element_resolver_v1_<stamp>/judge_pack_100.jsonl`
- [ ] `data/05_eval/xdoc_element_resolver_v1_<stamp>/dryrun_prompt_gate.md`
- [ ] Wiki update to `research-wiki/experiments/20260519_xdoc_pairing_module.md`
- [ ] Wiki log entry with exact counts and pass/fail gates

## Final Recommendation

The next assistant should not start with a large LLM judge. First implement explicit target numbering and L3 recovery. If v1 does not beat v0 on the free L3 check, a 100-item judge pack will mostly measure noise. If v1 does beat v0, judge only the stratified pack and use the result to decide whether to promote an explicit-target-only subset or the full v1 candidate generator.
