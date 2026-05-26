# Cross-Document Chain Judge: Fixed Entity Chains

Date: 2026-05-22

Input:

- `data/05_eval/cross_doc_chains_final_fixed.json`
- 38 fixed entity-bridge chains
- Each chain has 3 papers and 2 cross-document entity bridges

Judge:

- Script: `experiments/judge_cross_doc_chain.py`
- Model: `company:gpt-5.4`
- Output: `data/05_eval/cross_doc_chain_judge_fixed`
- API logging: `local_api_logger.wrap_requests_call`
- Token DB entry: `logs/token_usage.db`
- Tokens: 63,425 input + 22,467 output

## Result

| Metric | Count | Rate |
|---|---:|---:|
| total chains | 38 | 100.0% |
| strong_chain | 2 | 5.3% |
| weak_but_related | 14 | 36.8% |
| topic_only | 12 | 31.6% |
| broken_chain | 10 | 26.3% |
| production keep | 2 | 5.3% |
| production review | 14 | 36.8% |
| production drop | 22 | 57.9% |

Bridge-level quality is higher than full-chain quality:

| Bridge verdict | Count |
|---|---:|
| strong | 19 |
| weak | 45 |
| broken | 10 |
| topic_only | 2 |

This means many individual bridge edges are plausible, but the full 3-paper chain often fails because the middle paper does not act as a coherent scientific relay.

## Production-Keep Chains

Two chains passed as `strong_chain / keep`.

1. `xdoc_eb_fixed_0018`
   - Papers: `1804.09301 -> 1809.01496 -> 1904.03310`
   - Entities: `winobias`, `coreference resolution`, `ontonotes`
   - Why it works: all three papers are grounded in WinoBias/OntoNotes coreference-bias evaluation, and the middle paper relays the same benchmark/evaluation construct.

2. `xdoc_eb_fixed_0037`
   - Papers: `1804.06876 -> 1804.09301 -> 1904.03310`
   - Entities: `winobias`, `coreference resolution`
   - Why it works: the first paper grounds WinoBias examples, the middle paper analyzes WinoGender/WinoBias-style gender-bias difficulty, and the third evaluates WinoBias bias gaps.

Files:

- `data/05_eval/cross_doc_chain_judge_fixed/keep_chains.jsonl`
- `data/05_eval/cross_doc_chain_judge_fixed/review_chains.jsonl`
- `data/05_eval/cross_doc_chain_judge_fixed/drop_chains.jsonl`

## Failure Modes

| Main failure | Count | Interpretation |
|---|---:|---|
| disconnected_middle | 15 | each bridge is somewhat related, but the middle paper does not form a true relay |
| generic_entity | 11 | entities like `linear model`, `outcome`, `predictor`, or `adversarial training` are too broad |
| one_bad_bridge | 9 | one edge is usable while the other breaks |
| missing_context | 1 | element metadata is too thin |
| none | 2 | the two strong chains |

## Interpretation

The idea is not dead, but the current chain construction is not production-ready as-is.

The old endpoint-level entity judge overestimates quality because it asks whether a single bridge is plausible. The new chain-level judge asks whether two bridges compose into a useful 3-paper evidence path. That stricter question exposes the real bottleneck: composition through the middle paper.

The useful pattern is clear: named benchmark/task families work well, especially `WinoBias + coreference resolution + OntoNotes`. Generic method/theory families are too loose unless the elements expose the same named construct or metric.

## Recommendation Before Production

Do not run production over all fixed chains yet.

Run a narrowed production attempt with these filters:

1. Keep entity sets with named benchmarks/datasets/tasks/metrics:
   - `winobias`
   - `ontonotes`
   - `coreference resolution`
   - similar named benchmark families
2. Downweight or reject generic-only bridges:
   - `linear model`
   - `structural equation`
   - `outcome`
   - `predictor`
   - `adversarial training`
3. Require the middle paper to be a true relay:
   - either the same element participates in both bridges, or
   - two middle-paper elements share a named benchmark/task/metric relation.
4. Use chain-level judge as a gate before query generation.

Expected near-term yield from the current 53-paper corpus:

- high-precision production keep: about 5-10%
- reviewable weak data: about 35-45%
- ungated production data would be too noisy.
