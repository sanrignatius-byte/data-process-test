# QC Tightening Retroactive Audit — 2026-05-13

## Snapshot
- Source: `data/03_queries/graph_max20k_four_cells_snapshot_20260513_utc/*_pass.jsonl`
- Rows audited: **3251** (all currently `qc_pass=true`)

## Headline
- Would fail under new QC: **3146 (96.8%)**
- Would still pass:        **105 (3.2%)**

Reflects the truth that the prior gate did not enforce structural-vocabulary leakage,
superlative spoilers, bridge-as-metadata, or bridge_quality floor. The current pass set
is dominated by these patterns.

## Per-issue breakdown

| Issue | Count | Share | Enforcement |
|---|---|---|---|
| `bridge_narration_in_answer` | 2707 | 83.3% | hard fail |
| `bridge_one_sided` | 2201 | 67.7% | hard fail |
| `bridge_meta_pointer` | 776 | 23.9% | hard fail |
| `superlative_answer_spoiler` | 580 | 17.8% | hard fail |
| `premise_conclusion_meta_in_answer` | 186 | 5.7% | hard fail |
| `bridge_quality_too_low` | 110 | 3.4% | hard fail |
| `bridge_meta_leak_in_query` | 95 | 2.9% | hard fail |
| `premise_conclusion_paraphrase` | 7 | 0.2% | hard fail |

## Rule-to-check mapping

| Audit finding | Prompt rule | Code check |
|---|---|---|
| `the bridge X` narration in answer (83.3%) | Rules 12, 13 | `has_bridge_narration_in_answer` |
| `the premise/conclusion` meta (5.7%) | Rule 13 | `has_premise_conclusion_meta_in_answer` |
| `the bridge` referenced in query (2.9%) | Rule 13 | `has_bridge_meta_leak_in_query` |
| Superlative answer-spoiler (17.8%, incl 5.7% apostrophe) | Rule 14 | `has_superlative_answer_spoiler` |
| Bridge is metadata pointer (23.9%) | Rule 15 | `has_bridge_meta_pointer` |
| Bridge connects only one endpoint (67.7% upper bound) | Rule 16 | `check_bridge_one_sided` |
| Premise ≈ conclusion span (0.2%) | Rule 17 | `premise_conclusion_paraphrase_score` |
| `bridge_quality < 0.20` (3.4%) | new floor in pipeline | `bridge_quality_too_low` issue |
| Opening template homogenization (`how does the` 22%, `which X best` 15%) | Rule 19 | prompt only (low false-positive cost not worth code enforcement) |

## What's NOT enforced by code (prompt-only)

- Rule 18 (≥6 content words per evidence_span) — overlaps with existing `check_evidence_spans`
- Rule 19 (opening diversity) — too false-positive-prone to regex
- Rule 12 (paraphrase mechanism not 'the bridge') — already covered by code Rules 13 (forbidden phrases)

## Expected impact on next generation run

Under the new prompt + new QC, an aggressive estimate: ~30-50% pass rate (down from 53%).
The drop comes from (a) the 17.8% superlative cases that previously passed but now fail,
(b) the 23.9% metadata-bridge cases that previously passed, (c) the 67.7% one-sided cases.
Crucially, the model is expected to ADAPT to the new prompt (Rules 12-13 + revised example)
by naming the actual mechanism instead of saying 'the bridge explains', recovering most
of the bridge_narration losses in the next round.

## Tests

- 57/57 QC checks tests pass (added 23 new tests for the 6 new check functions).
- Full repo test suite: 82 pass (excluding pre-existing `_build_adjacency` import error
  in `tests/test_negative_sampling.py` — unrelated to this change).
