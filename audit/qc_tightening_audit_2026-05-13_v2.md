# QC Tightening v2 — Merged Audit (2026-05-13)

This update merges the copilot/design-triplet-construction prompt rewrite into
the claude/review-triplet-learning-design-4dEas branch, with 4 follow-up fixes
identified during independent review of the copilot work.

## What changed vs v1 (commit 1d65e80)

### Prompt (src/prompts/templates.py PROMPT_3STEP_REASONING_CHAIN)

Adopted copilot's structural rewrite of the 3-step prompt, with 4 follow-up
fixes applied:

| Fix | Issue | Resolution |
|---|---|---|
| F1 | GOOD example used `"leading accuracy"` — `leading` is a banned superlative under Rule 13 | Rewrote to `"convergence pattern observed in the cohort-level accuracy curves"` |
| F2 | Rule 13 forbidden list missing apostrophe forms + maximum/minimum/most-X variants vs QC regex coverage | Aligned Rule 13 to the QC regex superset: added `stronger`, `maximum`, `minimum`, `most accurate/common/robust/effective/stable/consistent/preferred/favored`, possessive forms `method's strongest` |
| F3 | "Middle paragraph" still appeared 32× in user-visible prose — same word-density problem that caused the original 83.3% bridge narration | Renamed canonical user-visible label to "Connecting paragraph" (30 occurrences). The 2 remaining "middle paragraph" references are inside Rule 17 forbidden-phrase alias list (intentional) |
| F4 | OPENER VARIETY used `<observation>` / `<design choice>` placeholders that some models echo literally | Replaced 5 templates with fully-formed compliant examples using FairBoost vocabulary |

Additional consistency fixes after F3 surfaced shuffled BAD/GOOD labels in
the ANTI-PATTERN block (BAD/GOOD swapped on the metadata-pointer example).

### Code QC (src/qc/checks.py, src/qc/pipelines.py)

Added one check on top of the v1 set of 6:

- `check_premise_contains_answer` — fires when step 1 (premise) provides
  more answer-bearing content tokens than step 3 (conclusion). 1.1% of the
  current pass set.

Total code QC additions: **9 hard-fail gates** (6 from v1 + bridge_one_sided
port + premise_contains_answer + bridge_quality floor wired through pipeline).

## Test Status

- `tests/test_qc_checks.py`: **60/60 passing** (added 3 tests for the new
  `check_premise_contains_answer`)
- Full repo (excl pre-existing `_build_adjacency` issue): **137/137 passing**

## Retroactive Audit on 3251-row PASS set (with ALL 9 gates)

| Issue | Count | Share |
|---|---|---|
| `bridge_narration_in_answer` | 2707 | 83.3% |
| `bridge_one_sided` | 2201 | 67.7% |
| `bridge_meta_pointer` | 776 | 23.9% |
| `superlative_answer_spoiler` | 580 | 17.8% |
| `premise_conclusion_meta_in_answer` | 186 | 5.7% |
| `bridge_quality_too_low` | 110 | 3.4% |
| `bridge_meta_leak_in_query` | 95 | 2.9% |
| `premise_contains_answer` | 36 | 1.1% |
| `premise_conclusion_paraphrase` | 7 | 0.2% |

Combined: **96.9%** of current pass set fails under the new gate.

## Vocabulary audit of new PROMPT_3STEP_REASONING_CHAIN

| Token | Original (snapshot prompt) | Copilot v1 | This branch v2 |
|---|---|---|---|
| `bridge*` (user-visible prose) | 22 | 7 | 9 (incl 2 alias mentions in Rule 17) |
| `<placeholder>` syntax | 0 | 5 | **0** |
| `Middle paragraph` (canonical) | 0 | 32 | 2 (alias only) |
| `Connecting paragraph` (canonical) | 0 | 0 | **30** |
| Banned superlatives in GOOD examples | (no examples) | 1 (`leading`) | **0** |

## Expected impact on next generation

| Configuration | Estimated PASS rate |
|---|---|
| Snapshot (baseline) | 53% |
| With copilot v1 prompt + my v1 code QC | 78–82% |
| With merged v2 (this branch) | **80–85%** |
| + construction-time data filter (separate work) | **>90%** |

The data-side filter (bridge_one_sided 67.7% + bridge_meta_pointer 23.9%)
remains the dominant volume gate; only construction-time enrichment changes
can move that floor.
