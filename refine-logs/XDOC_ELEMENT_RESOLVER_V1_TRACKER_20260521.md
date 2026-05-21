# Cross-Doc Element Resolver v1 Tracker

| Run ID | Milestone | Purpose | Command / Artifact | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| XRV1-R001 | M0 | Confirm v0 and G11 baseline counts | Read `predicted_xdoc_edges_chunks_filtered_stats.json` and v0 `summary.json` | 34,447 filtered edges; 5,000 v0 pairs | MUST | TODO | Stop if counts differ unexpectedly |
| XRV1-R002 | M1 | Implement v1 builder | `experiments/build_xdoc_element_resolver_v1.py` | Unit tests pass | MUST | TODO | Copy v0, keep v0 immutable |
| XRV1-R003 | M1 | Smoke v1 on 1,000 edges | `python3 experiments/build_xdoc_element_resolver_v1.py --stamp smoke --max-edges 1000 --max-pairs 500` | nonzero explicit target matches; schema valid | MUST | TODO | Inspect 20 examples manually |
| XRV1-R004 | M2 | Full v1 artifact | `python3 experiments/build_xdoc_element_resolver_v1.py --max-pairs 5000` | pair counts by method, fanout, type | MUST | TODO | Write latest symlink |
| XRV1-R005 | M3 | Build/discover cross-doc L3 gold | evaluator gold discovery | expected around 87 gold cross-doc rows | MUST | TODO | Report rejected rows and schema mismatch |
| XRV1-R006 | M3 | Compare v0 vs v1 on L3 recovery | `evaluate_xdoc_resolver_l3_recovery.py` | endpoint/doc-pair recall@K | MUST | TODO | v1 must improve at least one target metric |
| XRV1-R007 | M4 | Build stratified judge pack | `build_xdoc_resolver_judge_pack.py --n 100` | balanced strata, judge schema complete | MUST | TODO | No LLM judging yet unless pack is sane |
| XRV1-R008 | M5 | Query dry-run | `generate_multihop_l1_queries.py --allow-cross-doc-candidates --dry-run --limit 20` | 20/20 prompt render | MUST | TODO | Watch for "same document" template wording |
| XRV1-R009 | M4 | Optional manual/LLM judge execution | judge pack with fixed rubric | precision by stratum | NICE | TODO | Only after R006 passes |
| XRV1-R010 | M5 | Wiki/log update | update experiment page and log | exact counts recorded | MUST | TODO | Do not claim support before judge/recovery gates |

## Stop/Go Rules

- Stop after R003 if explicit target coverage is zero and examples show mostly source-local refs.
- Stop after R006 if v1 is worse than v0 and the gold set is valid.
- Proceed to R009 only if R006 passes or if R006 failure is clearly due to ID/schema mismatch.
- Promote to production only if explicit-target precision is high and prompt gate remains clean.
