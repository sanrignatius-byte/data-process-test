# Plan — Dense Ceiling Failure Profiling

**Date**: 2026-05-03
**Status**: Ready for execution (handed off to coder assistant)
**Owner**: third assistant (coder)
**Target wall time**: ~30 min (5 min check + 15 min GPU + 10 min decision)

---

## Why This Plan Exists

Best rerank R@10 = 0.6913 since 2026-04-17, capped by dense R@100 = 0.8636. Multiple corpus enhancements and cross-doc edges have all failed to break the ceiling. The previously proposed two plans (VL enrich-only comparison, cross-doc citation pipeline) target peripheral questions instead of the ceiling. This plan profiles the actual failure cases so the next experiment is data-driven, not speculative.

Two prior critique passes (this conversation, two assistants) converged on:
1. The two existing plans are correct in mechanism but low ROI for breaking 0.6913
2. The real bottleneck is the 121 partial+zero queries that dense retrieval cannot bring into top-100 well
3. **Formula is the dominant missed modality (49.6%)** — this is the key finding that reshapes the decision tree

---

## Pre-confirmed Inputs (do NOT redo)

| Data point | Value | Source |
|---|---|---|
| Best rerank R@10 | 0.6913 | `data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_only_fixed/metrics_graph_static_plus_neighbor.json` |
| Dense baseline R@100 | 0.8636 | same dir, `metrics_baseline.json` |
| Per-query R@100 buckets | zero=8 / partial=113 / full=352 | spot-check, second assistant |
| Missed qrel modality split | **formula 64 (49.6%) / figure 34 (26.4%) / table 31 (24.0%)** | spot-check, second assistant |
| Corpus bug (figure side) | 71.5% of figure passages have `text` = `[Image: xxx.jpg]` placeholder | spot-check, second assistant |
| Graph coverage | explicit_only graph maps 12.18% of 2809 pids (342 / 2809) | `query_graph_stats.json` |

---

## Phase A — Pre-flight Checks (5 min, no GPU)

1. Read [scripts/eval_dense_retrieval.py](scripts/eval_dense_retrieval.py) argparse. Note the flag name (`--top-k` vs `--top_k`) and default. If max top-k is hard-capped, decide whether to bump it or use Phase B path B2.
2. `find data/05_eval/dense_retrieval/rebuilt_20260417 -name "*.npy" -o -name "*embed*" -o -name "*encoded*"` — check whether query / passage embeddings are already cached for the M4query_v1 corpus. If yes, prefer the cached path in B2.
3. From `data/05_eval/dense_retrieval/rebuilt_20260417/augmented/corpus_v1_enriched.jsonl`, sample 5 random formula passages (pid containing `formula`) and dump their `text` field.
   - Expected good: `[FORMULA] D(M x, M y) \leq d(x, y) \tag{1}` (LaTeX form, encoder can read)
   - Bad signal: `[Formula]` placeholders or empty text
   - **Gate**: if formula passages are degraded like figures, STOP and escalate. The fix is corpus-side, not encoder-side, and the rest of this plan is moot until that's fixed.

---

## Phase B — Full-rank Lookup for Missed qrels (~15 min GPU, 1 slurm job)

Scope: only the 121 partial+zero queries. For each, find the actual rank of every missed qrel (qrels that fall outside top-100 of dense baseline).

Two implementation paths — pick the shorter one based on Phase A:

- **B1 (preferred if possible)**: rerun `eval_dense_retrieval.py` with `--top-k 2809` (full corpus), restricted to the 121 query subset if the script supports it; otherwise run on all 473 and slice afterward. Dump full rankings to a new directory under `data/05_eval/failure_analysis/`.
- **B2 (fallback)**: write `scripts/analyze_missed_qrel_rank.py` (≤80 LOC). If Phase A2 found cached embeddings, load them; otherwise re-encode 473 queries + 2809 passages with Qwen3-Embedding-4B (~5 min GPU). Compute full sim matrix. For each (query, missed_qrel_pid), look up rank.

Slurm: [slurm_scripts/44_failure_full_rank.sh](slurm_scripts/44_failure_full_rank.sh) (new). A6000 is enough; minerU env.

### Outputs

`data/05_eval/failure_analysis/missed_qrel_ranks.json`:
```json
{
  "l1_de_1104.3913_0073": [
    {"qrel_pid": "...", "modality": "formula", "rank": 1842},
    {"qrel_pid": "...", "modality": "table",   "rank": 87}
  ]
}
```

`data/05_eval/failure_analysis/decision_tables.md`:
- **T6**: missed qrel rank-bucket distribution
  - `(100, 500]` / `(500, 2000]` / `(2000, ∞)` counts and percentages
- **T7**: rank-bucket × modality cross-tab (rows = formula / figure / table; cols = the three rank buckets). The formula row is the most decision-relevant cell.

---

## Phase C — Decision Gate (10 min, write only)

Output: `refine-logs/CEILING_DECISION_20260503.md` (≤1 page).

Read T6 + T7 and apply the rules **in order, first match wins**:

```
Variables (from T6 / T7):
  m_form, m_fig, m_tab          modality share of missed qrels (T7 row totals)
  r_low, r_mid, r_high          T6 bucket shares: (100,500] / (500,2000] / (2000,∞)
  form_high                     fraction of formula misses with rank > 2000

Decision rules:

R1. if m_form ≥ 0.40 and form_high ≥ 0.50:
      → Recommend: math-aware encoder swap OR formula-side query expansion
      → Reason: Qwen3-Embedding-4B cannot bridge NL query ↔ LaTeX formula
      → Candidate moves:
          - swap encoder for the formula passages: MathBERT, SimCSE-Math,
            jina-embeddings-v3 LaTeX mode
          - LLM-rewrite formula queries into "expected LaTeX form +
            natural-language description" and concat before encoding

R2. elif r_low ≥ 0.60:
      → Recommend: cross-encoder rerank on dense top-500
      → Reason: encoder already brings evidence into the candidate pool;
                the gap is in ordering, not recall
      → Candidate moves: BGE-reranker-v2-m3, Qwen3-Reranker-4B

R3. elif (m_fig + m_tab) ≥ 0.50 and r_mid ≥ 0.40:
      → Recommend: fix corpus enrichment injection bug FIRST (71.5% figure
                   text = [Image:]), re-build corpus, re-evaluate before
                   investing in any new algorithm
      → Reason: no reranker can promote a passage whose text is a placeholder

R4. elif r_mid ≥ 0.40:
      → Recommend: HyDE query rewriting (gpt-5.4 generates a hypothetical
                   passage per query; retrieve with that)
      → Reason: encoder representation isn't strong enough to pull evidence
                into top-500 from query alone

R5. else:
      → Mixed signal. Open three small parallel pilots, pick winner by R@10
        delta on the 121-query slice.
```

Report must include:
- which rule matched (R1–R5)
- the numeric values that triggered the match (e.g. "m_form = 0.51, form_high = 0.62")
- recommended next experiment with estimated GPU time, $ cost, and a one-paragraph design

---

## Out-of-scope

- Don't re-do the modality spot-check (already done by second assistant)
- Don't run Plan 1 (VL enrich-only) — known answer
- Don't run Plan 2 (cross-doc citation) — keep on the side track
- Don't start any of the recommended next experiments — Phase C output triggers a separate planning round

---

## File Manifest

| Path | Action | Owner |
|---|---|---|
| `scripts/analyze_missed_qrel_rank.py` | New, ≤80 LOC, only if B2 path chosen | coder |
| `slurm_scripts/44_failure_full_rank.sh` | New, ~30 lines | coder |
| `data/05_eval/failure_analysis/missed_qrel_ranks.json` | Generated by Phase B | coder |
| `data/05_eval/failure_analysis/decision_tables.md` | Aggregated from Phase B output | coder |
| `refine-logs/CEILING_DECISION_20260503.md` | Phase C report, ≤1 page | coder |
| `research-wiki/experiments/20260503_failure_profiling.md` | New wiki experiment node | coder |
| `research-wiki/log.md` | Append timestamped one-line summary on completion | coder |
| `research-wiki/index.md` | Add link to new experiment node under Experiments | coder |

---

## Acceptance Criteria

1. `missed_qrel_ranks.json` covers ≥ 121 queries × at least 1 missed qrel each, modality field populated for every entry
2. `decision_tables.md` contains T6 and T7, numbers reconcile with `missed_qrel_ranks.json`
3. `CEILING_DECISION_20260503.md` contains: matched rule (R1–R5), the triggering numeric values, the recommended next experiment with estimated GPU/$/time
4. Wiki updated: experiment node + log.md timestamp + index.md link

---

## Notes for the coder

- The `eval_dense_retrieval.py` workflow probably already lives behind a slurm script; check `slurm_scripts/` for existing examples before writing 44_failure_full_rank.sh from scratch
- If Phase A spots the formula corpus bug, write a 3-line escalation to `refine-logs/CEILING_DECISION_20260503.md` instead of running Phase B, and stop there. That's a valid Phase C output.
- The plan deliberately gives no leeway on adding extra profiling tables (T1–T5 from earlier draft were rejected as redundant given the modality spot-check already done)
