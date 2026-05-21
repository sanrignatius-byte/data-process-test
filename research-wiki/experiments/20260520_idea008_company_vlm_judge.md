---
type: experiment
node_id: exp:20260520_idea008_company_vlm_judge
title: "idea:008 company-API VLM judge smoke on 3 xdoc candidates + external-method comparison"
date: 2026-05-20
status: smoke_completed
lane: experimental
---

# idea:008 company-API VLM judge smoke

## Purpose

Validate the planned VLM judge end-to-end on a tiny representative subset before
spending budget on all 160 Phase 0 candidates. Unlike
`exp:20260520_idea008_text_only_self_judge` (caption/context only, no pixels), this
smoke sends **both endpoint images** to the company VLM and asks for a conservative
edge verdict. It also confirms every call is recorded through the repository-standard
logging path.

## External-method comparison (research-lit)

Read against registered prior art on how others build/strengthen cross-document edges:

- `paper:hessel2019_multilink` — multi-retrieval / weak supervision for image-text links.
- `paper:hwang2026_connecting_dots` — connecting cross-document evidence.
- `paper:wang2026_s1mmalign` — multimodal alignment dataset/judge.
- `paper:tian2026_corank` — LLM rerank for retrieval.
- `paper:bsap2024_clip_retrieval_bias` — CLIP retrieval bias / false positives.

Axis that emerges: others use **weak supervision / recaption / VLM-direct judgment /
LLM rerank / CLIP calibration** to turn candidate edges into strong edges. Our current
pipeline is **CLIP recall + caption/enriched light rerank**, so the weak spot is exactly
the *judgment layer* — which is what this experiment targets.

## Input

- Source pack: `data/05_eval/idea008_phase0_judge_pack_latest/phase0_candidates.jsonl`
- Judged subset: 3 representative candidates
  - `idea008_phase0_0001` — high-confidence table positive (OntoNotes/WinoBias coref-bias F1)
  - `idea008_phase0_0009` — layout false-positive control (hollow-square markers)
  - `idea008_phase0_0013` — caption-zero-overlap but semantically strong (fairness causal DAG)
- Output: `data/05_eval/idea008_company_vlm_judge_smoke_latest`

## Protocol

- Script: `experiments/idea008_company_vlm_judge_smoke.py`
- Path: `src.api.call_llm(provider="company")` → wrapped by `local_api_logger` →
  `token_logger.log_run()`. No bypass of the logger.
- Each call carries both endpoint images as base64 data URIs + the structured
  judge prompt; model `gpt-5.4`, `temperature=0.0`, `max_tokens=850`.

Verdict labels and promotion rule unchanged from
`exp:20260520_idea008_text_only_self_judge`: promote only `strong_edge` with
confidence >= 0.70; keep `weak_related` as recall; reject the rest.

## Results

- Candidates attempted: 3; parsed OK: 3; failures: 0
- Verdicts: `strong_edge` 2, `visual_layout_only` 1
  - `0001` → `strong_edge` (conf 0.98): same OntoNotes/WinoBias coref evaluation role.
  - `0009` → `visual_layout_only` (conf 0.99): isolated legend-square markers, layout false positive.
  - `0013` → `strong_edge` (conf 0.94): both are causal DAGs decomposing a sensitive
    attribute's direct/indirect effect on an outcome (path-specific fairness).
- Validation: 3/3 `ok`.
- Tokens: 5292 input / 581 output (this run).

## Logging verification

- `api_logs_cannt_delete/calls/gpt-5.4/2026-05/2026-05-20.jsonl` — 3 new call records.
- `logs/token_usage.db` — run recorded (5292 in / 581 out).

## Interpretation

The VLM judge cleanly separates the positive controls from the layout false positive,
including the caption-zero-overlap causal-DAG case that text-only judging would find
harder. This confirms the recall+VLM-judge design is the right lane: the company VLM
can both confirm real strong edges and reject high-CLIP layout matches.

## Full-160 VLM judge (2026-05-20T06:25Z)

Ran the same script over all 160 Phase 0 candidates.
Output: `data/05_eval/idea008_company_vlm_judge_full160_latest`.

- Attempted 160; parsed OK 159; 1 failure (`0048`, truncated JSON of a
  `weak_related`/0.58 answer — would not be promoted regardless).
- Tokens: 251459 input / 28319 output. Logged: 163 total calls today in
  `api_logs_cannt_delete/calls/gpt-5.4/2026-05/2026-05-20.jsonl` + `token_usage.db`.

Verdicts (n=159):

- `strong_edge`: 10
- `weak_related`: 79
- `visual_layout_only`: 55
- `unrelated`: 15

**Strong-edge rate: 10/159 = 6.3%.** Promotable (`strong_edge` & conf>=0.70): **10/10**.

Strong edges by caption bucket (the gap:G10 / idea:008 hypothesis test):

| caption_bucket | strong / total |
|---|---|
| clean_text_overlap | 5 / 38 |
| clean_caption_zero_overlap | 3 / 42 |
| degraded_one | 2 / 42 |
| **degraded_both** | **0 / 37** |

All 10 strong edges are in the `strong_text_supported` tier and collapse into ~4
semantic clusters: OntoNotes/WinoBias coref-bias tables (0001, 0006, 0023),
fairness/causal DAGs (0004, 0013, 0015, 0025), Amazon Reviews domain-adaptation
table (0002), Stack Exchange answer-acceptance / Simpson's-paradox figures (0005, 0030).

## Interpretation

The recall+VLM-judge pipeline works as a **filter**: it cleanly separates the few
real strong edges from layout false positives (the 3-edge smoke confirmed this on
controls). But two findings temper idea:008's premise:

1. **Yield is low.** Full-pool strong-edge rate is 6.3%, barely above the original
   ~5% real-text-support estimate, and far below the text-only probe's 21.9% (which
   ran on a deliberately balanced subset, not a representative one).
2. **Degraded-caption recovery did NOT happen.** The novel niche of gap:G10/idea:008
   was recovering strong edges where parser-degraded captions hide them. But
   `degraded_both` yielded 0/37 strong, `degraded_one` 2/42, and every strong edge
   sits in the already-`strong_text_supported` tier. The VLM mostly *confirms* edges
   that already had text support rather than *recovering* edges from degraded ones.

Caveat: per-bucket n is small (37–42), so this is directional, not conclusive.

## Decision

Open question for the research lead (do not auto-pivot): the evidence weakens the
"degraded-caption recovery" framing of idea:008. Options: (a) pivot the claim to
"VLM judge as a high-precision filter over CLIP recall" (which IS supported), (b)
re-scope the candidate pool toward harder/non-near-duplicate pairs before re-judging,
or (c) keep degraded-caption recovery but gather more degraded_both samples (n=37 too
small). Until decided, promote only the 10 conf>=0.70 strong edges into
`cross_doc_pairs.py`; everything else stays as recall.

## Connections

[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
