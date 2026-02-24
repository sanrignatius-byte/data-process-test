# L1 Dual-Evidence Official Batch Report

- Source: `data/l1_dual_evidence_queries_slurm_img150_tuned_v4_official.jsonl`
- Total queries: **222**
- QC pass: **173**
- QC fail: **49**
- QC pass rate: **77.93%**

## Pair Type Distribution

| key | count |
|---|---:|
| figure+table | 144 |
| figure+formula | 62 |
| formula+table | 16 |

## Query Type Distribution

| key | count |
|---|---:|
| causal_explanation | 107 |
| discrepancy_analysis | 95 |
| hypothesis_verification | 20 |

## Quality Tier Distribution

| key | count |
|---|---:|
| silver | 210 |
| gold | 12 |

## Hop Distance Distribution

| key | count |
|---|---:|
| 1 | 222 |

## QC Issue Breakdown

| key | count |
|---|---:|
| anchor_leakage | 15 |
| weak_reasoning_connector | 12 |
| single_element_answer | 11 |
| numeric_leakage | 7 |
| template_shortcut | 4 |
| pseudo_multihop_parallel | 2 |
| meta_language | 2 |
| template_collapse | 1 |
| templated_opening | 1 |
| evidence_spans_incomplete | 1 |

## Reasoning Chain Checks

- Missing reasoning_chain: **0**
- Short reasoning_chain (<40 chars): **0**
