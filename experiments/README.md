# M2 experiment packaging workspace

This directory is the packaging target for the five-stage M2 execution plan.

## Planned outputs

- `exp_A_difficulty_gradient.json`: BM25-only difficulty validation across Level 1 / 2 / 3 queries.
- `exp_B_retrieval_enhancement.json`: retrieval comparison between BM25 and graph-enhanced retrieval, with Level-3 subset analysis.
- `exp_C_qa_triangle.json`: QA triangle validation that measures how many ground-truth evidence elements are covered by generated answers.
- `m2_execution_manifest.json`: current-state bootstrap manifest generated from the repo's existing assets.

## Current status

Use the bootstrap script below to refresh the manifest before starting a new execution round:

```bash
python scripts/bootstrap_m2_delivery.py --output experiments/m2_execution_manifest.json
```

The manifest is meant to answer three practical questions before long jobs are launched:

1. Which current datasets are good enough to serve as Level 1 / Level 2 baselines?
2. Which gaps still block M2 packaging, especially for Level 3 and experiment C?
3. Which retrieval and embedding artifacts already exist and can be reused immediately?
