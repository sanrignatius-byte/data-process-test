# M4query_smoke50 manifest

Built: 2026-05-10  seed=20260510
Source: M4query_v1 (473 queries / 946 qrels)

## Sampling rule
Stratified by **primary qrel modality** (rank-1 qrel's element_type).

## Mentor C6 spec vs M4query_v1 reality

| Spec (mentor 录音 60) | This sample | Why |
|---|---|---|
| 10 text | 0 | M4query_v1 has 0 text qrels; text not realizable on this dataset |
| 10 figure | 17 | reweighted to fill text quota |
| 10 formula | 17 | reweighted |
| 10 table | 16 | reweighted |

5/3 BGE pilot top-1 modality showing 348/473 = text was reranker error,
not ground truth. M4query_v1 qrels are 100% from {figure, formula, table}.

## Bucket stats

| Modality | Pool size | Quota | Picked | Query level dist |
|---|---:|---:|---:|---|
| figure | 218 | 17 | 17 | {'l1': 13, 'l3': 4} |
| formula | 138 | 17 | 17 | {'l1': 12, 'l3': 5} |
| table | 117 | 16 | 16 | {'l1': 10, 'l3': 6} |

Total queries: 50  qrels: 100

## Output files
- `data/03_queries/M4query_smoke50/queries.jsonl`
- `data/03_queries/M4query_smoke50/qrels.jsonl`