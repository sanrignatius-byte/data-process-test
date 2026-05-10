# Smoke50 — per-system × per-modality metrics

Built from: data/03_queries/M4query_smoke50 (50 queries)

## R@10 by modality

| System | figure (39 qrels) | formula (25 qrels) | table (36 qrels) | all (100 qrels) |
|---|---:|---:|---:|---:|
| dense_v1_enriched | 0.7179 | 0.5600 | 0.6111 | 0.6400 |
| graph_explicit_only_baseline | 0.8205 | 0.5600 | 0.6944 | 0.7100 |
| dense_formula_caption | 0.6923 | 0.4000 | 0.6111 | 0.5900 |
| graph_formula_caption | 0.7436 | 0.5200 | 0.6667 | 0.6600 |
| graph_explicit_lineno | 0.8205 | 0.5600 | 0.6944 | 0.7100 |
| graph_explicit_virtual_origpos | 0.6410 | 0.5200 | 0.6111 | 0.6000 |
| graph_explicit_virtual_lineno | 0.6410 | 0.5200 | 0.6389 | 0.6100 |
| graph_static_prior_baseline | 0.7692 | 0.5600 | 0.5833 | 0.6500 |
| bge_ce | 0.5128 | 0.2400 | 0.3611 | 0.3900 |
| qwen3_ce | 0.6667 | 0.5600 | 0.5278 | 0.5900 |
| split_4b_text_baseline | 0.5897 | 0.4000 | 0.4722 | 0.5000 |
| split_vl2b_t5_baseline | 0.3333 | 0.4000 | 0.0278 | 0.2400 |
| math_norm_dense | 0.6923 | 0.5600 | 0.6111 | 0.6300 |
| math_norm_graph | 0.8205 | 0.5600 | 0.7222 | 0.7200 |

## MRR by modality

| System | figure | formula | table | all |
|---|---:|---:|---:|---:|
| dense_v1_enriched | 0.4100 | 0.3517 | 0.3297 | 0.5986 |
| graph_explicit_only_baseline | 0.4441 | 0.3151 | 0.3628 | 0.6142 |
| dense_formula_caption | 0.4031 | 0.2083 | 0.3260 | 0.5392 |
| graph_formula_caption | 0.3997 | 0.3116 | 0.3560 | 0.5716 |
| graph_explicit_lineno | 0.4441 | 0.3151 | 0.3628 | 0.6142 |
| graph_explicit_virtual_origpos | 0.2562 | 0.1843 | 0.2029 | 0.3525 |
| graph_explicit_virtual_lineno | 0.2494 | 0.1726 | 0.2283 | 0.3582 |
| graph_static_prior_baseline | 0.4511 | 0.3331 | 0.3285 | 0.6016 |
| bge_ce | 0.2673 | 0.0762 | 0.1309 | 0.2809 |
| qwen3_ce | 0.1209 | 0.0945 | 0.1013 | 0.1514 |
| split_4b_text_baseline | 0.3459 | 0.1937 | 0.2973 | 0.4667 |
| split_vl2b_t5_baseline | 0.2380 | 0.2155 | 0.0146 | 0.2769 |
| math_norm_dense | 0.4099 | 0.3523 | 0.3323 | 0.5942 |
| math_norm_graph | 0.4293 | 0.3410 | 0.3636 | 0.6059 |

## R@1 / R@5 / R@100 (overall, smoke50 all qrels)

| System | R@1 | R@5 | R@10 | R@100 | MRR |
|---|---:|---:|---:|---:|---:|
| dense_v1_enriched | 0.2200 | 0.5700 | 0.6400 | 0.8400 | 0.5986 |
| graph_explicit_only_baseline | 0.2200 | 0.5700 | 0.7100 | 0.8400 | 0.6142 |
| dense_formula_caption | 0.1900 | 0.5000 | 0.5900 | 0.8500 | 0.5392 |
| graph_formula_caption | 0.2100 | 0.5500 | 0.6600 | 0.8500 | 0.5716 |
| graph_explicit_lineno | 0.2200 | 0.5700 | 0.7100 | 0.8400 | 0.6142 |
| graph_explicit_virtual_origpos | 0.0800 | 0.3900 | 0.6000 | 0.8400 | 0.3525 |
| graph_explicit_virtual_lineno | 0.0700 | 0.3800 | 0.6100 | 0.8400 | 0.3582 |
| graph_static_prior_baseline | 0.2100 | 0.6100 | 0.6500 | 0.8400 | 0.6016 |
| bge_ce | 0.0700 | 0.2500 | 0.3900 | 0.8200 | 0.2809 |
| qwen3_ce | 0.0000 | 0.2200 | 0.5900 | 0.8300 | 0.1514 |
| split_4b_text_baseline | 0.1800 | 0.4100 | 0.5000 | 0.7500 | 0.4667 |
| split_vl2b_t5_baseline | 0.0900 | 0.2000 | 0.2400 | 0.4200 | 0.2769 |
| math_norm_dense | 0.2200 | 0.5600 | 0.6300 | 0.8400 | 0.5942 |
| math_norm_graph | 0.2200 | 0.5800 | 0.7200 | 0.8400 | 0.6059 |