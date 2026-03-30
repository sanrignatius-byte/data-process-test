# Classic Eval (stdlib-only)

新增脚本：`scripts/evaluate_evidence_localization_stdlib.py`

## 支持算法

- `bm25`
- `bm25f`（caption=3x, enriched_content=2x, context=0.5x）
- `lm_dirichlet`
- `prf`（Rocchio 变体）
- `bm25_title_boost`
- `proximity`
- `hits`
- `graph_full`
- `rrf`（`bm25 + graph_full`）

## 支持指标

- `Recall@1/3/5/10/20`
- `MRR`
- `Coverage@10`
- `NDCG@10`（`observation/result=2`, `mechanism=1`, 其他=1）

## 运行示例

```bash
python3 scripts/evaluate_evidence_localization_stdlib.py \
  --q1 data/l1_dual_evidence_queries_v4_4_run1_pass.jsonl \
  --q2 data/l1_dual_evidence_queries_v3_pass.jsonl \
  --elements data111/multimodal_elements_enriched.json \
  --hub-candidates data111/hub_candidates_enriched_v2.json \
  --citation-graph data/citation_graph.json \
  --output data/m2/eval_classic_stdlib.json
```

如需只跑部分方法：

```bash
python3 scripts/evaluate_evidence_localization_stdlib.py \
  --methods bm25 bm25f lm_dirichlet prf rrf
```
