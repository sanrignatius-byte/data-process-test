# Phase-0 A/B 结果回落分析（2026-03-12）

## 现象
在 `scripts/run_phase0_eval_ab.py` 的当前实现下，`graph_hub_rerank` 相比 `bm25` 明显下降（Recall@10 / MRR 都下降）。

## 核心原因

1. **Hub prior 覆盖率低，且与 GT 对齐不足**
   - 当前 hub candidate 仅覆盖 161 个 element（语料总 chunk 为 1314）。
   - 在评测 query 的 GT 证据 element 中，仅约 35.7% 被 prior 覆盖。
   - 这会导致大量 query 的真实证据拿不到先验加分，反而由“被先验覆盖但不相关”的元素上浮。

2. **Hub prior 实际退化为二值强加分（缺少区分度）**
   - `hub_candidates_enriched_v2.json` 中 `quality_score` 全部是常数 0.8。
   - 在代码中按 max 归一化后，所有被覆盖元素 prior 全变为 1.0，等价于“覆盖即同等加分”。
   - 结果是先验无法表达“更可能相关”的强弱，只会粗暴重排。

3. **`graph_alpha=0.6` 在当前 prior 质量下过强**
   - `graph_hub_rerank` 使用 `score = norm_bm25 + graph_alpha * prior`。
   - 当 prior 近似二值且覆盖噪声存在时，0.6 的权重足以把 BM25 正确排序冲掉。
   - 实测 sweep 显示：
     - alpha=0.0：图方法与 BM25 完全一致；
     - alpha=0.2：仅轻微下降；
     - alpha=0.6：显著下降；
     - alpha=1.0：进一步崩塌。

4. **评测命中判据是 element_id 命中，不是语义相近**
   - 评测逻辑优先使用 `required_evidence_spans.element_id` 是否命中 top-k。
   - 任何对“具体 element 排名”的错误重排都会直接伤害 Recall@10 与 MRR。
   - 这使得“粗粒度先验”更容易带来负收益。

## 证据摘要
- 当前一次复现实验（本地）：
  - bm25: Recall@10=0.8448, MRR=0.5594
  - graph_hub_rerank(alpha=0.6): Recall@10=0.7292, MRR=0.4373
- Query 级别对比（graph vs bm25）：
  - RR 提升 52 条，下降 105 条，持平 120 条。
  - hit@10 从 1→0 的有 44 条，而 0→1 仅 12 条。

## 建议
1. **短期止损**：将 `graph_alpha` 降到 0.0~0.2（优先 0.1/0.2）并固定在配置里。  
2. **中期修复**：重建 `hub_candidates` 的打分分布（不要全常数），并做校准（例如分位归一化）。  
3. **长期优化**：把 prior 从“直接加分”改为“受控 rerank”（例如只在 BM25 top-N 内进行小幅 tie-break），避免大幅改写主排序。  
