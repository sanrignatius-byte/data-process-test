---
type: experiment
node_id: exp:20260503_failure_profiling
title: "Dense ceiling failure profiling — 121 partial+zero query rank-of-missed analysis"
date: 2026-05-03
status: completed
verdict: R2 (cross-encoder rerank on dense top-500)
slurm_job: 66324
related_experiments: [exp:20260417_dense_baseline_rebuilt, exp:20260417_explicit_rerank_fixed]
related_claims: [C4]
---

# 结果速览（2026-05-03 11:15 UTC，job 66324）

| variable | value | 说明 |
|---|---:|---|
| r_low (rank ∈ (100,500]) | **0.690** | 触发 R2 (≥ 0.60) |
| r_mid (rank ∈ (500,2000]) | 0.264 | — |
| r_high (rank > 2000) | 0.047 | — |
| m_form / m_fig / m_tab | 0.496 / 0.264 / 0.240 | 与 spot-check 对齐 |
| **form_high** (formula 中 rank>2000) | **0.016** | 否决 R1：encoder 并未在 formula 上崩塌 |

**决策**：R2 — cross-encoder rerank on dense top-500（BGE-reranker-v2-m3 / Qwen3-Reranker-4B）。理由：69% 漏掉的 qrel 已经在 top-500 候选池里，问题是排序不是召回；formula 在 (100, 500] 桶占 65.6%，与 figure/table 同构，模态盲的 reranker 即可。详见 [refine-logs/CEILING_DECISION_20260503.md](../../refine-logs/CEILING_DECISION_20260503.md)。

---


# 目的

Best rerank R@10 = 0.6913 since 4/17，由 dense R@100 = 0.8636 决定上限。多个 corpus 增强和 cross-doc 边均无法突破。本实验放弃猜测下一步实验方向，改为在 121 个 partial+zero query 上做 rank-of-missed profiling，让数据决定下一步该投 cross-encoder / HyDE / encoder swap / 还是修 corpus bug。

# 背景关键事实

| 指标 | 数值 | 含义 |
|---|---|---|
| Per-query R@100 桶 | zero=8 / partial=113 / full=352 | 121 query 是 dense 表征不够强的子集 |
| Missed qrel modality | **formula 64 (49.6%)** / figure 34 (26.4%) / table 31 (24.0%) | formula 是漏网主体，第二位助手 spot-check 给出 |
| Corpus bug | 71.5% figure passage `text = [Image: xxx.jpg]` | 已知 figure 端的数据问题，不在本实验直接修，但是 Phase C 决策树的一个分支 |
| Graph coverage | explicit_only 仅覆盖 12.18% of 2809 pids | 解释为什么 graph rerank 在 12 万-1 倍范围饱和 |

# 假设

- **C_FP1**: missed qrel 主要落在 rank > 500，意味着 encoder 召回不足，cross-encoder 救不了
- **C_FP2**: formula 失败集中在 rank > 2000，意味着 Qwen3-Embedding-4B 不能桥接 NL query ↔ LaTeX

# 实验块

- **Phase A** (5 min, 0 GPU): 检查 `eval_dense_retrieval.py` 是否支持 `--top-k 2809`，找缓存 embeddings，spot-check formula corpus 是否同样退化
- **Phase B** (~15 min GPU): 对 121 query 做全量 sim，dump 每个 missed qrel 的真实 rank。输出 T6 (rank 桶) + T7 (rank × modality)
- **Phase C** (10 min): 按决策树 R1–R5 给单一推荐

# 决策树（顺序匹配，命中即停）

| 规则 | 触发条件 | 推荐 |
|---|---|---|
| R1 | m_form ≥ 0.40 且 form_high ≥ 0.50 | math-aware encoder swap 或 formula query expansion |
| R2 | r_low ≥ 0.60（rank ∈ (100,500]） | cross-encoder rerank top-500 (BGE / Qwen3-Reranker) |
| R3 | (m_fig + m_tab) ≥ 0.50 且 r_mid ≥ 0.40 | 先修 corpus enrichment 注入 bug |
| R4 | r_mid ≥ 0.40 | HyDE query rewriting (gpt-5.4) |
| R5 | else | 多信号并存，开 3 个并行小实验 |

# 文件

- 计划：[refine-logs/CEILING_PROFILING_PLAN_20260503.md](../../refine-logs/CEILING_PROFILING_PLAN_20260503.md)
- 待创建代码：`scripts/analyze_missed_qrel_rank.py`（仅 B2 路径）
- 待创建 slurm：`slurm_scripts/44_failure_full_rank.sh`
- 输出：`data/05_eval/failure_analysis/missed_qrel_ranks.json`、`data/05_eval/failure_analysis/decision_tables.md`
- 决策报告：`refine-logs/CEILING_DECISION_20260503.md`

# 验收

1. missed_qrel_ranks.json 覆盖 ≥ 121 query
2. decision_tables.md 含 T6 + T7
3. CEILING_DECISION_20260503.md 含命中规则编号 + 触发的数值 + 单一推荐 + 成本预估

# 与已有计划的关系

- VL enrich-only (R130–R133)：known answer，不阻塞本实验
- Cross-doc citation (R140–R143)：副线，不阻塞本实验
- 本实验 Phase C 的 R1/R2/R4 输出可能直接定义下一个 R150+ 实验
