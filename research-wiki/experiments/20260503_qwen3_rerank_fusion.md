---
type: experiment
node_id: exp:20260503_qwen3_rerank_fusion
title: "Qwen3-Reranker-4B rerank + fusion ablation on anchor corpus"
date: 2026-05-03
status: completed
verdict: NEGATIVE_no_fusion_beats_graph_ceiling
slurm_jobs: [66395, 66401, 66405, 66408]
related_experiments: [exp:20260503_failure_profiling, exp:20260503_ce_rerank_bge, exp:20260503_corpus_enrich_fix]
related_claims: [C4, C8]
---

# 目的

CE rerank pilot (`exp:20260503_ce_rerank_bge`) 用 BGE-reranker-v2-m3 失败，verdict 是 reranker 严重 text-bias。第二位助手 14:55 综述里推荐的 P0 选项 A 是「换 reranker 家族」——本实验验证 Qwen3-Reranker-4B（同 encoder 家族，理论上 prior 跟 BGE 不同）能否破 0.6913 ceiling。

Anchor corpus 不动（rebuilt_20260417/augmented），只换 reranker 模型。同时跑 dense / graph / CE / 多种 RRF fusion 全套消融。

# 假设

- **C_QR1**: Qwen3-Reranker-4B 不会有 BGE 的 NL-bias，能在 figure/formula 上不退步
- **C_QR2**: rrf(dense, graph, qwen_ce) 比 rrf(dense, graph) 多带独立信号，至少撬动 R@100 像 BGE pilot 那样 +2pp 以上

# 自动化执行

`scripts/qwen3_rerank_autopilot_20260503T160617Z.log`，autopilot 流程：模型 cache (7.6GB) → submit slurm → 监控 → fusion 评测。

Slurm script: [slurm_scripts/48_ce_rerank_qwen3_fusion.sh](../../slurm_scripts/48_ce_rerank_qwen3_fusion.sh)（仅 rerank，复用 fp16 + max_length=2048 + bs=64 设置）

最终成品：job 66408（1h 14 min on a6000）。前三个 job (66395/66401/66405) 是环境/参数调试。

# 主要结果

| config | R@1 | R@5 | R@10 | R@100 | MRR | ΔR@10 vs graph |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.2336 | 0.5275 | 0.6195 | 0.8636 | 0.6122 | −7.2 |
| **graph** | 0.2209 | 0.5941 | **0.6913** | 0.8636 | 0.6017 | **0.0** |
| ce (qwen3 alone) | 0.0000 | 0.2336 | 0.5613 | 0.8594 | 0.1534 | −13.0 |
| rrf_dense_ce_k20 | 0.2336 | 0.5275 | 0.6195 | 0.8594 | 0.6120 | −7.2 |
| **rrf_graph_ce_k10** | 0.2505 | 0.6068 | **0.6702** | 0.8626 | **0.6417** | −2.1 |
| rrf_graph_ce_k20 | 0.2505 | 0.5920 | 0.6702 | 0.8626 | 0.6433 | −2.1 |
| rrf_dense_graph_k10 | 0.2304 | 0.5888 | 0.6681 | 0.8636 | 0.6231 | −2.3 |
| rrf_dense_graph_ce_k10 | 0.2326 | 0.5708 | 0.6617 | 0.8615 | 0.6226 | −3.0 |

**Verdict**：无任何 fusion config 能突破 graph 0.6913。Qwen3-Reranker-4B 单跑比 BGE 单跑稍好（0.5613 vs 0.4482）但仍远低于 dense baseline。

# 模态偏置（关键发现）

每个 reranker 的 top-1 模态选择对比：

| reranker | text | figure | table | formula | 偏向 |
|---|---:|---:|---:|---:|---|
| dense baseline | 26 | 265 | 115 | 67 | （正常分布）|
| graph rerank | 3 | 273 | 142 | 55 | （强化 figure/table）|
| **BGE-reranker-v2-m3** | **348** | 87 | 29 | 9 | **偏文本** |
| **Qwen3-Reranker-4B** | 36 | 144 | 45 | **248** | **偏公式** |

**两个 reranker 各偏一个模态、且方向相反**：BGE 把所有 query 推到 NL 段，Qwen3 把所有 query 推到 LaTeX 公式。两者各坏在不同方向，但都坏。

Per-modality R@10 对比：

| config | figure R@10 | table R@10 | formula R@10 |
|---|---:|---:|---:|
| dense | 0.6893 | 0.6254 | 0.4413 |
| **graph** | **0.7453** | **0.7257** | **0.4972** |
| Qwen3 CE | 0.6332 | 0.5457 | 0.4190 |
| rrf_graph_ce_k10 | 0.7313 | 0.6932 | 0.4804 |

**graph 在所有三个 modality 上都赢**——没有任何 fusion config 能在 figure/table/formula 三个模态同时拉过 graph。

# R@100 ceiling 复现失败

BGE pilot 给的唯一正向信号是 rrf(dense, BGE) **R@100 +2.3pp** (0.8636 → 0.8869)。Qwen3 没有复现：

| config | R@100 | Δ vs dense |
|---|---:|---:|
| dense | 0.8636 | — |
| BGE rrf_dense_ce_k20 | 0.8869 | **+2.3pp** |
| **Qwen3 rrf_dense_ce_k20** | 0.8594 | **−0.4pp** |
| Qwen3 rrf_graph_ce_k20 | 0.8626 | −0.1pp |

**结论**：那个 ceiling 移动信号是 BGE-specific，不是 generic CE 现象。Qwen3 跟 encoder 家族太近，rerank 是相关的、不带独立证据；BGE 跟 encoder 家族远，rerank 出乎意料地丰富了 candidate pool（虽然 R@10 跌但 R@100 涨）。

# 与 C8 的连接

`claim:C8`（MODORA 视觉描述对 text-style retrieval 净负）的根本机制是「passage modality style ≠ query modality style」。本实验在 reranker 端再次坐实：

- corpus 端：MODORA 注入 → dense 跌（C8 已证）
- reranker 端：BGE 拉文本 / Qwen3 拉公式 → 都跌（本实验证）

两个发现指向同一个结构性问题（gap:G7 modality-style mismatch）：**M4query_v1 的 query 是 paper-domain 文本式，任何模态盲的全局 reranker / 注入都会引入新的模态 bias**。

# 含义

可以从 shortlist 划掉的：
- ~~Option A: 换 reranker 家族~~ — Qwen3 反向偏置，证伪「换家族就能 escape bias」
- ~~Cohere rerank-3.5 / Voyage rerank-2~~ — 同理大概率有自己的偏置，预期收益已被削弱

仍在台面上的：
- **B**: Late-fusion VL lane（graph rerank 在 figure/table 上 +5pp，VL split 救活 figure lane 到 0.5397——尝试 graph + VL fusion 而不是 graph + CE fusion）
- **C**: HyDE query rewriting（从 query 端缩小 modality-style mismatch，绕开 reranker 的偏置问题）
- **新增 D**: 重新审视「0.6913 是不是真天花板」——目前的 M4query_v1 modality 分布严重不均（figure 56% / table 24% / formula 14% / text 5.5%），mentor C6 要求的 10/10/10/10 balanced smoke test 至今未做，可能 ceiling 的形状本身就是 artifact

# 文件

- 报告：[data/05_eval/cross_encoder_rerank/qwen3_reranker_4b_transformers_anchor_top500/fusion_report.md](../../data/05_eval/cross_encoder_rerank/qwen3_reranker_4b_transformers_anchor_top500/fusion_report.md)
- 评分：`scores_ce_qwen3_tf.jsonl`、`ranking_ce_qwen3_tf.jsonl`、`metrics_ce_qwen3_tf.json`
- Slurm: 66395 / 66401 / 66405 / 66408（最终成品 66408）
- Autopilot log: `logs/qwen3_rerank_autopilot_20260503T160617Z.log`

# 验收

1. ✅ Qwen3-Reranker-4B 在 anchor corpus 上完整跑过 473 query × 500 candidate
2. ✅ Fusion ablation 覆盖 dense / graph / ce / 三种 RRF (k=10/20/60) × 三类组合
3. ✅ 模态守门（top-1 分布 + 三模态 R@10）已记录
4. ✅ 决策：**无 fusion 突破 0.6913；Option A 失败；战略选项缩到 B/C/D**
