---
type: experiment
node_id: exp:20260505_smoke50_balanced_audit
title: "M4query Smoke50 balanced modality audit (mentor 录音 60 / C6)"
date: 2026-05-05
executed: 2026-05-10
status: completed
verdict: S2_ceiling_real_modality_mixed_formula_is_real_bottleneck
related_experiments: [exp:20260503_mentor_recording60_full_todo, exp:20260503_failure_profiling, exp:20260503_ce_rerank_bge, exp:20260503_qwen3_rerank_fusion]
related_claims: [C1, C4, C5, C7, C8, C10]
---

# 执行结果（2026-05-10）

## TL;DR

- **决策规则命中**: S2 — graph 在 figure (+10.3pp) / table (+8.3pp) 上赢 dense；formula 上四方持平 0.56；text 模态在 M4query_v1 不存在
- **smoke50 graph R@10 = 0.7100 vs full graph 0.6913 偏差 +1.87pp** (<5pp 阈值) → ceiling **不是** figure-heavy artifact
- **真正瓶颈是 formula** — graph 信号对 formula 节点零增益，dense/graph/qwen3 都卡在 0.56；要破需 math-aware encoder
- mentor C6 todo 状态 ❌ → ✅

## 现实修正：M4query_v1 没有 text qrel

| 数据点 | Plan 估计 | 实际 |
|---|---:|---:|
| text qrel 数 | 26（基于 BGE pilot top-1） | **0** |
| figure qrel | — | 218 |
| formula qrel | — | 138 |
| table qrel | — | 117 |

BGE pilot top-1 = text 348/473 是 reranker text-bias **错答**，不是 ground truth。
smoke50 调整为 17 figure / 17 formula / 16 table = 50（按 rank-1 qrel 模态分桶）。

## T1: Per-modality R@10

| System | figure | formula | table | overall |
|---|---:|---:|---:|---:|
| dense | 0.7179 | 0.5600 | 0.6111 | 0.6400 |
| **graph** | **0.8205** | 0.5600 (tie) | **0.6944** | **0.7100** |
| qwen3_ce | 0.6667 | 0.5600 (tie) | 0.5278 | 0.5900 |
| bge_ce | 0.5128 | 0.2400 | 0.3611 | 0.3900 |
| split_4b_text | 0.5897 | 0.4000 | 0.4722 | 0.5000 |
| split_vl2b_t5 | 0.3333 | 0.4000 | 0.0278 | 0.2400 |

## T2: Sample 代表性

| System | M4query_v1 | smoke50 | Δ |
|---|---:|---:|---:|
| graph (ceiling) | 0.6913 | 0.7100 | +1.87pp |

→ smoke50 是有代表性的子集。

## 假设验证

- **C_S1** (modality-uniform): ❌ formula 没赢
- **C_S2** (modality-mixed, 不同 winner): ✅ **命中**
- **C_S3** (figure-heavy artifact): ❌ ceiling 在均衡测试上稳定

## Implications

1. ceiling 0.6913 是真的 — 三轮 reranker 失败的真正原因不是 modality bias，而是 graph 已摘完低垂果子
2. graph 增益强烈 modality-selective: figure +10.3pp / table +8.3pp / formula 0pp
3. formula 是真正瓶颈，dense/graph/qwen3 都 0.56 — 需 math-aware encoder
4. VL split 全模态弱于 4B text — Phase C VL fusion 不触发
5. paper claim C1/C5/C7 必须加注 modality scope

## 推荐下一步

1. **F-formula**: math-aware encoder for formula passages（Qwen3-Math / Mistral-Math 编码 LaTeX 源码）。1h GPU + $0 LLM。Success bar: formula R@10 > 0.65
2. **Claim scope 加注**: C1/C5/C7 加 "graph helps figure/table; no significant effect on formula"
3. **Text-evidence query 集**: 暂缓（除非 paper 必须 cover text 模态）

## 副产品

- BGE failure 的真根因更新：BGE figure R@10=0.51 仍远低于 graph 0.82，即便没 text-bias 它也走不通
- Qwen3 ensemble 路线证伪：formula 已 saturate，加 ensemble 反而拖低 figure/table
- graph_static_prior 在 table 上比 graph_static_plus_neighbor 弱 11pp — neighbor propagation 在 table 上特别有用
- 创建 [claim:C10](../claims/C10_graph_rerank_modality_selective.md) — graph rerank effect is modality-selective (figure/table only)

## 文件

- 决策报告：[refine-logs/SMOKE50_DECISION_20260510.md](../../refine-logs/SMOKE50_DECISION_20260510.md)
- 数据：[data/05_eval/smoke50/per_system_per_modality.md](../../data/05_eval/smoke50/per_system_per_modality.md)
- 代码：[scripts/build_smoke50.py](../../scripts/build_smoke50.py), [scripts/eval_smoke50_slice.py](../../scripts/eval_smoke50_slice.py)
- 测试集：[data/03_queries/M4query_smoke50/manifest.md](../../data/03_queries/M4query_smoke50/manifest.md)

---

# 原计划（保留作历史记录）

## 目的（原文）

回应 mentor 录音 60 的 **C6** 显式 todo（10 文本 / 10 图 / 10 表 / 10 公式），同时回答三轮 reranker 实验后的核心战略问题：**0.6913 R@10 ceiling 是真天花板，还是 M4query_v1 modality 分布严重不均（figure 56% / table 24% / formula 14% / text 5.5%）造成的 figure-heavy artifact？**

## 决策树（顺序，命中即停）

| 规则 | 触发条件 | 命中 | 推荐 |
|---|---|---|---|
| S1 | smoke50 graph overall R@10 ∈ [0.66, 0.72] 且 graph 在 4 modality 全赢 | ❌ formula 持平 | — |
| **S2** | graph 只在 ≥2 modality 赢，其他有不同 winner | ✅ | route-aware retrieval pilot |
| S3 | smoke50 graph overall R@10 < 0.60 | ❌ 0.71 | — |
| S4 | text(10) bucket R@10 < 0.30 | N/A 无 text qrel | — |
| S5 | else（mixed signal） | — | — |

# 与 M4query 初心的对照

M4query 是 "multimodal academic papers" 的 retrieval/QA/SFT 数据，不是 figure-heavy 偏好的 benchmark。但 M4query_v1 缺 text qrel，**M4query 自己就 modality 不全**——下次构建 v2 时需补 text-evidence query，否则 paper claim 在 "multimodal" 上只能讲 figure/formula/table 三模态。
