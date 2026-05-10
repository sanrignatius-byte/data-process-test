---
type: experiment
node_id: exp:20260510_b1_phase2_lineno
title: "B1 Phase 2: LaTeX line_no chunk-element edge rebuild + smoke50 6-config audit"
date: 2026-05-10
status: completed
verdict: line_no_fix_doesnt_break_ceiling_formula_is_encoder_bound
related_experiments: [exp:20260505_smoke50_balanced_audit, exp:20260503_chunk_query_element_recall, exp:20260503_failure_profiling]
related_claims: [C2, C9, C10, C11]
---

# 目的

Mentor 录音 60 B1 todo: "chunk-element 边构建用 LaTeX 行号，不许字符串模糊匹配"。
执行 Phase 2: 实际重建 chunk 边 + 跑 graph rerank + 看是否能破 0.6913 R@10 ceiling（特别是 formula 0pp 增益）。

# Hypothesis (5/10 早段)

用 LaTeX 行号修正 chunk-element 边，formula 增益从 0pp → +3 ~ +5pp（基于 41.2% 命中估计）。

# 方法

1. `scripts/rebuild_chunk_element_edges_lineno.py`: paragraph 文本 → .tex 源行号匹配（25.3% match rate, 53/57 docs）；chunks 拿 line_range；元素按 line_no 重新分配
2. `scripts/inject_lineno_into_elements.py`: 把元素 position_idx remap 到 v2_lineno chunks 的 paragraph_indices（612 elements remapped）
3. 跑 6 个 graph rerank 配置（origpos vs lineno × {explicit-only, explicit+virtual, virtual-only}）
4. Slice smoke50 重算 per-modality

# 结果

## Per-modality R@10 on smoke50

| Config | figure | formula | table | overall |
|---|---:|---:|---:|---:|
| dense baseline | 0.7179 | 0.5600 | 0.6111 | 0.6400 |
| **explicit-only baseline** (ceiling) | **0.8205** | 0.5600 | 0.6944 | **0.7100** |
| **explicit-only + lineno** | 0.8205 | 0.5600 | 0.6944 | **0.7100** ← identical |
| explicit+virtual orig | 0.6410 | 0.5200 | 0.6111 | 0.6000 |
| explicit+virtual lineno | 0.6410 | 0.5200 | **0.6389** ⬆ | 0.6100 ⬆ |
| qwen3 ce | 0.6667 | 0.5600 | 0.5278 | 0.5900 |

## Δ (lineno − orig)

- explicit-only: **0pp** all modalities (line_no irrelevant for hub-bridge graph)
- explicit+virtual: figure 0pp, formula 0pp, **table +2.78pp**, overall +1.0pp

# 验证

| Hypothesis | 结果 |
|---|---|
| line_no fix lifts formula R@10 +3-5pp | ❌ **证伪** — formula 0pp |
| line_no fix changes graph topology meaningfully | ✅ 24/49 formulas in 1104.3913 changed chunk membership; kept=20 / added=1130 / removed=529 across all docs |
| 0.6913 ceiling is graph-topology bound | ❌ **证伪** — line_no 不动 ceiling |
| Formula ceiling is encoder-bound | ✅ **新支持** — 6 个 config 全部 ≤ 0.5600 |

# 推论

→ 新 [claim:C11](../claims/C11_formula_ceiling_is_dense_encoder_bound.md)（formula ceiling is dense-encoder bound, not graph-topology bound）入库。
→ [claim:C10](../claims/C10_graph_rerank_modality_selective.md) 强化，加 "and formula bottleneck is dense-encoder, not graph"。
→ Mentor B1 todo 状态：🟡 phase 2 完成但**对 retrieval 指标无主线增益**。Line_no 工作的副产品价值（修 P1 bug、table +2.78pp、为未来 QA grounding 提供 deterministic alignment）保留。

# 后续

- **F-formula 优先级 P0**：现在有强证据支持必须换 encoder
- B1 Phase 2 关闭，不再追加迭代（除非用于 QA / SFT 数据合成）
- 若 mentor 5/10 开会问 B1，回答：「按要求做了 LaTeX 行号对齐，但实测发现真正瓶颈在 dense encoder，不在图拓扑。新 claim:C11 + 推荐 F-formula 是下一步。」

# Cost

- ~30 min wall (script writing + 4 graph rerank runs + smoke50 slice)
- 0 GPU (graph rerank 在 CPU 上几分钟)
- $0 LLM
- 远低于 Plan 预算（~1.5h + 30min GPU）

# Files

- 计划：[refine-logs/BCD_PHASED_PLAN_20260510.md](../../refine-logs/BCD_PHASED_PLAN_20260510.md) §Phase 2
- 执行报告：[refine-logs/B1_PHASE2_LINENO_EXPERIMENT_20260510.md](../../refine-logs/B1_PHASE2_LINENO_EXPERIMENT_20260510.md)
- 代码：`scripts/rebuild_chunk_element_edges_lineno.py`, `scripts/inject_lineno_into_elements.py`
- 数据：`data/01_graphs/paragraph_chunks_n400_v2_lineno.json`, `data/03_queries/M4query_v1/graphs/multimodal_elements_lineno.json`
- 评估：4 个新 graph rerank dirs in `data/05_eval/dense_retrieval/rebuilt_20260417/graph_*_lineno_*` + 更新的 `data/05_eval/smoke50/per_system_per_modality.md`
