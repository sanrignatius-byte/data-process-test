---
type: experiment
node_id: exp:20260503_corpus_enrich_fix
title: "Corpus enrichment mapping fix — recover 293 dropped figure enrichments"
date: 2026-05-03
status: completed
verdict: D5_antipattern
related_experiments: [exp:20260503_failure_profiling, exp:20260503_ce_rerank_bge]
related_claims: [C4]
---

# 目的

CE rerank 实验（job 66349）暴露：BGE-reranker-v2-m3 把 figure passage 中的 `[Image: xxx.jpg]` 系统性压低，因为它们作为 NL 文本就是垃圾。直接核验 `corpus_v1_enriched.jsonl` × `multimodal_elements_enriched.json` 发现：

- **63.5% 的 figure passage (695/1095) 的 text 字段是 `[Image: xxx.jpg]`**
- **其中 42.2% (293/695) 在 enrichment source 里有完整 enriched_content**

也就是说，corpus build 时键映射漏掉了 293 条已经存在的 enrichment。这是数据 bug，不是算法选择。

本实验：诊断键映射错配 → 修 build 脚本 → 重建 corpus → 重测 dense baseline + graph rerank ceiling → 决定下一步是 lock-in / 重跑 BGE / 还是换方向。

# 假设

- **C_CF1**: 修复后 dense R@100 ≥ +0.030（撬动 ceiling）
- **C_CF2**: 修复后 graph rerank R@10 ≥ 0.72（直接突破 0.6913 历史天花板）
- **C_CF3**: 如果 C_CF1 成立但 C_CF2 不成立，BGE rerank 在新 corpus 上能达 R@10 ≥ 0.72

# 关键背景事实

| 指标 | 数值 |
|---|---|
| Figure passage 总数 | 1095 / 2809 |
| `[Image:]` 退化比例 | 695 (63.5%) |
| 退化但有 enrichment 可用 | **293 (42.2% of degraded)** |
| 全 figure 中有 enrichment 可用 | 664 (60.6%) |
| 历史最佳 R@10 (graph rerank) | 0.6913 since 4/17 |
| RRF (dense+CE) R@100 lift | +2.3pp (0.8636 → 0.8869) — 历史首次 ceiling 移动 |

# 实验块

- **Phase A** (15 min, 0 GPU): 诊断键映射错配，写 D1/D2/D3 三桶报告
- **Phase B** (10 min, 0 GPU): 最小补丁 + 重建 corpus，验证 `[Image:]` 比例 ≤ 25%
- **Phase C** (~10 min GPU): 重跑 dense baseline + graph rerank ceiling
- **Phase D** (~30 min GPU, 条件触发): 如 dense R@100 +≥2pp，重跑 BGE rerank + RRF
- **Phase E** (10 min): 按 D1–D5 决策树给单一推荐

# 决策树（顺序匹配，命中即停）

| 规则 | 触发 | 推荐 |
|---|---|---|
| D1 | g_R@10 ≥ +0.030 | corpus 修复独立打破 ceiling — lock in，重跑所有开放实验 |
| D2 | ce_R@10 ≥ 0.72 | corpus 修复 + BGE 达标 — 收 rerank 轨 |
| D3 | d_R@100 ≥ +0.030 但 R@10 不到 | dense 上限动了但 rerank 是新瓶颈 — 上 F1 (Qwen3-Reranker-4B) |
| D4 | d_R@100 < +0.010 | corpus bug 不是真瓶颈 — 跳 F1，换 reranker 家族或 query expansion |
| D5 | else | 部分 lift — 审 table/formula 是否有同类 bug |

# 文件

- 计划：[refine-logs/CORPUS_ENRICH_FIX_PLAN_20260503.md](../../refine-logs/CORPUS_ENRICH_FIX_PLAN_20260503.md)
- 代码：`scripts/diagnose_corpus_enrich_mapping.py`（新）+ patch 到 `scripts/build_graph_augmented_corpus.py`
- Slurm：`slurm_scripts/46_corpus_fix_rebaseline.sh`（新）
- 输出：`data/05_eval/corpus_fix_v1/`（diagnose / verify / delta / 各项 metrics）
- 决策报告：`refine-logs/CORPUS_FIX_DECISION_20260503.md`

# 与已有计划的关系

- VL enrich-only (R130–R133)：known answer，不阻塞
- Cross-doc citation (R140–R143)：副线，不阻塞
- CE rerank (job 66349)：本实验直接由其失败分析触发；如 D2 命中可在新 corpus 上重跑同 BGE 即可
- F1 (Qwen3-Reranker-4B) / F3 (HyDE)：均由本实验 Phase E 决策门触发

# 验收

1. Phase A 报告精确指出键格式错配规则
2. Phase B 重建 corpus 中 figure `[Image:]` ≤ 25%，enriched figure ≥ 690
3. Phase C delta 表覆盖 R@1/5/10/100/MRR × dense+graph
4. Phase E 决策报告含命中规则编号 + 触发数值 + 单一推荐

# Result (2026-05-03)

**Verdict: D5 — file as antipattern.** Detail in `refine-logs/CORPUS_FIX_DECISION_20260503.md`.

Phase A confirmed two independent bugs (D2=293, single coherent rule). Phase B
produced two variants: `corpus_fix_v1` (replace) and `corpus_fix_v2` (additive,
MODORA visual + graph paper-context concat). Phase C dense + graph rerank on
both (jobs 66371, 66384) showed regression vs the buggy 4/17 baseline:

| metric | anchor | fix_v1 | fix_v2 |
|---|---|---|---|
| dense R@10 | 0.6195 | 0.5106 (−10.9pp) | 0.5888 (**−3.1pp**) |
| dense R@100 | 0.8636 | 0.7569 (−10.7pp) | 0.8436 (**−2.0pp**) |
| graph_explicit_only static_plus_neighbor R@10 | 0.6913 | 0.5888 | **0.6860** (−0.5pp) |

Phase D **NOT triggered** (R@100 Δ < +2pp). Mechanism: MODORA visual
descriptions are domain-detached ("Histogram of small-valued metric") whereas
M4query_v1 queries are paper-domain text-style. Same direction as the BGE-CE
text-bias finding from `exp:20260503_ce_rerank_bge` — visual replacement is
net-noise on text-style retrieval.

**Code state**: `DEFAULT_ENRICHED_FILES` reverted (no MODORA by default).
Latent-bug fixes for `load_enriched_index` (`documents.elements` branch) and
`build_element_text` (additive priority) kept in place; they are no-ops under
default args.

**Recommendation forwarded**: do NOT pursue corpus-level visual enrichment
for text-style retrieval. Higher-EV next experiment is CE on the *anchor*
corpus with a non-text-biased reranker (e.g. multimodal CE) or explicit
late-fusion of a vision-language lane — not corpus-level text replacement.

