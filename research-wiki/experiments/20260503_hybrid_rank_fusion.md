---
type: experiment
node_id: exp:20260503_hybrid_rank_fusion
title: "Modality routing ablation — find optimal per-modality encoder assignment + RRF merge"
date: 2026-05-03
status: planned
verdict: pending
related_experiments: [exp:20260502_split_modality, exp:20260503_split_modality_vl_t5_rerun]
related_claims: [C_R1, C_R2, C_R3]
---

# 目的

通过 routing ablation 找到每个 modality 的最优 encoder 分配，然后用 RRF 合并。**不预设 figure/table→VL**，因为 per-modality 证据显示 4B text 在 figure/table 上比 VL 更强。

# 关键 Per-Modality 基线（修正）

| System | figure R@10 | table R@10 | formula R@10 |
|--------|------------:|-----------:|-------------:|
| `split_4B_text` mixed | **0.5307** | **0.4985** | 0.3017 |
| `split_4B_text` split | **0.7128** | 0.0000 | 0.0000 |
| `split_VL_2B_t5` mixed | 0.4102 | 0.0236 | **0.3352** |
| `split_VL_2B_t5` split | 0.5390 | 0.0000 | 0.0000 |

**核心发现**：
1. 4B text 在 figure 上 **强于** VL（0.71 split > 0.54 split，0.53 mixed > 0.41 mixed）
2. 4B text 在 table 上 **碾压** VL（0.50 mixed >> 0.02 mixed）
3. VL 仅在 formula mixed 模式有微弱优势（0.34 vs 0.30）
4. Split index 对 table/formula 是致命的（两个 encoder 都归零）

# 假设

- **C_R1**: 最优 routing + RRF R@10 ≥ 0.50，beat split_4B_text mixed (0.4767)
- **C_R2**: 4B text 是强主线，figure/table→4B ≥ figure/table→VL
- **C_R3**: Mixed index > split index（table/formula 不被杀）

# 待测 Routing Configs

| Config | figure | table | formula | text | 假设 |
|--------|--------|-------|---------|------|------|
| `r_4b_all` | 4B | 4B | 4B | 4B | Baseline (=split_4B_text mixed) |
| `r_vl_fig_tab` | VL | VL | 4B | 4B | 原始（错误）假设 |
| `r_4b_fig_tab` | 4B | 4B | VL | 4B | **新最佳猜测** |
| `r_vl_formula_only` | 4B | 4B | VL | 4B | 最小 VL 介入 |
| `r_vl_all_nontext` | VL | VL | VL | 4B | Pure VL split baseline |

# 文件

- 实验计划：[refine-logs/EXPERIMENT_PLAN_HYBRID_20260503.md](../../refine-logs/EXPERIMENT_PLAN_HYBRID_20260503.md)（已修正为 routing ablation）
- Tracker: [refine-logs/EXPERIMENT_TRACKER.md](../../refine-logs/EXPERIMENT_TRACKER.md)
- 代码：待创建 `scripts/eval_routing_ablation.py`
- Slurm：待创建 `slurm_scripts/42_routing_ablation.sh`
