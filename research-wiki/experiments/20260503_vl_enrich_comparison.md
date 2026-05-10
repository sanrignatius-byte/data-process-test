---
type: experiment
node_id: exp:20260503_vl_enrich_comparison
title: "VL Embedding controlled comparison — enrich-only text vs raw image"
date: 2026-05-03
status: planned
verdict: pending
related_experiments: [exp:20260502_split_modality, exp:20260503_split_modality_vl_t5_rerun]
related_claims: [C_VL1, C_VL2]
---

# 目的

Isolate VL image encoding's contribution: strip figure/table passages to enriched_content ONLY (no caption, no context_before/after, no raw content), then compare Qwen3-Embedding-4B (text) vs Qwen3-VL-Embedding-2B (image).

Previous per-modality numbers showed 4B text > VL on figure/table, but 4B had caption + context as text signal. This experiment gives both encoders the same information budget.

# 假设

- **C_VL1**: VL image encoding beats text enrichment alone on figures (R@10 +5pp)
- **C_VL2**: Better enrichment → smaller VL-text gap (negative correlation)

# 实验配置

| Config | figure text | table text | text passages | Encoder |
|--------|------------:|-----------:|--------------|---------|
| `text_enrich_only` | enriched_content only | enriched_content only | enriched_content only | Qwen3-Embedding-4B |
| `vl_image` | raw image | raw image | enriched_content only | Qwen3-VL-Embedding-2B |
| `text_full` (ref) | caption+ctx+enriched | caption+ctx+enriched | full text | Qwen3-Embedding-4B |

# 实验块

- **B1 (MUST)**: 三 config 对比，per-modality figure/table R@10。GPU ~25 min。
- **B2 (NICE)**: Enrichment 质量 vs VL-text delta 相关性分析。

# 成功标准

VL image R@10 > text_enrich_only R@10 by ≥ 5pp on figure → VL has irreducible value.
Otherwise: enrichment already captures all retrieval-relevant visual information (at VL-2B's capacity).

# 文件

- 计划：[refine-logs/EXPERIMENT_PLAN_VL_ENRICH_20260503.md](../../refine-logs/EXPERIMENT_PLAN_VL_ENRICH_20260503.md)
- 代码：待创建 `scripts/build_enrich_only_corpus.py`, `scripts/eval_vl_vs_text_enrich.py`
- Slurm：待创建 `slurm_scripts/43_vl_enrich_comparison.sh`
- 输出：`data/05_eval/dense_retrieval/vl_enrich_comparison/`
