---
type: experiment
node_id: exp:20260510_f_formula_caption
title: "F-formula caption injection — test claim:C11 with NL context"
date: 2026-05-10
status: completed
verdict: HD_caption_injection_regresses_C11_strengthened
related_experiments: [exp:20260505_smoke50_balanced_audit, exp:20260510_b1_phase2_lineno]
related_claims: [C5, C8, C10, C11]
---

# 执行结果（2026-05-10）

## TL;DR

- **HD verdict 命中**: caption injection 反伤每个模态。dense R@10 0.6195 → 0.5825 (−3.7pp); graph 0.6913 → 0.6691 (−2.2pp)
- **Formula bucket 跌得最厉害**: dense −16pp (0.5600 → 0.4000), graph −4pp (0.5600 → 0.5200)
- **8 configs 在 formula 上 0 突破 0.5600**——3 个甚至 regressed
- [claim:C11](../claims/C11_formula_ceiling_is_dense_encoder_bound.md) 强化：text-style augmentation strictly cannot rescue LaTeX representation
- 与 [claim:C8](../claims/C8_modora_visual_enrichment_net_negative.md) 同向：cross-modal style injection 全方向 net 负
- **F-formula Phase 2 必须真换 encoder**（Qwen3-Math / Mistral-Math），不再追 caption injection / HyDE

## 决策报告
[refine-logs/F_FORMULA_CAPTION_DECISION_20260510.md](../../refine-logs/F_FORMULA_CAPTION_DECISION_20260510.md)

---

# 目的

验证 [claim:C11](../claims/C11_formula_ceiling_is_dense_encoder_bound.md) 的细节预测：
- **C11 says**: formula R@10 ≈ 0.56 ceiling 是 dense encoder on LaTeX 的 representation bound
- **Caption injection 是软测试**: 若 C11 严格成立，给 formula passage 附加 NL context 后用同一 encoder（Qwen3-Embedding-4B）re-encode，R@10 不应突破 0.56
- **若 caption injection 帮 +N pp**: C11 部分弱化，意味着同一 encoder + 文本上下文足以 narrow gap，不需要换 math-aware encoder

这条路线唯一不需要 LLM API（已死）+ 不需要新模型，是**最便宜的 F-formula 测试**。

# 设计

| 配置 | Corpus | Encoder |
|---|---|---|
| Baseline | `corpus_v1_enriched.jsonl`（formula text = `[FORMULA] LaTeX`） | Qwen3-Embedding-4B |
| **F-formula caption** | `corpus_v1_enriched_formula_caption.jsonl`（formula text = `[FORMULA] LaTeX | Context: <last 300 chars of context_before>`） | Qwen3-Embedding-4B (same) |

仅 formula passage 的 text 改变，其他模态 corpus 不变。Encoder / queries / qrels 完全不变。

# 实现

`scripts/build_formula_caption_corpus.py`:
- 读 `multimodal_elements.json` 取每个 formula 的 `context_before`（last 300 chars）
- 1054 formula 通过 element_id 直接匹配（85.4%），199 通过 content jaccard ≥ 0.30 匹配（15.9%），仅 5 个无 context 可注入（0.4%）
- 平均每 formula passage 注入 303 chars NL 文本
- 输出 corpus 总量不变（2809 passages），仅 formula 文本变长

`slurm_scripts/50_f_formula_caption_inject.sh`:
- Step 1: dense retrieval on caption corpus (`eval_dense_retrieval.py` + Qwen3-Embedding-4B, ~5 min A6000)
- Step 2: graph rerank explicit-only (CPU, ~1 min)

Job: 68107 (gpu-a6000-1, submitted 2026-05-10)

# 假设 + 决策门

| Hypothesis | Caption R@10 vs Baseline 0.6195 | C11 stance | Decision |
|---|---|---|---|
| **HA**: NL context 给同一 encoder 加足够信号 | ≥ 0.66 (+4pp) | C11 partially refuted | F-formula 完成；不需 math encoder |
| **HB**: NL context 帮 dense 但 graph rerank 已饱和 | dense +N, graph 0.6913 unchanged | C10 strengthened, C11 unclear | 测 graph rerank delta |
| **HC**: NL context 几乎无帮助（< +1pp） | 0.62 ± 1pp | **C11 strongly supported** | 必须真换 encoder（Qwen3-Math 或 OPT-LaTeX）|
| **HD**: NL context 反伤 | < 0.6195 | 注入策略劣化 | 跟 C8 同向；text-style retrieval 可能对 formula 多说话有意见 |

Per-modality smoke50 重新 slice：观察 formula 25 qrels R@10 vs baseline 0.5600。

# 与 mentor C5 的对照

Mentor C5 原话：「多粒度 enrich，粗 summary + 细描述，仿 DocResearcher」。本实验是 **C5 最便宜版本**：
- 不依赖 LLM API（API 19 天死）
- 不引入 visual 描述（[claim:C8](../claims/C8_modora_visual_enrichment_net_negative.md) 已证 visual 注入 net 负）
- 仅用已有 mineru `context_before` 文本（已存储）
- 仅 formula 注入（其他模态保持不变，避免普遍稀释）

若 HA / HB 命中，可写成 paper claim "minimal-cost text caption injection on formula passages restores X pp without encoder swap"；若 HC，C11 升级为强 claim。

# 文件

- 计划：本文档
- Corpus build script: `scripts/build_formula_caption_corpus.py`
- Slurm: `slurm_scripts/50_f_formula_caption_inject.sh`
- 输入: `data/05_eval/dense_retrieval/rebuilt_20260417/augmented/corpus_v1_enriched_formula_caption.jsonl` (2809 passages, 1253 formula 注入了 ~303 chars NL context)
- 期待输出:
  - `data/05_eval/dense_retrieval/rebuilt_20260417/eval_report_v1_formula_caption.json`
  - `data/05_eval/dense_retrieval/rebuilt_20260417/ranking_v1_formula_caption.jsonl`
  - `data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_only_formula_caption/metrics_graph_static_plus_neighbor.json`

# 待补 (job 完成后)

1. 全集 R@1 / R@5 / R@10 / R@100 / MRR vs baseline + 0.6913 ceiling
2. 用 `eval_smoke50_slice.py` 加新 system，跑 per-modality
3. 决策门 HA/HB/HC/HD 哪一个命中
4. 更新 [claim:C11](../claims/C11_formula_ceiling_is_dense_encoder_bound.md) 状态
5. 是否进 F-formula Phase 2（真换 encoder）由本结果决定
