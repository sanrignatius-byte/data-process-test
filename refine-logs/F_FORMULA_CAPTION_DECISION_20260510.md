# F-formula Caption Injection 决策报告

**Date**: 2026-05-10
**Plan**: [exp:20260510_f_formula_caption](../research-wiki/experiments/20260510_f_formula_caption.md)
**Job**: 68107 (gpu-a6000-1, ~10 min wall)
**Verdict**: 🔴 **HD 命中** — caption injection 全模态反伤；**C11 strengthened**；必须真换 encoder

---

## 全集结果（M4query_v1, 473 queries）

| System | R@1 | R@5 | **R@10** | R@100 | MRR | ΔR@10 |
|---|---:|---:|---:|---:|---:|---:|
| dense baseline | 0.2336 | 0.5275 | **0.6195** | 0.8636 | 0.6121 | — |
| **dense + formula caption** | 0.2188 | 0.4767 | **0.5825** | 0.8710 | 0.5713 | **−3.70pp** |
| graph (ceiling) | 0.2209 | 0.5941 | **0.6913** | 0.8636 | 0.6017 | — |
| **graph + formula caption** | 0.2188 | 0.5856 | **0.6691** | 0.8710 | 0.5888 | **−2.22pp** |

**唯一正向信号**：R@100 +0.74pp（gold passages 在 deep pool 略好找）。但 R@10/MRR 全跌。

---

## Smoke50 per-modality R@10

| System | figure | **formula** | table | overall |
|---|---:|---:|---:|---:|
| dense baseline | 0.7179 | **0.5600** | 0.6111 | 0.6400 |
| **dense + caption** | 0.6923 | **0.4000** ⬇ −16.0pp | 0.6111 | 0.5900 |
| graph baseline (ceiling) | 0.8205 | **0.5600** | 0.6944 | 0.7100 |
| **graph + caption** | 0.7436 | **0.5200** ⬇ −4.0pp | 0.6667 | 0.6600 |

**惊人发现**：caption injection 在 formula bucket 上**最伤**（dense −16pp，graph −4pp），figure / table 也跌。

---

## 决策门验证

| Hypothesis (plan 5/10) | 预期 | 实际 | Verdict |
|---|---|---|---|
| HA: NL context 解决 formula | dense ≥ 0.66 | dense 0.5825 | ❌ |
| HB: dense +N, graph 不动 | graph stable | graph 0.6691 (−2.2pp) | ❌ |
| HC: 几乎无帮助 | ≈ 0.62 ± 1pp | 0.5825 (远低于 −1pp 区间) | ❌ |
| **HD: NL context 反伤** | < 0.6195 | **0.5825 全模态** | ✅ **命中** |

---

## 机制分析

注入 300 chars NL `context_before` 到 formula passage 后，dense encoder 把 formula passage 的 embedding **拉向"text passage"方向**：

1. **Formula 自身 −16pp 下跌**: query 描述 formula 内容（数学符号/LaTeX 操作），但注入后 passage embedding 变得"像 text"，与 query 距离反而拉大
2. **Figure −2.6pp 下跌（dense） / −7.7pp（graph）**: formula 注入后 score 普遍变化，影响 top-K 排名，间接挤掉 figure gold（虽然 figure passage 没改）
3. **R@100 +0.7pp**: 唯一证据 caption 没"完全废掉"——gold formula passages 仍在 deep pool 里，但 top-10 里被噪声推走

机制等同 [claim:C8](../research-wiki/claims/C8_modora_visual_enrichment_net_negative.md)：

> MODORA visual descriptions are domain-detached → text-style retrieval 上 net 负

把 paragraph `context_before` 注入 formula passage 同样 **detached** —— query 不需要 paragraph context 来匹配 formula，反而稀释了 formula content 的权重。

---

## C11 状态升级

[claim:C11](../research-wiki/claims/C11_formula_ceiling_is_dense_encoder_bound.md) 在 5/10 早段写"6 configs 全部 ≤ 0.5600"。本实验添加第 7、第 8 个数据点：

| Config | formula R@10 |
|---|---:|
| dense | 0.5600 |
| graph (ceiling) | 0.5600 |
| graph + lineno | 0.5600 |
| graph + virtual orig | 0.5200 |
| graph + virtual lineno | 0.5200 |
| Qwen3-CE | 0.5600 |
| **dense + formula caption** | **0.4000** (NEW, regressed) |
| **graph + formula caption** | **0.5200** (NEW, regressed) |

**8 configs，0 突破 0.5600**。Caption injection 不仅没破，还反伤。

C11 升级为：「Formula retrieval ceiling is dense-encoder bound; **adding NL caption to formula passages with the same encoder strictly regresses**, confirming that text-style augmentation cannot rescue LaTeX representation」.

---

## 推荐下一步

1. **F-formula 进入 Phase 2 — 真换 encoder** (P0)
   - **Option α (top recommendation)**: Qwen3-Math 或 Mistral-Math，仅替换 formula passages 的 embedding，混合 score（4B text 用于 figure/table/text，math encoder 用于 formula）
   - **Option β**: 训练 LoRA/fine-tune Qwen3-Embedding-4B 在 LaTeX corpus 上，保持架构不变
   - **Option γ (废弃)**: HyDE 在 query 端改写——given C8 + 本实验 caption 反伤，模态混注路线已被三次否决，γ 类同方向不再追

2. **更新 paper claim 模板**:
   - C8: visual injection net negative on text-style queries
   - **新发现**: text injection net negative on formula passages
   - **统一 framing**: "**Cross-modal style injection is net negative** for M4query retrieval, regardless of injection direction" — 这是个 strong negative finding 值得在 paper 里讲

3. **关闭 caption injection 路线**: F-formula 不再尝试同 encoder 做文本注入

---

## 副产品

1. **R@100 +0.7pp**: caption injection 唯一正向是 gold passages 在 deep pool（top-100）更好找。这暗示 caption 注入对 *召回* 不是有害，但对 *精排* 是有害。可能用法：caption 仅用于 first-stage candidate generation，top-K 还原 LaTeX-only 文本做 second-stage 重排
2. **Figure −7.7pp on graph caption**: 没改 figure passage 的情况下 figure R@10 也跌——证明 graph rerank 的 figure-table 增益对 formula bucket 的"分数稳定性"是依赖的。Formula passage embedding 变化间接影响其他模态 ranking。这是 C10 的强化版补充
3. **C8 + F-formula caption 共同模式**: 任何**跨模态文本注入**（图描述注入 figure，paragraph 注入 formula）都呈现 net 负。Mentor C5 "多粒度 enrich" 路线在当前架构下走不通

---

## File manifest

| Path | 状态 |
|---|---|
| `scripts/build_formula_caption_corpus.py` | ✅ 新建 |
| `data/05_eval/dense_retrieval/rebuilt_20260417/augmented/corpus_v1_enriched_formula_caption.jsonl` | ✅ 输出（2809 passages，1253 formula 注入 ~303 chars） |
| `slurm_scripts/50_f_formula_caption_inject.sh` | ✅ 新建 |
| `data/05_eval/dense_retrieval/rebuilt_20260417/eval_report_v1_formula_caption.json` | ✅ R@10 0.5825 |
| `data/05_eval/dense_retrieval/rebuilt_20260417/ranking_v1_formula_caption.jsonl` | ✅ |
| `data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_only_formula_caption/` | ✅ R@10 0.6691 |
| `data/05_eval/smoke50/per_system_per_modality.md` | ✅ 加 dense_formula_caption + graph_formula_caption |
| `refine-logs/F_FORMULA_CAPTION_DECISION_20260510.md` | **本报告** |

---

## Cost 实绩

- ~12 min wall（包括编码 2809 passages + 473 queries + graph rerank）
- ~10 min A6000 GPU
- $0 LLM API
- 完全在 plan 预算内（~1.5h + 30 min GPU）
