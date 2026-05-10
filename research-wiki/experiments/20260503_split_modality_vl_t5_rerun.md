---
type: experiment
node_id: exp:20260503_split_modality_vl_t5_rerun
title: "Split-modality retrieval re-run with transformers ≥ 5.2 (Qwen3-VL-Embedding-2B)"
date: 2026-05-03
status: completed
verdict: environment_fixed_model_usable_but_split_vl_underperforms_text4b
related_experiments: [exp:20260502_split_modality_vl_failed]
related_claims: []
---

# 目的

修复 [exp:20260502_split_modality_vl_failed](20260502_split_modality_vl_failed.md) 的根因——重跑同一份 Qwen3-VL-Embedding-2B 实验，唯一区别是 conda env 用 `transformers ≥ 5.2`。

# 根因复述

5/2 时把 R@10=0.0021 解释成"checkpoint 只发布 vision encoder"。复查后发现：

- HF 官方 [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) 文件清单只有**一个** 4.26 GB `model.safetensors`，是 **SentenceTransformer 包装**（modules.json 含 Transformer + Pooling + Normalize）。
- 本地 safetensors 共 **625 个 key**，全部以 `model.` 为前缀（含完整 28 层 `model.language_model.layers.*` + 24 个 `model.visual.blocks.*`）。语言塔权重**完整存在**。
- Job 66114 警告原文："Some weights of Qwen3VLModel were not initialized... `language_model.layers.0.*`..." —— 期望键是 `language_model.*`，文件键是 `model.language_model.*`，**多了一层 `model.` 前缀没被剥**。
- 我们环境 `transformers==4.57.6`、Qwen 官方 PR #19 标题 "making the script compatible with **transformers 5.2+**"。
- 随机文本编码器 + valid 视觉空间 → R@10 ≈ k/N ≈ 100/2809 ≈ 0.0036，与观测 0.0021 完全吻合。

# 修复方案

不污染 minerU env（vllm + compressed-tensors 钉死 transformers 4.57.6）：
1. **overlay**：`pip install --target /projects/myyyx1/envs/qwen3vl_tf5_overlay 'transformers>=5.2,<6'`
2. **runtime shadowing**：`PYTHONPATH=/projects/myyyx1/envs/qwen3vl_tf5_overlay:$PYTHONPATH`
3. **submit**：[slurm_scripts/41b_split_modality_vl_t5.sh](../../slurm_scripts/41b_split_modality_vl_t5.sh)（与 41 完全一致，除运行时优先加载 transformers 5.x）

# Setup 状态

- [x] HF 页面 + 本地 safetensors 验证根因（PR #19、官方文档）
- [x] [scripts/eval_split_modality_vl.py](../../scripts/eval_split_modality_vl.py) 不需改（已用 SentenceTransformer.encode 正确路径）
- [x] [slurm_scripts/41b_split_modality_vl_t5.sh](../../slurm_scripts/41b_split_modality_vl_t5.sh) 创建
- [x] overlay 安装完成：`/projects/myyyx1/envs/qwen3vl_tf5_overlay`，import sanity 看到 `transformers 5.7.0`
- [x] sanity test: load 后无 "newly initialized" warning（625/625 weights cleanly loaded）
- [x] sbatch submitted: Job `66243` on `gpu-a6000-1` → failed after clean weight load due old `mistral_common` missing `ReasoningEffort`
- [x] overlay dependency repaired: installed `mistral-common 1.11.1`
- [x] sbatch resubmitted: Job `66244` → failed in sanity import due `pydantic-core` mismatch
- [x] overlay dependency repaired: installed `pydantic-core==2.46.3`; `ReasoningEffort` import passed
- [x] sbatch resubmitted: Job `66248`
- [x] 等结果
- [x] eval_report.json + 与 Job 66114 / 66048 / v1_enriched_4B 对比

# 结果（Job 66248）

## 环境修复是否成功

成功。

- runtime: `transformers=5.7.0`, `sentence_transformers=5.4.1`
- `transformers_path=/projects/myyyx1/envs/qwen3vl_tf5_overlay/transformers/__init__.py`
- `mistral_common_ReasoningEffort=ReasoningEffort`
- 625/625 weights loaded；`logs/41b_split_modality_vl_t5_66248.err` 中无 `newly initialized` / `not initialized` warning。
- Query embeddings shape = `(473, 2048)`；image encoded = 1080；text passages encoded = 1729。

## 指标对比

| Config | R@1 | R@5 | R@10 | R@100 | MRR |
|--------|----:|----:|-----:|------:|----:|
| `v1_enriched_4B` unified | 0.2336 | 0.5275 | **0.6195** | 0.8636 | 0.6121 |
| `split_4B_text` (Job 66048) | 0.1934 | 0.3964 | 0.4767 | 0.7526 | 0.4995 |
| `split_VL_2B_t4_failed` (Job 66114) | 0.0011 | 0.0011 | 0.0021 | 0.0074 | 0.0028 |
| `split_VL_2B_t5` mixed-index baseline | 0.1205 | 0.2326 | 0.2579 | 0.4123 | 0.3217 |
| `split_VL_2B_t5` best split (`equal_split`) | 0.1406 | 0.2209 | 0.2442 | 0.6723 | 0.3633 |

Per-modality R@10 under best split:

| Type | Mixed-index R@10 | Split R@10 | Delta |
|------|-----------------:|-----------:|------:|
| figure | 0.4112 | **0.5397** | +0.1285 |
| table | 0.0236 | 0.0000 | -0.0236 |
| formula | 0.3352 | 0.0000 | -0.3352 |

# 结论

1. **5/2 误诊已纠正**：Qwen3-VL-Embedding-2B checkpoint 是可加载的；Job 66114 的 near-random R@10=0.0021 是 transformers 4.x / dependency 环境问题，不是 checkpoint 缺权重。
2. **模型不是随机，但不够强**：修复后 R@10 从 0.0021 升到 0.2579，证明 text/image embedding 正常工作；但仍显著低于 `split_4B_text` 的 0.4767 和 unified 4B 的 0.6195。
3. **VL 主要救了 figure**：figure split R@10 从 mixed 0.4112 → 0.5397；但 table 仍近乎不可用，formula 在 split 下归零。
4. **当前 split allocation 逻辑不适合 formula**：formula 是 LaTeX 文本，强行按 modality split 后与文本/图像共同排序的分配策略会牺牲它；下一步应把 formula 归入 text-like lane，或为 formula 单独使用 Qwen3-Embedding-4B。
5. **mentor 录音 60 的方向仍成立一半**：多模态 embedding 对 figure 有正收益，但"所有非文本都用 VL split"不是当前最佳方案。

推荐下一步：

- `figure/table`: 继续 VL lane，但 table 需要 image crop/caption 质量审计。
- `formula/text`: 回到 `Qwen3-Embedding-4B` text lane，而不是 Qwen3-VL-Embedding-2B。
- rerun hybrid split：`figure/table -> Qwen3-VL-Embedding-2B`，`formula/text -> Qwen3-Embedding-4B`，merge 时不要跨 embedding 空间直接比较分数，改用 rank fusion / reciprocal rank fusion。

# 期望结果

| Config | R@10 (Job 66114) | R@10 (期望 t5 rerun) |
|--------|------------------:|---------------------:|
| split_VL_2B (figure+table 真图编码) | 0.0021 | **>0.40**（接近 split_4B_text 0.4767） |
| per-modality figure | 0.0093 | **>0.20** |
| per-modality table | 0.0000 | **>0.20** |
| per-modality formula | 0.0000 | 持平（公式仍用文本编码） |

实际结果 R@10=0.2579，处于两者之间：checkpoint / environment 已修复，但当前 Qwen3-VL-Embedding-2B + split allocation 不足以作为主线检索方案。

# 文件

- 模型：`/projects/myyyx1/models/Qwen3-VL-Embedding-2B`（4.26 GB safetensors，已下载）
- transformers 5 overlay：`/projects/myyyx1/envs/qwen3vl_tf5_overlay`
- 提交脚本：[slurm_scripts/41b_split_modality_vl_t5.sh](../../slurm_scripts/41b_split_modality_vl_t5.sh)
- 评估代码：[scripts/eval_split_modality_vl.py](../../scripts/eval_split_modality_vl.py)（无变动）
- 输出 dir：`data/05_eval/dense_retrieval/split_modality_vl_t5/`
- Failed setup job: `66243`, logs `logs/41b_split_modality_vl_t5_66243.{out,err}`
- Failed setup job: `66244`, logs `logs/41b_split_modality_vl_t5_66244.{out,err}`
- Completed rerun: `66248`, logs `logs/41b_split_modality_vl_t5_66248.{out,err}`
