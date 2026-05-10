---
type: experiment
node_id: exp:20260502_split_modality_vl_failed
title: "Split-modality retrieval with Qwen3-VL-Embedding-2B — transformers 4.x load failure"
date: 2026-05-02
status: completed
verdict: superseded_by_transformers5_rerun
related_claims: []
---

# Purpose

Following mentor 录音 60 (2026-05-02), test whether replacing the text-only
split_modality eval (Job 66048, where figure/table/formula encode their `[Image: xxx.jpg]`
placeholders as text and collapse to R@10≈0) with a true VL embedding model rescues
non-text passage recall. Use `Qwen3-VL-Embedding-2B` so all four modalities live in a
unified 2048-dim space and a text query can match images via cross-modal alignment.

## Setup (consistency with Job 66048 baseline)

- **Corpus**: `data/03_queries/M4query_v1` (2809 passages — figure 1095 / formula 1253 / table 237 / text 224)
- **Queries**: 473
- **Qrels**: 946
- **Merge configs**: equal_split / prop_to_corpus / boost_formula / boost_figure (identical to 66048)
- **top_k = 100**, **batch_size = 16**
- **Model**: `/projects/myyyx1/models/Qwen3-VL-Embedding-2B`
- **Code**: [scripts/eval_split_modality_vl.py](../../scripts/eval_split_modality_vl.py),
  [slurm_scripts/41_split_modality_vl.sh](../../slurm_scripts/41_split_modality_vl.sh)

## Run history

| Job | Status | Issue |
|-----|--------|-------|
| 66105 | FAILED | `model.max_seq_length=512` truncated Qwen3-VL's ~1247 image tokens to 496, causing `Mismatch in image token count between text and input_ids. Got ids=[496] and text=[1247]` |
| 66114 | COMPLETED | Bug fixed: split `--text-max-length=512` and `--image-max-length=4096`, only image pass bumps `model.max_seq_length`. Encoded successfully but produces near-zero retrieval. |

## Results (Job 66114)

| Config | R@1 | R@5 | R@10 | R@100 | MRR |
|--------|----:|----:|-----:|------:|----:|
| `v1_enriched_4B` (Qwen3-Embedding-4B baseline) | 0.2336 | 0.5275 | **0.6195** | 0.8636 | 0.6121 |
| `split_4B_text` (Job 66048) | 0.1934 | 0.3964 | 0.4767 | 0.7526 | 0.4995 |
| **`split_VL_2B` (this run)** | **0.0011** | **0.0011** | **0.0021** | 0.0074 | 0.0028 |

### Per-modality split (best merge `boost_figure`)

| Type | Baseline R@10 | Split R@10 |
|------|--------------:|-----------:|
| figure | 0.0047 | 0.0093 |
| table | 0.0000 | 0.0000 |
| formula | 0.0000 | 0.0000 |

## Root cause

> ⚠️ **2026-05-03 CORRECTION**: 上面 5/2 当天的"checkpoint 只发布 vision encoder"
> 解释**是错的**。复查后真正根因是 transformers 版本不匹配。

### Real root cause (2026-05-03 verified)

- HF 官方 [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) 只有**一个** 4.26 GB `model.safetensors`，是 SentenceTransformer 包装。
- 本地 safetensors 共 625 个 key，**全部以 `model.` 为前缀**，含完整 28 层 `model.language_model.layers.*` + 24 个 `model.visual.blocks.*`。语言塔权重**完整存在**。
- 加载警告原文期望键是 `language_model.layers.0.*`（无 `model.` 前缀），文件实际键是 `model.language_model.layers.0.*`。**多了一层 `model.` 前缀，transformers 4.x 没剥**。
- Qwen 官方 PR #19 标题 "making the script compatible with **transformers 5.2+**"，确认模型需要 transformers ≥ 5.2。
- 我们环境 `transformers==4.57.6`，`vllm` + `compressed-tensors` 钉死该版本。
- 随机文本编码器 + valid 视觉空间 → R@10 ≈ k/N ≈ 100/2809 ≈ 0.0036，与观测 0.0021 完全吻合。

### Original wrong explanation (kept for historical record)

> Qwen3-VL-Embedding-2B's HuggingFace checkpoint only ships the vision encoder
> weights; the language_model decoder is reported as "newly initialized" at load
> time:
>
> ```
> Some weights of Qwen3VLModel were not initialized from the model checkpoint at
> /projects/myyyx1/models/Qwen3-VL-Embedding-2B and are newly initialized:
> ['language_model.embed_tokens.weight', 'language_model.layers.0.input_layernorm.weight',
>  ... (all 28 decoder layers + embeddings)]
> ```

## Verdict (UPDATED 2026-05-03)

Qwen3-VL-Embedding-2B as published is **loadable and usable**, just not under
transformers 4.x. See [exp:20260503_split_modality_vl_t5_rerun](20260503_split_modality_vl_t5_rerun.md)
for the rerun under transformers 5 overlay. The rerun proves the environment fix
works (`R@10 0.0021 → 0.2579`), but also shows this **pure VL split** is still
weaker than `split_4B_text R@10=0.4767` and `v1_enriched_4B R@10=0.6195`.

~~`Qwen3-VL-Embedding-2B` as published is **unusable** for any text-query → multimodal
retrieval task. Job 66114 is the negative control proving this; do not cite it as a
retrieval baseline, only as a model-availability finding.~~ (superseded 2026-05-03)

## Implications and next-step options

The split-modality direction (mentor 录音 60 requirement #3) is not invalidated.
The updated lesson from Job 66248 is: use VL selectively, not for all modalities.
Options to surface for next mentor sync:

1. **Hybrid rank fusion** — figure/table use Qwen3-VL-Embedding-2B; formula/text
   use Qwen3-Embedding-4B. Do not raw-score merge across embedding spaces; use RRF
   or rank-normalized fusion.
2. **Caption-then-text-embed** — use a VLM to generate captions/descriptions for
   figure/table passages, then encode everything (queries + captioned passages) with
   `Qwen3-Embedding-4B` (already validated at R@10=0.6195). This is essentially the
   `enriched_content` pathway already wired into `enrich_elements_modora.py`; the
   M4query_v1 corpus may already have these for some elements — needs an audit.
3. **Alternative VL model** — try `jinaai/jina-clip-v2`, `BGE-VL`, or
   `GME-Qwen2-VL` only if hybrid rank fusion still underperforms. Same eval pipeline,
   swap the model path.

## Files touched

- [scripts/eval_split_modality_vl.py](../../scripts/eval_split_modality_vl.py) — added `--text-max-length` / `--image-max-length`, applied per-pass max_seq_length swap
- [slurm_scripts/41_split_modality_vl.sh](../../slurm_scripts/41_split_modality_vl.sh) — uses new args

## Artifacts

- `data/05_eval/dense_retrieval/split_modality_vl/eval_report.json`
- `data/05_eval/dense_retrieval/split_modality_vl/ranking_*.jsonl`
- `logs/41_split_modality_vl_66114.{out,err}`
