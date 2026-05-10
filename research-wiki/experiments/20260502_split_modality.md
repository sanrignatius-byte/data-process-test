# Split Modality 分离式检索实验 (2026-05-02)

## 动机

Mentor 录音 60 建议：不要把所有模态塞进同一个 dense 空间。文本 query 应该查文本 corpus，figure/table query 用多模态 embedding 独立检索。目前所有检索实验都是 unified retrieval（所有 corpus passages 混在一起），模态间的 embedding 质量差异没有被显式处理。

## 实验设计

`scripts/eval_split_modality_retrieval.py`：

| 组件 | 描述 |
|------|------|
| 数据 | M4query_v1（473 queries, 2809 corpus, 946 qrels） |
| 模型 | Qwen3-Embedding-0.6B + Qwen3-Embedding-4B |
| Corpus 分片 | figure (1095) / formula (1253) / table (237) / text (224) |
| 检索策略 | query 按 evidence 类型路由到对应模态的 corpus 分片 |
| 指标 | R@1 / R@2 / R@5 / R@10 / R@100 / MRR |

### 与 Unified Retrieval 的区别

```
Unified:  query → [所有 2809 passages] → top-k
Split:    query → if evidence=figure → [1095 figure passages] → top-k_figure
                + if evidence=table  → [237 table passages]  → top-k_table
                + if text query      → [224 text passages]    → top-k_text
                → merge & re-rank → final top-k
```

## 运行状态

- **Job 66036** — slurm_scripts/41_split_modality_eval.sh，gpu-a6000-1
- 状态：✅ COMPLETED（最终成功 job: 66048；66036 OOM，66041 缺 faiss）

## 预期对照

| 配置 | 来源 | R@10 |
|------|------|------|
| v1_enriched unified (0.6B) | eval_report_v1_enriched.json | 0.1628 |
| v2_chunks unified (0.6B) | eval_report_v2_chunks.json | 0.1126 |
| split_modality mixed-index (0.6B) | **本实验** | 0.4302 |
| split_modality best split (0.6B) | **本实验** | 0.3288 |
| v1_enriched unified (4B) | rebuilt_20260417 | **0.6195** |
| v2_chunks unified (4B) | rebuilt_20260417 | 0.5085 |
| split_modality mixed-index (4B) | **本实验** | 0.4767 |
| split_modality best split (4B) | **本实验** | 0.3235 |

## 假设

1. **H1**：split modality 在 0.6B 下应优于 unified（减少跨模态噪音）
2. **H2**：4B 下 unified 已经较强，split 的边际收益可能较小
3. **H3**：figure/table corpus 的 embedding 质量仍是天花板（图片无 VLM caption 时 embedding 弱）

## 结论（2026-05-03 更新）

- **H1 不成立**：0.6B split 后 R@10 从 mixed-index 0.4302 降到 0.3288；但 mixed-index 本身比旧 `v1_enriched unified (0.6B)=0.1628` 强，说明语料/流程口径不同，不能直接把 split 当唯一变量解释。
- **H2 成立**：4B mixed-index 已强，split 后明显下降（0.4767 → 0.3235），说明简单按 modality 分配 top-k 会牺牲 top-10 精度。
- **H3 成立**：text-only embedding 对 figure/table 的 `[Image: xxx]` 占位符基本无语义；后续 VL rerun 见 [exp:20260503_split_modality_vl_t5_rerun](20260503_split_modality_vl_t5_rerun.md)。
- 最终路线：不要做"所有 modality 一个 split rule"；改成 hybrid lane：figure/table 用 VL 或 caption，formula/text 用 text embedding，再用 rank fusion 合并。

## Related

- [exp:20260502_chunk_element_coverage] — chunk→element 覆盖分析
- [exp:20260417_dense_baseline_rebuilt] — 4B dense baseline
- [exp:20260421_chunk_as_retrieval_unit] — chunk 检索单元实验
