# Experiment Audit Report

**Date**: 2026-04-21  
**Auditor**: GPT-5.4 xhigh（cross-model, read-only）  
**Project**: chunk-as-retrieval-unit experiment（Job 62353 / 62358）  
**Executor**: Claude Sonnet 4.6

---

## Overall Verdict: ⚠️ WARN

---

## Checks

### A. Ground Truth Provenance: ✅ PASS
- `eval_dense_retrieval.py` 从磁盘 `qrels.jsonl` 加载 GT，只在 metric 计算中使用，无 model-output → GT 循环
- `build_chunk_corpus.py` 的 qrels 重映射基于 dataset qrels + 文档结构（position_idx），不依赖 retrieval 输出
- 无 fake GT 风险

### B. Corpus Comparability: ⚠️ WARN
- **不是同等条件对比**：chunk passage 比 element passage 更宽（段落文本 + element 内容注入），固有检索难度下降
- 57-doc 限制内部一致，但比 1147-doc 全量语料检索难度小
- **结论**：chunk 即使指标提升，也不能直接断言是"图结构更好"，部分来自 passage 内容更丰富

### C. qrels Remapping Integrity: ⚠️ WARN
- position_idx → paragraph_indices → chunk_id 的对齐逻辑合理，**但从未验证 position_idx 命名空间一致性**
- Dedup `(query_id, chunk_id)` 是必要的，但改变了任务：原来 2 个 element positives 合并成 1 个 chunk positive → positives/query 从 2.00 降到 1.87
- **小 bug**：`build_report.json` 的 `mapped_elements` 字段实际计的是 qrel 行数，不是 unique element 数（`n_mapped` 在 per-qrel 循环内递增）
- 4 个 unmapped elements 不是任何 query 的唯一正例（query coverage 保持 473/473），影响可忽略

### D. Graph Rerank Validity: ❌ FAIL
- **核心问题**：`eval_dense_retrieval.py` 只保存 PID 列表，不保存 cosine score。`eval_chunk_graph_rerank.py` 用 `n - rank` 替代，**不等价于 cosine score**
- 对比：`eval_graph_topk_rerank.py` 用的是带归一化的 rank score 且有 boost cap；chunk reranker 无 cap → neighbor propagation 可能过度
- `exp_only` 在 `elem_to_chunk` 覆盖低时退化到接近 baseline（graceful），但稀疏边可能因 raw score scale 过度 boost
- **fix 要求**：在 `eval_dense_retrieval.py` 的 ranking 输出中加 scores 字段

### E. Expected Results Analysis: ⚠️ WARN
- positives/query 从 2.00 降到 1.87（≈1.07× 机械 recall lift），加上更小 corpus + 更丰富 passage，**R@1 和 R@10 应双升**
- 部分提升是 unit 变换的机械效果，不能全归因于图信号

---

## 预测表（GPT-5.4 xhigh）

| Config | R@1 预测 | R@10 预测 | 推断逻辑 |
|--------|---------|---------|---------|
| **dense chunk_n400** | **0.26–0.29** | **0.64–0.69** | 更小 corpus + 更丰富 passage + 机械 recall lift |
| seq_only / static_prior | 0.25–0.28 | 0.64–0.70 | 顺序覆盖广，但 n-rank 让 static_prior 偏弱 |
| seq_only / static+neighbor | 0.22–0.27 | 0.68–0.75 | top-10 recall 升，精度可能牺牲 |
| exp_only / static_prior | 0.26–0.29 | 0.64–0.69 | projected 边稀疏，接近 dense baseline |
| exp_only / static+neighbor | 0.24–0.28 | 0.66–0.72 | 取决于 elem→chunk 覆盖率 |
| seq+exp / static_prior | 0.25–0.28 | 0.65–0.71 | 最保守 |
| **seq+exp / static+neighbor** | **0.22–0.27** | **0.69–0.77** | R@10 最高，R@1 可能因过度传播下降 |

**参考基线**：v1_enriched R@1=0.2389, R@10=0.5994, MRR=0.6081

**最可能的失败模式**：R@10 因更小 corpus 上升，但这是 unit 变换效果而非图增强效果；R@1 和 MRR flat or worse after graph rerank due to uncapped propagation。

---

## 必须修复的问题（在结果出来前）

| 优先级 | 问题 | 修复方案 |
|--------|------|---------|
| P0 | `eval_dense_retrieval.py` ranking 不含 cosine scores | 加 `--save-scores` 选项，输出 `{"query_id":..., "top_k":[...], "scores":[...]}` |
| P1 | `eval_chunk_graph_rerank.py` neighbor propagation 无上限 | 参照 `eval_graph_topk_rerank.py` 加 `boost_cap` 参数 |
| P2 | `build_report.json` 的 `mapped_elements` 字段语义不准确 | 区分 `mapped_qrel_rows` vs `mapped_unique_elements` |

---

## Claim Impact

- **"chunk 检索优于 element 检索"**：即使 R@1/R@10 双升，也需加限定语：在同等 57-doc 封闭评估下，且部分提升来自 passage 粒度变换。**不能直接等同于"图增强有效"**
- **"图 rerank 提升 chunk 检索"**：seq+exp / static+neighbor 的 R@10 提升可信，但 R@1 下降需要解释；需修复 cosine score 缺失问题后才能做强声明
- **"chunk_sequence 边有效"**：R@10 在 seq_only 下的提升是有效信号（顺序结构真实存在），R@1 牺牲是可接受的 trade-off
