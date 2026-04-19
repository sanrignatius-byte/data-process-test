# Eval Changelog

用于记录所有会影响实验可比性的评估口径变更。

## Entry template
- Date:
- Version tag:
- Changed by:
- Change type: dataset | chunking | metric | threshold | candidate generation | qc rule
- Before:
- After:
- Reason:
- Expected impact:
- Related commit:

---

## Entries

### eval-v1.1 — 2026-03-12
- Version tag: eval-v1.1
- Change type: metric + chunking
- Before: hit 判定 = span 字面文本 substring overlap ≥ 0.5；chunk 字段 = caption + content + context_before + context_after
- After: hit 判定优先使用 `required_evidence_spans[].element_id` 做 element_id 精确匹配（当 gt_eids 非空时）；chunk 字段新增 `enriched_title` + `enriched_content`（MoDora enriched）
- Reason: 诊断发现语料 span 字面覆盖率只有 14%——`required_evidence_spans` 里的 span 是 Claude 看图生成的视觉内容字面文本（表格单元格值、图中标注文字），这些文字不在 caption/context 字段里。span overlap 方法导致正确 chunk 排到第一也被判 miss，指标严重低估。改用 element_id 匹配后，命中条件变为"对应元素的 chunk 出现在 top-k 中"，与任务目标一致。
- Expected impact: Recall@10 和 MRR 均大幅提升，可获得真实可比的 baseline vs graph delta
- Related commit: dceaa71

### eval-v1.2 — 2026-03-12
- Version tag: eval-v1.2
- Change type: candidate generation (graph prior granularity)
- Before: graph_hub_rerank 使用 doc-level hub prior（31/76 docs 有非零 prior，平均每 doc 17.3 个元素，~537 个 chunk 被 boost，占语料 41%）
- After: 优先使用 element-level prior，来自 `hub_candidates_enriched_v2.json` 中的 `element_a_id`/`element_b_id`（161 个元素被 boost，占语料 12%）；fallback 到 doc-level prior
- Reason: 诊断显示 hub 节点均为 paragraph 类型（非 figure/table/formula），doc-level boost 将同一文档内所有 17.3 个 elements 全部拉高，造成大量无关 chunk 被误排上位。graph < BM25 的根本原因是 prior 粒度太粗，信噪比差。
- Expected impact: graph_hub_rerank Recall@10 向 BM25 靠拢或超越
- Related commit: (本次)

### eval-v1.3 — 2026-04-19
- Version tag: eval-v1.3
- Change type: candidate generation (multi-source graph rerank: per-source weighting, weighted prior, typed cross-doc layer, cross_doc bug fix)
- Before:
  - `eval_graph_topk_rerank.py` 仅支持 `--graph-sources {explicit, virtual, cross_doc, summary}`，多源叠加固定用 `merge_adjacencies(combine='max')`，所有源等权。
  - graph prior 仅有 `static_prior` (degree-based, `log1p(deg)`)，对边权完全免疫。
  - `load_cross_doc_adjacency()` 试图从 `node_id` 字符串（如 `1104.3913_secsummary_1104.3913_secsum_1`）解析 (doc, section_title)，解析失败 → cross_doc 实际产生 **0 个 pid pair**，等同 cross_doc source 静默失效。
- After:
  - 新增 `typed_crossdoc` graph source，加载 `data/01_graphs/typed_crossdoc_edges.json`（figure/formula/table 三类元素的 0.6B embedding 跨文档相似度边，threshold 0.70 + top-K 10，bbl citation 命中 +0.05 boost；产物 16520 边 / 744 doc pair / 68.4% pid 覆盖）。
  - 新 CLI：`--explicit-weight / --virtual-weight / --crossdoc-weight / --typed-crossdoc-weight`（每源边权缩放）、`--merge-combine {max, sum}`（叠加时同边合并方式）、`--prior-mode {degree, weighted}`（degree = 老的 `log1p(deg)`，weighted = `log1p(sum-of-weights)`，配合 per-source weight 才能起作用）、`--typed-crossdoc-edges` 路径、`--typed-crossdoc-types` 子集筛选、`--typed-crossdoc-use-boost` 用 boosted_similarity 作权重。
  - 修复 `load_cross_doc_adjacency()`：优先使用 edge metadata 里的 `source_doc` / `source_section` 字段做 (doc, section_title) hint，解析失败才回退到 node_id 解析。修复后 cross_doc pid_pair 0 → **5135**，doc pair 0 → **167**。
- Reason:
  - 4 月 stacking 实验 A 发现 `explicit + cross_doc` 与 `explicit_only` 完全等指标 → 定位到 cross_doc 静默失效 bug。
  - 现有 `merge_adjacencies(combine='max')` + degree prior 让多源叠加无法表达"信号源强弱"，virtual edges 噪声直接污染排序。需要 per-source 权重 + 与之配套的 weighted prior 才能控制贡献度。
  - section-level cross_doc 信号粒度仍粗（句段相似），需要更细的 element-level 跨文档相似（figure/formula/table），并用 bbl 引用做先验加成。
- Expected impact:
  - cross_doc bug 修复：`explicit + crossdoc_sec` static_plus_neighbor R@10 从无效 → **0.6406**（vs explicit_only 0.6258，+1.5pp）。
  - typed_crossdoc：`explicit + typed (w=0.2)` 在 static_plus_neighbor 上 R@10 = **0.6406**（本项目 0.6B 历史最高）；static_prior 上 R@1=0.2304 / MRR=0.6060（接近 explicit_only 0.2357 / 0.6166）。
  - 多源叠加从此可在权重轴上系统消融，而不是固定等权 max-merge。
- Related commit: (本次)
