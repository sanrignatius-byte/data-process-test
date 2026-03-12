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
- Related commit: (本次)
