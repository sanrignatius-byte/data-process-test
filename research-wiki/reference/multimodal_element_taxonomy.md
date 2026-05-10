---
type: reference
node_id: ref:multimodal_element_taxonomy
title: "Multimodal Element Taxonomy (paragraph = text element)"
date: 2026-05-10
source: mentor 录音 60 (2026-05-02), B3 todo
---

# 1. Element 5 类（论文级术语）

按 mentor 录音 60 (5/2) 共识：**paragraph 是 element 的一种，与 figure/table/equation 平级**。论文级术语只用 element 与 chunk 二元，不再单独写 paragraph。

| Element type | 论文用语 | 内部代码字段 | 是否进 graph node |
|---|---|---|---|
| **text element** | "paragraph" / "text element" | `element_type=paragraph` 或 `=text` | ✅ |
| **figure** | "figure element" | `element_type=figure` | ✅ |
| **table** | "table element" | `element_type=table` | ✅ |
| **equation** (display) | "equation element" | `element_type=formula` (display 才算) | ✅ |
| **inline formula** `$...$` | **不作为独立 element** | 不抽取为节点 | ❌ |

Mentor 5/2 原话：

> equation 必须是独立的一回事，inline 就没了

---

# 2. 现状（M4query_v1 corpus，57 docs）

| Type | 元素数 | metadata source | 备注 |
|---|---:|---|---|
| figure | 714 | mineru 解析 | 含 sub-figure（mineru 拆分大图为 a/b/c），与 latex `\\label{fig:..}` 仅 49.7% 命中 |
| formula | 1054 | `display_math_in_text` | mineru 把所有 display math 都抽出 — 与 mentor "equation 必须独立" 一致 |
| section | 164 | mineru 解析 | 章节节点，不参与多模态 query |
| table | 232 | mineru 解析 | 与 latex `\\label{tab:..}` 67.3% 命中 |
| **inline math** | **0** | — | mineru 与 latex 都未把 inline `$x^2$` 当 element — 与 mentor 共识一致 |

→ 当前 corpus 与 mentor taxonomy **一致**，无需重抽 element。

---

# 3. mineru → latex 匹配率（element 级，real numbers）

详见 [refine-logs/B2_MINERU_LATEX_MATCH_AUDIT_20260510.md](../../refine-logs/B2_MINERU_LATEX_MATCH_AUDIT_20260510.md)。

| Type | latex labels | 匹配率 | smoke50 上 graph 增益（vs dense R@10） |
|---|---:|---:|---:|
| figure | 394 | 49.7% | +10.3pp |
| table | 202 | 67.3% | +8.3pp |
| **formula** | 331 | **0.0%** | **0.0pp** |

**Doc 级覆盖率 98.2%**（56/57 有 latex source）— 这是 user-2 在录音里讲的 92% 数字。

---

# 4. ID 命名空间

| ID 形式 | 来源 | 例 |
|---|---|---|
| `{doc_id}_figure_{n}` | mineru position_idx | `1104.3913_figure_2` |
| `{doc_id}_formula_{n}` | mineru | `1104.3913_formula_1` |
| `{doc_id}_table_{n}` | mineru | `1306.5204_table_2` |
| `fig:overview` / `tab:results` / `eq:lipschitz` | latex `\\label{}` | latex side, 不直接出现在 corpus |

**Bridge 字段**：mineru element 的 `latex_labels` 字段（由 [build_latex_reference_graph.py:128](../../scripts/build_latex_reference_graph.py) 注入）保存匹配上的 latex label keys。当前 49.7%/67.3%/0% 命中率下，大量 element 该字段为空。

---

# 5. 已知缺陷与 mentor todo 对照

| 缺陷 | mentor todo | 当前状态 |
|---|---|---|
| chunk-element 边用 caption 字符串模糊匹配 | B1 | ❌ 待 LaTeX 行号实现 |
| mineru→latex equation 0% 命中（formula 完全没 graph 信号） | B1 间接覆盖 | ❌ |
| 全 element enrich (10988/27209 = 40.4%) | B4 | ⚠️ API 死了，blocked |
| 论文混用 paragraph / element 两套名词 | A1 | ❌ handoff（5/3 user 决定）|
| `文档建图.md` 5/2 后未更新 | A2 | ❌ handoff |

---

# 6. 与 paper claim 的关系

C10 ([graph_rerank_modality_selective](../claims/C10_graph_rerank_modality_selective.md)) 已经把 modality scope 写明，但要在 paper 里讲 multimodal retrieval 必须诚实写：

> "M4query_v1 covers figure / table / formula evidence (no text-only-evidence query). Mineru-extracted elements align with latex `\\label{}` at 49.7% (figure), 67.3% (table), 0.0% (formula). The graph rerank uplift on figure/table is therefore measured against an already-undermatched element corpus and is a lower bound on attainable graph signal."

---

# 7. 不在本文档范围

- 实际改代码（A1 / B1 / B3 中的"代码命名清理"）
- 重抽 element / 重建 corpus（mentor 没要求；当前 taxonomy 已与共识一致）
- 重写 `文档建图.md`（A2，handoff 件）
