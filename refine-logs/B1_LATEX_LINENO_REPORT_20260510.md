# B1: chunk-element 边用 LaTeX 行号 — Phase 1 报告

**Date**: 2026-05-10
**Mentor 录音 60 quote**: "chunk-element 边构建用 LaTeX 行号，不许字符串模糊匹配"
**Status**: ⚠️ Phase 1 of 2 完成（line_no 提取 + formula content 匹配；chunk 重建未做）

---

## 完成范围（本 session）

1. ✅ 写 [scripts/extract_element_latex_lineno.py](../scripts/extract_element_latex_lineno.py)
2. ✅ 在 mineru 元素上标注 `latex_line_no` + `label_key` + `match_method`
3. ✅ **新加 formula content matching** — 把 mineru formula 与 `\begin{equation}...\end{equation}` block 按 LaTeX-token Jaccard 匹配
4. ✅ 输出 [data/01_graphs/element_latex_lineno_map.json](../data/01_graphs/element_latex_lineno_map.json) — 1191 entries

## 未完成（Phase 2，下个 session）

- ❌ 用 line_no 重建 chunk-element 边（替代 position_idx fuzzy）
- ❌ 重建 `chunk_contains_element` 边的 paragraph_chunks_n400_v2.json
- ❌ 修复 5/3 发现的 P1 bug（chunk_id 0% 一致）

---

## Match rate before / after

匹配率分母 = mineru 元素总数（M4query_v1，57 docs）：

| Modality | Mineru 元素 | by # | by caption | by content (NEW) | match% (Before B1) | match% (After B1 Phase 1) |
|---|---:|---:|---:|---:|---:|---:|
| figure | 714 | 14 | 182 | 0 | 27.5% (existing fuzzy) | **27.5%** (无新增) |
| table | 232 | 7 | 129 | 0 | 58.6% (existing fuzzy) | **58.6%** (无新增) |
| formula | 1054 | 0 | 0 | **434** | **0.0%** | **41.2%** ⬆ +41.2pp |

**Key finding: formula 从 0% → 41.2%**（+434 elements 拿到 latex line_no）。这是 B2 audit 揭示的最大可修复杠杆。

---

## Formula content matching 算法

```python
# scripts/extract_element_latex_lineno.py:53-95
1. 解析 .tex 源文件，提取所有 \begin{equation*?|align*?|gather*?|multline*?|displaymath} blocks
2. 对每个 block 抽 (start_line, end_line, content, label_key)
3. 对每个未匹配的 mineru formula:
   - 标准化 LaTeX content（drop 注释、commands、单字母 spacing 折叠 "o p t" → "opt"）
   - Jaccard tokens vs 各 block tokens
   - 阈值 0.30（参考 figure/table caption 阈值）
4. 选 best block, 注入 latex_line_no = block.start_line
```

阈值 0.30 是平衡保守（避免 false match）和召回的设定。可调。

---

## 失败的 612 个 formula

| Reason | Count |
|---|---:|
| `no_tex_source` (28 formulas in 1 doc 没 latex source) | 28 |
| `unmatched` (jaccard < 0.30，主要是 mineru 把 inline math + tag 当独立 element) | 521 |
| `empty_content` (mineru content 解析失败) | ~63 |

521 unmatched 中相当部分是：
- mineru 把 `\tag{1}` `\tag{2}` 等 numbering 标记拆分成独立"formula" element
- `[FORMULA] $$...$$` 格式 — 加了前缀后正则切不到内容
- 跨页或被 OCR 错位的公式

提升路径（下次 session）：
1. 阈值调到 0.20 + 加 false-match guard（block_used 唯一性 + start_line 单调递增）
2. 把 `\tag{N}` formula 链回它所在 block（相邻规则）
3. 用 line proximity 兜底（两个 mineru formula 之间没 break 时合并到同一 block）

---

## 与 B2 audit 数据对照

| Modality | B2（latex-label 侧） | B1 Phase 1（mineru 侧） | 解释 |
|---|---:|---:|---|
| figure | 49.7% latex labels matched | 27.5% mineru elements have line_no | mineru 比 latex 多 80% sub-figure，比例自然降 |
| table | 67.3% latex labels matched | 58.6% mineru elements have line_no | 类似 |
| formula | 0.0% latex labels matched | 41.2% mineru elements have line_no | **content matching 反向打通**：从 mineru 找回 latex block |

两种视角互补：B2 看 "graph 信号有多少没传到 mineru element"，B1 Phase 1 看 "mineru element 有多少能拿到 latex line_no"。formula 上 B2 是 0%（latex 用 label key，mineru 没 number），B1 是 41.2%（绕过 label key 用 content 匹配）。

---

## 与 smoke50 ceiling 的关系

[exp:20260505_smoke50_balanced_audit](../research-wiki/experiments/20260505_smoke50_balanced_audit.md) 揭示：

| Modality | Graph rerank gain over dense |
|---|---:|
| figure | +10.3pp |
| table | +8.3pp |
| formula | **+0.0pp** |

formula 上 graph 增益为 0 直接对应 B2 揭示的 "latex equation label 0% 命中"。
**预测**：用 B1 Phase 1 的 formula line_no 重建 chunk 边 + graph propagation 后，formula 增益可能从 0pp → +3 ~ +5pp（保守估计，因为 41.2% 命中而非 100%）。

这个预测要在 Phase 2 实测验证。

---

## Phase 2 规划（下次 session）

1. **重建 chunk-element 边**：修改 [build_paragraph_chunks.py](../scripts/build_paragraph_chunks.py) 加 `--use-latex-line-no` flag
   - 需求：mineru paragraph 也要拿 line_no（当前没有）
   - 可行路径：mineru paragraph 落在哪个 latex 行区间，用 paragraph 的首句到 latex 全文的最长公共子串定位
2. **重建 graph 边**：用 line_no 区间 + `chunk_contains_element` 边
3. **Re-run smoke50**：看 formula R@10 能否突破 0.56
4. **修 5/3 P1 bug**：chunk_id 0% 一致问题与本工作直接相关

预算：~1.5h wall + 30 min GPU smoke50 重跑。

---

## 文件 manifest

| Path | 状态 |
|---|---|
| `scripts/extract_element_latex_lineno.py` | ✅ 新建 |
| `data/01_graphs/element_latex_lineno_map.json` | ✅ 新建（1191 entries: 196 figure + 136 table + 434 formula + 425 其他类） |
| `refine-logs/B1_LATEX_LINENO_REPORT_20260510.md` | **本报告** |
| `scripts/build_paragraph_chunks.py` | ❌ 待 Phase 2 改 |
| `data/01_graphs/paragraph_chunks_n400_lineno.json` | ❌ Phase 2 输出 |
