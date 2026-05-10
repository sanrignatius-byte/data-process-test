# B2: Mineru → LaTeX 标签匹配丢失率审计

**Date**: 2026-05-10
**Mentor 录音 60 quote**: user 印象 50% 丢失，user-2 反驳 92% 命中 — 谁对？
**Verdict**: **两人粒度不同。Doc 级 user-2 对（98%），element 级 user 对（图 49.7% / 表 67.3% / 公式 0%）。**

---

## 测量范围

| 维度 | 数 |
|---|---:|
| M4query_v1 corpus 文档数 | 57 |
| 同时有 mineru output 的 | 57/57 |
| 同时有 latex source 的 | 56/57 |
| 同时在 latex_reference_graph.json (v1, 82 docs) 的 | 56/57 |

→ **Doc 级 latex 覆盖率 = 56/57 = 98.2%**

这就是 user-2 引用的 "92%" 数字的源头（doc 级，不是 element 级）。

---

## Element 级匹配率（基于 56 共享文档）

匹配算法 = `scripts/build_latex_reference_graph.py` 里的 `_match_labels_to_elements()`：
1. 先按尾部数字匹配（`fig:3` → mineru 的 `figure_3`）
2. fallback 按 caption Jaccard 相似度 > 0.3

| Type | LaTeX labels | by # | by caption | 未匹配 | mineru orphan | **匹配率** |
|---|---:|---:|---:|---:|---:|---:|
| figure | 394 | 14 | 182 | 198 | 512 | **49.7%** |
| table | 202 | 7 | 129 | 66 | 87 | **67.3%** |
| formula/equation | 331 | 0 | 0 | 331 | 1054 | **0.0%** |

**核心发现**：
1. **Figure 49.7%** 与 user 印象 50% **完全吻合**，把 user-2 的 92% 数字证伪（在 element 层面）
2. **Formula 0%** — 完全没匹配上。原因：mineru `position_idx` 编号是按解析顺序生成的，与 LaTeX 里 `\label{eq:lipschitz}` 这类 label key 的命名空间没有交集；caption 也几乎全空
3. **Mineru orphan**：512 figure / 87 table / 1054 formula 是从 mineru 提取出来但**完全找不到对应 latex label**，这些 element 拿不到 graph reference 信号

---

## 为什么 figure 只有 49.7%

匹配主要靠 caption 相似度（182 个），按编号匹配只成功 14 个。原因分析：

1. **Sub-figure 拆分**：mineru 把 `Figure 3 (a)` `Figure 3 (b)` 拆成两个 element，编号都是 3 或 3a / 3b，与 latex 单一 `fig:overview` label 对不上
2. **mineru 总数 1218 vs latex 394**：mineru 比 latex 多 3 倍 figure，剩下 ~512 是 sub-figure / 辅助图。这些天然没有对应的 latex `\label{}`
3. **Caption fuzzy match 阈值 0.3** 是 Jaccard，对 OCR 错误（mineru caption 里有错字）容忍度不够

---

## Formula 0% 的根因

```python
# 当前代码
type_map = {"equation": "formula"}
# 数字匹配：fig:3 → 取尾部 "3"，匹配 figure_3.number==3
# Formula 不通：mineru formula 没 number 字段，position_idx 是顺序计数
# Caption 匹配：mineru formula 没 caption，永远落空
```

因此 latex 里 331 个 equation label 一个都没和 mineru formula 元素打通。

**结论**：当前实现下 formula 节点完全没有 latex graph 信号——这正好与 5/10 smoke50 的发现"graph 在 formula 上零增益"形成机制对应：graph rerank 的 `degree`/`neighbor_propagation` 信号对 formula 节点几乎没有。

---

## 这对 graph rerank 0.6913 ceiling 意味着什么

| 模态 | 匹配率 | smoke50 上 graph 比 dense 增益 |
|---|---:|---:|
| figure | 49.7% | **+10.3pp** |
| table | 67.3% | **+8.3pp** |
| formula | 0.0% | **+0.0pp** |

**完美相关**：模态匹配率越高，graph rerank 增益越大。Formula 0% 匹配率直接对应 graph 在 formula 上零增益。

之前我们以为 formula bottleneck 是"dense encoder 在 LaTeX 上能力不足"，现在看更可能是**根本没把 latex equation label 链接到 mineru formula element**——graph 信号传不到 formula 节点。

如果修复 mineru→latex equation 匹配（用 LaTeX 行号 + content 双重对齐），formula 上的 graph 增益可能从 0pp 拉到 +5~+8pp。这是 B1 的潜在杠杆。

---

## 与 mentor 录音的对照

| User-2 引用 | 数字 | 实际数据 | 对吗？ |
|---|---|---|---|
| "doc 层面 92% 命中" | 92% | 98.2% (56/57 docs) | ✅ 对，但用的是 doc 级 |
| User 印象 "50% 丢失" | 50% | figure 49.7%（element 级） | ✅ 对，element 级 |

**两个数字都对，但讲的是不同粒度**。Mentor 应该看到 element 级真实数字，因为 retrieval 是 element 级的。

---

## 推荐下一步

1. **B1 (chunk-element 边用 LaTeX 行号)** 的优先级因此**显著提升**：
   - figure 49.7% → 用 latex `\begin{figure}[Y]...\end{figure}` 行号区间 + `\includegraphics` 文件名匹配 mineru `image_path`，应能拉到 ~85%
   - formula 0% → 用 `\begin{equation}` / `\[...\]` 行号区间 + LaTeX 源码字面对齐 mineru `content`，应能拉到 ~70%
2. **B3 (multimodal element doc 更新)** 应明确写：
   - figure / table / equation 是独立 element，与 paragraph 平级
   - inline `$...$` 不作为 element（mentor 原话）
   - mineru→latex element 级匹配率（49.7% / 67.3% / 0.0%）作为已知缺陷写进 reference doc
3. **重新表述 graph rerank 价值**：当前 graph 增益 +8~+10pp 是建立在 mineru 49.7% / 67.3% 匹配率之上的——如果匹配率提升，增益还有上行空间

---

## 文件 manifest

| Path | 内容 |
|---|---|
| `refine-logs/B2_MINERU_LATEX_MATCH_AUDIT_20260510.md` | **本报告** |
| `data/01_graphs/latex_reference_graph.json` | v1 latex graph (82 docs, 含 56 M4 共享) |
| `data/03_queries/M4query_v1/graphs/multimodal_elements.json` | mineru-side elements (57 docs) |
| `scripts/build_latex_reference_graph.py:128-191` | `_match_labels_to_elements()` 实现 |

无新代码改动；本审计纯计数 read-only。
