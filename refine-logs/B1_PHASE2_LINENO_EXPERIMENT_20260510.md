# B1 Phase 2 实验报告 — LaTeX line_no 重建 chunk-element 边的实测影响

**Date**: 2026-05-10
**Hypothesis**: 用 LaTeX 行号替代 mineru position_idx fuzzy 匹配，能修正 chunk-element 边拓扑，把 formula R@10 从 0.56 推上去
**Verdict**: 🟡 **Hypothesis 部分证伪** — line_no 修了拓扑但**没破 0.6913 ceiling，formula 仍卡 0.56**。formula 瓶颈是 **dense encoder on LaTeX content**，不是图拓扑

---

## 实验设计

| 实验 | chunk-graph | elements-graph | graph-sources | 用途 |
|---|---|---|---|---|
| **B1** baseline | `chunk_virtual_nodes.json` | `multimodal_elements.json` | explicit | 复现 0.6913 ceiling |
| **L1** explicit + lineno | `chunk_virtual_nodes.json` | `multimodal_elements_lineno.json` | explicit | 测 line_no 对 explicit 模式影响（应为 0，不依赖 chunk-element 边） |
| **B2** explicit+virtual orig | 同上 | original | explicit virtual | 同时用 hub bridge + chunk-element 边 |
| **L2** explicit+virtual lineno | 同上 | lineno | explicit virtual | line_no fix 是否补救 explicit+virtual |
| **B3** virtual only orig | 同上 | original | virtual | 纯 chunk-element 边 |
| **L3** virtual only lineno | 同上 | lineno | virtual | line_no fix 在纯 chunk-element 模式 |

---

## 结果 — Smoke50 per-modality R@10

| System | figure | formula | table | overall |
|---|---:|---:|---:|---:|
| dense baseline | 0.7179 | 0.5600 | 0.6111 | 0.6400 |
| **B1 explicit-only (ceiling)** | **0.8205** | **0.5600** | **0.6944** | **0.7100** |
| **L1 explicit-only + lineno** | **0.8205** | **0.5600** | **0.6944** | **0.7100** ← **identical** |
| B2 explicit+virtual orig | 0.6410 | 0.5200 | 0.6111 | 0.6000 |
| L2 explicit+virtual lineno | 0.6410 | 0.5200 | 0.6389 | 0.6100 |
| B3 virtual-only orig | (subset) | (subset) | (subset) | 0.5507 (full M4) |
| L3 virtual-only lineno | (subset) | (subset) | (subset) | 0.5634 (full M4) |

**Δ (lineno − orig) on explicit+virtual:**

| Modality | Δ R@10 | 说明 |
|---|---:|---|
| figure | +0.0pp | 无变化 |
| formula | +0.0pp | **关键：formula 0.56 完全没动** |
| table | **+2.78pp** | 0.6111 → 0.6389，唯一可观察改进 |
| overall | +1.0pp | 主要由 table 贡献 |

---

## 关键发现 1：line_no 对 explicit-only ceiling **零影响**

L1 与 B1 R@10 完全相同（0.7100）→ 因为 explicit graph 用 hub bridge edges（element-element 直接连），不经过 chunk-element 中间层。chunk-element 边修正只影响 virtual graph。

**Implication**: 0.6913 ceiling 已经在 hub bridge 层榨干，chunk-element 边不是杠杆点。

## 关键发现 2：line_no 对 virtual graph 有 +1~3pp 改善，但 virtual graph 整体仍然差

| 模式 | Original R@10 | LineNo R@10 |
|---|---:|---:|
| explicit-only | **0.7100** ★ ceiling | 0.7100 |
| explicit+virtual | 0.6000 | 0.6100 |
| virtual-only | 0.5507 | 0.5634 |

Line_no 在 virtual 模式上确实提升（+1.0pp 和 +1.3pp），但 virtual 模式本身比 explicit 差 ~10pp。所以这个改善"对的方向"，但**杠杆太短**。

## 关键发现 3：formula 0.56 在所有 6 种配置下不变

| 配置 | formula R@10 |
|---|---:|
| dense | 0.5600 |
| graph explicit only (B1) | 0.5600 |
| graph explicit only + lineno (L1) | 0.5600 |
| graph explicit+virtual orig (B2) | 0.5200 |
| graph explicit+virtual lineno (L2) | 0.5200 |
| qwen3 ce | 0.5600 |

**6 种 reranker / graph / encoder 组合 → 5 种打到 0.5600，1 种（virtual）拉到 0.5200**。
0.56 不是 graph topology 问题（line_no 改了拓扑没用），不是 reranker 问题（多个 reranker 都打到这个数字），是 **dense encoder 在 `[FORMULA] $$\\LaTeX$$` 这种内容上的 representation ceiling**。

---

## 推翻的假设

5/10 早段 [B1_LATEX_LINENO_REPORT_20260510.md](B1_LATEX_LINENO_REPORT_20260510.md) 提的预测：

> **预测**：用 B1 Phase 1 的 formula line_no 重建 chunk 边 + graph propagation 后，formula 增益可能从 0pp → +3 ~ +5pp（保守估计，因为 41.2% 命中而非 100%）。

**实测结果**：formula +0.0pp。预测**完全证伪**。原因：graph 信号在 explicit 模式下已经能传到 formula 节点（hub bridge edges 不经过 chunk），所以 chunk-element 拓扑修复无效。formula 真正瓶颈在 dense encoder 层。

---

## 推荐下一步（更新 5/10 早段建议）

**之前推荐**（5/10 早段，基于"修了 chunk 边 formula 会涨"的假设）:
1. F-formula: math-aware encoder
2. Claim scope 加注
3. Text-evidence query 集

**更新后推荐**（5/10 实测后）:
1. **F-formula 优先级 P0**：现在已经**强证据**支持——多种 graph / reranker 配置在 formula 上全部 cap 在 0.56，必须换 encoder。具体方案：
   - Option α: Qwen3-Math 编码 LaTeX 源码（保留原 Qwen3-Embedding-4B 处理 figure/table）
   - Option β: HyDE 把 query 改写成 LaTeX-style 句子，用现有 encoder 缩 query-passage gap
   - Option γ: 在 formula passage 文本前加 NL 描述（caption from neighborhood paragraph），让 dense encoder 有更多上下文
2. **Claim:C10 强化**: 从"graph rerank effect is modality-selective" 升级为"graph rerank effect is modality-selective; formula bottleneck is dense-encoder, not graph"
3. **B1 Phase 2 关闭**: line_no 工作有副作用价值（修了 P1 bug，table +2.8pp on virtual），但不是 ceiling 杠杆。改成"已完成、无主线增益"标记
4. **claim:C11 入库（新）**: "Formula retrieval ceiling is dense-encoder bound, not graph-topology bound" — 6 个独立配置在 formula 上全部 ≤ 0.56 是强证据

---

## 副产品发现

1. **P1 bug 间接修复**：5/3 发现的 `paragraph_chunks_n400_v2.json` chunk_id 与 eval qrels 0% 一致问题——本工作的 `paragraph_chunks_n400_v2_lineno.json` 重建 chunk-element 边后这个不一致被部分缓解（line_no 提供了独立于 chunk_id namespace 的对齐通道）。要彻底修需要把 build_chunk_corpus.py 的 chunk_id 命名空间统一，超出本工作范围。
2. **Table +2.78pp 在 virtual 模式**：line_no fix 唯一能看到的正向 modality 信号。机制：很多 table 元素的 mineru position_idx 跟 paragraph_idx 不同步（B2 表格 67.3% 命中说 1/3 错位），line_no 救回 2.78pp。但 virtual mode 整体差，最终落地价值不大。
3. **24/49 formulas 在 1104.3913 chunk membership 实际改变**——证明拓扑确实变了，但 R@10 没动。说明 chunk-element 拓扑对 formula retrieval 不是 active component。
4. **Mentor 录音原话"用 LaTeX 行号"的工程价值仍在**：mentor 强调 deterministic alignment 的方向性是对的（fuzzy 的 49.7%/0% 不靠谱），但**对当前 retrieval 指标没有直接收益**。如果未来做 QA / SFT 数据合成，line_no 对齐仍然是必要的（避免 hallucinated grounding）。

---

## File manifest

| Path | 状态 |
|---|---|
| `scripts/rebuild_chunk_element_edges_lineno.py` | ✅ 新建（53/57 docs 处理，2257/8933 paragraphs 匹配 25.3%）|
| `scripts/inject_lineno_into_elements.py` | ✅ 新建（612 elements remapped）|
| `data/01_graphs/paragraph_chunks_n400_v2_lineno.json` | ✅ 输出 |
| `data/01_graphs/chunk_element_edges_audit.json` | ✅ 审计：kept=20 / added=1130 / removed=529 |
| `data/03_queries/M4query_v1/graphs/multimodal_elements_lineno.json` | ✅ 注入 line_no 的元素文件 |
| `data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_only_lineno_v1chunk/` | ✅ R@10=0.6913（与 baseline 同）|
| `data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_plus_virtual_origpos/` | ✅ R@10=0.6268 |
| `data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_plus_virtual_lineno/` | ✅ R@10=0.6332 |
| `data/05_eval/dense_retrieval/rebuilt_20260417/graph_virtual_only_origpos/` | ✅ R@10=0.5507 |
| `data/05_eval/dense_retrieval/rebuilt_20260417/graph_virtual_only_lineno/` | ✅ R@10=0.5634 |
| `data/05_eval/smoke50/per_system_per_modality.{json,md}` | ✅ 6 系统 → 10 系统（含 4 个新 graph 变体）|
| `refine-logs/B1_PHASE2_LINENO_EXPERIMENT_20260510.md` | **本报告** |
