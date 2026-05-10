# Plan — Mentor 录音 60 BCD 全部分阶段执行

**Date**: 2026-05-10
**Source**: `/projects/myyyx1/标准录音 60.mp3_20260502_134902_精转文稿.docx`
**Todo extraction**: [exp:20260503_mentor_recording60_full_todo](../research-wiki/experiments/20260503_mentor_recording60_full_todo.md)
**Triggered by**: User 5/10 directive — "BCD全部分阶段执行，写进 research wiki"
**Owner**: this session

---

## 总览：14 项 todo（不含 A）

| 类别 | 项数 | 已完成 | 半完成 | 未做 |
|---|---:|---:|---:|---:|
| B 数据/匹配 | 4 | 0 | 1 (B4) | 3 |
| C 实验 | 7 | 3 (C1/C3/C4) + 1 deferred (C7) | 1 (C2) | 2 (C5/C6) |
| D 工作方式 | 3 | 1 (D2) | 1 (D3 节点未到) | 1 (D1) |

**Note**: A1/A2/A3（文档/术语层）已 5/3 handoff 给别人，本 plan 不动。

---

## 关键约束（5/10 重新核查）

1. **API logs 4/21 后停了**。最后成功调用 `api_logs/calls/gpt-5.4/2026-04/2026-04-21.jsonl`，2026-04 之后无新文件。任何 LLM 调用任务（B4 全 element enrich, C5 多粒度 enrich）**当前不可执行**。
2. **C6 smoke50 plan 已写**（[SMOKE50_BALANCED_PLAN_20260505.md](SMOKE50_BALANCED_PLAN_20260505.md)），$0 LLM、~10 min GPU，全套 ranking 文件已就绪可直接 slice。本轮直接执行。
3. **0.6913 R@10 ceiling 仍未破**。F1 (BGE-reranker)、F2 (corpus enrich fix)、Option A (Qwen3-Reranker-4B) 三轮证伪后，C6 smoke50 是最后一根能区分 "ceiling 真假" 的稻草。
4. **3 个独立证据收敛到 [claim:C8](../research-wiki/claims/C8_modora_visual_enrichment_net_negative.md)** — MODORA 视觉描述对 text-style retrieval 净负。这直接威胁 C5（多粒度 enrich）的可行性，C5 应等 smoke50 数据再决策。

---

## Phase 1 — 今天可立刻做的（$0，无 API，~3h wall）

并行 4 件，相互独立。

### Phase 1a: C6 Smoke50 执行（~30 min wall, 10 min GPU）

按已有 [SMOKE50_BALANCED_PLAN_20260505.md](SMOKE50_BALANCED_PLAN_20260505.md) 5 阶段执行：
- A: 写 `scripts/build_smoke50.py` + 输出 `data/03_queries/M4query_smoke50/{queries,qrels,manifest}.{jsonl,md}`
- B: 写 `scripts/eval_smoke50_slice.py` + 6 系统 × 4 modality slice metrics
- C: 条件触发的 VL fusion（仅 figure lane VL > 4B text 时）
- D: 决策报告 `refine-logs/SMOKE50_DECISION_20260510.md`，命中 S1–S5
- E: wiki 收尾（mentor C6 ❌→✅）

**Gate**: text bucket 必须满 10；smoke50 graph R@10 与 M4query_v1 全集 0.6913 偏差 ≤ 5pp 才算 sample 有代表性。

### Phase 1b: B2 mineru→latex 模糊匹配丢失率 audit（~30 min, read-only）

**Mentor 原话**：录音里 user 印象 50% 丢失，user-2 反驳说 92% 命中。**核查真实数字**。

输出 `refine-logs/B2_MINERU_LATEX_MATCH_AUDIT_20260510.md`：
1. 找到现有 manifest / log（`data/00_raw/mineru_output/` 的 conversion log，或 `01_graphs/manifest.json`）
2. 统计 latex source 文档数 vs 成功 mineru→latex 对齐的 element 数
3. 给两个数字：(a) 文档级丢失率（manifest 未对齐）；(b) element 级丢失率（按 modality 分）
4. 如果 manifest 不存在，从 `multimodal_elements.json` 反推（element 缺 latex source 字段的比例）

不改代码，纯报告。

### Phase 1c: B3 多模态元素文档更新（~20 min, write-only）

**Mentor 原话**：equation 必须独立、inline 不计。

新建 `research-wiki/reference/multimodal_element_taxonomy.md`：
1. 5 类 element：text / figure / table / equation / formula-inline（最后一类显式排除）
2. equation 必须独立成节点；inline `$...$` **不**作为 element
3. paragraph 是 element 的一种（=text element），与 figure/table/equation 平级（呼应 A1，但不动代码）
4. 给现有 corpus 一个 modality 分布快照（用 multimodal_elements.json 实数）

更新 `research-wiki/index.md` 加 reference 链接。

### Phase 1d: D1 mentor to-do list 草稿（~20 min, write-only）

**Mentor 原话**：自己写 deadline，自己写时间，"你写一个月后也没有关系"。

写 `refine-logs/MENTOR_TODO_DRAFT_20260510.md`：
- 按 mentor 录音 60 18 条整理成 markdown checklist
- 每条加自定 deadline（保守，user 可调整）
- 区分 "本轮 5/10 完成" / "下周可做" / "blocked on API/handoff"
- **不发，等 user 审**

---

## Phase 2 — 代码改动（~1h wall）

### Phase 2a: B1 chunk-element 边用 LaTeX 行号

**Mentor 原话**：chunk-element 边构建用 LaTeX 行号，不许字符串模糊匹配。

**当前状态**：`build_paragraph_chunks.py` 用 `position_idx`（顺序计数器）而非 LaTeX 行号。`element.position_idx` 与 `paragraph.para_idx` 共享 namespace，但这是 mineru 解析顺序，**不是 LaTeX 源行号**。

**改动方案**：
1. 在 `multimodal_elements.json` 里查 element 是否已有 `latex_line_no` / `line_number` / `latex_line_start` 字段
2. 如果有 → `build_paragraph_chunks.py` 加 `--use-latex-line-no` flag，按行号区间 inject element 到 chunk
3. 如果没有 → 写 `scripts/extract_element_latex_lineno.py`，从 `data/00_raw/latex_source/` parse 出每个 element 的源行号
4. 重建 `paragraph_chunks_n400_lineno.json` + 重建 `chunk_contains_element` 边
5. 与 5/3 发现的 P1 bug（`paragraph_chunks_n400_v2.json` 与 eval qrels chunk_id **0% 一致**）一并修复

**Gate**: 重建后 `chunk_contains_element` 边与 eval-time qrels chunk_id 一致率 ≥ 99%（之前 0%）。

输出 `refine-logs/B1_LATEX_LINENO_REPORT_20260510.md`。

---

## Phase 3 — API blocked（B4 / C5）

### Phase 3a: API 状态确认

ping company API endpoint，看是否回归。如果死，记录到 wiki，**不强推**。

### Phase 3b: B4 全 element enrich

- 当前 10988/27209 = 40.4%
- 余 ~16221 elements 在 hub_pair 之外，5/3 wiki 已写明"暂不 enrich"是 user 决策（不全量）
- 5/3 后 [claim:C8](../research-wiki/claims/C8_modora_visual_enrichment_net_negative.md) 进一步证明 visual enrich 净负，**继续推 B4 风险更高**
- 本 plan 决策：**B4 不强推，理由 (i) API 死 (ii) C8 反证 visual enrich 价值**。Wiki 标记为"deferred pending API + C8 重新评估"。

### Phase 3c: C5 多粒度 enrich

**Mentor 原话**：粗 summary + 细描述，仿 DocResearcher。

**与 C8 的张力**：C5 思路是"加更多 enrich"，C8 已证 enrich 在 text-style retrieval 上净负。但 C5 是"双粒度"不是"换 visual 描述"，机制不同——粗 summary 可能给 graph rerank 提供 cross-doc bridging 信号，与 corpus passage 不冲突。

**本 plan 决策**：C5 等 C6 smoke50 决策报告。如果 S2 / S3 命中（modality-mixed 或 ceiling 是 figure-heavy artifact），可以试 C5 做 figure-only summary enrich；如果 S1 命中（ceiling 真且 modality-uniform），C5 优先级降到最低。

---

## Phase 4 — Writeup（~30 min）

### Phase 4a: C2 chunk dilution 结论 finalize

5/3 [exp:20260503_chunk_query_element_recall](../research-wiki/experiments/20260503_chunk_query_element_recall.md) 已坐实 chunk 在双证据 query 上稀释信号。本轮把它升级成 claim：

新建 `research-wiki/claims/C9_chunk_dilutes_double_evidence_signal.md`：
- 基于 75% query evidence 跨 chunk + n500 partial-overlay 上 chunk R@10 = 0.678 vs elem R@10 = 0.530 的 15pp gap
- Scope: M4query_v1，elem-level qrels；不在 chunk-level qrels 上做声明
- Mentor C2 "重新审视 chunk 是不是噪声" 状态从 ⚠️ 升 ✅

---

## Phase 5 — Wiki 收尾

1. `research-wiki/log.md`：追加 5/10 完成日志，含每个 phase verdict
2. `research-wiki/index.md`：加新 experiment / reference / claim 链接
3. `research-wiki/experiments/20260503_mentor_recording60_full_todo.md`：批量更新 14 项状态
4. memory `MEMORY.md` 不动（项目级动态在 wiki，不下沉到 memory）

---

## 接受标准

| Phase | Gate |
|---|---|
| 1a (C6) | smoke50 plan E 段全部完成；mentor C6 ❌→✅ |
| 1b (B2) | 输出报告含两个真实数字（doc/element 级丢失率） |
| 1c (B3) | wiki reference 节点 + index 链接 + element 分布快照 |
| 1d (D1) | 草稿写好但**不发**；user 审后才发 |
| 2 (B1) | 重建 chunk_contains_element 与 eval qrels 一致率 ≥ 99%；P1 bug 同步修复 |
| 3 | 仅状态确认，不强推 |
| 4 (C2) | claim:C9 入库；C2 todo ⚠️→✅ |
| 5 | wiki 三处更新；log.md 追加完整 timeline |

---

## 不在范围内

- A1/A2/A3 文档/术语 handoff 件
- 跑新 reranker / 改 corpus / 改 graph 拓扑
- 师兄会议沟通
