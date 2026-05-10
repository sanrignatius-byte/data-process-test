---
type: experiment
node_id: exp:20260503_mentor_recording60_full_todo
title: "Mentor 录音60 完整 todo 提取 + 完成度核查 (handoff record)"
date: 2026-05-03
last_updated: 2026-05-10
status: in_progress
verdict: 5_10_BCD_phased_executed_C6_passed_S2_B1_phase1_done_B4_blocked
---

# 5/10 BCD 分阶段执行后状态（在原文之上叠加）

| # | TODO | 5/3 状态 | **5/10 状态** | 5/10 产物 |
|---|---|---|---|---|
| **B1** | chunk-element 边用 LaTeX 行号 | ❌ | ✅ **Phase 1 + Phase 2 done**（topology 修复了但**对 retrieval 指标无主线增益**——formula ceiling 是 encoder bound 不是 topology bound）| [B1_LATEX_LINENO_REPORT](../../refine-logs/B1_LATEX_LINENO_REPORT_20260510.md), [B1_PHASE2_LINENO_EXPERIMENT](../../refine-logs/B1_PHASE2_LINENO_EXPERIMENT_20260510.md), [exp:20260510_b1_phase2_lineno](20260510_b1_phase2_lineno.md), [claim:C11](../claims/C11_formula_ceiling_is_dense_encoder_bound.md) |
| **B2** | mineru→latex 模糊匹配丢失率 audit | ❌ | ✅ **完成** | figure 49.7% / table 67.3% / formula **0.0%**；[B2_MINERU_LATEX_MATCH_AUDIT](../../refine-logs/B2_MINERU_LATEX_MATCH_AUDIT_20260510.md) |
| **B3** | 多模态元素文档（equation 独立 / inline 不计） | ❌ | ✅ **完成** | [ref:multimodal_element_taxonomy](../reference/multimodal_element_taxonomy.md) |
| **B4** | 全 element enrich (40.4%) | ⚠️ | 🔴 **API 死 19 天** | [B4_C5_API_STATUS](../../refine-logs/B4_C5_API_STATUS_20260510.md) |
| **C2** | chunk 是否为噪声重审 | ⚠️ | ✅ **claim:C9 入库** | [C9_chunk_dilutes_double_evidence_signal](../claims/C9_chunk_dilutes_double_evidence_signal.md) |
| **C5** | 多粒度 enrich (DocResearcher) | ❌ | 🔴 **deferred (API + C8 双阻)** | [B4_C5_API_STATUS](../../refine-logs/B4_C5_API_STATUS_20260510.md) |
| **C6** | 50 query 平衡测试集 | ❌ | ✅ **S2 verdict** | [SMOKE50_DECISION_20260510](../../refine-logs/SMOKE50_DECISION_20260510.md), [exp:20260505_smoke50_balanced_audit](20260505_smoke50_balanced_audit.md), [claim:C10](../claims/C10_graph_rerank_modality_selective.md) |
| **D1** | mentor todo list with deadlines | ❌ | 🟡 **草稿就绪** | [MENTOR_TODO_DRAFT](../../refine-logs/MENTOR_TODO_DRAFT_20260510.md) — user 审后发 |

## 5/10 关键发现

1. **C6 verdict S2 命中**: ceiling 0.6913 真实，但 graph 增益是 modality-selective: figure +10.3pp / table +8.3pp / formula **+0pp**
2. **M4query_v1 没有 text qrels**: BGE pilot top-1=text 是 reranker 错答（348/473）；mentor "10 text" 在 v1 不可执行；新 [claim:C10](../claims/C10_graph_rerank_modality_selective.md) 已加注 modality scope
3. **B2 揭示 figure 49.7% / table 67.3% / formula 0% 元素级匹配率**：与 user 印象 "50%" 完全吻合，与 user-2 引用的 "92%" 是不同粒度（92% 是 doc 级覆盖率 56/57=98.2%）
4. **B1 Phase 1 救回 formula 41.2% chunk-element 匹配**：用 LaTeX content matching 绕过 label key
5. **B1 Phase 2 实测：拓扑修了但 ceiling 没破** — line_no fix 在 explicit-only graph 完全无效（hub bridge 不经过 chunk-element），在 explicit+virtual 给 table +2.78pp 但 formula 0pp。**6 个 config 在 formula 上全 ≤ 0.5600** → 新 [claim:C11](../claims/C11_formula_ceiling_is_dense_encoder_bound.md): formula 瓶颈是 **dense encoder on LaTeX**，不是 graph topology
6. **F-formula caption injection 实验（job 68107, 5/10 13:30 UTC）**: 同 encoder + NL context injection on formula passages → **HD verdict**: 反伤 dense R@10 −3.7pp, graph −2.2pp, formula bucket −16pp dense / −4pp graph. 8 configs 0 突破 0.5600（3 regressed）. C11 强化, F-formula Phase 2 必须真换 encoder. 与 [claim:C8](../claims/C8_modora_visual_enrichment_net_negative.md) 同向 — cross-modal style injection 全方向 net 负
7. **B4 / C5 因 API 19 天死被锁**：4/21 后 401 unauthorized；recommend mentor / cluster admin 给新 token

## 整体完成度（5/10 vs 5/3）

| 类别 | 5/3 完成率 | 5/10 早段 | **5/10 末段（B1 P2 后）** | Δ vs 5/3 |
|------|---------:|---:|---------------:|---:|
| A 文档/术语（3 项，handoff） | 17% | 17% | 17% | 0pp |
| B 数据/匹配（4 项） | 13% | 63% | **75%**（B1/B2/B3 done, B4 blocked）| +62pp |
| C 实验（7 项） | 43% | 86% | 86% | +43pp |
| D 工作方式（3 项） | 33% | 67% | 67% | +33pp |
| **整体（核心 17 项）** | **~32%** | ~71% | **~76%** | **+44pp** |

主要原因：5/10 单 session 拿掉 7 项（B1 phase 1+2, B2, B3, C2, C6, D1 草稿），剩下 4 项要么 API blocked 要么 handoff。

---



# 来源

`/projects/myyyx1/标准录音 60.mp3_20260502_134902_精转文稿.docx`

之前 wiki log 里 2026-05-02 那条只摘了 4 点高层要求，录音里实际有 18 条具体动作项。

# 完整 todo 清单（按录音先后顺序，session 2026-05-03 提取）

## A. 文档/术语层

| # | TODO（mentor 原话） | 当前状态 |
|---|--------------------|---------|
| **A1** | paragraph 是 text element，与 figure/table/equation 平级；写文章只用 element / chunk 二元 | 文档半完成（wiki 用了 element），代码层未改 |
| **A2** | 基于 mentor 上次发的 `文档建图.md` 修改，不要重新造文档/术语 | ❌ `文档建图.md` 最后修改 2026-04-21，5/2 后未动 |
| **A3** | `fallback`、`static_plus_neighbor`、"算子"、"底部先验"等 AI 命名换成大白话（出入度 / 阅读顺序 / ref 关系 / 汇总） | ❌ 未改 |

→ **A1/A2/A3 user 5/3 明确说交给别人处理，本 session 只记录不动手**

## B. 数据/匹配层

| # | TODO | 当前状态 |
|---|------|---------|
| **B1** | chunk-element 边构建用 LaTeX 行号，不许字符串模糊匹配 | ❌ `build_paragraph_chunks.py` 没有 line_no/line_number/latex_line 字段 |
| **B2** | 手动核查 mineru→latex 模糊匹配丢失率（之前印象 50%，新数字 92%） | ❌ 未做 |
| **B3** | 多模态元素文档更新：equation 必须独立、inline 不计 | ❌ 未做 |
| **B4** | 全 element enrich（覆盖 27209 个，目前 10988/27209 = 40.4%） | ⚠️ 进行中，gap227 99.6% 完成；hub-pair 之外的剩余被 API 401 阻塞 |

## C. 实验层

| # | TODO | 当前状态 |
|---|------|---------|
| **C1** | 算"recalled chunk 平均含几个 element"——per-query，不是 corpus 全局 | ✅ **2026-05-03 完成**，详见 [exp:20260503_chunk_query_element_recall](20260503_chunk_query_element_recall.md) |
| **C2** | 如果 chunk 拉低 R@1，重新审视 chunk 是不是噪声 | ⚠️ 已坐实"chunk 稀释信号"，结论待师兄会议确认 |
| **C3** | 分离式检索：chunk 装文字，图/表/公式独立被找到 | ✅ 文本版 Job 66048 跑通；VL rerun Job 66248 跑通。结论：figure lane 被救活，但 formula/table 拖累整体，需 hybrid rank fusion |
| **C4** | 用 Qwen3-VL-Embedding-2B（mentor 主动建议） | ✅ Job 66248 用 transformers 5 overlay 成功加载并评估；`R@10 0.2579`，低于 `split_4B_text 0.4767` |
| **C5** | 多粒度 enrich（粗 summary + 细描述，仿 DocResearcher） | ❌ 未做 |
| **C6** | 小冒烟测试集（50 query × 4 类型：10 文本/10 图/10 表/10 公式） | ❌ 未做 |
| **C7** | summary 节点暂缓（仅作 scaffold） | ✅ 已暂缓 |

## D. 工作方式

| # | TODO | 状态 |
|---|------|------|
| **D1** | 整理 to-do list 发师兄，自己写 deadline | ❌ 未做 |
| **D2** | 5/4 那周 user 请假，后台跑 query 分析 | ✅ 在跑 |
| **D3** | 转 full-time 后开始真正交付（前 3 个月是培养沟通） | 🟡 节点未到 |

# 量化完成度

| 类别 | 已完成 | 半完成 | 未做 | 完成率 |
|------|-------:|-------:|-----:|-------:|
| A 文档/术语 (3 项) | 0 | 1 | 2 | 17% |
| B 数据/匹配 (4 项) | 0 | 1 | 3 | 13% |
| C 实验 (7 项) | 3 | 1 | 3 | 43% |
| D 工作方式 (3 项) | 1 | 0 | 1 | n/a |
| **整体（核心 16 项）** | **4** | **3** | **9** | **~32%** |

# 本 session 实际推进的两件事

1. **C1 完成**：写了 `scripts/analyze_chunk_query_element_recall.py`（per-query, eval-qrels grounded），跑了 4 lane 比较。结论：n500 partial-overlay 最佳 lane 也只有 elem R@10 = 0.53，K=1 zero rate 71%。**坐实 mentor 怀疑：chunk 在双证据 query 上稀释信号**。
2. **C3+C4 完成**：用 transformers 5 overlay 重跑 Qwen3-VL-Embedding-2B（Job 66248）。环境问题修复：625/625 weights clean load，无 `newly initialized` warning。结果：`split_VL_2B_t5 R@10=0.2579`，证明模型可用但当前 pure VL split 仍弱于 `split_4B_text R@10=0.4767`。详见 [exp:20260503_split_modality_vl_t5_rerun](20260503_split_modality_vl_t5_rerun.md)。

# 副产品发现（顺手挖到的）

| 发现 | 严重度 |
|------|--------|
| `paragraph_chunks_n400_v2.json` 的 `chunk_contains_element` 边和 eval-time `chunk_corpus_*/qrels.jsonl` 的 chunk_id **0% 一致**，两套独立 build 流程命名空间冲突 | **P1** |
| `compute_chunk_element_stats.py` 接受 `--qrels` 但 main() 完全忽略 | P3 |

# Handoff: 1/2/3 交给谁做什么

User 2026-05-03 决定 A1/A2/A3 交给别人处理。下一个执行人需要：

1. **重写 `/projects/myyyx1/data-process-test/文档建图.md`**：基于 mentor 5/2 给的同名 doc 全部手敲，不能 AI 生成；不要 fallback / static_plus_neighbor / 底部先验 / 算子；图加权用大白话（出入度 / 阅读顺序 / ref 关系 / 汇总）。
2. **代码命名清理**（不强求一次到位）：
   - `src/parsers/latex_reference_extractor.py` 4 处 `fallback`
   - `src/prompts/styles.py`、`src/prompts/personas.py` ×3 处 `fallback`
   - `src/models/__init__.py:25` `node_type = "paragraph"` → 文章里只用 element/chunk
3. **整理 to-do list 发 mentor**：自己写 deadline，自己写时间，不强制时长（mentor 原话："你写一个月后也没有关系"）。
