# Mentor 录音 60 Todo List — 5/10 自定 deadline 草稿

> **草稿状态**：未发；user 审后才发师兄。
> **基线**：mentor 录音 60 (5/2) 18 条 todo + 5/10 BCD 阶段执行结果
> **格式**：mentor 5/2 原话「你写一个月后也没有关系」，所以 deadline 偏保守

---

## 已完成（5/3 → 5/10）

| # | TODO | 完成日 | 产物 |
|---|---|---|---|
| C1 | per-query chunk→element recall 分析 | 5/3 | `exp:20260503_chunk_query_element_recall` |
| C3 | 分离式检索（文本 + 多模态） | 5/2-5/3 | `exp:20260502_split_modality` + `exp:20260503_split_modality_vl_t5_rerun` |
| C4 | Qwen3-VL-Embedding-2B | 5/3 | `R@10 0.2579`，环境修复完成 |
| C7 | summary 节点暂缓 | confirmed | C6 claim line closed |
| **C6** | **50 query 平衡测试集** | **5/10** | `exp:20260505_smoke50_balanced_audit`，**S2 命中** |
| **B2** | **mineru→latex 匹配率审计** | **5/10** | figure 49.7% / table 67.3% / formula 0% |
| **B3** | **多模态元素文档 (equation 独立)** | **5/10** | `ref:multimodal_element_taxonomy` |

---

## 进行中 / 本周内（5/12-5/16）

| # | TODO | 计划 deadline | 备注 |
|---|---|---|---|
| **B1** | chunk-element 边用 LaTeX 行号 | **5/15** | B2 揭示 figure 50% / formula 0% 匹配率，B1 是关键杠杆，预期把 formula graph 增益从 0pp 拉到 +5pp |
| **C2** | chunk 是否为噪声重审 | **5/12** | 已坐实 chunk 在双证据上稀释信号，待写成 claim:C9 + paper scope 加注 |
| **D1** | 整理 todo 发师兄 | **5/10** | **本文档** |

---

## 下周 / 月内（5/19-6/10）

| # | TODO | 计划 deadline | 备注 |
|---|---|---|---|
| F-formula | math-aware encoder for formula passages | **6/3** | smoke50 S2 verdict 推荐项；Qwen3-Math 编码 LaTeX 源码；预算 1h GPU + $0 LLM |
| Claim scope 加注 | C1/C5/C7 加 modality scope | **5/16** | 配合 C10，paper claim 改诚实陈述 |
| C5 | 多粒度 enrich (DocResearcher 风格) | **6/10** | 与 [claim:C8](../research-wiki/claims/C8_modora_visual_enrichment_net_negative.md) 张力大，需 smoke50 决策后再设计 |

---

## Blocked / Handoff

| # | TODO | 状态 | 阻塞点 |
|---|---|---|---|
| **B4** | 全 27209 elements LLM enrich | 🔴 BLOCKED | API endpoint `az.gptplus5.com` 自 4/21 起 401 / 无新调用记录 |
| A1 | paragraph = text element 进代码 | 🔵 Handoff | 5/3 user 决定交给别人 |
| A2 | 重写 `文档建图.md` | 🔵 Handoff | 5/3 user 决定交给别人 |
| A3 | AI 命名换大白话 | 🔵 Handoff | 5/3 user 决定交给别人 |

---

## 节点未到

| # | TODO | 触发条件 |
|---|---|---|
| D3 | 转 full-time 后开始真正交付 | 节点未到 |

---

## 给 mentor 的 5/10 战略汇报（建议同时发）

> 师兄您好，5/2 录音 60 的 18 条 todo 进度更新：
>
> 1. **C6 智能冒烟测试集**已落地，**verdict S2**：
>    - graph rerank ceiling 0.6913 是真的（不是 figure-heavy artifact），smoke50 上 0.7100 ±2pp
>    - 但 **formula 是真正瓶颈**：dense / graph / Qwen3-Reranker 三家在 formula 上都卡在 R@10=0.56，graph rerank 对 formula 零增益
>    - figure +10.3pp / table +8.3pp / formula 0pp — graph 价值是模态选择性的
>
> 2. **关键现实修正**：M4query_v1 qrels 完全没有 text 元素，您 5/2 提的"10 文本"在当前数据上不可执行；BGE pilot 之前 top-1=text 是 reranker 错答，不是 ground truth。下次构建 v2 query 集应补 text-evidence query
>
> 3. **B2 mineru→latex 匹配率审计**揭示根因：figure 49.7% / table 67.3% / formula **0%**。Formula 0% 完全解释了 graph 在 formula 上没增益——latex equation label 与 mineru formula element 完全没打通，graph 信号传不到 formula 节点。**B1 (LaTeX 行号对齐)** 因此优先级显著提升，预期能把 formula 增益拉到 +5pp
>
> 4. **下一步推荐 (按 EV 排序)**：(a) B1 LaTeX 行号 5/15 截止；(b) F-formula math-aware encoder 6/3；(c) paper claim 全部加 modality scope。所有 reranker 路线（BGE / Qwen3 / corpus enrich）已证伪，不再追
>
> 5. **请假/blocked**：API endpoint 自 4/21 起死，B4 全 enrich 推不下去；如能恢复请告知

---

## 不发给师兄的内部备注（user 看）

1. mentor "10 text" 提的时候大概率没意识到 v1 没 text qrel——等于 mentor 提的 spec 与数据不一致。汇报时温和指出，别让 mentor 觉得"你跑偏了"
2. C5 多粒度 enrich 与 C8 (MODORA visual enrich net negative) 直接冲突，5/3 -> 5/10 三次实验都在敲打"加更多 enrich 容易反向"。我标 6/10 是给我自己留拒掉这条的余地——smoke50 verdict S2 已经把推荐改到 F-formula 了
3. D2 5/4 那周请假是 mentor 默认的，不需要正式汇报
4. F-formula 6/3 deadline 是给 user 自己定的；mentor 没要求时间表

---

## File manifest

| Path | 用途 |
|---|---|
| `refine-logs/MENTOR_TODO_DRAFT_20260510.md` | **本草稿** |
| `refine-logs/SMOKE50_DECISION_20260510.md` | C6 决策报告（汇报附件） |
| `refine-logs/B2_MINERU_LATEX_MATCH_AUDIT_20260510.md` | B2 数据（汇报附件） |
| `research-wiki/reference/multimodal_element_taxonomy.md` | B3 文档（汇报附件） |
| `research-wiki/claims/C10_graph_rerank_modality_selective.md` | claim 加注的依据 |

User 审完发即可，无需我再编辑。
