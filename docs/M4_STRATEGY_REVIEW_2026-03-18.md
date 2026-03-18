# M4 Strategy Review — 2026-03-18

## 诚实现状评估：我们在 M4 的哪里？

### 当前项目定位（建议口径）

> **Graph-backed Cross-modal Dual-evidence Benchmark (M4-Foundation)**
>
> 我们已证明 Document Graph 在跨模态 dual-evidence evidence localization 上有效（MRR +0.0403 vs BM25），
> 但严格意义上的 M4 尚未建立。下一阶段目标是在已验证的图基础上，逐步构建真正的多跳推理链、
> 元素级跨文档桥接、和多轮对话。

### M4 四维度达成度

| 维度 | 要求 | 当前状态 | 达成度 | 差距本质 |
|------|------|---------|--------|---------|
| **Multi-modal** | 跨模态证据 | fig+tab / fig+formula / formula+tab | ✅ ~90% | 基本达标，待纳入 text-only 对照 |
| **Multi-hop** | 多步推理链 | 图上 2-3 hop 路径，但 query 只需 2 片证据拼接 | ⚠️ ~30% | **"双证据并行取证" ≠ "串行推理链"** |
| **Multi-document** | 跨文档证据 | citation walk 为负→关闭；L2 暂停 | ❌ ~10% | 缺少 element-level 跨文档桥接 |
| **Multi-turn** | 多轮对话 | 零进展 | ❌ 0% | 无数据格式、无 QC、无生成策略 |

**结论：当前实际达成 M1.5（跨模态 + 伪多跳），不是 M4。**

---

## 差距逐维分析

### 1. Multi-hop：并行取证 vs 串行推理

**当前状态**（双证据拼接）：
```
Query: "模型公平性如何随数据分布变化？"
Answer = Figure 3 的趋势 + Table 2 的数值
推理结构: A ∧ B → Answer（并行组合）
```

**目标状态**（3+ 步因果推理链）：
```
Query: "为什么模型 X 在子群 A 上性能差？根本原因的数学解释是什么？"
Step 1: Figure 3 → 模型 X 在子群 A 上性能下降（观察）
Step 2: Table 5 消融实验 → 下降原因是特征 F 缺失（归因）
Step 3: Equation (7) → 特征 F 在子群 A 的分布满足不等式（解释）
推理结构: A → B → C → Answer（串行链）
```

**核心区别**：
- 当前 `hop_distance` 是**拓扑距离**（经过几条图的边）
- 真正的 multi-hop 是**推理深度**（需要几步逻辑推导，每步用到不同证据）
- 验证标准：**step-deletion test** — 删掉任意中间步骤，答案不可得

### 2. Multi-document：doc-level 引用 vs element-level 桥接

**当前瓶颈**：
- citation walk 在 doc-level（"A 引用了 B"），不知道 A 的哪段话和 B 的哪个元素相关
- L2 v3 仍有 anchor_leakage 问题（45% fail），桥接实体语义不足

**需要做的**：
- Element-level cross-doc edges：用 embedding 相似度（已有 Qwen3-4B 匹配结果）建立元素级跨文档边
- 每条边附带 `bridge_type` + `bridge_evidence` + `confidence`
- 在此基础上才能做真正的跨文档 multi-hop

### 3. Multi-turn：需要底层支撑

**前提条件**：Multi-turn 不是独立问题，而是 multi-hop 的 session 化表达。

```
Turn 1: 对应 Hop 1 → "Figure 3 显示了什么趋势？"
Turn 2: 对应 Hop 2 → "那 Table 5 中哪些消融实验和这个趋势一致？"（依赖 Turn 1）
Turn 3: 对应 Hop 3 → "公式 (7) 能解释这种不一致吗？"（依赖 Turn 1+2）
```

**没有 multi-hop 推理链，multi-turn 就是空中楼阁。**

---

## 执行路线图

### Phase 0（当前 → 本周）：锁定 M1.5 基线 + 定义 M4 schema

| 任务 | 交付物 | 说明 |
|------|--------|------|
| M4 Strategy Review | 本文档 | 诚实重定位 |
| Reasoning Step Schema | `docs/M4_SCHEMAS.md`（Schema-ready） | 定义 3-step 推理链数据格式；当前生成脚本尚未升级为 native generator |
| Reasoning-depth heuristic | `qc_reasoning_depth()` in generation script | 启发式 auto-tagging（非严格验证器），用连接词模式区分 parallel/serial |
| Step-deletion proxy | `step_deletion_proxy` metric | 因果连接词计数 ≥ min_depth-1 的代理指标；真正 step-deletion test 待 Phase 1 |
| Cross-doc bridge schema | `cross_doc_edges` schema（Schema-ready） | element-level 桥接数据格式 |
| Multi-turn session schema | `session` schema（Schema-ready） | 对话链数据格式 |

### Phase 1（1-2 周）：严格 Multi-hop

| 任务 | 交付物 | 说明 |
|------|--------|------|
| 3-4 hop 因果路径枚举 | 图上路径搜索升级 | 从拓扑路径→因果路径 |
| 推理链 query 生成 | LLM prompt + 生成脚本 | 每条 query 附带 `reasoning_steps[]` |
| **真正的** step-deletion test | 50-100 条 gold 3-step queries | 删 step 重判 answer derivability（替代当前 proxy） |
| Heuristic 误差审计 | 30-50 条人工标注 | 用 `scripts/audit_reasoning_depth_heuristic.py` 验证 `classify_reasoning_structure()` 的 precision/recall |

### Phase 2（1-2 周）：高精度 Multi-document

| 任务 | 交付物 | 说明 |
|------|--------|------|
| Element-level cross-doc edges | `cross_doc_edges_v1.jsonl` | embedding 相似度 > 阈值的跨文档元素对 |
| 跨文档 bridge 质量验证 | 小规模 eval | 证明 element-level > doc-level |
| 跨文档 multi-hop 路径 | 路径枚举升级 | 路径可跨越文档边界 |

### Phase 3（1-2 周）：Multi-turn Session

| 任务 | 交付物 | 说明 |
|------|--------|------|
| Turn 生成策略 | 路径→对话链生成 | 每个 hop 对应一个 turn |
| Turn dependency QC | `qc_turn_dependency()` | 删前轮信息后当前轮不可回答 |
| 50-100 条 gold sessions | 人工验证 | 对话自然度 + 依赖真实性 |

### Phase 4（1 周）：M4 联合验证

| 任务 | 交付物 | 说明 |
|------|--------|------|
| M4 全覆盖验证 | eval report | multi-hop + multi-doc + multi-turn + multi-modal |
| 基线对比 | BM25 / Dense / GraphRAG | 在新 M4 数据上的检索效果 |

---

## 关键认知

1. **图已具备 M4 的拓扑基础**（多模态节点 + backbone + 引用边 + 跨文档引用边），欠缺的不是图本身，而是在图上生成 M4 query 的策略和 QC 体系。

2. **不要同时铺开三条线**。优先把严格 multi-hop schema 和真正的 step-deletion 验证做实（Phase 1），这是回答"你们到底是在做 multi-hop 还是在做双证据拼接"的关键。

3. **每个 Phase 的交付物应该是质量而非数量**。50-100 条真正 3-step 的 gold 样本，比 500 条 2-evidence 拼接更有论文价值。

4. **项目对外口径**：Graph-backed M4-Foundation，不是"已完成 M4 数据闭环"。已验证的是图检索增益和跨模态 dual-evidence 生成能力，M4 的三个缺口正在系统性补齐。

5. **区分 Schema-ready 与 Generator-ready**。M4_SCHEMAS.md 定义了目标数据格式，但当前生成脚本仍是 dual-evidence pair 容器 + 新字段透传。3-step native generator（3+ element path 枚举 + Schema 1 原生输出 + element_ids 与 reasoning_steps 一致）是 Phase 1 的核心工程任务。

6. **不因战略升级停摆已有可交付**。在 M4 研究主线之外，保留并行的保底交付线（full-run + eval + GRAPH_ARCHITECTURE 扩充），确保 4 月节点有东西可交。
