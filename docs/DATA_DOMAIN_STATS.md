# Data Domain & Statistical Distributions — delivery_v1

`data/03_queries/delivery_v1_2026-04-13.jsonl`（473 条 queries，53 篇 arXiv 论文）。

源数据：`delivery_v1_2026-04-13_stats.json` + `delivery_v1_2026-04-13_stats_extended.json`。

## 1. 语料 (Corpus)

| 项 | 值 |
|----|----|
| **唯一文档数** | **53** |
| 文档来源 | arXiv，全部 CS 类（cs.LG / cs.CY / cs.AI） |
| 主题域 | **algorithmic fairness, ML bias, LLM** —— 来自种子论文 1908.09635 的引用网络 BFS |
| 模态来源 | MinerU 解析 PDF → multimodal_elements.json (1316 elements 覆盖 ~76 文档) |
| LaTeX 源覆盖 | 部分文档有 LaTeX 源码，用于构建 reference graph |

### Domain 单一性 — 重大缺口

**当前所有 53 篇文档都属于 CS / 公平性 / LLM 三个相互关联的子领域。** 没有医疗、金融、物理、生物等其他学科覆盖。要构造通用的多模态推理 benchmark，必须扩展非 CS 语料。

**正在准备的扩展**：1040 篇 pruned graph 已构建（`data/01_graphs/pruned_graph_v2.json`），但仍是同一引用网络的扩展，**仍以 CS 为主**。真正跨学科需要新的 seed papers + 重跑下载 + MinerU + 图构建。

## 2. 模态 (Modality)

| 模态 | 出现次数 (token level) | 说明 |
|------|------------------------|------|
| figure | 428 | 包含 plot / diagram / architecture 多种细分（细分标记在 enriched_metadata 中） |
| table | 339 | 包括实验结果表、信息汇总表 |
| formula | 179 | 来自 LaTeX equation block |
| **合计** | **946** | 每条 query 引用 2 个 element ⇒ 473 × 2 = 946 |

### Pair type 分布

| pair_type | 条数 | 占比 |
|-----------|-----|------|
| figure+table | 289 | 61.1% |
| figure+formula | 129 | 27.3% |
| formula+table | 50 | 10.6% |
| figure+figure | 5 | 1.1%（来自 long_chain 批次） |

**图像覆盖**：462 / 473 = **97.7%** 条 queries 至少含一张图片路径。

## 3. 推理深度 (Reasoning depth)

### 3.1 `hop_distance`（图上路径长度）

| hop_distance | 条数 |
|--------------|-----|
| 2 | 189 |
| 3 | 275 |
| 4 | 2 |
| 5 | 7 |

### 3.2 `reasoning_structure`（启发式标注）

| structure | 条数 | 占比 |
|-----------|-----|------|
| parallel（并行取证）| 318 | 67.2% |
| serial（串行链）| 124 | 26.2% |
| mixed | 19 | 4.0% |
| unknown | 12 | 2.5% |

### 3.3 `m4_step_deletion_proxy`

86 / 473 = **18.2%** 条 query 的 answer 含 ≥ depth-1 个因果连接词 → proxy 通过。

`m4_is_true_multihop`（最严格判定）**0 条**。这印证了目标定位中的"M4-Foundation 而非 M4 strict"。

## 4. Query 长度

| 桶 | 条数 |
|----|-----|
| short (< 15 词) | 152 |
| medium (15-25 词) | 26 |
| long (25-40 词) | 295 |

每个 candidate pair 通常同时产出 short + long 两条以保证 length mix。

## 5. LLM Grounding 置信度

| 指标 | 值 |
|------|----|
| 评估条数 (n) | 469 / 473（4 条 grounding 调用 error） |
| 平均 confidence | **0.915** |
| min | 0.000 |
| max | 0.990 |
| **% ≥ 0.85** | **86.4%** |

86.4% 的 query 由独立的 LLM judge 高置信判定为「answer 完全有 evidence 支撑」。

## 6. 风格与 Persona

### Query style

| style | 条数 |
|-------|-----|
| academic | 354 (74.8%) |
| real_user | 119 (25.2%) |

### Persona 分布

`none`（不开人设）339 条；其余 134 条覆盖 53 种不同 PersonaHub 人设（statistician_skeptic, ml_engineer_production, parent_concerned_about_ai, civil_rights_lawyer 等）。

**多样性**：persona 数量 / 启用条数 ≈ 0.40，平均每个人设出现 2.5 次，分布相对均匀（最高 6 条 `curious_teenager`）。

## 7. Top 10 文档（按 query 数）

| arXiv ID | queries |
|----------|---------|
| 1802.08139 | 24 |
| 1809.04737 | 24 |
| 2005.07293 | 22 |
| 2103.11320 | 22 |
| 1907.12059 | 19 |
| 1810.01943 | 18 |
| 1907.06430 | 17 |
| 1511.00830 | 16 |
| 1610.08452 | 16 |
| 1709.02012 | 15 |

Per-doc query count 分布（53 篇）：

| 区间 | 文档数 |
|------|--------|
| 1-4 | 16 |
| 5-8 | 15 |
| 9-15 | 13 |
| 16-25 | 9 |

中位数约 8 条 / 文档，最长尾文档 24 条。

## 8. 来源批次

| `_source_batch` | 条数 |
|------|------|
| sweep_m2_academic | 85 |
| sweep_l3_mixed | 71 |
| sweep_m2_mixed_persona | 65 |
| old_l3_v3 | 148 |
| sweep_l3_academic_persona | 24 |
| sweep_l3_academic | 23 |
| sweep_l3_mixed_persona | 17 |
| old_m2 | 28 |
| old_long_chain | 12 |

**Sweep（2026-04-12）** 配置 6 个：L3×{academic, academic+persona, mixed, mixed+persona} + M2×{academic, mixed+persona}。

## 9. 训练 split 分布（`data/07_training/delivery_v1/`）

| split | triplets |
|-------|----------|
| train | 407 (86.0%) |
| val | 44 (9.3%) |
| test | 22 (4.7%) |

按 `doc_id` hash 划分，同一文档全部进同一 split。负样本策略：`graph_aware`（默认 3 个 / query）。

## 10. 缺口与未来计划

| 维度 | 当前 | 目标 |
|------|------|------|
| 文档数 | 53 | 1000+（pruned graph 已有 1040，阻塞于 enrichment + API key） |
| 学科 | 仅 CS | + 医疗 / 金融 / 物理 / 生物 等 |
| 严格 multi-hop | 0 条 | 50-100 条 step-deletion 验证通过的 L3 |
| Cross-document | 0 条 | element-level cross-doc edges + 跨文档 query |
| Multi-turn | 0 条 | Phase 3 session 化 |
| Quality tier | 全部 `unknown` | 抽样 50-100 条手工标 gold/silver |
