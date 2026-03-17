# Document Graph Architecture v2

> 版本：v2.1  
> 更新日期：2026-03-15  
> 目标：将图结构定义从零散说明升级为**独立技术参考文档**，可直接支撑专利申请、周会汇报与论文撰写。

---

## 1. Overview

### 1.1 一句话定义
本项目采用**面向学术论文的多层异构图（multi-layer heterogeneous graph）**，用于统一承载 query 生成、QA、证据定位与多文档多跳推理。

### 1.2 战略定位
- 图（Graph）是核心创新主体，query 生成是图上的下游产物（byproduct）。
- 工程优先级：先保证拓扑可解释、可扩展、可复现，再叠加语义增强。

### 1.3 当前规模（基线快照）
- 文档规模：82 篇 arXiv 论文（LaTeX 可用场景）。
- 图规模：2551 nodes，3471 edges。
- 当前 report 明确统计的边：`paragraph_ref` / `backbone` / `element_ref` / `cross_doc_cite`。

---

## 2. 节点类型（架构定义 5 种；当前拓扑实例化 4 种）

> 关键口径：**“架构定义”与“当前 topology report 实例化”需要分开写**，避免把设计态和运行态混为一谈。

| 类型 | 来源 | 成本 | 关键字段 | 状态 |
|---|---|---|---|---|
| `paragraph` | MinerU / LaTeX 段落块 | 低 | `line_no`, `line_no_end`, `text_snippet`, `paragraph_order` | ✅ 当前实例化 |
| `figure` | MinerU + LaTeX 标签映射 | 低 | `caption`, `image_path`, `page_idx`, `position_idx` | ✅ 当前实例化 |
| `table` | MinerU + LaTeX 标签映射 | 低 | `caption`, `html_content`, `page_idx`, `position_idx` | ✅ 当前实例化 |
| `formula` | MinerU + LaTeX 标签映射 | 低 | `latex_source`, `context`, `page_idx`, `position_idx` | ✅ 当前实例化 |
| `section` | LaTeX `\section{}` 系列 | 低 | `title`, `level`, `line_no~line_no_end` | ⚠️ 设计层定义，当前全局分布未实例化 |

### 2.1 与代码对齐说明（避免误导）
- `scripts/analyze_latex_graph_topology.py` 是当前拓扑图构建、打分、候选枚举的主实现。  
- `src/linkers/unified_graph.py` 定义了另一套双层图接口（`DOCUMENT/FIGURE/TABLE/FORMULA/SECTION` 的 `NodeType`，以及 `EdgeType/Attribution` 枚举）；它可作为统一接口设计参考，但**与当前 topology 图（以 paragraph 为桥接核心）不是同一节点体系，不可混用统计口径**。

### 2.2 当前实例化节点分布（global）
来自 `data/latex_graph_topology_report.json`：
- `paragraph`: 1347
- `figure`: 532
- `table`: 221
- `formula`: 451

---

## 3. 边类型（5 种）

| 类型 | 语义 | 构建方法 | 成本 | 当前数量（report） |
|---|---|---|---|---|
| `backbone` | 阅读顺序边 | 段落按 `line_no` 排序后相邻连边 | 低 | 1269 |
| `paragraph_ref` | 段落引用元素 | `\ref{}` 调用点归属最紧跨度段落，再连目标元素 | 低 | 1688 |
| `element_ref` | 元素间显式引用/共引 | LaTeX `source_label -> target_label` | 低 | 80 |
| `cross_doc_cite` | 跨文档桥接边 | `.bbl` 匹配 citation pair，再做 top2×top2 扩展 | 低 | 434 |
| `section_contains_*` | 章节包含边 | section 到段落/元素最紧跨度匹配 | 低 | 当前 report 未单列 |

> `section_contains_paragraph` 与 `section_contains_element` 在代码逻辑中有构建路径；但当前 report 的全局 `edge_type_distribution` 未单独输出该家族计数。

---

## 4. 获取方式分层（成本矩阵）

### Layer A：纯自动化（低成本，可扩展）
- 输入：PDF + MinerU（不依赖 LaTeX）。
- 产出：paragraph/figure/table/formula 节点、backbone、弱引用边。
- 特点：成本最低，适合万级文档批处理。

### Layer B：LaTeX 增强（低成本，需要源码）
- 输入：`.tex` + `.bbl` + LaTeX 引用图。
- 产出：`paragraph_ref` / `element_ref` / `cross_doc_cite` 等强结构边。
- 关键：三级映射（数字匹配 → caption Jaccard → 顺序兜底）。

### Layer C：LLM 增强（中/高成本，按需）
- 元素级 MoDora `[T]/[M]/[C]` enrichment。
- hub 级语义摘要（50–80 词）。
- 成本估算：1316 元素 × `$0.003~0.005` ≈ `$4~7`（离线一次性）。

---

## 5. Hub 评分体系

### 5.1 Bridge vs Authority 分类

| 类别 | 判定条件 | 适合多跳? |
|---|---|---|
| `bridge` | paragraph 且出边覆盖 ≥2 模态 | ✅ |
| `authority` | 元素节点且 `in_degree > out_degree` | ❌ |
| `mixed` | 其他 | 视情况 |

### 5.2 Hub Score 公式（MVE locked）

```text
hub_score = 0.40×bridge_score
          + 0.35×connectivity_score
          + 0.25×core_module_score
          + 20×pagerank
          - penalty
```

子项口径：
1. `bridge_score`: 2+ 模态=100；1 模态=50；否则 0。  
2. `connectivity_score`: `min(1.0, (total_degree/degree_norm) + (cross_type_edges/degree_norm)) × 100`。  
3. `core_module_score`: 章节关键词权重映射。  
4. `penalty`: authority sink 条件触发时扣分。

### 5.3 当前 hub 相关统计（避免术语混淆）
- Top hubs（`compute_hubs` 取 top-60）：60；`hub_category_breakdown={bridge:60}`。  
- Bridge hubs（`compute_bridge_hubs`）：60（独立算法列表）。  
- Adjacent backbone bridges：369。  

> 备注：`traffic_hubs` 在代码中是基于 `in_degree>0 && out_degree>0` 的辅助列表，不等同于 `bridge_hubs`，也不作为主分类术语写入本节核心定义。

---

## 6. 多跳候选枚举

### 6.1 为什么不用 DFS
骨架边密集时，通用 DFS/BFS 容易把 hop 消耗在段落链上，难以稳定到达第二模态端点。

### 6.2 定向枚举三策略（已实现）
1. **2-hop 文档内**：`[A, hub, B]`  
2. **3-hop 骨架邻居**：`[A, hub, adj, B]`  
3. **跨文档引用**：`[A, hub, B_other]`

### 6.3 多样性控制
- `per_combo_cap`：每个 `(doc_id, modality_set, is_cross_doc)` 最多 5 条。  
- 结构去重：相同元素标签集视为重复。  
- 4 种种子轮换：WHY / WHAT_IF / MISMATCH / CONDITION。  
- 位置优先：优先 `page_idx/position_idx` 可用端点。

---

## 7. 质量门禁

### 7.1 图构建阶段
1. **G1（hub 去重）**：单元素在跨模态对中的出现上限（默认 top-3）。
2. **G2（交叉引用门禁）**：桥接文本须双向提及两端标签。

### 7.2 查询生成阶段（9 项 QC）
`numeric_leakage`, `meta_language`, `is_yes_no_question`, `query_too_long`, `anchor_leakage`, `single_element_answer`, `weak_reasoning_connector`, `template_shortcut`, `architecture_intent_missing`。

---

## 8. Pipeline 总览（6 阶段，详细 ASCII 复用）

```text
┌─────────────────────────────────────────────────────────────────┐
│                    LaTeX 源码（82 篇 arXiv 论文）                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                    │
│  │ .tex 文件 │    │ .bbl 文件 │    │ 引用关系  │                    │
│  └─────┬────┘    └─────┬────┘    └─────┬────┘                    │
│        │               │               │                         │
│        ▼               ▼               ▼                         │
│  ┌──────────────────────────────────────────┐                    │
│  │   阶段1: LaTeX 引用图构建                  │                    │
│  │   LaTeXReferenceExtractor                 │                    │
│  │   → labels + refs + edges + bib per doc   │                    │
│  └───────────────────┬──────────────────────┘                    │
│                      │                                           │
│                      ▼                                           │
│  ┌──────────────────────────────────────────┐                    │
│  │   阶段2: 跨文档引用图                      │                    │
│  │   4级标题匹配（arXiv ID → 归一化标题 →     │                    │
│  │   Jaccard≥0.55）                          │                    │
│  │   → 123 条 citation edges                 │                    │
│  └───────────────────┬──────────────────────┘                    │
└──────────────────────┼──────────────────────────────────────────┘
                       │
┌──────────────────────┼──────────────────────────────────────────┐
│  PDF 解析层（MinerU） │                                          │
│                      │                                           │
│  ┌───────────────────▼──────────────────────┐                    │
│  │   阶段3: 多模态元素提取                    │                    │
│  │   MultimodalRelationshipBuilder           │                    │
│  │   → 1316 elements + 1261 edges            │                    │
│  │   → content_list.json 修复 page_idx       │                    │
│  └───────────────────┬──────────────────────┘                    │
└──────────────────────┼──────────────────────────────────────────┘
                       │
┌──────────────────────┼──────────────────────────────────────────┐
│  图融合与候选生成层    │                                          │
│                      ▼                                           │
│  ┌──────────────────────────────────────────┐                    │
│  │   阶段4: 多层拓扑图融合                    │                    │
│  │                                           │                    │
│  │   ┌─ 3级标签映射 ──────────────────┐      │                    │
│  │   │ 数字匹配 → Jaccard → 顺序映射  │      │                    │
│  │   └────────────────────────────────┘      │                    │
│  │                                           │                    │
│  │   5类节点 + 5类边 → 2551 nodes, 3471 edges │                    │
│  │                                           │                    │
│  │   ┌─ PageRank + Bridge-First Hub 评分 ─┐  │                    │
│  │   │ bridge_score × 0.40                 │  │                    │
│  │   │ + connectivity × 0.35               │  │                    │
│  │   │ + core_module × 0.25                │  │                    │
│  │   │ + 20 × pagerank                     │  │                    │
│  │   │ - authority_sink_penalty             │  │                    │
│  │   └─────────────────────────────────────┘  │                    │
│  │                                           │                    │
│  │   ┌─ 邻接骨架桥检测 ────────────────┐      │                    │
│  │   │ para_i→figure + para_j→table     │      │                    │
│  │   │ → 369 adjacent bridges           │      │                    │
│  │   └────────────────────────────────┘      │                    │
│  │                                           │                    │
│  │   ┌─ 定向枚举（替代DFS）────────────┐      │                    │
│  │   │ 策略1: 2-hop [A, hub, B]        │      │                    │
│  │   │ 策略2: 3-hop [A, hub, adj, B]   │      │                    │
│  │   │ 策略3: 跨文档 [A, hub, B_other]  │      │                    │
│  │   │ + per_combo_cap + 结构去重       │      │                    │
│  │   │ → 500 候选路径                   │      │                    │
│  │   └────────────────────────────────┘      │                    │
│  └───────────────────┬──────────────────────┘                    │
│                      │                                           │
│                      ▼                                           │
│  ┌──────────────────────────────────────────┐                    │
│  │   阶段5: 语义增强（MoDora 整合）          │                    │
│  │                                           │                    │
│  │   元素级: [T]itle/[M]etadata/[C]ontent    │                    │
│  │   枢纽级: hub_semantic_summary            │                    │
│  │   （图感知聚合 ≠ 树感知聚合）              │                    │
│  └───────────────────┬──────────────────────┘                    │
│                      │                                           │
│                      ▼                                           │
│  ┌──────────────────────────────────────────┐                    │
│  │   阶段6: 查询生成 + 9项QC                 │                    │
│  │                                           │                    │
│  │   4 种模态组合 prompt 模板                 │                    │
│  │   + enriched context 注入                 │                    │
│  │   + 双证据信息缺口约束                     │                    │
│  │   + 长短混合强制                           │                    │
│  │   → 双证据多跳查询                        │                    │
│  └──────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. 与 MoDora 的系统对比（补全）

| 维度 | MoDora | 本项目 |
|---|---|---|
| 文档建模 | 单文档层级树（CCTree） | 多文档多层异构图 |
| 跨模态关系 | 隐式（同标题共现） | 显式边（`\ref`/citation/backbone） |
| 跨文档 | 多树简单合并 | citation + hub 引导连接 |
| 语义增强 | 树节点 cascade summarization | 图节点 `[T]/[M]/[C]` + hub 语义摘要 |
| 多跳推理 | 在线逐层剪枝 | 离线定向枚举 + 门禁质控 |
| 输出形态 | 在线检索排名 | 离线训练样本（双证据 query） |
| 成本模型 | 推理时持续 LLM 调用 | 预计算一次，在线低延迟复用 |

---

## 10. 泛化方案（无 LaTeX 场景）

### 10.1 Layer A/B/C 降级路径
- A（low）：MinerU + parser order。  
- B（medium）：embedding 相似边 + rerank。  
- C（high，可选）：仅 hub 邻域 LLM 补全。

### 10.2 Edge availability matrix

| Edge type | With LaTeX | PDF-only | 策略 |
|---|---|---|---|
| backbone | yes | yes | parser order |
| element_ref | yes | partial/no | caption/anchor + LLM fallback |
| paragraph_ref | yes | partial/no | discourse cue + embedding |
| cross_doc_cite | yes | partial | bibliography parse + title match |
| section_contains | yes | partial | heading detection + span heuristic |

---

## 11. 当前统计数据（Topology Report Snapshot）

### 11.1 全局规模
| 指标 | 数值 |
|---|---|
| nodes | 2551 |
| edges | 3471 |
| graph density | 0.00053359 |

### 11.2 节点分布（当前实例化）
| node type | count |
|---|---|
| paragraph | 1347 |
| figure | 532 |
| table | 221 |
| formula | 451 |

### 11.3 边分布（report 已输出）
| edge type | count |
|---|---|
| paragraph_ref | 1688 |
| backbone | 1269 |
| element_ref | 80 |
| cross_doc_cite | 434 |

### 11.4 关键派生指标
| 指标 | 数值 | 备注 |
|---|---|---|
| citation pairs | 123 | `.bbl` 匹配文档对 |
| cross_doc edges added | 434 | top2×top2 扩展 |
| label mapping rate (topology口径) | 49.75% | 599 / 1204 |
| label mapping rate (patent口径) | 49.8% | 958 / 1924（LaTeX labels 总量口径） |
| top hubs | 60 | `compute_hubs` 取 top-k |
| bridge hubs | 60 | `compute_bridge_hubs` 独立列表 |
| adjacent bridges | 369 | 骨架相邻桥接 |
| candidate_count | 500 | 其中跨文档 170 |

> 注：`hubs_summary.note_scoring` 在 report JSON 中是历史字符串，可能与当前代码公式不一致；本文件以代码实现公式为准。

---

## 12. 关键代码文件索引（创新点→实现）

| 创新点 | 文件 | 核心函数/结构 |
|---|---|---|
| 节点/边建图 | `scripts/analyze_latex_graph_topology.py` | `Node`, `Edge`, `build_topology_graph()` |
| 跨文档边扩展 | `scripts/analyze_latex_graph_topology.py` | `build_cross_doc_edges()` |
| Hub 打分 | `scripts/analyze_latex_graph_topology.py` | `compute_hubs()` |
| Bridge hubs 列表 | `scripts/analyze_latex_graph_topology.py` | `compute_bridge_hubs()` |
| 邻接骨架桥 | `scripts/analyze_latex_graph_topology.py` | `compute_adjacent_backbone_bridges()` |
| 定向枚举与去重 | `scripts/analyze_latex_graph_topology.py` | `enumerate_candidates_from_bridge_hubs()` |
| 双层图接口枚举 | `src/linkers/unified_graph.py` | `NodeType`, `EdgeType`, `Attribution` |

---

## 13. Mentor 周会问答模板

- **节点是什么？** 当前运行图的核心节点是 paragraph+figure+table+formula，section 是架构预留节点。  
- **边是什么？** backbone / paragraph_ref / element_ref / cross_doc_cite 为当前主统计边；section_contains 为实现层补充。  
- **成本多高？** Layer A/B 低成本可扩展，Layer C 按需启用（1316 元素约 `$4~7`）。  
- **哪些自动化？** 从解析、映射、打分、枚举到 QC 基本全自动，人工主要负责阈值校准与抽检。

---

## 14. 维护规范

1. 每次周会前若算法或统计变化，必须更新本文件（不得只落在 `CLAUDE.md`）。
2. 新增边类型或评分项时，至少同步更新第 3/5/11 节。
3. 专利与论文引用本文件时，必须注明统计口径（topology vs patent）。
