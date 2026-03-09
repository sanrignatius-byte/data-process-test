# 专利技术摘要：多源异构文档图构建与多跳跨模态候选生成方法

> 版本：v1.1 | 日期：2026-03-09
> 状态：内部技术梳理，供专利撰写参考

---

## 一、发明名称（建议）

**"一种融合 LaTeX 结构信号与多模态解析的多层文档图构建及跨模态多跳候选自动生成方法"**

英文：
*"A Multi-Layer Document Graph Construction Method Integrating LaTeX Structural Signals with Multimodal Parsing for Automated Cross-Modal Multi-Hop Candidate Generation"*

---

## 二、技术问题（要解决什么）

现有多模态文档理解方法存在以下缺陷：

1. **树状结构表达力不足**：现有方法（如 MoDora/CCTree）将文档建模为层级树，无法显式表达跨模态引用关系（如"段落同时引用图和表"）和跨文档引用边。
2. **OCR/解析层信息丢失**：PDF 解析工具（MinerU 等）提取的元素缺少结构语义，`page_idx` 常为 0，元素间关系主要靠空间共现推断。
3. **多跳路径发现效率低**：传统 DFS 在文档图中容易在段落骨架链上耗尽跳数，无法有效触达不同模态的元素。
4. **跨文档关联信号弱**：基于实体倒排索引的跨文档匹配噪声大，"相似 ≠ 多跳有用"。
5. **非文本元素语义表达不充分**：图、表、公式的原始 caption/content 语义可读性不稳定，直接影响下游查询生成质量。

---

## 三、技术方案总览

本发明提出一种 **六阶段流水线**，从 LaTeX 源码 + PDF 解析结果出发，构建多层异构文档图，并通过枢纽检测与定向枚举自动生成跨模态多跳候选对。

```
阶段1  LaTeX引用图构建         ──→ 文档内引用DAG（labels + refs + edges）
阶段2  跨文档引用图构建         ──→ 文档间citation edges（标题模糊匹配）
阶段3  多模态元素提取与关系建图  ──→ MinerU元素 + 跨模态对
阶段4  多层拓扑图融合与枢纽检测  ──→ 5类节点 × 5类边 + Bridge/Authority分类
阶段5  语义增强（MoDora [T]/[M]/[C]） ──→ 结构化元素描述 + 枢纽语义摘要
阶段6  多跳候选生成与质量控制    ──→ 双证据查询 + 9项QC门禁
```

---

## 四、核心创新点（权利要求基础）

### 创新点 1：多源异构多层文档图

**区别于现有技术**：不使用单一树或平面图，而是构建包含 **5 种节点类型 × 5 种边类型** 的多层异构图。

#### 节点类型（5 种）：
| 类型 | 来源 | 说明 |
|------|------|------|
| **section** | LaTeX `\section{}` | 章节节点，含层级、行号范围、标题 |
| **paragraph** | LaTeX 空行分隔的段落块 | 骨架节点，含行号范围、文本片段 |
| **figure** | LaTeX `\label{fig:...}` → MinerU 映射 | 图片元素，含真实 `page_idx`、`position_idx`、图像路径 |
| **table** | LaTeX `\label{tab:...}` → MinerU 映射 | 表格元素，含 HTML/Markdown 原始内容 |
| **equation** | LaTeX `\label{eq:...}` → MinerU 映射 | 公式元素，含 LaTeX 源码 |

#### 边类型（5 种）：
| 类型 | 语义 | 构建方法 |
|------|------|----------|
| **paragraph_ref** | 段落引用元素 | 每个 `\ref{}` 调用点归属到其包含段落（最紧跨度匹配） |
| **element_ref** | 元素间共引 | LaTeX 图内 source_label → target_label 的共引边 |
| **backbone** | 阅读顺序 | 同文档段落按行号排序后，相邻段落连边 |
| **section_contains_\*** | 章节包含 | 章节节点到其包含的段落/元素的包含边（最紧跨度匹配） |
| **cross_doc_cite** | 跨文档引用 | 源文档高出度段落 → 目标文档高入度元素 |

**关键技术细节**：
- **段落归属算法**（`_find_para_for_line`）：给定一个 `\ref{}` 出现的 `(file_path, line_no)`，遍历所有段落节点，找到行号范围覆盖该位置且跨度最小（最紧）的段落。
- **跨文档边构建**（`build_cross_doc_edges`）：对每条引用边，取源文档出度 top-2 段落和目标文档入度 top-2 元素，两两连边（最多 4 条/引用）。

---

### 创新点 2：LaTeX 标签到 MinerU 元素的多策略映射

**问题**：LaTeX 标签（如 `fig:arch`）和 MinerU 元素 ID（如 `1306.5204_figure_2`）之间没有直接对应关系，MinerU 的编号常与 LaTeX 编号存在 offset。

**三级级联匹配策略**（`map_label_to_element`）：

```
策略1: 数字匹配
  ├─ 提取标签键尾部数字（如 fig:3 → 3）
  ├─ 匹配 MinerU 同类型元素编号索引
  └─ 若失败，按分隔符切分标签键的每段再试

策略2: 标题Jaccard相似度
  ├─ 对 LaTeX 标签 caption 和 MinerU 元素 caption 做词级 Jaccard
  └─ 阈值 ≥ 0.25（从 0.35 放宽，提升召回）

策略3: 顺序匹配（兜底）
  ├─ 将同文档同类型的 LaTeX 标签按行号排序
  ├─ 将同文档同类型的 MinerU 元素按 position_idx 排序
  └─ 第 N 个标签对应第 N 个元素（1:1，不冲突已有映射）
```

**实际效果**：49.8% 标签匹配率（958/1924），含策略 3 后候选映射率达 41.2%（206/500）。

---

### 创新点 3：跨文档引用图构建（基于 .bbl 标题模糊匹配）

**四级级联匹配策略**（`match_bib_entry_topk`）：

```
策略1: 显式 arXiv ID（regex: arXiv:YYMM.NNNNN）→ 置信度 1.0
策略2: 裸 arXiv ID 模式（regex: \b\d{4}\.\d{4,5}\b）→ 置信度 0.9（带年月范围校验）
策略3: 精确归一化标题匹配 → 置信度 0.95
策略4: 模糊标题匹配（词级 Jaccard ≥ 0.55）→ 置信度按 Jaccard 值
```

**质量验证**：人工抽查 Jaccard ≥ 0.55 区间，误匹配率 0%。
**产出**：123 条跨文档引用边，最大连通分量 55 篇。

---

### 创新点 4：Bridge-First 枢纽检测算法

**问题**：传统 PageRank/度中心性评分会将高被引元素（如被 49 个段落引用的公式）排在顶部，但这类节点是"权威汇聚点"（authority sink），不适合作为多跳桥接。

**Bridge vs Authority 分类**：

| 类别 | 判定条件 | 适合多跳? |
|------|----------|-----------|
| **bridge** | paragraph 节点且出边覆盖 ≥2 种模态 | ✅ 最适合 |
| **authority** | 元素节点且入度 > 出度 | ❌ 被引多但不桥接 |
| **mixed** | 其他 | 视情况 |

**Hub 评分公式**：
```
hub_score = 0.40 × bridge_score + 0.35 × connectivity_score + 0.25 × core_module_score + 20 × pagerank - penalty

其中：
  bridge_score = 100 if 出边覆盖≥2模态, 50 if 1模态, 0 otherwise
  connectivity_score = min(1.0, (total_degree / degree_norm) + (cross_type_edges / degree_norm)) × 100
  core_module_score = 章节关键词匹配权重 × 100
    - introduction: 1.0, experiment/ablation: 0.9, method/architecture: 0.8
    - conclusion: 0.6, related_work: 0.3
    - 架构图额外加分
  penalty = 20.0 if (in_degree > 2×out_degree AND 仅1种模态)  # authority sink 惩罚
```

**排序策略**：Bridge-First —— bridge 类节点无条件排在 authority 之前，bridge 内部按 hub_score 排序。

**效果**：top-60 hub 100% 为 bridge 类别（覆盖 31 篇文档），authority sink 全部被清除出排名。

---

### 创新点 5：邻接骨架桥检测（Adjacent Backbone Bridge）

**核心思想**（来自导师建议）：如果骨架链上相邻的两个段落 para_i 和 para_j 分别引用不同模态的元素（如 para_i 引用图，para_j 引用表），则它们构成一个"桥接片段"。

**算法**（`compute_adjacent_backbone_bridges`）：
```
对每篇文档的段落按 paragraph_order 排序:
  for i in range(len(paragraphs) - 1):
    para_i, para_j = paragraphs[i], paragraphs[i+1]

    # 验证存在骨架边
    if para_j ∉ out_adj[para_i]: continue

    # 收集各自引用的模态
    mods_i = {出边指向的元素模态}
    mods_j = {出边指向的元素模态}

    # 必须各引至少1个元素，合并覆盖≥2种模态
    if |mods_i| == 0 or |mods_j| == 0: continue
    if mods_j ⊆ mods_i: continue  # j只是i的子集，无新模态
    if |mods_i ∪ mods_j| < 2: continue

    # 收集路径: [elem_from_i, para_i, para_j, elem_from_j]
    → 输出候选
```

**产出**：369 条邻接桥（覆盖 68 篇文档）。

---

### 创新点 6：定向枚举替代 DFS

**问题**：传统 DFS（`max_hops=5`）在骨架链（1269 条边）上迷失，消耗跳数但无法到达第二种模态。

**定向枚举三策略**（`enumerate_candidates_from_bridge_hubs`）：

```
对每个 bridge hub 段落（直接引用 ≥2 种模态）：

策略1: 2-hop 文档内
  [elem_A, hub_para, elem_B]
  条件: elem_A 和 elem_B 模态不同

策略2: 3-hop 骨架邻居
  [elem_A, hub_para, adjacent_para, elem_B]
  条件: adjacent_para 通过骨架边连接到 hub_para，
        elem_B 由 adjacent_para 引用，与 elem_A 模态不同

策略3: 跨文档引用
  [elem_A_src, hub_para, elem_B_tgt]
  条件: elem_B_tgt 在目标文档中，通过 cross_doc_cite 边到达，
        与 elem_A 模态不同
```

**多样性控制**：
- **Per-combo cap**：MAX_PER_COMBO = 5 per (doc_id, modality_frozenset, is_cross_doc)
- **结构去重**：元素标签集合相同的路径视为重复
- **4 种种子类型轮换**：WHY / WHAT_IF / MISMATCH / CONDITION，由 `hash(tuple(path)) % 4` 决定
- **位置优先排序**：有真实 `page_idx` 的元素优先配对（改善物理距离覆盖）

**效果**：500 条候选，496/500 独特种子（99.2%）；分布：figure+table 247 / figure+formula 153 / formula+table 100；文档内 330 + 跨文档 170。

---

### 创新点 7：MoDora [T]/[M]/[C] 图节点语义增强

**区别于 MoDora 原方法**：MoDora 对树节点做层级聚合（子→父），我们对图节点做跨模态桥接聚合。

#### 元素级增强（`enrich_elements_modora.py`）

对每个 figure/table/formula 节点，用类型特化 prompt 生成三元组：
```
[T]itle: 5-15词描述性标题（不重复caption）
[M]etadata: {figure_type/table_type/formula_type, keywords, 轴/列/变量信息}
[C]ontent: 2-4句全面语义描述（不用元语言）
```

三种类型的 prompt 差异：
| 类型 | 特化字段 | 分类标签 |
|------|----------|----------|
| **figure** | figure_type (12类), axes, num_series | line_plot, architecture_diagram, heatmap... |
| **table** | table_type (8类), columns, num_rows, best_values | results_comparison, ablation_study... |
| **formula** | formula_type (10类), variables, domain | loss_function, objective, constraint... |

#### 枢纽级聚合（`build_hub_semantic_summary`）

```
输入: 一对已 enriched 的元素 + edge_contexts
输出: hub_semantic_summary 字符串

格式: "[FIGURE A] <title>: <content> | [TABLE B] <title>: <content> | [BRIDGE] <edge_context> | [KEYWORDS] kw1, kw2, ..."
```

**与 MoDora cascade summarization 的关键区别**：
- MoDora：自底向上，子节点关键词聚合到父节点 → **层级语义传播**
- 本方法：从两个端点元素 + 桥接段落上下文 → **跨模态关系语义融合**，不依赖层级结构

---

### 创新点 8：MinerU `page_idx` 修复（content_list.json 顺序匹配）

**问题**：MinerU 输出的 `multimodal_elements.json` 中所有元素 `page_idx = 0`（parser bug）。

**修复方法**（`build_real_page_index`）：
```
对每篇文档:
  1. 读取 content_list.json（MinerU 的另一输出，含正确 page_idx）
  2. 按类型分组（image/table/equation）
  3. 将 multimodal_elements 同类型元素按 position_idx 排序
  4. 第 N 个 content_list 项对应第 N 个 multimodal_elements 元素
  5. 复制 page_idx
```

**覆盖率**：94.8%。

---

### 创新点 9：LaTeX 跨模态链接质量门禁（G1 + G2）

**G1（Hub 去重）**：每个元素在跨模态对中最多出现 3 次（按 quality_score 排序取 top-3）。
- 防止单个高频被引元素（如被引 49 次的 Table 9）产生 O(N) 虚假对。

**G2（交叉引用门禁）**：bridge_text 必须同时提及两端的标签。
- 条件：`ctx_a mention label_b OR ctx_b mention label_a`
- 否则硬性丢弃（proximity 仅含一端引用不算真正的桥接）。

---

### 创新点 10：9 项查询质量控制引擎

生成的每条查询经过 9+ 项自动 QC 检查：

| QC 检查 | 方法 | 阈值 |
|---------|------|------|
| **numeric_leakage** | 查询中包含具体数值（排除 0/1/年份） | ≥1 个可疑数字 |
| **meta_language** | 正则匹配禁用词（figure/table/equation/as shown in） | 任何命中 |
| **is_yes_no_question** | 检测 Do/Does/Is/Are/Can 等开头 | 任何命中 |
| **query_too_long** | 超过 30 词 | >30 words |
| **anchor_leakage** | 查询 token 与 evidence span token 的 Jaccard | >0.20 |
| **single_element_answer** | 答案必须引用双元素证据（overlap + balance 检查） | MIN_OVERLAP 按类型, BALANCE ≥0.20 |
| **weak_reasoning_connector** | 答案须含关系连接词（because/due to/consistent with...） | 至少 1 个 |
| **template_shortcut** | 禁止模板化开头（Which component.../How does X relate...） | 任何命中 |
| **architecture_intent_missing** | 架构图查询须含结构设计意图词 | 仅架构图 pair |

---

## 五、整体系统架构图

```
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

## 六、与现有技术对比

### 6.1 与 MoDora (SIGMOD 2026) 对比

| 维度 | MoDora | 本发明 |
|------|--------|--------|
| 文档建模 | 单文档层级树（CCTree） | 多文档多层异构图（5类节点×5类边） |
| 跨模态关系 | 隐式（同标题共现） | 显式边（LaTeX \ref 共引、proximity、citation） |
| 跨文档 | 多树简单合并 | 引用边 + 高度/出度启发式连接 |
| 语义增强 | 树节点 cascade summarization | 图节点 [T]/[M]/[C] + 枢纽桥接聚合 |
| 多跳推理 | 在线 LLM 逐层剪枝 | 离线预计算候选路径 + 定向枚举 |
| 公式处理 | 非核心 | 一等图节点（含类型分类与变量提取） |
| 输出 | 在线检索排名 | 离线训练数据（双证据查询） |
| 成本模型 | 推理时多次 LLM 调用 | 一次性离线生成，在线快速检索 |

### 6.2 与传统文档图方法对比

| 维度 | 传统方法 | 本发明 |
|------|----------|--------|
| 图结构 | 单类型节点+单类型边 | 异构多层（5+5） |
| 来源融合 | 单一来源（PDF或LaTeX） | LaTeX源码 + MinerU解析 双源融合 |
| 枢纽检测 | PageRank或度中心性 | Bridge-First 分类 + 多信号加权 + authority惩罚 |
| 路径发现 | BFS/DFS | 定向枚举（针对多模态桥接优化） |
| 骨架建模 | 无 | 段落阅读顺序骨架链 |
| 邻接桥 | 无 | 骨架相邻段落跨模态引用检测 |

---

## 七、实验数据支撑

| 指标 | 数值 |
|------|------|
| 文档规模 | 82 篇 arXiv 论文（种子论文 1908.09635） |
| 多模态元素 | 1316 个（841 figure + 334 table + 141 formula） |
| 图规模 | 2551 nodes, 3471 edges |
| 骨架边 | 1269 条 |
| 段落引用边 | 1688 条 |
| 跨文档引用边 | 434 条（123 citation edges × top2×top2 扩展） |
| 标签匹配率 | 49.8%（三级级联策略） |
| Bridge hub | 60 个（覆盖 31/82 篇文档） |
| 邻接骨架桥 | 369 条（覆盖 68/82 篇文档） |
| 候选对 | 500 条（figure+table: 247, figure+formula: 153, formula+table: 100） |
| enrichment 后候选 | 206 对（映射率 41.2%） |
| 查询生成 | 252 条，113 QC 通过（44.8%） |
| 最佳版本通过率 | 64.4%（v4.2, PhD persona + verb diversity） |

---

## 八、权利要求草案建议

### 独立权利要求 1（方法）
一种多层文档图构建方法，包括：
1. 从 LaTeX 源码提取标签（label）、引用（ref）和参考文献（bib），构建文档内引用有向无环图；
2. 通过参考文献标题的多级模糊匹配构建跨文档引用边；
3. 从 PDF 解析结果提取多模态元素（图/表/公式），通过多策略级联映射将 LaTeX 标签关联到解析元素；
4. 构建包含至少五种节点类型和五种边类型的多层异构文档图；
5. 基于 Bridge-First 枢纽检测算法识别跨模态桥接段落；
6. 通过定向枚举策略从桥接段落出发生成多跳候选路径。

### 独立权利要求 2（系统）
一种多跳跨模态候选生成系统，包括：
- LaTeX 引用图构建模块
- 跨文档引用图构建模块
- 多模态元素提取与映射模块
- 多层图融合与拓扑分析模块
- 枢纽检测与候选枚举模块
- 语义增强模块（元素级 + 枢纽级）
- 查询生成与质量控制模块

### 从属权利要求建议
- 三级标签映射策略（数字→Jaccard→顺序）
- 骨架边与邻接骨架桥检测
- PageRank 融合 bridge/connectivity/core_module 的加权评分
- Authority sink 惩罚机制
- content_list.json 顺序匹配修复 page_idx
- G1（hub 去重）+ G2（交叉引用门禁）质量门禁
- 4 种种子类型轮换（WHY/WHAT_IF/MISMATCH/CONDITION）
- MoDora [T]/[M]/[C] 图节点语义增强与枢纽级聚合
- 9 项查询质量自动控制

---

## 九、关键代码文件对应表

| 创新点 | 主文件 | 核心函数 |
|--------|--------|----------|
| 创新1 多层图 | `analyze_latex_graph_topology.py` | `build_topology_graph()` |
| 创新2 标签映射 | `analyze_latex_graph_topology.py` | `map_label_to_element()` |
| 创新3 跨文档引用 | `build_citation_graph.py` | `match_bib_entry_topk()` |
| 创新4 枢纽检测 | `analyze_latex_graph_topology.py` | `compute_hubs()`, `compute_bridge_hubs()` |
| 创新5 邻接骨架桥 | `analyze_latex_graph_topology.py` | `compute_adjacent_backbone_bridges()` |
| 创新6 定向枚举 | `analyze_latex_graph_topology.py` | `enumerate_candidates_from_bridge_hubs()` |
| 创新7 语义增强 | `enrich_elements_modora.py`, `enrich_hub_candidates.py` | `build_hub_semantic_summary()` |
| 创新8 page_idx修复 | `analyze_latex_graph_topology.py` | `build_real_page_index()` |
| 创新9 质量门禁 | `build_latex_cross_modal_links.py` | G1/G2 逻辑 |
| 创新10 QC引擎 | `generate_multihop_l1_queries.py` | `perform_qc()` 及 9 项检查函数 |

---

## 十、补充创新点（深度代码分析发现）

以下创新点是对前述 10 项的补充，涉及 pipeline 中更底层的算法设计。

### 补充创新点 A：LaTeX 主文件发现与置信度排序

**文件**：`src/parsers/latex_reference_extractor.py`（`_find_all_main_tex()`）

**问题**：arXiv 解压后的 LaTeX 源码目录中可能有多个 `.tex` 文件，需要识别真正的主文件。

**三级置信度排序**：
| 策略 | 条件 | 置信度 |
|------|------|--------|
| `\documentclass` + `\begin{document}` | 同时存在 | 0.95 |
| `\documentclass` 单独出现 | 无 `\begin{document}` | 0.85 |
| 约定文件名（main.tex, paper.tex 等） | 文件名匹配 | 0.70 |
| 最大文件 | 按文件大小兜底 | 0.40 |

**专利价值**：解决了碎片化 LaTeX 归档包的文件发现问题，提供可解释的置信度排名。

---

### 补充创新点 B：递归 `\input{}` 解析与行号溯源映射

**文件**：`src/parsers/latex_reference_extractor.py`（`_resolve_inputs()`）

**技术方案**：
1. 递归解析 `\input{file}` / `\include{file}` 链（含循环检测）
2. 构建 `merged_line_no → (original_line_no, source_file)` 映射
3. 支持花括号和无括号两种语法

**专利价值**：任何从合并文本中提取的引用都能精确追溯到原始 `.tex` 文件和行号，实现"来源可追溯性"（provenance tracking）。

---

### 补充创新点 C：环境栈跟踪与标签类型推断

**文件**：`src/parsers/latex_reference_extractor.py`（`_build_env_map()`, `_infer_label_type()`）

**标签类型推断三级级联**：
```
级别1: 冒号前缀策略
  fig:xxx → figure, tab:xxx → table, eq:xxx → equation

级别2: 环境栈策略
  当前处于 \begin{figure}...\end{figure} 内 → figure
  当前处于 \begin{equation}...\end{equation} 内 → equation

级别3: 子串启发式
  标签含 "figure" → figure, 含 "table" → table
```

**环境栈容错**：容忍不匹配的 `\begin{}`/`\end{}` 对（实际 LaTeX 源码常有此类错误）。

---

### 补充创新点 D：字符距离指数衰减质量评分

**文件**：`scripts/build_latex_cross_modal_links.py`（`_quality_score()`）

**公式**：
```
quality_score = min(conf_a, conf_b) × exp(-char_dist / DECAY_CONST)

其中:
  conf_a, conf_b: 两端标签映射置信度
  char_dist: LaTeX 源码中两个 \ref{} 调用的字符距离
  DECAY_CONST = 500.0（调优值）
```

**距离衰减效果**：
| 距离 | 乘数 | 含义 |
|------|------|------|
| 0（直接边） | ≈ 1.0 | 无衰减 |
| 500 字符 | ≈ 0.37 | 约 2-3 段 |
| 1000 字符 | ≈ 0.14 | 约 5-6 段 |

**专利价值**：提供连续型置信度度量，替代二值"接受/拒绝"判断；对三种发现策略（直接边、proximity、段落共引）统一适用。

---

### 补充创新点 E：跨文档引用匹配的歧义度检测

**文件**：`scripts/build_citation_graph.py`（`compute_match_margin()`, `is_ambiguous_match()`）

**方法**：
```
margin = top_confidence - runner_up_confidence
if margin < 0.10: 标记为 ambiguous
```

每条引用边标注 `match_margin` 和 `is_ambiguous` 字段。

**专利价值**：超越传统二值匹配（accept/reject），提供匹配可信度的梯度信息，可用于下游训练数据的置信度加权。

---

### 补充创新点 F：引用图度数预算抑制（Hub Suppression）

**文件**：`scripts/build_citation_graph.py`（`suppress_hub_citers()`）

**方法**：
- 出度超限（`> max_out_degree=10`）的"多产引用者"：保留置信度 top-N
- 入度超限（`> max_in_degree=15`）的"高被引论文"：保留置信度 top-N

**专利价值**：与创新点 9（G1 hub 去重）形成两级度数控制体系——G1 在元素级，本方法在论文级。

---

### 补充创新点 G：多模态元素嵌入式检测（从文本流中提取表/公式）

**文件**：`src/linkers/multimodal_relationship_builder.py`

**技术方案**：
- **嵌入式表格检测**（`_extract_tables_from_text()`）：识别 HTML `<table>` 和 Markdown `|...|` 格式的内嵌表格
- **嵌入式公式检测**（`_extract_formulas_from_text()`）：识别 `$$...$$` 显示数学环境，提取方程编号
- 两者均从文本正文中提取，而非仅依赖 MinerU 的独立元素记录

**专利价值**：弥补 PDF 解析器将部分表/公式识别为普通文本段落的遗漏，提升元素召回率。

---

### 补充创新点 H：质量分层标签体系（Gold/Silver/Trash）

**文件**：`scripts/build_latex_cross_modal_links.py`

**分层标准**：
| 层级 | 条件 |
|------|------|
| **Gold** | 直接边 + 双端匹配置信度 ≥ 0.9 |
| **Silver** | proximity 策略 + 字符距离 ≤ 200 + 无共引惩罚 |
| **Trash** | quality_score < 0.2 或字符距离 > 600 |

**专利价值**：支持课程学习（curriculum learning）——训练时先用 Gold 再逐步引入 Silver，Trash 永不进入训练集。

---

## 十一、权利要求补充建议

基于补充创新点，建议增加以下从属权利要求：

- **从属权利要求 10**（关于创新点 A）：所述 LaTeX 引用图构建步骤中，通过多级置信度排序自动发现 LaTeX 主文件。
- **从属权利要求 11**（关于创新点 B）：所述引用提取步骤包括递归解析 `\input` 链并维护行号到源文件的溯源映射。
- **从属权利要求 12**（关于创新点 D）：所述跨模态对质量评分使用字符距离指数衰减函数 `min(conf_a, conf_b) × exp(-d/τ)`。
- **从属权利要求 13**（关于创新点 E）：所述跨文档引用匹配包括歧义度检测步骤，其中匹配余量小于阈值的边被标记为不确定。
- **从属权利要求 14**（关于创新点 F）：所述跨文档引用图构建包括两级度数预算控制——引用图级别抑制 + 元素对级别去重。
- **从属权利要求 15**（关于创新点 H）：所述候选对被分为 Gold/Silver/Trash 三层质量等级，支持课程学习策略。

---

## 十二、完整创新点清单（总计 18 项）

| # | 创新点 | 类别 | 新颖性级别 |
|---|--------|------|------------|
| 1 | 5类节点×5类边多层异构图 | 图结构 | ★★★ 高（vs 树/平面图） |
| 2 | 三级标签映射（数字→Jaccard→顺序） | 实体对齐 | ★★ 中 |
| 3 | 四级跨文档标题模糊匹配 | 实体对齐 | ★★ 中 |
| 4 | Bridge-First 枢纽检测 + authority 惩罚 | 图分析 | ★★★ 高（独创） |
| 5 | 邻接骨架桥检测 | 图分析 | ★★★ 高（独创） |
| 6 | 定向枚举三策略（替代DFS） | 路径发现 | ★★★ 高（vs DFS） |
| 7 | MoDora [T]/[M]/[C] 图节点语义增强 | 语义增强 | ★★ 中（改造） |
| 8 | content_list.json 顺序匹配修复 page_idx | 数据修复 | ★★ 中 |
| 9 | G1+G2 双门禁 | 质量控制 | ★★ 中 |
| 10 | 9项查询QC引擎 | 质量控制 | ★★ 中 |
| A | LaTeX 主文件置信度发现 | 解析 | ★ 低 |
| B | 递归 \input 解析 + 行号溯源 | 解析 | ★★ 中 |
| C | 环境栈 + 标签类型三级推断 | 解析 | ★★ 中 |
| D | 字符距离指数衰减质量评分 | 评分 | ★★★ 高（独创） |
| E | 引用匹配歧义度检测 | 实体对齐 | ★★ 中 |
| F | 引用图度数预算抑制 | 质量控制 | ★ 低 |
| G | 嵌入式表/公式检测 | 元素提取 | ★ 低 |
| H | Gold/Silver/Trash 质量分层 | 质量控制 | ★★ 中 |
