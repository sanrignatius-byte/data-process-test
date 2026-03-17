# Document Graph for Document Understanding — 技术方案 v3

> 2026-03-17 | 汇报版

---

## 1. 问题与动机

### 1.1 要解决什么问题

学术论文中，一个研究问题的完整回答往往分散在**多种模态**的元素里——例如"模型的公平性如何随数据分布变化"，答案的一半在 **Figure 3**（性能曲线趋势）、另一半在 **Table 2**（不同子群的精确数值）。我们需要一个检索系统能**同时定位到这两个跨模态证据**。

### 1.2 BM25 为什么不够

BM25 是纯词面匹配。当 query 描述 figure 中的"accuracy curve"，BM25 能通过 figure caption 命中这张图；但与之关联的 Table 2 的 caption 可能写的是"Per-group metrics"——词面完全不同，BM25 定位不到。

**核心矛盾**：BM25 独立打分每个候选 chunk，不建模 chunk 之间的结构关联。Figure 3 和 Table 2 在同一篇论文中被同一段文字共同引用——这个"共引关系"是 BM25 看不到的信号。

### 1.3 核心洞察：文档自带结构信号

学术论文（尤其有 LaTeX 源码的）本身包含丰富的**显式结构信号**：

```
论文原文中的一段话（paragraph）：
  "As shown in Figure 3, the accuracy drops significantly
   for minority groups. The exact numbers are in Table 2."
```

这段话里有两个 `\ref{}`——一个指向 Figure 3，一个指向 Table 2。这意味着：
- 这个**段落**是一个**桥梁（Bridge）**：它同时连接了 figure 和 table 两种模态
- 如果 BM25 命中了 Figure 3，我们**沿着这个桥梁段落的引用边**就能找到 Table 2

**这就是本方法的核心思路**：把这些已有的引用关系、段落顺序构建成图，利用图的拓扑结构做检索增强。整个过程**零 LLM 成本**——不需要大模型提取任何东西，信号来自文档本身。

---

## 2. 核心发明点（三个层次）

### 发明点 1：多层异构文档图（Multi-layer Heterogeneous Document Graph）

**"多层"**：图中有 4 种不同语义的边（阅读顺序 / 段内引用 / 元素间引用 / 跨文档引用），每种边编码不同层次的文档结构关系。

**"异构"**：图中有 4 种节点（paragraph / figure / table / formula），不同类型节点间的边具有不同语义——paragraph→figure 是"这段话讨论了这张图"，paragraph→paragraph（backbone）是"这两段在原文中紧邻"。

**核心主张**：仅靠文档解析器的输出 + LaTeX 显式引用关系，就能自动构建这样的多层异构图，无需 LLM 参与。

### 发明点 2：Bridge Hub 识别（桥接枢纽识别）

不是所有段落都同等重要。我们区分两类高连接度节点：

- **Bridge Hub（桥接型）**：一个段落**同时引用了不同模态的元素**（如同时 `\ref{fig:3}` 和 `\ref{tab:2}`）。这类段落是跨模态信息的"交通枢纽"——查询链路的关键中间站。
- **Authority Node（权威型）**：被大量其他段落引用（如一个核心公式被全文 49 段引用），但自身不桥接不同模态。这类节点是信息的"目的地"，不是"中转站"。

旧方法：按总连接度排序 → authority node 排首位（高入度），但它们无法桥接任何跨模态路径。
**本方法**：桥接优先排序 → bridge hub 强制排在 authority node 之前。

此外我们发现了一种特殊 pattern——**Adjacent Backbone Bridge（相邻骨干桥接）**：

```
原文段落 i  : "Figure 3 shows the performance trend..."     → 引用 figure
原文段落 i+1: "The detailed numbers are listed in Table 2." → 引用 table
```

连续两段各引用不同模态——这在学术写作中极为常见（先说"图上看到什么"，接着说"表里具体数字"）。虽然没有单个段落同时 `\ref{}` 两种模态，但**阅读顺序上的相邻性**同样编码了跨模态关联。

**数据验证**：Bridge Hub 60 个（覆盖 31 篇），Adjacent Backbone Bridge 369 条（覆盖 68 篇）。纳入 adjacent bridges 后，hub 覆盖率从 9.53% 跃升至 **90.42%**，是效果提升最大的单一因素。

### 发明点 3：1-hop Neighbor Propagation（邻域标签传播检索增强）

在 BM25 初步检索的基础上，利用图的拓扑关系做 rerank。核心机制用一句话说：

> **BM25 命中了 Figure 3 → 沿图边找到 Figure 3 的邻居 Table 2 → 给 Table 2 加分 → 两个跨模态证据一起浮上来。**

这比"独立打分每个 chunk"多了一步"邻居传播"，但正是这一步让系统能捕获 BM25 看不到的跨模态关联。

**为什么只传 1 跳（1-hop）**：当前图平均度 ~2.7，2-hop 就会扩散到大量弱关联节点，引入噪声反而降低效果。实验验证：1-hop MRR 0.6045 > 2-hop 0.5962。

---

## 3. 图的形式化定义

$$G = (V, E, \tau_V, \tau_E)$$

### 3.1 节点类型 $\mathcal{T}_V$

| 类型 | 来源 | 成本 | 在图中的角色 |
|------|------|------|-------------|
| **paragraph** | 文档解析（MinerU / LaTeX） | 零 | 阅读顺序的基本单元；**Bridge Hub 的候选**——只有 paragraph 可以同时引用多种模态元素 |
| **figure** | 文档解析 | 零 | 视觉证据（图片 + caption） |
| **table** | 文档解析 | 零 | 结构化数据证据（HTML 表格 + caption） |
| **formula** | 文档解析 | 零 | 数学模型证据（LaTeX 公式 + 上下文） |

### 3.2 边类型 $\mathcal{T}_E$

| 类型 | 构建方式 | 数量 | 编码的语义 |
|------|---------|------|-----------|
| **backbone** | 同文档段落按行号排序 → $p_i \to p_{i+1}$ | 1269 | **阅读顺序**：连续段落在原文中紧邻，语义最相关。Adjacent Backbone Bridge 就靠这种边发现 |
| **paragraph_ref** | 段落文本中出现 `\ref{label}` → 段落→被引元素 | 1688 | **"这段话讨论了这个元素"**：Bridge Hub 就是通过这种边同时指向多种模态 |
| **element_ref** | 两个非 paragraph 元素间的直接 `\ref{}` | 80 | **元素间直接引用**（高置信但稀少） |
| **cross_doc_cite** | `.bbl` 参考文献标题与 corpus 内文档做模糊匹配（Jaccard ≥ 0.55） | 434 | **跨文档学术引用**：论文 A 引用了论文 B |

当前图规模：**2551 nodes / 3471 edges**，覆盖 82 篇文档。

### 3.3 Label 匹配（LaTeX label → 解析器元素）

LaTeX 中写 `\ref{fig:roc}` 指的是一个 label，但文档解析器输出的是 `figure_3` 这样的编号元素。两者之间需要做匹配：

1. **数字提取**：`fig:3` → 找同文档中 `number=3` 的 figure（高置信）
2. **Caption Jaccard fallback**：若数字匹配失败，用 caption 文本的 token overlap（阈值 0.25）

当前匹配率：**49.8%**（主要瓶颈：MinerU 编号与 LaTeX 编号有偏移）。

---

## 4. Hub 评分

### 4.1 Bridge Score（桥接分）

$$\text{bridge\_score}(h) = |\text{modalities}(h)| \times 15 + |\text{out\_to\_elements}(h)| \times 2$$

- 模态数乘 15：确保 3-模态 hub（45 分）与 2-模态 hub（30 分）有明确区分
- 元素出度乘 2：引用越多元素的段落桥接能力越强，但不如模态多样性重要

### 4.2 Quality Score（最终评分）

$$\text{quality\_score}(h) = 0.5 \times \hat{s}_{bridge} + 0.25 \times \hat{s}_{pagerank} + 0.25 \times \hat{s}_{out\_elem}$$

$\hat{s}$ 为 min-max 归一化。分布 [0.13, 0.88]（v2 版本是常量 0.8，无区分度）。

**排序规则**：`sort_key = (is_bridge, bridge_score, quality_score)` — 桥接类**强制排在**权威类之前，不论权威类的总分多高。

---

## 5. 检索增强算法

### 5.1 算法流程

```
输入：query q, 候选库 C, 文档图 G

Step 1 — BM25 初步打分
    对每个候选 c ∈ C 计算 s_bm25(c)

Step 2 — Hub Prior（静态先验）
    若 c 对应的元素在 hub 覆盖集中：
        s(c) ← s_bm25(c) × (1 + w_hub × quality_score(hub))
    直觉：hub 邻域内的元素更可能是有价值的跨模态证据，给予微量加分

Step 3 — 1-hop Neighbor Propagation（动态传播）
    对 c 的每个图邻居 n（通过 paragraph_ref / element_ref / backbone 相连）：
        prop ← max(s_bm25(n)) × (1 - λ_decay)
        s(c) ← max(s(c), s(c) + prop)
    直觉：BM25 命中了 figure → 分数沿图边流向关联的 table → table 排名上升

Step 4 — 输出 reranked top-k
```

### 5.2 参数与选择理据

| 参数 | 最优值 | 为什么 |
|------|--------|--------|
| $w_{hub}$ | **0.15** | hub prior 是轻微加分，不能喧宾夺主；>0.20 开始反噬 BM25 本身正确的排名 |
| $\lambda_{decay}$ | **0.20** | 传播保留 80% 的原始分数；过低（保留太多）→ 噪声扩散；过高（保留太少）→ 传播无效 |
| cite\_weight | **0**（关闭） | citation walk 实验为负贡献（doc-level 粒度 vs element-level 需求错位） |
| neighbor\_hops | **1** | 1-hop 严格优于 2-hop；当前图密度下 2-hop 扩散太多弱关联节点 |

### 5.3 一个具体的端到端例子

```
Query: "How does the fairness-accuracy tradeoff change across different demographic groups?"

BM25 打分:
  figure_3 (caption: "Accuracy vs fairness tradeoff curves")  → score = 0.82 ✓ 命中
  table_2  (caption: "Per-group metrics breakdown")            → score = 0.31 ✗ 词面不匹配

图中的结构:
  paragraph_17 —paragraph_ref→ figure_3
  paragraph_17 —paragraph_ref→ table_2
  (paragraph_17 是 bridge hub：同时引用了 figure 和 table)

1-hop propagation:
  figure_3 的邻居包含 table_2（通过 paragraph_17 桥接）
  table_2.score ← 0.31 + 0.82 × 0.80 = 0.966   ← 大幅提升

Reranked top-10: figure_3 和 table_2 都进入 → 跨模态 evidence pair 完整召回 ✅
```

---

## 6. 实验结果

### 主结果（261 条 dual-evidence queries，1314 候选 chunks）

| Method | Recall@10 | Δ vs BM25 | MRR | Δ vs BM25 |
|--------|-----------|-----------|-----|-----------|
| BM25 | 0.8467 | — | 0.5642 | — |
| TF-IDF dense | 0.7739 | -0.0728 | 0.4789 | -0.0853 |
| **Graph full** | **0.8736** | **+0.0269** | **0.6045** | **+0.0403** |

MRR **+7.1%** 相对提升，达到 continue\_expand 阈值（+0.03）✅

### Hub-overlap 子集（236 queries，90.42%）

| Method | Recall@10 | Δ | MRR | Δ |
|--------|-----------|---|-----|---|
| BM25 | 0.8602 | — | 0.5652 | — |
| **Graph full** | **0.8898** | **+0.0296** | **0.6102** | **+0.0450** |

### Per-query 命中分析

Graph full 拯救 **11 条** BM25 miss 的 queries → **全部是跨模态 dual-evidence 类型**（fig+tab: 5, fig+formula: 4, formula+tab: 2）。损失 4 条。净增 +7。

---

## 7. 消融实验（关键结论）

| 消融项 | 结论 |
|--------|------|
| **neighbor_prop 是核心信号** | 独立贡献 MRR +0.0313，占 graph_full 增益的 ~70% |
| **hub_prior 是辅助信号** | 独立仅 +0.0015 MRR，但与 neighbor_prop 协同后总增益达 +0.0403 |
| **citation_walk 为负** | -0.0153 Recall，0 wins / 4 losses → 已关闭（doc-level 粒度与 element-level 需求错位） |
| **1-hop > 2-hop** | 2-hop MRR 0.5962 < 1-hop 0.6045（扩散引入噪声） |
| **hub_overlap 是决定因素** | 9.53% → 90.42%（纳入 adjacent backbone bridges）是最大单一增益来源 |

### 迭代过程

| 版本 | MRR | Δ vs BM25 | 关键变化 |
|------|-----|-----------|---------|
| v1 | 0.5315 | -0.009 | 初始 |
| v2 | 0.5552 | -0.009 | alpha 修复 |
| v3-fix | 0.5939 | +0.030 | quality_score 重建 + hub coverage ×9.5 |
| **v3-tuned** | **0.6045** | **+0.040** | cite_weight=0 |

---

## 8. 成本与可扩展性

| 层级 | 内容 | 需要 LLM？ | 可扩展 |
|------|------|-----------|--------|
| **层级 1（零成本）** | 解析 + backbone + 引用边 + hub 识别 + rerank | **否** | **万篇+** |
| 层级 2（中） | Embedding 语义边 / MoDora 元素增强 | 可选 | 千篇 |
| 层级 3（高） | Figure 精分 / Hub 摘要重写 | 是 | 百篇 |

**层级 1 已产生 +0.0403 MRR 增益。** 这意味着无需任何 LLM 调用，仅靠文档自身结构信号即可显著超越 BM25。

---

## 9. vs 现有方法

| 维度 | 本方法 | GraphRAG | Dense Retrieval |
|------|--------|----------|-----------------|
| 建图成本 | **零 LLM** | 极高（per-doc LLM 实体提取） | 无图 |
| 多模态 | **原生支持**（fig/tab/formula 为一等节点） | 仅文本实体 | 仅文本 |
| 跨模态桥接 | **结构化（1-hop prop via bridge hub）** | 实体共现（无模态区分） | 无 |
| 可扩展 | **万篇+** | 百篇级（token 成本线性增） | 万篇+ |

**新颖性总结**：
- **信号来源不同**：GraphRAG 靠 LLM 提取语义关系；本方法靠文档已有的显式结构（`\ref{}`、阅读顺序）——成本低 2-3 个数量级
- **原生多模态**：figure / table / formula 是图中的一等公民节点，不是文本实体的附属
- **桥接机制独特**：Bridge Hub + Adjacent Backbone Bridge 是本方法特有的跨模态中间节点识别方式

---

## 10. 已知局限 & 下一步

| 局限 | 影响 | 下一步 |
|------|------|--------|
| 评测集由本系统辅助生成 | 循环评估风险 | P0：引入外部标注 + 统计显著性检验 |
| 86 篇 / 261 queries 规模有限 | 泛化性待检验 | P0：扩到 500+ queries（real-user + persona） |
| 3/4 边类型依赖 LaTeX | 纯 PDF 适用性受限 | P1：backbone + 正则引用模式替代 |
| Citation walk 为负 | 跨文档信号未被利用 | P1：element-level cross-doc linking |
| 35/82 篇零候选 | 覆盖不均匀 | P1：降 cap / adj_bridge 单独生成 |
