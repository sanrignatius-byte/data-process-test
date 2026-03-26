# Document Graph for Document Understanding — 技术方案 v4

> 2026-03-17 | 汇报版 + 专利素材

---

## 术语定义

| 术语 | 定义 |
|------|------|
| **Bridge Hub（桥接枢纽节点）** | 同时引用**至少两种不同模态**元素（figure / table / formula 中的两种及以上）的段落节点。是跨模态信息传播的关键中间节点 |
| **Authority Node（权威节点）** | 被大量其他节点引用（高入度），但自身不桥接不同模态的节点。是信息的目的地而非中转站 |
| **Adjacent Backbone Bridge（相邻骨干桥接）** | 文档阅读顺序中相邻的两个段落节点，分别引用不同模态的元素，且通过 backbone 边连接的桥接结构 |
| **Hub 覆盖集** | 所有 Bridge Hub 及 Adjacent Backbone Bridge 所引用的元素节点的集合 |
| **Hub 覆盖率（hub_overlap）** | 评测集中 ground truth 证据落入 hub 覆盖集的 **query 占比**（分母为总 query 数） |
| **1-hop 邻域** | 与目标节点通过 paragraph_ref / element_ref / backbone 边直接连接的节点集合 |
| **显式引用标记** | 文档中标记元素间引用关系的结构化标注，如 LaTeX `\ref{}`、Word 交叉引用、HTML `<a>` 锚点等 |
| **chunk** | 文档解析后的最小检索单元，对应一个多模态元素（段落 / 图表 / 表格 / 公式）及其上下文 |

---

## 1. 问题与动机

### 1.1 要解决什么问题

**技术问题**：在多模态文档的信息检索中，一个查询所需的完整证据往往分散在**不同模态**的元素中——例如"模型的公平性如何随数据分布变化"，答案的一半在 **Figure 3**（性能曲线趋势）、另一半在 **Table 2**（不同子群的精确数值）。现有基于词面匹配的检索方法仅对单个 chunk 独立打分，未利用文档自身的显式结构信号建立不同模态元素间的关联，导致**跨模态证据的召回率和准确率低**。

本方法要解决的核心技术问题是：**如何在不依赖大语言模型的前提下，利用文档的显式结构信号建立多模态元素间的关联，提升跨模态证据的检索召回率和排序准确率，同时保持低成本可扩展性。**

### 1.2 BM25 为什么不够

BM25 是纯词面匹配。当 query 描述 figure 中的"accuracy curve"，BM25 能通过 figure caption 命中这张图；但与之关联的 Table 2 的 caption 可能写的是"Per-group metrics"——词面完全不同，BM25 定位不到。

**核心矛盾**：BM25 独立打分每个候选 chunk，不建模 chunk 之间的结构关联。Figure 3 和 Table 2 在同一篇论文中被同一段文字共同引用——这个"共引关系"是 BM25 看不到的信号。

### 1.3 核心洞察：文档自带结构信号

多模态文档（尤其是学术论文、技术报告等含显式引用标记的文档）本身包含丰富的**显式结构信号**：

```
论文原文中的一段话（paragraph）：
  "As shown in Figure 3, the accuracy drops significantly
   for minority groups. The exact numbers are in Table 2."
```

这段话里有两个显式引用标记（如 LaTeX `\ref{}`、Word 交叉引用等）——一个指向 Figure 3，一个指向 Table 2。这意味着：
- 这个**段落**是一个**桥梁（Bridge Hub）**：它同时连接了 figure 和 table 两种模态
- 如果 BM25 命中了 Figure 3，我们**沿着这个桥梁段落的引用边**就能找到 Table 2

**这就是本方法的核心思路**：把这些已有的引用关系、段落顺序构建成图，利用图的拓扑结构做检索增强。图构建与 rerank 全过程**零 LLM 调用**——不需要大模型提取任何东西，信号来自文档本身的显式结构。

---

## 2. 核心发明点（三个层次）

### 发明点 1：多层异构文档图（Multi-layer Heterogeneous Document Graph）

**"多层"**：图中有 4 种不同语义的边（阅读顺序 / 段内引用 / 元素间引用 / 跨文档引用），每种边编码不同层次的文档结构关系。

**"异构"**：图中有 4 种节点（paragraph / figure / table / formula），不同类型节点间的边具有不同语义——paragraph→figure 是"这段话讨论了这张图"，paragraph→paragraph（backbone）是"这两段在原文中紧邻"。

**核心主张**：仅靠文档解析器的输出 + 文档内显式引用标记（如 LaTeX `\ref{}`、Word 交叉引用、HTML 锚点等），就能自动构建这样的多层异构图，无需 LLM 参与。

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

**数据验证**：Bridge Hub 60 个（覆盖 31 篇），Adjacent Backbone Bridge 369 条（覆盖 68 篇）。纳入 adjacent bridges 后，hub 覆盖率（评测集中 ground truth 证据落入 hub 覆盖集的 query 占比）从 9.53%（25/261）跃升至 **90.42%**（236/261），是效果提升最大的单一因素。

### 发明点 3：1-hop Neighbor Propagation（邻域标签传播检索增强）

在 BM25 初步检索的基础上，利用图的拓扑关系做 rerank。核心机制用一句话说：

> **BM25 命中了 Figure 3 → 沿图边找到 Figure 3 的邻居 Table 2 → 给 Table 2 加分 → 两个跨模态证据一起浮上来。**

这比"独立打分每个 chunk"多了一步"邻居传播"，但正是这一步让系统能捕获 BM25 看不到的跨模态关联。

**为什么只传 1 跳（1-hop）**：当前图平均度 ~2.7，2-hop 就会扩散到大量弱关联节点，引入噪声反而降低效果。实验验证：1-hop MRR 0.6045 > 2-hop 0.5962。

---

## 3. 图的形式化定义

$$G = (V, E, \tau_V, \tau_E)$$

### 3.1 节点类型 \(\mathcal{T}_V\)

| 类型 | 来源 | 成本 | 在图中的角色 |
|------|------|------|-------------|
| **paragraph** | 文档解析（MinerU / LaTeX） | 零 | 阅读顺序的基本单元；**Bridge Hub 的候选**——只有 paragraph 可以同时引用多种模态元素 |
| **figure** | 文档解析 | 零 | 视觉证据（图片 + caption） |
| **table** | 文档解析 | 零 | 结构化数据证据（HTML 表格 + caption） |
| **formula** | 文档解析 | 零 | 数学模型证据（LaTeX 公式 + 上下文） |

### 3.2 边类型 \(\mathcal{T}_E\)

| 类型 | 构建方式 | 数量 | 编码的语义 |
|------|---------|------|-----------|
| **backbone** | 同文档段落按解析器输出的阅读顺序排序 → \(p_i \to p_{i+1}\) | 1269 | **阅读顺序**：连续段落在原文中紧邻，语义最相关。Adjacent Backbone Bridge 就靠这种边发现 |
| **paragraph_ref** | 段落文本中出现显式引用标记（如 `\ref{label}`）→ 段落→被引元素 | 1688 | **"这段话讨论了这个元素"**：Bridge Hub 就是通过这种边同时指向多种模态 |
| **element_ref** | 两个非 paragraph 元素间的直接显式引用 | 80 | **元素间直接引用**（高置信但稀少） |
| **cross_doc_cite** | 参考文献列表（如 `.bbl`）中的标题与 corpus 内文档做模糊匹配（基于空格分词的 token 集合 Jaccard ≥ 0.55） | 434 | **跨文档引用**：文档 A 引用了文档 B |

当前图规模：**2551 nodes / 3471 edges**，覆盖 82 篇文档。

### 3.3 引用标记与解析器元素的对齐

文档源码中的引用标记（如 LaTeX `\ref{fig:roc}`）指向一个内部 label，但文档解析器输出的是 `figure_3` 这样的编号元素。两者之间需要做对齐匹配：

1. **数字提取**：从引用标记中提取数字（如 `fig:3` → `3`），匹配同文档中 `number=3` 的同类型元素（高置信）
2. **Caption Jaccard fallback**：若数字匹配失败，对引用上下文与候选元素 caption 做基于空格分词的 token 集合 Jaccard 相似度匹配（阈值 0.25）

当前匹配率：**49.8%**（主要瓶颈：解析器编号与源码编号有偏移）。纯 backbone 边不依赖此对齐步骤，因此即使匹配率有限，阅读顺序信号仍可完整保留。

---

## 4. Hub 评分

### 4.1 Bridge Score（桥接分）

$$\text{bridge\_score}(h) = |\text{modalities}(h)| \times 15 + |\text{out\_to\_elements}(h)| \times 2$$

- 模态数乘 15：确保 3-模态 hub（45 分）与 2-模态 hub（30 分）有明确区分
- 元素出度乘 2：引用越多元素的段落桥接能力越强，但不如模态多样性重要

### 4.2 Quality Score（最终评分）

$$\text{quality\_score}(h) = 0.5 \times \hat{s}_{bridge} + 0.25 \times \hat{s}_{pagerank} + 0.25 \times \hat{s}_{out\_elem}$$

\(\hat{s}\) 为 min-max 归一化。分布 [0.13, 0.88]（v2 版本是常量 0.8，无区分度）。

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
    neighbor_boost(c) ← λ × max_{n ∈ N(c)} s_bm25(n)
    其中 N(c) 为 c 在图 G 中通过 paragraph_ref / element_ref / backbone 边
    直接连接的 1-hop 邻域节点集合，λ = 1 - λ_decay
    s(c) ← s(c) + neighbor_boost(c)
    直觉：BM25 命中了 figure → 分数沿图边流向关联的 table → table 排名上升

Step 4 — 输出 reranked top-k
```

### 5.2 参数与选择理据

| 参数 | 最优值 | 为什么 |
|------|--------|--------|
| \(w_{hub}\) | **0.15** | hub prior 是轻微加分，不能喧宾夺主；>0.20 开始反噬 BM25 本身正确的排名 |
| \(w_{nprop}\) | **1.00** | neighbor propagation 是核心信号，2026-03-26 grid search 证明：nw=0.20→1.00 时 graph\_full MRR 从 0.6225 → 0.7234（+16.2%） |
| \(\lambda_{decay}\) | **0.20** | 传播保留 80% 的原始分数；过低（保留太多）→ 噪声扩散；过高（保留太少）→ 传播无效 |
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

1-hop propagation（λ = 1 - 0.20 = 0.80）:
  table_2 的 1-hop 邻域 N(table_2) 包含 figure_3（通过 paragraph_17 桥接）
  neighbor_boost = 0.80 × max(s_bm25(n)) = 0.80 × 0.82 = 0.656
  table_2.score ← 0.31 + 0.656 = 0.966   ← 大幅提升

Reranked top-10: figure_3 和 table_2 都进入 → 跨模态 evidence pair 完整召回 ✅
```

---

## 6. 实验结果

### 6.1 Exp B — 检索增强（261 条 queries，1314 候选 chunks）

| Method | Recall@10 | Δ vs BM25 | MRR | Δ vs BM25 |
|--------|-----------|-----------|-----|-----------|
| BM25 | 0.8467 | — | 0.5642 | — |
| TF-IDF dense | 0.7739 | -0.0728 | 0.4789 | -0.0853 |
| **Graph full** | **0.8736** | **+0.0269** | **0.6045** | **+0.0403** |

MRR **+7.1%** 相对提升，达到 continue\_expand 阈值（+0.03）✅

#### Hub-overlap 子集（236 queries，90.42%）

| Method | Recall@10 | Δ | MRR | Δ |
|--------|-----------|---|-----|---|
| BM25 | 0.8602 | — | 0.5652 | — |
| **Graph full** | **0.8898** | **+0.0296** | **0.6102** | **+0.0450** |

#### Per-query 命中分析

Graph full 拯救 **11 条** BM25 miss 的 queries → **全部是跨模态 dual-evidence 类型**（fig+tab: 5, fig+formula: 4, formula+tab: 2）。损失 4 条。净增 +7。

### 6.2 Exp A — 难度梯度（974 + 210 + 115 条 queries）

用 evidence coverage（全部证据命中比例）验证三级 query 的难度递增：

| Level | n | Evidence Coverage | Recall@10 | MRR |
|-------|---|------------------|-----------|-----|
| L1 (single_element) | 974 | **0.971** | 0.712 | 0.508 |
| L2 (dual_evidence) | 210 | **0.610** | 0.833 | 0.553 |
| L3 (reasoning_chain) | 115 | **0.617** | 0.965 | 0.746 |

- **L1→L2 陡降 -37%**：单模态→跨模态是核心难度分水岭
- L2≈L3 持平：推理链长度差异不足以在 evidence coverage 上拉开差距
- Recall@10 反升是统计假象：多证据 query 有更多"中奖机会"

### 6.3 Exp C — QA 三角印证（Graph 检索→LLM 回答质量）

用 LLM (gpt-5.4) 在 BM25 vs Graph 检索结果上分别做 QA，对比 evidence mention：

| 条件 | L2 检索Δ | L2 QA Δ | L3 检索Δ | L3 QA Δ |
|------|---------|---------|---------|---------|
| raw elements (n=157/89) | +0.96% | +1.91% | **+8.99%** | +2.25% |
| enriched elements (n=210/115) | +1.90% | -0.48% | **+6.09%** | -1.74% |

- **图一致提升检索覆盖**：L3 +6%~9%，推理链越深图的增益越大
- **QA mention 在 enriched 环境下中性**：enrichment 已让 BM25 提供"足够好"的 evidence
- **核心结论**：Graph 的价值在检索层（尤其 raw/规模化场景），QA 层需更好的评估指标

---

## 7. 消融实验

### 7.1 图组件消融

| 消融项 | 结论 |
|--------|------|
| **neighbor_prop 是核心信号** | 独立贡献 MRR +0.0313，占 graph_full 增益的 ~70% |
| **hub_prior 是辅助信号** | 独立仅 +0.0015 MRR，但与 neighbor_prop 协同后总增益达 +0.0403 |
| **citation_walk 为负** | -0.0153 Recall，0 wins / 4 losses → 已关闭（doc-level 粒度与 element-level 需求错位） |
| **1-hop > 2-hop** | 2-hop MRR 0.5962 < 1-hop 0.6045（扩散引入噪声） |
| **hub_overlap 是决定因素** | 9.53%→90.42% hub 覆盖率提升是最大单一增益来源 |

### 7.2 Enrichment 消融（核心发现）

同一 261 条 queries，仅切换 elements 文件（raw vs enriched）：

| 方法 | R@10 (raw) | R@10 (enr) | MRR (raw) | MRR (enr) |
|------|-----------|-----------|----------|----------|
| BM25 | 0.8314 | 0.8467 | 0.5508 | 0.5642 |
| BM25+Graph | 0.8314 | **0.8736** | **0.5685** | **0.6045** |

| 对比 | MRR Δ | 成本 |
|------|-------|------|
| Enrichment alone | +0.0134 | ~$3 LLM |
| **Graph alone** | **+0.0177** | **$0** |
| Both combined | +0.0537 | ~$3 LLM |

**三个关键发现**：
1. **Graph 零成本 MRR +0.018 > Enrichment $3 MRR +0.013**
2. **两者合用超线性**：各自之和 0.031，合用 0.054（×1.73 倍）
3. **规模化路径**：万篇级用图为主（$0），局部高价值元素加 enrichment

### 7.3 Section Enrichment 对比（2026-03-26）

Section-level enrichment（1417 section 节点 LLM 语义总结）注入 query 生成 prompt：

| 方法 | Baseline MRR (n=284) | Section-Enriched MRR (n=329) |
|------|---------------------|------------------------------|
| bm25 | 0.486 | 0.531 |
| neighbor_prop | 0.670 | **0.715** |
| graph_full (hw=0.15,nw=0.20) | 0.575 | 0.623 |

- Section enrichment 提升 BM25 baseline（+0.045 MRR）——更好的词面锚点
- Graph lift 基本持平（ΔMRR ~+0.184），但绝对值随 baseline 上升
- L3 pass 率从 48% → 66%（37 → 80 条），是最大收益

### 7.4 graph\_full 权重调优（2026-03-26）

Grid search on section-enriched queries（329 条），cite\_weight=0 固定：

| hw | nw | R@10 | MRR | ΔMRR vs current |
|----|------|--------|--------|-----------------|
| 0.15 | 0.20 | 0.8602 | 0.6225 | — (旧配置) |
| 0.05 | 1.00 | 0.9058 | 0.7200 | +0.0975 |
| 0.10 | 1.00 | 0.9027 | 0.7211 | +0.0986 |
| **0.15** | **1.00** | **0.9027** | **0.7234** | **+0.1009** |
| 0.00 | 1.00 | 0.9058 | 0.7145 | +0.0920 |

**结论**：nprop\_weight 从 0.20 → 1.00 是最大单一改进（MRR +16.2%）。hub\_weight 保留 0.15 有正贡献（+0.009 vs hw=0）。最优配置 **hw=0.15, nw=1.00, cw=0**。

### 迭代过程

| 版本 | MRR | Δ vs BM25 | 关键变化 |
|------|-----|-----------|---------|
| v1 | 0.5315 | -0.009 | 初始 |
| v2 | 0.5552 | -0.009 | alpha 修复 |
| v3-fix | 0.5939 | +0.030 | quality_score 重建 + hub coverage ×9.5 |
| v3-tuned | 0.6045 | +0.040 | cite_weight=0 |
| **v4-section-tuned** | **0.7234** | **+0.192** | section enrichment + nw=1.00 |

---

## 8. 成本与可扩展性

| 层级 | 内容 | 需要 LLM？ | 可扩展 |
|------|------|-----------|--------|
| **层级 1（零成本）** | 解析 + backbone + 引用边 + hub 识别 + rerank | **否** | **万篇+** |
| 层级 2（中） | Embedding 语义边 / MoDora 元素增强 | 可选 | 千篇 |
| 层级 3（高） | Figure 精分 / Hub 摘要重写 | 是 | 百篇 |

**层级 1 已产生 +0.0403 MRR 增益。** 这意味着图构建与 rerank 环节无需任何 LLM 调用，仅靠文档自身结构信号即可超越 BM25。

---

## 9. vs 现有方法

**主要参考文献**：BM25 [Robertson et al., "The Probabilistic Relevance Framework: BM25 and Beyond", FnTIR 2009]；GraphRAG [Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization", arXiv 2024]；PRF [Rocchio, 1971; Lavrenko & Croft, "Relevance-Based Language Models", SIGIR 2001]。

| 维度 | 本方法 | GraphRAG | PRF | Dense Retrieval |
|------|--------|----------|-----|-----------------|
| 建图成本 | **零 LLM**（图构建+rerank） | 极高（per-doc LLM 实体提取） | 无图 | 无图 |
| 增强对象 | **候选 chunk 的分数**（沿结构边传播） | 社区摘要→全局答案 | **query 本身**（加词扩展） | 无增强 |
| 增强信号来源 | **文档显式结构边** | LLM 提取的语义关系 | 初始 top-k 文档的词频 | 无 |
| 多模态 | **原生支持**（fig/tab/formula 为一等节点） | 仅文本实体 | 无（仅文本） | 仅文本 |
| 跨模态桥接 | **结构化（1-hop prop via bridge hub）** | 实体共现（无模态区分） | 无 | 无 |
| 可扩展 | **万篇+** | 百篇级（token 成本线性增） | 万篇+ | 万篇+ |

**与 PRF 的核心区别**：PRF 扩展的是 query 的表达（在 query 中加入 top-k 文档的高频词），本方法扩展的是**候选 chunk 的分数**（沿文档结构边从高分邻居传播分数到低分邻居）。PRF 无法利用文档内的结构关系，也不区分模态。

**新颖性总结**：
- **信号来源不同**：GraphRAG 靠 LLM 提取语义关系；PRF 靠初始检索结果的词频统计；本方法靠文档已有的显式结构（引用标记、阅读顺序）——零额外 LLM 成本
- **原生多模态**：figure / table / formula 是图中的一等公民节点，不是文本实体的附属
- **桥接机制独特**：Bridge Hub + Adjacent Backbone Bridge 是本方法特有的跨模态中间节点识别方式，利用了学术写作中"先描图再述表"的独有 pattern

---

## 10. 改进方向 & 下一步

| 当前状态 | 改进方向 | 优先级 |
|----------|---------|--------|
| L1=974, L2=210, L3=115（共 1299 条） | **量产 1000 条 production query**（L2+L3 为主），配严格 QC | P0 |
| 仅基于 LaTeX \ref 的显式引用边 | **Embedding 语义边**：用 embedding model 算节点间相似度，阈值以上连边（补充隐式关联） | P0 |
| paragraph_ref / element_ref 边当前基于 LaTeX 引用标记 | 扩展至正则引用模式（"Figure X" / "Table Y"），使方法适用于纯 PDF 文档 | P1 |
| Citation walk 当前为负贡献（doc-level 粒度） | 改进为 element-level cross-doc linking | P1 |
| QA evaluation 用 evidence mention（不够好） | 改进为 answer correctness / completeness 评估 | P1 |

> **注**：backbone 边（阅读顺序）和 Adjacent Backbone Bridge 不依赖任何引用标记格式，仅依赖文档解析器的段落顺序输出，因此天然适用于任何格式的文档。
