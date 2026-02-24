# M4 数据工程进度汇报

**日期**：2026-02-24
**汇报周期**：约两周（2026-02-11 → 2026-02-24）
**汇报范围**：Dual-evidence L1 生成迭代 → Triplet 构建 → 跨文档 Embedding 匹配 → Utility-aware Rerank

---

## 0. 两周前的起点：问题在哪里

在进入本轮进展之前，先明确我们两周前面对的核心困境，否则无法理解这两周做的事情的意义。

### 0.1 L1 的状态与争议

L1（文档内跨模态 query）的 v3 版本在 2026-02-10 前后完成，产出 974 条 query，QC 通过率 97.2%。但 Mentor 和外部评审对这批数据提出了三条有实质意义的质疑：

1. **模态严重偏科**：plot（实验图）占 71.3%，table 仅 0.6%，formula 几乎为零。"多模态"名实不符。
2. **"多跳"名不副实**：我们当时叫 multi-hop 的数据，经统计发现 298/300 条 path length = 2，本质是"双证据并行查找"而非链式推理，概念需要厘清。
3. **单元素可答（45%）**：相当比例的 query 实际上只用一侧证据就能回答，双证据设计形同虚设。

> **本轮最重要的一个决定**：在 Mentor 确认"先深耕 L1"方向后，我们接受了这些批评，把 multi-hop 改名为 **dual-evidence**，并设计了三轮 prompt/QC 迭代，目标是真正的双证据必要性。

### 0.2 Cross-doc 的状态与停滞

L2 跨文档 query 在 v3 完成后（42 条，19 QC pass），暴露了一个结构性问题：anchor leakage 是主要失败原因（21/23 fail），但 L2 v3 已经把 QC 做得很严了。继续在实体倒排索引方案上堆 QC 是错误方向。

> **另一个关键决定**：L2 暂停，转用 **Citation Graph** 替代实体倒排索引作为跨文档候选来源。

---

## 1. 本轮完成的工作

### 1.1 LaTeX 引用基础设施（Week 1）

**动机**：解决 formula+table 配对 pass rate 仅 3.3% 的根因——模型不知道两个元素为什么有关联。

**核心洞察**：LaTeX 源码里，一段话经常同时 `\ref{}` 多个元素（figure/table/equation），这段上下文（LatexRefEdge.context）就是作者亲笔写的"为什么这两个元素相关"的解释。这比 MinerU 位置邻近法有本质提升。

**执行结果**：
- LaTeX 源码下载：73/76 篇 .tex，65 篇 .bbl
- 文档内引用 DAG：2021 labels，7423 refs，3019 edges（集群版本更新后）
- **跨文档 Citation Graph**：**123 条**引用边，55 篇最大连通分量
- 人工质检：title_fuzzy 匹配**误匹配率 0%**，Jaccard ≥ 0.55 阈值在 fairness 语料可信

**Step 0 v3.2（LaTeX 跨模态 pair 构建）**：
- 输入：MinerU 的 1316 个多模态元素 + LaTeX 引用图
- 发现两个质量问题并修复：
  - **G1（hub 去重）**：单个高频被引 element 产生 O(N) 虚假对，限制每 element ≤ 3 pairs
  - **G2（共引门禁）**：proximity 配对缺语义门禁，要求 bridge_text 中必须有两端 `\ref{}` 的共现证据
- 最终输出：**118 对**（proximity: 105 + direct: 13），gold: 6 + silver: 112

### 1.2 Dual-evidence 生成四轮迭代（Week 1–2）

这是本轮工作量最大的部分，共经历了四轮有实质差异的 prompt/QC 迭代。

**输入统一**：118 对 latex_cross_modal_pairs（figure+table / figure+formula / formula+table）

| 版本 | 核心改动 | 通过率 | 费用 | 主要失败原因 |
|------|----------|--------|------|--------------|
| v3（LaTeX bridge 注入） | 在 prompt 中注入作者原文 bridge_text | 72/236（30.5%） | $1.66 | bridge_entity_leakage: 84, single_element_answer: 63 |
| v4（Conceptual Masking） | 实体匿名化；新增 cross-modal operator 约束；required_evidence_spans 字段 | 139/236（58.9%） | $2.07 | single_element_answer: 60, anchor_leakage: 20 |
| v4.1（opus figure+formula 重设计） | 区分 quantitative / structural figure 两种策略；双 field 强制解耦；operator 扩展 14 个 | 138/236（58.5%） | $2.39 | single_element_answer: 62, anchor_leakage: 39↑（回归） |
| **v4.2（PhD persona + 句法多样性）** | Persona 从"学术评审"改为"组会 PhD 生"；动词黑名单；5 种句法结构约束；CROSS_MODAL_OPERATORS 扩展自然词 | **152/236（64.4%）** | $2.57 | single_element_answer: 57, anchor_leakage: 29↓ |

**v4.1 的一个意外发现**：重新设计 figure+formula prompt 后，anchor_leakage 从 20 条**回升到 39 条**。根因是新 prompt 生成了更细致的 visual_anchor 描述，词汇与 query 重叠上升。这是 prompt 工程里常见的 trade-off：提升一个维度，另一个维度反弹。解决方案在 v4.2 中通过 persona 更换间接缓解。

**v4.2 的关键洞察**：原 persona "rigorous academic reviewer" 导致所有 query 退化为 `Which X validates/quantifies Y...` 双子句模板（句法拓扑坍缩），训练集会产生 dataset artifact。改为 PhD 组会场景后，句式自然多样化，同时通过黑名单动词（validate/quantify/justify/demonstrate 等）强制阻止学术腔。

**按 pair_type 分布（v4.2）**：
- figure+table：111/146（**76.0%**）
- figure+formula：34/74（45.9%）
- formula+table：7/16（43.8%）

figure+formula 从最初的 32.4% 提升到 45.9%，但仍有 5 篇论文（含 1803.04383）全部失败，architecture diagram + 复杂 loss function 是剩余的 hard case。

### 1.3 官方批次：集群 slurm 生产版

本轮最终的"生产"批次不是 v4.2，而是在集群上用专门调优参数（`img150_tuned_v4_official`）跑出的：

| 指标 | 结果 |
|------|------|
| 总量 | 222 条 |
| QC pass | 173 条（77.93%） |
| figure+table | 144 |
| figure+formula | 62 |
| formula+table | 16 |

**77.93%** 是本项目 dual-evidence 任务的最高通过率，相比最初 v1 的 14.3%，提升约 4 倍。

### 1.4 Triplet 构建（v1 + v2）

Triplet 是对比学习训练数据的最终格式，格式为（query, positive bundle, negative bundles）。

| 版本 | 负例策略 | avg difficulty | positive 图像覆盖 |
|------|----------|---------------|-----------------|
| v1 | in_doc_swap + same_type_hard | 0.6248 | - |
| **v2** | in_doc_swap + same_type_hard_plus + text_short | **0.7288** | **100%** |

v2 负例更难（difficulty 0.72 vs 0.62），体现在 BM25 基线压力测试上：

| 版本 | local acc@1 | global acc@1 |
|------|-------------|--------------|
| v1 pass | 0.8092 | 0.5549 |
| v2 pass 全文 | 0.7514 | 0.4451 |

BM25 性能下降是预期结果：harder negatives = 更少的词法捷径。

### 1.5 跨文档 Embedding 匹配 + Utility-aware Rerank

**Stage A（候选召回）**：
- 模型：本地 Qwen3-Embedding-4B
- 规模：590 source elements × top-k=20 = 11800 条匹配
- 约束验证：同文档违规 0，类型不一致 0

**匹配质量基线审计**（存在的问题）：
- top10 target 集中度 = **0.3153**（hub 效应强，30% 的候选堆在 10 个热点目标上）
- unique top1 targets = **186**（仅 186 个目标被选为任意 source 的 top1）
- suspicious candidates = **241**

**Stage B（Utility-aware Rerank）**：

核心设计理念：embedding 优化的是"相似度"，我们需要的是"多跳有用性"。这两者在数学上不等价（详见 §2）。Rerank 引入四类惩罚信号：
- 目标 hub 惩罚
- 目标文档热度惩罚
- 列表内多样性惩罚
- 全局 top1 per-target cap

| 配置 | top1 mean | top10 集中度 | unique top1 targets | suspicious |
|------|-----------|------------|---------------------|------------|
| 基线（无 rerank） | 0.8822 | 0.3153 | 186 | 241 |
| 严格版（cap=8） | 0.8635 | 0.1271 | 275 | 140 |
| **平衡版（cap=10，推荐）** | **0.8690** | **0.1305** | **286** | **146** |

平衡版用约 1.5% 的相似度分数，换来：top10 集中度降低 58%，唯一 top1 目标数增加 54%，互惠率从 0.7051 升至 0.8119。

---

## 2. 本轮最重要的方法论认识

这一节是对外汇报中最值得讲清楚的部分，因为它不是一个数字，而是一个**方向性的认识转变**。

### 2.1 Objective Mismatch：相似 ≠ 有用

Embedding 模型在训练时优化的是"query 与文档相似"，但我们的任务需要的是"这个跨文档 element 能提供下一跳新增的推理证据"。这两个目标在数学上是不等价的：

- 高相似度的跨文档元素，往往是语义重复的平行描述（讲的是同一件事）
- 真正有多跳价值的 element，往往是在视角/模态/结论上**互补**的，相似度反而可能不是最高的

这个认识解释了为什么 hub 效应会出现：模型把所有人都映射到几个"语义中心"元素上，但这些中心元素对大多数 source 而言不是最好的下一跳候选。

### 2.2 三段式产线架构

基于上述认识，我们确定了跨文档候选生成的三段架构：

```
Stage A：候选召回（高 recall 为主，不过分追求 precision）
     ↓
Stage B：Utility-aware Rerank（hub 抑制 + 多样性 + 去冗余）
     ↓
Stage C：构链约束 + Answerability 检验（还未实现）
```

当前 Stage A + B 已落地。Stage C 是下一步的核心缺口。

### 2.3 主 KPI 的转变

从本轮开始，我们不再以 **top-1 平均分** 作为主 KPI，而是引入：

| 指标 | 意义 |
|------|------|
| unique top1 targets | 候选池覆盖是否充分，不过度集中 |
| top10 target 集中度 | hub 效应强弱 |
| top1 reciprocal rate | 候选的"稳定性"（A→B 同时 B→A 为 top1） |
| hop_utility（下一步） | 候选 element 是否能提供新增推理信息 |

---

## 3. 当前存在的主要问题

诚实评估当前数据的缺陷，比列好看的数字更重要。

### 3.1 最大缺口：没有评估闭环

从 2026-02-10 的讨论起，我们就把"30 条人工测试集 + BM25 baseline + Recall@10/MRR"列为最高优先级，但截至本次汇报，**评估闭环仍未建立**。

当前所有的迭代判断（通过率提升、hub 集中度下降）都是**生成侧指标**，没有一条数字能回答"这些数据训出来的 embedding 比 BM25 检索效果好不好"。这是最需要向 Mentor 坦承的问题。

### 3.2 没有 hop_utility 基准

我们现在用的是 embedding 相似度（已知有 objective mismatch），但连一个小规模的人工标注集来验证"哪些候选真的对多跳推理有用"都没有。100–300 条标注集（relevance / hop_utility / redundancy / error_type）是最紧迫的数据工程任务。

### 3.3 Formula+figure 的 hard case 未解决

1803.04383 等 5 篇论文的 figure+formula 配对全部失败，根因是 architecture diagram（结构图）与复杂 loss function 的 token overlap proxy 失效——answer 用数学术语，但 figure caption 只写"Model architecture"，overlap 为零。单独为 architectural diagram 设计处理策略是遗留工作。

### 3.4 All-rank 层面仍有热点目标

Rerank 改善了 top1 的 hub 效应，但 all-rank 候选池中，热点目标仍然出现在大量 source 的候选列表中（只是不再稳坐 top1）。Stage C 的全局路径约束是真正解决这个问题的手段。

### 3.5 无图路径问题

公式类 element 没有 image_path，约 12% 的候选记录存在来源缺失。训练时的 fallback 规则（纯文本 vs 跳过 vs 用 LaTeX 渲染图）还没有明确定义。

---

## 4. 下一步（优先级排序）

| 优先级 | 任务 | 依赖 |
|--------|------|------|
| **P0** | 建立 200 条人工标注集（relevance / hop_utility / redundancy / error_type） | 当前 v2b_cap10 候选集 |
| **P0.1** | Citation-based L2 候选替换：用 123 条引用边构建有方向性的跨文档 query 候选，替代实体倒排索引 | Citation Graph（已就绪） |
| **P1** | 生成 triplet v3：保留 in_doc_swap，加入 reranked cross-doc hard negatives | v2b_cap10（已冻结） |
| **P2** | 最小消融实验：embedding-only vs +hub/diversity vs +context rerank | 标注集（P0 完成后） |
| **P3** | figure+formula hard case 专项：1803.04383 等 architectural diagram + loss function 场景 | - |

**当前冻结文件**：`data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2b_cap10.jsonl`
作为下游所有工作的默认跨文档候选输入。

---

## 5. 关键文件索引

| 文件 | 说明 |
|------|------|
| `data/l1_dual_evidence_queries_slurm_img150_tuned_v4_official.jsonl` | 官方生产批次（222 条，173 pass） |
| `data/l1_dual_evidence_queries_v3_pass.jsonl` | v4.2 通过集（152 条） |
| `data/l1_dual_evidence_triplets_v2_all.jsonl` | 当前训练用 triplet（222 条，avg_difficulty 0.7288） |
| `data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2b_cap10.jsonl` | **冻结的跨文档候选集（推荐使用）** |
| `data/latex_cross_modal_pairs.json` | LaTeX 增强跨模态对（118 对，含 bridge_text） |
| `data/citation_graph.json` | 跨文档引用图（123 条引用边，55 篇连通分量） |
| `scripts/rerank_mineru_crossdoc_matches.py` | Utility-aware rerank 脚本 |
| `scripts/audit_mineru_crossdoc_embedding_matches.py` | 匹配质量审计脚本 |
