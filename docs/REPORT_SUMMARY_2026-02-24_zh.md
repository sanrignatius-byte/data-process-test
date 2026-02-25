# M4 数据工程进度汇报

**日期**：2026-02-24
**汇报周期**：约两周（2026-02-11 → 2026-02-24）

---

## 先说背景：两周前我们卡在哪里

直接进数字之前，想先交代一下两周前的处境，不然很多决定看起来会没来由。

### L1 的争议

L1 v3 在 2 月 10 号前后跑完，974 条 query，QC 通过率 97.2%。数字看起来很漂亮，但当时 Mentor 和外部评审提了三条批评，我觉得都有道理，没有强辩：

一是**模态偏科很严重**——71% 是 plot，table 只有 0.6%，formula 几乎没有。叫"多模态"其实名实不符。

二是**"多跳"的概念用错了**。我们当时把 dual-evidence 叫 multi-hop，但实际统计 298/300 条的 path length = 2，这不是链式推理，只是两个证据并排放。后来把名字改成 dual-evidence，既是承认问题，也是把任务定义收紧到更可验证的范围。

三是**单元素可答率 45%**——接近一半的 query 只用一侧的证据就能回答，双证据的设计目标完全没实现。

在 Mentor 确认"先深耕 L1"的方向后，我们就把接下来的两周放在这上面了。

### Cross-doc 的暂停

L2 v3 完成后（42 条，19 QC pass），反复分析失败原因发现 anchor leakage 占了 21/23，而 QC 已经做得比较严了。继续在实体倒排索引方案上加规则感觉是越堆越脆。当时决定先停掉 L2，改用 Citation Graph 做跨文档候选来源——这个方案的信号更强，因为引用关系本身代表作者认为两篇论文有关联。

---

## 这两周做了什么

### 基础设施：LaTeX 引用图

formula+table 配对在最初的版本里通过率只有 3.3%，根因是模型拿到两个元素却不知道它们为什么有关联。

这里有一个比较关键的发现：LaTeX 源码里，作者写正文的时候经常会在同一句话里 `\ref{}` 多个元素（比如"Figure 3 和 Table 2 都说明了 X"）。这段上下文本质上是作者亲笔写的"这两个元素为什么在一起"。用 MinerU 的位置邻近法是找不到这个的。

基于这个思路，我们把 LaTeX 源码下载（73 篇）和引用 DAG 构建好了，最终拿到 118 对跨模态元素，每对都附带 bridge_text（作者的原文引用上下文）。过程中发现了两个质量问题并修了：

- **G1**：一些高频被引的元素（比如 1409.0575 的 Table 9）产生了大量虚假对，加了每个元素最多 3 对的上限
- **G2**：纯靠位置邻近配的对里有的 bridge_text 其实只提到了一侧的元素，加了共引门禁

另外把跨文档 Citation Graph 也跑好了，123 条引用边，人工抽查 title 匹配误匹配率 0%。

### Dual-evidence 生成的四轮迭代

这是本轮工作量最大的部分，前后共四轮，每轮都有实质性的 prompt 改动：

**v3（LaTeX bridge 注入）**：把 bridge_text 直接喂给模型，通过率 72/236（30.5%）。主要失败原因是 bridge_entity_leakage（84 条）——模型把 bridge_text 里的实体名直接写进 query，answer 里也出现了，变成"看名字就能猜答案"。

**v4（Conceptual Masking）**：引入实体匿名化，要求模型用功能描述代替名字，同时加了 cross-modal operator 约束和 required_evidence_spans 字段。通过率跳到 139/236（58.9%）。

**v4.1（figure+formula 专项重设计）**：figure+formula 的通过率一直是短板（最初 32%），专门为这个 pair_type 用 opus 重写了 prompt，区分 quantitative figure 和 structural diagram 两种策略。结果 figure+formula 提升到 40.5%，但 anchor_leakage 从 20 条反弹到 39 条——这是个 trade-off，新 prompt 让模型描述视觉细节更细致，词汇和 query 的 overlap 反而上去了。

**v4.2（PhD persona + 句法多样性）**：这轮的出发点是发现上几轮 query 的句法拓扑在坍缩——大量 query 都是 "Which X validates/quantifies Y" 的双子句模板。根因是 persona "rigorous academic reviewer" 在语言风格上会把人往这个模式拉。改成"组会上的 PhD 生"之后，句式自然散开了很多，同时加了 5 种句法结构约束（GIVEN-WHY / WHAT-IF / WHY-INCONSISTENT / WHEN-CONDITION / WHAT-CAUSES）和 verb 黑名单（validate/quantify/justify/demonstrate 等）。最终通过率 152/236（64.4%）。

按 pair_type 看 v4.2 的结果：figure+table 76.0%，figure+formula 45.9%，formula+table 43.8%。figure+formula 还有 5 篇论文完全 0 pass，下面会单独说。

### 官方生产批次

最终在集群上跑的生产批次（`img150_tuned_v4_official`）拿到了 173/222（77.93%）的通过率，是目前 dual-evidence 任务的最高点，相比最初 v1 的 14.3% 大概翻了 4 倍多。

### Triplet 构建

训练格式是（query, positive bundle, negative bundles）。做了 v1 和 v2 两版：

v2 加入了 `same_type_hard_plus` 负例策略和 `text_short` 字段，avg_difficulty 从 0.62 升到 0.73，positive 图像覆盖率 100%。BM25 在 v2 上的性能（global acc@1 0.45）低于 v1（0.55），这是预期结果——负例更难了，词法捷径更少了。

### 跨文档 Embedding 匹配 + Utility-aware Rerank

用本地的 Qwen3-Embedding-4B 对 590 个 source elements 做了 top-20 匹配（共 11800 条记录）。

审计完发现了比较严重的 hub 效应：top10 target 集中度 0.3153，只有 186 个 target 能被选为任意 source 的 top1。这背后有一个比较基础的 objective mismatch——embedding 优化的是"相似"，我们需要的是"多跳有用"，高相似度的跨文档元素往往是平行描述同一件事，反而不是好的下一跳候选。

为了解决这个问题，在 Stage B 加了 Utility-aware Rerank，引入四类惩罚：target hub 惩罚、文档热度惩罚、列表内多样性惩罚、全局 top1 per-target cap。最终平衡版（cap=10）的结果：用约 1.5% 的相似度分数换来 top10 集中度下降 58%，unique top1 targets 从 186 增加到 286，互惠率从 0.71 升到 0.81。

---

## 这轮最想说清楚的一件事

数字之外，有一个认识上的转变我觉得比较值得说：**top-1 平均相似度不是一个好的 KPI**。

我们在这个问题上走了一段弯路——花了不少精力优化 embedding 的 top-1 分，但后来意识到这完全是 objective mismatch。embedding 模型的训练目标是"相似"，但我们需要的是"这个元素能为多跳推理提供新的视角"。两件事在数学上不等价，而且方向有时是相反的：真正有多跳价值的元素往往是在视角、模态、结论上互补的，相似度反而可能比平行描述低。

这个认识确定了之后，当前的评估重心是：unique top1 targets（候选覆盖是否充分）、top10 集中度（hub 效应有多强）、互惠率（候选稳定性）。更重要的 hop_utility 指标还没有——那需要人工标注。

---

## 坦诚说：现在最大的问题是什么

### 最大的问题：评估闭环一直没建

从 2 月 10 号的讨论开始，"30 条人工测试集 + BM25 baseline + Recall@10/MRR"就列为最高优先级了，但截至今天，这件事还没做。

当前所有的迭代依据（pass rate 提升、hub 集中度下降）都是生成侧的代理指标，没有一个数字能回答"训出来的 embedding 对检索有没有帮助"。这是目前最需要解决的问题，也是下周最优先的事情。

### figure+formula 的 hard case

1803.04383 等 5 篇论文的 figure+formula 配对全部 0 pass。根因是 architecture diagram 这种图的 caption 往往只写"Model architecture"，没有语义锚点，而 formula 那侧是复杂的 loss function，两端完全无法用 token overlap 建立关联。这个 case 暂时还没有好的解决思路，先标记在这里。

### 公式类元素没有图片路径

约 12% 的候选记录是公式类 element，没有 image_path。训练时是要跳过、纯文本处理、还是用 LaTeX 渲染？这个 fallback 规则还没定，是架构债。

### All-rank 层面的 hub 问题

Rerank 改善了 top1 的集中度，但在 all-rank 候选池里，热点 target 还是会反复出现在很多 source 的候选列表里（只是不再稳坐 top1 了）。真正解决这个问题需要 Stage C 的全局路径约束，这一段还是空的。

---

## 下一步（按优先级）

**最紧迫**：建立 200 条人工标注集（relevance / hop_utility / redundancy / error_type），这是后续所有消融实验的前提。用当前冻结的 v2b_cap10 候选集做输入。

**然后**：用 123 条引用边替换实体倒排索引做 L2 候选。引用关系信号比实体共现强，leakage 风险也更低。

**再然后**：在 v2b_cap10 基础上加入 reranked cross-doc hard negatives，生成 triplet v3，做最小消融（embedding-only vs +hub/diversity rerank vs +context rerank）。

**遗留**：figure+formula 里 architectural diagram + 复杂 loss function 的 hard case，等有了评估闭环再来判断要不要专项处理。

---

## 关键文件（方便查找）

| 文件 | 说明 |
|------|------|
| `data/l1_dual_evidence_queries_slurm_img150_tuned_v4_official.jsonl` | 生产批次（222 条，173 pass，77.93%） |
| `data/l1_dual_evidence_queries_v3_pass.jsonl` | v4.2 通过集（152 条） |
| `data/l1_dual_evidence_triplets_v2_all.jsonl` | 当前训练用 triplet（222 条，avg_difficulty 0.73） |
| `data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2b_cap10.jsonl` | **冻结的跨文档候选集** |
| `data/latex_cross_modal_pairs.json` | LaTeX 增强跨模态对（118 对，含 bridge_text） |
| `data/citation_graph.json` | 跨文档引用图（123 条引用边） |
| `scripts/rerank_mineru_crossdoc_matches.py` | Utility-aware rerank |
| `scripts/audit_mineru_crossdoc_embedding_matches.py` | 匹配质量审计 |
