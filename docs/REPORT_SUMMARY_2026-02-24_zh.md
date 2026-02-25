# M4 数据工程进度汇报

**日期**：2026-02-24
**汇报周期**：约两周（2026-02-11 → 2026-02-24）

---

## 对上次 Mentor 建议的执行情况

上次汇报结束时，Mentor 给了三条方向性建议，这里先逐条交代进展，再展开细节。

**① 丰富模态：引入 table/formula/figure，并尝试对各模态做细分**

这条做了一半。table 和 formula 已经进入了 dual-evidence 的正式 pipeline——当前生产批次的 pair_type 包含 figure+table、figure+formula、formula+table 三种组合，不再是 L1 v3 那样清一色的图文问题。通过率也有明显差异：figure+table 76%，figure+formula 46%，formula+table 44%，说明三种模态组合的难度结构很不一样，值得分开处理。

但**细分**那一层（模型图 vs 实验结果图 vs Chart vs 信息汇总图）还没做。原因是当时判断先把 dual-evidence pass rate 做上来比细分优先，但这确实是个欠账，后面回来补。

**② 构建文档内部 links/structure，自然实现多跳——①LaTeX 引用图，②MinerU 路线**

LaTeX 路线全部落地了，包括文档内引用 DAG、跨模态 pair 构建（带 bridge_text），以及跨文档 Citation Graph。细节在正文里展开。

MinerU 路线当时判断"较难"之后就搁置了，这两周没有推进，还是在观望。思路上可行（从 MinerU 解析出的 section/caption/ref 结构直接建图），但 MinerU 输出的编号和 LaTeX 编号经常对不上（已知 label 匹配率只有 28.8%），在 LaTeX 路线还没充分验证之前先不动这条。

**③ 展望：用 embedding 在隐空间探索跨文档文本相似性**

这条有推进，但遇到了一个比较根本的问题。我们用 Qwen3-Embedding-4B 跑了 590 个 source elements 的 top-20 跨文档匹配，审计结果发现：embedding 找到的"最相似"跨文档 element，往往是平行描述同一件事的元素，而不是有多跳价值的元素——相似度高 ≠ 多跳有用。这不是 Qwen3-4B 的问题，是 objective mismatch。

具体怎么应对，在正文的"这轮最想说清楚的一件事"里有讨论。

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

## 各版本代表性 Query 示例

五个阶段各取一条，从简单到复杂展示 query 质量的演进。

---

### ① L1 v3（单图文，974 条，QC 97.2%）

> **图**：Fair PCA 在 LFW 数据集上的重建误差折线图（Figure 3）
> **类型**：`anomaly_cause`

**Query：**
> Why do all curves show steeper descent between 2.5 and 7.5 features compared to the gradual decline after 10 features, given that k groups exist?

**Answer：**
> With more than two groups, optimal solutions may not assign identical loss initially, but as features increase beyond k groups, the fairness constraints stabilize and reconstruction error improvements plateau, explaining the transition from steep to gradual decline.

**特征**：只依赖单张图，视觉锚点明确（x=2.5-7.5 段的斜率），答案需要把视觉观察和公平约束理论挂钩。这是 L1 最干净的状态。

---

### ② L1 multihop v3（LaTeX bridge 注入，72 pass，30.5%）

> **元素 A**：Formula 1（equalized odds 优化目标）
> **元素 B**：Figure 4（graphical model DAG）
> **类型**：`theory_vs_experiment` | 论文：1610.02413

**Query：**
> Why does the optimal predictor construction require both the regression score and the sensitive demographic attribute in the dependency structure?

**Answer：**
> The graphical model shows that the sensitive attribute directly influences the outcome and connects to the regression score, while the optimization framework demonstrates that deriving optimal equalized odds predictors requires combining the Bayes optimal regressor with the protected attribute. This structural dependency explains why fairness-constrained prediction cannot ignore demographic information even when seeking unbiased decisions.

**特征**：这是 bridge evidence 首次注入的版本——模型拿到了 LaTeX 原文中"An optimal equalized odds predictor can be derived from the Bayes optimal regressor R and the protected attribute A"这句话作为桥接。但句式仍然是偏学术的 "requires both X and Y" 模式。

---

### ③ L1 dual-evidence v4.2（PhD persona + 句法多样性，152 pass，64.4%）

> **元素 A**：Formula 1（equalized odds 优化目标，同上）
> **元素 B**：Figure 4（graphical model DAG，同上）
> **论文**：1610.02413——**与②完全相同的元素对**

**Query：**
> Given that the protected attribute directly connects to the outcome through the regressor node, why does the optimal predictor require access to both the protected attribute and the regressor?

**Answer：**
> The graphical model shows that the protected attribute A has a direct path to the regressor R, which then influences the outcome Y, creating conditional dependencies. The optimization formulation minimizes expected loss over the true outcome and predictor, subject to group-specific constraints that must be evaluated separately for each protected group. This necessitates conditioning on the protected attribute to compute group-conditional error rates (false positives and true positives) that define the fairness constraints. Therefore, the optimal predictor must access both the regressor (for accuracy) and the protected attribute (for fairness evaluation) because the constraints operate on conditional distributions that cannot be derived from the regressor alone.

**特征**：相同元素对，改用 GIVEN-WHY 句法结构后，query 从"要求式陈述"变成了"条件触发的因果追问"，更像研究生在组会上真实提问的方式。anchor_leakage 降为 0（`anchor_leak_jaccard: 0.0`）。

---

### ④ 官方生产批次（77.93%，最高通过率）

> **元素 A**：Figure 1（Google Translate 把匈牙利语职业句子翻成英文的截图）
> **元素 B**：Table 3（各语言职业句子模板，含匈牙利语 `ő egy ⟨occupation⟩`）
> **pair_type**：`figure+table` | 论文：1809.02208

**Query：**
> Why does translating occupation sentences using the single-pronoun structure from gender-neutral languages produce systematically gendered outputs for traditionally stereotyped professions?

**Answer：**
> The Hungarian template 'ő egy ⟨occupation⟩' uses a gender-neutral pronoun structure, but when translated to English, the system must assign gendered pronouns (he/she). This forces the translation algorithm to infer gender based on occupational stereotypes, resulting in nurse and baker being translated with 'she' while CEO is translated with 'he', consistent with traditionally male-dominated versus female-dominated fields.

**特征**：figure（截图，视觉证据：she's a nurse / he's a CEO）和 table（模板结构）两侧证据各司其职——去掉任何一侧都无法完整回答。`required_evidence_spans` 标注了双侧各自的具体 span。

---

### ⑤ L2 v2（跨文档，16 pass）

> **文档 A**：1511.00830（VFAE 原论文，Figure 13：t-SNE 可视化）
> **文档 B**：1805.09458（后续工作，Figure 1：adversarial loss 柱状图）
> **类型**：`cross_synthesis`

**Query：**
> How does the VFAE adversarial loss of approximately 0.75 on the Adult dataset bar chart compare to the gender-factoring success visible in the red-blue overlapping t-SNE clusters when VFAE uses MMD regularization?

**Answer：**
> The VFAE adversarial loss of ~0.75 on Adult (0-layer configuration) reflects moderate adversary accuracy on the protected attribute, while the t-SNE visualization with MMD shows heavily overlapping red (female) and blue (male) clusters where linear and non-linear accuracy approaches random chance. Both metrics confirm successful gender-information factoring: the bar chart quantifies adversarial confusion at 0.75, and the t-SNE overlap demonstrates indistinguishability of gender representations in the latent space.

**特征**：这是当前唯一跨文档的样例——两张图来自两篇发表时间不同的论文，answer 需要同时引用两侧数值（0.75 和 t-SNE 聚类重叠）。代表了我们最终希望大量生产的 query 类型，也是目前产量最少、质量最难控的地方。

---

### ⑥ Triplet v2（训练格式，222 条，avg_difficulty 0.73）

Triplet 是最终喂给 embedding 训练的格式，结构是 `(query, positive_bundle, [neg₁, neg₂])`。正例是 dual-evidence 双元素包，负例有两种策略。下面用同一条 query 完整展示结构。

> **来源 query**：`l1_de_1409.0575_0000` | pair_type: `figure+table` | difficulty: 0.61

**Query：**
> Why does the hierarchical query progression from general to specific categories enable the human annotator to outperform the automated classifier despite evaluating many images?

**✅ Positive bundle（正例，双元素）**

| | 元素 |
|---|---|
| Evidence unit 1 | **Figure 6**：动态查询算法示意图（绿=正标注，红=负标注，展示 cat 类别在一批图上的逐步推进过程） |
| Evidence unit 2 | **Table 14**：A1 与 A2 人工标注 top-5 误差对比（GoogLeNet 6.8% vs A1 5.1% vs A2 12.0%） |
| bridge_evidence | "With a sufficient amount of training, a human annotator is still able to outperform the GoogLeNet result (p=0.022) by approximately 1.7%." |

**❌ Negative 1：`in_doc_swap`（文档内替换，score=0.26）**

> 保留 Figure 6，把 Table 14 换成**同文档的 Table 1**（ILSVRC 任务标注概览表）。Table 1 和 query 讲的完全不是同一件事，但模态类型（figure+table）一致、文档一致——这是最基础的干扰项，考察模型能否区分"同类型但错误内容"。

**❌ Negative 2：`same_type_hard_plus`（跨文档同主题，score=0.22）**

> 来自文档 **1707.09457**（视觉识别中的性别偏差）的 figure+table bundle：Table 1（vSRL/MLC 任务统计）+ Figure（MS-COCO bias analysis）。主题同属"公平性+视觉识别"，query token overlap 高（0.25），但证据内容和正例讲的是完全不同的实验。这类负例是真正的 hard negative，`sim_query=0.14`、`sim_bundle=0.24`——和正例的语义距离比 in_doc_swap 近得多。

**两版负例策略对比（同一条 query）：**

| 负例类型 | v1（`same_type_hard`） | v2（`same_type_hard_plus`） |
|---------|----------------------|--------------------------|
| 跨文档负例选取依据 | 同 pair_type + query 词汇 overlap | 加入 bundle/span/bridge 多维相似度加权 |
| avg_difficulty | 0.62 | **0.73** |
| 对 BM25 baseline 的影响 | global acc@1 = 0.55 | global acc@1 = 0.45 |

avg_difficulty 上升、BM25 下降——说明 v2 的负例更难通过词法捷径区分，符合预期。

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
