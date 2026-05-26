# 当前技术方案

_整理时间: 2026-05-25 · 状态快照_

## 这个项目到底在干什么

输入: 一堆 PDF 论文（多模态 ML 论文为主，1147 篇规模；公平性/偏见 53 篇子集用来做跨文档实体桥探索）。

输出: 一批 M4 风格的多模态检索三元组 —— 每个 query 需要从一篇论文里同时召回**图 + 表 + 公式 + 段落 + chunk** 多个粒度的证据，并能跟该论文的其他干扰元素以及跨论文的噪声区分开。

为什么不简单: 这些 PDF 没有 LaTeX 源码，只能靠 MinerU 解析。解析出来的 caption 87% 跟邻居 caption 一个词都对不上，所以单靠 CLIP 跨文档连边都是噪声。整个方案的核心问题就是: **没有 LaTeX 的时候，怎么把图、表、公式跨文档地正确串起来。**

---

## 三个 query 示例（取自实际交付）

**交付物**: [data/03_queries/M4query_v2_clean_chunk_aug/](../data/03_queries/M4query_v2_clean_chunk_aug/)（5/18 推送的版本）
**规模**: 8,104 条 query × 3-4 positive × 5 hard_neg × 5 random_neg；corpus 169,671 条 passage（paragraph 117K / chunk 29K / figure 9K / table 7K / formula 6.9K）
**说明**: 这是给检索训练用的三元组格式。下面三条原样从 [train_triplets.jsonl](../data/03_queries/M4query_v2_clean_chunk_aug/train_triplets.jsonl) 抽，positive 里的图直接嵌进来。

---

### 示例 1: figure + formula + 段落 + chunk

`query_id: l3_de_2506.18504_0006`（VLM prompt design）

> **Query**: What prompt design addresses the timeline's growing domain-diverse adaptation pressure when frozen prompts are insufficient under significant domain gap?
>
> **Reasoning chain**: The field is increasingly focused on diverse task-specific adaptation problems rather than relying only on general-purpose models. Because domain gaps can defeat frozen-prompt methods, the paper motivates moving toward transfer mechanisms that adapt more explicitly to domain differences. The proposed prompt structure answers that need by combining shared prompts with domain-specific prompts for source and target domains.
>
> **Answer**: The timeline shows rapid expansion of task-specific adaptation methods alongside general MLLMs, implying increasing pressure to handle varied transfer settings. The bridge explains that frozen prompt learning can be inadequate when the domain gap is large, motivating stronger transfer mechanisms. The resulting design uses both domain-invariant prompts and domain-specialized prompts for source and target domains, capturing shared structure while adapting to each domain.

**Positives (4 个)**:

- `2506.18504_figure_3` — *"Fig. 3. An overview of recent advances in vision-language systems, including adapting VLMs (lower part) and building generalist multimodal models (upper part)."*

  ![fig3](../data/03_queries/M4query_v2_clean/images/2506.18504/eabf5a7f737e728bc6a0f2a0945d27a3be118656b03e96d41c4332dd677925c6.jpg)

- `2506.18504_formula_6` — `$$t_c^d = [U]_0, ..., [U]_k [V]_{k+1}^d, ..., [V]_m^d [\mathrm{class}]_c$$`（domain-aware prompt 模板）

- `2506.18504_paragraph_92` — *"Prompt-based methods introduced in Section 3 learn additional prompts with original parameters frozen, which may not be sufficient for tasks with significant domain gap..."*

- `2506.18504_chunk_18` — section-level 聚合，覆盖 paragraph_92 所在节

**Hard negatives** (同文档干扰): `chunk_38`, `chunk_39`, `figure_5`, `figure_1`；外加一个 GPT-3 论文段落 `2005.14165_paragraph_322`
**Random negatives**: 5 条来自其它论文的混合粒度

---

### 示例 2: figure + table + 段落 + chunk

`query_id: l3_de_1809.01696_1562`（Video QA, 多模态 answer selection）

> **Query**: Which baseline is motivated by the multi-stream model's answer selection setup because correct answers tend to be longer than wrong ones across question types?
>
> **Reasoning chain**: The task setup explicitly evaluates candidate answers paired with a question using a scoring model. Because answer candidates are being selected, the authors use answer-length bias as the mechanism for defining their first baseline. The per-question-type length columns verify that correct answers are longer on average, supporting the longest-answer baseline.
>
> **Answer**: The model in the diagram scores candidate answers for each question-answer pair, and the bridge explains that this motivates a simple baseline that picks the longest candidate because correct answers are on average longer. The statistics confirm that pattern by showing CA Len. exceeds WA Len. across the question-type rows and in the total row. Therefore, the baseline is to select the longest answer for each question.

**Positives (4 个)**:

- `1809.01696_figure_4` — *"Figure 4: Illustration of our multi-stream model for Multi-Modal Video QA. Our full model takes different contextual sources..."*

  ![fig4](../data/03_queries/M4query_v2_clean/images/1809.01696/0db57598946da7ecb06b03a76ddfa321ebbe7f6f25704a9f80014c79a414b2f9.jpg)

- `1809.01696_table_1` — *"Table 1: Statistics for different question types based on first question word."*

  ![tab1](../data/03_queries/M4query_v2_clean/images/1809.01696/2144dce3b1aea6f18bf97d5a447850a2ae7b842c545fcb572586e1667f45eb1a.jpg)

- `1809.01696_paragraph_75` — 描述 longer-answer baseline 的动机段
- `1809.01696_chunk_17` — 段落 75 所在的 chunk

**Hard negatives** + **random negatives**: 略（5 + 5，分布同 EX1）

这条最干净: query 同时锚 figure（模型图）和 table（统计表），需要把"correct answers longer than wrong"这个观察和"multi-stream"这个模型设计 join 起来。

---

### 示例 3: formula + table + 段落 + chunk

`query_id: l3_de_1810.06553_0957`（Recipe-image cross-modal retrieval, AdaMine）

> **Query**: Which retrieval advantage for AdaMine on Recipe1M follows from training recipe-image alignment with cosine similarity and then ranking queries in the common space by cosine similarity?
>
> **Reasoning chain**: The method is trained with a cosine-based objective that structures recipe and image embeddings in a shared space. Because evaluation ranks candidates by cosine similarity in the same common space learned by the loss, improved alignment should directly improve retrieval. AdaMine verifies the expected retrieval advantage by achieving the strongest performance across both retrieval directions.
>
> **Answer**: The loss uses cosine similarity to pull matched recipe-image pairs together and push mismatched pairs apart in a shared embedding space. The evaluation then ranks im2recipe and recipe2im candidates by cosine similarity in that same common space, so better alignment should translate into stronger retrieval. In the AdaMine row, that expectation is confirmed by the best retrieval results, with the lowest median ranks and highest recall values among the listed methods.

**Positives (4 个)**:

- `1810.06553_formula_1` — cosine loss 定义: `$$L_{cos}(\phi^r, \phi^v, y) = \begin{cases} 1 - \cos(\phi^r, \phi^v), & y=1 \\ \max(0, \cos(\phi^r, \phi^v) - \alpha), & y=-1 \end{cases}$$`

- `1810.06553_table_3` — *"TABLE 3. Im2recipe retrieval comparisons on Recipe1M. Median ranks and recall rate at top K are reported for baselines and our method."*

  ![tab3](../data/03_queries/M4query_v2_clean/images/1810.06553/759e448ef20805293d2e66d6fdcd8bfeb616832a6597969cd619866f73fd690c.jpg)

- `1810.06553_paragraph_118` — *"For evaluation, given a test query image, we use cosine similarity in the common space for ranking the relevant recipes..."*

- `1810.06553_chunk_27` — 评估协议 chunk

**Hard negatives**: `1810.06553_chunk_18`, `figure_6`, `figure_8`（同文档），`2409.00147_chunk_14`, `2412.07213_chunk_5`（跨文档）
**Random negatives**: 5 条跨论文混合粒度

这条体现公式 bucket 在产线里的位置: query 描述 cosine 训练 + cosine 排序的方法学闭环，positive 必须把 loss 定义（formula）和实际指标（table）都召回。

---

### 三条共有的产线特征

1. **正例都来自同一文档** —— 整个 8,104 条 query 都是 intra-doc 多粒度正例（cross-doc 那条线没进这一版交付，见后文 T2）
2. **每条都有一个 chunk 兜底** —— 即使图/表/公式没召回，section-level chunk 也能给检索一个降级信号
3. **Hard negative 主要来自同文档** —— 训练时引导模型分辨"同篇的别张图" vs "正确那张图"
4. **Visual negative 占比 ~45%** ——文本元素不再欠采样

---

## 主力生产管线（已交付到 M4query_v2_clean_chunk_aug）

**两个关键事实**:
1. 当前交付的 8,104 条 query 全部是**单文档内**多模态检索。跨文档相关的能力虽然有验证（见下一节），但**从未进入主力生产**。
2. 当前交付的**拓扑骨架还在用 LaTeX 引用图**（`latex_reference_graph_v2.json` + `latex_hub_multihop_candidates_v2.json`），MinerU-only 那条线虽然已验证（见下一节 b. C15），但**没进这一版交付**。

### A. PDF → 结构化元素

- MinerU 解析 PDF，每个元素带 `doc_id + element_type + page_idx + position_idx + content/caption/context + image_path + bbox`
- 1147 篇全集都跑过

### B. LaTeX 引用图作为拓扑骨架

- `latex_reference_graph_v2.json`: 从 `.tex` 源码抽 `\ref{}` / `\cite{}` 得到的硬引用图
- `latex_hub_multihop_candidates_v2.json`: 从引用图里挑出 hub element（被多段正文引用的中心图/表/公式）以及它们的多跳邻居
- 在此基础上**严格 intra-doc 过滤**生成两个候选池:
  - `l3_candidates_v4_intra_doc.json` —— L3 dual-evidence 候选
  - `m2_diverse_candidates_intra_doc.json` —— M2 多样性候选

### C. Query 生成: m2/m15 reasoning_path 线

主力脚本: [scripts/generate_multihop_l1_queries.py](../scripts/generate_multihop_l1_queries.py)
配置矩阵 (`slurm_scripts/12_production_sweep.sh` 6 个 array job):

| Array | 候选池 | style | persona | 量 |
|---|---|---|---|---|
| 0 | L3 | academic | off | 88 |
| 1 | L3 | academic | on | 88 |
| 2 | L3 | mixed | off | 88 |
| 3 | L3 | mixed | on | 88 |
| 4 | M2 | academic | off | 108 |
| 5 | M2 | mixed | on | 108 |

每条 query 的 schema 关键字段:
- `query` + `answer`
- `reasoning_chain`: 从 endpoint element + 段落桥推导出的因果/比较推理链
- `path`: `[elem_a, ::p::00010, elem_b]` 或类似的元素-段落桥结构
- `element_ids`: 两端 endpoint
- `required_evidence_spans`: 每个 element 上锁定的具体证据片段
- `dual_evidence: true` / `cross_modal: true` / `query_type ∈ {causal_explanation, comparison, ...}`

更大规模跑: `slurm_scripts/53_m2_m15_reasoning_path_production.sh` 和 `57_graph_max20k_reasoning_production.sh`（`hub_pairs_graph_max20000_production_full_*`，2 万对量级）。

### D. 交付打包: clean → chunk_aug

[scripts/prep_delivery_chunks.py](../scripts/prep_delivery_chunks.py) + `scripts/build_clean_chunk_aug.py`:

- 每条通过 QC 的 query → 3-4 个正例
  - 同篇的 figure / table / formula / paragraph 原始证据
  - section-level chunk (~400 词，按 section 边界聚合) 兜底
- 5 个 hard negative：同文档其它干扰元素为主 + 少量跨文档相邻话题
- 5 个 random negative：跨文档随机抽
- 原 bridge 节点（跨文档段落桥）的处理:
  - 能在 1147 篇 corpus 里找到 source paragraph 的 (7,471 条) → 直接替换为 source paragraph + chunk
  - 找不到 source 或 source 是 figure/table/formula 的 (1,118 条) → 把 bridge 文本本身当作 type=paragraph 的假节点入库
- Visual:text negative 比例从原 72:28 重平衡到 ~45:55

### E. QC 闸门

- **Pass-only filter**: 通过 LLM judge 的 query 才入 `*_pass.jsonl`，通过率 ~25-30%
- **Evidence grounding judge**: 每个 positive 必须能被 query 直接锚到
- **HopWeaver 规则 QC**: hop 不重复用单文档；不能跨越式桥接；因果方向 premise → intermediate → conclusion
- **Local API logger**: 所有 LLM 调用走 `wrap_requests_call`，token 全程留痕

**主力交付物**: [M4query_v2_clean_chunk_aug](../data/03_queries/M4query_v2_clean_chunk_aug/) —— 8,104 query × (3-4 pos + 5 hard_neg + 5 rand_neg)，corpus 169,671 passage。

---

## 已验证但未进入主力交付的能力

这四件事**有证据、有产物**，但都还**只活在 `data/01_graphs/` 和 `data/04_xdoc_citation/`**，没参与过当前交付的 query 生成或三元组打包。它们是下一版交付要消化的素材。

### a. MinerU 替代 LaTeX 作为文档内拓扑骨架（C15）

- 从 MinerU 输出里 regex 抽 "Figure N" / "Table M" / "Eq. K" 这类显式引用
- 跟 LaTeX `\ref` 在 52 篇重叠文档上 A/B：**84% 召回**、26/52 篇 100% 召回、人工抽样 6/6 正确
- **现状**: 验证了"MinerU 可以替代 LaTeX"，但当前交付的拓扑还是 `latex_reference_graph_v2.json`；下一版要拿这个替换掉 LaTeX 依赖，把 corpus 从 1147 篇 LaTeX 可用论文扩到任意 PDF

### b. 跨文档引用预测（C18）

- 用 LaTeX 引用图当 ground truth 训 XGBoost：chunk-level **AUC 0.852 / F1 0.746**，top-50 precision = 1.0
- 推到 1147 篇全集，预测出 **53,435 条跨文档引用边**（75% 概率 ≥ 0.95）
- 主特征：`title_match`（论文标题出现在引用文本里），占 88% 重要性
- **现状**: 边没用进当前交付的 query 生成；下一版要靠它做跨文档 query 候选预过滤

### c. 公式专用 encoder（C17）

- CLIP text 上所有公式相似度挤在 0.92 附近，标准差只有 0.027 —— 不可区分
- 换 `math-similarity/Bert-MLM_arXiv-MP-class_arXiv` 后标准差到 0.172 —— 可用
- 产出 4,331 条公式相似度边
- **现状**: 用于图边构建，没用进当前交付的 retrieval encoder

### d. 跨文档视觉相似度（C16，仅召回层）

- CLIP 视觉 + caption rerank 产出 3,238 条候选边
- 诚实结论：**87% 的候选 caption 一个词都不重合，只有 5% 有真实文本支撑**
- 解析出来的 caption 35% 是 "(a)(b)" 子图标 / 残缺 OCR / 太短不可用
- **现状**: 当召回层够用，当硬边不够，等更强的语义判官

---

## 探索中的跨文档路线（都还没进交付）

跨文档是下一步的核心目标。三条候选路线在并行验证：

### 路线 1: 实体桥接链

**做法**: 两篇论文如果共享 ≥2 个高 IDF 实体（如 "winobias" + "coreference resolution"），就建实体桥；BFS 找 3-paper 2-bridge 的元素链。

**当前数据**（53 篇公平性子集）:
- 83 对 entity-bridge pair 经 LLM judge，**25.3% 端到端 strong**（21/83）—— 这是目前最强的跨文档精度信号
- 进一步组成 38 条 3-paper 链，其中 3 条两端 bridge 都 strong、9 条至少一端 strong
- 已经能投影成多轮 session（Trinity E2E：16/16 生成、50% turn-dependency QC 通过）

**下一步 (T2)**: 在 1147 篇全集上跑（用上面的 a. citation prediction 做预过滤），目标 ≥ 100 条链级 strong → 才能升级为主力交付的跨文档输入。

### 路线 2: 元素级语言学验证

**做法**: 直接对（caption + enriched_content）元素对跑 Genette + RST 关系分类，跳过此前 section→element 笛卡尔投影那一步。

**前期教训**: section 级 100 条边里 46% usable，但投影到 element 级后链头到链尾 0% strong —— 笛卡尔积稀释了信号。

**下一步 (T1)**: 直接元素对判定，top-1500 高 CLIP 相似度元素对，预算 ~$10。链级 head-to-head usable ≥ 15% 救活，< 5% 关闭。

### 路线 3: 公式 routing 全量验证

**做法**: query 含公式 anchor 时路由到数学专用 encoder + RRF 融合。

**现状**: smoke50 上公式 bucket R@10 +7.3pp（0.5600 → 0.6313），但全量集没跑过。

**下一步 (T3)**: 1147 篇全量重跑。≥ +5pp 进 paper 主结果，≤ +2pp 标 smoke-only。

---

## 技术细节（部分可考虑申请技术秘密）

⭐⭐⭐ = 强建议申请；⭐⭐ = 可申请；⭐ = 内部 know-how。每条只列已经在 claims / experiments 文档里写过的口径，没扩出验证之外的实现细节。

### T1. ⭐⭐⭐ 把 LaTeX 引用图当弱标签，训 XGBoost 预测跨文档引用边（claim C18）

- 训练数据：4,028 对 LaTeX → MinerU 对齐的 citation pair
- 特征：只用 MinerU 解析后能算的字段。**`title_match`（论文标题出现在 chunk 文本里）单独占 88.1% 的特征重要性**
- 评估：按 source doc 分组的 5 折交叉验证（同篇 chunk 不会跨折叠泄露），**AUC 0.852 / F1 0.746，每折 top-50 precision 都是 1.0**
- 推理：在 1147 篇全集上跑出 53,435 条预测边，**75% 概率 ≥ 0.95**
- 适用范围：MinerU 输出里 paper title 保留在 reference list 且正文里有 citation marker 的论文；OCR 噪声大或 reference 段解析差的论文表现会变弱

**为什么可申请**：用 LaTeX bibliography 做弱标签 + 只用 PDF 解析字段做特征，是为了让模型训完之后能在**没有 LaTeX 的纯 PDF** 上推断。这个组合方案公开文献里没有。

### T2. ⭐⭐⭐ 公式相似度从 CLIP 切到数学专用 encoder（claim C17）

- 诊断：抽 200 条公式样本测 CLIP 文本 encoder 给的两两相似度，**标准差只有 0.027**（任何两条公式 CLIP 都觉得"差不多像"，threshold 切不开）
- 替换：换成 `math-similarity/Bert-MLM_arXiv-MP-class_arXiv`（768-d），同样 200 条样本上**标准差涨到 0.172**，范围 0.036 ~ 0.977，可用
- 自动阈值：基于该分布把 threshold 从 0.45 重标定到 0.85
- 入图：写进 `build_mineru_vl_edges.py` 的 `--formula-backend math_similarity`，跟 CLIP 视觉/文本分离独立；全量产出 4,331 条公式相似度边
- 后续：smoke50 上把 query 含公式 anchor 时路由到这个 encoder + RRF 融合，formula bucket R@10 +7.3pp（0.5600 → 0.6313）

**为什么可申请**：业内做多模态检索默认对公式走 CLIP 文本通道。"用 encoder 输出的标准差诊断 → 切换数学专用 encoder → 仅在 query 含公式 anchor 时路由"这套组合是我们独有的发现路径。

### T3. ⭐⭐⭐ Bridge 节点的两段式 fallback（让跨文档训练信号"装进" intra-doc 三元组）

跨文档 L3 推理链原本带一个"桥段落"（跨论文的概念衔接文本）。直接保留会让 corpus 多出一类不属于任何论文的 passage。我们的处理（来自 `M4query_v2_clean_chunk_aug` README 的事实）：

- **第一段：source paragraph 替换（7,471 条 / 86.9% 成功）**——在 1147 篇 corpus 里找到该桥文本对应的真实 source paragraph 时，把 bridge 节点替换为 source paragraph + 它所在的 chunk，正例数 3 → 4
- **第二段：假 paragraph 保留（1,118 条 / 13.1% 失败）**——找不到 source（435 条），或 source 是 figure / table / formula（683 条）时，把桥文本本身**改 type 为 paragraph 当合成节点入库**，passage_id 用 `<query_id>_bridge` 命名以免和真实 paragraph ID 冲突

**为什么可申请**：这是让 8,104 条 query 表面格式上都是 intra-doc、但训练时仍保留跨文档推理信号的核心 trick。公开的多跳问答数据集（HotpotQA、MultiHopQA 等）都没人这么处理跨文档桥。

### T4. ⭐⭐ Chunk 聚合规则

- 按 section 边界 + 约 400 词软上限聚合 paragraph 成 chunk
- **不跨 section**：即使前后段语义连续也不合并
- **不滑窗、不重叠**：跟主流检索框架（LangChain / LlamaIndex 默认 512 token + 50 overlap）不一样
- **每条 query 必附一个 chunk 正例兜底**：即使 figure / table 没召回，section-level chunk 也能给检索一个降级信号
- corpus 169,671 passage 里 chunk 占 29,237（17.2%）

**为什么可申请**："不跨 section + 必附 chunk 兜底"是从 C9（chunk-as-retrieval-unit 稀释 dual-evidence 信号）的失败反推出来的设计，不是行业默认。

### T5. ⭐⭐ 负样本三层重平衡（README 直接证据）

- **Hard negative 按来源分层**：以同文档其它元素为主（强迫模型分辨"同篇别的图" vs "正确那张图"），少量跨文档话题相邻（防止只学到文档 ID 这种 shortcut）
- **Visual / 文本 比例从 72:28 调到 ~45:55**：原始按 corpus 模态分布抽是 visual 占 72%，手动压到 45 让文本元素不再欠采样
- **抽取概率**：hard_neg 和 random_neg 各随机抽 2-3 个文本类（chunk / paragraph）slot——75% 概率 3 个，25% 概率 2 个

**为什么可申请**：公开的多模态检索数据集都不做模态级重平衡，训出来的 retriever 对 visual 模态过拟合。

### T6. ⭐⭐ Turn-dependency QC 协议（多轮 session 用，跨文档链路上的核心闸门）

- **抹除测试**：把 Turn N-1 的 assistant 回答抹掉 → 重问 Turn N → 如果 LLM 还能答对，这条 session **作废**
- **指代强制**：每个 Turn N ≥ 2 必须含至少 1 个指代表达式（pronoun / 定指 NP / 省略），抗"独立可答"作弊
- **Evidence 锁定**：style pass 只允许加 persona / 指代变形，`element_ids` 和 `required_evidence_spans` 不动
- 当前验证：L3 链投影 60% 通过、entity-bridge 链投影 100% 通过（smoke）

**为什么可申请**：把"多轮 session"从"两轮拼接"升级到"判定真依赖"的 QC 协议。公开的多轮 LLM 评测都没这么严格。

### T7. ⭐⭐ Entity-bridge 链的 IDF 双阈值（跨文档链构造）

- `min_idf = 2.5`：抽出的实体必须在 corpus 里 IDF ≥ 2.5（剔除 "method" / "model" / "result" 这类高频通用词）
- `min_elem_overlap = 2`：两篇 paper 共享 entity 必须 ≥ 2 个，单实体不构成桥
- `max_hops = 2`：BFS 限 2 跳，防止链过长稀释推理信号
- 53 篇子集上的效果：83 对 entity-bridge pair 经 LLM judge **25.3% 端到端 strong**（21/83）—— 当前最强的跨文档精度信号

**为什么可申请**：实体桥本身是公开方法，但这套阈值组合是反复 ablate 出来的；公开文献没有匹配的设置。

### T8. ⭐ HopWeaver 规则 QC 三件套（生产管线已集成）

- 每个 hop 必须用不同文档
- 不能有单文档桥接不相邻 hop
- 因果链方向必须是 premise → intermediate → conclusion

**为什么不必申请**：HopWeaver 文献已发表，我们是实现者不是发明者，作为内部工程 know-how 保留即可。

### T9. ⭐ Pass-only filter 通过率

- 用 gpt-5.4 做 evidence-grounding judge
- 通过率稳定在 25-30%（多 array job 验证）

**为什么不必申请**：通用 LLM-as-judge 模板，没有独特做法。

---

## 一句话总结

**当前主力交付（M4query_v2_clean_chunk_aug，8,104 条 query）是 LaTeX 引用图 + 严格 intra-doc 过滤跑出来的单文档多粒度检索。MinerU 替代 LaTeX、跨文档引用预测、公式 encoder、跨文档视觉召回这四件事都已经验过单点能力，但没参与过这一版交付。下一版的两个独立目标: 把拓扑骨架从 LaTeX 切到 MinerU（解锁任意 PDF），把跨文档真正做进 query 生成（解锁多文档推理）。**
