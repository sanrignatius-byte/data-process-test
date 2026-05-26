# 当前技术方案

_整理时间: 2026-05-26 · 状态快照_

## 这个项目到底在干什么

**一句话**：把一堆 arXiv 论文变成 16,000+ 条"需要同时看图的多个元素才能回答"的检索训练数据。

具体来说，我们拿了 3,200+ 篇论文（1,147 篇 CS 机器学习 + 2,103 篇数学/物理/天文），给每篇论文里的 figure、table、formula、paragraph 建了一张图，然后在这张图上找"单看一个元素答不出来，必须串 2-3 个元素才能推理"的路径，让 LLM 把这些路径写成自然语言 query。

最终产出是检索训练用的三元组：每个 query 配上 3-4 个必须召回的 passage（正例）+ 5 个同文档干扰 + 5 个跨文档噪声。

**已经交付了两版**：

| 交付包 | 论文 | query 数 |
|--------|------|----------|
| `M4query_v2_clean_chunk_aug` | 1,147 篇 CS/ML | 8,104 |
| `M4query_noncs2000_final` | 2,103 篇 非CS | 8,204 |

---

## 图是什么——这是整个方案的核心

### 为什么需要图

一篇论文里，Figure 3 引用了 Table 2 的数据，Section 4 解释了 Formula 7，Formula 7 又被 Table 2 的参数表用到——这些"谁引用了谁"的关系，是构造多跳推理 query 的唯一依据。没有图，你只能随机抓两个元素让 LLM 硬编关系，编出来大概率是假的。

### 节点：五种元素

从每篇论文的 PDF（MinerU 解析）和 LaTeX 源码中提取五种元素作为图的节点：

| 节点类型 | 从哪来 | 内容 |
|----------|--------|------|
| **figure** | MinerU（图片 + caption） | 图片文件 + VLM 生成的视觉描述 |
| **table** | MinerU（表格截图 + HTML body） | markdown 表格文本 + caption + VLM 描述 |
| **formula** | LaTeX 源码 `\begin{equation}` | LaTeX 公式文本 + VLM 语义描述 |
| **paragraph** | LaTeX 源码正文段落 | 纯文本 |
| **section** | LaTeX section 标题 + 正文 | 章节标题 + 正文 + LLM enriched 摘要（含 section_type 和 keywords） |

一个典型的 paper 有 30-80 个节点。

### 边：两套来源，合在一起用

边的来源分两路，**两路合并**才构成完整的图：

**路 1 —— LaTeX 引用边（精确，覆盖率 ~50%）**

从 `.tex` 源码里直接解析 `\ref{fig:result}`、`\cite{he2016deep}`、`\label{tab:data}` 这类 LaTeX 交叉引用命令。这是一对一的精确边——Figure 3 明确引用了 Equation 7。

局限：只有有 LaTeX 源码的论文才能提取；MinerU 解析和 LaTeX 的元素不是 100% 对齐的（figure 匹配率 ~50%，table ~67%，formula 0%——公式压根没有 LaTeX label 到 MinerU ID 的映射）。

**路 2 —— MinerU 内容边（模糊但全覆盖）**

从 MinerU 解析的正文文本里，用正则匹配 "Figure 3"、"Table 2"、"Eq. (7)" 这类显式引用文本，建到对应 element 的边。84% 召回率 vs LaTeX 引用，不需要 LaTeX 源码就能跑。

另外还有跨文档的边：从 `\cite{}` 建的论文级引用图，以及用 CLIP visual encoder 算出来的跨文档图片相似度边（不准但能当召回层用）。

**合并策略**：LaTeX 边精确度最高，优先采用；MinerU 边覆盖面更广，作为补充。两套边合并后建 DAG，然后跑 hub 检测找"连接多条路径的关键节点"。

### 图带来了什么提升

在 M4query_v1 的 473 条 query 上做检索评测，加上图 rerank 之后：

| 方法 | R@10 | 说明 |
|------|------|------|
| 纯 dense embedding | 0.6195 | Qwen3-Embedding-4B |
| **+ graph hub rerank** | **0.6913** | 图传播提升 +7.2pp |
| + graph neighbor prop | 0.7100 (smoke50) | 沿图邻居传播 |

**这里的图提升是怎么来的**：不是 embedding 变好了，而是检索时用图结构把排名重排了——跟 query 里的 element 在图上有直接连边的 passage，排名往上提；同文档但图上没连边的，排名往下压。本质上是用论文作者写的引用关系来纠正 embedding 的排序错误。

**图增益是模态选择性的**（C10）：图 rerank 在 figure 上 +10.3pp，table +8.3pp，但在 formula 上 +0pp——formula 的瓶颈是 dense encoder 在 LaTeX 内容上的表达上限（C11），不是图拓扑问题。

---

## 三个 query 示例

**交付物**: `data/03_queries/M4query_v2_clean_chunk_aug/`（5/18 推送的版本）
**规模**: 8,104 条 query × 3-4 positive × 5 hard_neg × 5 random_neg；corpus 169,671 条 passage（paragraph 117K / chunk 29K / figure 9K / table 7K / formula 6.9K）
**说明**: 这是给检索训练用的三元组格式。下面三条原样从 `train_triplets.jsonl` 抽，positive 里的图直接嵌进来。

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

  ![fig3](images/ex1_vlm_overview.jpg)

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

  ![fig4](images/ex2_videoqa_model.jpg)

- `1809.01696_table_1` — *"Table 1: Statistics for different question types based on first question word."*

  ![tab1](images/ex2_question_stats.jpg)

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

  ![tab3](images/ex3_recipe1m_table.jpg)

- `1810.06553_paragraph_118` — *"For evaluation, given a test query image, we use cosine similarity in the common space for ranking the relevant recipes..."*

- `1810.06553_chunk_27` — 评估协议 chunk

**Hard negatives**: `1810.06553_chunk_18`, `figure_6`, `figure_8`（同文档），`2409.00147_chunk_14`, `2412.07213_chunk_5`（跨文档）
**Random negatives**: 5 条跨论文混合粒度

这条体现公式 bucket 在产线里的位置: query 描述 cosine 训练 + cosine 排序的方法学闭环，positive 必须把 loss 定义（formula）和实际指标（table）都召回。

---

### 三条共有的产线特征

1. **正例都来自同一文档** —— 当前交付的 query 都是 intra-doc 多粒度正例（跨文档能力已验证但没进这版交付）
2. **每条都有一个 chunk 兜底** —— 即使图/表/公式没召回，section-level chunk 也能给检索一个降级信号
3. **Hard negative 主要来自同文档** —— 训练时引导模型分辨"同篇的别张图" vs "正确那张图"
4. **Visual negative 占比 ~45%** ——文本元素不再欠采样

---

## 生产管线（统一）

一条管线，跑了两批论文（CS 1,147 篇 + 非CS 2,103 篇），产出两版交付。管线的每一步对两批论文都一样，只是具体文件不同。

### 第 1 步：论文采集

从 arXiv 搜 survey/review 论文作为种子 → 下载种子论文的 LaTeX 源码 → 从 `.bbl` / `.bib` 文件里提取所有引用论文的 arXiv ID → 下载这些论文的 PDF + LaTeX → 得到论文池。

- CS 论文（v2）：关键词搜 "multimodal LLM survey" 等 topic
- 非 CS 论文（noncs2000）：从 math / astro-ph / cond-mat / hep 等类别各搜 survey
- 华为域论文（进行中）：83 个 topic query 覆盖无线/光通信/AI/计算/终端/能源/汽车/芯片/材料

### 第 2 步：建图

三件事按顺序跑：

1. **MinerU 解析 PDF** → 提取 figure / table / formula / paragraph，输出 `multimodal_elements.json`
2. **LaTeX 建引用图**（`build_latex_reference_graph.py`）→ 从 `.tex` 源码解析 `\ref` / `\cite` / `\label`，建 DAG
3. **拓扑分析**（`analyze_latex_graph_topology.py`）→ 在图里找 hub（度数高于均值 2σ 的节点），以 hub 为中心提取 2-hop 和 3-hop 路径，输出 hub 多跳候选池

### 第 3 步：Enrichment（三层）

候选池里的节点需要语义信息才能让 LLM 生成有意义的 query。Enrichment 分三层：

**层 1: Element 级（figure / table / formula）**

`scripts/enrich_elements_modora.py`：用 VLM 读懂图/表的内容，用 LLM 解释公式的含义。输出 `enriched_content` 字段（图的视觉描述、公式的语义解释）。

- v2（CS 论文）：1,316 个 element，1,285 个 enriched（97.6%）
- noncs2000：196,748 个 element，174,049 个 enriched（88.5%）

支持 `--num-shards --shard-index` 多进程并行加速。

**层 2: Section 级**

`scripts/enrich_section_nodes.py`：对每篇论文的 section / subsection / subsubsection 节点，用 LLM 生成三样东西：
- `enriched_title`：把 "Introduction" 这种通用标题改写成具体内容描述（如 "Motivation and aims of two-field inflation perturbation study"）
- `enriched_content`：section 的 1-2 句摘要
- `enriched_metadata`：section_type（introduction / methods / results 等）+ keywords

noncs2000 产出 23,940 个 enriched section。Query 生成时，section 摘要是 prompt 的重要背景信息，帮助 LLM 理解论文的主题和方法。

**层 3: Hub candidate 级**

`scripts/enrich_hub_candidates.py`：对拓扑分析出的 hub 多跳候选 pair，填充 bridge 文本、edge context、quality_score 等字段。这是 query 生成的直接输入。noncs2000 从 14,638 条拓扑候选中产出 6,521 条 L3 enriched pair。

### 第 3.5 步：Chunk 聚合

Paragraph 是论文正文的自然段落，粒度太细（几十到几百词不等），直接在检索里用容易把排名打散。Chunk 的设计就是把相邻的 paragraph 按 section 边界聚合成更大的语义单元，作为检索的"降级兜底"信号。

`scripts/build_hierarchical_chunks.py`：

1. **按 section 边界分桶**：同一个 section 下的 paragraph 归到一个桶里，section 变了就切新桶。不跨 section，不滑窗，不重叠
2. **~400 词软上限**：每个桶内的 paragraph 依次拼起来，超过 400 词就封口成一个 chunk，继续拼下一个
3. **注入 element 语义**：每个 chunk 记录它包含了哪些 figure / table / formula（通过 chunk→element 归属关系），并把对应的 enriched_content 注入 chunk 的 `enriched_description` 字段

**为什么这么做**：检索评测显示，只用 paragraph 做检索单元时 visual 元素（figure/table/formula）在训练中被欠采样。Chunk 作为一个更大的文本粒度，能跟 visual 元素平衡分布（corpus 里 chunk 占 ~17%）。而且从训练角度看，即使模型没把具体的 figure 召回，只要召回了它所在的 chunk，也给了检索一个合理的降级信号。

noncs2000 产出了 46,991 个 chunk，M4query_v2 产出了 29,237 个。最终 corpus 是五粒度共存：figure + table + formula + section + chunk。

### 第 4 步：Query 生成

核心脚本 `generate_multihop_l1_queries.py`：把 enriched candidate pair + 图上 3-hop 路径的桥文本 + section 摘要 喂给 LLM，让它写出一个"需要同时看两个元素才能回答"的推理型 query + answer + reasoning chain。

**多轮生成策略**：同一批 candidate，用不同配置跑多轮，每轮产出风格不同的 query：

| 配置 | 风格 | 说明 |
|------|------|------|
| `academic` | 学术论文口吻 | "Which mechanism explains..." |
| `academic + persona` | 学术 + 人设 | 76 种学术人设随机分配（phd / postdoc / 工程师） |
| `mixed + persona` | 混合 + 人设 | 50% 学术 50% 真用户问法 |
| `real_user` | 真用户口吻 | 5 种模板轮换（factual / summary / comparison / how_works / what_if） |

每轮跑完后，统计哪些 candidate 还没产出 pass query（`--skip-done`），下一轮接着处理（retry）。两批论文共用这套策略。

**通过率**：CS 论文上 25-30%，非 CS 论文上 47-52%。差距的原因是非 CS 论文的 hub candidate 质量更稳定（hop≥3 的路径中 bridge 文本更完整），不是 enrichment 覆盖率的问题——v2 的 element enrichment 现在也是 97.6%。retry 轮 pass rate 略低于 sweep 轮（47% vs 52%），根因是 gpt-5.4 输出中 anchor_leakage 增多——生成更多 "blue curve" / "upper panel" 类视觉描述词，和 anchor 文本的 Jaccard 重叠超过 0.20 阈值。**已在 prompt 模板加 Rule 13 压制**。

### 第 5 步：QC 双层闸门

只有两层都过的 query 才进最终交付。

**Layer 1: 规则 QC** —— 15+ 项原子检查：

- 不能说 "the figure shows..."（meta_language）
- 不能直接抄数字（numeric_leakage）
- query 不能和 visual anchor 文本重叠超过 20%（anchor_leakage）
- query 必须问一件事，不能 "and what" 问两件（parallel_dual_ask）
- 每个 evidence element 单独拿出来都不能回答 query（否则是伪多跳）

**Layer 2: LLM judge** —— 规则过了之后再让 LLM 判两次：

- **单元素消融**：依次只给一个 evidence element，看 LLM 能不能答对。任何一个 element 单独就能答 → 伪多跳，作废
- **答案 grounding**：给 LLM 看 evidence（文本 + 图），判断 answer 的每句话能不能从 evidence 推出来。出现幻觉 → 作废

### 第 6 步：打包

`package_noncs2000_final.py`（可适配不同论文池）：

1. 合并所有 pass 文件，按 query_id 去重
2. 从 MinerU 元素 + section + chunk 建 corpus（figure / table / formula / section / chunk 五粒度）
3. 把 corpus 里的 `image_path` 从 MinerU 输出路径改写为 `images/{doc_id}/{hash}.jpg`，物理拷贝图片
4. 清理裸图（有图片但无 caption 无 description）、破损引用
5. 从 evidence span 回填缺失的 description
6. 建 triplet：每条 query 配 3-4 正例 + 5 同文档 hard_neg + 5 跨文档 random_neg
7. 负样本重平衡：强制保留 2-3 个文本类 slot，visual:text 比例从原始 ~72:28 压到 ~45:55

**交付格式**：

```
M4query_xxx/
├── corpus.jsonl.gz      # gzip 压缩的 passage 文件
├── train_triplets.jsonl  # 三元组
├── images/               # 所有被引用的图片
└── README.md
```

passage 格式：`{passage_id, type, text, caption, image_path, description}`
triplet 格式：`{query_id, query, positive_passages[], hard_negative_passages[], random_negative_passages[]}`

### 第 7 步：API 审计

所有 LLM 调用走 `src.utils.token_logger.log_run()`，写到 `api_logs_cannt_delete/calls/`，含 prompt / response / token 数 / latency。

---

## 已验证但未进入主力交付的能力

这四件事**有证据、有产物**，但都还只活在实验数据里，没参与过当前交付的 query 生成或三元组打包。它们是下一版交付要消化的素材。

### a. MinerU 替代 LaTeX 作为文档内拓扑骨架（C15）

- 从 MinerU 输出里 regex 抽 "Figure N" / "Table M" / "Eq. K" 这类显式引用
- 跟 LaTeX `\ref` 在 52 篇重叠文档上 A/B：**84% 召回**、26/52 篇 100% 召回、人工抽样 6/6 正确
- **现状**: 验证了"MinerU 可以替代 LaTeX"，但当前交付的拓扑还是 LaTeX 引用图；下一版要拿这个替换掉 LaTeX 依赖，把 corpus 从需要 LaTeX 源码扩到任意 PDF

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

**做法**: 两篇论文如果共享 ≥2 个有区分度的实体，就建实体桥；BFS 找 3-paper 2-bridge 的元素链。

**实体从哪来**：`scripts/build_entity_skeleton_xdoc.py`，纯规则，零 LLM 成本。用四组正则从论文的 caption + content + enriched_content 文本里抽：
- **方法名**：CNN / BERT / Transformer / fine-tuning / pre-trained 等
- **数据集名**：ImageNet / COCO / SQuAD / CelebA 等
- **指标名**：accuracy / F1 / BLEU / RMSE / demographic parity 等
- **公式变量**：`\theta` / `\lambda` / loss function / regularization 等

抽出来后过滤掉通用停用词（"model" / "method" / "result" / "data"），只保留 IDF ≥ 2.5 的有区分度实体。每个元素建好实体集合后，跨文档 pair 的实体重叠数 + Jaccard 相似度融合成 entity skeleton score，用于 rerank 跨文档候选边。

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

⭐⭐⭐ = 强建议申请；⭐⭐ = 可申请；⭐ = 内部 know-how。

### T1. ⭐⭐⭐ 把 LaTeX 引用图当弱标签，训 XGBoost 预测跨文档引用边（claim C18）

- 训练数据：4,028 对 LaTeX → MinerU 对齐的 citation pair
- 特征：只用 MinerU 解析后能算的字段。**`title_match`（论文标题出现在 chunk 文本里）单独占 88.1% 的特征重要性**
- 评估：按 source doc 分组的 5 折交叉验证（同篇 chunk 不会跨折叠泄露），**AUC 0.852 / F1 0.746，每折 top-50 precision 都是 1.0**
- 推理：在 1147 篇全集上跑出 53,435 条预测边，**75% 概率 ≥ 0.95**

### T2. ⭐⭐⭐ 公式相似度从 CLIP 切到数学专用 encoder（claim C17）

- 诊断：抽 200 条公式样本测 CLIP 文本 encoder，**标准差只有 0.027**（任何两条公式 CLIP 都觉得"差不多像"，threshold 切不开）
- 替换：换成 `math-similarity/Bert-MLM_arXiv-MP-class_arXiv`（768-d），**标准差涨到 0.172**，范围 0.036 ~ 0.977，可用
- 入图：写进 `build_mineru_vl_edges.py` 的 `--formula-backend math_similarity`，全量产出 4,331 条公式相似度边
- 后续：smoke50 上把 query 含公式 anchor 时路由到这个 encoder + RRF 融合，formula bucket R@10 +7.3pp

### T3. ⭐⭐⭐ Bridge 节点的两段式 fallback（让跨文档训练信号"装进" intra-doc 三元组）

跨文档 L3 推理链原本带一个"桥段落"（跨论文的概念衔接文本）。直接保留会让 corpus 多出一类不属于任何论文的 passage。我们的处理：

- **第一段：source paragraph 替换（7,471 条 / 86.9% 成功）**——在 corpus 里找到桥文本对应的真实 source paragraph 时，把 bridge 替换为 source paragraph + chunk，正例数 3 → 4
- **第二段：假 paragraph 保留（1,118 条 / 13.1% 失败）**——找不到 source，或 source 是 figure / table / formula 时，把桥文本本身改 `type=paragraph` 当合成节点入库

### T4. ⭐⭐ Chunk 聚合规则

- 按 section 边界 + 约 400 词软上限聚合 paragraph 成 chunk
- **不跨 section**、**不滑窗、不重叠**
- **每条 query 必附一个 chunk 正例兜底**：即使 figure / table 没召回，chunk 降级信号还在
- corpus 里 chunk 占 17.2%（v2 29,237 / 169,671）

### T5. ⭐⭐ 负样本三层重平衡

- **Hard negative 按来源分层**：同文档其他元素为主（分"同篇别张图"），少量跨文档话题相邻
- **Visual:text 从 72:28 调到 ~45:55**：避免 visual 过拟合
- **抽取概率**：75% 概率抽 3 个文本 slot，25% 概率抽 2 个

### T6. ⭐⭐ Turn-dependency QC 协议（多轮 session）

- **抹除测试**：把 Turn N-1 的回答抹掉 → 重问 Turn N → 如果 LLM 还能答对，session 作废
- **指代强制**：每个 Turn N ≥ 2 必须含至少 1 个指代表达式
- **Evidence 锁定**：style pass 只允许加 persona / 指代变形，element_ids 不动
- 当前：L3 链投影 60% 通过，entity-bridge 链投影 100% 通过（smoke）

### T7. ⭐⭐ Entity-bridge 链的 IDF 双阈值

- `min_idf = 2.5`：实体必须在 corpus 里 IDF ≥ 2.5
- `min_elem_overlap = 2`：两篇 paper 共享实体 ≥ 2 个
- `max_hops = 2`：BFS 限 2 跳
- 53 篇子集：83 对 entity-bridge pair，**25.3% 端到端 strong**（当前最强跨文档精度信号）

### T8. ⭐ HopWeaver 规则 QC 三件套

- 每个 hop 用不同文档 / 不能单文档桥接 / 因果链方向 premise → intermediate → conclusion
- 实现不是发明，内部工程 know-how

### T9. ⭐ Pass-only filter 通过率

- gpt-5.4 做 evidence-grounding judge
- 通过率 25-52%（取决于论文领域和 enrichment 覆盖率）

---

## 一句话总结

**一条管线，两批论文，两个交付包（合计 16,308 条 triplet）。核心是 LaTeX+MinerU 合并的引用图 → hub 多跳候选 → LLM 生成推理链 → 双层 QC → 多粒度 chunk aug 打包。图 rerank 在检索上带来 +7pp R@10。MinerU 替代 LaTeX、跨文档引用预测、公式 encoder、跨文档视觉召回四件事已验过但没进交付。下一步：把拓扑从 LaTeX 切到 MinerU（解锁任意 PDF），把跨文档做进 query 生成（解锁多文档推理），把华为域接进产线。**
