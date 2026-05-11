# 周报 5/10 — 数据图组（DPT）

汇报人：你（草稿，自己审一遍再发）
覆盖区间：2026-05-02 周 → 2026-05-10
上次对应：5/2 mentor 录音 60 那次会议（你给的 4 条 to-do + 18 条录音 todo）

---

## 零、上次 4 条 to-do 的总览

| # | To-do | 状态 | 一句话结论 |
|---|---|---|---|
| 1 | 修改 enrich 方式，模仿 summary+细节 格式参与召回 | ❌ 证伪 + API blocked | 5/3 跑了 corpus enrich fix（拿已经存在的 MODORA 视觉描述做 summary+细节）整体 R@10 跌 3.1pp，落地为 `claim:C8`；做加强版（多粒度 enrich）的公司 API 自 4/21 已 401 共 19 天 |
| 2 | 用 VL-Embedding 让图片参与 dense retrieval | ⚠️ 跑两轮全负 + 发现真正的 bug | 5/2 第一轮挂在模型 checkpoint 缺 language head；5/3 修好环境重跑还是输给纯文本 4B；5/10 复盘时挖到 **table 图片在 corpus 构建阶段被丢光**，VL 上 table R@10 归零不是模型问题 |
| 3 | 统计 chunk 平均含几个 element、查 R@10 低的根因 | ✅ 完成并升级为 claim | 964 chunk 平均 1.94 element / 52% chunk 只有 1 个 element / 75% 双证据 query 跨 chunk / K=1 时 71% query 一个证据都召不到 → `claim:C9` |
| 4 | 原有 md 文档更新，说人话 | ✅ 完成 | `文档建图.md` 5/10 重写：术语统一、图加权改大白话、第十节重名修复、第十一节加 modality scope |

完成率 **2/4**：#3、#4 全做完；#1 路线本身被证伪 + 加强方案卡 API；#2 找到真正根因但补丁还没跑。

---

## 一、按 to-do 详述（5/02 → 5/10 的时间线）

### To-do #3 — Chunk 含几个 element / R@10 低的根因 ✅

**5/2 上午**：写了 `scripts/analyze_chunk_element_coverage.py`，在 M4query_v1 的 57 篇 gold docs 上跑。

| 指标 | 数值 |
|---|---|
| 总 chunk 数 | 964 |
| 平均 element / chunk | 1.94（中位 2） |
| 只含 1 个 element 的 chunk | 501 / 964（52.0%） |
| 双证据落在**同一**个 chunk 的 query | 10 / 473（2.1%） |
| 双证据落在**不同** chunk 的 query | 357 / 473（75.5%） |

**5/3**：在此基础上做 per-query chunk→element recall，发现 K=1 时 71% query 一个 evidence 都召不到（实验 `20260503_chunk_query_element_recall`）。

**结论已写成 claim:C9**：chunk 作为检索单元会**稀释**双证据信号。检索 target 从 1798 个 element 缩到 964 个 chunk，但 75% query 又需要命中两个不同 chunk 才完整，R@K 自然吃亏。
落点是：**element 当检索单元，chunk 留给下游 QA 消费**，不要再回头硬推 chunk-as-retrieval。

### To-do #2 — VL-Embedding 让图片参与 dense retrieval ❌→bug 已定位

这条线跑了三轮，每轮死法不一样：

**5/2 第一轮（Job 66048 文本 baseline + 66114 VL 第一版）**
- 提交 split_modality 评测（4B 纯文本 baseline R@10 = 0.4767）
- VL 版 Qwen3-VL-Embedding-2B 跑出 R@10 = **0.0021**（基本是随机）
- 根因当时误诊为"split allocation 对 table 不适合"
- 真实根因（5/3 复盘）：HF 上的 checkpoint **只发了 vision encoder，language head 28 层是随机初始化的**，启动 warning 一堆 newly initialized，整个 query+text 编码就是噪声

**5/3 第二轮（Job 66243→66244→66248）**
- 用 transformers 5 overlay 把环境装对，weight load 干净（625/625），无 newly initialized
- 跑出来 split_VL_2B 仍然不如 4B 纯文本：figure VL 0.54 vs 4B 0.71，table VL 0.02 vs 4B 0.50，formula VL 0.34 vs 4B 0.30
- 含义：VL-2B 在 figure/table 上**全面输**给纯文本 4B；只有 formula 上有 4pp 微弱优势
- 当时归因到"模型本身在 text→multimodal retrieval 上能力弱"

**5/10 复盘挖到真正的 bug**

| 层级 | figure 带图比例 | table 带图比例 |
|---|---:|---:|
| mineru 原始输出 | 100% | **100%**（163/163 抽样） |
| `multimodal_elements.json` 图层 | 841/841 | 237/334（其余 97 是 inline HTML，本来就无图） |
| `corpus_v1_enriched.jsonl` **检索 corpus** | 842/842 ✅ | **0/2 ❌** |

也就是说：**mineru 给的 table crop 图片在 corpus 构建那一步几乎全丢**——1798 条 passage 里只剩 2 条 table 类型，且都不带 `image_path`。VL 脚本里 `resolve_image_path(table) → None → 走文本编码 → table R@10 = 0`。

5/3 那篇归因写成"split allocation 不适合 table"是**错的**。真正修法不是改 VL 脚本，而是改 corpus 构建：把 237 个 table-with-image 显式作为独立 passage 注入 corpus + 带 `image_path` 字段。修完再给 VL 一次机会，table lane 才有公平评测的可能。

**下周排进去**：corpus 修复 + VL 重跑，大约 1 天 + 30 min A6000。

### To-do #1 — Summary+细节式 enrich 参与召回 ❌（路线证伪）+ ⚠️（加强版卡 API）

**5/3 跑了一版用现有 MODORA 视觉描述当 summary 注入 corpus**
- 诊断阶段先抓出两个独立 bug：① `load_enriched_index` 只认嵌套 layout，flat 的 `element_a_id` 一个都没读到；② `build_element_text` 优先级写成 OR，MODORA 在的时候直接跳过 graph caption/context 分支
- 修完后做了两个变体：v1 replace（mean fig len 405）、v2 additive（visual + paper context，mean fig len 683）
- 结果对比 anchor（rebuilt_20260417）：

| 指标 | anchor | fix_v1 | fix_v2 |
|---|---|---|---|
| dense R@10 | 0.6195 | 0.5106（**−10.9pp**） | 0.5888（−3.1pp） |
| dense R@100 | 0.8636 | 0.7569 | 0.8436 |
| graph explicit_only R@10 | 0.6913 | 0.5888 | 0.6860（−0.5pp） |

**结论：净负，不 promote**。机制上跟 BGE-CE 那次的 text-bias 是同一方向——MODORA 给的视觉描述是 domain-detached 的（"Histogram of small-valued metric"），而 M4query_v1 的 query 用的是论文领域语言，把视觉描述塞进 passage 反而冲淡了原本能匹上的领域词面。

这条结论**直接落地为 `claim:C8`**：MODORA-style 视觉 enrich 对 text-style retrieval 是净负。

**加强方案（多粒度 enrich，DocResearcher 风格的真正"summary + 细节"）的状态**
- 这是 mentor 录音 60 里的 C5 todo，思路是给同一个 element 同时生成"概要句 + 细节句"两段，分别参与 dense 召回
- 跟 C8 不冲突（C8 是说视觉描述硬塞会负向；多粒度是另一种结构），但**前提是公司 API 通**
- 公司 API endpoint `az.gptplus5.com` 从 4/21 401 至今 19 天，B4 全量 enrich 卡在 10988 / 27209 = 40.4%，C5 这条加强线也跟着挂着
- OpenAI 直连可以 fallback 但成本约 **10x**，**等你拍板是否切**

### To-do #4 — 文档建图.md 更新 ✅

按你 5/2 给的版本改：

- 元素（element）：text / figure / table / formula 四种平级，inline 公式不算独立元素
- 切片（chunk）：合并连续文本元素得到的检索单元
- 图加权用大白话写：出入度、阅读顺序分数传播、引用边分数传播、汇总加权
- 投影边：元素层引用边映射到切片层
- 第十节标题之前跟第七节重复了，改掉
- 第十一节 verdict 加了 modality 选择性的说明（对应 C10）

---

## 二、这周顺便做掉的事（不在 4 to-do 里但有意义）

### 1. F-formula 三阶段 — 公式检索瓶颈被推开一道口子（这周最大的技术信号）

公式题的 R@10 在 10 个之前的配置里都卡在 0.56，这周连开三发：

**Phase 1 (5/10 13:30, Job 68107) — caption injection**
- 给 formula passage 前面拼 300 chars 的 NL `context_before`
- 结果：R@10 仍是 0.5600。机制上文本上下文确实是 NL 但没解决 LaTeX 本体编码差的问题
- 结论：caption 不是杠杆 → 强化 C11（formula ceiling 是 dense encoder bound）

**Phase 2 (5/10 14:54, Job 68131) — LaTeX 表面归一化**
- `\operatorname` → opt，`\leq` → <=，`\mathbb{E}` → E，`\frac{a}{b}` → (a)/(b)
- 结果：R@10 = 0.5600（一动没动）
- 结论：surface form 假说被杀掉。Qwen3-Embedding-4B 对 LaTeX 不是"长得不顺所以编不进"，是**真编不了**

**Phase 3 (5/10 17:15, Job 68281) — Qwen2.5-Math-7B routing**
- 思路：换一个**专门在 LaTeX 源码上训练过**的数学模型，单独编码公式段落，再跟原 Qwen3-Embedding-4B 的排名用倒数排名融合（k=60）

| 指标 | 之前（10 配置） | 这次（k=60） | Δ |
|---|---|---|---|
| 公式题 R@10（179 道，全集） | 0.5600 | **0.6313** | **+7.3pp** |
| 公式题 R@10（25 道，平衡冒烟集） | 0.5600 | 0.6000 | +4.0pp |
| 公式题 R@100（全集） | 0.8636 | 0.9441 | +8.0pp |

**4/17 之后第一个真正推动公式天花板的信号**。原来"模型本身就编码不了 LaTeX"这个结论被部分推翻——只是 4B 一个家族编不了。

**代价**：整体 R@10 从 dense 0.6195 跌到 0.5222（−9.7pp）。原因是融合权重做得太简单，公式段落同时拿到 Qwen3 和 Math 两票，图/表只有 Qwen3 一票，所以图/表的相对位置被压下来了。

**下周怎么修**：只在 query 看起来是公式题时才触发 Math 编码器（query 含数学符号 / 公式关键词）。图/表 query 走原来的路径不动，公式 query 走融合路径。

### 2. 50 道平衡冒烟测试集（mentor 录音 60 的 C6 todo）

按 mentor 要求"10 文本 / 10 图 / 10 表 / 10 公式"做。当前 query 集合里没有纯文本题，实际比例 17 fig / 17 tab / 16 formula。

结论：之前 0.6913 那个上限**不是数据偏置造出来的**（不是因为 figure 占 56% 才高），冒烟集上图增益 +1.87pp 在误差范围内，上限是真的。

但有一个重要的修正：**图增益不均匀** —— fig +10.3pp / tab +8.3pp / **formula 0pp**。意味着论文里 C1 / C5 / C7 三个 claim 都要明确写"对图、表有效；对公式无效"，不能笼统说"图增益"。这条落地为 `claim:C10`（graph rerank modality selective）。

### 3. mineru ↔ LaTeX 元素级匹配率审计（B2）

之前一直只有"大概 50%"和"92%"两个说法，没具体数字。手动核完：

- figure: 49.7% 元素级匹配
- table: 67.3% 元素级匹配
- formula: **0%** 元素级匹配

92% 是文档级覆盖率（56/57 = 98.2%），跟元素级是两回事。

**这件事解释了 #2 里 formula 为什么图增益归零**：mineru 给的 formula label 跟 LaTeX 源里的 label 完全没打通，图上根本没有边连到 formula 节点上去，图信号传不进来。

### 4. 用 LaTeX 行号重建 chunk-element 边（B1）

按 mentor 要求把字符串模糊匹配换成 LaTeX 行号。
- Phase 1 用内容 jaccard 绕过 label key 通道，把 formula 元素匹配从 0% 救到 41.2%
- Phase 2 重建 chunk-element 边，新边和老边几乎完全不重合（修了 1130 条，删了 529 条）

**实测**：拓扑改了，但 explicit-only 这个图配置下 R@10 完全不动。原因是这个配置走桥接节点（hub）这条路，不经过 chunk-element 这条边。

**结论**：B1 的工程价值在 QA / SFT 数据合成那边（对齐准了，证据定位不容易出错），不在 retrieval ceiling。retrieval ceiling 卡在公式编码上（已被 F-formula Phase 3 部分推开）。

### 5. claim 入库

本周新增 4 个 claim 文件：

- `C8`：MODORA 视觉 enrich 对 text-style retrieval 净负（对应 to-do #1 的证伪）
- `C9`：chunk 检索稀释双证据信号（对应 to-do #3）
- `C10`：graph rerank 的增益是 modality selective（fig/tab 有，formula 没有）
- `C11`：formula ceiling 是 dense encoder bound（被 F-formula Phase 3 部分推翻）

### 6. BCD 阶段执行整体完成度从 32% 推到 76%（mentor 录音 60 的 14 项 BCD todo）

---

## 三、卡住的事

### B4 全量 element enrich
公司 API endpoint `az.gptplus5.com` 自 4/21 起 401，19 天没恢复。当前覆盖 10988 / 27209 = 40.4%，剩下的元素一个都加不了。

### C5 多粒度 enrich（DocResearcher 风格的真正 summary+细节）
同样卡 API。这条线就是 to-do #1 的加强方案——但要注意它跟 `claim:C8` 有冲突，**API 通了也不能直接 plug-and-play**，需要先重新设计：summary 内容必须用论文领域语言（而非 MODORA 那种 domain-detached 描述），否则会重蹈 C8 覆辙。

---

## 四、下周打算（5/13–5/17）

| 优先级 | 事 | 估时 |
|---|---|---|
| P0 | F-formula Phase 2a：query 感知路由（query 含数学符号才触发 Math 编码器，把 −9.7pp 整体回归修回去） | 1-2 天 + ~30 min A6000 |
| P0 | **corpus 端补 table-with-image passage**，重跑 VL split（对应 to-do #2 的真补丁） | 1 天 + 30 min A6000 |
| P0 备用 | F-formula Phase 2b：两阶段——dense 取 top-100 后只对公式候选重排 | 同上 |
| P1 | 给 C1 / C5 / C7 论文 claim 加 modality scope（配合 C10） | 半天 |
| P2 | 拿 jina-embeddings-v3 跑一次独立家族对照（排除"是不是 Qwen 系列偏置"的疑问） | 半天 + ~20 min A6000 |

---

## 五、要你拍板的事

1. **API 切不切 OpenAI 直连**？切了 B4（全量 enrich）和 to-do #1 加强方案（C5 多粒度 enrich）才能动；不切的话这两条线继续挂着。成本约公司 API 的 10x。
2. **F-formula 整体 R@10 −9.7pp 的代价能不能接受**？我倾向 Phase 2a 把它修回去（保住整体的同时留住公式 +7.3pp 增益）。如果你觉得"公式增益本身就够发论文"，那 Phase 2 可以省掉直接收工。
3. **VL split 还要不要继续推（to-do #2）**？现在已知 corpus 端有 bug 把 table 图丢了。修完 corpus 后值得再给 VL 一次机会，但 smoke50 上 figure VL 0.33 vs graph 0.82 这个 49pp 的差距摆在那，**就算 table 修好，VL 路线大概率还是输给 graph rerank**。我倾向"修一次 + 跑一次"作为收尾，结论清楚后归档，不再继续推。
4. **to-do #1 这条线整体方向调整**：MODORA 视觉式 summary 已经被 C8 证伪；C5 多粒度 enrich 是另一条路，但需要先在 prompt 设计上避开"domain-detached 描述"的坑。要不要等 API 通了后**先做 50 条小冒烟**验证方向再大规模铺？
5. **下次同步节奏**：还按 mentor 录音 60 之前的两周一次，还是改成每周？

---

## 附：所有材料的位置

- 实验记录：`research-wiki/experiments/20260502_*`、`20260503_*`、`20260505_*`、`20260510_*`
- 决策报告：`refine-logs/*_20260503.md`、`refine-logs/*_20260510.md`
- 数据 / ranking / qrels：`data/05_eval/dense_retrieval/qwen25math_formula_routing/`
- claim 定义：`research-wiki/claims/C{8,9,10,11}_*.md`
- BCD 主计划：`refine-logs/BCD_PHASED_PLAN_20260510.md`
- 文档更新：`文档建图.md`

本周 GitHub commits：`fa29c84`（5/10 上午 BCD 执行）→ `cbd3a01` → `4d859d0` → `904189d`（5/10 傍晚 F-formula 结果）。
