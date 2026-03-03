# M4 数据工程进度汇报（对齐版）

**日期**：2026-03-03  
**汇报周期**：2026-02-25 -> 2026-03-03  
**本次重点**：LaTeX Graph 拓扑强化、Query 人类化改造、架构图专项修复  
**说明**：按上次沟通，本轮不包含 MinerU 服务化部署任务

---

## 0. 先说结论（3 分钟）

这周我主要做了三件事，并且都已经有可核验产物：

1. 我把 LaTeX 引用图从“同段共引”升级成了“backbone + bridge-first + cross-doc”的拓扑图，能够量化连接密度、识别交通枢纽、并生成 500 条多跳候选。
2. 我按您建议把 Query 从“偏长偏模板化”改成“短长混合”，已经跑出 v4.4 run1：252 条里 113 条通过，pass 集中 short/long 基本平衡。
3. 架构图问答质量还没有达标，目前是本轮最大短板；hard negatives 我先保持可选策略，不抢正向 Query 质量优先级。

目前阶段性判断是：
- 不是“做没做出来”的问题，而是“质量稳定性还不够”的问题。
- 瓶颈已经从环境可用性转移到数据质量控制（长度混合、架构图意图、过长 Query）。

---

## 1. 对齐上次建议：我做了什么

| 上次建议 | 本周执行 | 当前结果 |
|---|---|---|
| 看 Graph 连接密度，不只看同段引用 | 新增 backbone 边、cross-doc 边、全图统计 | 2551 nodes / 3471 edges，backbone 1269，cross-doc 434 |
| 找“交通枢纽”节点做更复杂构链 | 改 hub 评分为 bridge-first，导出 hub 列表 | 60 个 bridge hubs，adjacent bridges 369 |
| 看跨页距离方差，避免全是临近 | 补 real page_idx，统计 page gap / variance | 候选中 95/500 有 page_span，page variance 已可算 |
| Query 要贴近人类，长短都要有 | v4.4 加短长硬约束与 QC 项 | run1(pass) 中 short 59 / long 54 |
| 架构图质量要专项处理 | 增加 architecture case 检测与 intent 约束 | 68 条架构图样本，pass 23（仍是短板） |
| hard negative 先别过早优化 | 仅做可选开关，不改主流程优先级 | 正向质量优先策略保持不变 |

---

## 2. LaTeX Graph 拓扑：从“配对”到“结构化网络”

### 2.1 图结构与连接密度（已落地）

我现在用的图不是只看同段引用，而是把“阅读顺序 backbone + 引用边 + 跨文档引用边”统一到一个图里：

- Nodes: 2551
  - paragraph: 1347
  - figure: 532
  - table: 221
  - formula: 451
- Edges: 3471
  - paragraph_ref: 1688
  - backbone: 1269
  - cross_doc_cite: 434
  - element_ref: 80
- 全局有向密度：0.00053359

这一步对应解决的是“图是不是只有顺序边、有没有结构 richness”的问题。

### 2.2 交通枢纽识别（bridge-first）

我把 hub 识别从“高入度节点优先”改成“桥接能力优先”，避免被 authority sink 节点带偏。

当前结果：
- bridge_hubs: 60
- adjacent_backbone_bridges: 369

这个结果可以直接支持后续多跳构链起点选择，而不是随机起点。

### 2.3 多跳候选构造（500 条）

候选分布：
- 总数：500
- 2-hop: 181
- 3-hop: 319
- 跨文档：170（34%）
- 模态组合：
  - figure+formula: 247
  - figure+table: 153
  - formula+table: 100
- source docs 覆盖：47

对比之前 DFS 方案“长链里迷路”的问题，这版已能稳定产出可用规模。

具体例子（跨文档 2-hop）：
- 路径：`1904.03310::el::tab:lm_cor -> 1904.03310::p::00001 -> 1707.09457::el::fig:cooking`
- 对应短问句种子：`Under what conditions does tab:lm_cor predict fig:cooking (1707.09457)?`
- 这个例子体现的是：以段落 hub 为中介，把“本篇表格证据”桥接到“被引论文图证据”，而不是只在同一段里拼接两个元素。

### 2.4 物理距离方差（已可计算，但覆盖仍偏低）

我补了 real page_idx 后，已经可以算 page gap variance：
- with_page_span: 95 / 500（19%）
- with_line_no_span: 500 / 500（100%）
- pair_gap_global: mean 11.65，variance 115.76

目前问题是 page 级覆盖仍低，核心原因是 label 对齐率限制（不是算法没实现）。

---

## 3. Query 构造：v4.4 run1 实际结果

### 3.1 产物与总体通过率

- `l1_dual_evidence_queries_v4_4_run1.jsonl`: 252
- `l1_dual_evidence_queries_v4_4_run1_pass.jsonl`: 113
- pass rate: 44.8%

### 3.2 长短句混合（人类提问习惯）

all 分布：
- short: 104
- long: 87
- medium: 19
- too_long: 42

pass 分布：
- short: 59
- long: 54

说明：短长并存目标已经落地，但“过长句控制”仍需再压。

具体例子（同一 pair 的短/长对照）：
- pair：`1306.5204_pair_1`（figure+table）
- 短句：`Which city-level terms dominate the Streaming API cloud consistent with the keyword filter set?`
- 长句：`Beyond the keyword list, what spatial constraint in the collection parameters supports the prominence of both Arabic and English city-name terms in the Streaming API tag cloud?`
- 这个对照体现的是：短句偏“快速检索入口”，长句偏“机制追问”，两类都保留。

### 3.3 架构图专项（当前短板）

- architecture 样本：68
- architecture pass：23（33.8%）

失败主因（architecture 子集）：
- `architecture_intent_missing`: 29
- `length_mix_missing`: 22
- `query_too_long`: 9

这说明“架构图问题怎么问得像真实研究者”这个点，我已经有专项约束，但约束强度和稳定性不够。

失败例子（架构图意图缺失）：
- `query_id`: `l1_de_1611.07438_0050`
- 问句：`Why do gender-stratified admission rates diverge despite identical stratum-level rates?`
- `qc_issues`: `architecture_intent_missing`
- 问题本质：句子本身可读，但没有明确落在“结构总结/关键组件机制/公式-模块联动/实验效应解释”这些架构图意图槽位里，导致被判失败。

### 3.4 按 pair_type 的当前通过率

- figure+table：74 / 178（41.6%）
- figure+formula：21 / 44（47.7%）
- formula+table：18 / 30（60.0%）

### 3.5 口径说明（避免误读）

- 2026-02-24 的官方批次（222 条，173 pass）与本次 run1 不是同一批候选来源，不能只看 pass rate 做横向结论。
- 本次 run1 的意义主要是验证 v4.4 的长度控制和架构图专项约束是否真正落地。
- 拓扑 v2 新产生的 500 条 hub 候选，下一轮会作为主输入做独立对比批次。

---

## 4. 我对当前状态的判断

### 4.1 已经完成的“能力建设”

1. 拓扑层面：连接密度、交通枢纽、物理距离方差都已具备可重复计算能力。  
2. 生成层面：短长句混合已经不再停留在 prompt 描述，run1 有真实产物。  
3. 流程层面：从候选到 Query 的链路已打通并可复跑。

### 4.2 仍需解决的核心缺口

1. `length_mix_missing` 和 `query_too_long` 数量偏高，长度控制不稳定。  
2. 架构图问题意图仍偏泛化，学者视角问题模板还不够强。  
3. page-level 覆盖率 19%，跨页 variance 的统计样本仍偏少。
4. topology v2 的 500 候选还没有完整接入 run1 主干，需要下一轮做同口径复跑。

---

## 5. 下周计划（按优先级）

### P0：先把 run1 的主要失败项打下来

目标：`length_mix_missing` 和 `query_too_long` 各下降 50% 以上。  
做法：
- 对每个 pair 增加“短句/长句成对校验 + 失败重试”
- 将 too_long 作为硬失败前置，先重写再进入后续 QC

### P1：架构图专项 v2

目标：architecture pass 从 33.8% 提到 50%+。  
做法：
- 将“结构总结/创新点/组件-公式机制/实验效应”拆成显式问法槽位
- 架构图 case 采用更严格的 intent checker 与拒收规则

### P1.5：提升跨页统计覆盖

目标：page_span 覆盖从 19% 提升到 30%+。  
做法：
- 继续修 label 对齐 fallback
- 对未对齐样本补 position-level 代理统计

### P2：hard negatives 进入下一阶段

保持“正向先稳”的前提下，逐步启用 related-random negatives 做增量实验，不在本周抢主线资源。

---

## 6. 我希望本周会后确认的 3 个决策

1. 下周主目标是优先“提升 architecture 质量”还是“扩大全文档覆盖率”？  
2. v4.4 下一轮是否继续保留 34% 跨文档候选比例，还是先降跨文档比重换稳定通过率？  
3. hard negatives 是否继续维持“可选开关”，等正向 pass 稳定后再转主线？

---
