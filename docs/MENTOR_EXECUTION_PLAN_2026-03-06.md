# Mentor 要求执行方案（基于当前进度）

> 参考输入：`CLAUDE.md` 当前状态（LaTeX Topology v2 + Hub 候选已落地）与 `docs/DISCUSSION_LOG.md` 历史问题复盘。

## 0. 执行目标（两周）

- 把“单文档 LaTeX 图谱 + 轻量检索”打造成可稳定复现实验基线。
- 明确 Node 粒度、Hub 评分、轻量关键词召回、Query 与评测协议。
- 先把 **Retrieve 能力** 做扎实，再让生成/回答模块消费检索证据。

---

## 1. Node 定义细化（最高优先）

### 1.1 新的节点类型（必须落地到 schema）

- `section`：`\\section{}`
- `subsection`：`\\subsection{}`
- `paragraph`：正文段落（不再整段 section 打包）
- `figure`：图（caption + label + page_idx）
- `table`：表（caption + label + page_idx）
- `equation`：公式（display/block + label）

> 执行收敛（本周）：先只落地以上 6 类结构节点。
> `result_claim` / `intro_claim` 暂不作为独立节点；Week 2 先以 paragraph 标签方式试运行。

### 1.2 新的边类型（区分结构边与语义边）

- `contains`：section/subsection → paragraph/figure/table/equation
- `adjacent`：同层相邻 paragraph（保留 backbone，但降权）
- `refers_to`：paragraph → figure/table/equation（由 `\\ref{}` / 文本模式触发）

> 执行收敛（本周）：先落地 `contains/adjacent/refers_to`。`supports/motivates` 等语义边留到 Week 2 误差分析后。

### 1.3 验收标准

- 随机抽样 10 篇文档：
  - 每篇至少能看到 section/subsection 层级节点。
  - figure/table/equation 的 label 解析召回率 ≥ 90%。
  - paragraph 不再“整节打包”。

---

## 2. Hub（桥接节点）评分体系

## 2.1 评分函数（先规则版，后学习版）

定义节点 `v` 的桥接分：

`HubScore(v) = 0.40 * BridgeRole + 0.35 * EdgeConnectivity + 0.25 * CoreModuleCoverage`

- `BridgeRole`（0-1）：
  - 是否连接至少两类模态（text/figure/table/equation）
  - 是否位于 section→evidence→claim 的中间路径
- `EdgeConnectivity`（0-1）：
  - 归一化度数（入度+出度）
  - 跨类型边数量（`refers_to`/`supports` 优先）
- `CoreModuleCoverage`（0-1）：
  - 是否命中核心模块关键词（模型结构、实验主结果、消融、限制）

### 2.2 防止“authority sink”

- 对纯高入度但无桥接功能节点加惩罚项：
  - `Penalty = 0.2` if `in_degree >> out_degree` 且仅单模态连接。
- Top-N hub 必须满足：
  - 至少 2 种模态 + 至少 1 条 `refers_to/supports` 边。

### 2.3 验收指标

- Top-60 hub 中 bridge 类占比 ≥ 85%。
- 用 hub 作为种子生成候选路径时，2-hop/3-hop 可解释路径占比 ≥ 70%。

---

## 3. 轻量级关键信息筛选（正则优先）

## 3.1 关键词规则池（先中文/英文通用）

- `MODEL_FIG`: `architecture|framework|overview|pipeline|model diagram|Figure\\s*1`
- `INTRO_MOTIVATION`: `we propose|motivation|challenge|problem setting|our contribution`
- `MAIN_RESULTS`: `main result|state-of-the-art|outperform|improvement|Table\\s*[0-9]+`
- `ABLATION`: `ablation|w/o|without|sensitivity`
- `LIMITATION`: `limitation|failure case|future work`

## 3.2 加权策略

- 命中 `MODEL_FIG` 的 figure/table 节点：+3
- 命中 `INTRO_MOTIVATION` 的段落：+2
- 命中 `MAIN_RESULTS` 的段落/表格：+3
- 句子同时命中“结果+对比词（better than / compared with）”：额外 +2

## 3.3 触发 LLM 的条件（严格收敛）

- 仅在以下情况调用大模型：
  - 规则冲突（同一段命中多个互斥标签）
  - 召回不足（Top-K 证据缺口）
  - 人工抽检失败样本复判

---

## 4. Query 与检索逻辑：从 Answer 导向转为 Evidence 导向

## 4.1 目标重定义

- Query 系统的主 KPI：**证据召回率**，不是“直接答对复杂问题”。

## 4.2 检索评测指标（必须上线）

- `Recall@K (evidence-node)`：目标证据节点是否进入 Top-K。
- `Path Hit@K`：是否召回包含“动机→方法图→主结果”的路径。
- `Evidence Diversity`：返回证据涉及的模态种类数。
- `Latency`：单 query 检索时延（规则版目标 < 200ms/文档，离线可放宽）。

## 4.3 Query 集合重建（贴近真实用户）

只保留短问句、多样化模板，每类至少 20 条：

- 结构认知：
  - “这个模型架构是什么？”
  - “图 1 主要展示了什么？”
- 动机理解：
  - “这篇文章要解决什么问题？”
- 结果总结：
  - “主实验结论是什么？”
  - “相比基线提升了多少？”
- 主客观混合：
  - “总结一下这篇文章。”
  - “作者最重要的贡献是什么？”

每个 query 需要绑定“期望节点群”（gold node set），形成评测闭环。

---

## 5. 单文档优先 + 多跳预留架构

## 5.1 当前里程碑（M1）

- 输入：单篇 LaTeX + MinerU 元素。
- 输出：文档内图谱 + evidence 检索接口。
- 暂不追求跨文档召回最优，只保留跨文档接口定义。

## 5.2 预留点（不立即重投入）

- 节点 ID 设计：`doc_id:node_id`，天然支持 future cross-doc merge。
- Query API 预留 `conversation_state` 与 `history_nodes`，支持后续多轮追问。
- Path search 模块保留 `max_hops` 与 `cross_doc` 开关。

---

## 6. 两周执行排期（建议）

### Week 1：打牢底座

1. Node/Edge schema 重构与解析器改造。
2. 正则规则池 + 核心模块加权上线。
3. HubScore v1 实现与 authority sink 惩罚。
4. 先建立 30 条“真实短 query + gold 证据节点”小基准（baseline）。

### Week 2：检索闭环与误差分析

1. 先在 30 条基准上跑 Recall@K / Path Hit@K / Diversity / Latency。
2. 对失败样本做 error taxonomy（漏召回、错桥接、节点粒度错误）。
3. 仅对失败簇做最小增量修复（规则 or 图边补充）。
4. 将基准扩展到 100 条并复跑完整评测，产出 v1.1 报告。
5. 决策是否引入 `result_claim` / `intro_claim` 独立节点并进入跨文档阶段。

---

## 7. 风险与止损

- **风险 1：节点类型激增导致噪声上升**
  - 止损：先限制在 section/subsection/paragraph/figure/table/equation 六类核心节点。
- **风险 2：规则过拟合某些论文写法**
  - 止损：按子领域抽样（CV/NLP/ML）做规则泛化测试。
- **风险 3：Hub 分数被“高频结果段”劫持**
  - 止损：强制 bridge 条件门槛，不满足则不得进入 Top hub。

---

## 8. 建议的“完成定义（DoD）”

满足以下条件才进入下一阶段（跨文档/多跳增强）：

1. 单文档 Node 粒度抽检通过率 ≥ 90%。
2. 关键 query 集上 Recall@10 ≥ 60%（v1 基线门槛，后续逐步提升）。
3. Top hub 中 bridge 合格率 ≥ 85%。
4. 至少 80% query 能返回“多节点证据包”（而非单段文本）。
