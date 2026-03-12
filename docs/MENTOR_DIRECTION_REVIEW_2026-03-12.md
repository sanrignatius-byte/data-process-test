# Mentor 方向讨论评估（2026-03-12）

## 总评
这次更新质量高，方向从“做 query”升级为“做可泛化的 Document Graph”，并给出了明确时间线（4 月专利、5 月论文）。从研究叙事和执行管理两个角度看，属于一次有效的战略收敛。

## 做得好的地方
1. **核心贡献重新聚焦**：把 graph 定位为主产品、query 降级为首个应用，避免“以产物代替方法”的叙事风险。
2. **里程碑明确**：有月度目标和阶段出口（专利→论文），便于周会管理和资源排期。
3. **路线扩展合理**：Persona Hub / C-Pool / Graph RAG / 泛化方案并行纳入 roadmap，为后续实验提供多抓手。
4. **文档治理进步**：明确要求图架构独立文档，减少历史上“信息散落在大日志里”的协作摩擦。

## 仍需补齐的关键缺口
1. **成功标准仍偏叙事化**：需要把“验证效果”细化成可量化 KPI（如 QA F1、evidence localization Recall@K、构图成本/doc）。
2. **多方向并行存在分散风险**：新方向多，若没有 P0/P1 闸门，容易再次陷入“同时推进很多事但闭环慢”。
3. **query 作为副产物的定义需更硬**：应明确“query 只是验证 graph 的 probe”，并固定最小评估集，避免重新回到 prompt 工程导向。

## 修正后执行方案（按优先级，直接开工）

### Phase 0（本周）：先把“可验证闭环”立起来
1. **冻结一个最小可评估版本（MVE）**
   - 冻结对象不是仅边类型，而是：
     - 图结构：`backbone + element_ref + paragraph_ref + cross_doc_cite`
     - **Hub 节点选取/排序公式**：`bridge_score + PageRank`（双锁定）
   - 任何变更都必须进入 `docs/EVAL_CHANGELOG.md`。

2. **发布《Evaluation Protocol v1》并锁口径**
   - 任务：`evidence localization`（主）+ `QA`（辅）。
   - Ground-truth 来源（固定）：
     - `data/l1_dual_evidence_queries_v4_4_run1_pass.jsonl`（113）
     - `data/l1_dual_evidence_queries_v3_pass.jsonl`（152）
     - 去重后并集作为评估测试集。
   - Localization 判定：retrieved chunk 与 `required_evidence_spans` **字符级 overlap ≥ 0.5**。
   - Baseline：BM25 + dense retrieval（同一 chunking 口径）。

3. **决策阈值写死（防止 p-hacking）**

| 决策 | 条件 |
|------|------|
| 继续扩量 | Graph Recall@10 ≥ BM25 + **5%**，或 MRR ≥ BM25 + **3%** |
| 暂停扩量并回查 | 任一指标未达阈值 |
| 放弃当前 graph 配置 | 连续 2 轮实验均未达阈值 |

### Phase 1（1-2 周）：交付专利需要的“方法 + 证据”
1. **提交图架构文档到仓库（可追踪）**
   - 文件：`docs/GRAPH_ARCHITECTURE.md`（必须 commit）。
   - 必含：节点/边定义、构建流程、复杂度、成本分层、Hub 评分公式。

2. **跑第一轮 A/B 实验（Graph vs Baseline）**
   - 固定使用 Phase 0 的测试集（禁止候选集/测试集混用）。
   - 同一评估脚本、同一统计口径。
   - 输出：总体 + 分类型（single-hop / multi-hop、intra-doc / cross-doc）结果表。

3. **沉淀失败案例集（至少 20 条，统一模板）**
   - 标注模板（JSON）：

```json
{
  "query_id": "...",
  "failure_type": "bridge_error | evidence_gap | retrieval_bias | parse_noise",
  "failure_description": "一句话说清楚",
  "fix_action": "改图结构 | 改QC | 改prompt | 改解析",
  "priority": "P0 | P1 | ignore"
}
```

### Phase 2（2-4 周）：在不破坏主线前提下扩展新方向
1. **C-Pool 小规模上线（先 50 条）**
   - 仅做 evidence localization，不做 query 质量评分。
   - 作为通用能力探针，不干扰 academic 多跳主评估集。

2. **Persona Hub 先做轻量版本**
   - 先支持 2 类 persona（PhD + practitioner），观察检索分布变化。
   - 达标后再扩展到 5 类。

3. **Graph RAG 调研以“对比框架”输出，不直接改主 pipeline**
   - 先产出对照表（成本、可扩展性、可解释性、适配当前数据程度）。
   - 只有当评估证明收益显著，才进入主干。

4. **新增：纯 PDF 场景泛化方案设计（专利覆盖宽度）**
   - 文件：`docs/GRAPH_GENERALIZATION.md`。
   - 内容：无 LaTeX 时的降级路径（MinerU 可得边、LLM 补全边、成本估算）。
   - 当前阶段只设计文档，不强制实现。

## 执行治理（防跑偏）
1. **单周只允许 1 个 P0 目标**：其余全部降级为 P1/P2。
2. **口径变更强制记录**：所有影响评估口径的改动，必须写入 `docs/EVAL_CHANGELOG.md` 并附版本 tag。
3. **周会固定三页材料**：
   - Page 1：本周指标（vs baseline；若首周无图结果则先给 baseline）
   - Page 2：失败案例与修复计划
   - Page 3：下周唯一 P0

## 一句话结论
方向已经对了，接下来不要再“加想法”，而是用固定测试集 + 固定阈值把“图确实更好”打成硬证据；先赢下 4 月专利验证，再扩展 Persona/C-Pool/Graph RAG。
