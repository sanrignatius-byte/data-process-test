# 进度总结（自上次汇报以来）

日期：2026-02-24
范围：dual-evidence L1 query 生成流程 → triplet 构建 → 跨文档 embedding 匹配 → utility-aware rerank

---

## 0) 与前期 L1 报告的衔接

参考文档：`docs/L1_query_iteration_report.md`

**历史基线（figure-text L1 主线）：**
- 73 篇论文，351 个 figure-text 对作为输入
- v3 最终输出：974 条 query
- QC 通过率：97.2%
- 验证清洁率：84.3%

**本轮主线说明：**
- 本报告从单 figure-text L1 生成切换至更严格的 **dual-evidence** 流程（figure+table / figure+formula / formula+table），用于 M4 检索训练。
- 因此，绝对数量不可与旧版 L1 总量直接比较。
- 可比维度在于：**质量门禁更严格，训练数据现已包含显式困难负例 + 跨文档候选控制。**

**继承自前期 L1 工作的方法论：**
- Prompt 硬约束 + QC 优先生成策略
- 元语言（meta-language）过滤
- 可复现的脚本与报告产物

---

## 1) 已完成事项

1. L1 dual-evidence query 官方批次生成完毕。
2. Triplet 数据（v1 与 v2 两版）构建完毕。
3. 本地 Qwen3-Embedding-4B 跨文档匹配完毕。
4. 匹配质量审计完毕。
5. Stage-B utility-aware rerank 已实现并执行（v2 严格版 + v2 平衡版）。

---

## 2) 所用方法

### A. Query 生成 + QC

- 数据来源：`data/l1_dual_evidence_queries_slurm_img150_tuned_v4_official.jsonl`
- Prompt / QC 策略：
  - 生成前置 `reasoning_chain`（因果推理链）
  - 实体特赦（entity amnesty）+ 因果拓扑约束
  - 模板塌陷检测（template-collapse checks）
  - 锚点泄漏检测（anchor leakage checks，含特赦机制）
  - 双证据重叠检测（dual-evidence overlap checks）

### B. Triplet 构建

- v1：`in_doc_swap + same_type_hard`
- v2：`in_doc_swap + same_type_hard_plus`
- 新增文本压缩字段：
  - `text`（完整文本）
  - `text_short`（训练友好精简版）
- 新增正例与负例 bundle 的图像覆盖率检查。

### C. 跨文档 Embedding 匹配

- 模型：本地 `Qwen3-Embedding-4B`
- 输出：`data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B.jsonl`
- 审计脚本：`scripts/audit_mineru_crossdoc_embedding_matches.py`
- 审计维度：
  - 分数分布（整体 + top1 + 按排名分层）
  - 约束有效性（跨文档约束、类型一致性）
  - Hub 集中度
  - 互惠率（reciprocity）
  - 可疑候选抽样

### D. Utility-aware Rerank（新增）

- 脚本：`scripts/rerank_mineru_crossdoc_matches.py`
- 惩罚信号：
  - 目标节点 hub 惩罚（target hub penalty）
  - 目标文档热度惩罚（target-doc popularity penalty）
  - 列表内多样性惩罚（intra-list diversity penalty）
  - 全局每目标 top1 上限（global top1 per-target cap）
- 产出文件：
  - 严格版：`..._v2_rerank.jsonl`（cap=8）
  - 平衡版：`..._v2b_cap10.jsonl`（cap=10）

---

## 3) 关键结果

### A. Query 官方批次（v4）

数据来源：`data/l1_dual_evidence_queries_slurm_img150_tuned_v4_official_report.json`

| 指标 | 数值 |
|------|------|
| 总 query 数 | 222 |
| QC 通过数 | 173 |
| QC 通过率 | 77.93% |
| Pair 类型 — figure+table | 144 |
| Pair 类型 — figure+formula | 62 |
| Pair 类型 — formula+table | 16 |

### B. Triplet 数据

**v1 全量（`data/l1_dual_evidence_triplets_v1_all.jsonl`）：**
- Triplet 总数：222
- 平均负例数/triplet：2.0
- 平均难度：0.6248

**v2 全量（`data/l1_dual_evidence_triplets_v2_all.jsonl`）：**
- Triplet 总数：222
- 平均负例数/triplet：2.0
- 平均难度：0.7288
- 正例图像覆盖率：100%
- 负例图像覆盖率：99.55%

**BM25 检索基线压力测试（pass 子集）：**

| 版本 | local acc@1 | global acc@1 |
|------|-------------|--------------|
| v1 pass（`...v1_pass_baseline_report.json`） | 0.8092 | 0.5549 |
| v2 pass 全文（`...v2_pass_baseline_text_report.json`） | 0.7514 | 0.4451 |

**解读：** v2 负例更难（难度提升），词法基线性能下降，这与我们减少捷径行为的目标相符，属于预期结果。

### C. Embedding 匹配审计（4B 基线）

数据来源：`data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_audit.json`

| 指标 | 数值 |
|------|------|
| 记录数 | 590（top-k=20，共 11800 条匹配） |
| 同文档违规数 | 0 |
| 类型不一致数 | 0 |
| top1 平均分 | 0.8822 |
| top10 目标集中度 | 0.3153 |
| 唯一 top1 目标数 | 186 |
| top1 互惠率 | 0.7051 |
| 可疑候选数 | 241 |

**解读：** 检索结果稳定，但 hub 效应明显，top1 集中度过高，不利于多跳 utility。

### D. Utility-aware Rerank 效果对比

**严格配置（cap=8）：**
数据来源：`data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2_rerank_report.json`

| 指标 | rerank 前 | rerank 后 |
|------|-----------|-----------|
| top1 平均分 | 0.8822 | 0.8635 |
| top10 集中度 | 0.3153 | 0.1271 |
| 唯一 top1 目标数 | 186 | 275 |
| 可疑候选数 | 241 | 140 |

**平衡配置（cap=10，当前推荐）：**
数据来源：`data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2b_cap10_report.json`

| 指标 | rerank 前 | rerank 后 |
|------|-----------|-----------|
| top1 平均分 | 0.8822 | 0.8690 |
| top10 集中度 | 0.3153 | 0.1305 |
| 唯一 top1 目标数 | 186 | 286 |
| top1 互惠率 | 0.7051 | 0.8119 |
| 可疑候选数 | 241 | 146 |

**决策：** 采用 `..._v2b_cap10.jsonl` 作为默认下游候选集。

---

## 4) 当前问题 / 缺陷

1. **目标函数错位（objective mismatch）仍然存在：**
   - 相似度分数 ≠ 多跳 utility。
   - 我们降低了 hub 集中度，但仍未直接优化 `hop_utility`。

2. **尚无人工标注的 utility 基准：**
   - 需要至少 100–300 条标注对，覆盖以下维度：
     - relevance（相关性）
     - hop_utility（多跳有用性）
     - redundancy（冗余度）
     - error_type（错误类型分类）

3. **Hub 效应在全排名池层面依然存在：**
   - top1 已改善，但全排名集中度仍显示热点目标存在。

4. **缺失图像路径比例不为零：**
   - 来源缺失率约 12%（公式密集文档可预期，但仍需制定 data-loader 处理策略）。

5. **Rerank 后 margin 指标需谨慎解读：**
   - top1 由 utility-aware 策略选出，而非原始分数排序。
   - rerank 后，原始 `margin12` 不再是可靠的质量信号。

---

## 5) 推荐下一步（近期）

1. **冻结平衡版 rerank 输出作为当前生产候选文件：**
   - `data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2b_cap10.jsonl`

2. **构建人工评测集（200 对），带分类标签，并计算：**
   - HopUtility@1 / @5 / @20
   - 错误桶分布（error bucket distribution）

3. **将 reranked 跨文档候选纳入 triplet v3 负例挖掘：**
   - 保留 `in_doc_swap`
   - 用 reranked 跨文档负例替换/增强 `same_type_hard_plus`

4. **新增一条 context-aware reranker 基线（cross-encoder 或 LLM judge），用于消融实验：**
   - embedding-only vs. +hub/diversity rerank vs. +context rerank
