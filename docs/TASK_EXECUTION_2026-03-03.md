# 新任务执行记录（2026-03-03）

日期：2026-03-03  
范围：LaTeX 引用图拓扑分析 + Query 多样性优化 + Hard Negative 策略补充

---

## 1) 引用图拓扑结构与连接密度

已新增脚本：`scripts/analyze_latex_graph_topology.py`

### 1.1 图连接密度（paragraph / figure / table / formula）

基于 `data/latex_reference_graph.json` + `data/multimodal_elements.json` 实际统计结果：

- 节点总数：2551
- 边总数：1768
- 全局有向边密度：0.00027179
- 节点分布：
  - paragraph: 1347
  - figure: 532
  - table: 221
  - formula: 451
- 边分布：
  - paragraph_ref: 1688
  - element_ref: 80

结果文件：
- `data/latex_graph_topology_report.json`

### 1.2 “交通枢纽”节点识别（拓扑算法）

实现方式：
- 入度 + 出度 + PageRank 组合评分
- 单独导出 traffic hub（要求 in_degree>0 且 out_degree>0）

示例 top hub（traffic）：
- `1602.05352::el::tab:illuexp` (table) in=3 out=3
- `1603.07025::el::fig:comment_length` (figure) in=9 out=2
- `1306.5204::el::fig:histograms` (figure) in=2 out=8

结果文件：
- `data/latex_graph_hubs.json`（含 `hubs` 与 `traffic_hubs`）

### 1.3 从 hub 出发构建长链多跳候选

已基于 traffic hub 构建候选路径：
- 候选数：300
- 典型链路类型示例：`table -> figure -> paragraph -> formula -> paragraph`

结果文件：
- `data/latex_hub_multihop_candidates.json`

### 1.4 跨页物理距离方差

针对已建立连接的跨模态 pair 计算页跨度统计：
- 样本数：17
- 均值：0.0
- 方差：0.0

当前候选中跨页跨度基本为同页（后续可通过放宽跨页配对策略扩展）。

---

## 2) Query 构造与多样性优化

### 2.1 先验失败分析（已执行）

已新增脚本：`scripts/analyze_query_quality_focus.py`  
输入：`data/l1_dual_evidence_queries_slurm_img150_tuned_v4_official.jsonl`

关键结果：
- 总 query：222
- 长度分布：`long=212`, `medium=10`, `short=0`
- pair 级长短句混合覆盖率：`0/111 = 0.0`
- figure+formula 主要失败项：
  - `numeric_leakage`: 6
  - `weak_reasoning_connector`: 5
  - `single_element_answer`: 4

结果文件：
- `data/query_quality_focus_report_v4_official.json`

### 2.2 生成策略改造（已落地代码）

已修改：`scripts/generate_multihop_l1_queries.py`

改动点：
- Prompt 级硬约束：每对 query 必须一短一长
  - short: 8-14 words
  - long: 18-30 words
- QC 新增：
  - `query_too_short` / `query_too_long`
  - pair 级 `length_mix_missing`
  - 输出 `query_length_bucket`
- 模型架构图专项修复：
  - 自动检测 architecture figure
  - 在 figure+formula prompt 注入 failure-case guidance（学者视角：结构总结/创新点/组件-公式机制/实验效应）
  - QC 新增 `architecture_intent_missing`

说明：
- 这部分需要重新跑生成流程后，才会在新批次中体现提升。

---

## 3) Hard Negative（优先保证正向 Query 质量 + 可选增强）

已修改：`scripts/build_dual_evidence_triplets.py`

默认行为保持不变（正向质量优先，不强制增加额外 hard negative）。  
新增可选能力：

- `--enable-related-random-negative`
- `--related-random-scope {same_doc,same_or_similar}`
- `--related-min-overlap <float>`

该策略会随机抽取同文档/相似候选中的相关片段作为额外负样本（`negative_type=related_random`）。

smoke test（20 条测试集）：
- `related_random` 生成 19 条
- 总负样本类型：`in_doc_swap + same_type_hard_plus + related_random`

---

## 4) 本次新增/更新文件

- 新增：
  - `scripts/analyze_latex_graph_topology.py`
  - `scripts/analyze_query_quality_focus.py`
  - `docs/TASK_EXECUTION_2026-03-03.md`
- 修改：
  - `scripts/generate_multihop_l1_queries.py`
  - `scripts/build_dual_evidence_triplets.py`
- 产出数据：
  - `data/latex_graph_topology_report.json`
  - `data/latex_graph_hubs.json`
  - `data/latex_hub_multihop_candidates.json`
  - `data/query_quality_focus_report_v4_official.json`

---

## 5) 复现命令

```bash
cd data-process-test

# 图拓扑 + hub + 跨页方差
python3 scripts/analyze_latex_graph_topology.py

# Query 质量聚焦分析
python3 scripts/analyze_query_quality_focus.py \
  --queries data/l1_dual_evidence_queries_slurm_img150_tuned_v4_official.jsonl \
  --elements data/multimodal_elements.json \
  --output data/query_quality_focus_report_v4_official.json
```
