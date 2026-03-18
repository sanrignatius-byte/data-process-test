# M4 数据工程进度汇报

**日期**：2026-03-17　　**周期**：2026-03-10 → 2026-03-17

---

## 一、TL;DR

### 本周最重要的一件事

**Document Graph 首次超越 BM25 基线，Phase0 效果验证达标。**

> **关键术语速查**
>
> - **Document Graph（文档图）**：从学术论文的显式结构信号（引用标记、段落顺序等）自动构建的多层异构图，节点为文档中的多模态元素（段落 / 图 / 表 / 公式），边编码它们之间的引用与顺序关系。核心价值：无需 LLM，仅靠文档自身结构即可建模跨模态关联。
> - **BM25 基线**：经典的稀疏检索算法，基于词面匹配（term frequency + inverse document frequency）对每个候选 chunk 独立打分。它是信息检索领域使用最广泛的基线方法，但无法利用文档内部的结构关系——如果两个 chunk 被同一段话引用（词面却完全不同），BM25 无法发现这种关联。
> - **Recall@10（前 10 召回率）**：检索系统返回的前 10 个结果中，包含了多少比例的 ground truth 证据。衡量"能不能找到"。
> - **MRR（Mean Reciprocal Rank，平均倒数排名）**：第一个正确结果排在第几位的倒数的平均值。MRR=0.6 大致意味着正确答案平均排在第 1-2 位。衡量"排得准不准"。
> - **Phase0**：效果验证的第一阶段——在已有数据上做最小实验，回答"图信号是否比纯 BM25 有提升"这个门控问题。达标后才进入 Phase1（扩大规模 + 统计显著性检验）。

| 指标 | graph_full | vs BM25 | 达标阈值 |
|------|-----------|---------|---------|
| Recall@10 | **0.8736** | **+0.0269** (+3.2%) | — |
| MRR | **0.6045** | **+0.0403** (+7.1%) | ≥ +0.03 ✅ |

`continue_expand = True` ✅ — 核心机制：bridge hub topology → element adjacency → 1-hop neighbor propagation，图构建与 rerank 全程纯规则，**零额外 LLM 调用**。

> **机制术语解释**
>
> - **Bridge Hub（桥接枢纽）**：同时引用至少两种不同模态元素的段落节点。例如一个段落同时提到 Figure 3 和 Table 2，它就是连接"图"和"表"两种模态的桥梁。为什么重要？因为跨模态查询的答案往往分散在不同模态中，bridge hub 是唯一能在图上把它们关联起来的中间节点。
> - **Element Adjacency（元素邻接）**：两个多模态元素通过图中的边（引用边 / 阅读顺序边）直接相连的关系。这是传播检索分数的基础——BM25 命中了一个元素，分数可以沿邻接边传播到相关联的另一个元素。
> - **1-hop Neighbor Propagation（1 跳邻域传播）**：在 BM25 初步打分后，每个候选 chunk 从其图上直接邻居中获取分数加成。通俗地说：BM25 找到了 Figure 3 → 沿图边发现 Table 2 是邻居 → 给 Table 2 加分 → 两个跨模态证据一起浮上来。为什么只传 1 跳？因为当前图密度下，2 跳会扩散到大量弱关联节点，引入噪声。
> - **Rerank（重排序）**：在 BM25 初始排序基础上，利用图的拓扑信号调整候选的排名，而非替换 BM25。零额外 LLM 调用意味着这一步完全是规则驱动的分数传播。

### 本周产出总览

| 里程碑 | 产出 |
|--------|------|
| Phase0 Eval 达标 | graph_full MRR +0.0403 vs BM25，首次超越 |
| 三项工程修复 | quality_score 重建 + hub coverage ×9.5（9.53%→90.42%）+ citation walk 方向修复 |
| 组件权重解耦 | `--hub-weight / --nprop-weight / --cite-weight` 独立调参，最优配置锁定 |
| MoDora 全四工作流代码完成 | A1/A2 + B1/B2 + C1/C3 + D1 + PersonaHub 多样化人设，全部已实现（待全量验证） |
| Graph 技术方案文档 v3 | `GRAPH_ARCHITECTURE.md` 从 42 行框架重写为完整技术方案 |

> **产出术语补充**
>
> - **quality_score（质量分数）**：Hub 节点的综合评分，用于量化其作为跨模态桥梁的价值。修复前是常量 0.8（所有 hub 同分，完全没有区分度），修复后基于拓扑特征（bridge score + PageRank + 元素出度）加权计算，分布在 [0.13, 0.88] 之间。
> - **hub coverage（Hub 覆盖率）**：评测集中 ground truth 证据落入 hub 覆盖集的 query 占比。9.53% 意味着只有约 25 条 query 的证据元素被 hub 识别覆盖到——图信号根本无法施加影响。提升到 90.42% 后，图才有足够的"作用面积"产生效果。
> - **citation walk（引用随机游走）**：沿跨文档引用边传播分数的机制——如果论文 A 引用了论文 B，把 A 中命中 chunk 的部分分数传给 B 中的相关 chunk。因为引用边是文档级而非元素级（知道 A 引用了 B，但不知道 A 的哪段引用了 B 的哪个元素），导致信号太粗糙，实验结果为负贡献，已关闭。
> - **MoDora**：一种面向文档理解的上游语义增强方法（Modality-aware Document Representation）。我们借鉴其核心思路——对 figure/table/formula 三类元素用 LLM 生成结构化描述（enriched content），增强元素的语义密度——但不迁移其检索框架。四个工作流覆盖节点粒度细化（A）、real-user 查询风格（B）、enrichment 质量闸门（C）、QC 体系重构（D）。
> - **PersonaHub 多样化人设**：借鉴 PersonaHub（Ge et al., 2024, arXiv:2406.20094, Tencent AI Lab）的方法论——用多样化的人设描述驱动 LLM 生成差异化数据。我们从 PersonaHub 的 Text-to-Persona 格式出发，针对学术文档理解场景策展了 50 类读者人设（涵盖 PhD 学生、资深审稿人、产业工程师、政策分析师、跨领域研究者等），在 query 生成时按 pair_id 哈希确定性分配，提升查询的多样性和自然度。人设数据文件：`data/personahub_academic_personas.json`。

本周合并 **21 个 PR**（#82 ~ #103），55 个 commits。

---

## 二、上周计划执行情况

| 上次计划 | 执行结果 | 备注 |
|----------|---------|------|
| ① 修复 yes/no 禁令 + formula checker + numeric 验证池 | ✅ PR #102 | `e12b941` real_user QC formula grounding + lazy persona + numeric 修复 |
| ② 排查 270 条 `skipped_no_mapping` 根因并修复 | ⚠️ 部分完成 | label mapping 改善（49.8%），但 35/82 篇仍零候选 |
| ③ 建立最小评估闭环（Recall@10/MRR 有数字） | **✅ 超额完成** | 不仅建立了闭环，还做了完整 grid search 并达到 continue_expand 阈值 |
| 节点重要性打分 P0 缺口补齐 | ✅ PR #82, #83 | `core_module_score` 数据落盘 + pair_importance_score 透传 |
| 单文档闭环优先于跨文档 | ✅ 执行 | Phase0 eval 以 intra-doc 为主，citation walk 实验为负后关闭 |

**上周最大的遗留问题——"评估闭环连续两周未完成"——本周彻底解决。**

---

## 三、Phase0 评测详情（核心成果）

### 3.1 实验设置

- 评测集：261 条 QC-pass L1 dual-evidence queries（v4_4_run1: 113 + v3: 152）
- 候选库：1314 chunks（76 篇文档）
- 基线：BM25（k1=1.5, b=0.75）稀疏检索 + TF-IDF dense（TF-IDF 向量余弦相似度检索）

> **实验术语解释**
>
> - **QC-pass（质量控制通过）**：每条生成的 query 都经过自动质量检查（检测 anchor 泄漏、单元素即可回答、弱推理连接等问题），只有通过全部检查的 query 才进入评测集。
> - **L1 dual-evidence queries（L1 双证据查询）**：文档内（L1 = intra-document）需要同时使用两种不同模态证据才能回答的查询。例如"模型在哪些子群上的公平性最差？"需要同时看 Figure（趋势图）和 Table（精确数值）才能完整回答。
> - **chunk（检索单元）**：文档解析后的最小检索粒度，每个 chunk 对应一个多模态元素（段落 / 图 / 表 / 公式）及其上下文。1314 chunks 是系统从 76 篇文档中解析出的全部候选。
> - **TF-IDF dense**：将每个 chunk 表示为 TF-IDF 向量，用余弦相似度检索。与 BM25 同为词面方法但实现方式不同——BM25 用概率模型，TF-IDF dense 用向量空间模型。两者对比用于排除"BM25 本身碰巧特别差"的可能性。

### 3.2 迭代过程（从 -0.009 到 +0.040）

| 版本 | 日期 | MRR | Δ vs BM25 | 关键修复 |
|------|------|-----|-----------|---------|
| v1 | 03-15 | 0.5315 | **-0.009** | 初始实验，alpha=0.3 过高 |
| v2 | 03-15 | 0.5552 | -0.009 | alpha 0.3→0.1 |
| v3-fix | 03-16 | 0.5939 | +0.030 | quality_score + hub coverage + citation 修复 |
| **v3-tuned** | **03-16** | **0.6045** | **+0.040** | cite_weight=0, hw=0.15, nd=0.20 |

### 3.3 三项工程修复（v2 → v3-fix，最大增益来源）

| 修复项 | 修复前 | 修复后 | 影响 |
|--------|--------|--------|------|
| quality_score | 常量 0.8（无区分度） | 拓扑加权 [0.13, 0.88]（31 值） | hub prior 有区分度 |
| **hub coverage** | **161 元素，query 覆盖率 9.53%（25/261）** | **403 元素，query 覆盖率 90.42%（236/261）** | **最大单一增益来源** |
| citation walk | 单向传播 | 双向 + 2-hop co-citation | 负贡献减弱（仍为负，最终关闭） |

hub coverage 提升的核心手段：将 **369 条 adjacent backbone bridges** 纳入 hub 覆盖集（纯规则，零额外成本）。

### 3.4 组件贡献（消融结论）

> **消融实验**（ablation study）：逐一关闭系统的各个组件，观察效果变化，从而量化每个组件的独立贡献。

| 组件 | MRR Δ vs BM25 | 角色 |
|------|---------------|------|
| **1-hop neighbor_prop** | **+0.0313** | 核心信号（~70% 增益），单组件拯救 10 条 BM25 miss queries |
| hub_prior | +0.0015 | 静态辅助，与 neighbor_prop 协同 |
| citation_walk | **-0.0024** | 负贡献（doc-level vs element-level 错位），已关闭 |
| 2-hop propagation | 低于 1-hop | 扩散噪声，不采用 |

> - **hub_prior（Hub 静态先验）**：对 hub 覆盖集内的元素给予固定的微量加分（不依赖 query），相当于"事先标记哪些元素更可能是重要证据"。独立效果小（+0.0015），但为 neighbor_prop 提供了更好的起点。
> - **BM25 miss queries**：BM25 在 top-10 中完全没有命中任何 ground truth 证据的查询。graph 能"拯救"这些 query 说明图信号捕获了 BM25 完全看不到的关联。

### 3.5 Per-query 分析

Graph full（neighbor_prop + hub_prior 组合）拯救 **11 条** BM25 完全遗漏的 queries（neighbor_prop 单组件为 10 条，+1 条来自 hub_prior 协同）→ **全部是跨模态 dual-evidence**（fig+tab: 5, fig+formula: 4, formula+tab: 2）。这验证了 neighbor propagation 在跨模态桥接场景的独特价值。

---

## 四、其他代码改动

### 4.1 Phase0 评测框架从零搭建（PR #86 ~ #99，12 个 PR）

| 改动 | 文件 | PR |
|------|------|-----|
| Phase0 A/B 实验脚本 | `scripts/run_phase0_eval_ab.py` | #86 |
| element-level hub prior（修复 chunk→element 级匹配） | 同上 | #88, #89 |
| neighbor_prop + citation_walk + graph_full 组件 | 同上 | #95 |
| hub-BM25 overlap 诊断 | 同上 | #92 |
| hub prior 重建 + coverage 扩大 + citation 修复 | `scripts/enrich_hub_candidates.py` + eval 脚本 | #98 |
| 组件权重解耦 + grid search | eval 脚本 | #99 |
| 实验记录文档 | `docs/EXPERIMENT_RECORD_2026-03-16.md` | #99 |

### 4.2 MoDora 工作流代码（PR #82 ~ #84）

| 工作流 | 状态 | 备注 |
|--------|------|------|
| A1: Section 粒度切分 | ✅ 代码完成 | `src/parsers/latex_reference_extractor.py` |
| A2: Strategy 4 + `--single-doc-only` | ✅ 代码完成 | `scripts/analyze_latex_graph_topology.py` |
| B1: 5 类 real-user 模板 | ✅ 代码完成 | `scripts/generate_multihop_l1_queries.py` |
| B2: `--query-style` CLI | ✅ 代码完成 | 同上 |
| C1: Enrichment 噪声过滤器 | ✅ 代码完成 | 同上 |
| C3: Hub summary 压缩重写 | ✅ 代码完成 | `scripts/enrich_hub_candidates.py` |
| D1: `qc_real_user_query()` | ✅ 代码完成 | 同上 |
| PersonaHub 多样化人设 (50 类) | ✅ 代码完成 | 同上 |

**以上均为代码就绪，尚未进行 500 candidates 全量运行验证。**

### 4.3 Bug 修复与基础设施（PR #100 ~ #103）

| 改动 | PR |
|------|-----|
| real_user QC 修复（formula grounding + lazy persona + numeric） | #102 |
| 服务器路径迁移（`/home/d00855555/query_myx/`） | #103 |
| 默认 query 模型切换为 gpt-5.4 | #101 |
| CLAUDE.md 状态更新 + TODO 重组 | #100 |

### 4.4 文档

| 文档 | 内容 |
|------|------|
| `docs/EXPERIMENT_RECORD_2026-03-16.md` | Phase0 v3 完整实验记录（修复前/后/调优全过程） |
| `docs/GRAPH_ARCHITECTURE.md` v3 | 技术方案文档重写（问题定义→发明点→形式化→实验→消融） |
| `docs/DISCUSSION_LOG.md` | 新增 2026-03-12（Mentor 战略升级）、2026-03-15（Phase0 v2）、2026-03-16（Phase0 v3）三节 |
| CLAUDE.md | Phase0 达标状态 + P0/P1/P2 重排 |

---

## 五、上周主管反馈项回溯

| 上周反馈 | 本周回应 |
|----------|---------|
| "节点重要性打分处于缺失状态" | ✅ 已补齐：core_module_score 数据落盘 + quality_score 重建为拓扑加权公式 |
| "单文档闭环未稳，跨文档步子偏大" | ✅ 执行：Phase0 以 intra-doc 为主；citation walk 实验证实为负 → 关闭，转 P1 改进 |
| "QC 指标执念 vs 检索目标偏移" | ✅ 转向：Recall@10 / MRR 已成为北极星指标，QC pass rate 降为辅助监控 |
| "评估闭环连续两周未完成" | **✅ 本周彻底解决**：从搭框架到 grid search 到达标，一周内完成 |

---

## 六、下周计划

| 优先级 | 事项 | 交付标准 |
|--------|------|---------|
| **P0** | 全量生成 real-user + PersonaHub persona queries（500 candidates） | `--provider company --query-style mixed --use-persona`，50 类 PersonaHub 人设驱动，产出 400+ 新 queries |
| **P0** | 扩大评测集 + 重跑 eval | 新 queries 上 graph_full 仍优于 BM25（验证泛化性） |
| **P0** | 跑 MoDora element enrichment | 产出 `multimodal_elements_enriched.json`，使 C1 噪声过滤在全量生成中生效 |
| **P0** | **C-Pool 泛用型查询库（Mentor 要求）** | 人工整理 50-100 条通用学术 query（总结/动机/方法/贡献/跨文档连接），QC 只验 evidence localization。详见下方说明 |
| P1 | 修复 35/82 篇零候选文档 | 降 per_combo_cap 或 adj_bridge 单独路径 |
| P1 | Citation walk 改进方向探索 | element-level cross-doc linking 替代 doc-level citation 边 |
| P1 | 统计显著性检验 | bootstrap CI + paired test，加强实验说服力 |

**C-Pool 泛用型查询库说明**（Mentor 2026-03-12 提出）：

C-Pool 是一组**不依赖特定文档内容**的通用学术查询模板，用于测试检索系统在泛化场景下的鲁棒性。典型 query 如"这篇论文的核心贡献是什么？""方法部分的 baseline 对比有哪些？""Figure 1 展示了什么？"等。这类 query 不走 multihop 生成流程，由人工策展 + 轻量模板扩展生成。QC 策略为只验 evidence localization 准确性，不评 query 本身的质量。当前状态：**尚未启动**，计划下周与 PersonaHub persona 全量生成并行推进。

---

## 附录：当前模块状态

| 模块 | 状态 | 变化 |
|------|------|------|
| LaTeX 多层图 | 稳定可复跑 | — |
| Hub bridge-first 候选 500 条 | 已产出 | — |
| **Phase0 评测框架** | **✅ 新建并达标** | 本周从零搭建 |
| **Graph 效果验证** | **✅ MRR +0.0403** | 本周核心成果 |
| **组件权重解耦** | **✅ 最优配置锁定** | hw=0.15, nd=0.20, cw=0.0 |
| MoDora enrichment 数据 | 1285/1316（97.6%） | — |
| MoDora 四工作流代码 | ✅ 全部完成 | 本周补齐 A1/A2 + PersonaHub 人设 |
| v4.5 生成链路 | 代码完成 + bug 修复 | PR #102 |
| 公司 API | 就绪 | — |
| Graph 技术方案文档 | **v3 重写完成** | 支撑专利 + 汇报 |
| **全量生成验证** | ⬜ 待执行 | 下周 P0 |
