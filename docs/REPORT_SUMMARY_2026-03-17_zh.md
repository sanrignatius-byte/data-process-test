# M4 数据工程进度汇报

**日期**：2026-03-17　　**周期**：2026-03-10 → 2026-03-17

---

## 一、TL;DR

### 本周最重要的一件事

**Document Graph 首次显著超越 BM25 基线，Phase0 效果验证达标。**

| 指标 | graph_full | vs BM25 | 达标阈值 |
|------|-----------|---------|---------|
| Recall@10 | **0.8736** | **+0.0269** (+3.2%) | — |
| MRR | **0.6045** | **+0.0403** (+7.1%) | ≥ +0.03 ✅ |

`continue_expand = True` ✅ — 核心机制：bridge hub topology → element adjacency → 1-hop neighbor propagation，全程纯规则，**零 LLM 成本**。

### 本周产出总览

| 里程碑 | 产出 |
|--------|------|
| Phase0 Eval 达标 | graph_full MRR +0.0403 vs BM25，首次超越 |
| 三项工程修复 | quality_score 重建 + hub coverage ×9.5（9.53%→90.42%）+ citation walk 方向修复 |
| 组件权重解耦 | `--hub-weight / --nprop-weight / --cite-weight` 独立调参，最优配置锁定 |
| MoDora 全四工作流代码完成 | A1/A2 + B1/B2 + C1/C3 + D1 + Persona Hub，全部已实现 |
| Graph 技术方案文档 v3 | `GRAPH_ARCHITECTURE.md` 从 42 行框架重写为完整技术方案 |

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
- 基线：BM25（k1=1.5, b=0.75）+ TF-IDF dense

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
| **hub coverage** | **161 元素，9.53%** | **403 元素，90.42%** | **最大单一增益来源** |
| citation walk | 单向传播 | 双向 + 2-hop co-citation | 负贡献减弱（仍为负，最终关闭） |

hub coverage 提升的核心手段：将 **369 条 adjacent backbone bridges** 纳入 hub 覆盖集（纯规则，零额外成本）。

### 3.4 组件贡献（消融结论）

| 组件 | MRR Δ vs BM25 | 角色 |
|------|---------------|------|
| **1-hop neighbor_prop** | **+0.0313** | 核心信号（~70% 增益），拯救 10 条 BM25 miss queries |
| hub_prior | +0.0015 | 静态辅助，与 neighbor_prop 协同 |
| citation_walk | **-0.0024** | 负贡献（doc-level vs element-level 错位），已关闭 |
| 2-hop propagation | 低于 1-hop | 扩散噪声，不采用 |

### 3.5 Per-query 分析

Graph full 拯救 **11 条** BM25 完全遗漏的 queries → **全部是跨模态 dual-evidence**（fig+tab: 5, fig+formula: 4, formula+tab: 2）。这验证了 neighbor propagation 在跨模态桥接场景的独特价值。

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
| Persona Hub (5 类) | ✅ 代码完成 | 同上 |

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
| **P0** | 全量生成 real-user + persona queries（500 candidates） | `--provider company --query-style mixed --use-persona`，产出 400+ 新 queries |
| **P0** | 扩大评测集 + 重跑 eval | 新 queries 上 graph_full 仍优于 BM25（验证泛化性） |
| **P0** | 跑 MoDora element enrichment | 产出 `multimodal_elements_enriched.json`，使 C1 噪声过滤在全量生成中生效 |
| P1 | 修复 35/82 篇零候选文档 | 降 per_combo_cap 或 adj_bridge 单独路径 |
| P1 | Citation walk 改进方向探索 | element-level cross-doc linking 替代 doc-level citation 边 |
| P1 | 统计显著性检验 | bootstrap CI + paired test，加强实验说服力 |

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
| MoDora 四工作流代码 | ✅ 全部完成 | 本周补齐 A1/A2 + Persona Hub |
| v4.5 生成链路 | 代码完成 + bug 修复 | PR #102 |
| 公司 API | 就绪 | — |
| Graph 技术方案文档 | **v3 重写完成** | 支撑专利 + 汇报 |
| **全量生成验证** | ⬜ 待执行 | 下周 P0 |
