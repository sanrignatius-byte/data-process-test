# 进度独立审计（2026-03-16）

> 目的：对“已实现 / 未实现”状态做代码级核验，并给出下一步改进计划（按可执行优先级）。

## 一、核验结论（与当前仓库一致）

| 工作项 | 你给出的状态 | 独立审计结果 | 证据 |
|---|---|---|---|
| CLAUDE.md 新增 2026-03-16 状态节（Phase0 v3 达标 + MoDora 完成度表 + TODO 重排） | 待确认 | ✅ 已完成 | `CLAUDE.md` 已有 “2026-03-16 更新”状态节，含 Phase0 v3 指标与 TODO 方向 |
| DISCUSSION_LOG.md 完整实验叙事 | 之前已提交 | ✅ 已存在 | `docs/DISCUSSION_LOG.md` 包含完整讨论历史与 03-10/03-12 扩展记录 |
| EXPERIMENT_RECORD_2026-03-16.md 独立实验记录 | 之前已提交 | ✅ 已完成 | `docs/EXPERIMENT_RECORD_2026-03-16.md` 完整记录 Phase1/Phase2、最优参数与最终指标 |
| MoDora C1：Enrichment 噪声过滤器 | ❌ 未实现 | ✅ **已实现（代码层）** | `scripts/generate_multihop_l1_queries.py` 存在 `_is_noisy_enrichment()` 与 `build_enriched_context_section()`，并在 prompt 构造中启用 |
| MoDora A1/A2：Section 粒度细化 + 路径枚举 | ❌ 未实现 | ✅ **已实现（代码层）** | `scripts/analyze_latex_graph_topology.py` 已实现 section/subsection 节点、`section_contains_*` 边与 targeted enumeration |
| MoDora B1/B2：Real-user 模板 + `--query-style` | ❌ 未实现 | ✅ **已实现（代码层）** | `scripts/generate_multihop_l1_queries.py` 已支持 real-user 模板路由与 `--query-style` 参数 |
| C-Pool：50-100 条通用 query | ❌ 未建立 | 🟡 **部分完成（43 条）** | 已有 `data/c_pool_universal_queries.json`，当前 `queries`=43，未达 50-100 目标 |
| Persona Hub：5 类用户人设 | ❌ 未实现 | ✅ **已实现（代码层）** | `scripts/generate_multihop_l1_queries.py` 已有 `--use-persona` 开关与 5 类 persona 选择逻辑 |
| GRAPH_ARCHITECTURE.md 完整度 | ✅ 已存在（需确认） | 🟡 **存在但偏薄** | `docs/GRAPH_ARCHITECTURE.md` 仅架构骨架，缺少实验结果、参数、公式与失败分析 |

---

## 二、核心判断（防止“已写代码=已交付”误判）

当前状态更准确表述应为：

1. **MoDora A/B/C + Persona Hub：代码实现基本完成，但实验验证未闭环**（尤其是 full-run、质量分布、失败案例分析）。
2. **C-Pool：已起步但数量不足**（43 条，未达到你设定的 50-100 条）。
3. **GRAPH_ARCHITECTURE.md：文档存在，但远未达到“专利/论文答辩级”完整度。**

---

## 三、改进计划（建议按 2 周冲刺执行）

### P0（本周必须完成，直接影响可汇报性）

1. **统一“完成度口径”**
   - 在 `CLAUDE.md` 与周报中改成三态：`代码完成 / 小样验证 / 全量验证`。
   - 把 A/B/C/Persona 从“已实现”拆成“代码✅，全量验证⬜”。

2. **跑通 MoDora 全链路小样验证（n=50）**
   - 固定：`--query-style real_user --use-persona`。
   - 产出：pass rate、issue top-5、evidence localization（R@10/MRR）。
   - 验证目标：确认 real-user 与 academic 风格是否存在显著质量差异。

3. **补齐 C-Pool 到 60 条（先过 50 门槛）**
   - 每类意图（summary/comparison/how/why/debug）至少 10 条。
   - 每条附 `expected_evidence_types`，保证可定位评估。

4. **扩写 GRAPH_ARCHITECTURE.md 到“评审可读版 v2”**
   - 必含：节点/边统计、hub 公式、最优配置、ablation 结论、适用边界。

### P1（下周完成，形成“可决策”证据）

5. **500 candidates 全量验证**
   - 对比：`academic` vs `real_user` vs `mixed`。
   - 切片：persona × hop_distance × pair_type。
   - 输出：一页结论表（哪种组合在 evidence localization 上最稳）。

6. **Persona Hub A/B 测试**
   - baseline：关闭 persona。
   - treatment：开启 persona（5 类均衡采样）。
   - 指标：query 多样性、QC pass、R@10/MRR。

7. **C-Pool 专项评测流程固化**
   - 明确“只评 evidence localization，不评 query quality”的协议文件。
   - 建立固定评测脚本命令（便于每周复现）。

### P2（后续，面向专利/论文）

8. **Graph 架构文档与实验记录互相链接**
   - 在 `GRAPH_ARCHITECTURE.md` 内引用 03-16 实验关键结论，形成“设计→验证”闭环。

9. **失败案例库（20 条）**
   - 按 `bridge_error / evidence_gap / retrieval_bias / parse_noise` 分类沉淀。

---

## 四、建议使用的“状态定义”

- **代码完成**：功能存在且可运行（通过静态核验）。
- **小样验证**：n≤50 跑通并有指标。
- **全量验证**：n≈500 跑通并有对照实验。
- **可汇报完成**：全量验证 + 文档闭环（架构、实验、结论一致）。

按该定义，当前项目最接近：
- MoDora A/B/C + Persona：**代码完成**
- C-Pool：**代码/数据起步（未达标）**
- Graph Architecture：**文档起步（未达汇报级）**
