# M4 数据工程进度汇报

**日期**：2026-03-10　　**周期**：2026-03-04 → 2026-03-10

---

## 一、TL;DR & 核心行动项

### 生产进度（三件事已落盘）

| 里程碑 | 产出 |
|---|---|
| MoDora 语义增强 | `multimodal_elements_enriched.json`：76 篇文档 / **1285/1316 = 97.6%** 元素完成 enrichment |
| Hub 语义摘要链路 | `hub_candidates_enriched_v2.json`：**230 条** enriched pairs，全部带 `hub_semantic_summary` |
| v4.5 生成器 + 公司 API | 噪声过滤、real-user 模板、新 QC 链路三项核心升级上线；公司 API 全链路就绪 |

三个模块已形成可复跑的最小闭环：`enriched_elements → enriched_hub_pairs → query_generation`。

### Query 质量现状（img_20 批次，32 条）

v4.5 生成链路接入 enriched context 后，最新一批 32 条 query（`data111/l1_img_run_20.jsonl`）整体 QC pass 率 **65.6%（21/32）**，较上批 v4.4（~44.8%）有明显提升。figure+table 双模态已达可用水位；figure+formula 存在一个系统性 QC 问题待修（见第四节）。

| pair_type | style | 总条 | pass | pass 率 |
|---|---|---|---|---|
| figure+table | academic | 20 | 18 | **90%** |
| figure+table | real_user | 6 | 5 | **83%** |
| figure+formula | real_user | 6 | 0 | **0%** |
| **合计** | | **32** | **21** | **65.6%** |

### 本周 P0 交付序列

| 顺序 | 事项 | 交付标准 |
|---|---|---|
| **①（周二前）** | 建立最小评估闭环 | 20-30 条人工 query + BM25 + Recall@10/MRR 有数字 |
| **②（周三前）** | 修复 figure+formula `formula_symbol_grounding_missing` + 排查 270 条 `skipped_no_mapping` | figure+formula pass 率 ≥ 50%；mapping rate ≥ 70% |
| **③（周五前）** | 修复后重跑全量，产出可训练子集 | figure+table 可入训练集；formula 对 QC 通过 |

评估闭环（①）已连续两周被排后，本周提升为第一优先，哪怕样本量只有 20 条，**有 Recall@10/MRR 数字比无数字有决定性意义**。

---

## 二、上周计划执行情况

| 上次计划 | 本周执行 | 结果 | 备注 |
|---|---|---|---|
| C1：噪声过滤 | `_is_noisy_enrichment()` 上线，7 类正则 + 15 字符门限 | ✅ | |
| B1/B2：real-user 模板 | 5 类模板 + `--query-style` CLI 参数 | ✅ | |
| D1：`qc_real_user_query()` | 完成，放宽 6 类 academic 约束 | ✅ | |
| 公司 API 打通 | `--provider company` + SSE 解析 + token 日志 | ✅ | |
| A1/A2：段落/section 精细化 | `_extract_paragraphs()` + section label 融入拓扑图 | ✅ | |
| C3：hub summary + 产出数据 | 230 条产出，全部带 summary | ⚠️ 部分完成 | 目标 500 条，实际 230 条（46%）；270 条映射丢失问题本周未解决 |
| MoDora enriched 数据实际生成 | 1285/1316 落盘 | ✅ | |

C3 代码功能已上线，但数据覆盖率只达到 46%，这部分目标未达成，是本周遗留的最大技术债。

---

## 三、数据管线产出指标

### 3.1 MoDora 语义增强

对 figure / table / formula 三类元素分别用专用 prompt 生成结构化语义描述（enriched_title / enriched_metadata / enriched_content），供下游 prompt 注入。

**实际产出：**
- 覆盖 76 篇文档，figure **811/841**、table **334/334**、formula **140/141**，合计 **1285/1316（97.6%）**
- `enriched_title` 均长 64 字符，`enriched_content` 均长 553 字符，已达可用语义密度
- 剩余 31 条写回失败，原因尚未完整归因，为 P2 排查项

### 3.2 Hub 语义摘要合成

| 指标 | 数值 |
|---|---|
| 输入候选 | 500 |
| 产出 enriched pairs | **230（mapping rate 46%）** |
| 跳过（`skipped_no_mapping`） | 270 |
| pair_type 分布 | figure+table 132 / figure+formula 74 / formula+table 24 |
| hop 分布 | 2-hop 82 / 3-hop 148；cross-doc 74；覆盖 38 篇文档 |

**270 条丢失的根因已基本定位：** 这不是 doc_id 前缀小差异，而是 **LaTeX label-style node id**（如 `1904.03310::el::tab:lm_cor`、`1709.02012::el::fig:general-disparity`）无法稳定 join 到 **MinerU ordinal element id**（如 `1904.03310_table_1`、`1709.02012_figure_4`）。我抽样对比后发现，候选端点里 **853/1000** 仍是 `fig:/tab:/eq:` 这类 label-style 标识，而 enriched 落盘侧用的是 `figure_1/table_5/formula_1` 编号体系；当前损耗本质上是 **label mapping / schema join 缺失**，不是可以容忍的边角问题。周三前必须修到可复跑。

### 3.3 v4.5 生成器核心升级

- **噪声过滤器**：检测到 OCR 残留、Unicode 装饰符等低质 enriched 字段时静默回退原始 context（已在生产数据中发现真实噪声样本）
- **real-user 问题风格**：5 类模板（factual_lookup / summary / comparison / how_works / what_if），`--query-style` 支持 academic / real_user / mixed
- **`qc_real_user_query()`**：保留 4 项核心 QC，放宽 8 项 academic 约束，新增 `retrievability_score` 启发式评分

---

## 四、Query 质量分析（img_20，32 条）

来自 `data111/l1_img_run_20.jsonl`，32 条 query（query_id 0000–0031），总 pass 率 **65.6%（21/32）**。

### 4.1 pass/fail 明细

| pair_type | style | 总条 | pass | fail | 主要 fail 原因 |
|---|---|---|---|---|---|
| figure+table | academic | 20 | 18 | 2 | `single_element_answer`（2） |
| figure+table | real_user | 6 | 5 | 1 | `single_element_answer`（1） |
| figure+formula | real_user | 6 | 0 | 6 | `formula_symbol_grounding_missing`（6） |

### 4.2 两个系统性问题

**问题 A：`formula_symbol_grounding_missing`（figure+formula 全军覆没，6 条，来自 1709.02012）**

real_user 风格 query 直接引用 `c_fp`、`h_t*`、`μ_t` 等裸 LaTeX 符号，未加自然语言释义。QC `_formula_symbol_hit()` 检测到 ≥4 个符号 term 但 answer 中无自然语言对应项，全部拦截。

修复方向：real_user 模板要求引用公式符号时括注语义，例如 `c_fp (false-positive cost rate)` 而非裸符号。定位代码：`generate_multihop_l1_queries.py` 第 1327–1363 行（`_extract_formula_symbol_terms` + `_formula_symbol_hit`）。预期修复后 figure+formula pass 率 ≥ 50%。

**问题 B：`single_element_answer`（5 条，跨 pair_type）**

受影响：query_id 0008（answer_balance=0.14）、0017（0.20）、0022（0.17）、0026（0.08）、0027（0.11）。answer 主体只围绕一端 element，另一端仅做结构性引用。修复方向：prompt 层加约束——两端各须有一句含具体数值或观察的句子。

### 4.3 通过批次质量指标

| 指标 | 状态 |
|---|---|
| anchor_leak_jaccard | ✅ 全部 < 0.15（0002 = 0.143，边界内） |
| short+long query 配对覆盖 | ✅ 所有 academic 对均有短长两条 |
| dual_evidence / cross_modal 标记 | ✅ 全部正确 |
| reasoning_chain 非空 | ✅ academic 对全有；real_user 按设计为空，可接受 |
| `has_cross_modal_operator` | ⚠️ 部分 long query 缺失（0001、0007、0010、0011），非 fail 触发条件，下轮加约束 |

### 4.4 版本对比

| 批次 | 条数 | QC pass | pass 率 |
|---|---|---|---|
| v4.4（上批） | 252 | 113 | 44.8% |
| v4.5 img_20（本批） | 32 | 21 | **65.6%** |

figure+table 已达可用水位（90%），可继续扩量。figure+formula 全批 fail，须在下次全量跑前修复符号 grounding 问题。

---

## 五、节点重要性打分体系（本报告此前遗漏的核心模块）

> **上周主管明确要求**：定义规则量化节点重要性，具体维度包括：承上启下桥接功能加分、包含核心模块（Introduction/Main Results）加分、边数多的加分。本节如实说明当前落地状态，包含一处此前误报的重要修正。

### 5.1 实际运行的打分公式（从 `data111/latex_graph_topology_report (1).json` 反查）

```
hub_score = bridge_score + authority_score + 60 × pagerank

其中：
  bridge_score   = num_modalities × 15 + out_to_elements × 2
  authority_score = in_from_paragraphs × 2
```

**此为 2026-03-03 生成 `data/latex_graph_hubs.json` 时实际使用的公式**，已通过 `data111` 中的 topology report `note_scoring` 字段和 hub 数值反推验证。每个 hub 输出两个分量字段：`bridge_score`（段落到多模态元素的桥接能力）和 `authority_score`（被段落引用的次数，即"权威度"）。

| 分量 | 计算方式 | 对应主管要求 |
|---|---|---|
| `bridge_score` | 覆盖模态数 × 15 + 出向元素边数 × 2 | ✅ 桥接功能加分、边数多的加分 |
| `authority_score` | 被段落引用次数 × 2 | ⚠️ 近似于"入度多的加分"，但偏向 authority sink 而非桥接 |
| `pagerank × 60` | 全图 PageRank × 60 | ✅ 结构性中心度 |
| **核心模块加分** | — | ❌ **从未计算**（Introduction/Main Results 正则加分在当前运行数据中缺失） |

### 5.2 重要修正：此前报告存在误报

上一版本报告中描述的 4 分量公式（含 `connectivity_score` 和 `core_module_score`）**存在于当前代码的 `compute_hubs()` 函数（第 813 行），但该代码路径从未在实际数据上运行过**。`data111/latex_graph_hubs (1).json` 中所有 60 个 hub 节点均无 `connectivity_score` / `core_module_score` 字段，因为数据实际由旧版逻辑生成。

因此主管的批评完全成立：**"核心模块加分（Introduction/Main Results）处于缺失状态"**——不是在报告里没说，而是在数据上真的没有算。

### 5.3 差距全图

| 主管要求 | 代码状态 | 数据状态 | 缺口等级 |
|---|---|---|---|
| 桥接功能加分 | ✅ 已实现 | ✅ 已落盘（`bridge_score`） | 无缺口 |
| 边数多的加分 | ✅ 已实现（`out_to_elements×2`） | ✅ 已落盘 | 无缺口 |
| 核心模块加分（Introduction/Main Results） | ✅ 代码已写（`core_module_score`，第 792-811 行） | ❌ **从未执行，数据中不存在** | **P0 缺口** |
| 打分透传到 pair 级输出 | ❌ 未实现 | ❌ JSONL 无 `pair_importance_score` 字段 | P0 缺口 |
| 节点重要性进入训练信号 | ❌ 未实现 | ❌ | P1 缺口 |
| 权重可调节（CLI 参数） | ❌ 硬编码 | — | P1 缺口 |

### 5.4 本周补齐计划

1. **P0（周二前）**：重跑 `analyze_latex_graph_topology.py`，启用含 `core_module_score` 的 4 分量公式，更新 `data/latex_graph_hubs.json`；在 `enrich_hub_candidates.py` 输出的 pair JSON 中新增 `pair_importance_score = (hub_score_A + hub_score_B) / 2`，并将 `bridge_score`、`core_module_score` 作为 sub-fields 透出
2. **P1（周三前）**：在 `analyze_latex_graph_topology.py` argparse 中暴露 `--bridge-weight / --connectivity-weight / --core-weight` 三个权重参数，支持消融实验
3. **P2（下周）**：在 JSONL 训练输出中加入 `node_importance` 字段，作为训练时 sample weight 候选

---

## 六、三项主管反馈的自查与修正

### 问题1：节点重要性打分体系"处于缺失状态"

**修正**：打分体系在代码层面已实现（见第五节），但存在两个问题：① 报告中完全未提及，造成"不存在"的误判；② 分数未透传到 pair 级输出和最终 JSONL，无法驱动训练时的 sample weighting。本周 P0 补齐 ① 的可见性，并启动 pair_importance_score 透传。

### 问题2：单文档闭环未稳，跨文档步子偏大

**自查**：当前 230 条 enriched pairs 中有 **74 条（32%）为 cross-doc**，而此时 label→MinerU ID 映射成功率仅 46%（270/500 条因 schema join 失败丢弃）。主管的判断是正确的：单文档的 ID 映射链路都还没稳定，74 条 cross-doc pair 只会增加排错变量。

**修正行动**：
- 本周三之前，label mapping 修复优先于 cross-doc 扩量
- 修复后的重跑策略：先只保留 `is_cross_doc=False` 的 intra-doc pairs 做闭环验证，达到 mapping rate ≥70% + QC pass ≥50% 后，再开放 cross-doc
- 在 `enrich_hub_candidates.py` 新增 `--single-doc-only` flag，与 CLAUDE.md 中 A2 workstream 对齐

### 问题3：QC 指标的执念 vs. 检索目标的偏移

**自查**：上版报告中 Recall@10/MRR 评估闭环被排在 ③ 号交付位（周五），前两项都是生成侧 bug 修复。主管的批评准确：当前所有 prompt 优化、QC 调整，如果没有检索命中率数据作为北极星，就是在黑盒里调参。

**修正行动**：
- **调整交付顺序**：将最小评估闭环（20-30条人工query + BM25 + Recall@10/MRR）提升为 **① 号交付（周二前）**，哪怕样本量只有 20 条，有数字比无数字有决定性意义
- **北极星指标明确化**：本周所有 QC 修复的合格标准不再是"pass rate ≥50%"，而是"修复后 Recall@10 不低于修复前"
- QC pass rate 降级为辅助监控指标，不再作为周报的核心 KPI

---

## 附录：当前模块状态

| 模块 | 状态 |
|---|---|
| LaTeX 多层图 | 稳定可复跑 |
| Hub bridge-first 候选 500 条 | 已产出；接入 enrichment 后覆盖 230/500 |
| MoDora enrichment 数据 | 1285/1316（97.6%） |
| Hub semantic summary | 230 条 enriched pairs |
| v4.5 生成链路 | 代码完成，smoke test 跑通 |
| 公司 API | 就绪 |
| **节点重要性打分** | **代码已实现（hub_score 四分量），pair 级透传和 CLI 权重参数本周补齐** |
| **单文档闭环** | **label mapping 46%，本周修复目标 ≥70%；cross-doc 暂缓扩量** |
| **端到端评估闭环** | **本周①必须交付（提前至周二），Recall@10/MRR 作为北极星** |
