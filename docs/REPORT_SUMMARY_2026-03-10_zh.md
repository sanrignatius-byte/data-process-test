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

### 最大危机与澄清

Smoke test QC 通过率暴跌至 **18.75%（3/16）**，比上批约 56% 大幅下降。

已定位到三个具体问题（见第四节）：一是 formula checker 的 plain-text notation 匹配 bug，二是 yes/no 禁令被本轮 prompt 改写意外删掉，三是 `numeric_unsupported` 的验证池只看 `text_evidence`，存在明显多模态盲区。**前两项是确定性 bug，第三项是 QC 口径问题。** 这轮下跌不是模型能力退化，也不是 enrichment 方向有问题，而是生成门禁与验证逻辑没跟上数据形态。

### 本周 P0 交付序列

| 顺序 | 事项 | 交付标准 |
|---|---|---|
| **①（周一前）** | 修复 yes/no 禁令 + formula checker plain-text 匹配 + `numeric_unsupported` 验证池扩到 `caption/content/enriched_content` | 对同一批 16 条重跑，pass 率 ≥ 50%，并复核 QC 误杀是否清除 |
| **②（周三前）** | 排查 270 条 `skipped_no_mapping` 根因并修复 | mapping rate 提升至 ≥ 70% |
| **③（周五前）** | 建立最小评估闭环 | 20-30 条人工 query + BM25 + Recall@10/MRR 有数字 |

评估闭环（③）已连续两周列为 P0 但未完成，原因是前两项 bug 修复占用了预期外的排查时间。本周①②必须在前三天解决，为③留出时间，不再允许顺延。**如果周五仍拿不出 Recall@10 / MRR，这周所有生成侧优化都只能算局部排障，不算闭环交付。**

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

## 四、深度诊断：为何通过率跌至 18.75%？

Smoke test 16 条 query，3 pass / 13 fail。**已定位到三个具体问题，且都有明确修复动作。**

### 病灶 1：Formula Checker 假阳性拦截（主因，7 条 fail，其中 ≥3 条确认误判）

Checker 要求 answer 包含 LaTeX `$...$` 格式的数学符号。但模型输出普遍使用平文写法（`epsilon_m`、`p(H_m|A)`、`f(A,C)`、`g+h`），触发 false positive。

已手动核查 7 条 fail 中的 3 条：answer 实质上正确引用了数学变量，只是格式不符合 checker 的强匹配规则，确认为 checker bug。剩余 4 条混有真实 grounding 缺失，需要修复后重跑区分。

**修复预期**：将 plain-text notation 纳入 grounding 匹配，figure+formula pass 率可从 12.5% 推至 ≥ 50%。

### 病灶 2：Yes/No 句式退化（3 条 fail）

本轮 prompt 迭代将 "what/which X?" 改成了 "does X?"，触发 `yes_no_answer` 检测。上批同类问题是通过的，改了反而失败。已定位到 prompt 改写的具体位置，回滚即可。

### 病灶 3：`numeric_unsupported` 存在多模态盲区（2 条 fail，当前不能直接定性为模型幻觉）

当前 `numeric_unsupported` 的判断口径只验证 `text_evidence`，这在 multimodal RAG 场景下是站不住的：精确数字本来就可能只存在于 table / figure / enriched 描述，而不在正文片段里。因此，这两条样本不应先被定性为“模型幻觉”，而应先被定性为**QC 证据池不完整**。

我复核了 0002 / 0003 两条样本：`text_evidence` 的确不含数字；现有写回的 `enriched_content` 也没有把 `91 / 106 / 59 / 44` 显式展开。这说明当前更准确的结论是：**现有验证池不足以下支持或反驳这些精确数字**。因此 P0 修复方向不是先约束模型“别报数字”，而是先把 `caption / content / enriched_content` 一并纳入 numeric validation pool，再区分真幻觉与 QC 误杀。

### 两批对比

| 批次 | 条数 | QC pass | pass 率 | 主因 |
|---|---|---|---|---|
| 上批（v4.4） | 16 | ~9 | ~56% | — |
| 本批（v4.5 smoke） | 16 | 3 | **18.75%** | formula checker bug（7）+ yes/no 回退（3）+ numeric validation blind spot（2） |

本批 3 条 pass query 的 anchor_leak_jaccard 均低于 0.17，answer_balance 在 0.25-0.44 之间。样本量太小（3 条），不足以断言 enriched context 对质量有正面作用，但至少没有引入新的 leakage 模式，这是一个初步的正向信号。

更重要的是：**QC pass rate 只是生成侧代理指标，不是最终 KPI。** 按 `DISCUSSION_LOG.md`、`CLAUDE.md` 和既定计划，本周周五前必须拿到最小评估闭环的 BM25 baseline、Recall@10、MRR；否则这轮所有 prompt / QC 优化都只能算局部调试，不能算方法有效性验证。

---

---

## 五、节点重要性打分体系（本报告此前遗漏的核心模块）

> **上周主管明确要求**：定义规则量化节点重要性，具体维度包括：承上启下桥接功能加分、包含核心模块（Introduction/Main Results）加分、边数多的加分。本节补充说明该体系的实际落地状态。

### 5.1 当前打分公式（已在代码中实现，`analyze_latex_graph_topology.py:813`）

```
hub_score = 0.40 × bridge_score
          + 0.35 × connectivity_score
          + 0.25 × core_module_score
          + 20.0 × pagerank
          - penalty
```

各分量含义：

| 分量 | 权重 | 计算方式 | 对应主管要求 |
|---|---|---|---|
| `bridge_score` | **40%** | `bridge_role × 100`；`bridge_role=1.0` 当段落节点出边覆盖 ≥2 种元素模态（figure/table/formula）；`=0.5` 仅覆盖1种 | ✅ 桥接功能加分 |
| `connectivity_score` | **35%** | `min(1, total_degree/degree_norm + cross_type_edges/degree_norm) × 100`；degree_norm 取全图 90 百分位 | ✅ 边数多的加分（自适应归一化） |
| `core_module_score` | **25%** | 对节点 section_title + label + text_snippet 做正则匹配：introduction=1.0, main_result/experiment=0.9, method/framework=0.8, conclusion=0.6, related_work=0.3 | ✅ 核心模块加分 |
| `pagerank` | 系数20 | 全图 PageRank，捕捉全局引用权威性 | ✅ 结构性中心度 |
| `penalty` | -20 | 当 in_deg > out_deg×2 且跨模态出边 ≤1 时扣分（authority sink 惩罚） | ✅ 抑制虚假枢纽 |

figure 节点还有额外 `core_module_score` 加成：含 architecture/framework/overview/pipeline 等词 → 满分1.0；含 result/performance/comparison/ablation → 0.8。

### 5.2 当前状态与差距

**已落地**：公式已在 `compute_hubs()` 函数中实现，每个 hub 节点均输出 `bridge_score / connectivity_score / core_module_score / hub_score` 四个字段，可在 `data/latex_graph_hubs.json` 中查询。

**工程化差距（主管要求 vs. 当前实现）**：

| 主管要求 | 当前状态 | 缺口 |
|---|---|---|
| 打分体系可解释 | 四字段全量输出，注释说明在 report JSON | ⚠️ 报告中未透出，不可视 |
| 权重可调节 | 权重硬编码在函数体 | ❌ 无 CLI 参数，不支持调参实验 |
| 打分驱动候选筛选 | hub_score 已用于 top-K hub 排序 | ⚠️ 但 500 条候选的 **pair 级重要性分**（两端 hub_score 聚合）尚未暴露到 enriched pair 输出 |
| 节点重要性进入训练信号 | 未接入 | ❌ 无 `importance_weight` 字段在 JSONL 输出中 |

### 5.3 本周补齐计划

1. **P0（周二前）**：在 `enrich_hub_candidates.py` 输出的 pair JSON 中新增 `pair_importance_score` 字段，定义为 `(hub_score_A + hub_score_B) / 2`，并将 `bridge_score` 和 `core_module_score` 作为 sub-fields 透出
2. **P1（周三前）**：在 `analyze_latex_graph_topology.py` 的 argparse 中暴露 `--bridge-weight / --connectivity-weight / --core-weight` 三个权重参数，支持消融实验
3. **P2（下周）**：在 JSONL 训练输出中加入 `node_importance` 字段，作为训练时的 sample weight 候选

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
