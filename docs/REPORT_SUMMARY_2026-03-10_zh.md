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

### 病灶 1：`formula_symbol_grounding_missing` 假阳性拦截（主因，7 条 fail，figure+formula 全军覆没）

QC 函数 `_formula_symbol_hit()` 从 formula 的 `caption/content` 字段提取 LaTeX `$...$` 区域内的符号词（如 `epsilon`、`p(h_m|a)` 等），然后检查 answer 是否提到其中至少一个。

**根因**：当 formula 内容以 plain text 存储（无 `$...$` wrapper）时，提取逻辑 fallback 到整段文本，提取出的 terms 与模型 answer 中的自然语言表述无法匹配，导致 false positive。16 条样本中 figure+formula 共 8 条，其中 **7 条**触发此问题，pass 仅 1/8（12.5%）。

已定位到 `generate_multihop_l1_queries.py` 第 1327-1363 行（`_extract_formula_symbol_terms` + `_formula_symbol_hit`）。需将 plain-text 数学符号写法（下划线记法 `epsilon_m`、函数记法 `P(A|B)` 等）加入匹配，并对无 `$` 内容的 formula 降低 grounding 要求。

**修复预期**：figure+formula pass 率从 12.5% → ≥ 50%。

### 病灶 2：Yes/No 句式退化（3 条 fail）

本轮 prompt 迭代将 "what/which X?" 改成了 "does X?"，触发 `yes_no_answer` 检测。上批同类问题是通过的，改了反而失败。已定位到 prompt 改写的具体位置，回滚即可。

### 病灶 3：`numeric_unsupported` 存在多模态盲区（2 条 fail，当前不能直接定性为模型幻觉）

当前 `numeric_unsupported` 的判断口径只验证 `text_evidence`，这在 multimodal RAG 场景下是站不住的：精确数字本来就可能只存在于 table / figure / enriched 描述，而不在正文片段里。因此，这两条样本不应先被定性为“模型幻觉”，而应先被定性为**QC 证据池不完整**。

我复核了 0002 / 0003 两条样本：`text_evidence` 的确不含数字；现有写回的 `enriched_content` 也没有把 `91 / 106 / 59 / 44` 显式展开。这说明当前更准确的结论是：**现有验证池不足以下支持或反驳这些精确数字**。因此 P0 修复方向不是先约束模型“别报数字”，而是先把 `caption / content / enriched_content` 一并纳入 numeric validation pool，再区分真幻觉与 QC 误杀。

### 两批对比与精确故障分布

| 批次 | 条数 | QC pass | pass 率 | 主因 |
|---|---|---|---|---|
| 上批（v4.4） | 16 | ~9 | ~56% | — |
| 本批（v4.5 smoke） | 16 | 3 | **18.75%** | `formula_symbol_grounding_missing`（7）+ `yes_no_answer`（3）+ `numeric_unsupported`（2）+ `weak_reasoning_connector`（2）+ `length_mix_missing`（2）+ `template_shortcut`（1） |

**pair_type 细分**（来自 `data111/l1_img_run_20.jsonl`，各 8 条）：

| pair_type | total | pass | fail 主因 |
|---|---|---|---|
| figure+formula | 8 | 1（12.5%） | `formula_symbol_grounding_missing` 7 条（几乎全军覆没） |
| figure+table | 8 | 2（25%） | `yes_no_answer` 3、`numeric_unsupported` 2、`weak_reasoning_connector` 2 |

本批 3 条 pass query 的 anchor_leak_jaccard 均低于 0.17，answer_balance 在 0.25-0.44 之间。样本量太小（3 条），不足以断言 enriched context 对质量有正面作用，但至少没有引入新的 leakage 模式，这是一个初步的正向信号。

更重要的是：**QC pass rate 只是生成侧代理指标，不是最终 KPI。** 按 `DISCUSSION_LOG.md`、`CLAUDE.md` 和既定计划，本周周五前必须拿到最小评估闭环的 BM25 baseline、Recall@10、MRR；否则这轮所有 prompt / QC 优化都只能算局部调试，不能算方法有效性验证。

### 本轮生成样本展示（来自 `data111/l1_img_run_20.jsonl`，共 16 条）

#### ✅ 通过样本（3 条）

**[Pass-1] figure+table | academic | 1904.03035（debiasing LSTM 论文）**
```
Hub 语义摘要（生成 query 所基于的 enriched 上下文）：
  [TABLE A] Effect of lambda on bias metrics and language-model perplexity:
  Ablation over λ compares bias-related summary statistics (μ, σ, β) under fixed
  vs infinite context settings and reports Penn Treebank perplexity (Ppl.).
  λ=0.5 yields the best perplexity (116.19) and the lowest β (0.14)...

Query:  The three-layer LSTM applies bias minimization to embedding matrices —
        which specific bias metrics demonstrate the effectiveness of this architectural choice?

Answer: The table's β (beta) and σ (sigma) columns provide bias measurements
        that quantify the effectiveness of the LSTM's embedding bias minimization,
        because these metrics directly measure the gender bias reduction achieved by
        the lambda-controlled regularization shown in the figure's architecture.

QC metrics: anchor_jaccard=0.12, answer_balance=0.44
```
> 评注：query 给出了"建筑前提"（三层 LSTM + bias minimization），要求学生从 table 指出具体度量指标，从 figure 建立架构对应；answer 准确引用 β/σ 并连接了正则化参数 λ，属于典型双证据闭环。

---

**[Pass-2] figure+formula | academic | 1801.07593（对抗去偏论文）**
```
Query:  How does the geometric arrangement of vectors in the projection operation
        correspond to the dual adversarial terms in the gradient update?

Answer: The vector diagram illustrates the geometric decomposition where proj_h g
        isolates the component of the predictor gradient orthogonal to the adversarial
        direction. This corresponds to the middle projection term in the formula that
        removes adversary-aligned components, while the final alpha term activates
        only when the adversary succeeds...

QC metrics: anchor_jaccard=0.17, answer_balance=0.25
```
> 评注：这是 16 条中唯一通过的 figure+formula，质量也是本批最高——几何直觉（向量投影图）与数学形式（梯度更新公式）精准互通，reasoning chain 完整。

---

#### ❌ 失败样本（按病灶分类）

**[Fail-A] formula_symbol_grounding_missing × 7 — 根因示例**
```
pair: 1802.08139_figure_2 × 1802.08139_formula_3（因果公平论文）

Query:  Which pathways from A to Y does the penalty term block?

Answer: The penalty term blocks both the direct A→Y pathway and the indirect A→M→Y
        pathway because these represent unfair influence channels. The causal graph
        explicitly shows these two green-highlighted paths as the problematic routes
        that must be eliminated, whereas the formula's penalty...

QC: formula_symbol_grounded=False, formula_symbol_term_count=6 ← checker 提取到 6 个符号词但 answer 未命中任何一个
```
> 根因：checker 从 formula caption/content 的 plain text 中提取出 `Y_0`、`Y_1`、`delta` 等符号词，但模型 answer 用自然语言描述路径（"A→Y pathway"），而非重复符号名，导致误判。Answer 语义完全正确，这是 checker 的精确匹配过窄。

---

**[Fail-B] yes_no_answer × 3 — prompt 改写引入退化**
```
pair: 1904.03035_figure_1 × 1904.03035_table_1

Query:  Regularization flattens bias curves — does perplexity increase with stronger lambda values?
Answer: Yes, the table shows perplexity increases from lower to higher lambda values...

QC: yes_no_answer ← "does X?" 句式触发
```
> 根因：本轮 prompt 迭代将 WH 问句（"what/which X?"）改成了极性问句（"does X?"），是一行 prompt 改写造成的退化，回滚即可。

---

**[Fail-C] numeric_unsupported × 2 — QC 证据池多模态盲区**
```
pair: 2005.07293_figure_4 × 2005.07293_table_5（公平性偏好调查论文）

Query:  How do the rating distributions for equity versus parity compare to the actual preference counts?
Answer: ...in Scenarios 3 and 4, more people actually preferred parity solutions
        (91 and 106) over equity solutions (59 and 44)...

QC: unsupported_answer_numbers=['106', '44', '59', '91']
    → text_evidence 中不含这四个数字
```
> 根因：数字 91/106/59/44 来自 table 的格子，而 `numeric_unsupported` checker 只验证 `text_evidence` 文本片段。这是 QC 证据池不完整，不是模型幻觉。

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
