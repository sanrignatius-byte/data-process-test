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

## 附录：当前模块状态

| 模块 | 状态 |
|---|---|
| LaTeX 多层图 | 稳定可复跑 |
| Hub bridge-first 候选 500 条 | 已产出；接入 enrichment 后覆盖 230/500 |
| MoDora enrichment 数据 | 1285/1316（97.6%） |
| Hub semantic summary | 230 条 enriched pairs |
| v4.5 生成链路 | 代码完成，smoke test 跑通 |
| 公司 API | 就绪 |
| **端到端评估闭环** | **缺失，本周③必须交付** |
