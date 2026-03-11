# M4 数据工程进度汇报

**日期**：2026-03-11　　**周期**：2026-03-03 → 2026-03-11

---

## 0. 先说结论

本周完成了从"元素语义增强"到"枢纽摘要合成"再到"查询生成"的完整闭环升级，核心结果：

| 里程碑 | 数字 |
|---|---|
| MoDora 元素语义增强 | 1285 / 1316 元素（97.6%），覆盖 76 篇文档 |
| Hub 语义摘要合成 | 230 条 enriched pairs，全部带 `hub_semantic_summary` |
| v4.5 Query 质量（img_20） | 整体 pass **65.6%**（21/32）；figure+table 达 **90%** |

相比上一版 v4.4（pass 44.8%），本轮整体提升约 21 个百分点。figure+table 双模态已达可直接入训练集的水位。

---

## 1. 方法演进：MoDora 上游语义增强

### 核心思路

借鉴 MoDora [T]/[M]/[C] 的思路，在生成多跳 query 之前，对 figure / table / formula 三类元素先做一轮专用 prompt 的结构化语义描述（enrichment），再将 enriched 字段注入 query 生成的 context。这比直接用原始 caption 更有语义密度，有助于模型"看懂"元素而非仅复述标题。

**判断**：我们多层图（citation + cross-modal + backbone）对跨文档/跨模态表达力优于 CCTree 树合并，因此只采用 MoDora 的"上游语义增强"思路，不迁移其树结构或在线检索框架。

### 三级 Pipeline

```
enrich_elements_modora.py         →  multimodal_elements_enriched.json
    ↓
enrich_hub_candidates.py          →  hub_candidates_enriched_v2.json
    ↓
generate_multihop_l1_queries.py   →  l1_dual_evidence_queries_*.jsonl
```

**第一级（元素增强）**：对每个 figure / table / formula 独立调用专用 prompt，输出 `enriched_title` / `enriched_metadata` / `enriched_content` 三个新字段，不覆盖原字段。

**第二级（Hub 摘要）**：聚合两端元素的 enriched 描述 + bridge context + keywords，压缩为 50-80 词的 `hub_semantic_summary`（规则压缩，无额外 LLM 调用）。

**第三级（查询生成）**：`_context()` 优先读取 `enriched_content`；检测到 OCR 残留、Unicode 装饰符等低质字段时，噪声过滤器（`_is_noisy_enrichment()`）静默回退原始 context。

---

## 2. 图结构升级（A workstream）

### 段落按 section 边界切分（A1）

`_extract_paragraphs()` 在每个 `\section / \subsection / \subsubsection` 命令处先 flush 当前 block，段落不再跨 section 边界合并。这使段落语义更纯，downstream 路径枚举中的"桥接段落"更精准。

### Section 节点参与路径枚举（A2）

新增 Strategy 4：通过 `section_contains_element` 边，枚举 `[elem_A → sec_node → elem_B]` 的 2-hop 路径，条件是 section 节点同时含 ≥2 种模态的元素。此前路径枚举只走 backbone/element_ref 边，现在加入 section 节点后，从同一 section 内部直接桥接两个不同模态的元素成为可能。

新增 `--single-doc-only` flag，支持单文档闭环验证时排除跨文档候选，降低变量。

### Page span bug 修复

`build_real_page_index` 中，当 `position_idx == 0`（MinerU parser 已知 bug）时，改用元素的 `number` 字段作为排序 fallback，确保第 N 个 figure 正确对应 content_list.json 中的第 N 个 image 条目，而非按字母序错位。跨文档候选的 `page_span` 统一设为 `None`（跨论文的页码不可比较）。

---

## 3. Query 生成升级（B / D workstream）

### Real-user 查询风格（B1/B2）

新增 5 类贴近真实用户的问题模板（`factual_lookup / summary / comparison / how_works / what_if`），`--query-style` 支持 `academic`（默认，向后兼容）/ `real_user` / `mixed` 三档切换。Real-user 模板特点：

- 自然英文，≤25 词，无学术腔
- 仍强制 dual-element 覆盖
- mixed 模式按固定周期轮换，保证分布可控

### Node group 支持（B3）

`enrich_hub_candidates.py` 改为收集路径中**所有**不同 element 端点，存为 `node_group` 列表（1-3 个元素）。原 `element_a / element_b` 字段向后兼容保留。这使 3-hop 路径能完整传递所有中间元素的语义，而不再只看首尾两端。

### Real-user QC（D1）

新增 `qc_real_user_query()`，保留 4 项核心 QC（dual_evidence / leakage / balance / length），放宽 8 项 academic 专用约束（yes/no 禁令、模板动词检测等），新增 `retrievability_score` 启发式评分（检索友好度）。Academic 路径仍走原有 `qc_multihop_query()`，两轨并行。

---

## 4. 查询质量结果（img_20，32 条）

来自 `data111/l1_img_run_20.jsonl`，query_id 0000–0031。

### 总体与分类

| pair_type | style | 总条 | pass | pass 率 |
|---|---|---|---|---|
| figure+table | academic | 20 | 18 | **90%** |
| figure+table | real_user | 6 | 5 | **83%** |
| figure+formula | real_user | 6 | 0 | 0% |
| **合计** | | **32** | **21** | **65.6%** |

### 版本趋势

| 版本 | 条数 | pass | pass 率 |
|---|---|---|---|
| v4.2（2026-02-22） | 236 | 152 | 64.4% |
| v4.4（2026-03-03） | 252 | 113 | 44.8% |
| **v4.5 img_20（本批）** | **32** | **21** | **65.6%** |

v4.4 的下滑来自 hub 候选来源切换导致难度增加；v4.5 接入 enriched context 后回升至 65.6%，figure+table 已稳定在 90%。

### 通过批次质量指标

| 指标 | 状态 |
|---|---|
| anchor_leak_jaccard | ✅ 全部 < 0.15 |
| short+long 配对覆盖 | ✅ 所有 academic 对均有短长两条 |
| dual_evidence / cross_modal 标记 | ✅ 全部正确 |
| reasoning_chain | ✅ academic 对全有；real_user 按设计为空 |

### 当前瓶颈：figure+formula（0% pass）

`formula_symbol_grounding_missing`：real_user 模板直接引用 LaTeX 裸符号（`c_fp`、`μ_t` 等），QC `_formula_symbol_hit()` 无法将其与 answer 的自然语言描述匹配。

修复方向：模板约束符号引用时加括注释义，如 `c_fp (false-positive cost rate)`；同步调整 `_extract_formula_symbol_terms` 对 plain-text 公式的匹配逻辑（第 1327–1363 行）。预期修复后 figure+formula pass 率 ≥ 50%。

次要问题：`single_element_answer` 5 条（answer_balance < 0.25），修复方向为 prompt 层强制两端各有一句含具体数值的句子。

---

## 5. 节点重要性打分体系

当前运行公式（已在 `data/latex_graph_hubs.json` 落盘）：

```
hub_score = bridge_score + authority_score + 60 × pagerank

bridge_score    = num_modalities × 15 + out_to_elements × 2
authority_score = in_from_paragraphs × 2
```

| 分量 | 对应设计目标 | 状态 |
|---|---|---|
| `bridge_score`（桥接能力） | 覆盖模态数 + 出向元素边数 | ✅ 已落盘 |
| `authority_score`（被引度） | 入度加分 | ✅ 已落盘 |
| `pagerank × 60`（结构中心度） | 全图 PageRank | ✅ 已落盘 |
| `core_module_score`（核心 section 加分） | Introduction/Main Results 正则加分 | ⚠️ 代码已写，本周重跑补齐 |
| `pair_importance_score` 透传 | 下游 JSONL 携带打分 | ⚠️ 本周补齐 |

当前 bridge hubs 60 个，`top-60 hubs 100% category=bridge`（authority sink 已从排名中清除）。

---

## 6. 下周计划

| 优先级 | 事项 | 交付标准 |
|---|---|---|
| **P0** | 建立最小评估闭环 | 20-30 条人工 query + BM25 + Recall@10/MRR 有数字 |
| **P0** | 修复 figure+formula 符号 grounding | figure+formula pass 率 ≥ 50% |
| **P1** | 修复 270 条 `skipped_no_mapping`（label-style → ordinal ID join） | mapping rate ≥ 70% |
| **P1** | 重跑 `analyze_latex_graph_topology.py` 启用 `core_module_score`，透传 `pair_importance_score` | hubs JSON 含 4 分量，JSONL 有字段 |
| **P2** | 修复后全量重跑，产出可训练子集 | figure+table 入训练集；formula 对 QC 通过 |

当前所有 prompt / QC 优化如果没有检索命中率作为北极星，只能算生成侧代理指标。**评估闭环是本周最高优先，无论其他项进展如何，周五前必须有 Recall@10/MRR 数字。**
