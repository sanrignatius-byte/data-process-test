# M4 数据工程进度汇报

**日期**：2026-03-10
**汇报周期**：2026-03-04 → 2026-03-10
**本次重点**：公司 API 打通、MoDora 语义增强落地、v4.5 生成器（real-user 模板 + 噪声过滤 + 新 QC 链路）、Query 质量深度分析

---

## 0. 先说结论（3 分钟）

这周做了四件事，都有代码产物：

1. **公司 API（yunwu.ai）打通**：`generate_multihop_l1_queries.py` 现在支持 `--provider company`，通过 SSE 流式解析完成 token 统计，全量生成链路已就绪。
2. **MoDora 语义增强落地**：新增 `enrich_elements_modora.py`（751 行），对 figure/table/formula 三类元素生成 `[T]itle / [M]etadata / [C]ontent` 结构化描述；`enrich_hub_candidates.py` 同步升级，支持传入 enriched 数据并合成 `hub_semantic_summary`。
3. **v4.5 生成器三项核心升级**：C1 enrichment 噪声过滤器、B1/B2 real-user 问题风格（5 类模板 + `--query-style` 开关）、D1 `qc_real_user_query()` 函数。
4. **Query 质量深度人工分析**：对两批共约 30 条生成结果做了系统性解剖，定位出 yes_no_answer 退化、numeric 幻觉、formula_symbol_grounding 误判等三类问题，并给出了修复路径。

当前最大的缺口是：**enriched elements 的完整数据还没跑出来**（需要 `COMPANY_API_KEY` 有额度才能执行），所以 MoDora 上下文注入对生成质量的实际提升目前还没有数据验证。

---

## 1. 对齐上次建议：执行情况

| 上次计划 | 本周执行 | 结果 |
|---|---|---|
| C1：低质量 enrichment 噪声过滤 | `_is_noisy_enrichment()` 落地，7 类正则 + 15 字符门限 | ✅ 已上线到生成脚本 |
| B1/B2：real-user 模板 + `--query-style` | 5 类模板全写完，CLI 参数支持 academic/real_user/mixed | ✅ 已上线 |
| D1：`qc_real_user_query()` | 完成，放宽 6 类 academic 约束，新增 retrievability_score | ✅ 已上线 |
| 公司 API 打通 | `--provider company` 完整支持，含 SSE 解析 + token 日志 | ✅ 已上线 |
| A1：段落按 section 边界切分 | `_extract_paragraphs()` + `Paragraph` dataclass 落地 | ✅ 已上线 |
| A2：section 节点参与路径枚举 | 拓扑图 section label 含标题，paragraph 节点精细化 | ✅ 已上线 |
| C3：hub summary 压缩重写 | `build_hub_semantic_summary()` 聚合 enriched 描述 + edge context | ✅ 已上线，待数据验证 |
| MoDora enriched 数据实际生成 | 需要 API 额度，脚本已就绪但未跑完整批次 | ⬜ 待执行 |

---

## 2. MoDora 语义增强落地

### 2.1 整合结论

MoDora（SIGMOD 2026）的 CCTree 框架里有一个子思路值得借鉴：对每个非文本元素（图/表/chart）用类型特化的 LLM prompt 生成结构化描述，统一转为可检索文本表示。我们不迁移它的树结构和在线检索逻辑，只引入这一层"上游语义增强"。

两个项目的根本差异：MoDora 做推理时的 Document QA（零训练），我们做训练数据生成。检索框架层面各自有最优解，不存在谁替换谁的问题。

### 2.2 新脚本：`enrich_elements_modora.py`（751 行）

对 figure、table、formula 三类元素，各自用专用 prompt 生成：

```
[T] enriched_title  — 15-20词，描述性标题（非原 caption）
[M] enriched_metadata — keywords, element_type, key_metrics, axes 等结构化字段
[C] enriched_content — 200-300词，详细语义描述
```

三种 prompt 分别针对不同模态的信息密度：
- figure prompt：关注视觉结构（axes, data series, trends, anomalies）
- table prompt：关注数值和行列关系（metric values, comparisons）
- formula prompt：关注数学结构（variables, constraints, optimization objective）

支持 `--incremental` 增量模式（已处理过的跳过）、`--dry-run`、三种 provider。

### 2.3 `enrich_hub_candidates.py` 升级

新增 `--enriched-elements` 参数，接收 enriched elements JSON 后：
1. 把每个 candidate pair 两端元素的 enriched 字段注入（`enriched_title / enriched_metadata / enriched_content`）
2. 调用 `build_hub_semantic_summary()` 生成 `hub_semantic_summary`：拼接双端 enriched 描述 + edge context + keywords，作为每对候选的语义摘要

### 2.4 `generate_multihop_l1_queries.py` 上下文注入

新增 `build_enriched_context_section()`：当 pair 中有 enriched 字段时，自动在 prompt 末尾注入增强上下文。`_context()` 函数优先读 `enriched_content`，回退到原始 content。

**向后兼容**：无 enriched 字段时，行为与之前完全一致。

---

## 3. v4.5 生成器核心改动

### 3.1 C1：Enrichment 噪声过滤器

在查询生成前检测 enriched 字段质量，命中噪声模式则静默回退到原始 context：

```python
_ENRICHMENT_NOISE_PATTERNS = [
    r"[\u2460-\u2473\u25a0-\u25ff\u2600-\u26ff\u2700-\u27bf]",  # Unicode 装饰符号
    r"\b(glyph|icon|marker|symbol|bullet|arrow|checkmark|watermark)\b",  # OCR 噪声标签
    r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]",  # 圆圈数字列表标记
    r"\b(ocr error|illegible|unreadable|corrupted text|extraction failed)\b",
    r"^[\W\d\s]{0,20}$",  # 全非字母填充
]
# 另：长度 < 15 字符直接判噪声
```

### 3.2 B1/B2：Real-user 问题风格

5 类模板覆盖不同读者场景：

| 模板名 | 场景 | 问法举例 |
|---|---|---|
| `factual_lookup` | 快速查事实 | "What value does X take when Y?" |
| `summary` | 综合两侧证据 | "What do these two results tell us about X?" |
| `comparison` | 对比两侧结论 | "How do these results compare to X?" |
| `how_works` | 理解机制 | "Why does X happen in this setup?" |
| `what_if` | 假设推理 | "What would happen if X changed?" |

`--query-style` 参数控制：
- `academic`（默认）：原有 4 个 prompt，向后兼容
- `real_user`：按 `hash(pair_id) % 5` 确定性轮换 5 类模板
- `mixed`：按 `hash(pair_id) % 2` 各占 50%

### 3.3 D1：`qc_real_user_query()`

Academic QC 有 12 项检查，real-user 风格保留 4 项、去掉 8 项：

**保留**：meta_language、yes_no_question、single_element_answer、query_too_long（上限放宽至 35 词）

**去掉**：template_shortcut、weak_reasoning_connector、length_mix_missing、architecture_intent_missing、pseudo_multihop_parallel、templated_opening 等

**新增**：`retrievability_score`（0-3）——基于 anchor 词汇覆盖率和 text_evidence 存在与否的启发式评分。

---

## 4. 工程基础设施

### 4.1 公司 API 整合

`--provider company` 通过 `local_api_logger` 的 `wrap_requests_call` 发送 OpenAI-compat 格式请求，SSE 流式解析自动记录 token 统计。图像走 `image_url` 格式（yunwu.ai 是 OpenAI 代理）。

已配置为系统默认 provider，无需额外传参：`.env` 里已有 `COMPANY_API_KEY` 和 `COMPANY_API_URL`。

### 4.2 路径可移植性修复

之前脚本里有若干 `/cluster/...` 和 `/projects/...` 硬编码路径，本周统一改为：
1. `Path(__file__).resolve().parent.parent` 做项目根目录定位
2. 未找到时有 `/data/` 通用 fallback

### 4.3 大规模数据下载基础

新增两个下载脚本：
- `download_pdf_latex_pairs_snowball.py`：雪球采样，从种子论文出发通过引用关系扩展，支持崩溃恢复（checkpoint）
- `download_pdf_latex_pairs.py`：直接下载 PDF+LaTeX 源码对

这是为后续数据规模化预备的，目前尚未触发大规模扩采。

---

## 5. Query 质量人工分析：两批诊断

### 5.1 第一批（上轮生成，16 条）

分析后发现 3 类系统性问题：

**（1）"描述图—查表"退化结构**

Query 前半句把 figure 信息嵌进问句，实质上只需查表就能回答：
- 坏例：`"RULE shows steepest gender bias correlation — what correlation strength explains this?"` → figure 已告知结论，答案只是查 0.87
- 这类问题的 answer_balance 通常 < 0.25，已有对应 QC 指标，但未触发 fail

**（2）Real-user 数字幻觉**

0002/0003 两条 real_user 类问题，reasoning_chain 为空，但 answer 中引用了极精确的数字（91/106/59/44/134/115）：这些数字来源只有两种可能——视觉读取或幻觉。`text_evidence` 里没有任何数字，倾向判定为幻觉。

**（3）text_evidence 泄漏**

0012/0014/0015 的 answer 是 text_evidence 的改写，跨模态推理贡献接近 0。

### 5.2 第二批（本次迭代，16 条，QC 通过率 3/16 = 18.75%）

QC 率从上批 ~56% 大幅下降，根因分解：

| fail 类型 | 条数 | 根因 |
|---|---|---|
| `formula_symbol_grounding_missing` | 7 | 其中 4 条真实失败（answer 未引用公式符号），3 条疑似 checker 误判（answer 有 LaTeX 符号但仍 fail） |
| `yes_no_answer` | 3 | 提示词迭代把 "what X?" 改成了 "does X?"，触发 yes/no 检测，属于**直接回退** |
| `numeric_unsupported` | 2 | 新 QC 规则正确捕获的幻觉数字 |
| `weak_reasoning_connector` | 2 | 其中至少 1 条有争议（answer 包含 "because/resulting in" 等连接词） |

**最关键的一个发现**：yes_no_answer 的 3 条失败全部来自对上批 queries 的迭代改写——把 "what/which" 改成了 "does"。上批这几条是通过的，改了反而失败，说明提示词改动方向有问题。

### 5.3 两批分析后的修复清单

按优先级：

1. **P0**：恢复 yes/no 禁令到生成 prompt（这是之前版本已经修过的，改丢了）
2. **P0**：检查 `formula_symbol_grounded` checker 对 LaTeX 符号的匹配逻辑（疑有 3 条误判）
3. **P1**：real_user 路径强制 reasoning_chain 非空；answer 引用数字时只能来自 text_evidence
4. **P1**：figure+formula prompt 加约束：answer 必须引用至少一个具体数学符号
5. **P2**：answer_balance < 0.25 加入 soft fail 门限

---

## 6. 当前状态判断

### 已完成的能力建设

| 模块 | 状态 |
|---|---|
| LaTeX 多层图（backbone + cross-modal + citation） | 稳定可复跑 |
| Hub bridge-first 候选 500 条 | 已产出，可接入生成 |
| MoDora [T]/[M]/[C] enrichment 脚本 | 代码完成，待数据 |
| Hub semantic summary 合成 | 代码完成，待数据 |
| v4.5 real-user 生成链路 | 代码完成，已有小批次样本 |
| 公司 API 全链路 | 就绪，依赖有效 key |

### 当前最大缺口

1. **MoDora enriched 数据未产出**：`enrich_elements_modora.py` 还没有跑完整批次（需要 API 额度）。上下文注入对生成质量的实际提升目前没有数据支撑。
2. **yes_no 退化和 formula grounding checker 的 bug 还未修复**：第二批 QC 率 18.75% 是异常低值，主因是这两个问题，不是方向错了。
3. **评估闭环仍然缺失**：从 2 月 10 日起就是 P0，目前还是空的。所有迭代依据都是生成侧代理指标（QC pass rate），没有一个数字能回答检索效果有没有改善。

---

## 7. 下周计划（按优先级）

### P0：修 yes_no + formula_grounding checker，把 QC 率恢复正常水位
- 恢复 yes/no 禁令到 prompt
- 排查 `formula_symbol_grounded=false` 在 LaTeX 符号存在时仍失败的原因
- 预期：QC 率应恢复到 40-50%+ 水位

### P0：执行 enriched elements 全量生成
- 用 `enrich_elements_modora.py` 对 1316 个 multimodal elements 跑一批
- 跑完后接入 enrich_hub_candidates → generate 的完整 pipeline，对比有 / 无 enrichment 的 QC 通过率差异

### P1：建立最小评估闭环
- 这件事拖了快一个月了，必须做
- 20-30 条人工测试 query + BM25 baseline + Recall@10/MRR
- 用当前 pass 集（来自 v4.4 run1 的 113 条 + 新批次）作为正例，不需要完整标注

### P2：用 500 条 hub 候选跑完整生成批次
- 目前 hub 候选 500 条没有完整跑过一次生成，只有小批次 smoke test
- 跑一次全量，看 pass rate 和 pair_type 分布是否合理

---

## 附：本周关键文件变动

| 文件 | 改动性质 | 代码量 |
|---|---|---|
| `scripts/enrich_elements_modora.py` | **新增**，MoDora enrichment 主脚本 | 751 行 |
| `scripts/generate_multihop_l1_queries.py` | **升级到 v4.5**，C1+B1+B2+D1 四项 | +569 行（总 2614 行） |
| `scripts/enrich_hub_candidates.py` | 升级，`--enriched-elements` + hub summary | +104 行（总 639 行） |
| `src/parsers/latex_reference_extractor.py` | 段落粒度重构，`Paragraph` dataclass | 较大改动 |
| `scripts/analyze_latex_graph_topology.py` | section 节点升级，paragraph 精细化 | 较大改动 |
| `main.py` | 新增，公司 API 连通性测试脚本 | 141 行 |
| `scripts/download_pdf_latex_pairs_snowball.py` | 新增，崩溃恢复式雪球采样下载 | — |
| `docs/MODORA_INTEGRATION_ANALYSIS.md` | 新增，MoDora 整合分析文档 | — |
| `docs/PATENT_TECHNICAL_SUMMARY.md` | 新增，专利技术总结（v1.1 含 8 项补充创新） | — |
