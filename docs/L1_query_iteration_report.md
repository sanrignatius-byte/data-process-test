# L1 Cross-Modal Query 迭代改进报告

## 1. 任务目标

为 M4 多模态文档检索系统生成 **L1 intra-document cross-modal queries**：每条 query 必须**同时依赖图片和文本**才能回答，用于训练 multimodal document retrieval embedding。

输入：351 figure-text pairs（73 篇 arXiv 论文，MinerU 解析）

---

## 2. 三轮迭代总览

| | v1 | v2 | v3 (最终) |
|---|---|---|---|
| **模型** | Qwen3-VL-30B (本地 4×A5000) | Qwen3-VL-30B (本地) | Claude Sonnet 4.5 (API) |
| **解析成功率** | 286/335 (85.4%) | 21/335 (6.3%) | 334/335 (99.7%) |
| **产出 queries** | 604 | 33 | **974** |
| **QC 通过率** | — | — | 97.2% (28 dropped) |
| **Clean rate** | — | — | **84.3%** |
| **花费** | GPU 时间 ~21min | GPU 时间 ~2h (失败) | **$4.59** |

---

## 3. 各版本详细分析

### v1：基线（Qwen3-VL 本地推理）

**做法**：基础 prompt，让模型为每张图生成 cross-modal queries。用 vLLM TP=4 在 4×A5000 上跑。

**结果**：604 条 queries，但质量有严重问题：

| 问题 | 严重程度 | 数据 |
|------|---------|------|
| 缺少 visual anchor | 严重 | 63.4% 的 query 没有任何视觉锚点 |
| "看图说话"而非跨模态推理 | 严重 | 很多 query 不看文本也能回答 |
| Meta-language 泛滥 | 中等 | "According to the text", "the figure shows" |
| Why/How 占比过高 | 中等 | 37.3% 是解释型，难以 ground |
| 类型字段污染 | 中等 | `requires_figure` 有 2 个非 bool 值 |
| 绝对路径不可复现 | 轻微 | image_path 包含服务器绝对路径 |

**Reviewer 评价**：*"这些是看图说话，不是真正的 cross-modal reasoning"*

### v2：Prompt 重设计（仍用 Qwen3-VL）

**改进措施**：
- 添加 Blindfold Test 要求（遮住图/文任一都不能答）
- 要求明确 `visual_anchor` 和 `text_evidence` 字段
- 添加 banned patterns 列表
- 新增 `validate_queries.py` QC 脚本
- 定义 4 种 query type（value_context / comparison_explanation / anomaly_cause / visual_definition）

**结果**：Thinking 模式的 `<think>` 块消耗了大量 output token，导致只有 21/335 成功解析。但成功的 33 条质量确实提升：

| 指标 | v1 | v2 |
|------|-----|-----|
| Visual anchor 有 | 36.6% | **75.8%** |
| 有具体数值 | ? | **63.6%** |

**新问题（Reviewer 二次反馈）**：

| 问题 | 描述 |
|------|------|
| "拼盘"非"融合" | query 用 "and" 拼接两个子问题，没有真正融合 |
| Meta-language 残留 | "the text states" 仍然出现 |
| Text evidence 复用 | 同一图片的 3 条 query 引用同一段文本 |
| Query 太长 | 平均 ~29 词，像考试题 |
| Comparison 太少 | 只占 12% |
| 解析率太低 | Thinking 模式吃 token，6.3% 成功率不可用 |

### v3：融合 Prompt + Claude API（最终版）

**根本性改进**：

1. **换模型**：Qwen3-VL 本地 → Claude Sonnet 4.5 API
   - 原因：Qwen3-VL 在 4×A5000 上 OOM（max_model_len=16384 挂死），且 Thinking 模式浪费 token
   - 效果：99.7% 解析率，$4.59 总花费

2. **Prompt 重构**：
   - "每条 query 只能是一个问题，最多 25 词，不准用 and 拼接子问题"
   - 完全禁止 meta-words："text", "caption", "figure", "paper", "section", "according to" 等
   - 每条 query 必须引用**不同的**文本段落
   - 偏好 comparison/trend/anomaly，减少纯读数
   - 提供 BAD/GOOD 对比示例

3. **QC Pipeline 加强**：
   - Meta-language 检测（anywhere in query, not just prefix）
   - Visual anchor 最低 5 字符
   - Text evidence 最低 50 字符
   - 自动归一化 image_path 为相对路径

---

## 4. v3 最终质量指标

### 基础统计
- **974 条 queries**，覆盖 334 张图、73 篇论文
- **QC 通过率 97.2%**（28 条被过滤）
- **Validation clean rate 84.3%**（821/974 无任何 warning）

### 关键质量指标对比

| 指标 | v1 | v3 | 改善 |
|------|-----|-----|------|
| Visual anchor 有 | 36.6% | **74.8%** | +38.2pp |
| 有具体数值 | — | **59.4%** | — |
| Comparison 类型占比 | 12% | **41.9%** | +29.9pp |
| 平均 query 长度 | ~29 词 | **17.9 词** | -38% |
| Meta-language | 大量 | **0**（QC 过滤） | 清除 |
| 绝对路径 | 是 | **否**（自动归一化） | 修复 |

### Query Type 分布

```
comparison_explanation  408  (41.9%)  ████████████████████
value_context          319  (32.8%)  ████████████████
anomaly_cause          129  (13.2%)  ██████
visual_definition      118  (12.1%)  ██████
```

### Figure Type 分布

```
plot          694  (71.3%)  ██████████████████████████████████
diagram       201  (20.6%)  ██████████
example        51  ( 5.2%)  ██
architecture   12  ( 1.2%)  █
photo           7  ( 0.7%)
table           6  ( 0.6%)
```

### Query 示例

**comparison_explanation（融合型）**：
> "Why does the solid blue curve overtake the dashed red one only after epoch 12, given that both use the same base architecture?"

**anomaly_cause（异常型）**：
> "Why does 'syria' dominate the tag cloud when the dataset uses a boundary box causing strong Asian bias?"

**value_context（值+语境型）**：
> "Does RLR's 0.68 accuracy at fairness=0.95 support the claim that repair performance varies across algorithms?"

---

## 5. 遗留问题与后续计划

| 问题 | 状态 | 计划 |
|------|------|------|
| 74.8% visual anchor（非100%） | 可改进 | validation 已标注，可人工审核剩余 25% |
| Figure type 偏 plot (71.3%) | 数据限制 | 受限于 arXiv 论文本身图片类型分布 |
| Table 模态几乎没有 (0.6%) | 数据限制 | Table 在 MinerU 中多解析为文本而非图片 |
| 数值答案可靠性 | 待验证 | MLLM 生成的数值可能有幻觉，可抽样验证 |

### 5.1 当前诊断（2026-02-09 扫描）

基于 `data/l1_cross_modal_queries_v3.jsonl` 的快速审计：

| 诊断项 | 数量 | 占比 |
|------|------|------|
| **Warning 总数** | 153 | 15.7% |
| `no_visual_anchor` | 94 | 9.7% |
| `ungrounded_why` | 81 | 8.3% |
| `text_evidence` 提到 **Table #** | 36 | 3.7% |
| 任意位置提到 table（evidence/query/answer） | 38 | 3.9% |

**解读**：剩余问题主要来自“视觉锚点不落地 + why 问句悬空”。表格信息更多体现在 **文本证据引用**，而非 query 自身显式点名表格。

---

## 6. 技术决策总结

| 决策 | 原因 |
|------|------|
| 从 Qwen3-VL 本地 → Claude API | GPU OOM + Thinking 模式 token 浪费，API 99.7% 成功率 |
| Sonnet 4.5 而非 Opus | 性价比：$4.59 处理 335 张图，质量足够 |
| 25 词上限 | 避免"考试题"式长 query，强制融合而非拼接 |
| 禁止所有 meta-words | 彻底解决 "the text states" 类污染 |
| 3 条 query 引用不同段落 | 解决 text evidence 复用问题 |
| 自动化 QC pipeline | 可复现、可扩展，不依赖人工逐条检查 |

---

## 7. 代码解析与当前使用模块（重点）

**当前在用的生成链路（v3）**：
1. **图文对齐（上游）**  
   - 入口：`data/figure_text_pairs.json`  
   - 产出方式：由 `src/linkers/figure_text_associator.py` 将 MinerU 解析结果中的图像与上下文文本进行关联。  

2. **L1 Query 生成（当前主流程）**  
   - 脚本：`scripts/batch_figure_understanding_api.py`（**当前使用**）  
   - 关键流程：  
     - 读取 `figure_text_pairs.json`  
     - 对图片做 base64 编码，拼接 caption + 前后文 + references  
     - 调用 Anthropic API（Claude Sonnet 4.5）返回 JSON  
     - 解析 JSON，保留 `queries / visual_elements / figure_type`  
     - 在脚本内做第一次 QC（禁 meta-language、要求 visual_anchor & text_evidence）  
     - 输出：  
       - `data/figure_descriptions_v3_api.json`（完整原始响应）  
       - `data/l1_cross_modal_queries_v3.jsonl`（过滤后的最终 L1）  

3. **质量审计（统计与诊断）**  
   - 脚本：`scripts/validate_queries.py`（**当前使用**）  
   - 作用： schema 检查、视觉锚点检测、禁用模式识别、分布统计  
   - 输出：`data/validation_report_v3.json`（clean rate、类型分布等）  

**当前不再使用 / 历史脚本（v1/v2）**：
- `scripts/batch_figure_understanding.py`：本地 vLLM 推理脚本（Qwen3-VL），因 OOM + 解析率低而弃用。  

**代码层关键设计点**：
- 生成端已经把 **QC 的硬约束嵌入主脚本**（不是完全依赖后处理）。  
- 输出中统一 **相对路径**，便于复现与迁移。  
- 生成脚本 + QC 脚本分离：便于替换模型、对比实验。  

---

## 8. 当前策略与我的思路（给 mentor）

**总体思路**：先把 L1 的“跨模态融合”做稳，再把模态多样性与证据链结构化能力做强，为 L2/L3 的跨文档/多跳做铺垫。

**当前策略拆解**：
1. **融合优先于规模**  
   - 先确保 query 本身能“盲测不过”（去掉图/文任一都无法回答），再扩大规模。  
   - 这也是我选择 **强约束 prompt + QC 过滤** 的原因：宁可少一点，也要是可训练的。

2. **显式视觉锚点是第一约束**  
   - 统计表明主要噪声来自“缺锚点”与“why 问句悬空”。  
   - 下一步不是盲目加量，而是把 **视觉锚点写入 query 本体**（而不仅是 `visual_anchor` 字段）。

3. **多样性是第二阶段目标**  
   - plot 偏重是数据分布的客观结果，但 L2/L3 需要更强的 table/formula 参与度。  
   - 我会在 **采样层/模板层** 加入 table/formula 的配额约束，并优化 MinerU 的表格/公式结构化落盘。

4. **可复现是工程底线**  
   - 输出中统一相对路径、记录模型与 prompt 版本，保证“可复查、可比较、可迭代”。  
   - 这是把生成数据当作“实验资产”而非“一次性产出”的原则。

**我希望 mentor 看到的重点**：  
我不是在“堆 query”，而是在构建一个 **可持续可扩展的跨模态数据生成框架**。  
L1 解决“融合”，L2/L3 解决“证据链”，table/formula 解决“模态均衡”，QC/日志解决“可复现”。

---

## 9. L1 Triage 与 L2 跨文档推进（2026-02-10）

### 9.1 L1 三分法分拣结果

对 974 条 L1 v3 queries 进行自动化门禁分拣：

| Grade | Count | Pct | 含义 |
|-------|-------|-----|------|
| **A (keep)** | 751 | 77.1% | 直接可用于训练 + L2 输入 |
| **B (clean)** | 223 | 22.9% | 有问题但可修复，或作为 hard negative |
| **C (drop)** | 0 | 0.0% | v3 QC 已过滤最差样本 |

门禁命中分布：

| 问题 | Count | Pct |
|------|-------|-----|
| value_leakage（query 含答案小数） | 126 | 12.9% |
| ocr_only_anchor（视觉锚点仅含文字，无几何/颜色） | 101 | 10.4% |
| ungrounded_why（why 无视觉锚点） | 26 | 2.7% |
| evidence_truncated（文本证据截断） | 2 | 0.2% |

### 9.2 L2 跨文档候选构建

从 751 条 A-class L1 queries 中提取实体，构建跨文档桥接：

- **436 个 unique entities**（方法名/数据集/指标/概念）
- **55 个 cross-document entities**（出现在 2+ 篇文档）
- **711 个候选文档对**
- Top bridge entities: fairness (33 docs), accuracy (22 docs), COMPAS (6 docs), disparate impact (5 docs)

### 9.3 下一步

1. 对 top-50 候选对调用 Claude API 生成 L2 queries
2. 复用 L1 QC pipeline + 跨文档证据完整性门禁
3. 人工写 30 条测试 query，建立评估闭环

---

## 10. 文件清单

| 文件 | 说明 |
|------|------|
| `scripts/batch_figure_understanding.py` | vLLM 本地推理脚本 (v1/v2) |
| `scripts/batch_figure_understanding_api.py` | Anthropic API 推理脚本 (v3) |
| `scripts/validate_queries.py` | Query QC & validation |
| `scripts/triage_l1_v3.py` | **L1 三分法分拣 (A/B/C)** |
| `scripts/build_l2_candidates.py` | **L2 跨文档候选对构建** |
| `scripts/generate_l2_queries.py` | **L2 query 生成 (Claude API)** |
| `data/l1_cross_modal_queries_v3.jsonl` | **最终输出：974 条 L1 queries** |
| `data/l1_triage_v3.jsonl` | **L1 分拣结果（含 triage 字段）** |
| `data/l1_triage_report_v3.json` | **L1 分拣统计报告** |
| `data/l2_candidate_pairs_v1.json` | **L2 候选文档对（top-100）** |
| `data/figure_descriptions_v3_api.json` | 完整 API 返回（含 raw response） |
| `data/validation_report_v3.json` | Validation 报告 |
