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
| 缺少 visual anchor | 🔴 严重 | 63.4% 的 query 没有任何视觉锚点 |
| "看图说话"而非跨模态推理 | 🔴 严重 | 很多 query 不看文本也能回答 |
| Meta-language 泛滥 | 🟡 中等 | "According to the text", "the figure shows" |
| Why/How 占比过高 | 🟡 中等 | 37.3% 是解释型，难以 ground |
| 类型字段污染 | 🟡 中等 | `requires_figure` 有 2 个非 bool 值 |
| 绝对路径不可复现 | 🟢 轻微 | image_path 包含服务器绝对路径 |

**Reviewer 评价**：*"这些是看图说话，不是真正的 cross-modal reasoning"*

### v2：Prompt 重设计（仍用 Qwen3-VL）

**改进措施**：
- ✅ 添加 Blindfold Test 要求（遮住图/文任一都不能答）
- ✅ 要求明确 `visual_anchor` 和 `text_evidence` 字段
- ✅ 添加 banned patterns 列表
- ✅ 新增 `validate_queries.py` QC 脚本
- ✅ 定义 4 种 query type（value_context / comparison_explanation / anomaly_cause / visual_definition）

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
| 74.8% visual anchor（非100%） | 🟡 可改进 | validation 已标注，可人工审核剩余 25% |
| Figure type 偏 plot (71.3%) | 🟡 数据限制 | 受限于 arXiv 论文本身图片类型分布 |
| Table 模态几乎没有 (0.6%) | 🟡 数据限制 | Table 在 MinerU 中多解析为文本而非图片 |
| 数值答案可靠性 | ⚪ 待验证 | MLLM 生成的数值可能有幻觉，可抽样验证 |

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

## 7. 文件清单

| 文件 | 说明 |
|------|------|
| `scripts/batch_figure_understanding.py` | vLLM 本地推理脚本 (v1/v2) |
| `scripts/batch_figure_understanding_api.py` | Anthropic API 推理脚本 (v3) |
| `scripts/validate_queries.py` | Query QC & validation |
| `data/l1_cross_modal_queries_v3.jsonl` | **最终输出：974 条 L1 queries** |
| `data/figure_descriptions_v3_api.json` | 完整 API 返回（含 raw response） |
| `data/validation_report_v3.json` | Validation 报告 |
