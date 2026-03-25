# 代码逐函数解读：`scripts/generate_multihop_l1_queries.py`

> Query 生成的核心脚本，处理 L1/L2/L3 三个难度层级的多模态跨证据 query 生成。内含 prompt 模板、质量控制、PersonaHub、bridge grounding、公司 API 调用等全部逻辑，3850 行。

---

## 整体架构

```
main()
 ├── 加载候选对 + enriched elements + bridge texts（P0）
 ├── 对每个 candidate pair：
 │    ├── build_prompt()       ← 拼接 prompt + 注入 bridge + 注入 persona
 │    ├── call_api()           ← 调 LLM（Anthropic / OpenAI / Company）
 │    ├── parse_json()         ← 解析 LLM 输出
 │    ├── qc_multihop_query()  ← 质量控制（11+ 检查项）
 │    └── 写入 JSONL
 └── log_run()                 ← 铁律：token 记录
```

---

## 一、Bridge Text 加载系统（P0 核心）

### `load_reference_graph_bridge_texts(ref_graph_path, topology_candidates_path="")`
**作用**：预加载 `latex_reference_graph.json` 中所有边的上下文文本，供 prompt 注入。

**核心数据结构**：
- `_BRIDGE_TEXT_CACHE`：`{doc_id: {label: cleaned_bridge_text}}`
- `_ELEMENT_TO_LABELS`：`{element_id: [latex_label1, ...]}`（MinerU ID → LaTeX label 映射，1317 个）

**流程**：
1. 遍历 reference graph 每篇文档的所有边
2. 提取边的 `context` 字段并清洗（去 LaTeX 命令）
3. 按 `(doc_id, target_label)` 索引存入 cache
4. 如果提供 topology_candidates，额外构建 element_id→label 反向映射

**设计意图**：bridge text 是作者写的**真实连接句**（"We show that Figure 3 and Table 2 together demonstrate..."），注入 prompt 后 LLM 的 query 会使用论文原有术语，显著提升 BM25 词面匹配（MRR +0.135）。

---

### `_clean_latex_bridge(text) -> str`
**作用**：清洗 LaTeX 命令，保留语义内容。

去掉 `\textbf{}`, `\emph{}`, `\cite{}`, `$...$` 外壳，保留内容文字。公式直接去掉 `$$...$$`（避免生成 query 里出现裸 LaTeX）。

---

### `resolve_bridge_texts_for_path(pair) -> List[str]`
**作用**：给定一个 candidate pair，找出其 multi-hop 路径上的 bridge 文本列表。

**策略**：
1. 从 `element_a_id` / `element_b_id` 提取 doc_id
2. 用 `_ELEMENT_TO_LABELS` 找到每个 element 对应的 LaTeX label
3. 在 `_BRIDGE_TEXT_CACHE` 中查找这些 label 的 bridge text
4. Fallback：用 pair 自带的 `bridge_contexts` 字段

**返回**：最多 3 条去重 bridge text 列表。

---

### `score_bridge_quality(bridge_text) -> float`
**作用**：对 bridge 段落评质量分（0.0–1.0），用于筛选候选对和 QC 时的 bridge span 检验。

**评分信号**：

| 信号 | 权重 | 说明 |
|------|------|------|
| 长度 | max 0.30 | 200–600 chars 最优 |
| 语义动词 | max 0.35 | show/compare/indicate/demonstrate 等 |
| 交叉引用标记 | +0.15 | 含 `[fig\|tab\|eq\|cite]` 等标记 |
| 公式比例惩罚 | -0.2 | alpha 字符 < 40% 则惩罚 |
| 样板句惩罚 | -0.1 | "see also"/"e.g."/"note that" 等 |

**分类**：HIGH ≥ 0.7、MEDIUM 0.4–0.7、LOW < 0.4。

---

## 二、Prompt 模板系统

脚本内有 **6 套 prompt 模板**，每套 200–300 行：

| 模板 | 适用场景 | 核心约束 |
|------|---------|---------|
| `PROMPT_FIGURE_TABLE_1HOP` | figure + table，1-hop | SHORT(8-14词)+LONG(18-30词)双 query；OBSERVATION INJECTION |
| `PROMPT_FIGURE_TABLE_2HOP` | figure + table，2-hop | 中间 bridge element；3-step 依赖链 |
| `PROMPT_FIGURE_FORMULA` | figure + formula | FIGURE TYPE STRATEGY（量化图 vs 架构图分别处理） |
| `PROMPT_FORMULA_TABLE` | formula + table | 公式变量必须语义化；不得照抄符号 |
| `PROMPT_3STEP_REASONING_CHAIN` | L3，3-step 串行 | `reasoning_steps[]` + `depends_on_steps`；bridge grounding rule；serial chain 强制示例 |
| `PROMPT_REAL_USER_*`（3个）| real_user style | factual_lookup / summary / comparison 三类，放宽学术腔要求 |

**所有模板通用规则**：
- **FORBIDDEN**：yes/no 题、meta-language（"the figure shows"）、直接抄 anchor 坐标、裸 LaTeX 符号
- **MANDATORY**：OBSERVATION INJECTION（用自然语言描述视觉现象，如"the curve drops sharply after epoch 50"）
- **ENTITY AMNESTY**：允许使用论文原有术语（F1 score、regularization strength）

---

## 三、Quality Control 系统

### `qc_multihop_query(obj, pair) -> Tuple[bool, List[str]]`

L1/L2 dual-evidence query 的质检门，11+ 个检查项。返回 `(是否通过, 失败原因列表)`。

**Hard-fail 检查（任一失败则 QC 不通过）**：

| 检查项 | 函数 | 判定逻辑 |
|--------|------|---------|
| yes/no 问题 | `is_yes_no_question()` | query 以 Do/Does/Is/Are/Can/Should 开头 |
| anchor 泄漏 | `anchor_leak_jaccard()` | query token 与 visual_anchor 的 Jaccard > 0.20 |
| 单元素答案 | `single_element_answer()` | answer 中只引用了一个 element（双证据要求两个都用到） |
| bridge 实体泄漏 | `bridge_entity_leakage()` | query 中直接出现了 bridge 实体名（"Table 2 shows..."） |
| 模板捷径 | `has_template_collapse()` | "Which component", "How does X relate to Y" 等弱模板 |
| query 过长 | `query_too_long()` | 超过 35 词 |
| 弱连接词 | `has_weak_reasoning_connector()` | answer 缺少 because/leads to/constrains 等因果词 |
| visual anchor 泛化 | `check_visual_anchor_specificity()` | visual_anchor 全是泛型词（无 row/col/axis/marker 等具体位置） |
| bridge span 为空 | `check_bridge_span()` | L3 query 必须有非空 bridge claim |

**Soft warning（记录但不 hard-fail）**：
- `parallel_reasoning`：reasoning 结构被判定为 parallel（不是 serial chain）
- `semantic_category_mismatch`：query 语义类型与 pair 类型不符
- `premise_answer_contradiction`：前提和答案逻辑不一致

---

### `qc_real_user_query(obj, pair) -> Tuple[bool, List[str]]`

real_user style 的 QC，规则更宽松：
- 不强制 length_mix（real user 说话没那么规整）
- 不禁止 "Given that" 等表达
- 新增 `retrievability_score` 检查（query 能否被 dense 检索命中）
- 新增 query_type 必须是 factual_lookup / summary / comparison 之一

---

### `qc_reasoning_depth(obj) -> Dict`

对 L3 query 的推理深度打标，**advisory only**（不影响 pass/fail）：
- `classify_reasoning_structure()`：通过连接词模式（because/therefore/hence 为 serial；and/also/additionally 为 parallel）区分推理结构
- `m4_is_true_multihop`：proxy heuristic（answer 里因果连接词数 ≥ min_depth - 1）
- **已知局限**：写作风格可欺骗；30-50 条人工标注误差审计待做

---

## 四、PersonaHub 系统

### `_load_personahub_personas(path=None) -> List[Dict]`
加载 `data/personahub_academic_personas.json`，包含 50 类学术读者人设（PhD student, industry ML engineer, medical researcher...），每条有 name/description/query_style 三个字段。

### `resolve_persona(pair_id) -> str`
**确定性分配**：`hash(pair_id) % num_personas`，同一 pair 每次跑都分到同一人设。

### `inject_persona_prefix(prompt, persona) -> str`
在 prompt 最前面注入人设描述，引导 LLM 用该读者的视角来问问题。

---

## 五、API 调用层

### `encode_image(path) -> Optional[Tuple[str, str]]`
**作用**：将图片文件加载并 base64 编码，用于多模态 API 调用。

**路径解析顺序**：
1. 直接路径
2. 去掉集群前缀重拼接（`/projects/_hdd/myyyx1/`）
3. 从 `/data/mineru_output/` 重新 root

超过 5MB 的图片跳过（避免 API 超限）。

---

### `call_api(client, model, prompt, system_prompt, images, provider) -> Tuple[str, int, int]`
**作用**：统一 API 调用入口，支持三种 provider。

| Provider | 调用方式 | 图片格式 |
|---------|---------|---------|
| `anthropic` | `client.messages.create()` | native base64 source |
| `openai` | `client.chat.completions.create()` | `image_url` data URI |
| `company` | `local_api_logger.wrap_requests_call()` + SSE 解析 | OpenAI 兼容格式 |

**返回**：`(response_text, input_tokens, output_tokens)`

---

### `_collect_company_stream(stream_generator) -> Tuple[str, int, int]`
**作用**：解析 yunwu.ai 返回的 SSE（Server-Sent Events）流。

逐行读取 `data: {...}` 格式的 JSON chunk，累积 `delta.content`，从 `usage` 字段提取 token 计数。

---

## 六、Build Prompt & Parse

### `build_prompt(pair, query_style="academic", use_persona=False) -> str`

**流程**：
1. 根据 `pair_type`（fig+tbl/fig+formula/formula+tbl）+ hop_distance 选择模板
2. 读取 element_a/b 的 caption、context_before/after、enriched 字段
3. 调 `resolve_bridge_texts_for_path()` 注入 bridge 原文
4. 如果有 enriched elements（MoDora），调 `build_enriched_context_section()` 追加 enriched 字段
5. 可选注入 persona
6. 格式化所有占位符，返回完整 prompt 字符串

**Enrichment 过滤（C1）**：若 enriched 字段含 glyph/icon/marker 等噪声关键词，自动回退到原始 context，不盲目信任 enrichment。

---

### `parse_json(txt) -> Optional[Dict]`
**作用**：从 LLM 响应中健壮地提取 JSON。

1. 去掉 ` ```json ` 代码块标记
2. 找第一个 `{`，手动匹配括号找对应的 `}`
3. 尝试 `json.loads()`

**容错**：LLM 有时会在 JSON 前后加解释文字，这个函数能跳过它们。

---

## 七、Main Pipeline

### `main()`

**CLI 参数速查**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--candidates` | — | 候选对文件（enriched candidates） |
| `--output` | — | 输出 JSONL |
| `--reference-graph` | auto | LaTeX reference graph（bridge text 来源） |
| `--enriched-elements` | None | MoDora enriched elements |
| `--query-style` | academic | academic / real_user / mixed |
| `--use-persona` | False | 启用 PersonaHub |
| `--provider` | company | anthropic / openai / company |
| `--model` | gpt-5.4 | 模型名 |
| `--pass-only` | False | 只输出 QC pass 的 query |
| `--limit` | 0(all) | 最多处理多少 candidates |
| `--dry-run` | False | 不调 API，只看 prompt |
| `--delay` | 0.5 | 调用间隔（秒），避免限速 |

**输出格式**（每条 JSONL）：
```json
{
  "query_id": "l3_1802.08139_hub_pair_11",
  "query": "How does the green-path causal setup...",
  "answer": "The causal graph marks A's direct path...",
  "reasoning_steps": [{...}, {...}, {...}],
  "required_evidence_spans": [{"element_id": "...", "span": "..."}],
  "visual_anchors": ["row 3 col 2", "x-axis epoch"],
  "qc_pass": true,
  "qc_issues": [],
  "bridge_quality": 0.85,
  "reasoning_structure": "serial"
}
```

**铁律**：`main()` 最末尾必须调 `log_run()`，记录本次跑了多少 token、处理了多少 pair、pass 多少条。
