# 代码逐函数解读：两版 Enrich 脚本

> 本文档覆盖两个 enrichment 脚本：
> - `scripts/enrich_elements_modora.py`（元素语义增强，MoDora 风格）
> - `scripts/enrich_hub_candidates.py`（Hub 候选对格式转换 + 语义摘要）
>
> 两者在 pipeline 中的位置：`enrich_elements` → `enrich_hub_candidates` → `generate_multihop_l1_queries`

---

## Part 1：`enrich_elements_modora.py`

**目标**：用 LLM 对每个 figure/table/formula 生成结构化语义描述，输出三个新字段：
- `enriched_title`：一句话总结（10-20 词）
- `enriched_metadata`：结构化 JSON（类型/关键词/坐标/变量等）
- `enriched_content`：2-4 句语义描述

这些字段让原本只有 caption 的多模态元素获得更丰富的文本表示，BM25 和 LLM prompt 都能利用。

---

### `resolve_image_path(raw_path) -> Optional[Path]`
**作用**：将存储在 JSON 里的图片路径解析为当前环境下实际存在的路径。

**问题背景**：`multimodal_elements.json` 中的 `image_path` 是在集群上生成的绝对路径（`/projects/_hdd/myyyx1/...`），在本地 Windows 机器或不同服务器上会失效。

**四级解析策略**：
1. 直接检查路径是否存在
2. 去掉集群前缀 `/projects/_hdd/myyyx1/`，重拼 PROJECT_ROOT
3. 截取 `/data/mineru_output/` 后的部分，接到 PROJECT_ROOT
4. 找 `/data/` 后的所有内容，接到 PROJECT_ROOT

**返回**：找到则返回 `Path`，否则 None（跳过该 element 的图片，只用文本）。

---

### `load_image_b64(image_path) -> Optional[Tuple[str, str]]`
**作用**：读取图片文件，base64 编码用于 API。

**限制**：超过 5MB 的图片跳过（避免 API token 超限）。
**返回**：`(base64_string, mime_type)` 或 None。

---

### 三套 Prompt 模板（PROMPT_FIGURE / PROMPT_TABLE / PROMPT_FORMULA）

每个模板要求 LLM 输出固定结构的 JSON，温度 0.2（低随机性，保证结构一致）：

#### `PROMPT_FIGURE`
```json
{
  "title": "一句话描述图的核心发现",
  "metadata": {
    "figure_type": "line_plot|bar_chart|heatmap|scatter_plot|architecture_diagram|...",
    "keywords": ["accuracy", "epoch", "fairness"],
    "axes": {"x": "epoch", "y": "accuracy"},
    "n_series": 3
  },
  "content": "2-4 句语义描述，重点讲图展示了什么现象/趋势/对比"
}
```

#### `PROMPT_TABLE`
```json
{
  "title": "一句话描述表格的核心内容",
  "metadata": {
    "table_type": "results_comparison|ablation_study|dataset_statistics|...",
    "keywords": ["F1", "precision", "baseline"],
    "columns": ["Method", "Accuracy", "F1"],
    "n_rows": 5,
    "best_values": {"Accuracy": "0.923 (row 3)"}
  },
  "content": "2-4 句语义描述，说明表格比较了什么、最佳结果是什么"
}
```

#### `PROMPT_FORMULA`
```json
{
  "title": "一句话描述公式的作用",
  "metadata": {
    "formula_type": "loss_function|objective|constraint|definition|...",
    "keywords": ["regularization", "KL divergence"],
    "variables": {"λ": "regularization weight", "θ": "model parameters"},
    "domain": "optimization|probability|information_theory|..."
  },
  "content": "2-4 句语义描述，解释公式的数学含义和在论文中的角色"
}
```

---

### `build_element_prompt(element) -> str`
**作用**：为单个元素构造 enrichment prompt。

**输入字段截取限制**：
- `caption`：最多 400 chars（避免超长 caption 干扰）
- `content`（LaTeX 原文）：最多 1200 chars
- `context_before`/`context_after`：各最多 300 chars

按 element_type 选择对应模板，填入上述字段。

---

### `extract_json(text) -> Optional[Dict]`
**作用**：从 LLM 响应中提取 JSON。

**比 `parse_json()` 更鲁棒**：手动跟踪 `{` `}` 嵌套深度和字符串转义，能处理嵌套 JSON（如 metadata 内的 dict）。

---

### `validate_enrichment(result, etype) -> List[str]`
**作用**：验证 enrichment 结果是否完整。

**检查项**：
- `title` 不为空且长度 ≥ 3 chars
- `content` 不为空且长度 ≥ 20 chars
- `metadata` 存在
- 类型特定字段存在（`figure_type` / `table_type` / `formula_type`）
- `keywords` 列表至少 1 个

**返回**：问题列表（空 = 验证通过）。

---

### `process_elements(mm_data, client, model, provider, no_images, dry_run, existing_enriched) -> Tuple[enrichments, in_tok, out_tok, processed, failed]`

**作用**：遍历所有 elements，调 LLM 生成 enrichment。

**关键参数**：
- `no_images`：只用文本，不发送图片（节省 token；formula 天然走这条路）
- `dry_run`：只打印 prompt，不调 API（用于调试）
- `existing_enriched`：已有 enriched 结果的 element_id 集合（增量模式跳过）

**处理流程（每个 element）**：
1. `build_element_prompt(element)` → 构造 prompt
2. `load_image_b64()` → 加载图片（如果不是 formula 且不是 no_images 模式）
3. `call_api()` → 调 LLM
4. `extract_json()` → 解析
5. `validate_enrichment()` → 验证
6. 记录结果：通过则存入 enrichments dict；失败则记录 issues

**返回**：enrichments dict + token 统计 + 处理数量。

---

### `merge_enrichments(mm_data, enrichments) -> Dict`
**作用**：将 enrichment 结果合并回 `multimodal_elements` 原始结构。

对每个已 enrich 的 element，追加四个字段：
- `enriched_title`：一句话总结
- `enriched_metadata`：结构化描述 dict
- `enriched_content`：语义描述段落
- `enrichment_issues`：如果有验证问题，记录在这里

**不覆盖原始字段**（`caption`、`content`、`context_before/after` 保持不变）。

---

### `main()`

**CLI 参数速查**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--input` | multimodal_elements.json | 输入 |
| `--output` | multimodal_elements_enriched.json | 输出 |
| `--provider` | company | anthropic / openai / company |
| `--model` | 自动选 | 各 provider 默认模型 |
| `--limit` | 0(all) | 最多处理多少 elements |
| `--no-images` | False | 纯文本模式（公式默认走这条） |
| `--dry-run` | False | 调试模式，不调 API |
| `--incremental` | False | 跳过已有 enrichment 的元素 |
| `--delay` | 0.3 | API 调用间隔 |

**规模**：1316 个 elements（841 figure + 334 table + 141 formula），全量 enrichment 约 $3。

**铁律**：结尾调 `log_run()` 记录 token。

---

---

## Part 2：`enrich_hub_candidates.py`

**目标**：把 `analyze_latex_graph_topology.py` 输出的 500 条 hub 候选（LaTeX 内部 node_id 格式）转换为 `generate_multihop_l1_queries.py` 能直接读取的格式（MinerU element_id + 完整 element 详情），同时注入 hub semantic summary。

**定位**：格式转换 + 语义增强的"胶水层"，不调 LLM（纯规则）。

---

### `build_mm_index(mm_data) -> Dict`
与 `analyze_latex_graph_topology.py` 中的同名函数功能相同：
```
{
  "by_doc": {doc_id: {"by_number": {type: {n: eid}}, "by_caption": {type: [(eid, tokens)]}}},
  "all_elements": {eid: element_dict}
}
```

---

### `build_node_element_map(latex_data, mm_index, mm_data) -> Dict[str, str]`
**作用**：将拓扑图中所有 LaTeX 节点 ID 映射到 MinerU element_id。

**三阶段映射**：

| 阶段 | 策略 | 说明 |
|------|------|------|
| Phase 1 | 数字 + Caption Jaccard ≥ 0.25 | 优先精确匹配 |
| Phase 2 | 同 Phase 1，但 Jaccard 阈值降到 0.20 | 扩大召回 |
| Phase 3 | 顺序匹配（fallback） | 按 `line_no` 排序后第 N 个 LaTeX label → 第 N 个 MinerU element |

**关键**：Phase 3 是兜底，解决 LaTeX 和 MinerU 编号完全不对应的情况（如 MinerU 跳编号、LaTeX 用字母 label）。

---

### `build_edge_context_index(latex_data) -> Dict[Tuple[str, str], List[Dict]]`
**作用**：建立边上下文的快速查找索引。

从 reference graph 的所有 edge 中提取 `context` 字段，按 `(source_node_id, target_node_id)` 对为 key 存储。

后续 `enrich_candidates()` 会用这个索引找到路径上每条边的原文 bridge 上下文。

---

### `build_hub_semantic_summary(el_a, el_b, edge_contexts) -> str`

**作用**：为候选对生成压缩的语义摘要（50-80 词），**纯规则，不调 LLM**。

**MoDora 风格的级联摘要策略**：
1. 从 `el_a["enriched_title"]` 取前 ~20 词（如果有 enriched）
2. 从 `el_b["enriched_title"]` 取前 ~20 词
3. 从最强 bridge context 取前 ~15 词
4. 合并两个元素 `enriched_metadata.keywords` 的交集（最多 5 个）
5. 格式化为 `[FIGURE A] ... | [TABLE B] ... | [BRIDGE] ... | [KEYWORDS] ...`
6. 硬截断到 80 词

**目的**：给 `generate_multihop_l1_queries.py` 的 prompt 提供一个紧凑的桥接语义摘要，减少 LLM 需要"自己理解"的工作量。

---

### `_build_hub_quality_scores(hub_data) -> Dict[str, float]`
**作用**：从 hub 拓扑特征计算质量分，用于候选对的 `quality_score` 字段。

**公式**：
```
score = 0.5 × norm(bridge_score)
       + 0.25 × norm(pagerank)
       + 0.25 × norm(out_to_elements)
```

全部归一化到 [0.1, 1.0]（保证每对至少有最低质量分）。

`quality_score` 在下游的 `load_element_hub_prior()` 和 `load_element_adjacency()` 中被用作边权重。

---

### `enrich_candidates(hub_data, mm_index, node_to_element, edge_ctx_index, enriched_elements=None) -> Dict`

**作用**：核心转换函数，将 topology candidates 逐一转为 generation-ready 格式。

**对每个 candidate pair 的处理**：
1. **提取端点**：从 `path` 取第一个和最后一个节点，通过 `node_to_element` 映射到 MinerU element_id
2. **解析路径边上下文**：沿 path 逐条边查 `edge_ctx_index`，收集所有 bridge context snippets
3. **构建 node_group**：path 中所有 distinct element endpoint（1-3 个）
4. **生成 hub_semantic_summary**：调 `build_hub_semantic_summary()`
5. **计算 quality_score**：调 `_build_hub_quality_scores()`
6. **组装输出**：保留所有原始字段，追加 `element_a_id`, `element_b_id`, `element_a`, `element_b`（完整 element dict）, `bridge_contexts`, `hub_semantic_summary`, `quality_score`

**Adjacent Bridge 处理**：将 `adjacent_backbone_bridges` 中的 LaTeX node_id 也映射为 MinerU element_id，生成 `adjacent_bridge_elements` 和 `adjacent_bridge_adjacency` 两个字典，供检索实验的 `load_element_hub_prior` 和 `load_element_adjacency` 使用。

**返回**：
```json
{
  "pairs": [...],           // 转换后的候选对
  "summary": {...},         // 统计
  "adjacent_bridge_elements": {eid: score},  // hub 覆盖扩充
  "adjacent_bridge_adjacency": [...]         // 邻接边扩充
}
```

---

### `main()`

**CLI 参数速查**：

| 参数 | 说明 |
|------|------|
| `--hub-candidates` | 拓扑分析输出（latex_hub_multihop_candidates.json） |
| `--elements` | multimodal_elements.json |
| `--latex-graph` | latex_reference_graph.json |
| `--enriched-elements` | multimodal_elements_enriched.json（可选） |
| `--hubs` | latex_graph_hubs.json（用于质量分计算） |
| `--output` | 输出 enriched candidates JSON |

**典型输出**：~490/500 pairs 成功映射（约 2% 因 label 匹配失败而丢失）。

---

## 两版 Enrich 的关系与区别

| 维度 | `enrich_elements_modora.py` | `enrich_hub_candidates.py` |
|------|----------------------------|---------------------------|
| 输入粒度 | 单个 element（figure/table/formula） | 候选 pair（两个 element + path） |
| 调 LLM | ✅ 是（每 element 一次 API call） | ❌ 否（纯规则） |
| 主要输出 | enriched_title / metadata / content | element_id 映射 + hub_semantic_summary |
| 依赖关系 | 独立运行，输出被 enrich_hub_candidates 消费 | 依赖 enrich_elements 的输出（可选） |
| 成本 | ~$3（1316 elements） | 免费 |
| 目的 | 让 figure/table/formula 有语义文本 | 把拓扑候选转换为生成可用格式 |

**Pipeline 顺序**：
```
enrich_elements_modora.py          → multimodal_elements_enriched.json
        ↓
enrich_hub_candidates.py           → hub_candidates_enriched.json
  (可选传入 --enriched-elements)
        ↓
generate_multihop_l1_queries.py    → L1/L2/L3 queries JSONL
```
