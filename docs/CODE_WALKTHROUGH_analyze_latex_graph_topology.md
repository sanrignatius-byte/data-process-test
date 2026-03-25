# 代码逐函数解读：`scripts/analyze_latex_graph_topology.py`

> 图结构生成的核心脚本。从 LaTeX 源码解析出的引用图出发，构建带 backbone 边的异构图，识别 bridge hub 节点，枚举多跳候选路径，输出 500 条 candidate pairs。2117 行。

---

## 整体架构

```
main()
 ├── 加载 latex_reference_graph.json + multimodal_elements.json
 ├── build_mm_index()              ← 建立 MinerU 元素查找索引
 ├── build_topology_graph()        ← 构建带 backbone 的异构图
 │    ├── section 节点
 │    ├── element 节点（figure/table/formula）
 │    ├── paragraph 节点
 │    ├── paragraph_ref 边（\ref{} 调用）
 │    ├── element_ref 边（共引）
 │    ├── backbone 边（段落阅读顺序）
 │    └── section 包含边
 ├── build_cross_doc_edges()        ← 跨文档引用边
 ├── compute_bridge_hubs()          ← 识别 bridge hubs（60 个）
 ├── compute_adjacent_backbone_bridges()  ← 邻接 backbone bridge spans
 └── enumerate_candidates_from_bridge_hubs()  ← 枚举 500 条路径
```

---

## 一、核心数据结构

### `@dataclass Node`
图中每个节点的完整描述：

| 字段 | 说明 |
|------|------|
| `node_id` | 唯一标识，格式 `{doc_id}::{type}::{label}` |
| `doc_id` | 所属论文 arXiv ID |
| `node_type` | section / subsection / paragraph / figure / table / equation |
| `label` | LaTeX label（如 `fig:accuracy`） |
| `mapped_element_id` | 对应的 MinerU element_id（如 `1306.5204_figure_2`） |
| `page_idx` | 页码（从 content_list.json 获取的真实页码） |
| `line_no` | 在 LaTeX 源码中的行号（用于 backbone 排序） |
| `section_title` | 所在章节标题 |

### `@dataclass Edge`

| 字段 | 说明 |
|------|------|
| `edge_type` | `paragraph_ref` / `element_ref` / `backbone` / `cross_doc_cite` / `section_contains_*` |
| `weight` | `paragraph_ref` 边：`log2(1 + ref_count)`；backbone 边：1.0 |

---

## 二、MinerU 元素索引

### `build_mm_index(mm_data) -> Dict`
**作用**：从 `multimodal_elements.json` 建立三层查找索引。

```
返回结构：
{
  "by_doc": {
    "doc_id": {
      "by_number": {"figure": {1: eid, 2: eid}, "table": {1: eid}},
      "by_caption": {"figure": [(eid, tokens)], ...}
    }
  },
  "all_elements": {eid: element_dict}
}
```

`by_caption` 用于 Jaccard 匹配；`by_number` 用于编号直接匹配。

---

## 三、LaTeX Label → MinerU Element 映射

### `map_label_to_element(doc_id, label_key, label_info, mm_index, label_ordinal=None) -> Optional[str]`

这是整个图构建流程中**最关键的映射函数**，负责把 LaTeX label（如 `fig:results`）转换为 MinerU element_id（如 `1801.04385_figure_3`）。

**六级策略（按优先级）**：

| 优先级 | 策略 | 说明 |
|--------|------|------|
| 1 | 数字提取 | 从 label key 提取数字后缀，匹配 `by_number` |
| 2 | 后缀拆分 | 按 `-_:` 分割 label，逐个数字部分尝试 |
| 3 | ±2 偏移 | 尝试 number ± 1/2（MinerU 与 LaTeX 编号 offset 问题） |
| 4 | 顺序匹配 | 第 N 个 LaTeX label → 第 N 个 MinerU element（按 line_no vs position） |
| 5 | Caption Jaccard | label 的 caption 与 element caption token 重合度 ≥ 0.20 |
| 6 | 上下文匹配 | label 的 context text 与 element caption 的 Jaccard |

**实际效果**：49.8% 匹配率（1317/2644 labels）。主要失败原因：MinerU 解析遗漏元素、极端 offset、纯数字 label。

---

## 四、图构建核心

### `build_topology_graph(latex_data, mm_index) -> Tuple[nodes, edges, out_adj, in_adj, stats]`

**作用**：构建完整的文档内异构图，这是整个系统的核心函数。

**六类节点/边的构建逻辑**：

#### 1. Section 节点
从 `latex_data["documents"][doc_id]["sections"]` 提取，每个 section/subsection/subsubsection 建一个节点，记录 `section_level`（1/2/3）和标题。

#### 2. Element 节点
遍历所有 LaTeX labels，调 `map_label_to_element()` 映射，为 figure/table/formula 各建节点。`mapped_element_id` 用于后续与 MinerU 数据对齐。

#### 3. Paragraph 节点
从 `latex_data["documents"][doc_id]["paragraphs"]` 提取。每段记录 `line_no`（源码行号）和 `section_title`（所在章节）。**Paragraph 是 bridge 信息的载体**——段落引用元素，段落之间构成 backbone。

#### 4. `paragraph_ref` 边
含义：段落 P 通过 `\ref{label}` 引用了元素 E。

**权重**：`log2(1 + ref_count)`，多次引用同一元素权重更高。

这类边是 bridge 发现的基础：高权重 paragraph_ref 说明段落强关联该元素。

#### 5. `element_ref` 边（共引）
含义：同一段落同时引用了元素 A 和元素 B（`\ref{figA}...\ref{tabB}`）。

这类边直接把两个元素关联起来，是 L2 dual-evidence 候选的主要来源。

#### 6. `backbone` 边 ← **设计核心**
含义：**同一文档中按 `line_no` 排序的相邻段落之间**，用 backbone 边连接。

```python
sorted_paragraphs = sorted(paragraphs, key=lambda p: p.line_no)
for i in range(len(sorted_paragraphs) - 1):
    add_edge(sorted_paragraphs[i], sorted_paragraphs[i+1], type="backbone")
```

**为什么重要**：backbone 边编码文档的**线性阅读顺序**，让 DFS/BFS 可以沿着文档流走，找到物理相邻但可能引用不同模态的段落对。这是让 bridge 路径从"语义图遍历"变成"文档流遍历"的关键。

**返回**：
- `nodes`：`{node_id: Node}`
- `edges`：`List[Edge]`
- `out_adj`：`{node_id: {neighbor_id: Edge}}`（正向邻接）
- `in_adj`：`{node_id: {neighbor_id: Edge}}`（反向邻接）
- `stats`：每篇文档的节点/边统计

---

### `build_cross_doc_edges(citation_graph, nodes, out_adj, in_adj) -> Tuple[edges, cited_pairs]`
**作用**：在文档图之间添加跨文档引用边。

**策略**：对每条引用 (src_doc → tgt_doc)：
1. 在 src_doc 中找**出度最高的段落**（引用元素最多的段落，代表最"活跃"的跨文档讨论段）
2. 在 tgt_doc 中找**入度最高的元素**（被引用最多的核心元素）
3. 连接这对 (段落, 元素)，类型 `cross_doc_cite`

**效果**：123 条跨文档引用边，覆盖 55 篇文档的最大连通分量。

---

## 五、Hub 检测

### `compute_bridge_hubs(nodes, out_adj, in_adj) -> List[Dict]`
**作用**：识别 "bridge hub" 节点——连接 2+ 种模态的关键段落。

**Bridge Score 公式**：
```
bridge_score = num_modalities × 15 + out_to_elements × 2
```

- `num_modalities`：该节点的邻居中有多少种元素类型（figure/table/formula）
- `out_to_elements`：直接引用元素的数量

**Filter**：只保留 paragraph 和 element 类型节点；排除 authority sinks（高入度、低出度的被大量引用的公式节点）。

**结果**：60 个 bridge hubs，all-3 modality（figure+table+formula）31 个，fig+formula 25 个。

---

### `compute_hubs(nodes, out_adj, in_adj, top_k=60, keyword_boost=True) -> List[Dict]`
**作用**：更综合的 hub 评分，加入 PageRank 和关键词 boost。

**Hub Score 组成**：

| 组件 | 权重 | 说明 |
|------|------|------|
| Bridge score | 0.5 | 多模态覆盖（同上） |
| PageRank | 0.25 | 结构中心性（迭代 20 次） |
| Keyword boost | 0.15 | 节点在 Introduction/Methods/Experiments 章节中 |
| Out-connectivity | 0.10 | 出度归一化 |

`keyword_boost=True`（`--keyword-boost` 开关）：检测 section_title 是否包含核心章节关键词，命中则给 +0.2 bonus。

---

### `compute_adjacent_backbone_bridges(nodes, edges, out_adj) -> List[Dict]`
**作用**：找 backbone 上相邻段落形成的 bridge span，扩充 hub coverage。

**逻辑**：遍历所有 backbone 边 (P_i → P_{i+1})：
- 如果 P_i 引用 element_A（类型 X）
- 且 P_{i+1} 引用 element_B（类型 Y）
- 且 X ≠ Y（跨模态）

→ 记录为一个 adjacent_backbone_bridge，包含 {P_i, P_{i+1}, element_A, element_B}。

**效果**：369 条 adjacent backbone bridge spans，将 hub coverage 从 9.53% 扩展到 90.95%。

---

## 六、候选路径枚举

### `enumerate_candidates_from_bridge_hubs(nodes, edges, out_adj, in_adj, hubs, ...) -> List[Dict]`
**作用**：从 bridge hub 出发，枚举 2-hop 和 3-hop 多模态路径，生成候选对。

**三种路径类型**：

```
2-hop direct:   hub → [backbone] → element
3-hop chain:    hub → para1 → [backbone] → para2 → element
cross-doc:      element_A → [cross_doc_cite] → element_B（不同论文）
```

**去重与 cap**：
- 按 `(element_a_id, element_b_id)` 去重（无向）
- 每个 hub 每种 pair_type（fig+tbl/fig+formula/formula+tbl）最多贡献 `per_combo_cap` 条

**输出字段**：
```json
{
  "element_a_id": "1802.08139_figure_3",
  "element_b_id": "1802.08139_table_1",
  "path": ["node_a", "bridge_para", "node_b"],
  "hop_distance": 3,
  "hub_node_id": "1802.08139::p::00012",
  "page_span": {"a": 3, "b": 5},
  "line_no_span": {"a": 1204, "b": 1897},
  "short_query_seed": "Why does Figure 3 contradict Table 1?",
  "long_query_seed": "What if the method in Figure 3 were applied..."
}
```

---

### `build_query_seeds(nodes, path) -> Tuple[str, str]`
**作用**：为候选对生成确定性的 seed question，作为 prompt 中的出发点。

**四类 seed 类型（按 `hash(tuple(path)) % 4` 轮换）**：

| 类型 | 模板示例 |
|------|---------|
| WHY | "Why does {elem_a} produce values consistent with {elem_b}?" |
| WHAT_IF | "What if the approach in {elem_a} were applied to the scenario shown in {elem_b}?" |
| MISMATCH | "What discrepancy exists between {elem_a} and {elem_b}?" |
| CONDITION | "Under what conditions would {elem_a} and {elem_b} yield the same result?" |

---

## 七、Main Pipeline

### `main()`
**关键 CLI 参数**：

| 参数 | 说明 |
|------|------|
| `--reference-graph` | `latex_reference_graph.json` |
| `--citation-graph` | `citation_graph.json` |
| `--multimodal-elements` | `multimodal_elements.json` |
| `--output` | 候选对输出（500 条） |
| `--output-hubs` | Hub 列表输出 |
| `--output-report` | 拓扑统计报告 |
| `--keyword-boost` | 启用关键词 boost（推荐） |
| `--single-doc-only` | 只生成文档内路径（不含跨文档） |
| `--max-candidates` | 每个 hub 最多多少条候选 |

**输出统计（keyword_boost 版，2026-03-24）**：
- 节点：2551，边：3471
- bridge_hubs：60 个，adjacent_backbone_bridges：369 条
- 候选对：500 条（figure+formula:247 / figure+table:153 / formula+table:100）
- 跨文档：170/500（34%）
- label 匹配率：49.8%

---

## 附：关键设计决策汇总

| 决策 | 原因 |
|------|------|
| Backbone 边按 line_no 排序，而非语义相似度 | 文档流是最可靠的无监督结构信号；DFS/BFS 沿 backbone 走自然产生物理相邻的跨模态对 |
| Authority sink 排除 | 高被引公式（in-degree 高）会主导旧评分，但它们是被引用目标，不是 bridge 媒介 |
| 用 log2 加权 paragraph_ref | 多次引用同一元素的段落更可能是核心 bridge，但边际效益递减 |
| adjacent_backbone_bridges 将 hub_overlap 从 9.5%→90.9% | 不仅 top-60 hub 节点参与候选，邻接 backbone 段落也可作为 bridge，大幅扩充覆盖 |
| 顺序映射作为 label 匹配的第 4 级 fallback | LaTeX 第 N 个 figure label 很可能对应 MinerU 第 N 个 figure，无论编号格式如何 |
