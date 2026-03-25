# 代码逐函数解读：`scripts/run_phase0_eval_ab.py`

> 这是本项目**使用频率最高的核心脚本**，负责所有检索方法的评测对比（BM25 基线 vs 六种 Graph 增强方法）。每次跑实验、调参、验证图效果都依赖此脚本。

---

## 整体架构

```
main()
 ├── 加载数据：load_jsonl → dedupe_queries → build_chunks
 ├── 加载图结构：load_element_hub_prior / load_element_adjacency / load_citation_adjacency
 ├── 构建 BM25 + TF-IDF 索引
 └── 循环跑 7 种方法 → evaluate_method() → 汇总 report → JSON 输出
```

---

## 数据结构

### `Chunk`（dataclass）
```python
@dataclass
class Chunk:
    chunk_id: str   # 即 element_id，如 "1801.04385_table_2"
    doc_id: str     # 论文 arXiv ID，如 "1801.04385"
    text: str       # caption + content + context_before/after + enriched 字段拼接，最长 1800 chars
```
检索的基本单元。每个 element（figure/table/formula）对应一个 Chunk。

---

## 函数解读

### `tokenize(text)`
**作用**：正则提取所有 `[A-Za-z][A-Za-z0-9_-]{1,}` 的 token，全小写。
**设计意图**：只保留英文词，过滤纯数字、单字符、LaTeX 符号。BM25 和 TF-IDF 都用这个分词。

---

### `BM25Lite.__init__(docs, k1=1.5, b=0.75)`
**作用**：构建 BM25 索引。
**关键数据结构**：
- `self.df`：每个 token 在多少篇文档中出现（document frequency）
- `self.tf_docs`：每篇文档的词频 Counter
- `self.avgdl`：平均文档长度，用于长度归一化

**参数含义**：
- `k1=1.5`：词频饱和系数，越大词频权重越线性
- `b=0.75`：长度归一化强度，1.0=完全归一化，0=不归一化

### `BM25Lite.idf(term)`
**公式**：`log(1 + (N - df + 0.5) / (df + 0.5))`
Okapi BM25 标准 IDF，罕见词得分高，stopword 近零。

### `BM25Lite.score(query_tokens, doc_idx)`
**公式**：`Σ idf(t) × (f × (k1+1)) / (f + k1 × (1 - b + b × dl/avgdl))`
对 query 中每个唯一 token 计算 BM25 分值累加，返回该文档对这条 query 的相关性分数。

---

### `load_jsonl(path)`
读取 `.jsonl` 文件，返回 dict 列表。每行是一条 query 的 JSON 对象。

### `dedupe_queries(rows)`
**作用**：去重，避免同一 query 重复计入评测。
**去重键**：优先用 `query_id`，没有则用 query 文本的 hash。

---

### `build_chunks(elements_json, max_chars=1800)`
**作用**：从 `multimodal_elements.json` 构建 Chunk 列表（候选库）。
**拼接字段顺序**：`caption → content → context_before → context_after → enriched_title → enriched_content`
**关键细节**：
- 若有 enriched 字段（MoDora 语义增强后）则追加，让 BM25 能匹配更丰富的语义描述
- 超过 1800 chars 截断，避免单个 chunk 过大影响 BM25 分布

---

### `load_doc_hub_prior(hubs_json)`
**作用**：从 `latex_graph_hubs.json` 加载**文档级** hub prior。
**逻辑**：每篇文档取其所有 hub 节点中最高的 `hub_score`，全局归一化到 [0,1]。
**用途**：`graph_hub_rerank` 的粗粒度 fallback（当 element 级 prior 找不到时）。

### `load_element_hub_prior(hub_candidates_json)`
**作用**：从 enriched hub candidates 加载**元素级** hub prior。
**逻辑**：
1. 遍历所有 hub candidate pair，记录每个 `element_a_id`/`element_b_id` 的最高 `quality_score`
2. 追加 `adjacent_bridge_elements`（由 `enrich_hub_candidates.py` 生成的邻接 bridge 元素）
3. 全局归一化到 [0,1]

**关键**：这里的 prior 是**静态的**，与 query 无关——只要 element 出现在 hub 候选对中就有加分。

### `load_element_adjacency(hub_candidates_json)`
**作用**：构建**元素邻接图**（element → 邻居 set + 边权重）。
**两种边来源**：
1. Hub candidate pairs：`element_a_id ↔ element_b_id`，权重 = `quality_score`
2. `adjacent_bridge_adjacency`：相邻 bridge 覆盖的元素对，默认权重 0.6

**返回**：
- `adj`：`element_id → set[neighbor_ids]`（无向图）
- `adj_weights`：`element_id → {neighbor_id: weight}`（每个元素出发的邻边权重，局部归一化）

### `load_citation_adjacency(citation_graph_json)`
**作用**：加载跨文档引用图。
**返回**：`doc_id → {"cites": set, "cited_by": set}`
即每篇文档引用哪些论文、被哪些论文引用（1-hop 邻居）。

---

### `span_overlap(span, text)`
**作用**：计算 evidence span 与 chunk 文本的字符重合率。
**逻辑**：
1. 若 span 是 text 的子串，直接返回 1.0
2. 否则用 `SequenceMatcher` 找最长公共子串，返回 `LCS长度 / span长度`

**用于**：当 query 没有 `required_evidence_spans` 的 element_id 时，用文本 overlap 作为 hit 判定的 fallback。

### `query_spans(q)` / `query_element_ids(q)`
从 `required_evidence_spans` 字段中分别提取 span 文本列表和 element_id 列表，作为 ground truth。

### `reciprocal_rank_binary(hit_ranks)`
`1 / min(hit_ranks)`，取命中排名中最高位的倒数，即 MRR 的单 query 计算。

---

### `_bm25_norm_scores(bm25, q_toks, n_chunks)`
**作用**：对所有 chunks 计算 BM25 分数，然后 min-max 归一化到 [0,1]。
**原因**：各 Graph 方法的 boost 值也在 [0,1] 范围，归一化后才能做线性叠加。

---

### `evaluate_method(method, queries, chunks, bm25, ..., top_k, ...)` ← 核心函数

对所有 queries 跑一个检索方法，返回 Recall@10、MRR 和 per-query 明细。

**内部流程（每条 query）**：
1. 按 method 计算每个 chunk 的得分
2. 按得分降序排列，取 top_k
3. 判断是否命中（GT element_id 在 top_k 中，或 span overlap ≥ threshold）
4. 累加 Recall@10 和 MRR

**六种方法的评分计算**：

#### `bm25`（基线）
```
score[i] = BM25(query, chunk_i)
```
纯文本词频检索，无图结构信息。

#### `dense`（TF-IDF cosine 基线）
```
score[i] = cosine(TF-IDF(query), TF-IDF(chunk_i))
```
用 sklearn TfidfVectorizer + 余弦相似度，用于与 BM25 对比。

#### `graph_hub_rerank`（静态 hub prior）
```
score[i] = norm_bm25[i] + alpha × hub_prior[chunk_id]
           （仅对 BM25 top-100 candidates 施加 boost）
```
- **hub_prior**：element 是否参与过 hub candidate pair，是则有加分（静态，与 query 无关）
- **candidate_set 限制**：只在 BM25 top-N 内做 hub boost，避免将排名很低的 hub element 无故拉高
- **hub_bm25_overlap_rate**：统计 BM25 top-10 与 hub 集合的重叠率，诊断 hub prior 对 BM25 的覆盖程度

#### `graph_neighbor_prop`（动态邻居传播）← 最重要的信号
```
neighbor_boost[n] += norm_bm25[seed] × decay × edge_weight(seed, n)
                     （对 BM25 top-N 中每个 seed 的所有 1-hop 邻居）
score[i] = norm_bm25[i] + neighbor_boost[i]
```
**直觉**：如果一个 figure 和这条 query 相关（BM25 高分），那么与它同在 hub 路径上的 table/formula 也可能是证据，应该往上提。
**关键参数**：
- `neighbor_decay=0.5`：控制传播强度，调参实验证明 0.5 是最优
- `edge_weight`：由 hub candidate quality_score 归一化得来，强连接传播更多
- `neighbor_hops=1`：调参实验证明 2-hop 反而更差

#### `graph_ppr`（Personalized PageRank）
```
teleport[node] = norm_bm25[node]（BM25 top-N 中有邻居的节点）
PPR 迭代 20 次，damping=0.85
score[i] = norm_bm25[i] + ppr[chunk_id] × decay
```
在元素邻接图上做带个性化向量的 PageRank。相比 neighbor_prop 能平滑多跳传播，但实验中效果与 neighbor_prop 相近（PPR≈0.8561 vs nprop≈0.8515）。

#### `graph_citation_walk`（跨文档引用传播）
```
doc_score[d] = max(norm_bm25[i] for i in chunks of d)
citation_boost[neighbor_doc] += doc_score[d] × citation_decay
                                （d cites neighbor → 全权；d cited_by neighbor → 半权；2-hop co-citation → 0.3权）
score[i] = norm_bm25[i] + citation_decay × citation_boost[chunk.doc_id]
```
**直觉**：如果论文 A 与 query 相关，则 A 引用的论文（基础参考文献）也可能含有证据。
**问题**：实验中 citation_walk 为负贡献（R@10 -0.012），原因是 doc 级粒度太粗，传播方向与 element 级证据位置不匹配。最优配置已关闭此组件（`cite_weight=0`）。

#### `graph_full`（组合方法）
```
score[i] = norm_bm25[i]
         + hub_w   × hub_prior[i]         （默认 0.1）
         + nprop_w × neighbor_boost[i]     （默认 1.0，内部有 decay 控制）
         + cite_w  × citation_boost[doc_i] （默认 0，已关闭）
```
三个组件的线性叠加，权重通过 `--hub-weight/--nprop-weight/--cite-weight` 独立调节。调参实验（A/B/C/D 四组）证明默认参数就是最优。

---

### `decision(graph_full_metrics, bm25_metrics)`
**作用**：判断是否达到论文/专利的实验达标线。
**规则**：`continue_expand = True if ΔR@10 ≥ 0.05 OR ΔMRR ≥ 0.03`
当前结果：ΔR@10=+0.0859, ΔMRR=+0.1165，**双双达标**。

---

### `main()`
**流程**：
1. 解析 CLI 参数（q1/q2/q3 query 文件、elements、hubs、hub-candidates、citation-graph 等）
2. 加载并去重 queries，构建 Chunk 候选库
3. 加载图结构数据（hub prior、邻接图、引用图）
4. 可选：合并 embedding 语义边（`--embedding-edges`，为 Phase 2B 预留）
5. 构建 BM25Lite + TF-IDF 索引
6. 顺序跑 7 种方法，每种调 `evaluate_method()`
7. **分层评测**（layered evaluation）：把 queries 按 GT element 是否在 hub prior 中分成两组（hub-overlap vs non-hub），分别统计 — 用于诊断图信号对哪类 query 有效
8. 输出 JSON 报告 + 控制台打印汇总表

**关键 CLI 参数速查**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--q1/--q2/--q3` | pass 文件 | query 输入文件（可叠加多个） |
| `--elements` | enriched elements | chunk 候选库来源 |
| `--hub-candidates` | enriched candidates | hub prior + 邻接图来源 |
| `--top-k` | 10 | Recall@k 的 k 值 |
| `--graph-alpha` | 0.2 | hub_rerank 的 boost 系数 |
| `--neighbor-decay` | 0.5 | 邻居传播衰减（调参最优值） |
| `--hub-weight` | None（同 alpha） | graph_full 中 hub 组件权重 |
| `--nprop-weight` | None（默认 1.0） | graph_full 中 nprop 组件权重 |
| `--cite-weight` | None（同 decay） | graph_full 中 citation 组件权重（设 0 禁用） |
| `--neighbor-hops` | 1 | 传播跳数（调参证明 1 最优） |
| `--embedding-edges` | None | 可选 embedding 语义边文件 |

---

## 当前最优配置（锁定）

```bash
python scripts/run_phase0_eval_ab.py \
    --q1 data/m2/l2_production_..._pass.jsonl \
    --q3 data/m2/l3_production_..._pass.jsonl \
    --elements data111/multimodal_elements_enriched.json \
    --hubs data/latex_sections_rebuild_2026-03-24/latex_graph_hubs_keyword_boost.json \
    --hub-candidates data/m2/hub_candidates_enriched_keyword_boost_full_2026-03-24.json \
    --citation-graph data/citation_graph.json \
    --hub-weight 0.1 --nprop-weight 1.0 --cite-weight 0.0 \
    --top-k 10
```

**结果**：graph_full R@10=0.8585（+8.59%），MRR=0.6339（+11.65%），`continue_expand=True`。
