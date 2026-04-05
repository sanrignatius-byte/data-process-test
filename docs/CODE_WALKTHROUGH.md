# Document Graph 代码讲解手稿

> 写给 mentor 看的，但其实先是写给自己复习的。
> 截止 2026-04-05，基于最近活跃代码整理。

---

## 这个项目到底在干嘛

一句话：**给学术论文建多层异构图，然后基于图结构生成高质量的跨模态检索 query，最终打包成可训练数据。**

图是核心贡献，query 是图的第一个应用——但也是现阶段的主要交付物。

整个 pipeline 长这样（后面一节一节讲）：

```
论文 PDF/LaTeX 下载
  ↓
MinerU 解析 + LaTeX \ref{} 解析
  ↓
多层图构建（文档内 DAG + 跨文档引用 + backbone 阅读链 + embedding 语义边）
  ↓
Hub 检测 + 候选路径枚举（500 对跨模态候选）
  ↓
MoDora 语义增强（给 figure/table/formula 加结构化描述）
  ↓
LLM Query 生成（L1/L2/L3 三级难度）
  ↓
QC 质量闸门（15+ 原子检查组合）
  ↓
检索评测（BM25 基线 vs 图增强，5 种方法对比）
  ↓
训练数据导出（normalize → triplet → doc-level split → JSONL）
```

---

## 完整 Pipeline 走一遍

下面按实际执行顺序，把每个阶段对应的脚本和 src 模块串起来。这是那种"从头跑一遍会经过什么"的流程。

### 第一步：搞论文

`scripts/download_latex_sources.py` 从 arXiv 批量下载 LaTeX 源码。种子论文是 `1908.09635`（算法公平性方向），通过引用网络 BFS 扩展到 86 篇。另外还有 `scripts/download_papers_semantic_scholar.py` 从 Semantic Scholar API 拉更多论文。

PDF 这边，`src/parsers/pdf_downloader.py` 下载 PDF，然后用 MinerU 解析——MinerU 相关的解析逻辑在 `src/parsers/mineru_parser.py`（1057 行，处理 MinerU 的 content_list.json 和输出格式），`src/parsers/modal_extractor.py` 负责从解析结果中提取 figure/table/formula 等模态元素。

最终产物：`data/multimodal_elements.json`——1316 个多模态元素（841 figure + 334 table + 141 formula），每个元素带 caption、content、context_before、image_path 等字段。

### 第二步：建图——这是核心中的核心

图有好几层，每层用不同的方式构建：

**层一：文档内 LaTeX 引用 DAG**

`src/parsers/latex_reference_extractor.py`（1454 行，仅次于最大脚本）是解析引擎。`LaTeXReferenceExtractor` 用正则扫 .tex 文件，把 `\label{}`、`\ref{}`、`\cite{}` 全部提取出来。它不用 AST 解析——82 篇论文各种奇葩 LaTeX 写法，正则反而最鲁棒。有个 `_merged_idx` 字段专门处理 `\input{}` 多文件合并后的行号重映射，这个设计挺讲究的。

`scripts/build_latex_reference_graph.py` 调用上面的 extractor，输出每篇文档的 labels / refs / edges / bib_entries。这里有个关键步骤 `merge_with_multimodal()`——把 LaTeX 的 label（比如 `fig:results`）映射到 MinerU 解析出来的 element（比如 `1306.5204_figure_2`）。匹配率只有 49.8%，用的是两级策略：先精确 Jaccard（阈值 0.25），失败则用数字后缀 fallback（"Figure 3" 匹配第 3 个 figure element）。这个匹配率不高是已知瓶颈，但足够用了。

**层二：文档内多模态关系 DAG**

`src/linkers/multimodal_relationship_builder.py`（1318 行）从 MinerU 的输出构建文档内多模态 DAG。这一层跟 LaTeX 层互补——MinerU 能看到 PDF 的物理布局，LaTeX 能看到语义引用关系。

**层三：跨文档引用图**

`scripts/build_citation_graph.py` 读 .bbl 文件里的参考文献条目，跟语料库中的论文做匹配。匹配策略有四级优先：arXiv ID 精确(1.0) → bare ID(0.9) → 精确标题(0.95) → fuzzy 标题(Jaccard≥0.55)。还有个 `compute_match_margin()` 算最佳和次佳匹配的置信度差距，差距太小就拒绝——宁可不匹配也不要匹配错。最终产出 123 条跨文档引用边，55 篇论文形成最大连通分量。

**层四：拓扑分析 + Hub 检测**

`scripts/analyze_latex_graph_topology.py` 是整个项目最长的脚本（2015 行），干了四件大事：

1. **Backbone edges**：把同文档内的段落按 line_no 排序连成阅读顺序链（para→para→para），1269 条边
2. **Bridge-first hub 评分**：`hub_score = num_modalities×15 + out_to_elements×2`。故意压制 authority hub（那种被引用 49 次但只连一种模态的公式节点）——我们要的是连接多种模态的"桥梁"，不是单纯被引用多的"权威"
3. **Targeted enumeration**：最初用 DFS 找跨模态路径，但 backbone 边形成的长链（para→para→para...）会让 DFS 在里面转圈到不了不同模态。改成直接枚举 2-hop direct + 3-hop via backbone neighbor + cross-doc 路径
4. **Real page index**：MinerU 的 `multimodal_elements.json` 里 page_idx 全是 0（parser bug），但 content_list.json 里有真实页码，用 sequential type-order matching 实现 94.8% 覆盖

产出：60 个 bridge hubs + 369 个 adjacent backbone bridges + **500 条候选对**。

数据结构方面，图的节点和边用 `src/models/__init__.py` 里的 `Node` 和 `Edge` dataclass 表示。`Edge.key()` 返回 `(source_id, target_id, edge_type)` 三元组，用于 set 去重——图构建过程中重复边很多，这个 O(1) 去重很关键。

**层五（实验性）：Embedding 语义边**

`scripts/build_embedding_edges.py`（797 行）用 embedding 相似度给元素之间加虚拟边。支持两个后端：sentence-transformers（纯文本，快）和 Qwen3-VL-Embedding（多模态，需要 GPU）。输出兼容评测脚本的 `--embedding-edges` 参数。这是 Phase 2B 的东西，还在实验阶段。

### 第三步：Enrichment——给元素"化妆"

这一步的目标是让每个元素的文本表示更丰富、更语义化，方便下游的 query 生成和检索。

**元素级 enrichment（MoDora 风格）**

`scripts/enrich_elements_modora.py` 对 figure/table/formula 三类元素分别用特化 prompt 生成结构化描述：
- `[T]` enriched_title——简洁标题
- `[M]` enriched_metadata——关键词、元素子类型、关键变量
- `[C]` enriched_content——详细语义描述

每种模态用不同的 prompt（figure 强调视觉特征，table 强调列头和行关系，formula 强调符号语义），输出结构化 JSON。这个脚本用了 `src/api/` 统一调 LLM，支持 `--incremental` 断点续跑，`--dry-run` 不花钱先看效果。

**Section 级 enrichment**

`scripts/enrich_section_nodes.py` 给 1417 个 section/subsection 节点生成描述。同样走 `src/api.call_llm` + `src/api.parse_json`——跟 element enrichment 共享 API 层，不是各写各的。支持 `--incremental` + `--flush-every` 断点续跑。

**Hub 候选富化**

`scripts/enrich_hub_candidates.py` 把拓扑层面的 `(node_a, node_b, path)` 三元组映射回 MinerU 的完整元素数据（图像路径、caption、context），还生成 `hub_semantic_summary`（50-80 词的压缩重写）。这里的 `load_section_enrichments()` 会把 section enrichment 结果也注入进去，给路径上经过的 section 节点加上 `[SECTION]` 标签。

这一步的产物就是 `hub_candidates_enriched.json`——格式跟 `generate_multihop_l1_queries.py` 直接兼容，可以无缝喂进去。

### 第四步：Query 生成——让 LLM 干活

`scripts/generate_multihop_l1_queries.py`（1310 行）是生成阶段的核心。它的主循环：

```
遍历 candidate pairs
  → 编码图像（src/utils/image_utils.py 的 encode_image/load_image_b64）
  → 选 prompt 模板（src/prompts/styles.py 的 select_template）
  → 可选注入 persona（src/prompts/personas.py 的 inject_persona_prefix）
  → call_llm()（src/api/ 统一调用层）
  → parse_json()（两级策略：先 json.loads 快路径，失败走花括号深度扫描）
  → qc_multihop_query()（src/qc/pipelines.py 的 15+ 检查组合）
  → 写 JSONL（full 版 + pass 版）
  → 脚本末尾 log_run()（铁律：不记 token 不合并）
```

**三条 LLM 调用路径**（在 `src/api/__init__.py` 里）：
- `provider="anthropic"` → client.messages.create() 直连 Claude
- `provider="openai"` → client.chat.completions.create() 走 OpenAI SDK
- `provider="company"` → wrap_requests_call() + SSE 流式解析走公司代理 yunwu.ai

company 路径的 `collect_company_stream()` 逐行读 `data: {...}` 格式的 SSE 流，从每个 chunk 的 `choices[0].delta.content` 拼接文本，最后从 `usage` 字段取 token 数。

**Prompt 工程层**（`src/prompts/`）：
- `templates.py`：11 个模板——6 个学术风格（按模态组合 figure+table / figure+formula / formula+table，再按 hop 距离分）+ 5 个 real-user 风格（factual / summary / comparison / how_works / what_if）
- `styles.py`：`select_template()` 是路由器，先按 style 分轨，再按模态组合选模板。`mixed` 模式用 pair_id 的 md5 哈希 50/50 分配，保证可复现
- `personas.py`：76 个 PersonaHub 人设，`resolve_persona()` 同样用 md5 哈希确定性分配。`inject_persona_prefix()` 用正则找到 prompt 首句 "You are a ..."，替换成 persona 描述——不是简单拼接，是替换，避免双重角色定义

**L3 的核心创新：Bridge Text Injection**。通过 `--reference-graph` 从 `latex_reference_graph.json` 提取边 context（作者在论文中连接两个元素时写的原句），注入到 prompt 里。这样 LLM 会用论文实际术语生成 query，BM25 基线 MRR 从 0.597 跳到 0.733——涨了 0.135，这个提升相当暴力。

**其他生成脚本**：
- `scripts/generate_l2_queries.py`：跨文档 L2 query 生成
- `scripts/generate_long_chain_iterative_queries.py`（1105 行）：一个全新的生成范式——不是一次性让 LLM 生成整个 query，而是沿图路径逐步生成：先给每个中间节点出子查询，再基于 bridge facts 出最终 query。还跑 ablation QC（删掉中间节点后还能答 = fake_long_chain）
- `scripts/generate_multiturn_sessions.py`（729 行）：把 L2/L3 单跳 query 分解成 2-3 轮对话。含 `context_isolation_score()`（Jaccard 代理，阈值 0.35）和 intent_shift 类型系统。M4 Phase 3 的关键脚本
- `scripts/build_latex_long_chain_candidates.py`：为迭代生成脚本提供长链候选（hop≥2）

### 第五步：QC 质量闸门——不合格的一个都不放过

QC 体系住在 `src/qc/`，分成四个文件：

**`constants.py`** 存阈值和词表。几个关键数：
- `ANCHOR_LEAK_THRESHOLD = 0.20`：query 与 visual anchor 的 Jaccard 超过 20% 就算泄漏
- `ANSWER_BALANCE_THRESHOLD = 0.20`：答案不能只依赖一个元素
- `MAX_QUERY_WORDS = 30`：query 不能超过 30 词
- `BAD_META_PATTERNS`：硬禁 "figure"、"table"、"equation" 等元语言——出现就 fail，因为会让检索变成简单关键词匹配

**`checks.py`** 是原子检查函数（25+ 个），每个只干一件事：
- `has_numeric_leakage(query)` — 检查 query 是否泄露具体数值（0、1 和年份豁免）
- `check_single_element_answer()` — 答案是不是只用了一个元素就能答（计算双端 token 重叠的 balance）
- `anchor_leak_jaccard()` — query 跟 visual anchor 的 Jaccard
- `is_yes_no_question()` — 不只看开头词，还处理倒装句（"In the context of X, does..."）
- `has_template_collapse()` — 检测 "How does X relate to Y" 这种高频 shell pattern
- `formula_symbol_hit()` — figure+formula 对里答案必须引用公式符号

**`pipelines.py`** 把原子检查组合起来：
- `qc_multihop_query()` 跑 15+ 个检查（有先后顺序），返回 `(issues, metrics)`。anchor leakage 有 amnesty 机制——如果重叠 token 全是领域必需词（像 "accuracy"、"f1"）就豁免，不误杀
- `qc_real_user_query()` 更宽松——不查 template shortcuts，yes/no 只是 issue 不是硬 fail，新增 retrievability_score

**`reasoning.py`** 做推理结构分析：
- `classify_reasoning_structure()` 用因果连接词（because/therefore/leads to）区分 parallel（两个证据并行取证）和 serial（真正多跳推理链）
- `classify_query_intent()` 分类 query 意图为 objective 或 subjective

### 第六步：检索评测——图到底有没有用？

**主评测脚本：`scripts/run_phase0_eval_ab.py`**（1350 行）

固定 query pool + 1314 chunks 候选库，跑 5 种检索方法对比：

```
bm25               — 纯 BM25 基线
graph_hub_rerank    — BM25 + hub 静态先验加权
graph_neighbor_prop — BM25 + 1-hop 邻域标签传播（核心信号！）
graph_citation_walk — BM25 + 跨文档引用传播
graph_full          — hw×hub + nw×nprop + cw×cite 加权组合
```

BM25 实现在 `src/retrieval/__init__.py` 里的 `BM25Lite`——纯标准库（math + collections），零外部依赖。标准 Okapi BM25 公式，k1=1.5，b=0.75。同文件还有评测指标：`reciprocal_rank_binary`（MRR）、`coverage_at_k`（Recall@k）、`ndcg_at_k`。

关键发现：**neighbor_prop 是最有效的图信号**。原理是：如果某个 chunk 的 element 在图中跟 query evidence 的 element 是 1-hop 邻居，就传播一个 boost。2-hop 反而不如 1-hop（信号衰减 > 噪声引入）。最优配置 `nw=1.00, hw=0.15, cw=0`（citation walk 是负贡献，关掉了）。

graph_full 最终成绩：R@10=0.8736 (+0.0269 vs BM25), MRR=0.6045 (+0.0403 vs BM25)。

**其他评测脚本**：
- `scripts/run_m2_classic_eval.py`（502 行）——实现了 8 种检索方法（bm25 / bm25f / lm_dirichlet / prf / bm25_title_boost / oracle_proximity / hits / rrf），是 enrichment 消融实验的底层引擎
- `scripts/eval_cpool_keyword_boost_graph.py`（621 行）——C-Pool（78 条通用学术 query）专用评测
- `scripts/run_ablation_enrich.py`——2×2 Enrichment 消融实验（raw/enriched query × raw/enriched corpus），核心发现是双端 enrichment 的超线性增益——L3 R@10 从 0.333 翻到 0.690
- `scripts/run_exp_a_difficulty.py` / `scripts/run_exp_c_qa_triangle.py`——M2 阶段的实验脚本

### 第七步：训练数据导出——最后收割

这一步是 Phase A Training Pipeline 的交付物，分两步走：

**Step 1：Normalize**

`scripts/normalize_queries.py` 把历史上各种格式的 L1/L2/L3 JSONL 统一成 `StandardQuery` schema（定义在 `src/models/training.py`）。

`StandardQuery` 是个 Pydantic BaseModel，重点字段：
- `difficulty_level`：L1（单元素）/ L2（双证据/跨文档）/ L3（多跳推理链）
- `evidence_spans`：至少 1 个（`@field_validator` 强制），每个是 `EvidenceSpan`（element_id + doc_id + span_text）
- `reasoning_steps`：L3 专属（`@model_validator` 强制——L3 没有推理步骤直接报错）
- `ReasoningStep` 里的 `depends_on_steps` 建立步骤间依赖（step 2 依赖 step 1 的结论），`reasoning_role` 分 premise / bridge / conclusion

**Step 2：Export**

`scripts/export_training_data.py` → `src/export/dataset_builder.py` 的 `DatasetBuilder`：

```
加载 StandardQuery
  → _build_triplets()：query + positive evidence + NegativeSampler 采负样本 → Triplet
  → _doc_stratified_split()：按 doc_id 的 md5 哈希分 train/val/test
  → 写 JSONL + manifest.json
```

**doc-level hash split 是防泄漏的关键**：同一篇论文的所有 query 只会出现在同一个 split 里。用 `md5(doc_id) % 1000`，< 900 进 train，900-949 进 val，≥ 950 进 test。确定性、可复现、零泄漏。

**负样本采样**住在 `src/sampling/negative_sampler.py`：
- `NegativeSampler` 是 Protocol（结构化类型），不是 ABC——只要实现 `sample()` 方法就行
- `HeuristicNegativeSampler`：两种策略
  - `random`——排除 positive 后随机采
  - `in_doc_swap`——优先从同文档采（hard negative），不够从其他文档补
- `GraphAwareNegativeSampler`——stub，预留图感知负样本，当前 fallback 到 random
- `build_sampler()` 工厂函数按 config 实例化

有个被 review 揪出来的 bug 值得说：`_in_doc_swap` 原来从 element_id 字符串 rsplit 猜 doc_id，arXiv ID 有多个下划线会猜错。改成直接用 `Chunk.doc_id` 字段。另外 `pos_doc_ids` 必须在过滤 pool **之前**从 candidates 中提取——不然 positive chunks 已被移除就取不到 doc_id 了。教训：**永远不要从 element_id 字符串里猜 doc_id，用 doc_id 字段。**

---

## `src/` 共享库速览

上面 pipeline 走过的模块已经讲完了，这里把其余的补齐：

### `src/utils/`——工具层

**`text_utils.py`** 有三种分词函数（别搞混了）：
- `tokenize()` — `[a-zA-Z]{3,}` 小写化，用于拓扑分析和 hub enrichment
- `tokenize_words()` — `\w+` 小写化，含数字，用于 enrich_hub_candidates
- `tokenize_for_retrieval()` — 字母开头 2+ 字符，返回 list（保留顺序和重复），给 BM25 用

还有一堆工具函数：`jaccard()` / `overlap_ratio()`（集合相似度）、`content_tokens()`（去停用词后的内容 token，给 QC 用）、`extract_formula_variables()`（从 LaTeX 公式提取变量名，给 prompt 用）、`extract_table_headers()`（从 HTML/markdown table 提取表头）。

**`image_utils.py`** 最值得说的是 `resolve_image_path()`——维护了一个 `_KNOWN_PREFIXES` 列表，处理集群 / 本地 / CI 三种环境下图像路径不一致的问题。四级策略：直接路径 → 已知前缀剥离 → `/data/mineru_output/` 后缀提取 → 通用 `/data/` 重定向。

**`token_logger.py`**——铁律的执行者。每次调 LLM 的脚本结束时必须 `log_run()`，记到 SQLite（`logs/token_usage.db`）。支持 CLI 查看：`python src/utils/token_logger.py --days 7`、`--csv > report.csv` 导出给老板看。

### `src/linkers/`——关系构建

- `multimodal_relationship_builder.py`（1318 行）从 MinerU 输出建文档内多模态 DAG
- `cross_document_linker.py`（619 行）做跨文档实体链接 + 证据链构建。`EvidenceChain` 有三个维度验证：`.is_valid_multi_hop()` / `.is_valid_multi_doc()` / `.is_valid_multi_modal()`

### `src/graph/`——规划中

目前 `__init__.py` 里 `__all__ = []`，但 docstring 已经规划了 PageRank、label propagation、hub detection、candidate enumeration。当前图算法逻辑在 `analyze_latex_graph_topology.py` 里，计划迁入这里。

---

## 测试

62 个 pytest（`python -m pytest tests/ -v`）：
- `test_qc_checks.py`（27 个）——QC 原子检查的正反例
- `test_schema.py`（9 个）——Pydantic 模型验证（L3 强制推理步骤等）
- `test_text_utils.py`（14 个）——分词 / Jaccard / 公式符号提取
- `test_negative_sampling.py`（12 个）——负样本策略（random / in_doc_swap / padding）

---

## 几条铁律

1. **Token 必须记录**——任何调 LLM 的脚本结束时必须调 `log_run()`，不记不合并
2. **不要从 element_id 字符串猜 doc_id**——`EvidenceSpan` 和 `Chunk` 都有 `doc_id` 字段，直接用
3. **doc-level hash split 防泄漏**——同文档 query 只出现在同一个 train/val/test split
4. **图是核心贡献，query 是副产物**——graph 构建零 LLM 成本，LLM 只用于 enrichment 和 query 生成
