# MoDora 整合分析：CCTree 思路与我们文档建图 pipeline 的对接

> 日期：2026-03-09
> 论文：[MoDora: Tree-Based Semi-Structured Document Analysis System](https://arxiv.org/abs/2602.23061) (SIGMOD 2026)
> 代码：https://github.com/weAIDB/MoDora

---

## 1. MoDora 核心方法论

MoDora 提出 **Component-Correlation Tree (CCTree)** 来表示半结构化文档，分 4 阶段：

### 1.1 OCR + Local-Alignment Aggregation
- PaddleOCR 提取碎片化 block → 按布局聚合为自包含 Component
- 每个 Component 保留 type / bbox / page 信息
- 目的：解决 OCR 输出碎片化、丢失语义上下文的问题

### 1.2 Enrichment（语义增强）
- 对每个非文本 component（图/表/chart），裁切 PDF 区域发给 LLM
- LLM 返回结构化三元组：
  - `[T]itle`：标题
  - `[M]etadata`：关键词、图表类型、坐标轴等
  - `[C]ontent`：详细内容描述
- 三种专用 prompt（image / chart / table），格式统一
- **关键**：将所有模态统一转为可搜索的文本表示

### 1.3 CCTree 构建
- **标题层级校正**：裁切标题区域图片 + 文本，发 LLM 判断 `#`/`##`/`###` 层级（比纯 OCR 检测更准）
- **Stack-based 建树**：按 title_level 驱动 push/pop，非文本元素附到当前 heading scope 下
- **Bottom-up Cascade Summarization**：
  - 叶节点：LLM 生成 n₀ 个关键词
  - 父节点：聚合子节点 metadata + 自身内容 → LLM 生成 summary
  - 关键词数量公式：`ceil(fraction_part + log₂(total_child_keywords^growth_rate) + 1)`
  - 保证高层节点有更多聚合关键词，形成**多粒度语义索引**

### 1.4 Question-Type-Aware Retrieval
- 位置类问题：3×3 grid 空间匹配
- 语义类问题：LLM 自顶向下剪枝（每层选择相关子节点下探）+ embedding fallback
- Impact tracking：每个被检索的 node 增加 `impact` 计数器，用于热力图可视化

---

## 2. 两个项目的根本差异

| 维度 | MoDora | 我们的项目 |
|------|--------|-----------|
| **目标** | 推理时 Document QA（零训练） | 生成训练数据 → 训练多模态检索 embedding |
| **文档表示** | 层级树（CCTree） | 多层图（backbone + cross-modal edges + citation edges） |
| **跨模态关系** | **隐式**（同一标题下共现即关联） | **显式**边（LaTeX `\ref{}` 共引、proximity、cross-doc citation） |
| **多跳推理** | LLM 自顶向下逐层剪枝（在线） | 预计算 multi-hop candidate paths + seed questions（离线） |
| **公式处理** | 未特殊处理 | LaTeX formula 作为一等公民图节点 |
| **跨文档** | 简单树合并（共享根节点，无边） | 123 条引用边 + 实体匹配 + utility-aware rerank |
| **模态增强** | LLM 生成 [T]/[M]/[C] 结构化描述 | MinerU 原始提取（caption + content） |
| **可扩展性** | 每 query 多次 LLM 调用（推理时昂贵） | 一次性生成成本，embedding 推理快速 |

**核心判断**：两个项目解决的是文档理解 pipeline 中不同环节的问题。MoDora 专注"如何更好地表示和检索文档"，我们专注"如何从文档中挖掘高质量多跳训练数据"。不存在直接替换关系，但有若干具体技术点值得借鉴。

---

## 3. 可借鉴的技术点

### 3.1 ⭐ P0.5：非文本元素 [T]/[M]/[C] Enrichment

**MoDora 做法**：
```
# 以 table 为例
[T] Table 2: Comparison of F1 scores across models
[M] Keywords: F1-score, BERT, RoBERTa, model comparison, NLP benchmark
[C] A performance comparison table with 4 rows and 3 columns...
```

**我们当前状况**：
- `multimodal_elements.json` 中的 `caption` 和 `content` 来自 MinerU 原始提取
- table 的 `content` 是 HTML 片段，formula 是 LaTeX 源码
- 给 `generate_multihop_l1_queries.py` 的 context 质量直接影响生成的 query 质量

**整合方案**：
```
新建 scripts/enrich_elements_modora.py
  输入：data/multimodal_elements.json + MinerU 输出的图片文件
  处理：对每个 figure/table/formula：
    1. 裁切对应区域图片（或直接用 image_path）
    2. 发 LLM（公司 API）+ 类型专用 prompt
    3. 返回 [T]itle / [M]etadata / [C]ontent
  输出：data/multimodal_elements_enriched.json
    - 新增字段：enriched_title, enriched_metadata, enriched_content
    - 保留原始字段不变
```

**预期收益**：
- query 生成时可以用 enriched_content 替代原始 content，提供更好的语义上下文
- embedding 训练时 element 的文本表示更丰富
- 成本：~1316 × LLM 调用 ≈ $2-3

**对下游的影响**：
- `enrich_hub_candidates.py` 可以读取 enriched 版本
- `generate_multihop_l1_queries.py` 中构建 prompt 时优先使用 enriched description
- 有望提升 QC pass rate（特别是 `single_element_answer` 失败场景——更好的 element 描述 → LLM 更容易理解如何构造需要两个 element 的 query）

---

### 3.2 ⭐ P1：Hub Cascade Summary（多粒度语义聚合）

**MoDora 做法**：
- 叶节点生成关键词 → 父节点聚合子节点 metadata → 逐层向上
- 高层节点自动获得其子树的语义摘要

**我们的改进点**：
- 当前 `enrich_hub_candidates.py` 只做了 LaTeX → MinerU 的 ID 映射
- hub 的 `bridge_text` 来自 LaTeX 源码原文，没有经过语义理解

**整合方案**：
```
改造 scripts/enrich_hub_candidates.py：
  对每个 bridge hub paragraph：
    1. 收集其引用的所有 element 的 enriched description（来自 3.1）
    2. 加上 hub paragraph 自身的 text_snippet
    3. 用 LLM 生成一段 hub_semantic_summary（50-100 词）
    4. 写入 enriched candidate 的 bridge_text_enriched 字段
```

**预期收益**：
- 更好的 bridge context → LLM 生成 query 时更准确地理解两个 element 之间的语义关系
- 有望降低 `bridge_entity_leakage` 和 `weak_reasoning_connector` 失败率

---

### 3.3 P2：LLM 辅助标题层级校正

**MoDora 做法**：裁切标题区域图片 + 发 LLM 判断层级

**我们的现状**：
- 有 LaTeX 源码 → `\section{}`/`\subsection{}` 可以直接解析
- 但 label 匹配率 49.8%（LaTeX label → MinerU element 的映射）还有提升空间

**判断**：当前有 LaTeX 源码时收益有限。如果后续要支持无源码的 PDF-only 场景，这个思路很重要。**暂缓**。

---

### 3.4 P3：Impact Tracking（检索热度追踪）

**MoDora 做法**：检索时 node.impact += 1（中间节点）或 += 2（叶节点）

**我们的应用**：
- 在评估闭环中，跑 BM25/embedding retrieval 时记录每个 element 被命中的频率
- 识别"冷 element"（从未被检索到）→ 针对性补充 query 覆盖
- 识别"热 element"（被过度检索）→ 可能需要 hard negative 去重

**判断**：等评估闭环落地时再实现，优先级最低。

---

## 4. 不适合借用的部分

| 部分 | 原因 |
|------|------|
| CCTree 树结构 | 我们的图（2551 nodes, 3471 edges, 4 种边类型）表达力远强于树 |
| LLM 逐层剪枝检索 | 每 query 多次 LLM 调用，太贵；我们是离线生成，不需要在线检索 |
| 多文档树合并 | 纯嵌套，无跨文档语义边；远不如我们的 123 条 citation edge |
| 零训练理念 | 我们的目标就是生成训练数据 |
| 隐式跨模态关系 | "共现即关联" 信号太弱；我们的 `\ref{}` 共引是显式、精确的信号 |

---

## 5. 代码层面的整合路线

### Phase 1：Element Enrichment（1-2 天）
```
scripts/enrich_elements_modora.py   [新建]
  ├── 读取 data/multimodal_elements.json
  ├── 对 figure/table/formula 分别使用专用 prompt
  ├── 支持 --provider company/anthropic/openai
  ├── 输出 data/multimodal_elements_enriched.json
  └── 增量模式：已有 enriched 的 element 跳过
```

### Phase 2：Hub Summary 增强（半天）
```
scripts/enrich_hub_candidates.py    [改造]
  ├── 读取 enriched elements（Phase 1 产出）
  ├── 对每个 bridge hub 聚合子元素 metadata
  ├── LLM 生成 hub_semantic_summary
  └── 写入 hub_candidates_enriched_v2.json
```

### Phase 3：Query 生成 prompt 升级（半天）
```
scripts/generate_multihop_l1_queries.py   [小改]
  ├── 读取 enriched elements 作为 context
  ├── 使用 hub_semantic_summary 替代原始 bridge_text
  └── 预期：QC pass rate 从 44.8% → 55%+
```

---

## 6. MoDora 代码可直接复用的片段

### 6.1 Enrichment Prompt 模板
来源：`MoDora-backend/src/modora/core/prompts/enrichment.py`

三套 prompt（image/chart/table），格式统一：
```
You are given an image of [type]. Extract the following:
[T] Title: ...
[M] Metadata: keywords describing ...
[C] Content: detailed description of ...
```
可以直接翻译/适配为我们的 enrichment prompt，无需从头设计。

### 6.2 OCRBlock 类型分类方法
来源：`MoDora-backend/src/modora/core/domain/ocr.py`

`is_title()`, `is_figure()`, `is_header()` 等方法的分类标签集合可以参考，用于改进我们的 MinerU 元素类型判断。

---

## 7. 总结

MoDora 的核心贡献（CCTree + LLM 检索）与我们的目标（训练数据生成）是正交的，但他们在**非文本元素语义增强**方面的工程做得很好。最值得借鉴的是：

1. **[T]/[M]/[C] Enrichment**：统一格式的元素描述 → 直接提升我们 query 生成的输入质量
2. **Cascade Summary**：多粒度语义聚合 → 改善 hub bridge context 质量

两者都是"上游数据质量"的改进，会通过 pipeline 传播到最终的 QC pass rate。建议先做 P0.5（element enrichment），效果验证后再做 P1。
