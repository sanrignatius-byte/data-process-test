# Data Delivery — README (delivery_v1)

入口文档。介绍当前交付的数据是什么、怎么生成的、怎么用。

- **当前交付包**：`data/03_queries/delivery_v1_2026-04-13.jsonl`（473 条 cross-modal multi-hop queries，53 篇 arXiv CS 论文）
- **训练格式**：`data/07_training/delivery_v1/{train,val,test}.jsonl`（407 / 44 / 22 triplets）
- **演示物料**：`data/06_evidence_export/demo_showcase/`（10 条精选）+ `data/06_evidence_export/delivery_v1/index.md`（全量 473 条目录）

## 配套文档

| 文档 | 内容 |
|------|------|
| [DATA_SCHEMA.md](./DATA_SCHEMA.md) | delivery JSONL 27 个字段的逐字段说明 + 训练 Triplet 字段 |
| [DATA_DOMAIN_STATS.md](./DATA_DOMAIN_STATS.md) | 语料 / 模态 / 推理深度 / grounding 置信度等统计分布 |
| [GRAPH_ARCHITECTURE.md](./GRAPH_ARCHITECTURE.md) | 图构建原理（节点 / 边 / hub 评分），本文不重复 |
| [M4_SCHEMAS.md](./M4_SCHEMAS.md) | 设计阶段的 multi-hop / cross-doc / multi-turn schema 草案 |

## 一句话概括数据是什么

每条样本是一个面向**学术论文中跨模态证据链**的问答对：query 同时引用 ≥2 个模态元素（figure / table / formula 中的两个或多个），answer 必须建立在两端元素的联合证据之上，并附带 reasoning chain、required evidence spans、visual anchors 等结构化标注。

## 生成原理（5 步流水线）

```
  arXiv PDF/LaTeX
        │
        ▼
  ① MinerU 解析       →  multimodal_elements.json
                        （figure/table/formula + caption + context + page_idx）
        │
        ▼
  ② LaTeX 引用图       →  latex_reference_graph.json
                        （\ref{} / \cite{} / \label{} 构成 DAG，含 bridge paragraph 边）
        │     → 见 GRAPH_ARCHITECTURE.md
        ▼
  ③ 候选 pair 选取     →  intra_doc pairs / hub_candidates
                        （图上 2–5 hop 路径 + 严格 intra-doc 边界）
        │
        ▼
  ④ LLM 生成 (Claude/GPT)
        │   ・prompt 注入桥接段落原文（P0–P4 bridge grounding）
        │   ・两种风格：academic / real_user
        │   ・可选 PersonaHub 人设
        ▼
  ⑤ 多层 QC
            ・Rule QC（anchor_leak / single_element_answer / template_shortcut … ~20 项）
            ・LLM Ablation（单证据 / drop-element 不可答性测试）
            ・LLM Grounding（answer 是否真有 evidence 支撑）
            ─────────────
        ▼
  delivery_v1_2026-04-13.jsonl  (qc_pass=true only)
```

图构建是核心创新，详见 [`GRAPH_ARCHITECTURE.md`](./GRAPH_ARCHITECTURE.md)。

## Quick start

```bash
# 1. 浏览原始交付
head -1 data/03_queries/delivery_v1_2026-04-13.jsonl | python3 -m json.tool

# 2. 看摘要
cat data/03_queries/delivery_v1_2026-04-13_stats.json
cat data/03_queries/delivery_v1_2026-04-13_stats_extended.json

# 3. 看 10 条精选演示
xdg-open data/06_evidence_export/demo_showcase/README.md

# 4. 看全量 473 条逐条 evidence MD
xdg-open data/06_evidence_export/delivery_v1/index.md

# 5. 重新生成训练 triplet（修改负样本策略 / 比例）
python3 scripts/normalize_queries.py \
    --l1 /dev/null \
    --l2 <l2_records>.jsonl \
    --l3 <l3_records>.jsonl \
    --output data/05_eval/delivery_v1_normalized.jsonl
python3 scripts/export_training_data.py \
    --queries data/05_eval/delivery_v1_normalized.jsonl \
    --elements data/01_graphs/multimodal_elements.json \
    --output data/07_training/delivery_v1 \
    --num-negatives 3 --negative-strategy graph_aware
```

## 已知局限

1. **Domain 单一**：53 篇文档全部来自 arXiv CS（种子 1908.09635 引用网络），主要为 algorithmic fairness / LLM 相关。**待扩**：1040 篇 pruned graph 已就绪，待 enrichment 与新 API key。
2. **L3 真多跳条数偏低**：`m4_is_true_multihop` heuristic 标记的严格多跳为 0 条；`step_deletion_proxy` 通过 86 条。当前数据更多是 dual-evidence（两端证据组合）而非严格三步因果链。详见 [`M4_STRATEGY_REVIEW_2026-03-18.md`](./M4_STRATEGY_REVIEW_2026-03-18.md)。
3. **Cross-document = 0**：本批次全部为 intra-doc pairs。跨文档 element-level edges 在路线图 Phase 2。
4. **Multi-turn = 0**：本批次全部为单轮 query。Multi-turn 在 Phase 3。

## 路线图衔接

| 状态 | 目标 |
|------|------|
| ✅ 已交付 | 473 条 intra-doc cross-modal queries，含 reasoning_chain + qc_metrics |
| ⏳ 进行中 | 1040 篇 enrichment + bridge enrichment（阻塞于 API key） |
| ⏳ 下一步 | 非 CS 领域语料采集（医疗 / 金融 / 物理）+ Method C scale-up |
| ⏳ Phase 2 | element-level cross-document edges |
| ⏳ Phase 3 | multi-turn session 化 |
