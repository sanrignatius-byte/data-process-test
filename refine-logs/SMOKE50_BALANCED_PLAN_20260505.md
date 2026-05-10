# Plan — M4query Smoke50 Balanced Modality Audit (Mentor 录音 60 / C6)

**Date**: 2026-05-05
**Status**: Ready for execution
**Owner**: next coder assistant
**Target wall time**: ~90 min（30 min 采样 + 30 min slice & 评估 + 10 min GPU per-modality 重算 + 20 min 决策 & wiki）
**Budget**: $0 LLM, ~10 min GPU（仅 split modality 的 VL lane 需要重算 smoke50 子集）

---

## Why This Plan Exists

三轮 reranker / corpus 实验后，0.6913 R@10 ceiling 都没破：

| 路线 | 结果 |
|---|---|
| F2 corpus enrichment (MODORA 注入) | D5 antipattern（`exp:20260503_corpus_enrich_fix`） |
| Option A1: BGE-reranker-v2-m3 | 单跑 R@10 0.4482，rrf 0.6258（`exp:20260503_ce_rerank_bge`） |
| Option A2: Qwen3-Reranker-4B | 单跑 R@10 0.5613，rrf 0.6702（`exp:20260503_qwen3_rerank_fusion`） |

每次失败都给了一个"模态 bias"线索：BGE 偏文本，Qwen3 偏公式，MODORA 视觉描述偏 domain-detached——即 **passage modality style ≠ query modality style** (gap:G7)。

**但回看 M4query_v1 的 qrels modality 分布（用 BGE pilot top-1 gold passage 估算）**：

| modality | top-1 gold count | 占比 |
|---|---:|---:|
| figure | 265 | 56.0% |
| table | 115 | 24.3% |
| formula | 67 | 14.2% |
| text | 26 | 5.5% |

graph rerank 在 figure (R@10 0.7453) / table (0.7257) 这两个占评测 80% 的模态上特别强，但 formula (0.4972) 上明显弱，text 因样本太少不知道。**可能我们一直在打的 0.6913 ceiling 主要是 figure-table 性能的加权平均，而不是真正的"系统能力天花板"。**

Mentor 录音 60 的 **C6 显式要求**：「**50 query × 4 类型：10 文本 / 10 图 / 10 表 / 10 公式**」，至今未做。这个 plan 关闭 C6，并用它作为诊断工具回答："0.6913 是真天花板还是 figure-heavy artifact？"

---

## Pre-confirmed Inputs (do NOT redo)

| Data point | Value | Source |
|---|---|---|
| M4query_v1 总 query 数 | 473 | `data/03_queries/M4query_v1/queries.jsonl` |
| M4query_v1 总 qrel 数 | 946（avg 2 qrels/query） | `data/03_queries/M4query_v1/qrels.jsonl` |
| Top-1 gold modality 占比 | figure 56% / table 24% / formula 14% / text 5.5% | BGE pilot fusion_report.md |
| 当前 0.6913 ceiling | graph_explicit_only_fixed + static_plus_neighbor on M4query_v1 | `rebuilt_20260417/graph_explicit_only_fixed/metrics_graph_static_plus_neighbor.json` |
| 已有 ranking 文件（直接 slice 用，无需重跑）| dense / graph / BGE / Qwen3 / VL | 见 §Phase B 文件清单 |

---

## Phase A — 构建 Smoke50 (30 min, no GPU)

**目标**：从 M4query_v1 stratified-sample 出 50 query，10 / 10 / 10 / 10 严格按 query 主证据 modality 分类。

不要 hand-craft 新 query——用已有的、qrels 已经标注好的、已经被所有系统跑过的 M4query_v1 子集。这一步就是采样 + 验证。

1. 写 `scripts/build_smoke50.py`（≤80 LOC）：
   - 读 `data/03_queries/M4query_v1/qrels.jsonl` + `data/01_graphs/multimodal_elements.json`
   - 对每个 query，用其 **rank-1 gold qrel** 的 element_type（figure/table/formula/text）打标
   - 4 个 bucket 各 stratified-sample 10 个，random_state=20260505
   - 优先选 query level 多样的（l1/l2/l3）保证不全是简单 query
   - **Gate**：text bucket 必须能选满 10（M4query_v1 共 26 个 text qrel query，应该够）。如不够，退到现有数 + 标注 unbalanced
   - 输出 `data/03_queries/M4query_smoke50/queries.jsonl` + `qrels.jsonl` + `manifest.md`（采样统计 + 每 bucket query level 分布）

2. **人眼复检**：随机抽 5 / 50 query，dump query 文本 + gold passage 文本前 200 字，确认 modality 标注正确，没有 query/qrel 错乱（之前发现过 `chunk_contains_element` 0% 一致这类 bug）。

---

## Phase B — Slice 现有 ranking 到 Smoke50（30 min, no GPU）

**关键**：所有主流系统都已经在 M4query_v1 全集上跑过了——直接 slice 出 smoke50 的 50 个 query 重新算 metrics 即可，**几乎不要新计算**。

### 直接 slice 的 ranking 文件

| 系统 | ranking 文件 |
|---|---|
| dense baseline | `data/05_eval/dense_retrieval/rebuilt_20260417/ranking_v1_enriched.jsonl` |
| graph rerank ceiling | `data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_only_fixed/ranking_graph_static_plus_neighbor.jsonl` |
| BGE CE alone | `data/05_eval/cross_encoder_rerank/bge_v2m3_top500/ranking_ce_bge_v2m3.jsonl` |
| BGE rrf_dense_ce_k20 | 同目录的 fusion ranking（如无独立文件，从 scores 重算 RRF） |
| Qwen3 CE alone | `data/05_eval/cross_encoder_rerank/qwen3_reranker_4b_transformers_anchor_top500/ranking_ce_qwen3_tf.jsonl` |
| Qwen3 rrf_graph_ce_k10 | 同目录 fusion ranking |
| VL split (Qwen3-VL-2B-Embedding) | `data/05_eval/dense_retrieval/split_modality_vl_t5/ranking_*.jsonl` |
| Split 4B text | `data/05_eval/dense_retrieval/split_modality_4b/ranking_*.jsonl` |

### 输出

写 `scripts/eval_smoke50_slice.py`（≤120 LOC，直接复用 `eval_dense_retrieval.py` 的 metrics 计算逻辑）：
- 输入：上面任一 ranking + smoke50 qrels
- 输出：每系统 × 每 modality (figure/table/formula/text) × 5 metrics (R@1/5/10/100, MRR)
- 总输出 `data/05_eval/smoke50/per_system_per_modality.json` + `per_system_per_modality.md`（人读表）

---

## Phase C — VL lane 局部重算（条件触发，~10 min GPU）

仅当 Phase B 显示 **VL lane 在 figure/table 上单跑超过 dense lane**（即 split_VL_2B figure R@10 > split_4B_text figure R@10）时，才在 smoke50 上跑 graph + VL lane 的 RRF fusion。否则跳过。

复用 `scripts/cross_encoder_rerank.py` 的 RRF 逻辑，只换输入 ranking。slurm: `slurm_scripts/49_smoke50_vl_fusion.sh`（条件触发，新建）。

---

## Phase D — Per-modality Winner 分析 + 决策门 (20 min, 写)

输出 `refine-logs/SMOKE50_DECISION_20260505.md`，≤1.5 页。

### 必须包含的两张表

**T1 — Per-modality R@10 winner**：

|         | text(10) | figure(10) | table(10) | formula(10) | overall(50) |
|---|---:|---:|---:|---:|---:|
| dense | … | … | … | … | … |
| graph | … | … | … | … | … |
| BGE rrf | … | … | … | … | … |
| Qwen3 rrf | … | … | … | … | … |
| VL split | … | … | … | … | … |

**T2 — 同 query 在 M4query_v1 全集的 R@10 vs smoke50 R@10**（验证 smoke50 的代表性）：
- 如果 smoke50 上 graph R@10 接近 0.69，说明子集仍能复现 ceiling
- 如果 smoke50 上 graph R@10 显著不同（比如 0.55 或 0.80），说明 ceiling 形状高度依赖 modality 分布

### 决策规则（顺序，命中即停）

```
S1. if smoke50_graph_overall ∈ [0.66, 0.72] AND graph 在 4 个 modality 都赢:
      → ceiling 是真的、modality-uniform。
      → next: 不做 route-aware retrieval；走 Option C (HyDE)
              从 query 端缩小 modality-style mismatch

S2. elif graph 只在 ≥ 2 个 modality 赢，但其他 modality 有不同 winner:
      → ceiling 形状是 modality-mixed，存在 route-aware 收益。
      → next: 实现 query 模态分类器 + per-modality routing，用本 plan
              发现的 per-modality winner 做 routing rule。复用 split
              modality 的 VL lane 处理 figure 类 query。

S3. elif smoke50_graph_overall < 0.60:
      → 0.6913 主要是 figure-heavy artifact，graph 在均衡测试上没那么强。
      → next: 重新审视所有"graph beats X"的 claim（C1/C5/C7），加注 scope；
              开 Option B 的 late-fusion VL lane 实验（VL 在 figure 上能补
              dense 弱项）。

S4. elif text(10) bucket 表现远差于其他三个 (text R@10 < 0.30):
      → text query 是真正的硬骨头，跟之前认为「text query 简单」相反。
      → next: 失败案例分析专门针对 text query，可能需要 query 改写
              或者 text passage 的 semantic 表征改进。

S5. else (mixed signal):
      → 不同模态间的 winner 差异显著但 ceiling 仍然稳定 0.69 附近。
      → next: 开两条并行小实验：(a) route-aware retrieval pilot，
              (b) HyDE on text bucket only。$5 + 30 min GPU 预算。
```

报告必须含：
- 命中的规则（S1–S5）
- T1 + T2 数表
- 推荐的下一个实验（单线）+ 预估成本
- mentor C6 完成度更新（从 ❌ → ✅）

---

## Phase E — Wiki & Mentor Todo 收尾 (15 min)

1. 新增 `research-wiki/experiments/20260505_smoke50_balanced_audit.md`（experiment 节点，verdict = 命中规则编号）
2. 更新 `research-wiki/experiments/20260503_mentor_recording60_full_todo.md`：把 C6 从 ❌ 改为 ✅
3. 追加 `research-wiki/log.md` 一行 timestamp 完成总结
4. 更新 `research-wiki/index.md` 加入新 experiment 链接

---

## 与 M4query 初心的对照

M4query_v1 项目目标（`research-wiki/index.md` Project Direction）：
> Build a document graph over multimodal academic papers and test whether graph signals improve evidence localization, QA support, and synthesis of high-quality SFT data.

「multimodal academic papers」是 **多模态** 的，不是"figure-heavy 偏好"。如果 0.6913 是 figure-heavy artifact，那现在的 graph rerank 在 paper claim 上是过度泛化的——只在 figure/table 上有效，没在 formula 上验证，text 上几乎没数据。

C6 smoke50 是 **claim 校准工具**：让我们能写出 paper-grade 的诚实陈述（"graph helps figure/table retrieval +5pp on M4query_smoke50, no significant effect on formula/text"），而不是过度泛化的"graph improves multimodal retrieval"。

mentor 录音 60 todo 状态对照（仅与本 plan 相关项）：

| # | TODO | 之前状态 | 本 plan 完成后 |
|---|---|---|---|
| C6 | 50-query balanced smoke test | ❌ 未做 | ✅ |
| C2 | chunk dilutes signal — 已坐实，结论待师兄确认 | ⚠️ 半 | （不变，待师兄会议）|
| C3 | 分离式检索 figure/table 独立 | ✅ split modality 已跑 | （强化）—Phase D 决策可能触发 route-aware 实现 |
| C5 | 多粒度 enrich (DocResearcher) | ❌ 未做 | （C8 提示这条路风险高，本 plan 后再讨论）|

**不在本 plan 范围内的 mentor todo**（以免散焦）：
- A1/A2/A3（文档/术语层）：user 5/3 已交给别人
- B1/B2/B3（数据/匹配层）：独立工作流
- C5（多粒度 enrich）：与 C8 直接冲突，等 smoke50 数据再判
- D1（to-do list 发师兄）：用户层动作不动

---

## Out-of-scope

- 不构建新 query（所有 query 都从 M4query_v1 stratified-sample）
- 不重训练任何模型
- 不改 corpus
- 不开 HyDE / route-aware retrieval / late-fusion 的实现（Phase D 触发时另写 plan）

---

## File Manifest

| Path | Action | Owner |
|---|---|---|
| `scripts/build_smoke50.py` | New, ≤80 LOC | coder |
| `scripts/eval_smoke50_slice.py` | New, ≤120 LOC | coder |
| `slurm_scripts/49_smoke50_vl_fusion.sh` | New（条件触发） | coder |
| `data/03_queries/M4query_smoke50/{queries,qrels,manifest}.{jsonl,md}` | Phase A 输出 | coder |
| `data/05_eval/smoke50/per_system_per_modality.{json,md}` | Phase B 输出 | coder |
| `refine-logs/SMOKE50_DECISION_20260505.md` | Phase D 决策报告 | coder |
| `research-wiki/experiments/20260505_smoke50_balanced_audit.md` | New 实验节点 | coder |
| `research-wiki/experiments/20260503_mentor_recording60_full_todo.md` | C6 状态更新 ❌→✅ | coder |
| `research-wiki/log.md` | 追加完成日志 | coder |
| `research-wiki/index.md` | 加 experiment 链接 | coder |

---

## Acceptance Criteria

1. **Phase A**：smoke50 含完整 50 query × 已知 modality bucket，manifest 含每 bucket 的 query level 分布。如果 text bucket < 10，必须显式标注（不强行扩到 hand-craft）
2. **Phase B**：T1 per-system × per-modality R@10 表覆盖 dense / graph / BGE rrf / Qwen3 rrf / VL split / Split 4B text 共 6 系统
3. **Phase B**：T2 显示 graph 在 smoke50 全集 R@10 vs M4query_v1 全集 R@10（0.6913）的对比
4. **Phase D**：`SMOKE50_DECISION_20260505.md` 命中 S1–S5 中的一个，给出单线 next-experiment 推荐
5. **Phase E**：mentor C6 todo 标 ✅，wiki 三处更新

---

## Notes for the coder

- 核心 trick：**all systems already ran on M4query_v1**——你不需要任何 GPU 跑新模型，只需要 slice 已有的 ranking 文件。Phase C 的 VL fusion 是唯一可能要 GPU 的地方，且仅在条件触发时才做。
- `multimodal_elements.json` 里 `element_type` 字段就是 figure/table/formula/text/equation 的源——`text` 类型在某些版本可能叫 `paragraph`，按 mentor A1 术语统一映射成 element/text；本 plan 内部不强求改代码命名（A 类 todo 已交给别人）。
- 如果你发现 smoke50 sampling 过程中 modality 标注有 ≥ 10% 不一致（例如 query 主证据写的是 figure 但 element_type 是 table），停下来先 dump 不一致样例——这可能是 P1 类 bug（ID 命名空间错位），值得另外开 issue。
- T2 的 sanity check（smoke50 graph R@10 vs M4query_v1 graph R@10）是关键——如果偏差 > 5pp，意味着我们的 ceiling 数字本身依赖于 modality 分布，所有 paper 数据点都需要重新加 scope。
- mentor 录音 60 引用：`/projects/myyyx1/标准录音 60.mp3_20260502_134902_精转文稿.docx`；C6 原文「50 query × 4 类型：10 文本/10 图/10 表/10 公式」。
