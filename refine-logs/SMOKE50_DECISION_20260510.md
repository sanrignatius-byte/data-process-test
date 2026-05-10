# Smoke50 Decision Report — Mentor 录音 60 C6 关闭

**Date**: 2026-05-10
**Plan**: [SMOKE50_BALANCED_PLAN_20260505.md](SMOKE50_BALANCED_PLAN_20260505.md)
**Verdict**: **S2 命中** — ceiling 真但 modality-mixed；graph 在 figure/table 强、formula 与 dense 持平
**Mentor C6 status**: ❌ → ✅

---

## 关键现实修正：M4query_v1 没有 text qrels

Plan 5/5 估计 26 个 text query 是从 BGE pilot top-1 反推的——那是 **reranker 错答（把 text passage 当成 gold）**，不是 ground truth。
M4query_v1 qrel modality 实际：figure 218 / formula 138 / table 117 / **text 0**。

**调整**：smoke50 改为 17 figure / 17 formula / 16 table = 50（按 rank-1 qrel 模态分桶）。
Mentor "10 text" 在 v1 数据上不可执行，需另立 query 集（不是本 plan 范围）。

---

## T1 — Per-modality R@10 winner（核心表）

| System | figure (39 qrels) | formula (25 qrels) | table (36 qrels) | overall (100 qrels) |
|---|---:|---:|---:|---:|
| dense_v1_enriched | 0.7179 | 0.5600 | 0.6111 | 0.6400 |
| **graph_static_plus_neighbor** | **0.8205** ★ | 0.5600 | **0.6944** ★ | **0.7100** ★ |
| graph_static_prior | 0.7692 | 0.5600 | 0.5833 | 0.6500 |
| bge_ce | 0.5128 | 0.2400 | 0.3611 | 0.3900 |
| qwen3_ce | 0.6667 | **0.5600** ★ tie | 0.5278 | 0.5900 |
| split_4b_text | 0.5897 | 0.4000 | 0.4722 | 0.5000 |
| split_vl2b_t5 | 0.3333 | 0.4000 | 0.0278 | 0.2400 |

**Per-modality winner**:
- **figure**: graph (0.82, +10.3pp over dense)
- **table**: graph (0.69, +8.3pp over dense)
- **formula**: **3-way tie** at 0.56 — graph / graph_static_prior / dense / qwen3_ce 都打到这个数字

**关键**：graph 在 formula 上**没有任何增益**——它"赢"的是因为 dense 也是 0.56。这意味着 graph rerank 信号对 formula 检索不传播。

---

## T2 — Sample 代表性 sanity check

| System | M4query_v1 全集 R@10 | smoke50 R@10 | Δ |
|---|---:|---:|---:|
| dense | 0.6195 | 0.6400 | +2.05pp |
| **graph (ceiling)** | **0.6913** | **0.7100** | **+1.87pp** |
| BGE CE | 0.4482 | 0.3900 | −5.82pp |
| Qwen3 CE | 0.5613 | 0.5900 | +2.87pp |

graph 偏差 +1.87pp，**远低于 5pp 阈值** → smoke50 是 M4query_v1 的代表性子集。
即"0.6913 是 figure-heavy artifact"假说被否定（S3 不命中）。

---

## 决策规则命中：S2

```
S1 (ceiling 真且 modality-uniform; graph 在 4 modality 都赢)
  ❌ formula 上 graph 与 dense / qwen3_ce 三方持平 — 没赢

S2 (graph 在 ≥2 modality 赢，其他 modality 有不同 winner)
  ✅ graph 赢 figure + table；formula 三方 tie；text 不可测

S3 (smoke50_graph_overall < 0.60: ceiling 是 figure-heavy artifact)
  ❌ smoke50 graph = 0.71 与 full graph = 0.69 一致

S4 (text < 0.30; text query 是硬骨头)
  ⚠️  Not applicable — M4query_v1 没有 text qrel

S5 (mixed signal)
  ❌ S2 已命中
```

---

## 实质含义

### 1. ceiling 0.6913 是真的，不是 artifact

smoke50 上 graph 0.71 与 M4query_v1 全集 0.69 偏差 < 2pp。即使刻意把 modality 分布拉到接近均匀（figure 39% / formula 25% / table 36%，vs full set figure 46% / formula 29% / table 25%），graph 仍然守住 ~0.7 的水平。三轮 reranker 失败的真正原因不是 modality bias，而是 graph 已经把所有低垂果子摘完了。

### 2. graph 增益是 figure + table 选择性

| modality | graph - dense (pp) |
|---|---:|
| figure | +10.3 |
| table | +8.3 |
| formula | **0.0** |

5/3 failure profiling 已经指出 missed qrels 49.6% 是 formula——现在 smoke50 给出**机制证据**：graph rerank 在 formula 上零增益。所有 paper claim（C1/C5/C7）需加注 modality scope：「graph improves figure/table retrieval; no significant effect on formula」。

### 3. formula 是真正瓶颈

3 个 reranker 在 formula 上各有偏差：
- dense: 0.56
- graph: 0.56（graph 信号不传到 formula 节点）
- qwen3_ce: 0.56（formula-bias 但 figure/table 上拉低）
- bge_ce: 0.24（text-bias 全模态崩）

四家 0.56 像是 dense embedding 在 formula 上的天然能力上限。要破这个上限，需要 **formula-specific encoder**（math BERT / OPT-LaTeX / Qwen3-Math 等），不是 reranker，也不是 graph。

### 4. VL split 全模态弱于 4B text

split_vl2b_t5 figure 0.33 vs split_4b_text figure 0.59 vs graph figure 0.82。即使 figure 是 VL 的目标模态，VL 仍然 50pp 弱于 graph。**Phase C VL fusion 不触发**（plan 条件：VL figure > 4B text figure；实际 0.33 < 0.59）。

---

## 推荐下一步（单线，按 EV 排序）

1. **F-formula：math-aware encoder for formula passages** — 在 anchor corpus 上仅替换 formula passages 的 embedding（用 Qwen3-Math / Mistral-Math 编码 LaTeX 源码），其他模态保持 Qwen3-Embedding-4B。预算：1h GPU + $0 LLM。Success bar: formula R@10 > 0.65。
2. **claim 加注 modality scope** — 更新 C1/C5/C7 paper claim：「graph rerank improves figure/table R@10 by +8-10pp; no significant effect on formula retrieval. Demonstrated on M4query_v1 (figure-heavy) and balanced M4query_smoke50.」零成本，纯 wiki 改动。
3. **构建 text-evidence 子集** — M4query_v1 完全没有 text query；mentor C6 想测 text 模态需要新 query 集（从 paper 生成 text-only evidence query）。预算：~$5 LLM + ~30 min GPU。**优先级低**——除非 paper 必须 cover text，否则放着。

---

## 副产品 / 二阶发现

1. **BGE 失败的根因更新**：5/3 我们以为 BGE text-bias 是问题，现在看 smoke50 figure R@10=0.51 仍远低于 graph 0.82。即便 BGE 完美没 text-bias，它在 figure/table 上也输给 graph rerank。Reranker 路线对此 corpus 整体走不通。
2. **Qwen3-Reranker formula-bias 不是病而是优势**：Qwen3 formula 0.56 等于 graph，但 figure/table 弱。如果搞 ensemble (graph + qwen3 fusion)，预期 formula 没增益（已经 saturate），figure/table 会被 qwen3 拖低。**ensemble 路线证伪**。
3. **dense 的 figure ceiling 是 0.72**：dense 单跑 figure R@10=0.7179；graph 拉到 0.82。意味着 graph 给 figure 模态贡献了 ~14% 的额外召回。这是 graph 价值最强的实证点。
4. **graph_static_prior 比 graph_static_plus_neighbor 在 table 上弱 11pp**（0.58 vs 0.69），说明 neighbor propagation 在 table 上特别有用——可能因为 table caption 跨页 neighbor 关系密集。

---

## File manifest（本 plan 产物）

| Path | 用途 |
|---|---|
| `scripts/build_smoke50.py` | 建 smoke50 |
| `scripts/eval_smoke50_slice.py` | slice ranking + 算 metrics |
| `data/03_queries/M4query_smoke50/queries.jsonl` | 50 query |
| `data/03_queries/M4query_smoke50/qrels.jsonl` | 100 qrel |
| `data/03_queries/M4query_smoke50/manifest.md` | 采样统计 |
| `data/05_eval/smoke50/per_system_per_modality.json` | 全量 metrics JSON |
| `data/05_eval/smoke50/per_system_per_modality.md` | 人读表 |
| `refine-logs/SMOKE50_DECISION_20260510.md` | **本报告** |
| `research-wiki/experiments/20260505_smoke50_balanced_audit.md` | 实验节点（待更新） |

---

## Mentor C6 todo 收尾

| 字段 | 之前 | 之后 |
|---|---|---|
| 状态 | ❌ | ✅ |
| 完成方式 | plan 5/5 写、5/10 执行 | smoke50 实跑 + 决策报告 |
| 决策 | — | S2: route-aware retrieval is the implication, but formula is the real bottleneck |
| 下一步 | — | F-formula (math-aware encoder)；claim scope 加注 |

录音原话「50 query × 4 类型」打折落地为「50 query × 3 类型 (figure/formula/table)」，差额由"M4query_v1 没有 text qrel"承担，这个发现本身值得汇报给 mentor。
