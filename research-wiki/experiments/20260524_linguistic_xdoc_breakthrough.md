# Linguistic Cross-Document Bridge Validation — Exploratory 2026-05-24

> ⚠️ **Status: EXPLORATORY, not yet validated.** 此页保留首轮叙事和数据，但 TL;DR 与 "0 → 2007" 的对比基于不公平 baseline（baseline 注空集而非现有 raw CLIP 边）。**Fair-baseline 复盘见本文末 "Fair-baseline addendum" 节**，请同时阅读。

## TL;DR（原始版，已被 fair baseline 推翻）

**纯图找不到任何跨文档推理链 → 语言学验证后找到 2007 条。成本 $0.78/100 边。**

→ 复盘后实际结论：chain 数量受 BFS cap 主导无法区分变体；语言学过滤的真实效果是把 raw CLIP 同源 100 条的 doc-pair 覆盖从 270 砍到 78（−71%），但是否更高质量未判定。详见末尾 addendum。

## 问题

M4 的跨文档（Multi-document）维度一直是瓶颈：
- CLIP-based cross-doc edges: 3238 条，87% `caption_sim=0`，仅 5% 有实质语义
- Entity-bridge: 83 对/39 papers，仅 2 条 strong chain
- 纯图（intra-doc candidate pairs）: **0 条跨文档链**

## 方法

借鉴两套语言学理论：

1. **Genette 的 Transtextuality 理论 (1982)** — 五种文本间关系：
   - `direct_quotation`（直接引用）
   - `transformation`（B 在 A 基础上推导/扩展）
   - `commentary`（B 评论/benchmark A）
   - `paratextual`（共享问题域）
   - `architextual`（同方法家族）

2. **Rhetorical Structure Theory (RST)** (Mann & Thompson) — 七种修辞关系：
   - Elaboration, Cause-Effect, Background, Evidence, Contrast, Summary, Joint

结合 McManus & Lau (2024, arXiv:2410.15145) 和 Chen et al. (2025, IP&M) 的计算方法。

### Pipeline

```
Cross-doc edge (section-level)
  → Phase 1: Decontextualization (LLM 去上下文化)
  → Phase 2: RST relation classification (Genette type + RST relation)
  → Phase 3: Asymmetric verification (A→B ≠ B→A)
  → Phase 4: Genette quality tier (gold/strong/weak/topical/noise)
  → Inject strong+weak as hard cross-doc edges into Document Graph
  → Chain finding on enhanced graph
```

## 实验结果

### 100 条边的语言学验证 ($0.78)

| Genette 类型 | 数量 | % |
|-------------|------|-----|
| transformation | 33 | 33% |
| architextual | 27 | 27% |
| paratextual | 23 | 23% |
| commentary | 10 | 10% |
| direct_quotation | 5 | 5% |
| unknown | 2 | 2% |

| 质量层级 | 数量 | % |
|---------|------|-----|
| gold | 3 | 3% |
| strong | 25 | 25% |
| weak | 18 | 18% |
| topical | 50 | 50% |
| noise | 4 | 4% |

**Usable (gold+strong+weak): 46/100 (46%)**
**Causal-type edges: 25/100 (25%)** (Genette type ∈ {transformation, commentary}; these are *section-level edges* with causal relation, not multi-hop chains)

### Graph + Linguistics Fusion

| 指标 | Graph Only | Graph + Linguistics |
|------|-----------|-------------------|
| Cross-doc chains | **0** | **2007** |
| Linguistic edges injected | 0 | 43 |
| Element-level cross-doc edges | 0 | 6056 |
| Modality coverage | — | table+fig(852), fig+tab(517), fig+formula(191), tab+formula(189), formula+fig(155) |

## 复现步骤

```bash
cd /projects/_hdd/myyyx1/data-process-test

# Step 1: 语言学验证 (需要 LLM API)
set -a && source .env && set +a
python3 experiments/build_linguistic_xdoc_bridges.py \
  --cross-doc-edges data/01_graphs/cross_doc_sim_edges.json \
  --elements data/02_enriched/multimodal_elements_enriched.json \
  --limit 100 --delay 0.3

# 输出: data/05_eval/linguistic_xdoc_<timestamp>/
#   - linguistic_validated_edges.jsonl  (每条边含 genette_type, rst_relation, quality_tier)
#   - summary.json

# Step 2: Graph + Linguistics Fusion
python3 experiments/build_graph_linguistic_fusion.py \
  --linguistic-edges data/05_eval/linguistic_xdoc_<timestamp>/linguistic_validated_edges.jsonl \
  --elements data/02_enriched/multimodal_elements_enriched.json \
  --hub-candidates data/02_enriched/hub_candidates_enriched_v4_intra_doc.json \
  --min-hops 2 --max-hops 4 --max-chains 2000

# 输出: data/05_eval/graph_linguistic_fusion_<timestamp>/
#   - enhanced_chains.json  (2007 条跨文档链)
#   - baseline_chains.json  (0 条，纯图)
#   - fusion_summary.json
```

## 全量扩展预估

- 当前已跑: 100/3238 edges, $0.78
- 全量成本: ~$25
- 预期 usable edges: ~1500
- 预期跨文档链: ~60K

## 关键文件

| 文件 | 说明 |
|------|------|
| `experiments/build_linguistic_xdoc_bridges.py` | 语言学验证主脚本 (Genette + RST + asymmetry) |
| `experiments/build_graph_linguistic_fusion.py` | Graph + Linguistics 融合 + 链发现 |
| `data/05_eval/linguistic_xdoc_20260524T124648Z/` | 100 边验证结果 |
| `data/05_eval/graph_linguistic_fusion_20260524T131330Z/` | Fusion 结果 (2007 chains) |

## 同时新增的其他 M4 增强

本次 session 还实现了：

1. **P0.1 HopWeaver 规则 QC** (`src/qc/checks.py` + `src/qc/pipelines.py`)
   - `check_fact_distribution()`: 每条 hop 必须用不同文档
   - `check_no_shortcut()`: 不能有单文档桥接不相邻 hop
   - `check_causal_chain_direction()`: premise→intermediate→conclusion

2. **P0.2 BMGQ NLI 关系分类** (`scripts/type_graph_relations.py`)
   - 在图路径上做 observation/attribution/explanation/verification/prediction 分类

3. **P1 Entity Skeleton 跨文档 Reranker** (`scripts/build_entity_skeleton_xdoc.py`)
   - 零成本规则从元素文本提取实体，entity overlap 做跨文档精排

4. **P2 DocTalk Discourse Planning** (`experiments/build_chain_to_session_v2.py`)
   - chain_to_session 加入 discourse planning 阶段，提升 turn_dependency

## 参考文献

- Genette, G. (1982). *Palimpsests: Literature in the Second Degree*. (1997 English translation, U Nebraska Press)
- McManus, S. & Lau, P.K. (2024). "Mining Asymmetric Intertextuality." arXiv:2410.15145.
- Chen, X., Li, P., & Zhu, Q. (2025). "Improving Cross-Document Event Coreference Resolution by Discourse Coherence and Structure." Information Processing & Management, 62(4).
- Gao, Q. et al. (2024). "Enhancing Cross-Document Event Coreference Resolution by Discourse Structure and Semantic Information." LREC-COLING 2024.
- Mann, W. & Thompson, S. (1988). "Rhetorical Structure Theory: Toward a functional theory of text organization." Text, 8(3).

---

## Fair-baseline addendum（2026-05-24 当天复盘）

### 原版三个方法论硬伤

1. **Baseline 是空集，不是 raw CLIP**
   原脚本 `experiments/build_graph_linguistic_fusion.py:283-290` 把 baseline 设为 `build_enhanced_graph([], …)`——空 list，根本没有放入 corpus 已存在的 raw CLIP cross-doc edges。所以 "0 → 2007" 等价于"加边 vs 不加边"，不是"语言学过滤 vs 不过滤"。

2. **6056 element-level edges 是 src_doc × tgt_doc 笛卡尔积**
   `resolve_edge_to_elements` 把 section-level edge 投到 element 级时连接两端文档所有 figure/table/formula；同一条 section edge 衍生的 element pairs 共享 `confidence=0.85, asymmetry_score=0.7`。去重后实际只有 2615 条 unique element pairs（不是 6056）。验证只到 section 级。

3. **2007 chains 是 BFS 拓扑可达性，不是 chain-level 质量**
   现有 chunk-bridge judge（300/300，60% usable / 8% strong）从未对这 2007 条 chain 跑过 head-to-head。

### Fair-baseline 数据（chain cap = 5000, seed = 42）

脚本：`experiments/build_graph_linguistic_fusion_fair.py`

| 变体 | 输入 section edges | 去重 element pairs | chains（BFS, cap=5000） | 唯一 endpoint pairs | 唯一 doc pairs |
|------|-------------------|-------------------|-----------------------|--------------------|--------------|
| **A** 空 cross-doc baseline | 0 | 0 | 0 | 0 | 0 |
| **B** raw CLIP 同 100 条不过滤 | 100 | 8332 | 5092 ⓒ | 5012 | **270** |
| **C** 语言学 strong+weak | 43 | 2615 | 5081 ⓒ | 4855 | **78** |
| **D** raw CLIP 全量 2467 | 2467 | 130663 | 5016 ⓒ | 4981 | **423** |

ⓒ = 顶到 chain cap。

### Fair-baseline 真实发现

- **Chain count 是无效指标**：B/C/D 全部顶 cap，差异只反映 BFS 探索顺序，不反映质量。
- **Doc-pair 覆盖是真实可对比维度**：raw CLIP 同 100 条覆盖 270 doc-pair；语言学过滤后剩 78 doc-pair（−71%）。
- **语言学过滤 = 高聚焦低覆盖**：每个保留的 doc-pair 上 chain 密度更高（C: 62 endpoints/doc-pair vs B: 18）。
- **真问题未答**：那 78 doc-pair 是不是比被砍掉的 192 更高质量？需要 chain-level judge 才能判定。

### 真正可投稿的下一步

1. **Chain-level head-to-head**：从 chains_B / chains_C 各抽 100 条用 chunk-bridge judge 评分，看 usable rate；再对比"语言学过滤 vs raw CLIP"是否真的提升 chain 质量。
2. **新旧 judge 一致性**：chunk-bridge judge（60% usable / 8% strong）vs Genette+RST judge（46% usable / 25% strong）跑同一批 100 边，看分类是否一致。
3. **去除 cartesian 投影**：把 element-level grounding 真的做到位（caption + decontext 文本匹配，而非 src_doc × tgt_doc 笛卡尔积）。

### Provenance

- 原 fusion 数据：`data/05_eval/graph_linguistic_fusion_20260524T131330Z/`
- Fair 复盘数据：`data/05_eval/graph_linguistic_fusion_fair_20260524T134916Z/`
- Linguistic validation 100 边：`data/05_eval/linguistic_xdoc_20260524T124648Z/`

---

## Chain-level head-to-head（addendum #2，2026-05-24 晚）

### Setup

脚本：`experiments/judge_chain_quality_headtohead.py`
做 LLM-as-judge 的 head-to-head。两组都从 B（raw matched 100）chains 池中抽样，唯一差别是**所跨 doc-pair 是否通过语言学过滤**：

- **VALIDATED** — 跨过的 doc-pair 在语言学 gold/strong/weak 集（15 doc-pairs，B 池中 2829 条 chains）
- **REJECTED**  — 跨过的 doc-pair 在语言学 topical/noise 集（30 doc-pairs，B 池中 2263 条 chains）

按 hop count 分层各抽 40（实得 39 / 39），judge 给 usable / weak / noise 三档。模型 gpt-5.4，$0.28。

### 结果

| 组 | usable | weak | noise | bridge_quality=none |
|----|--------|------|-------|-------------------|
| VALIDATED | **0.0%** (0/39) | 10.3% (4/39) | 89.7% (35/39) | 89.7% |
| REJECTED  | **0.0%** (0/39) | 17.9% (7/39) | 82.1% (32/39) | 76.9% |

**Usable delta（VALIDATED − REJECTED）= 0.0**。
弱关系率上 REJECTED 反而略高 7.6 个百分点，但 N=39 下 95% CI 重叠，不显著。

### 结论：当前用法下，语言学过滤不能转化为 chain 级别的质量改善

判官的 reasoning 是 discriminative 的（举例：noise 判定写"step 1 是 fairness post-processing 公式，step 2 是 health dataset 的图，没有具体跨文档桥接"——具体且合理）。所以 0% usable 不是判官懒，是 **cartesian 投影**这一步真的把语言学验证的有用信号稀释掉了。

### 病灶定位（关键洞察）

- Section-level Genette+RST 判断本身是 discriminative 的（46% usable 边、25% causal chains）；
- 但从 section-level edge → element-level pair 用 src_doc × tgt_doc 笛卡尔积投影，把"两个 section 在概念上相关"扩散为"两篇 paper 里所有 figure/table/formula 两两相连"；
- 这一步把语言学过滤选中的稀少高质量信号摊薄到大量噪声中。

### 真正应做的下一步（按可行性排序）

1. **Element-level 语言学验证**：放弃 section→element 笛卡尔，直接对 element-element pair（caption + enriched_content 文本对）做 Genette+RST 判定。已有 `multimodal_elements_enriched.json`，每篇 ~16 元素，10⁴ 量级元素对里挑高 CLIP 相似度前 1-2k 做判定，预算 ~$5-8。
2. **Bridge edge constraint at chain time**：BFS 时强制要求跨文档跳跃的两端 element 之间必须有 element-level（不只是 section-level）证据，例如 caption token 重叠 ≥ 阈值，或者 CLIP element-element 相似度 ≥ 阈值。
3. **Drop linguistic xdoc 这条线**：如果元素级证据稀薄到根本不存在，那 cross-doc 多跳链对当前 corpus（85 篇 fairness 论文）就不是天然丰富，需要换更大的 corpus 或换问题（intra-doc + entity-bridge）。

### Provenance

- Judge 脚本：`experiments/judge_chain_quality_headtohead.py`
- Verdicts: `data/05_eval/chain_judge_h2h_20260524T143410Z/verdicts.jsonl`
- Summary: `data/05_eval/chain_judge_h2h_20260524T143410Z/summary.json`
- Token log: $0.28 / 51,177 in + 8,539 out
