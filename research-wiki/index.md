# Research Wiki Index

Last updated: 2026-04-19T17:30:00Z

---

## Project Direction

Build a document graph over multimodal academic papers and test whether graph signals improve evidence localization, QA support, and synthesis of high-quality SFT data.

Deliverables (priority order): ① high-quality SFT data  ② patent / trade secret  ③ paper (optional)

---

## Track A — Graph Research (图增强检索 + 虚拟节点/边)

**Goal**: Prove and extend the graph's value for retrieval. Explore virtual node/edge designs (typed cross-doc, paragraph merge, entity/keywords). Target: clean experimental story for patent + paper.

### Ideas
- [idea:001](ideas/001_explicit_graph_rerank.md): Explicit bridge-edge rerank with hub-aware prior. Stage: **active / dominant**.
- [idea:002](ideas/002_cross_doc_summary_edges.md): Cross-doc summary edges. Stage: **pivoted** — retrieval uplift closed (C6); summary nodes remain cross-doc structural scaffold.
- [idea:004](ideas/004_typed_crossdoc_element_edges.md): Typed cross-doc element edges (figure/formula/table). Stage: **validated** (R@10 = 0.6406 on v1_enriched; project high now 0.6522 via C7).

### Experiments
- [exp:20260417_dense_baseline_rebuilt](experiments/20260417_dense_baseline_rebuilt.md)
- [exp:20260417_explicit_rerank_fixed](experiments/20260417_explicit_rerank_fixed.md)
- [exp:20260418_graph_source_audit](experiments/20260418_graph_source_audit.md)
- [exp:20260418_cross_doc_summary_pending](experiments/20260418_cross_doc_summary_pending.md) — superseded by `exp:20260419_typed_crossdoc`
- [exp:20260419_deliverable_420](experiments/20260419_deliverable_420.md)
- [exp:20260419_multi_source_stacking](experiments/20260419_multi_source_stacking.md)
- [exp:20260419_cross_doc_bug_fix](experiments/20260419_cross_doc_bug_fix.md)
- [exp:20260419_typed_crossdoc](experiments/20260419_typed_crossdoc.md)
- [exp:20260419_summary_line_closed](experiments/20260419_summary_line_closed.md)
- [exp:20260419_combo_plan](experiments/20260419_combo_plan.md) — **DONE** R100–R103 (job 61463, 2026-04-19T04:30)

### Claims
- [C1](claims/C1_explicit_static_prior_improves_rerank.md): Explicit rerank with static prior improves precision. **Supported.**
- [C2](claims/C2_intra_doc_virtual_edges_dilute_precision.md): Intra-doc virtual edges dilute top-rank precision. **Supported.**
- [C3](claims/C3_cross_doc_summary_edges_can_help.md): Cross-doc section edges help R@10 after bug fix. **Partially supported** (superseded at element level by C5).
- [C4](claims/C4_graph_value_must_be_proven_on_three_axes.md): Graph value must be proven on retrieval / QA / data-synthesis axes. **Active framing.**
- [C5](claims/C5_typed_crossdoc_element_edges_lift_r10.md): Typed cross-doc element edges lift R@10 to 0.6406 on v1_enriched. **Supported** (R@10 high title transferred to C7).
- [C6](claims/C6_summary_virtual_nodes_no_retrieval_uplift.md): Summary virtual nodes give no **retrieval** uplift (R@1/MRR). **Supported.** Summary nodes remain active as cross-doc scaffold and embedding input — only the retrieval-signal usage is closed.
- [C7](claims/C7_explicit_only_static_plus_neighbor_r10_high.md): explicit_only + v1_enriched + static_plus_neighbor = R@10 **0.6522**, project high. **Supported.**

### Next runs (Track A)
| Run | Config | Cost | Status |
|-----|--------|------|--------|
| R100–R103 | combo typed_crossdoc + chunk_v2, weight sweep | CPU ~10 min | ✅ **DONE** (job 61463) |
| R104–R105 | paragraph merge n=400/500, dense baseline | pro6000 ~2h | 🔄 **RUNNING** (job 61516, gpu-pro6000-3) |
| R106 | paragraph merge n=best, graph rerank | GPU ~30 min | TODO (after R104/R105) |
| R107–R108 | bbl coverage expansion → rebuild typed_crossdoc | CPU ~30 min | NICE |
| R109 | C-Pool QA validation | BLOCKED on qrels | BLOCKED |
| R110 | clean ablation: explicit_only vs explicit+typed on v1_enriched | CPU ~5 min | **NEW** — needed to fairly resolve C5 vs C7 |

Full plan: `refine-logs/EXPERIMENT_PLAN_RETRIEVAL_2026-04-19.md`
Tracker: `refine-logs/EXPERIMENT_TRACKER.md` (R100–R109)

---

## Track B — Query Production (批量 SFT 数据生产)

**Goal**: Produce 500+ high-quality pass queries for SFT delivery. QC is lightweight rule-based + LLM judge sampling; do not over-engineer.

### Current Inventory (2026-04-19)

| File | Pass | Type |
|------|------|------|
| sweep_2026-04-12/l3_academic_pass.jsonl | 23 | L3 |
| sweep_2026-04-12/l3_academic_persona_pass.jsonl | 30 | L3 |
| sweep_2026-04-12/l3_mixed_pass.jsonl | 73 | L3 |
| sweep_2026-04-12/l3_mixed_persona_pass.jsonl | 58 | L3 |
| sweep_2026-04-12/m2_academic_pass.jsonl | 85 | M2 |
| sweep_2026-04-12/m2_mixed_persona_pass.jsonl | 100 | M2 |
| l3_enriched_v3_rerun2_pass.jsonl | 93 | L3 (old hub) |
| l3_enriched_v3_new82_rerun2_pass.jsonl | 53 | L3 (new82) |
| m2_diverse_v1_hub_kb_pass.jsonl | 29 | M2 |
| long_chain_iterative_pass.jsonl | 12 | Long-chain |
| **Total** | **556** | |

> ✅ **已超过 500 条目标**。下一步是打包 delivery 并对 LLM QC 抽样验证。

### Ideas
- [idea:003](ideas/003_method_c_query_synthesis.md): Method C long-chain synthesis. Stage: **deferred** — pass rate too low (8.5%) for bulk production; useful for data diversity.

### Next steps (Track B)
| Task | Script | Status |
|------|--------|--------|
| 打包 delivery（qrels + triplets） | `scripts/build_full_delivery.py` | TODO |
| LLM QC 抽样验证（新 sweep 结果） | `scripts/rerun_llm_qc.py` | TODO |
| L3 LLM QC 全量重跑（230 pairs，bug 已修） | `scripts/generate_multihop_l1_queries.py` | TODO |
| neg evidence 标注方案 | 待设计 | TODO |

Full tracker: `refine-logs/PRODUCTION_TRACKER.md`

---

---

## Track C — 1040-doc Corpus 建设（元素 enrich + hub pair 生产）

**Goal**: 为 1040-doc 全量 corpus 补齐 LLM element enrichment，产出可用于 query 生成的 hub pair candidates。Method C（长链 4+ hop）已废除，不再追踪。

### 1040-doc 现状快照

| 组件 | 状态 | 数量 |
|------|------|------|
| Reference graph v2 | ✅ 完成 | 1425 docs（含 1040） |
| Chunk merge v2 | ✅ 完成 | 1147 docs，42273 chunks |
| Element extraction | ✅ 完成 | 1040 docs，27209 elements |
| LLM enrich（[T]/[M]/[C]） | ⚠️ 部分 | 6707/27209（长链路径子集） |
| LLM summary nodes | ❌ 未做 | 仅旧 53 docs；暂不做 |
| Typed cross-doc edges | ❌ 未做 | 仅旧 53-doc corpus；暂不做 |

### Enrich 策略（已确认）

**不全量 enrich** 27209 elements。策略：
- `hub_candidates_v2`（2-3 hop，top 25% by hub_score）的 ~1000-1200 endpoint elements → LLM enrich
- gap227（227 docs，4301 elements）→ LLM enrich（为将来构建 1040-doc corpus 预备）
- 813-doc 集的其余 16201 elements（不在任何 hub pair 路径上）→ **暂不 enrich**

### Jobs 状态

| Job | Script | 内容 | 状态 |
|-----|--------|------|------|
| 61526 | `34_enrich_gap227.sh` | gap227 4301 elements LLM enrich | ❌ 全部 403（与 61529 并发）；0 enriched |
| 61529 | `35_hub_shortchain_enrich.sh` | hub_shortchain element enrich + pair gen | ❌ Step2 全部 403；Step1 ✅ subset 就绪 |
| 61647 | `36_combined_enrich_pairs.sh` | 两批 enrich + gap227 pairs + merge | ❌ 取消（coverage gate 等 bug） |
| **61649** | **`36b_combined_enrich_pairs.sh`** | 修复版：顺序 enrich + gate + 正确去重 + adjacency 保留 | 🔄 **RUNNING** |

### Step 1 结果（job 61529）

- 1250 candidates → 1156 pairs（92.5% mapping rate）
- By type: figure+formula 315 / formula+table 188 / figure+table 653
- By hop: 2-hop 736 / 3-hop 420；Docs covered 349
- Subset for enrich: 292 docs / 759 elements（已写入 `hub_shortchain_elements_subset.json`）

### 完成后下一步

| Task | 前置 | 备注 |
|------|------|------|
| 提交 `35b_hub_shortchain_enrich_retry.sh` | job 61526 结束 | Step 2 retry，enrich 759 elements |
| `generate_multihop_l1_queries.py` on `hub_candidates_v2_top25.json` | job 35b | 1040-doc 首批 query 生产 |
| 确认 `production_full.json` 覆盖率 | job 61526 | 合并 gap227 enriched → 1040 docs 全量 enrich |

---

## Gaps

- `gap:G1` [Track A]: Dense retrieval quality on multimodal docs. **Unresolved.** (structural: 899 formula + 842 image passages with weak embeddings)
- `gap:G2` [Track A]: Virtual edges mostly intra-doc. **Partially addressed** by C5 (typed cross-doc R@10 +1.5pp on v1_enriched). Clean ablation (R110) needed.
- `gap:G3` [Track B/C]: SFT data delivery pipeline + 1040-doc query production. **Active** — 556 pass queries (旧 53-doc 集); 1040-doc hub pair enrich 正在跑（job 61529）。
- `gap:G4` [Track A]: QA-side graph value proof. **Unresolved** (blocked on C-Pool qrels).
- `gap:G5` [Track A/B]: Repo organization complexity. **Acknowledged.**
