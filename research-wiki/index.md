# Research Wiki Index

Last updated: 2026-05-10T14:54:00Z

## Project Direction

Build a document graph over multimodal academic papers and test whether graph signals improve evidence localization, QA support, and synthesis of high-quality SFT data.

Latest requirement anchors:
- `标准录音 60.mp3_20260502_134902_精转文稿.docx` (2026-05-02): 术语统一（paragraph=text element），chunk 公平性先验证再讨论，分离式检索优先，虚拟边暂缓。完整 18 条 todo 提取见 [exp:20260503_mentor_recording60_full_todo](experiments/20260503_mentor_recording60_full_todo.md)，5/10 BCD 分阶段执行后整体完成度 ~71%（+39pp from 32%）。详见 [refine-logs/BCD_PHASED_PLAN_20260510.md](../refine-logs/BCD_PHASED_PLAN_20260510.md)。
- `4.16.md` top-level plan: deliver SFT data first, patent or trade secret second, paper optional.
- `标准录音 57.mp3_20260417_190739_精转文稿.docx` later section: prioritize retrieval uplift, QA uplift, and data synthesis value; keep `summary` as the only immediate virtual-node priority; simplify QC; do not widen the story around virtual edges yet.

Latest discussion log:
- 2026-05-03: 录音 60 完整 todo 提取 + per-query chunk→element recall 分析 + Qwen3-VL Embedding 根因复查 + transformers≥5.2 env clone → `research-wiki/log.md`
- 2026-05-02: Mentor 录音 60 对齐 + chunk 公平性分析 + split modality 实验 → `research-wiki/log.md`
- 2026-04-21: Chunk 革新与评估对齐讨论（含录音要点）→ `research-wiki/log.md`
- 2026-04-22: 基础设施解耦 Round 1（综合 Plan A + Plan B）→ `research-wiki/experiments/20260422_decoupling_round1.md`、`docs/DECOUPLING_PLAN_2026-04-22.md`

## Ideas

- [idea:001](/projects/_hdd/myyyx1/data-process-test/research-wiki/ideas/001_explicit_graph_rerank.md): Explicit bridge-edge rerank with hub-aware prior. Stage: active.
- [idea:002](/projects/_hdd/myyyx1/data-process-test/research-wiki/ideas/002_cross_doc_summary_edges.md): Cross-document summary similarity edges with citation boost. Stage: active validation.
- [idea:003](/projects/_hdd/myyyx1/data-process-test/research-wiki/ideas/003_method_c_query_synthesis.md): Long-chain query synthesis via compressed bridge chains. Stage: deferred to supporting role.

## Experiments

- [exp:20260417_dense_baseline_rebuilt](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260417_dense_baseline_rebuilt.md)
- [exp:20260417_explicit_rerank_fixed](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260417_explicit_rerank_fixed.md)
- [exp:20260418_graph_source_audit](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260418_graph_source_audit.md)
- [exp:20260418_cross_doc_summary_pending](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260418_cross_doc_summary_pending.md)
- [exp:20260421_chunk_as_retrieval_unit](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260421_chunk_as_retrieval_unit.md) — chunk 作检索单元 + element 注入 + qrels 重映射。当前 graph-only fair、partial-overlay、BM25 都已完成；fair enriched 仍被 API auth / budget 阻塞。
- [exp:20260421_trial57_fairness_repair](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260421_trial57_fairness_repair.md) — old-trial `57 gold docs` 与 `1040 production` 已分离，partial enrich fairness 问题已量化并加 guard，partial-overlay exploratory 结果已补齐。
- [exp:20260421_api_logging_compliance](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260421_api_logging_compliance.md) — `local_api_logger -> api_logs` 为所有公司代理调用的合规铁律。
- [exp:20260421_crossdoc_gold57_validation](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260421_crossdoc_gold57_validation.md) — gold-57 BBL+embedding 跨文档边生成+验证+chunk→element 投影。跨法机制正确（BBL 85 对覆盖 72 对），但作为 rerank 信号对 M4query_v1 净负；explicit_only 仍是本地 SOTA (neighbor `R@10=0.6892`)。
- [exp:20260422_decoupling_round1](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260422_decoupling_round1.md) — 综合 Plan A / Plan B 的基础设施解耦 Round 1：`src/` 库 / `experiments/` 一次性 / `scripts/` 薄壳三层分层；拆 `src/api/llm.py` 和 `src/retrieval/{bm25,metrics}.py`；新增 `src/io/` + `src/cli/`；删 `endpoint_anchor.py` 575 行死代码；写入 SOT + R1–R6 硬规则；`pytest` 107 全绿，零行为变化。
- [exp:20260502_chunk_element_coverage](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260502_chunk_element_coverage.md) — chunk→element 映射统计：964 chunks 平均覆盖 1.94 elements，75% query 双证据分散在不同 chunk。
- [exp:20260502_split_modality](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260502_split_modality.md) — 分离式检索实验（0.6B + 4B）：text-only split 跑通但弱于 unified 4B，`split_4B_text R@10=0.4767` vs `v1_enriched_4B R@10=0.6195`。
- [exp:20260502_split_modality_vl_failed](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260502_split_modality_vl_failed.md) — Qwen3-VL-Embedding-2B (Job 66114) R@10=0.0021。**5/2 误诊为 checkpoint 缺权重；2026-05-03 复查发现真正根因是 transformers 4.57 vs 模型要 5.2+ 的版本不匹配（Qwen 官方 PR #19 确认）**。Checkpoint 完整可用，已 superseded。
- [exp:20260503_chunk_query_element_recall](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260503_chunk_query_element_recall.md) — per-query 视角答 mentor 录音 60："recall 出来的 chunk 里平均含几个 element"。结论：n500 partial-overlay 最优 lane elem R@10=0.530（vs chunk R@10=0.678 有 15pp 鸿沟），K=1 zero rate 71%。坐实 chunk 在双证据 query 上稀释信号。同时发现 `paragraph_chunks_n400_v2.json` 的 `chunk_contains_element` 边和 eval-time qrels chunk_id **0% 一致**（P1 bug）。
- [exp:20260503_split_modality_vl_t5_rerun](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260503_split_modality_vl_t5_rerun.md) — 用 transformers 5 overlay 重跑 Job 66114。环境修复成功（625/625 weights clean load，无 newly initialized warning），`split_VL_2B_t5 R@10=0.2579`，证明 checkpoint 可用但当前 VL split 仍弱于 `split_4B_text R@10=0.4767`；figure lane 有提升（0.411→0.540），formula/table 拖累整体。
- [exp:20260503_mentor_recording60_full_todo](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260503_mentor_recording60_full_todo.md) — 录音 60 完整 18 条 todo 抽取 + 完成度核查（整体 ~26%）。A1/A2/A3（文档术语类）2026-05-03 user 决定交给别人处理。
- [exp:20260503_hybrid_rank_fusion](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260503_hybrid_rank_fusion.md) — **Planned (revised)**: Modality routing ablation。R120-R127。
- [exp:20260503_vl_enrich_comparison](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260503_vl_enrich_comparison.md) — **Planned**: VL Embedding controlled comparison — enrich_only text (4B) vs raw image (VL-2B)。R130-R133。
- [exp:20260503_cross_doc_citation](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260503_cross_doc_citation.md) — **Planned**: Cross-document element-level citation query pipeline。R140-R143。
- [exp:20260503_failure_profiling](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260503_failure_profiling.md) — **Completed (verdict R2)**: 121 partial+zero query rank-of-missed profiling。69% 漏掉的 qrel 在 rank (100, 500]，formula 中 form_high=0.016 否决 R1。**决策：cross-encoder rerank on dense top-500**（BGE-reranker-v2-m3 / Qwen3-Reranker-4B），目标 R@10 ≥ 0.72 vs ceiling 0.6913。详见 [refine-logs/CEILING_DECISION_20260503.md](../refine-logs/CEILING_DECISION_20260503.md)。
- [exp:20260503_ce_rerank_bge](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260503_ce_rerank_bge.md) — **Completed (NEGATIVE on R@10)**: BGE-reranker-v2-m3 rerank dense top-500. R@10 跌到 **0.4482 (−17pp)**，MRR 跌 24pp。原因：reranker 严重 text-bias，top-1 modality {text 348, figure 87, table 29, formula 9} vs dense {figure 265, table 115, formula 67, text 26}。RRF(dense, CE, k=20) 救回 R@100=0.8869 (+2.3pp) 但 R@10 仍 0.6258，未破 0.6913 ceiling。R2 路径在 BGE-reranker-v2-m3 上证伪。后续：F1 = Qwen3-Reranker-4B（同家族），F2 = 修 corpus 端 figure/formula 退化 bug，F3 = HyDE。
- [exp:20260503_corpus_enrich_fix](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260503_corpus_enrich_fix.md) — **Completed (verdict D5, NOT promoted)**: 修了两个独立 bug — `load_enriched_index` 只认 nested `pair.element_a.element_id` 错过 MODORA `documents.elements` 格式 (1285 enrichments)；`build_element_text` 用 OR 优先级在 enrich 命中时跳过 graph context。两个 corpus 变体：`fix_v1` (replace) dense R@10=0.5106 (−10.9pp)，`fix_v2` (additive: visual + paper context, mean figure len 683) dense R@10=**0.5888 (−3.1pp)** / R@100=0.8436 (−2.0pp) / graph_static_plus_neighbor R@10=**0.6860 (−0.5pp)**。Phase D 因 R@100 Δ<+2pp 不触发。机制：MODORA 视觉描述 domain-detached，M4query_v1 是 paper-domain 文本式 query，方向与 BGE-CE text-bias 同。`DEFAULT_ENRICHED_FILES` 已回滚；潜在 bug 修复保留为 no-op。详见 [refine-logs/CORPUS_FIX_DECISION_20260503.md](../refine-logs/CORPUS_FIX_DECISION_20260503.md)。
- [exp:20260503_qwen3_rerank_fusion](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260503_qwen3_rerank_fusion.md) — **Completed (NEGATIVE)**: Qwen3-Reranker-4B + fusion 全消融（jobs 66395/66401/66405/66408，~75 min A6000）。无 fusion 突破 graph 0.6913；ce 单跑 0.5613，rrf_graph_ce_k10 0.6702 (−2.1pp)。**关键发现**：Qwen3 跟 BGE 模态偏置完全相反——BGE 偏 text (top-1 348/473)，Qwen3 偏 formula (top-1 248/473)。BGE pilot 给的 R@100 +2.3pp 唯一正向信号 Qwen3 没复现 (Qwen3 rrf_dense_ce R@100 −0.4pp)。Option A（换 reranker 家族）证伪。
- [exp:20260505_smoke50_balanced_audit](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260505_smoke50_balanced_audit.md) — **Completed 5/10 (S2 verdict)**: smoke50 graph R@10=0.7100 vs full 0.6913 (+1.87pp, < 5pp threshold) → ceiling 真，不是 figure-heavy artifact。Per-modality: figure +10.3pp / table +8.3pp / **formula +0pp**。Mentor 提的 "10 text" 在 M4query_v1 不可行（v1 无 text qrel）。Recommendation: F-formula (math-aware encoder for formula passages)。
- [exp:20260510_b1_phase2_lineno](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260510_b1_phase2_lineno.md) — **Completed 5/10**: B1 Phase 2 重建 chunk-element 边用 LaTeX 行号 + 6 graph rerank 配置消融。Topology 确实变了（kept=20/added=1130/removed=529），但 explicit-only ceiling 0.7100 不变，formula R@10 6 配置全 ≤ 0.5600。**新 [claim:C11](/projects/_hdd/myyyx1/data-process-test/research-wiki/claims/C11_formula_ceiling_is_dense_encoder_bound.md)**: formula 瓶颈是 dense encoder bound, 不是 graph topology bound。
- [exp:20260510_f_formula_caption](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260510_f_formula_caption.md) — **Completed 5/10 (HD verdict)**: F-formula caption injection 把 mineru `context_before` 注入 formula passages，同 Qwen3-Embedding-4B encoder 重编码。Dense R@10 0.6195→0.5825 (−3.7pp)，graph R@10 0.6913→0.6691 (−2.2pp)，formula bucket 跌 16pp。8 configs 0 突破 0.5600（3 regressed）。**C11 强化**：text augmentation strictly cannot rescue LaTeX。F-formula Phase 2 必须真换 encoder。
- [exp:20260510_f_formula_math_norm](/projects/_hdd/myyyx1/data-process-test/research-wiki/experiments/20260510_f_formula_math_norm.md) — **Completed 5/10 (HD FAIL)**: LaTeX surface normalization 对 formula R@10 0.5600 完全无提升；C11 升级为 10 configs，全未突破公式桶天花板。下一步只剩 math-aware encoder。

## Claims

- [claim:C1](/projects/_hdd/myyyx1/data-process-test/research-wiki/claims/C1_explicit_static_prior_improves_rerank.md): Supported.
- [claim:C2](/projects/_hdd/myyyx1/data-process-test/research-wiki/claims/C2_intra_doc_virtual_edges_dilute_precision.md): Supported.
- [claim:C3](/projects/_hdd/myyyx1/data-process-test/research-wiki/claims/C3_cross_doc_summary_edges_can_help.md): Reported, pending validation.
- [claim:C4](/projects/_hdd/myyyx1/data-process-test/research-wiki/claims/C4_graph_value_must_be_proven_on_three_axes.md): Active framing claim.
- [claim:C5](/projects/_hdd/myyyx1/data-process-test/research-wiki/claims/C5_typed_crossdoc_element_edges_lift_r10.md): Supported (scope: v1_enriched only).
- [claim:C6](/projects/_hdd/myyyx1/data-process-test/research-wiki/claims/C6_summary_virtual_nodes_no_retrieval_uplift.md): Supported (line closed).
- [claim:C7](/projects/_hdd/myyyx1/data-process-test/research-wiki/claims/C7_explicit_only_static_plus_neighbor_r10_high.md): Supported.
- [claim:C8](/projects/_hdd/myyyx1/data-process-test/research-wiki/claims/C8_modora_visual_enrichment_net_negative.md): **Supported (2026-05-03)** — MODORA-style visual enrichment is net-negative on text-style scientific QA retrieval; corpus replacement and additive both regress, while graph rerank only drops −0.5pp (corpus quality is not the ceiling).
- [claim:C9](/projects/_hdd/myyyx1/data-process-test/research-wiki/claims/C9_chunk_dilutes_double_evidence_signal.md): **Supported (2026-05-10)** — Chunk-as-retrieval-unit dilutes signal on double-evidence queries (15pp gap at R@10, 71% K=1 zero-recall). Mentor C2 todo closed via this claim.
- [claim:C10](/projects/_hdd/myyyx1/data-process-test/research-wiki/claims/C10_graph_rerank_modality_selective.md): **Supported (2026-05-10, strengthened by C11)** — Graph rerank effect is modality-selective: figure +10.3pp / table +8.3pp / formula 0.0pp. Paper claims C1/C5/C7 must add modality scope.
- [claim:C11](/projects/_hdd/myyyx1/data-process-test/research-wiki/claims/C11_formula_ceiling_is_dense_encoder_bound.md): **Supported & strengthened (2026-05-10)** — Formula retrieval R@10 ≈ 0.56 ceiling is dense-encoder bound. 10 configs (line_no fix + caption injection + LaTeX normalization + reranker families) all hit ≤ 0.56 on formula. Surface normalization and NL injection cannot rescue LaTeX; math-aware encoder swap is the only untested lever.

---

## Track C — 1040-doc Corpus 建设（元素 enrich + hub pair 生产）

**Goal**: 为 1040-doc 全量 corpus 补齐 LLM element enrichment，产出可用于 query 生成的 hub pair candidates。Method C（长链 4+ hop）已废除，不再追踪。

### 1040-doc 现状快照

| 组件 | 状态 | 数量 |
|------|------|------|
| Reference graph v2 | ✅ 完成 | 1425 docs（含 1040） |
| Chunk merge v2 | ✅ 完成 | 1147 docs，42273 chunks |
| Element extraction | ✅ 完成 | 1040 docs，27209 elements |
| LLM enrich（[T]/[M]/[C]） | ⚠️ 部分 | 10988/27209（40.4%） |
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

## Reference

- [ref:data_architecture](/projects/_hdd/myyyx1/data-process-test/research-wiki/reference/data_architecture.md) — 数据架构全景：Track A（实验轨 53-doc）vs Track B（生产轨 1040-doc），含数据流、关键文件、已知问题。
- [ref:p1_chunk_element_edge_bug](/projects/_hdd/myyyx1/data-process-test/research-wiki/reference/p1_chunk_element_edge_bug.md) — P1 Bug：chunk_contains_element 边与 eval qrels 不一致的根因、影响范围、修复方案。
- [ref:multimodal_element_taxonomy](/projects/_hdd/myyyx1/data-process-test/research-wiki/reference/multimodal_element_taxonomy.md) — Mentor B3 todo：5 类 element（paragraph=text, figure, table, equation, [no inline]），mineru→latex 元素级匹配率 49.7%/67.3%/0%。

## Gaps

- `gap:G1`: Evidence localization is still limited by dense retrieval ranking quality on multimodal scientific documents.
- `gap:G2`: Current virtual edges are mostly intra-document and do not support cross-document retrieval well.
- `gap:G3`: The project needs a clean, low-cost data delivery path with simpler QC and negatives.
- `gap:G4`: QA-side proof of graph value is still missing.
