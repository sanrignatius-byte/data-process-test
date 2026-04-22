# Research Wiki Index

Last updated: 2026-04-21T12:00:00Z

## Project Direction

Build a document graph over multimodal academic papers and test whether graph signals improve evidence localization, QA support, and synthesis of high-quality SFT data.

Latest requirement anchors:
- `4.16.md` top-level plan: deliver SFT data first, patent or trade secret second, paper optional.
- `标准录音 57.mp3_20260417_190739_精转文稿.docx` later section: prioritize retrieval uplift, QA uplift, and data synthesis value; keep `summary` as the only immediate virtual-node priority; simplify QC; do not widen the story around virtual edges yet.

Latest discussion log:
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

## Claims

- [claim:C1](/projects/_hdd/myyyx1/data-process-test/research-wiki/claims/C1_explicit_static_prior_improves_rerank.md): Supported.
- [claim:C2](/projects/_hdd/myyyx1/data-process-test/research-wiki/claims/C2_intra_doc_virtual_edges_dilute_precision.md): Supported.
- [claim:C3](/projects/_hdd/myyyx1/data-process-test/research-wiki/claims/C3_cross_doc_summary_edges_can_help.md): Reported, pending validation.
- [claim:C4](/projects/_hdd/myyyx1/data-process-test/research-wiki/claims/C4_graph_value_must_be_proven_on_three_axes.md): Active framing claim.

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

- `gap:G1`: Evidence localization is still limited by dense retrieval ranking quality on multimodal scientific documents.
- `gap:G2`: Current virtual edges are mostly intra-document and do not support cross-document retrieval well.
- `gap:G3`: The project needs a clean, low-cost data delivery path with simpler QC and negatives.
- `gap:G4`: QA-side proof of graph value is still missing.
