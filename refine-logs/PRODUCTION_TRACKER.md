# Production Tracker — Track B

> Track B focus: bulk SFT query production and delivery packaging.
> Graph research experiments are in `EXPERIMENT_TRACKER.md` (Track A).

---

## Inventory Snapshot (2026-04-19)

| File | Pass | Type | Candidate Source |
|------|------|------|-----------------|
| sweep_2026-04-12/l3_academic_pass.jsonl | 23 | L3 | l3_candidates_v4_intra_doc (88 pairs) |
| sweep_2026-04-12/l3_academic_persona_pass.jsonl | 30 | L3 | same |
| sweep_2026-04-12/l3_mixed_pass.jsonl | 73 | L3 | same |
| sweep_2026-04-12/l3_mixed_persona_pass.jsonl | 58 | L3 | same |
| sweep_2026-04-12/m2_academic_pass.jsonl | 85 | M2 | m2_diverse_candidates_intra_doc (108 pairs) |
| sweep_2026-04-12/m2_mixed_persona_pass.jsonl | 100 | M2 | same |
| l3_enriched_v3_rerun2_pass.jsonl | 93 | L3 | hub_candidates_enriched_v3 |
| l3_enriched_v3_new82_rerun2_pass.jsonl | 53 | L3 | new82 docs |
| m2_diverse_v1_hub_kb_pass.jsonl | 29 | M2 | m2_diverse hub_kb |
| long_chain_iterative_pass.jsonl | 12 | Long | chain 3-11 hop |
| **TOTAL** | **556** | | |

**500 条目标已达成。** 当前阶段：打包 delivery + QC 验证。

---

## Task Runs

| Run ID | Task | Script | Input | Priority | Status | Notes |
|--------|------|--------|-------|----------|--------|-------|
| P001 | **打包 delivery v2** | `scripts/build_full_delivery.py` | 上方 10 个 pass 文件 | **MUST** | TODO | 合并去重 + 生成 qrels/triplets/corpus |
| P002 | **LLM QC 抽样验证** | `scripts/rerun_llm_qc.py` | sweep_2026-04-12 pass 文件（抽 50 条） | **MUST** | TODO | 验证新 sweep 质量；`log_run()` 未接入，先补 |
| P003 | L3 LLM QC 全量重跑 | `scripts/generate_multihop_l1_queries.py` | hub_candidates_enriched_v3 (230 pairs) | NICE | TODO | ablation bug 已修（见 src/qc/llm_judge.py）；需重跑 |
| P004 | neg evidence 标注方案设计 | 待设计 | — | NICE | TODO | 目前只有正例 evidence chain，缺负例 |
| P005 | supply 扩充（如需超过 700 条） | `scripts/run_production_batch.py` | 新 intra-doc candidate pool | NICE | STANDBY | 当前 556 已超目标，触发条件：mentor 要求更多 |

---

## 关键命令参考

```bash
cd /projects/myyyx1/data-process-test
set -a && source .env && set +a

# P001: 打包 delivery
python scripts/build_full_delivery.py \
  --pass-files \
    data/03_queries/sweep_2026-04-12/l3_academic_pass.jsonl \
    data/03_queries/sweep_2026-04-12/l3_academic_persona_pass.jsonl \
    data/03_queries/sweep_2026-04-12/l3_mixed_pass.jsonl \
    data/03_queries/sweep_2026-04-12/l3_mixed_persona_pass.jsonl \
    data/03_queries/sweep_2026-04-12/m2_academic_pass.jsonl \
    data/03_queries/sweep_2026-04-12/m2_mixed_persona_pass.jsonl \
    data/03_queries/l3_enriched_v3_rerun2_pass.jsonl \
    data/03_queries/l3_enriched_v3_new82_rerun2_pass.jsonl \
    data/03_queries/m2_diverse_v1_hub_kb_pass.jsonl \
    data/03_queries/long_chain_iterative_pass.jsonl \
  --output data/06_delivery/delivery_v2_2026-04-19.jsonl

# P002: LLM QC 抽样（先确认 rerun_llm_qc.py 已接入 log_run）
python scripts/rerun_llm_qc.py \
  --input data/03_queries/sweep_2026-04-12/l3_mixed_pass.jsonl \
  --sample 50 --provider company
```

---

## 已关闭 / 不再追踪

| 旧 Run ID | 原计划 | 原因 |
|-----------|-------|------|
| R001 (EXPERIMENT_TRACKER) | Production readiness audit | ✅ Done (4.18) |
| R002-R007 | First production sweep | ✅ Superseded by sweep_2026-04-12 (Job 58722) |
