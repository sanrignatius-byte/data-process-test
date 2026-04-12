# 周报 2026-04-07 ～ 2026-04-12

### 本周完成

- **方案C从 pilot 切到 scale-up**：主线从“小样本验证长链值不值得做”切到“在 1000+ 论文图上做真正的多跳资产建设”
- **图清洗 + hub 打分完成**：`pruned_graph_v2.json`、`hub_scores_v2.json`、`prune_report_v2.json` 已产出；图从 1425 篇清到 1040 篇，top 20% hubs 共 36175 个
- **Method C pilot 收回现有主干**：`pilot_method_c.py` 按 `generate_l1 + rule QC + LLM QC` 的现有结构重写，不再自起一套 QC 旁路
- **旧 enriched 数据实跑过一轮**：确认现在不是“跑不通”，而是“能跑通，但多跳必要性还不够强”；主要问题是 `llm_fake_multihop`
- **Method C enrich 目标策略定稿**：不按全局 hub 粗暴全 enrich，改成按 **long-chain bundle** enrich，覆盖 endpoint element + 中间 modal element + bridge 节点
- **scale-up 脚手架补齐**：新增 target builder、bridge enrichment、slurm 作业脚本，并给 `enrich_elements_modora.py` 补了 `--flush-every`，支持长作业断点续跑

### 当前进度

- `min_hops=4` 的 long-chain bundle 目标已经建好：
  - `12640` 条 candidate
  - `6718` 个 multimodal element
  - `5380` 个 bridge 节点
- slurm 作业 `58353` 已启动，当前在跑 Stage 2 element enrichment
- 截至 **2026-04-12 03:10 UTC**：
  - live log 进度 `460 / 6718`
  - 已落盘 `400` 个 element
  - parse fail `3`
- 这轮就算 24 小时跑不完整体，也不会白跑，因为 element / bridge 两段都已经接了增量续跑

### 风险 / 下周计划

- 当前最大的技术风险不在 enrichment，而在后续 query 生成：老数据测试显示桥虽然找到了，但还没有稳定变成“不可删的必要桥”
- 下周优先把 enrichment 跑完，然后在新 enrich 资产上重跑 Method C，重点压 `llm_fake_multihop` 和 `formula_symbol_grounding_missing`
- 如果 scale-up 跑完质量仍不稳，下一步就收紧 bridge 选择规则：长链用于找路，短证据链用于出题，完整链用于验题
