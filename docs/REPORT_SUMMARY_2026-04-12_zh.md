# 周报 2026-04-07 ～ 2026-04-12

### 本周完成

- **方案C从 pilot 切到 scale-up**：主线从“小样本验证长链值不值得做”切到“在 1000+ 论文图上做真正的多跳资产建设”
- **方案C范式定稿**：`docs/METHOD_C_GENERATION_SPEC_2026-04-12.md` 明确“长链找路 / 短证据链出题 / 完整链验题”三层分工,prompt 不再把整条长链直接喂模型
- **图清洗 + hub 打分完成**：`pruned_graph_v2.json`、`hub_scores_v2.json`、`prune_report_v2.json` 已产出；图从 1425 篇清到 1040 篇,top 20% hubs 共 36175 个
- **Long-chain 生成脚本加固到生产级**（4/9 集中落地）：
  - `--skip-done` + append + flush/fsync：进程被 kill 不再丢数据,支持断点续跑
  - `--use-persona` / `--query-style` / `--reference-graph`：长链脚本能力和 multihop 脚本拉平（persona、风格模板、LaTeX bridge 文本注入）
  - `--max-pass-per-pair` / `--dedup-jaccard` / `qc_summary_label`：同 pair 语义重复和过采样被硬门禁
  - `--max-query-hops 5`：11-hop 超长链自动切成 ≤5-hop 子链,避免 `query_too_long`
  - company LLM timeout 180s → 300s,解决长 prompt 边缘超时
  - 配套 `slurm_scripts/02_generate_long_chain.sh`
- **`src/pairing` 模块抽出**：`chain_finder` + `intra_doc_pairs` + `context_dedup` + CandidatePair schema,47 个新测试,零跨文档泄漏
- **两个新 QC check**：`conditional_hedge_overload`（答案 ≥3 个 if/assume 即 underdetermined）、`bridge_overclaim`（query 强因果 / answer 弱 hedge）
- **Evidence Markdown 导出**：`scripts/export_evidence_md.py` 生成 per-query 的 MD（含图像、evidence、reasoning chain）,便于人工审 pass 样本
- **Method C pilot 收回现有主干**：`pilot_method_c.py` v3 按 `generate_l1 + rule QC + LLM QC` 现有结构重写,修复 v1 把 answer 当 text_evidence 的 adapter bug（之前 100% 触发 `text_evidence_over_reliance`）
- **旧 enriched 数据实跑过一轮**：确认现在不是“跑不通”,而是“能跑通,但多跳必要性还不够强”；主要问题是 `llm_fake_multihop`
- **Method C enrich 目标策略定稿**：不按全局 hub 粗暴全 enrich,改成按 **long-chain bundle** enrich,覆盖 endpoint element + 中间 modal element + bridge 节点
- **scale-up 脚手架补齐**：新增 target builder、bridge enrichment、slurm 作业脚本,并给 `enrich_elements_modora.py` 补了 `--flush-every`,支持长作业断点续跑
- **Handoff 文档**：`docs/HANDOFF_LONG_CHAIN_PIPELINE_2026-04-09.md` 记录 pipeline 数据流、图像解析、链构造算法、已修和未修项

### 当前进度

- `min_hops=4` 的 long-chain bundle 目标已建好：
  - `12640` 条 candidate
  - `6718` 个 multimodal element
  - `5380` 个 bridge 节点
- slurm 作业 `58353` 已启动,当前跑 Stage 2 element enrichment
- 截至 **2026-04-12 03:10 UTC**：
  - live log 进度 `460 / 6718`
  - 已落盘 `400` 个 element
  - parse fail `3`
- 这轮就算 24h 跑不完,也不会白跑,element / bridge 两段都接了增量续跑
- **现有 pass query 存量**（可直接进下周交付池）：
  - L3 rerun2 合并去重：**145 条** pass（old 93 + new82 53,1 条重叠）
  - m2_diverse_v1_hub_kb：**29 条** pass（总 111,pass 率约 26%）
  - long_chain_v2_2026-04-07 试跑：0 / 20 pass,原因待分析（4-hop 长链 QC 过严 or prompt 需迭代）
  - 合计约 **170+ 条**,离 1500+ 的交付目标仍有一个数量级差距

### 风险 / 下周计划

- 当前最大技术风险不在 enrichment,而在后续 query 生成：老数据测试显示桥虽然找到了,但还没有稳定变成“不可删的必要桥”
- **下周 query 交付走双线并行**,不押注单一路径：
  - **A 线（保底）**：用 4/9 加固后的 long-chain 脚本 + 老的 `hub_candidates_enriched_v3/v4.json` 在 slurm 上跑一轮量产,目标加 500+ pass,不依赖 scale-up enrichment
  - **B 线（主线）**：等 scale-up enrichment 跑完（element + bridge 两段）,在新资产上重跑 Method C v3,重点压 `llm_fake_multihop` 和 `formula_symbol_grounding_missing`
- 如果 B 线跑完质量仍不稳,下一步收紧 bridge 选择规则：长链用于找路,短证据链用于出题,完整链用于验题
- **开跑前先做小批量 pilot 验证**：B 线 scale-up 一上来不做全量,先跑 50-100 pair 测 pass rate 基线,避免烧掉一整轮 enrichment token 之后才发现压缩桥策略不对
