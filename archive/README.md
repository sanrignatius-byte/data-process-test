# Archive

归档目录，保存已从主 pipeline 移出的脚本和数据文件。
这些文件在研究过程中产生，可能对论文写作、消融分析或方法回溯有参考价值。

---

## scripts/

| 子目录 | 内容 |
|--------|------|
| `01_download/` | 早期下载脚本（PDF snowball / 批量下载），已由 `download_papers_semantic_scholar.py` 取代 |
| `02_figure_understanding_vllm/` | vLLM + Claude API 批处理时代的图文理解脚本，功能已并入 `generate_multihop_l1_queries.py` |
| `03_pipeline_v1/` | 早期 pipeline 组件：候选预筛、L2 候选构建、LaTeX 跨模态链接、图合并、QC guardrails |
| `04_crossdoc_embedding/` | 跨文档 Qwen3-Embedding 匹配 + rerank + 审计流程（已由 `build_embedding_edges.py` 接替） |
| `05_one_off_tools/` | 一次性工具：引用抓取、图像重命名、HuggingFace 上传、grid 摘要、图文关联测试 |

## slurm/

vLLM 服务（Qwen3-VL）和早期生成任务的 SLURM 脚本（job 04-08），已切换为 Claude API 路径。

## data/

| 子目录 | 内容 | 大小 |
|--------|------|------|
| `l1_early_iterations/` | L1 早期迭代：figure_descriptions v1/v2，cross_modal_queries v1/v2，multihop_queries v1-v3，dual_evidence_queries v1/v2，triplets v1 | ~20M |
| `l2_early_iterations/` | L2 早期迭代：candidate_pairs v1（含 generic entities），queries v1/v2 及人工标注 tagged | ~0.5M |
| `crossdoc_embedding_matches/` | Qwen3-Embedding-4B 跨文档匹配原始结果及 v2 rerank（strict cap=8），推荐版为主 data/ 下的 v2b_cap10 | ~5.6M |
| `long_chain_candidates/` | 旧格式 LaTeX long-chain 候选对（q0 策略），已由 hub multi-hop candidates 取代 | ~2.4M |
| `batch_keyword_boost_2026-03-24/` | 2026-03-24 keyword-boost 批次的 L2/L3 生产数据及评测，已由 2026-03-26 section-enriched 批次取代 | ~4.5M |
| `batch_phase2a/` | Phase 2a bridge-grounded L3 批次（reasoning_chain_queries），已由 section-enriched 批次取代 | ~2.3M |
| `latex_graph_rebuild_2026-03-24/` | 2026-03-24 图拓扑重建快照（含 keyword_boost 版拓扑报告和候选对） | ~12M |
| `misc_backup/` | data111/ 中的非 enrich 备份文件（带括号重复副本）及早期图像 run | ~1M |
| `empty_placeholders/` | 0B 空文件，iterative long-chain 实验的占位输出 | 0B |
