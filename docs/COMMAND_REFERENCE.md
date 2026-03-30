# Document Graph 项目 — 命令行参考手册

> 最后更新：2026-03-21 | 所有路径相对于 `/projects/myyyx1/data-process-test/`

---

## 0. 环境准备

```bash
# 学校集群
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU
export $(grep -v '^#' /projects/myyyx1/data-process-test/.env | xargs)
cd /projects/myyyx1/data-process-test

# 公司电脑 — 确认 .env 中有 COMPANY_API_KEY 和 COMPANY_API_URL
export $(grep -v '^#' .env | xargs)
```

---

## 1. 数据采集

### 1.1 下载 arXiv LaTeX 源码

```bash
python scripts/download_latex_sources.py \
  --from-mineru data/mineru_output \
  --output data/latex_sources \
  --delay 4.0 \
  --extract-only        # 仅解压已下载的
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ids` | — | 直接指定 arXiv ID 列表 |
| `--id-file` | — | 从文件读 arXiv ID |
| `--from-pairs` | — | 从 paper_pairs 目录推断 |
| `--from-mineru` | — | 从 mineru_output 目录推断 |
| `--output` | `data/latex_sources` | 输出目录 |
| `--delay` | `4.0` | 请求间隔（秒） |
| `--extract-only` | False | 仅解压 |
| `--no-verify` | False | 跳过 SSL 验证 |

### 1.2 下载参考文献 PDF

```bash
python scripts/download_references_by_arxiv.py --arxiv-id 1908.09635
```

### 1.3 雪球式下载 PDF + LaTeX

```bash
python scripts/download_pdf_latex_pairs_snowball.py \
  --seeds 1908.09635 \
  --target-count 100 \
  --max-depth 2 \
  --output data/arxiv_pairs_snowball
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seeds` | **必填** | 种子 arXiv ID |
| `--target-count` | `5000` | 目标论文数 |
| `--max-depth` | `2` | 引用链最大深度 |
| `--max-refs-per-paper` | `500` | 每篇最多下载引用数 |
| `--arxiv-delay-s` | `4.0` | arXiv 请求间隔 |
| `--s2-delay-s` | `6.0` | Semantic Scholar 请求间隔 |
| `--checkpoint-every` | `50` | 每 N 篇保存 checkpoint |
| `--expand-failed` | False | 重试之前失败的 |
| `--s2-api-key` | — | S2 API key |

---

## 2. 图构建

### 2.1 构建 LaTeX 引用图

```bash
python scripts/build_latex_reference_graph.py \
  --source-dir data/latex_sources/extracted \
  --output data/latex_reference_graph.json \
  --report data/latex_reference_report.json
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source-dir` | `data/latex_sources/extracted` | LaTeX 源码目录 |
| `--output` | `data/latex_reference_graph.json` | 输出引用图 |
| `--report` | `data/latex_reference_report.json` | 输出报告 |
| `--doc-ids` | — | 仅处理指定文档 |
| `--max-hops` | `3` | 最大跳数 |
| `--merge-with` | — | 合并已有 elements |
| `--merged-output` | `data/multimodal_elements_v2.json` | 合并输出 |

### 2.2 构建跨文档引用图

```bash
python scripts/build_citation_graph.py \
  --input data/latex_reference_graph.json \
  --output data/citation_graph.json
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `data/latex_reference_graph.json` | 输入引用图 |
| `--output` | `data/citation_graph.json` | 输出引用图 |
| `--from-sources` | — | 直接从 LaTeX 源码构建 |

### 2.3 构建跨模态链接（Step 0 v3.2）

```bash
python scripts/build_latex_cross_modal_links.py \
  --elements data/multimodal_elements.json \
  --latex-graph data/latex_reference_graph.json \
  --output data/latex_cross_modal_pairs.json
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--elements` | `data/multimodal_elements.json` | MinerU 元素 |
| `--latex-graph` | `data/latex_reference_graph.json` | LaTeX 引用图 |
| `--output` | `data/latex_cross_modal_pairs.json` | 输出跨模态 pair |
| `--min-match-conf` | `0.35` | 最低匹配置信度 |

### 2.4 图拓扑分析 + Hub 识别 + 候选生成

```bash
python scripts/analyze_latex_graph_topology.py \
  --latex-graph data/latex_reference_graph.json \
  --elements data/multimodal_elements.json \
  --citation-graph data/citation_graph.json \
  --mineru-output data/mineru_output \
  --output-hubs data/latex_graph_hubs.json \
  --output-candidates data/latex_hub_multihop_candidates.json \
  --max-candidates 500
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--latex-graph` | `data/latex_reference_graph.json` | 输入引用图 |
| `--elements` | `data/multimodal_elements.json` | MinerU 元素 |
| `--cross-modal-pairs` | `data/latex_cross_modal_pairs.json` | 跨模态 pair |
| `--citation-graph` | `data/citation_graph.json` | 引用图 |
| `--mineru-output` | `data/mineru_output` | MinerU 输出目录 |
| `--output-report` | `data/latex_graph_topology_report.json` | 拓扑报告 |
| `--output-hubs` | `data/latex_graph_hubs.json` | Hub 列表 |
| `--output-candidates` | `data/latex_hub_multihop_candidates.json` | 候选 pair |
| `--top-k-hubs` | `60` | 保留 top-K hub |
| `--min-hops` | `2` | 候选最小跳数 |
| `--max-hops` | `5` | 候选最大跳数 |
| `--max-candidates` | `500` | 候选上限 |
| `--single-doc-only` | False | 仅文档内候选 |

### 2.5 构建 Embedding 语义边（新）

```bash
python scripts/build_embedding_edges.py \
  --elements data111/multimodal_elements_enriched.json \
  --hub-candidates data111/hub_candidates_enriched_v3.json \
  --output data/embedding_edges.json \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --threshold 0.8 \
  --inspect \
  --save-embeddings data/element_embeddings.npy
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--elements` | `data111/multimodal_elements_enriched.json` | 元素文件（推荐 enriched） |
| `--output` | `data/embedding_edges.json` | 输出边 |
| `--model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding 模型 |
| `--threshold` | `0.80` | 余弦相似度阈值 |
| `--max-edges-per-element` | `10` | 每元素最大边数 |
| `--batch-size` | `64` | 编码批大小 |
| `--device` | 自动 | `cuda` / `cpu` / `mps` |
| `--hub-candidates` | — | 已有图边（排除重复） |
| `--inspect` | False | 打印 top 边供人工检查 |
| `--top-k` | `30` | inspect 显示数量 |
| `--same-doc-only` | False | 仅文档内边 |
| `--cross-doc-only` | False | 仅跨文档边 |
| `--no-enriched` | False | 不使用 enriched_content |
| `--save-embeddings` | — | 保存 embedding 为 .npy |
| `--load-embeddings` | — | 加载预计算 embedding |

**依赖**：`pip install sentence-transformers torch`
**不需要 LLM API**，纯本地计算。

---

## 3. 候选筛选 & 富化

### 3.1 富化 Hub 候选（映射 MinerU 元素）

```bash
python scripts/enrich_hub_candidates.py \
  --hub-candidates data/latex_hub_multihop_candidates.json \
  --elements data/multimodal_elements.json \
  --latex-graph data/latex_reference_graph.json \
  --output data/hub_candidates_enriched.json \
  --enriched-elements data111/multimodal_elements_enriched.json \
  --hubs data/latex_graph_hubs.json
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hub-candidates` | `data/latex_hub_multihop_candidates.json` | 原始候选 |
| `--elements` | `data/multimodal_elements.json` | MinerU 元素 |
| `--latex-graph` | `data/latex_reference_graph.json` | 引用图（取 label 信息） |
| `--output` | `data/hub_candidates_enriched.json` | 输出 |
| `--enriched-elements` | — | MoDora 富化后的元素文件 |
| `--hubs` | `data/latex_graph_hubs.json` | Hub 拓扑数据 |
| `--limit` | `0` | 限制候选数（0=全部） |

**不需要 LLM API**，纯映射 + 文本拼接。

### 3.2 筛选 L3 候选（3-hop）

```bash
python scripts/filter_l3_candidates.py \
  --output data/m2/l3_candidates_filtered.json \
  --min-hops 3 \
  --max-candidates 200
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--candidates` | 自动检测 | 输入候选 |
| `--output` | `data/m2/l3_candidates_filtered.json` | 输出 |
| `--min-hops` | `3` | 最小跳数 |
| `--max-candidates` | `200` | 上限 |

### 3.3 多跳候选选择（旧版，L1 用）

```bash
python scripts/select_multihop_candidates.py \
  --elements data/multimodal_elements.json \
  --existing-l1 data/l1_cross_modal_queries_v3.jsonl \
  --output data/multihop_l1_candidates.json \
  --max-pairs 150 --max-per-doc 5 --max-hops 2
```

### 3.4 L2 跨文档候选构建

```bash
python scripts/build_l2_candidates.py \
  --input data/l1_triage_v3.jsonl \
  --output data/l2_candidate_pairs_v2.json \
  --topk 100 --min-class A
```

---

## 4. Query 生成（需要 LLM API）

### 4.1 生成 L1/L2/L3 Query（主力脚本）

```bash
# L2（2-hop）+ L3（3-hop）混合生成
python scripts/generate_multihop_l1_queries.py \
  --candidates data/m2/hub_candidates_enriched_full.json \
  --output data/m2/l2_new_batch.jsonl \
  --pass-only \
  --provider company --model gpt-5.4 \
  --query-style mixed \
  --delay 0.5

# 仅 L3 候选
python scripts/generate_multihop_l1_queries.py \
  --candidates data/m2/l3_candidates_filtered.json \
  --output data/m2/l3_new_batch.jsonl \
  --pass-only \
  --provider company --model gpt-5.4
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--candidates` | `data/latex_cross_modal_pairs.json` | 输入候选 pair |
| `--output` | `data/l1_dual_evidence_queries_v3.jsonl` | 输出 JSONL |
| `--pass-only` | False | 额外输出 `_pass.jsonl` |
| `--provider` | `company` | `company` / `anthropic` / `openai` |
| `--model` | 自动选 | company→gpt-5.4, anthropic→claude-sonnet |
| `--company-api-url` | `$COMPANY_API_URL` | 公司 API 地址 |
| `--company-api-key` | `$COMPANY_API_KEY` | 公司 API 密钥 |
| `--limit` | `0` | 限制处理数（0=全部） |
| `--shuffle` | False | 打乱候选顺序 |
| `--delay` | `0.5` | API 调用间隔（秒） |
| `--dry-run` | False | 仅打印 prompt，不调 API |
| `--no-images` | False | 不发送图片 |
| `--query-style` | `academic` | `academic` / `real_user` / `mixed` |
| `--use-persona` | False | 注入 PersonaHub 人设 |

**LLM API**：✅ 必需（company/gpt-5.4 或 anthropic/claude）
**Token Logger**：✅ 已接入 `log_run()`
**L3 判定**：候选中 `reasoning_chain_target=True` → L3（1 query/pair），否则 L2（2 queries/pair）

### 4.2 生成 L2 跨文档 Query

```bash
python scripts/generate_l2_queries.py \
  --pairs data/l2_candidate_pairs_v2.json \
  --output data/l2_queries_v3.jsonl \
  --limit 50 \
  --provider anthropic
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pairs` | `data/l2_candidate_pairs_v2.json` | 候选 pair |
| `--output` | `data/l2_queries_v3.jsonl` | 输出 |
| `--limit` | `50` | 限制数量 |
| `--provider` | `anthropic` | 供应商 |
| `--model` | `claude-sonnet-4-5-20250929` | 模型 |
| `--delay` | `0.5` | 间隔 |
| `--anchor-leak-threshold` | `0.15` | 锚点泄漏阈值 |
| `--evidence-closure-threshold` | `0.50` | 证据闭合阈值 |
| `--no-images` | False | 不发图 |
| `--dry-run` | False | 干跑 |

### 4.3 迭代式长链 Query 生成

```bash
python scripts/generate_long_chain_iterative_queries.py \
  --candidates data/latex_long_chain_pairs_all_q0.json \
  --output data/l1_dual_evidence_long_chain_queries_v2_iterative.jsonl \
  --pass-only \
  --model claude-sonnet-4-5-20250929 \
  --repair-attempts 1
```

### 4.4 量产 Pipeline（新，自动过滤+生成+合并）

```bash
python scripts/run_production_batch.py \
  --provider company --model gpt-5.4 \
  --batch-name production_v1 \
  --query-style mixed

# 带 enrichment 前置步骤
python scripts/run_production_batch.py \
  --enrich-first \
  --raw-candidates data/latex_hub_multihop_candidates.json \
  --batch-name production_v2 \
  --provider company --model gpt-5.4
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enriched-candidates` | `data/m2/hub_candidates_enriched_full.json` | 已富化候选 |
| `--raw-candidates` | `data/latex_hub_multihop_candidates.json` | 原始候选（配合 --enrich-first） |
| `--enrich-first` | False | 先富化再生成 |
| `--batch-name` | `production_v1` | 批次名（影响输出文件名） |
| `--provider` | `company` | 供应商 |
| `--model` | `gpt-5.4` | 模型 |
| `--limit` | `0` | 限制数 |
| `--delay` | `0.5` | 间隔 |
| `--query-style` | `mixed` | 风格 |
| `--dry-run` | False | 仅显示计划 |
| `--skip-package` | False | 跳过末尾的 package_m2_levels |

**自动流程**：收集已用 pair_id → 过滤候选 → 生成 → 拆分 L2/L3 → 合并到 level 文件 → 重新打包

---

## 5. 元素富化（需要 LLM API）

### 5.1 MoDora 元素富化

```bash
python scripts/enrich_elements_modora.py \
  --input data/multimodal_elements.json \
  --output data/multimodal_elements_enriched.json \
  --provider company \
  --incremental \
  --delay 0.3
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `data/multimodal_elements.json` | 输入元素 |
| `--output` | `data/multimodal_elements_enriched.json` | 输出 |
| `--provider` | — | `anthropic` / `openai` / `company` |
| `--model` | 自动 | 模型名 |
| `--delay` | `0.3` | 间隔 |
| `--limit` | `0` | 限制数 |
| `--no-images` | False | 不发图 |
| `--dry-run` | False | 干跑 |
| `--incremental` | False | 增量模式（跳过已富化的） |

**LLM API**：✅ | **Token Logger**：✅

### 5.2 批量图片理解（API 版）

```bash
python scripts/batch_figure_understanding_api.py \
  --input data/multimodal_elements.json \
  --output data/figure_descriptions_v3_api.json \
  --model claude-sonnet-4-5-20250929 \
  --delay 0.5
```

### 5.3 批量图片理解（本地 vLLM 版）

```bash
python scripts/batch_figure_understanding.py \
  --model /path/to/Qwen3-VL-30B \
  --tp-size 4 \
  --input data/multimodal_elements.json \
  --output data/figure_descriptions.json
```

---

## 6. 检索评测

### 6.1 Phase0 A/B 检索评测（主力）

```bash
# 标准运行（v3 tuned 配置）
python scripts/run_phase0_eval_ab.py \
  --q1 data/l1_dual_evidence_queries_v4_4_run1_pass.jsonl \
  --q2 data/l1_dual_evidence_queries_v3_pass.jsonl \
  --elements data111/multimodal_elements_enriched.json \
  --hub-candidates data111/hub_candidates_enriched_v3.json \
  --citation-graph data/citation_graph.json \
  --output data/phase0_eval_report_v3_tuned.json \
  --hub-weight 0.15 --nprop-weight 0.20 --cite-weight 0 \
  --neighbor-hops 1

# 融合 embedding 边
python scripts/run_phase0_eval_ab.py \
  --embedding-edges data/embedding_edges.json \
  --hub-candidates data111/hub_candidates_enriched_v3.json \
  --output data/phase0_eval_with_embedding.json \
  --hub-weight 0.15 --nprop-weight 0.20 --cite-weight 0
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--q1` | `data/l1_dual_evidence_queries_v4_4_run1_pass.jsonl` | Query 文件 1 |
| `--q2` | `data/l1_dual_evidence_queries_v3_pass.jsonl` | Query 文件 2 |
| `--q3` | — | 可选第三个 query 文件 |
| `--elements` | `data111/multimodal_elements_enriched.json` | 元素文件 |
| `--hubs` | `data111/latex_graph_hubs (1).json` | Hub 文件 |
| `--hub-candidates` | `data111/hub_candidates_enriched_v2.json` | 富化候选（用于先验+邻接） |
| `--citation-graph` | `data/citation_graph.json` | 引用图 |
| `--embedding-edges` | — | **新** embedding 边文件 |
| `--output` | `data/phase0_eval_report.json` | 输出报告 |
| `--top-k` | `10` | 评估 top-K |
| `--overlap-threshold` | `0.5` | span overlap 阈值 |
| `--graph-alpha` | `0.2` | Hub 先验权重（hub_rerank 用） |
| `--graph-rerank-topn` | `100` | 候选集大小 |
| `--neighbor-decay` | `0.5` | 邻域传播衰减因子 |
| `--citation-decay` | `0.3` | 引用游走衰减因子 |
| `--hub-weight` | `=graph-alpha` | graph_full 中 hub 权重 |
| `--nprop-weight` | `1.0` | graph_full 中邻域传播权重 |
| `--cite-weight` | `=citation-decay` | graph_full 中引用权重（设 0 关闭） |
| `--neighbor-hops` | `1` | 传播跳数（1 或 2） |
| `--max-chars` | `1800` | chunk 最大字符数 |

**不需要 LLM API**。

**最优配置**：`--hub-weight 0.15 --nprop-weight 0.20 --cite-weight 0 --neighbor-hops 1`

### 6.2 Exp A — 难度梯度验证

```bash
python scripts/run_exp_a_difficulty.py --top-k 10
```

读取 `data/m2/level{1,2,3}_*.jsonl`，输出 `data/m2/exp_a_difficulty_gradient.json`。

### 6.3 Exp C — QA 三角印证

```bash
python scripts/run_exp_c_qa_triangle.py \
  --elements data111/multimodal_elements_enriched.json \
  --provider company --model gpt-5.4 \
  --output data/m2/exp_c_qa_triangle_enriched.json \
  --top-k 5
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--elements` | `data/multimodal_elements.json` | 元素文件 |
| `--enriched-elements` | `""` | enriched 元素（可选） |
| `--top-k` | `5` | 检索 top-K |
| `--limit` | `0` | 限制 query 数 |
| `--provider` | `company` | 供应商 |
| `--model` | 自动 | 模型 |
| `--delay` | `0.5` | 间隔 |
| `--dry-run` | False | 干跑 |
| `--output` | `data/m2/exp_c_qa_triangle.json` | 输出 |

**LLM API**：✅ | **Token Logger**：✅

### 6.4 证据定位评测

```bash
python scripts/evaluate_evidence_localization.py \
  --queries data/m2/level2_dual_evidence.jsonl \
  --elements data/multimodal_elements.json \
  --mode bm25 \
  --ks 1 5 10 \
  --pass-only
```

---

## 7. 数据打包 & 交付

### 7.1 打包 M2 三级数据

```bash
python scripts/package_m2_levels.py
```

自动读取所有 L1/L2/L3 源文件（包括 `l2_production_*_pass.jsonl`），去重合并，输出：
- `data/m2/level1_single_element.jsonl`
- `data/m2/level2_dual_evidence.jsonl`
- `data/m2/level3_reasoning_chain.jsonl`
- `data/m2/all_levels_combined.jsonl`
- `data/m2/exp_b_retrieval_enhancement.json`

### 7.2 L1 分诊

```bash
python scripts/triage_l1_v3.py \
  --input data/l1_cross_modal_queries_v3.jsonl \
  --output data/l1_triage_v3.jsonl \
  --report data/l1_triage_report_v3.json
```

### 7.3 构建训练三元组

```bash
python scripts/build_dual_evidence_triplets.py \
  --queries data/l1_dual_evidence_queries_slurm_img150_tuned_v4_official.jsonl \
  --elements data/multimodal_elements.json \
  --output data/l1_dual_evidence_triplets_v2.jsonl \
  --pass-only
```

---

## 8. 跨文档 Embedding 匹配

### 8.1 跨文档元素 Embedding 匹配

```bash
python scripts/match_mineru_crossdoc_with_embeddings.py \
  --backend openai \
  --elements data/multimodal_elements.json \
  --model qwen3-embedding \
  --output data/mineru_crossdoc_embedding_matches.jsonl \
  --report data/mineru_crossdoc_embedding_report.json
```

### 8.2 匹配结果审计

```bash
python scripts/audit_mineru_crossdoc_embedding_matches.py \
  --matches data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B.jsonl \
  --elements data/multimodal_elements.json \
  --report data/mineru_crossdoc_embedding_audit.json \
  --suspicious-csv data/mineru_crossdoc_embedding_suspicious.csv \
  --sample-size 60
```

### 8.3 Utility-aware 重排

```bash
python scripts/rerank_mineru_crossdoc_matches.py \
  --input data/mineru_crossdoc_embedding_matches.jsonl \
  --output data/mineru_crossdoc_embedding_matches_rerank.jsonl \
  --report data/rerank_report.json \
  --hub-lambda 0.03 --doc-lambda 0.01 --diversity-lambda 0.02
```

---

## 9. 常用端到端 Pipeline

### 9.1 从零构建完整图（新数据集）

```bash
# Step 1: 下载 + 解析
python scripts/download_latex_sources.py --from-mineru data/mineru_output --output data/latex_sources

# Step 2: 构建图
python scripts/build_latex_reference_graph.py --source-dir data/latex_sources/extracted --output data/latex_reference_graph.json
python scripts/build_citation_graph.py --input data/latex_reference_graph.json --output data/citation_graph.json
python scripts/build_latex_cross_modal_links.py --elements data/multimodal_elements.json --latex-graph data/latex_reference_graph.json --output data/latex_cross_modal_pairs.json

# Step 3: 拓扑分析 + 候选
python scripts/analyze_latex_graph_topology.py --max-candidates 500

# Step 4: 富化候选
python scripts/enrich_hub_candidates.py --hub-candidates data/latex_hub_multihop_candidates.json --output data/hub_candidates_enriched.json --enriched-elements data111/multimodal_elements_enriched.json

# Step 5: 生成 query
python scripts/generate_multihop_l1_queries.py --candidates data/hub_candidates_enriched.json --output data/queries.jsonl --pass-only --provider company --model gpt-5.4

# Step 6: 评测
python scripts/run_phase0_eval_ab.py --hub-candidates data/hub_candidates_enriched.json --output data/eval_report.json --hub-weight 0.15 --nprop-weight 0.20 --cite-weight 0
```

### 9.2 量产 query（当前阶段）

```bash
# 一键量产（自动跳过已生成的）
python scripts/run_production_batch.py \
  --provider company --model gpt-5.4 \
  --batch-name production_v1 \
  --query-style mixed
```

### 9.3 Embedding 边实验

```bash
# Step 1: 生成 embedding 边
pip install sentence-transformers torch
python scripts/build_embedding_edges.py \
  --elements data111/multimodal_elements_enriched.json \
  --hub-candidates data111/hub_candidates_enriched_v3.json \
  --output data/embedding_edges.json \
  --threshold 0.8 --inspect --save-embeddings data/element_embeddings.npy

# Step 2: 融合到检索评测
python scripts/run_phase0_eval_ab.py \
  --embedding-edges data/embedding_edges.json \
  --hub-candidates data111/hub_candidates_enriched_v3.json \
  --output data/phase0_eval_with_embedding.json \
  --hub-weight 0.15 --nprop-weight 0.20 --cite-weight 0
```

---

## 10. 关键数据文件速查

### 输入文件

| 文件 | 说明 | 大小 |
|------|------|------|
| `data/multimodal_elements.json` | MinerU 解析的 1316 个元素（76 文档） | ~2 MB |
| `data111/multimodal_elements_enriched.json` | enriched 版（1285/1316 有 enriched_content） | ~6 MB |
| `data/latex_reference_graph.json` | LaTeX 引用图（2551 nodes, 3471 edges） | ~7 MB |
| `data/citation_graph.json` | 跨文档引用图（123 edges） | ~109 KB |
| `data/latex_hub_multihop_candidates.json` | 500 候选 pair（181×2-hop + 319×3-hop） | ~486 KB |

### 中间文件

| 文件 | 说明 |
|------|------|
| `data/m2/hub_candidates_enriched_full.json` | 230 已映射的候选 pair |
| `data111/hub_candidates_enriched_v3.json` | Phase0 用的 v3 候选（403 elements, 530 edges） |
| `data/latex_graph_hubs.json` | Hub 拓扑评分 |
| `data/embedding_edges.json` | Embedding 语义边（待生成） |

### 输出文件（当前数据集）

| 文件 | 条数 | 说明 |
|------|------|------|
| `data/m2/level1_single_element.jsonl` | 974 | L1 单元素 query |
| `data/m2/level2_dual_evidence.jsonl` | 210 | L2 双证据 query |
| `data/m2/level3_reasoning_chain.jsonl` | 115 | L3 推理链 query |
| `data/m2/all_levels_combined.jsonl` | 1299 | 合并 |

### 实验报告

| 文件 | 说明 |
|------|------|
| `data/m2/exp_a_difficulty_gradient_enriched.json` | Exp A（L1=0.971 > L2=0.610 > L3=0.617） |
| `data/m2/exp_b_retrieval_enhancement.json` | Exp B（graph_full MRR +0.0403） |
| `data/m2/exp_c_qa_triangle_enriched.json` | Exp C（graph 检索 +6.1%, QA neutral） |
| `data/m2/ablation_raw_elements.json` | Enrichment 消融 |
| `data/phase0_eval_report_v3_tuned.json` | Phase0 完整 6 方法对比 |

---

## 11. LLM API 使用规则

1. **所有调用 LLM 的脚本必须通过 `local_api_logger` 记录**
2. **公司 API**：provider=company, model=gpt-5.4, 地址在 `.env` 中
3. **已接入 token logger 的脚本**：
   - `generate_multihop_l1_queries.py` ✅
   - `generate_l2_queries.py` ✅
   - `batch_figure_understanding_api.py` ✅
   - `enrich_elements_modora.py` ✅
   - `run_exp_c_qa_triangle.py` ✅
4. **不调用 LLM 的脚本**（无需 logger）：
   - `build_embedding_edges.py`、`run_production_batch.py`（包装脚本）、`run_phase0_eval_ab.py`、`package_m2_levels.py`、`enrich_hub_candidates.py`、所有 `build_*` / `analyze_*` 脚本

## M2 一键式经典检索评测（stdlib 扩展版）

```bash
bash scripts/run_m2_classic_eval_oneclick.sh data111/multimodal_elements_enriched.json
```

- Query 输入固定为 M2：L1/L2/L3 pass 集合。
- 输出：`data/m2/phase0_classic_eval_m2_oneclick.json`。
- 如元素库不在默认路径，传入第一个参数覆盖。
