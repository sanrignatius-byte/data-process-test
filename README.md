# Document Graph for Document Understanding

基于多层异构图的学术文档理解系统。核心创新是面向学术论文的多层图构建方法，支持多种下游任务（query 生成、QA、文档检索、证据定位）。

## 概述

本项目实现了完整的数据处理流水线：

1. **PDF + LaTeX 下载**：从 arXiv 获取论文 PDF 和 LaTeX 源码
2. **MinerU 解析**：提取文本、表格、图片和公式
3. **构建 MinerU 基础图**：多模态元素 DAG（figure/table/formula/section 关系）
4. **构建 LaTeX 基础图**：引用关系 DAG（\label/\ref/\cite）+ 跨文档引用图
5. **合并为统一图**：拓扑分析 + Hub 检测 + 多跳候选路径
6. **Enrichment**：无 LLM 版（纯规则/拓扑特征）和调用 LLM 版（[T]/[M]/[C] 语义增强 + section 摘要）
7. **Query 生成**：11 种 prompt 模板 × 76 种学术人设 × 3 种 query 风格
8. **QC 检查**：25+ 原子检查，学术/真实用户双轨 QC
9. **实验评测**：BM25 vs Graph 检索、难度梯度验证、消融实验

## 架构

```
arXiv PDF+LaTeX → [下载器] → MinerU 解析 → 多模态元素提取
                      ↓                          ↓
               LaTeX 源码解析          build_multimodal_relationships.py
                      ↓                          ↓
            build_latex_reference_graph.py   multimodal_elements.json
            build_citation_graph.py              ↓
                      ↓                   ┌──────┴──────┐
             LaTeX 引用 DAG + 跨文档引用图 │              │
                      └──────────┬────────┘              │
                    analyze_latex_graph_topology.py       │
                                 ↓                       │
                    Hub 检测 + 多跳候选路径               │
                                 ↓                       ↓
                    enrich_hub_candidates.py ← enrich_elements_modora.py
                                 ↓              enrich_section_nodes.py
                    generate_multihop_l1_queries.py
                                 ↓
                    QC (src/qc/) → 实验评测 (run_phase0_eval_ab.py)
```

## 安装

### 前置条件

- Python 3.9+
- CUDA GPU（推荐4x A2000）
- MinerU（`pip install mineru[all]`）

### 环境设置

```bash
# 克隆仓库
git clone <repository-url>
cd data-process-test

# 安装依赖
pip install -r requirements.txt

# 单独安装MinerU（依赖较复杂）
pip install mineru[all]

# 设置API密钥
export ANTHROPIC_API_KEY="your-api-key"
# 或
export OPENAI_API_KEY="your-api-key"
```

## 使用方法

### 本地查看 `local_api_logger` 月度用量（2026-03 示例）

如果你已经有本地日志目录（例如 Windows 路径 `D:\Code_store\data-process-test\api_logs`），可以直接用独立脚本统计某个月的 token 用量（不修改 `local_api_logger/viewer.py`）：

```bash
python scripts/view_api_usage_monthly.py \
  --log-dir "D:\Code_store\data-process-test\api_logs" \
  --month 2026-03
```

按模型过滤（例如 `stats/claude-sonnet-4-20250514/*.jsonl`）：

```bash
python scripts/view_api_usage_monthly.py \
  --log-dir "D:\Code_store\data-process-test\api_logs" \
  --month 2026-03 \
  --model "claude-sonnet-4-20250514"
```

导出 2026-03 明细为 CSV：

```bash
python scripts/view_api_usage_monthly.py \
  --log-dir "D:\Code_store\data-process-test\api_logs" \
  --month 2026-03 \
  --export-csv "D:\Code_store\data-process-test\api_logs\usage_2026-03.csv"
```

按你给的 `LogViewer` 风格输出并保存为 TXT：

```bash
python scripts/view_api_usage_monthly.py \
  --log-dir "D:\Code_store\data-process-test\api_logs" \
  --month 2026-03 \
  --output-txt "D:\Code_store\data-process-test\api_logs\usage_2026-03.txt"
```

### 方式1：完整 Pipeline（推荐）

```bash
# 1. 下载引用论文 PDF + LaTeX 源码
python scripts/download_references_by_arxiv.py \
    --arxiv-id 2501.09959 --output data/raw_pdfs
python scripts/download_latex_sources.py

# 2. MinerU 解析 PDF（在集群上用 GPU）
# 使用 slurm_scripts/02_parse_pdfs.sh 或直接调用 MinerU CLI

# 3. 构建 MinerU 基础图
python scripts/build_multimodal_relationships.py

# 4. 构建 LaTeX 基础图 + 跨文档引用图
python scripts/build_latex_reference_graph.py \
    --source-dir data/latex_sources/extracted \
    --output data/latex_reference_graph.json
python scripts/build_citation_graph.py \
    --input data/latex_reference_graph.json \
    --output data/citation_graph.json

# 5. 合并为统一图 + Hub 检测
python scripts/analyze_latex_graph_topology.py

# 6. Enrichment（LLM 版）
python scripts/enrich_elements_modora.py \
    --input data/multimodal_elements.json \
    --output data/multimodal_elements_enriched.json
python scripts/enrich_section_nodes.py \
    --reference-graph data/latex_reference_graph.json \
    --output data/section_nodes_enriched.json
python scripts/enrich_hub_candidates.py \
    --hub-candidates data/latex_hub_multihop_candidates.json \
    --elements data/multimodal_elements.json \
    --latex-graph data/latex_reference_graph.json \
    --output data/hub_candidates_enriched.json

# 7. Query 生成（3 种风格 × 76 种人设）
python scripts/generate_multihop_l1_queries.py \
    --candidates data/hub_candidates_enriched.json \
    --output data/l1_queries.jsonl \
    --pass-only \
    --query-style mixed \
    --use-persona

# 8. 实验评测
python scripts/run_phase0_eval_ab.py
```

### 方式2：在 Slurm 上执行下载+解析

```bash
# 1) 完整流程（下载引用 → MinerU 解析）
./slurm_scripts/submit_all.sh --arxiv-id 2501.09959

# 2) 控制下载规模
./slurm_scripts/submit_all.sh \
    --arxiv-id 2501.09959 \
    --max-references 300 \
    --min-citations 3

# 3) 跳过下载，只跑解析
./slurm_scripts/submit_all.sh --skip-download
```

解析完成后，在本地继续执行图构建 → enrichment → query 生成。

## 配置说明

编辑 `configs/config.yaml`：

```yaml
# MinerU解析设置
mineru:
  backend: "auto"  # auto, pipeline, hybrid, vlm
  devices: ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
  num_workers: 4
  standardize_image_names: true  # 标准化图片命名

# Query生成设置
query_generation:
  provider: "anthropic"  # anthropic 或 openai
  model: "claude-sonnet-4-20250514"
  queries_per_element: 3
  batch_size: 10
  rate_limit: 60

# 负例采样设置
negative_sampling:
  strategy: "modal_mixed"  # random, modal_same, modal_mixed, semantic_hard
  num_negatives: 3
  distribution:
    hard_same_modal: 0.6
    cross_modal: 0.3
    random: 0.1
```

## 输出格式

### Query 输出格式（L1 Dual-evidence）

```json
{
  "pair_id": "abc123",
  "query": "When the loss curve flattens above 0.8 threshold, ...",
  "answer": "The plateau corresponds to Table 3 row 4 where ...",
  "query_type": "figure_table_2hop",
  "query_style": "academic",
  "persona_id": "phd_ml_fairness",
  "required_evidence_spans": [
    {"element_id": "doc_fig_1", "span": "loss curve plateau", "evidence_type": "visual"},
    {"element_id": "doc_tbl_3", "span": "row 4 threshold value", "evidence_type": "tabular"}
  ],
  "visual_anchors": [...],
  "qc_pass": true,
  "qc_metrics": {...}
}
```

## 目录结构

```
data-process-test/
├── data/
│   ├── raw_pdfs/                         # 下载的 PDF
│   ├── latex_sources/extracted/          # LaTeX 源码
│   ├── mineru_output/                    # MinerU 解析输出
│   ├── multimodal_elements.json          # MinerU 基础图
│   ├── latex_reference_graph.json        # LaTeX 引用 DAG
│   ├── citation_graph.json               # 跨文档引用图
│   ├── latex_graph_hubs.json             # Hub 节点
│   ├── latex_hub_multihop_candidates.json # 多跳候选路径
│   ├── hub_candidates_enriched.json      # Enriched 候选对
│   ├── personahub_academic_personas.json # 76 种学术人设
│   └── m2/                               # 评测数据
├── scripts/
│   ├── download_references_by_arxiv.py   # PDF 下载
│   ├── download_latex_sources.py         # LaTeX 源码下载
│   ├── build_multimodal_relationships.py # MinerU 图构建
│   ├── build_latex_reference_graph.py    # LaTeX 图构建
│   ├── build_citation_graph.py           # 跨文档引用图
│   ├── analyze_latex_graph_topology.py   # 统一图 + Hub 检测
│   ├── enrich_elements_modora.py         # 元素 [T]/[M]/[C] 增强（LLM）
│   ├── enrich_section_nodes.py           # Section 语义摘要（LLM）
│   ├── enrich_hub_candidates.py          # 候选对组装（无 LLM）
│   ├── generate_multihop_l1_queries.py   # 核心 Query 生成
│   ├── run_phase0_eval_ab.py             # BM25 vs Graph 检索评测
│   └── ...                               # 更多评测/工具脚本
├── src/
│   ├── api/                              # 统一 LLM API 客户端
│   ├── models/                           # 共享数据模型 (Node, Edge, Chunk)
│   ├── parsers/                          # PDF/LaTeX 解析器
│   ├── linkers/                          # 跨文档关联
│   ├── prompts/                          # Prompt 模板 + 人设管理
│   ├── qc/                               # 质量检查系统 (25+ 原子检查)
│   ├── retrieval/                        # 检索模块 (BM25Lite)
│   └── utils/                            # 工具函数
├── slurm_scripts/                        # SLURM 集群作业脚本
├── local_api_logger/                     # API 调用日志
└── README.md
```

## 核心模块说明

### CrossDocumentLinker

跨文档实体关联模块，负责：
- **实体提取**：从文档中提取方法、数据集、指标、任务等实体
- **跨文档链接**：基于名称相似度建立文档间实体关联
- **Evidence Chain构建**：构建带有桥接实体的推理链

### Query 生成系统

`generate_multihop_l1_queries.py` 是核心 query 生成脚本，特点：
- **11 种 prompt 模板**：5 种学术风格（figure+table 1/2-hop, figure+formula, formula+table, 3-step reasoning chain）+ 5 种真实用户风格（factual, summary, comparison, how_works, what_if）+ system prompt
- **76 种学术人设**（PersonaHub）：从 `data/personahub_academic_personas.json` 加载，按 pair_id 稳定哈希分配
- **3 种 query 风格**：`academic`（默认）/ `real_user` / `mixed`（50/50 混合）
- **25+ QC 检查**：双轨制（学术严格 QC + 真实用户宽松 QC）

Prompt 模板和人设管理从 `src/prompts/` 导出，可被其他脚本复用：
```python
from src.prompts import (
    # 模板
    PROMPT_FIGURE_TABLE_1HOP, PROMPT_REAL_USER_FACTUAL,
    REAL_USER_TEMPLATES, REAL_USER_STYLE_CYCLE,
    # 人设
    load_personahub_personas, resolve_persona, resolve_persona_id,
    inject_persona_prefix,
    # 风格
    resolve_query_style, select_template,
)
```

## 性能预估

### 处理 80+ 篇文档

| 阶段 | 时间 | 说明 |
|------|------|------|
| PDF 解析 | 2-4 小时 | MinerU GPU 并行 |
| 图构建 | < 5 分钟 | 纯规则，零 LLM 成本 |
| Element Enrichment | ~$8 | GPT-5.4 / Claude |
| Section Enrichment | ~$8 | GPT-5.4 / Claude |
| Query 生成 | ~$5-15 | 取决于候选数和模型 |

## 故障排除

### MinerU 未找到

```bash
pip install mineru[all]
```

### API 速率限制

- 调整 `--delay` 参数（默认 0.3s）
- 使用 `--dry-run` 先测试
- 使用 `--incremental` 断点续跑

## 学术参考

- **PersonaHub** (2024): Scaling Synthetic Data Creation with 1,000,000,000 Personas (arXiv:2406.20094)
- **MoDora** (2024): CCTree 文档理解框架

## License

MIT License
