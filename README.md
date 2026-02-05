# MinerU 多模态对比学习数据工厂

用于从PDF文档生成多模态对比学习训练数据的生产级流水线。专为NTU EEE集群（4x A2000 GPU）设计。

## 概述

本项目实现了完整的数据处理流水线：

1. **PDF下载**：从arXiv获取论文（或使用本地PDF）
2. **文档解析**：使用MinerU提取文本、表格、图片和公式
3. **模态提取**：按模态分类和组织内容
4. **Query生成**：使用LLM生成多样化的查询
5. **负例采样**：构建对比学习所需的硬负例
6. **数据输出**：导出JSONL格式用于训练

### 新功能：M4跨文档Query生成

基于最新学术研究（M4DocBench、TRACE、CoQA等），实现了增强的M4 Query生成：

- **Multi-hop（多跳推理）**：需要2+个证据点的推理链
- **Multi-modal（多模态）**：需要2+种模态（文本、表格、图片、公式）
- **Multi-document（多文档）**：需要跨2+个文档的证据综合
- **Multi-turn（多轮对话）**：自然的多轮对话格式，包含代词指代

## 架构

```
PDF源 → [下载器] → MinerU解析 → 模态分割 → Passage构建
                                                    ↓
                                         [实体提取] → [跨文档关联]
                                                    ↓
JSONL输出 ← 负例采样 ← Query生成(LLM) ← Evidence Chain ←┘
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

### 方式1：M4跨文档Query生成（推荐）

```bash
# 1. 先解析PDF文档
python scripts/parse_only.py --input data/raw_pdfs --output data/mineru_output

# 2. 生成M4跨文档Query
python scripts/generate_m4_queries.py \
    --input data/mineru_output \
    --output data/m4_queries/queries \
    --max-docs 10 \
    --num-queries 20
```

#### M4脚本参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | MinerU输出目录 | `data/mineru_output` |
| `--output` | Query输出路径 | `data/m4_queries/queries` |
| `--max-docs` | 最大处理文档数 | 10 |
| `--num-queries` | 目标Query数量 | 10 |
| `--provider` | LLM提供商 | `anthropic` |
| `--model` | 模型名称 | `claude-sonnet-4-20250514` |
| `--relaxed` | 放宽M4要求 | False |
| `--dry-run` | 仅统计，不调用LLM | False |

#### 使用示例

```bash
# Dry Run：只看实体和关联统计
python scripts/generate_m4_queries.py --input data/mineru_output --dry-run

# 指定特定文档
python scripts/generate_m4_queries.py \
    --input data/mineru_output \
    --doc-ids 2401.00001 2401.00002 2401.00003

# 放宽M4要求（不强制四个维度全满足）
python scripts/generate_m4_queries.py --input data/mineru_output --relaxed

# 使用OpenAI
python scripts/generate_m4_queries.py \
    --input data/mineru_output \
    --provider openai \
    --model gpt-4o-mini
```

### 方式4：输入arXiv编号，直接下载其引用文献PDF（新）

```bash
python scripts/download_references_by_arxiv.py \
    --arxiv-id 2501.09959 \
    --output data/referenced_pdfs
```

可选参数：

- `--max-references`: 限制处理的引用数量（默认全部）
- `--min-citations`: 只保留最小引用数以上的文献
- `--api-key`: Semantic Scholar API key（可提升速率限制）

脚本会优先尝试 `arXiv PDF`、`openAccessPdf`，并回退到 `doi.org` 跳转链接；最终在输出目录写入 `reference_download_report.json` 记录每篇文献的下载状态。

### 方式5：在 Slurm 上执行完整任务（推荐集群）

```bash
# 1) 提交完整流程（下载引用 -> 解析 -> 生成M4）
./slurm_scripts/submit_all.sh --arxiv-id 2501.09959

# 2) 控制下载规模
./slurm_scripts/submit_all.sh \
    --arxiv-id 2501.09959 \
    --max-references 300 \
    --min-citations 3

# 3) 只跑解析与生成（跳过下载）
./slurm_scripts/submit_all.sh --skip-download
```

如果你想单独提交下载任务，也可以直接：

```bash
sbatch \
  --export=ALL,ARXIV_ID=2501.09959,MAX_REFERENCES=200,MIN_CITATIONS=0 \
  slurm_scripts/01_fetch_references.sh
```

常用检查命令：

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
```

### 方式2：完整Pipeline

```bash
# 处理200篇文档
python scripts/run_pipeline.py --target-docs 200

# 使用自定义配置
python scripts/run_pipeline.py --config configs/config.yaml --target-docs 200

# 跳过下载，使用已有PDF
python scripts/run_pipeline.py --skip-download

# 跳过下载和解析，仅重新生成Query
python scripts/run_pipeline.py --skip-download --skip-parse
```

### 方式3：分阶段执行

```bash
# 仅下载PDF
python scripts/download_only.py --count 200 --categories cs.CL cs.CV

# 仅解析PDF
python scripts/parse_only.py --input ./data/raw_pdfs --output ./data/mineru_output
```

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

### M4 Query输出格式

```json
{
  "query_id": "m4_abc123",
  "query_type": "full_m4",
  "turns": [
    "BERT模型在SQuAD数据集上的F1分数是多少？",
    "它与GPT相比有什么优势？"
  ],
  "answer": "BERT在SQuAD上达到88.5% F1，主要优势在于双向编码...",
  "evidence_chain": {
    "nodes": [
      {"doc_id": "doc_a", "modal_type": "table", "content_snippet": "..."},
      {"doc_id": "doc_b", "modal_type": "text", "content_snippet": "..."}
    ],
    "reasoning_steps": [
      "从doc_a的表格获取BERT的F1分数",
      "通过BERT实体桥接到doc_b",
      "从doc_b的文本获取与GPT的比较"
    ],
    "modalities": ["table", "text"],
    "docs": ["doc_a", "doc_b"]
  },
  "validation": {
    "is_multi_hop": true,
    "is_multi_modal": true,
    "is_multi_doc": true,
    "is_multi_turn": true,
    "satisfies_full_m4": true
  }
}
```

### 对比学习Triplet格式

```json
{
  "query": "该方法在PubMed数据集上的F1分数是多少？",
  "query_type": "factual",
  "positive": {
    "text": "| 方法 | 数据集 | F1 |\n|------|--------|----|\n| Ours | PubMed | 0.89 |",
    "modal_type": "table",
    "image_path": "mineru_output/doc_001/images/table_1.png"
  },
  "negatives": [
    {
      "text": "| 方法 | 数据集 | F1 |\n|------|--------|----|\n| Baseline | PubMed | 0.75 |",
      "modal_type": "table",
      "negative_type": "hard_same_modal"
    }
  ],
  "difficulty_score": 0.7
}
```

## 目录结构

```
data-process-test/
├── configs/
│   └── config.yaml              # 主配置文件
├── data/
│   ├── raw_pdfs/                # 下载的PDF
│   ├── mineru_output/           # MinerU解析输出
│   ├── m4_queries/              # M4 Query输出
│   ├── contrastive_data/        # 最终数据集
│   └── checkpoints/             # Pipeline检查点
├── docs/
│   └── M4_RESEARCH_NOTES.md     # M4研究笔记
├── logs/                         # 执行日志
├── scripts/
│   ├── run_pipeline.py          # 完整Pipeline入口
│   ├── generate_m4_queries.py   # M4 Query生成脚本
│   ├── download_only.py         # PDF下载脚本
│   ├── parse_only.py            # 解析脚本
│   ├── download_references_by_arxiv.py  # 按arXiv拉取引用PDF（新）
│   └── standardize_image_names.py
├── src/
│   ├── parsers/                 # PDF下载和解析
│   │   ├── mineru_parser.py
│   │   ├── modal_extractor.py
│   │   ├── pdf_downloader.py
│   │   └── reference_pdf_collector.py
│   ├── generators/              # Query生成
│   │   ├── query_generator.py   # 基础Query生成
│   │   └── m4_query_generator.py # M4增强生成
│   ├── linkers/                 # 跨文档关联（新）
│   │   └── cross_document_linker.py
│   ├── samplers/                # 负例采样
│   │   └── negative_sampler.py
│   ├── utils/                   # 工具函数
│   └── pipeline.py              # 主Pipeline
├── requirements.txt
└── README.md
```

## M4模块说明

### CrossDocumentLinker

跨文档实体关联模块，负责：
- **实体提取**：从文档中提取方法、数据集、指标、任务等实体
- **跨文档链接**：基于名称相似度建立文档间实体关联
- **Evidence Chain构建**：构建带有桥接实体的推理链

```python
from src.linkers import CrossDocumentLinker

linker = CrossDocumentLinker()
# 提取实体
entities = linker.build_document_entities(passages, doc_id)
# 查找跨文档链接
links = linker.find_cross_document_links()
# 构建Evidence Chain
chain = linker.build_evidence_chain(passages, bridge_entities)
```

### M4QueryGenerator

增强的Query生成器，特点：
- 基于实体关联选择passage组合
- 分步生成：桥接问题→多轮转换→完整M4
- 内置Evidence验证

```python
from src.generators import create_m4_generator

generator = create_m4_generator(provider="anthropic")
queries = generator.generate_queries_for_passages(
    passages,
    num_queries=10,
    require_full_m4=True
)
```

## 性能预估

### 处理200篇文档

| 阶段 | 时间 | 说明 |
|------|------|------|
| PDF解析 | 2-4小时 | 4x A2000并行 |
| M4 Query生成 | 30-60分钟 | 取决于API速率 |

### 输出规模

- 约6,000个对比三元组
- 模态分布：表格~40%、图片~30%、公式~20%、文本~10%

### API成本估算

| 提供商 | 模型 | 6000 Queries成本 |
|--------|------|-----------------|
| Anthropic | claude-sonnet | ~$10-15 |
| OpenAI | gpt-4o-mini | ~$3-4 |

## 故障排除

### MinerU未找到

```bash
pip install mineru[all]
# 或使用conda
conda install -c conda-forge mineru
```

### CUDA内存不足

- 减少配置中的 `num_workers`
- 使用 `backend: "pipeline"` 回退到CPU

### API速率限制

- 调整 `rate_limit` 配置
- 使用 `--dry-run` 先测试

### 无法生成跨文档Query

- 确保有至少2个文档
- 检查文档是否属于同一领域（共享实体）
- 尝试 `--relaxed` 放宽要求

## 学术参考

本项目的M4实现参考了以下研究：

- **M4DocBench** (2025): 多模态文档深度研究基准
- **TRACE** (EMNLP 2024): 知识三元组推理链构建
- **CoQA**: 对话式问答数据集
- **HiRAG**: 层次化检索增强生成

详细研究笔记见 `docs/M4_RESEARCH_NOTES.md`

## License

MIT License
