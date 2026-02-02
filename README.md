# MinerU Multimodal Contrastive Learning Data Factory

A production-grade pipeline for generating multimodal contrastive learning data from PDF documents. Designed for deployment on NTU EEE Cluster with 4x A2000 GPUs.

## Overview

This pipeline automates the process of:
1. **PDF Download**: Fetch papers from arXiv (or use local PDFs)
2. **Document Parsing**: Extract text, tables, figures, and formulas using MinerU
3. **Modal Extraction**: Classify and organize content by modality
4. **Query Generation**: Generate diverse queries using LLMs (GPT-4o-mini)
5. **Negative Sampling**: Construct hard negatives for contrastive learning
6. **Dataset Output**: Export in JSONL format for training

## Architecture

```
PDF Source → [Downloader] → MinerU Parser → Modal Splitter → Passage Builder
                                                                    ↓
JSONL Output ← Negative Sampler ← Query Generator (LLM) ←──────────┘
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPUs (4x A2000 recommended)
- MinerU installed (`pip install mineru[all]`)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd data-process-test

# Install dependencies
pip install -r requirements.txt

# Install MinerU separately (has complex dependencies)
pip install mineru[all]

# Set up API keys (for query generation)
export OPENAI_API_KEY="your-api-key"
```

## Usage

### Full Pipeline

```bash
# Process 200 documents from scratch
python scripts/run_pipeline.py --target-docs 200

# Use custom configuration
python scripts/run_pipeline.py --config configs/config.yaml --target-docs 200
```

### Individual Stages

```bash
# Download PDFs only
python scripts/download_only.py --count 200 --categories cs.CL cs.CV

# Parse PDFs only (using existing downloads)
python scripts/parse_only.py --input ./data/raw_pdfs --output ./data/mineru_output

# Skip download, use existing PDFs
python scripts/run_pipeline.py --skip-download

# Skip download and parse, regenerate queries
python scripts/run_pipeline.py --skip-download --skip-parse
```

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
# MinerU Parser Settings
mineru:
  backend: "auto"  # auto, pipeline, hybrid, vlm
  devices: ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
  num_workers: 4

# Query Generation
query_generation:
  provider: "openai"
  model: "gpt-4o-mini"
  queries_per_element: 3

# Negative Sampling
negative_sampling:
  strategy: "modal_mixed"  # random, modal_same, modal_mixed, semantic_hard
  num_negatives: 3
```

## Output Format

The pipeline generates JSONL files with the following structure:

```json
{
  "query": "What is the F1 score of the proposed method on PubMed?",
  "query_type": "factual",
  "positive": {
    "text": "| Method | Dataset | F1 Score |\n|--------|---------|----------|\n| Ours | PubMed | 0.89 |",
    "modal_type": "table",
    "image_path": "mineru_output/doc_001/images/table_1.png",
    "metadata": {"rows": 3, "cols": 3}
  },
  "negatives": [
    {
      "text": "| Method | Dataset | F1 Score |\n|--------|---------|----------|\n| Baseline | PubMed | 0.75 |",
      "modal_type": "table",
      "negative_type": "hard_same_modal"
    }
  ],
  "difficulty_score": 0.7
}
```

## GPU Parallelization

The pipeline automatically distributes work across available GPUs:

- **4x A2000**: ~50 documents/hour for parsing
- Each GPU processes documents independently
- Configurable via `mineru.devices` and `mineru.num_workers`

## Directory Structure

```
data-process-test/
├── configs/
│   └── config.yaml           # Main configuration
├── data/
│   ├── raw_pdfs/             # Downloaded PDFs
│   ├── mineru_output/        # Parsed document outputs
│   ├── contrastive_data/     # Final dataset
│   └── checkpoints/          # Pipeline checkpoints
├── logs/                      # Execution logs
├── scripts/
│   ├── run_pipeline.py       # Main entry point
│   ├── download_only.py      # PDF download script
│   └── parse_only.py         # Parse-only script
├── src/
│   ├── parsers/              # PDF download and parsing
│   ├── generators/           # Query generation
│   ├── samplers/             # Negative sampling
│   ├── utils/                # Utilities
│   └── pipeline.py           # Main pipeline
├── requirements.txt
└── README.md
```

## Expected Output

For 200 documents:
- ~6,000 contrastive triplets
- Modal distribution: Table ~40%, Figure ~30%, Formula ~20%, Text ~10%
- Processing time: ~2-4 hours (with 4x A2000)

## Cost Estimation

- **GPU time**: ~20 hours on A2000 for 200 documents
- **API costs**: ~$3.60 for GPT-4o-mini (6,000 queries)

## Troubleshooting

### MinerU not found
```bash
pip install mineru[all]
# Or use conda
conda install -c conda-forge mineru
```

### CUDA out of memory
- Reduce `num_workers` in config
- Use `backend: "pipeline"` for CPU fallback

### API rate limits
- Adjust `rate_limit` in query_generation config
- Use local LLM with `provider: "local"`

## License

MIT License
