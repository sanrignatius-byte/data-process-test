# Graph Architecture v1

## 1. Objective
Document Graph for Document Understanding：支持 query 生成、QA、证据定位、多文档推理。

## 2. Node types
| Node | Source | Cost | Notes |
|------|--------|------|-------|
| Paragraph | MinerU / LaTeX | low | 基础文本单元 |
| Figure | MinerU | low | caption + image path |
| Table | MinerU | low | caption + html |
| Formula | MinerU | low | formula context |
| Section | LaTeX | low | section 结构 |
| Enriched element | LLM | medium/high | enriched_title/metadata/content |

## 3. Edge types（MVE locked）
| Edge | Source | Cost | Used in MVE |
|------|--------|------|-------------|
| backbone | parsing order | low | yes |
| element_ref | LaTeX `\\ref{}` | low | yes |
| paragraph_ref | paragraph references | low | yes |
| cross_doc_cite | `.bbl` + title match | low | yes |

## 4. Hub scoring（MVE locked）
- Bridge score（rule-based）
- PageRank（graph centrality）
- MVE 评估锁定为：`bridge_score + PageRank` 组合排序。

## 5. Build pipeline
1) Parse docs → nodes
2) Build edges
3) Compute hub scores
4) Retrieve evidence / evaluate against protocol

## 6. Complexity & cost layers
- Low: parsing + rule edges
- Medium: embedding / rerank
- High: LLM enrichment / optional graph augmentation

## 7. Output expectations
- 可追踪版本化（commit 历史）
- 与 `docs/EVAL_PROTOCOL_V1.md` 对齐，不随实验临时改定义。
