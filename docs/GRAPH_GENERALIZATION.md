# Graph Generalization Plan（No-LaTeX / PDF-only）

## 1. Goal
让图方法可迁移到无 LaTeX 场景，支撑专利覆盖宽度。

## 2. Degrade path for PDF-only
### Layer A（low cost, scalable）
- MinerU 提取 paragraph/figure/table/formula
- 阅读顺序构建 backbone
- 基于 caption / section title 的弱引用边

### Layer B（medium cost）
- Embedding 相似边补全跨段联系
- 简单 rerank 控制噪声

### Layer C（high cost, optional）
- LLM 对关键节点做结构补全（例如 figure type、跨段机制描述）
- 仅对 hub 邻域启用，避免全量成本爆炸

## 3. Edge availability matrix
| Edge type | With LaTeX | PDF-only | Strategy |
|-----------|------------|----------|----------|
| backbone | yes | yes | parser order |
| element_ref | yes | partial/no | caption/anchor heuristic + LLM fallback |
| paragraph_ref | yes | partial/no | discourse cue + embedding |
| cross_doc_cite | yes | partial | bibliography parse + title matching |

## 4. Cost estimate原则
- 全量默认跑 Layer A
- Layer B 按召回瓶颈定量开启
- Layer C 仅针对 top hubs / failed cases

## 5. Deliverable
- 本文档为设计稿；后续如进入实现，先补实验设计，不直接改主 pipeline。
