# Evaluation Protocol v1（Locked）

## 1) Scope
用于验证 Document Graph 检索是否优于 baseline。主任务为 evidence localization，辅任务为 QA。

## 2) Locked test set（禁止临时改切片）
- `data/l1_dual_evidence_queries_v4_4_run1_pass.jsonl`（113）
- `data/l1_dual_evidence_queries_v3_pass.jsonl`（152）
- `data111/l1_img_run_20.jsonl`（16，最新 enriched-hub run，3 qc_pass）
- 去重并集作为最终测试集（按 `query_id`，若缺失则按 query 文本哈希）。

## 2b) Corpus & Hub 版本（最新）
- Elements（用于 chunking）：`data111/multimodal_elements_enriched.json`（MoDora enriched）
- Hubs：`data111/latex_graph_hubs (1).json`

## 3) Ground-truth
- 使用 query 内 `required_evidence_spans` 作为证据标注。

## 4) Localization 判定规则
- 命中条件：retrieved chunk 与任一 `required_evidence_spans` 的**字符级 overlap ≥ 0.5**。
- 主指标：Recall@10、MRR。

## 5) Baselines
- BM25
- dense retrieval
- 要求：使用同一 chunking 规则与同一测试集。

## 6) Decision thresholds（预注册）
| 决策 | 条件 |
|------|------|
| 继续扩量 | Graph Recall@10 ≥ BM25 + 5%，或 MRR ≥ BM25 + 3% |
| 暂停扩量并回查 | 任一指标未达阈值 |
| 放弃当前 graph 配置 | 连续 2 轮实验均未达阈值 |

## 7) Governance
- 任何影响评估口径的改动（数据切片、chunking、QC 阈值、候选规则）必须记录到 `docs/EVAL_CHANGELOG.md`。
