# MinerU 跨文档视觉链接管线 — Baseline 快照（2026-05-20）

延续 session "Complete CLIP-based cross-document visual linking pipeline"，把之前
中断的待办全部跑通并锁定一版基线。语料：53 个 MinerU 解析的 arXiv 文档。
运行环境：conda env `glm46v_py310`（torch 2.8 CPU、open_clip、sentence-transformers 5.4）。

## 1. 本次相对上一版的改动

| 改动 | 文件 | 说明 |
|---|---|---|
| 公式 embedding 换后端 | `build_mineru_vl_edges.py` | 新增 `--formula-backend math_similarity`（`math-similarity/Bert-MLM_arXiv-MP-class_arXiv`，768 维）。CLIP text 对公式区分度太差（probe p50=0.966 / std=0.027），math 模型 p50=0.817 / std=0.172。公式阈值按新分布从 0.45 自动提到 0.85。 |
| generic caption 检测增强 | `rerank_mineru_crossdoc_vl_edges.py` | 除 "Figure 9" 类裸编号外，新增 placeholder/marker/icon/logo/arrow 等无语义小图的识别，自动降权。 |
| rerank 升为建图必经步 | `build_mineru_crossdoc_bridges.py` | 跨文档边在建图时即带 `support_tier` 和 rerank 置信度；`visual_only_risky` 默认丢弃，`--keep-risky` 可 opt-in 给离线审核。 |
| 回归驱动 | `run_mineru_pipeline_regression.sh` | VL → rerank → bridges → hub 一键串跑 + 快照。 |
| 同文档 A/B | `audit_latex_vs_mineru_intradoc.py` | 新增 LaTeX `\ref` 硬边 vs MinerU `regex_reference` 的同文档对照。 |
| 阈值可移植性 | `audit_rerank_threshold_portability.py` | split-half 稳定性代理检查。 |

## 2. 锁定的产物（latest 指向的固定时间戳目录）

```
mineru_vl_edges_v1             -> mineru_vl_edges_v1_20260520T020257Z
mineru_crossdoc_text_rerank_v1 -> mineru_crossdoc_text_rerank_v1_20260520T023724Z
mineru_crossdoc_bridges_v1     -> mineru_crossdoc_bridges_v1_20260520T023730Z
mineru_hub_candidates_v1       -> mineru_hub_candidates_v1_20260520T023733Z
latex_vs_mineru_intradoc       -> latex_vs_mineru_intradoc_20260520T021731Z
rerank_threshold_portability   -> rerank_threshold_portability_20260520T023822Z
```

## 3. 边集合快照

**VL 边**（backend=open_clip，formula_backend=math_similarity）
- cross_doc_visual_sim: 3238
- visual_similarity: 2520
- text_describes_figure / figure_described_by_text: 2703 / 2703
- formula_similarity: 4331（math_similarity@0.85；公式 embedding 768 维）

**rerank tiers**（3238 条 cross-doc 视觉边）
- strong_text_supported: 587
- strong_enriched_supported: 64
- text_supported_candidate: 970
- weak_text_support: 1431
- visual_only_risky: 186  ← 默认过滤
- generic-caption-both @ top100: **0**（纯 CLIP 时是 72%）

**cross-doc bridges**（已默认丢弃 186 条 risky）
- crossdoc 边: 4741（含 section/paragraph/visual）
- sentence bridges: 909
- VL alignments: 2703
- orphan visual nodes: 19

**hub candidates**: 100 hubs / 500 candidates / cross-doc 137

## 4. 核心评价：MinerU 能否替代 LaTeX 文档内硬边？

同文档 A/B，52 个同时有 `.tex` 源和 MinerU 解析的文档（注：之前 session 报 overlap=0
是因为比对的是另一语料的 `latex_reference_graph_v2`；原始 `.tex` 实际 52/53 都在）：

| 指标 | 值 | 含义 |
|---|---:|---|
| 图/表抽取召回 | **90.8%** | LaTeX 定义的 511 图表，MinerU 抽到并对上 464 个 |
| 引用边召回 | **84.0%** | LaTeX `\ref` 的图表里，MinerU `regex_reference` 也连上 84% |
| 逐文档中位数 | **1.0 / 1.0** | 26/52 文档引用召回满分 |
| MinerU 独占引用 | 139 条 | LaTeX 没 `\ref` 但 MinerU 连了（隐式引用 + 部分噪声）|

**结论**：可行。MinerU 路线在同文档上能恢复 ~84% 的 LaTeX 硬引用边、~91% 的图表，
给"以后大量非 LaTeX PDF"一个明确预期下限。两个低分离群点（1607.06520 抽取 0.455、
1709.02012 引用 0.462）主因是 caption 文本差异导致跨解析对齐失败，非 MinerU 漏解析。

## 5. 阈值可移植性

verdict = **marginal**。
- 评分尺度可移植：各分数字段中位数 A/B 两半 delta ≤ 0.041。
- tier 边界会漂移：strong/weak 比例两半差 ±0.14（两半视觉相似度分布不同）。
- 实践含义：上新 PDF 批次时 `combined_score` 绝对阈值（0.45）可直接复用，但
  strong/weak 的相对占比需复核一眼。
- 局限：本语料仅 1/53 是纯 PDF（1805.03677），无法做真·LaTeX-vs-PDF 检验，
  此处是 split-half 稳定性代理。

## 6. 复跑方式

```bash
PY=/projects/_hdd/myyyx1/envs/glm46v_py310/bin/python
cd /projects/myyyx1/data-process-test
# 全量重建（含 VL，CPU 上慢）：
bash experiments/run_mineru_pipeline_regression.sh --rebuild-vl
# 仅重跑下游（复用现有 VL 边）：
bash experiments/run_mineru_pipeline_regression.sh
# 单独 A/B 与可移植性：
$PY experiments/audit_latex_vs_mineru_intradoc.py
$PY experiments/audit_rerank_threshold_portability.py
```

## 7. 后续可做（非阻塞）

- caption 跨解析对齐用 figure 序号 + 文本双信号，提升 A/B 匹配率（当前低分离群点主因）。
- 攒一批真·纯 PDF 文档后，把可移植性从 split-half 代理换成真 LaTeX-vs-PDF。
- `formula_similarity` 现已用 math 模型，可再加一层公式符号集合 Jaccard 复核高分边。
