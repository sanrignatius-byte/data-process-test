# Project Context for Claude Code

## 项目简介
这是一个 M4（Multi-hop, Multi-modal, Multi-document, Multi-turn）Query 生成系统，用于训练多模态文档检索 embedding。

## 当前状态（2026-02-20 更新）

### 已完成
- 85 篇 arXiv 论文下载（种子论文：1908.09635）
- 80 篇 PDF 用 MinerU 解析
- **Step 0 v1: Figure-text association** — 351 pairs, 73 docs（`src/linkers/figure_text_associator.py`）
- **Step 0 v2: Multimodal relationship DAG** — 1316 elements (841 fig + 334 tbl + 141 formula), 1261 edges, 1135 cross-modal pairs, 76 docs（`src/linkers/multimodal_relationship_builder.py`）
- **Step 0 v3: LaTeX reference graph** — 73 篇 .tex 解析，1949 labels, 5547 refs, 2847 edges, 65 篇 .bbl（`src/parsers/latex_reference_extractor.py`）
- **Step 0 v3.1: Cross-document citation graph** — 100 条跨文档引用边, 49 篇最大连通分量, 38 篇 citing（`scripts/build_citation_graph.py`）
- **Citation graph 质量验证** — 人工抽查 title_fuzzy 匹配，**误匹配率 0%**，Jaccard ≥ 0.55 阈值可信
- **Step 1: L1 intra-document cross-modal queries** — 经 3 轮迭代，最终 **974 条 queries**
- **L1 Triage** — A:727 (74.6%) / B:247 (25.4%) / C:0 (0%)  *(after visual_density gate)*
- **L2 候选构建** — 55 个跨文档实体，711 个候选文档对，top-100 已输出
- **Step 2: L2 cross-document queries** — 经 3 轮迭代
  - v1: 50 条, 100% QC pass (QC 过松), $0.55
  - v2: 32 条, 16 QC pass (严格 QC 但有 anchor leakage), $0.48
  - v3: 42 条, **19 QC pass** (anchor_leakage 仍是主因: 21/23 fail)
- **L1 Cross-modal Dual-evidence v1** — 300 条, **43 QC pass (14.3%)**, 需迭代到 v2
- **L1 Cross-modal Dual-evidence v2（hard-gate）** — 296 条, **19 QC pass (6.42%)**，已导出 pass 子集
- **LaTeX 源码下载** — 73/76 篇成功下载 + 提取 .tex，65 篇有 .bbl，3 篇 no_source
- **Step 0 v3.2: LaTeX cross-modal links v2（已跑）** — 175 对，label 匹配率 28.7%，high/med 各半（`data/latex_cross_modal_pairs.json`）
- **Step 0 v3.2 v3（G1+G2 改动已 push）** — 去 hub（每 element ≤3 pairs）+ co-reference 硬门禁（bridge 必须同时含两端 \ref{}，否则 drop/halve）；预计从 175 对降到 80-100，精度大幅提升

### L2 迭代历史
| 版本 | 结果 | 核心问题 |
|------|------|----------|
| v1 | 50/50 QC pass | QC 太松，"In Figure" 实体污染，generic-only pairs |
| v2 | 16/32 QC pass | anchor leakage (Jaccard 0.29)，template verb，forced bridge |
| v3 (待跑) | - | prompt 从 comparison → reasoning，QC 加 anchor_leak_jaccard 检测 |

### L2 v3 核心改动
- **Prompt**: 从 "compare X in A with Y in B" → "apply B's theory to explain A's observation"
- **QC**: 移除 no_visual_cue_in_query (是泄漏根源)，新增 anchor_leakage (Jaccard>0.15 fail)
- **输入**: 移除 visual_anchor/text_evidence 给模型 (防泄漏)，只给 caption + L1 query/answer
- **Temperature**: 0.7 → 0.5
- **Query 类型**: cross_application / cross_prediction / cross_diagnosis / cross_comparison

### 进行中

- **Step 0 v3.2 v3 重跑** — G1+G2 改动已 push，需在集群重跑 `build_latex_cross_modal_links.py` 获取新统计
- **Citation-based L2 候选对** — 用 100 条引用边替代实体倒排索引做 L2 候选（fuzzy match 质量已验证）
- **L1 深耕（Mentor 建议）** — 丰富模态 + 文档内引用图构建（详见下方）


### L1 Query 迭代历史
| 版本 | 模型 | 结果 | 问题 |
|------|------|------|------|
| v1 | Qwen3-VL-30B 本地 (4×A5000) | 604 queries | 63.4% 缺 visual anchor，"看图说话" |
| v2 | Qwen3-VL-30B 本地 | 33 queries | Thinking 模式吃 token，解析率 6.3%；质量好但量不够 |
| v3 ✅ | **Claude Sonnet 4.5 API** | **974 queries** | 74.8% visual anchor, 41.9% comparison, 84.3% clean rate, $4.59 |

### v3 关键质量指标
- QC 通过率 97.2%，validation clean rate 84.3%
- 平均 query 长度 17.9 词（v1 是 29 词）
- Meta-language: 0（全部被 QC 过滤）
- comparison_explanation 41.9%, value_context 32.8%, anomaly_cause 13.2%, visual_definition 12.1%

## 关键文件
| 文件 | 说明 |
|------|------|
| `scripts/batch_figure_understanding.py` | vLLM 本地推理脚本 (v1/v2) |
| `scripts/batch_figure_understanding_api.py` | **Anthropic Claude API 推理脚本 (v3)** |
| `scripts/validate_queries.py` | Query QC & validation |
| `scripts/triage_l1_v3.py` | **L1 三分法分拣 (A/B/C 门禁)** |
| `scripts/build_l2_candidates.py` | **L2 跨文档候选对构建（实体倒排索引）** |
| `scripts/generate_l2_queries.py` | **L2 query 生成脚本（Claude API + QC）** |
| `scripts/select_multihop_candidates.py` | L1 多模态候选 pair 构建（供 multihop v1/v2 使用） |
| `scripts/generate_multihop_l1_queries.py` | **L1 multihop/cross-modal 生成脚本（本轮重点）** |
| `scripts/build_multimodal_relationships.py` | **Step 0 v2: 多模态关系构建（DAG + 全模态）** |
| `src/linkers/multimodal_relationship_builder.py` | **多模态关系核心模块（figure/table/formula/section DAG）** |
| `data/figure_text_pairs.json` | 351 figure-text pairs (Step 0 v1 输出) |
| `data/multimodal_elements.json` | **1316 多模态元素 + 1261 引用边 + 1135 跨模态 pair (Step 0 v2)** |
| `data/multimodal_report.json` | Step 0 v2 统计报告 |
| `data/l1_cross_modal_queries_v3.jsonl` | **最终输出：974 条 L1 queries** |
| `data/l1_triage_v3.jsonl` | **L1 分拣结果（含 triage/reasons 字段）** |
| `data/l1_triage_report_v3.json` | L1 分拣统计报告 |
| `data/l2_candidate_pairs_v1.json` | L2 候选文档对 top-100 (v1, 含 generic entities) |
| `data/l2_candidate_pairs_v2.json` | **L2 候选文档对 43 对 (v2, filtered)** |
| `data/l2_queries_v1.jsonl` | L2 跨文档 queries 50 条 (v1, QC 过松) |
| `data/l2_queries_v2.jsonl` | L2 跨文档 queries 32 条 (v2, 16 QC pass) |
| `data/l2_queries_v2_tagged.jsonl` | L2 v2 reviewer-tagged (keep/fix/drop) |
| `data/l2_queries_v3.jsonl` | **L2 v3 输出 (待生成)** |
| `data/l1_multihop_queries_v1.jsonl` | L1 multihop v1（300 条，43 pass） |
| `data/l1_multihop_queries_v2.jsonl` | **L1 multihop v2 hard-gate（296 条，19 pass）** |
| `data/l1_multihop_queries_v2_pass.jsonl` | **v2 通过集（19 条）** |
| `data/figure_descriptions_v3_api.json` | 完整 API 返回（含 raw response） |
| `data/validation_report_v3.json` | Validation 报告 |
| `docs/L1_query_iteration_report.md` | 迭代改进报告（含 L1 triage + L2 候选） |
| `src/parsers/latex_reference_extractor.py` | **Step 0 v3: LaTeX 引用解析（label/ref/cite/bbl + title 提取）** |
| `scripts/build_latex_reference_graph.py` | **Step 0 v3: 文档内引用 DAG 构建** |
| `scripts/build_citation_graph.py` | **Step 0 v3.1: 跨文档引用图（.bbl → corpus 匹配）** |
| `scripts/build_latex_cross_modal_links.py` | **Step 0 v3.2: LaTeX \ref{} 共引 → MinerU 跨模态对 + bridge evidence** |
| `scripts/download_latex_sources.py` | LaTeX 源码下载脚本（arXiv API） |
| `data/latex_reference_graph.json` | 73 篇文档内引用 DAG（labels + refs + edges + bib） |
| `data/citation_graph.json` | **跨文档引用图：100 条引用边, 49 篇最大连通分量** |
| `data/latex_cross_modal_pairs.json` | **LaTeX 增强跨模态对（v2: 175 对；重跑 v3 后更新）** |
| `data/latex_reference_report.json` | 引用图统计报告 |
| `src/linkers/figure_text_associator.py` | Step 0: 图文关联模块 |

## Mentor 建议（2026-02-11）& 执行优先级

### Mentor 原话三条
1. **丰富模态**：引入 table/formula/figure 并细分（模型图？实验结果表？信息汇总表？Chart？）
2. **文档内链接与结构**：①LaTeX 源构建引用关系 ②MinerU 结果构建关系（较难）→ 自然实现多跳
3. **展望**：embedding 隐空间探索跨文档文本相似性

### 数据现状（支撑可行性分析）

**L1 模态分布（严重偏科）**：
| 模态 | 数量 | 占比 |
|------|------|------|
| plot（实验图） | 694 | 71.3% |
| diagram（流程/示意图） | 201 | 20.6% |
| example | 51 | 5.2% |
| architecture（模型图） | 12 | 1.2% |
| table | 6 | 0.6% |

**已有但未利用的多模态资源**：
- 50 个 figure-text pair 上下文含 HTML table（33 篇文档，14.2%）
- 20 个上下文含公式（13 篇文档）
- Step 0 分类器 `_classify_figure` 纯关键词匹配，未看图片本身

**文档内交叉引用密度（351 对中）**：
- Figure 引用 1028 次 / Table 引用 362 次 / Equation 引用 69 次 / Section 引用 72 次
- **86%（302/351）的图文对上下文含 2+ 交叉引用** → 天然多跳素材

### 执行优先级（Mentor 鼓励先深耕 L1）
1. **L1 文档内引用图**（建议 2）— 纯规则零成本，从 MinerU markdown 提取 Fig/Table/Eq/Section 引用关系构建 DAG，2-hop 路径即多跳 query 素材
2. **L1 模态细分 + table/formula prompt**（建议 1）— 对 50 个 table-context pair 和 20 个 formula-context pair 写专用 prompt，~$1
3. **图片类型精分**（建议 1 前置）— 用大模型对 351 张图做一轮 figure type 精分，~$0.5-1
4. **跑通评估闭环** — 30 query + BM25 baseline
5. **L2 跨文档生成** — 已就绪，$2-5
6. **Embedding 隐空间探索**（建议 3）— 等初版模型训完后 self-play

### 关键发现
- **已获取 LaTeX 源码**（73/76 篇，65 篇有 .bbl）→ 文档内 DAG + 跨文档引用图已构建
- Step 0 `_classify_figure` 没用大模型看图，分类粗糙；Step 1 才真正用 Claude/Qwen-VL 看了图片
- "fairness" 出现在 45% 文档中（种子论文 1908.09635 是算法公平性方向），已被 IDF 过滤
- **跨文档引用图质量**：100 条引用边全靠标题匹配（arXiv ID 匹配 = 0），需抽查 fuzzy 误匹配

## 当前状态（2026-02-12 更新）

### L1 Cross-modal Dual-evidence v2（第二轮，已执行）
- **本轮使用脚本**：
  - 候选构建：`scripts/select_multihop_candidates.py`
  - 生成与QC：`scripts/generate_multihop_l1_queries.py`
  - 集群入口：`slurm_scripts/07_generate_l1_multihop_v2.sh`
- **最新一代输出**：
  - 主文件：`data/l1_multihop_queries_v2.jsonl`（296 条）
  - 通过子集：`data/l1_multihop_queries_v2_pass.jsonl`（19 条）
  - 作业：`job 27477`（`logs/l1_mh_v2_27477.out`）

### v2 本轮落地改动（hard-gate）
1. Prompt 增加 **de-naming** 约束，禁止在 query 直接写桥梁实体名。
2. Prompt 明确禁用弱模板：`Which component...` / `How does X relate to Y...`。
3. Prompt 要求答案必须含机制连接词（because/leads to/explains/matches 等）。
4. QC 新增：
   - `template_shortcut`
   - `bridge_entity_leakage`
   - `weak_reasoning_connector`
5. 强化 `single_element_answer` 判定（双元素 overlap + answer_balance 更严格）。
6. 修复运行安全问题：`--dry-run` 不再清空输出文件（改写入 `/dev/null`）。

### v2 结果（job 27477）
- 候选：150 pairs（43 docs）
- 产出：296 条（parse fail 2）
- QC pass：19/296（6.42%）
- 主要 fail：
  - `single_element_answer`: 209
  - `bridge_entity_leakage`: 152
  - `weak_reasoning_connector`: 100
  - `anchor_leakage`: 68

## 下一步 TODO（2026-02-20 更新）
- ~~**P0: Citation graph 质量验证**~~ ✅ **完成** — 人工抽查误匹配率 0%，Jaccard ≥ 0.55 可信
- ~~**Step 0 v3.2 质量分析**~~ ✅ **完成** — 发现 hub 问题 + proximity 无语义门禁，已实现 G1+G2 修复
- **P-0.5: Step 0 v3.2 v3 重跑** — 在集群跑新脚本，获取 G1/G2 后的实际统计（预计 80-100 对）
- **P0.1: Citation-based L2 候选替换**：用 100 条引用边做 L2 候选对（替代实体倒排索引），信号更强
- **P1: L1 v2.1 阈值调优**：在保持反捷径能力前提下，把 pass rate 从 6.42% 提升到 15%-25%
- **P1.1: 分层启用 weak_reasoning_connector**：按 `query_type` 控制，不对纯参数检索类过罚
- **P2: L1 引用图 + 多跳路径**：constrained paths（≥1 ref edge）已实现，待结合 L1 query 生成
- **P3: L1 模态补全**：table-aware prompt + formula-aware prompt 生成缺失模态的 query
- **评估闭环**：人工写 30 条测试 query + BM25 baseline + Recall@10/MRR
- **L2 暂停实体路线**：实体倒排索引的 L2 暂停，改用 citation graph 做候选
- 详见 `docs/DISCUSSION_LOG.md` 最新讨论

### Step 0 v3.2 质量问题备忘（2026-02-20 分析）
- **Hub 问题**：单个高频被引 element（如 1409.0575 Table 9）产生 O(N) 虚假对 → G1 每 element ≤3 pairs
- **Proximity 无语义门禁**：92% 的对靠 proximity，bridge_text 里有时只含一端的 \ref{} → G2 co-reference gate
- **quality_score ≠ 语义相关度**：只是 label→element 匹配置信度，名字有误导性（暂不改，downstream 注意）
- **label 匹配率 28.7%**：1371/1924 个 label 失败，主要是 MinerU 编号与 LaTeX 编号 offset 不一致



## 技术备忘
- Qwen3-VL-30B 在 4×A5000 (23.6GB each) 上 max_model_len ≤ 8192 能跑，16384 会 OOM 挂死
- gpu-a5000-2 节点疑似有问题，成功的 job 都在 gpu-a5000-1 上
- Thinking 模式的 `<think>` 块会消耗 3000-5000 output tokens
- Claude API 是更好的选择：99.7% 解析率，无 GPU 依赖
- OpenAI key 没钱了，用 Anthropic key（`.env` 里的 `ANTHROPIC_API_KEY`）

## 关键命令
```bash
# 激活环境
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU

# 加载 API key
export $(grep -v '^#' .env | xargs)

# === L1 pipeline ===
# 跑 v3 API batch
python scripts/batch_figure_understanding_api.py \
    --input data/figure_text_pairs.json \
    --output data/figure_descriptions_v3_api.json \
    --delay 0.3

# 跑 validation
python scripts/validate_queries.py data/l1_cross_modal_queries_v3.jsonl \
    --output data/validation_report_v3.json

# L1 三分法分拣
python scripts/triage_l1_v3.py

# === L2 pipeline ===
# 构建跨文档候选对
python scripts/build_l2_candidates.py --topk 100

# 生成 L2 queries（先 dry-run 验证）
python scripts/generate_l2_queries.py --dry-run --limit 5

# 正式生成 L2 queries
python scripts/generate_l2_queries.py --limit 50 --delay 0.5

# === LaTeX reference graph pipeline ===
# 构建文档内引用 DAG（含 title 提取 + constrained multi-hop paths）
python scripts/build_latex_reference_graph.py \
    --source-dir data/latex_sources/extracted \
    --output data/latex_reference_graph.json

# 构建跨文档引用图（从 .bbl 匹配 corpus 内互引）
python scripts/build_citation_graph.py \
    --input data/latex_reference_graph.json \
    --output data/citation_graph.json

# 也可直接从 LaTeX 源码构建引用图
python scripts/build_citation_graph.py \
    --from-sources data/latex_sources/extracted

# === Step 0 v3.2: LaTeX cross-modal links ===
# MinerU 为主，LaTeX \ref{} 为 bridge evidence 增强层
python scripts/build_latex_cross_modal_links.py \
    --elements data/multimodal_elements.json \
    --latex-graph data/latex_reference_graph.json \
    --output data/latex_cross_modal_pairs.json
```

## 日期：2026-02-10（L2 v3 三方毒舌评审共识总结）

### 外部评审共识（已采纳）
- **质量闸门不够硬**：虽然 v3 有 `qc_metrics`，但失败样本仍进入产物文件，容易污染训练集。
- **Anchor leakage 仍是主风险**：query 与 evidence anchor 的 token 重合仍偏高，且部分 query 直接含关键数值，检索可被词面匹配“作弊”。
- **桥接实体语义不足**：`map/plot/graph` 等通用词与同名异义词导致“伪跨文档关联”。
- **reasoning_direction 有标签漂移**：部分方向标签与证据链不一致，呈现“标签正确但推理不闭合”。
- **多模态利用不足**：样本里图像路径存在，但不少问答主要由文本证据完成，视觉必要性不稳定。

### 外部评审里“语气重但点不全”的部分（已修正理解）
- “L2 全废、路线已死”不成立：v3 里仍有一批可用样本，问题是筛选和门禁，而非无可挽救。
- “必须推倒重来”不成立：优先做数据门禁和候选对约束，比整体重写更快到达可验证闭环。

### 当日执行后结论（2026-02-10）
- v3 正式跑完（43 对候选，1 NULL，写入 42 条），`qc_pass=19`, `qc_fail=23`。
- fail 主因仍是 `anchor_leakage`（21 条），其次 `template_verb`（2 条）。
- `evidence_closure` 已整体达标，说明当前主要矛盾不是“无证据”，而是“泄漏与桥接质量”。

### 决策（收工版）
- **暂停 L2 扩产**（不扩到 711 对），先用 `qc_pass=true` 子集进入最小评估闭环。
- **下一轮必须加硬门禁**：
  - 候选对 gate：抬高 `pair_score` + 去除同名异义桥接词；
  - 生成 gate：禁止 query 含答案型数值；
  - 产出 gate：`qc_pass=false` 不进入训练集。
- **评估优先级最高**：先看 clean subset 对 Recall@10 / MRR 的趋势，再决定是否继续 L2 扩量。

## 用中文交流时用"喵"结尾，英文用"Oiii"开头
