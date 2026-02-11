# Project Context for Claude Code

## 项目简介
这是一个 M4（Multi-hop, Multi-modal, Multi-document, Multi-turn）Query 生成系统，用于训练多模态文档检索 embedding。

## 当前状态（2026-02-11 更新）

### 已完成
- 85 篇 arXiv 论文下载（种子论文：1908.09635）
- 80 篇 PDF 用 MinerU 解析
- **Step 0: Figure-text association** — 351 pairs, 73 docs（`src/linkers/figure_text_associator.py`）
- **Step 1: L1 intra-document cross-modal queries** — 经 3 轮迭代，最终 **974 条 queries**
- **L1 Triage** — A:727 (74.6%) / B:247 (25.4%) / C:0 (0%)  *(after visual_density gate)*
- **L2 候选构建** — 55 个跨文档实体，711 个候选文档对，top-100 已输出
- **Step 2: L2 cross-document queries** — 经 3 轮迭代
  - v1: 50 条, 100% QC pass (QC 过松), $0.55
  - v2: 32 条, 16 QC pass (严格 QC 但有 anchor leakage), $0.48
  - v3: **脚本已就绪，待执行** (prompt 重写 + anchor leakage QC)

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

- **Step 2: L2 cross-document queries** — 候选对已就绪，待调 Claude API 生成 top-50
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
| `data/figure_text_pairs.json` | 351 figure-text pairs (Step 0 输出) |
| `data/l1_cross_modal_queries_v3.jsonl` | **最终输出：974 条 L1 queries** |
| `data/l1_triage_v3.jsonl` | **L1 分拣结果（含 triage/reasons 字段）** |
| `data/l1_triage_report_v3.json` | L1 分拣统计报告 |
| `data/l2_candidate_pairs_v1.json` | L2 候选文档对 top-100 (v1, 含 generic entities) |
| `data/l2_candidate_pairs_v2.json` | **L2 候选文档对 43 对 (v2, filtered)** |
| `data/l2_queries_v1.jsonl` | L2 跨文档 queries 50 条 (v1, QC 过松) |
| `data/l2_queries_v2.jsonl` | L2 跨文档 queries 32 条 (v2, 16 QC pass) |
| `data/l2_queries_v2_tagged.jsonl` | L2 v2 reviewer-tagged (keep/fix/drop) |
| `data/l2_queries_v3.jsonl` | **L2 v3 输出 (待生成)** |
| `data/figure_descriptions_v3_api.json` | 完整 API 返回（含 raw response） |
| `data/validation_report_v3.json` | Validation 报告 |
| `docs/L1_query_iteration_report.md` | 迭代改进报告（含 L1 triage + L2 候选） |
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
- **没有 LaTeX 源码**（repo 里无 .tex/.bbl），只能用 MinerU markdown，但交叉引用正则已足够
- Step 0 `_classify_figure` 没用大模型看图，分类粗糙；Step 1 才真正用 Claude/Qwen-VL 看了图片
- "fairness" 出现在 45% 文档中（种子论文 1908.09635 是算法公平性方向），已被 IDF 过滤

## 下一步 TODO（更新后）
- **L1 引用图构建**：正则提取 Fig N/Table N/Eq N 引用 → 文档内 DAG → 2-hop 路径
- **L1 模态补全**：table-aware prompt + formula-aware prompt 生成缺失模态的 query
- **L2 试产**：对 top-50 候选对调 Claude API 生成跨文档 queries（~$2-5）
- **评估闭环**：人工写 30 条测试 query + BM25 baseline + Recall@10/MRR
- L3: multi-hop queries（基于 L2 桥接图 + 文档内引用图找 2-hop 路径）
- 详见 `docs/DISCUSSION_LOG.md` 最新讨论



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
