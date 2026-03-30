# Project Context for Claude Code

## 运行环境默认配置

- **默认 provider**：`company`（yunwu.ai），所有脚本已改为默认走公司 API
- **API 配置**：已在 `.env` 中配好 `COMPANY_API_KEY` 和 `COMPANY_API_URL`，无需额外传参
- **`local_api_logger/`**：已放入项目根目录
- **运行方式**：直接 `python scripts/xxx.py` 即可，不需要加 `--provider` 或 `--company-api-key`
- **备选 provider**：如需切换，手动加 `--provider anthropic` 或 `--provider openai`

## 铁律（Iron Rules）— 所有开发必须遵守

### 铁律 1：Token 使用必须官方记录

**任何调用 LLM API 的脚本，结束时必须调用 `src.utils.token_logger.log_run()` 记录本次运行的 token 消耗。无例外。**

```python
# 必须在脚本顶部 import
from src.utils.token_logger import log_run

# 必须在脚本结束时调用（dry-run 除外，log_run 内部会自动跳过 0 token）
log_run(
    script="your_script_name",           # 脚本名，不含路径
    model=f"{provider}:{model}",          # provider:model 格式
    purpose="简述本次运行做了什么",         # 人可读
    input_tokens=total_in_tok,
    output_tokens=total_out_tok,
    extra={                               # 可选但强烈建议
        "pairs_processed": N,
        "qc_pass": M,
        "output": str(output_path),
    },
)
```

**合规检查清单**：
- `generate_multihop_l1_queries.py` ✅ 已接入
- `batch_figure_understanding_api.py` ✅ 已接入
- `generate_l2_queries.py` ✅ 已接入
- `enrich_elements_modora.py` ✅ 已接入（v1.1 补入）
- `run_exp_c_qa_triangle.py` ✅ 已接入
- `build_embedding_edges.py` — 不调用 LLM，无需接入
- `run_production_batch.py` — 包装脚本，内部调用 generate_multihop_l1_queries.py（已接入）
- **新增任何调用 LLM 的脚本时必须同步接入**

**违规判定**：任何发起 API 请求但未调用 `log_run()` 的 PR 视为未通过 review。

---

## 项目简介
这是一个以 **Document Graph for Document Understanding** 为核心的研究系统。核心创新是面向学术论文的多层异构图构建方法，支持多种下游任务（query 生成、QA、文档总结、多文档推理、证据定位）。M4 Query 生成（Multi-hop, Multi-modal, Multi-document, Multi-turn）是图的第一个应用示例，也是当前主要交付物。

**战略定位（2026-03-12 Mentor 确认）**：图是核心贡献，query 是副产物；图应具备泛化到非 LaTeX 文档的能力；计划 4 月申请专利（公司），之后开放论文投稿。

## 当前状态（2026-03-24 更新｜P0-P4 Bridge Grounding 增强 + L3 质量验证）

### 本轮完成（相对 2026-03-21）

- **L3 Query 质量诊断**：发现旧 L3 115 条全部 bridge_paragraph content 为空、reasoning_structure 100% parallel；根因是 `hub_candidates_enriched` 的 `edge_contexts` 全空，bridge 文本从未传入 prompt
- **P0-P4 五项增强实施**
  - **P0 Bridge 文本注入**：从 `latex_reference_graph.json` 提取边 context，通过 element_id→LaTeX label 映射（1317 个映射），实现 **209/230 pair (90.9%)** bridge 文本覆盖（之前 0%）
  - **P1 图路径编码**：重写 `PROMPT_3STEP_REASONING_CHAIN`，注入图路径描述 + bridge 原文 + 质量标签 + serial chain 强制示例 + bridge grounding rule
  - **P2 Bridge 质量评分**：`score_bridge_quality()` 基于动词密度/长度/公式比/引用标记评 0-1；HIGH 77, MEDIUM 97, LOW 35 pairs
  - **P3 Hub-aware QC**：bridge span 长度检查 + bridge claim 非空 + parallel L3 hard-fail + `pseudo_multihop_parallel` 从 L3 soft issues 中移除
  - **P4 Anchor 特异性**：visual_anchors 必须含具体位置标记（row/col/axis/marker），全 generic 则 fail
- **新增参数**：`--reference-graph data/latex_reference_graph.json`（默认自动加载）
- **测试批次运行**：40 pair 生成，13 pass (37%)；所有 40 条 reasoning_steps 都有正确依赖链
- **检索评测验证**：新 bridge-grounded L3 vs 旧空 bridge L3 全面提升

### Bridge Grounding 检索评测对比（n=40 each）

| Method | Old L3 R@10 | New L3 R@10 | Δ | Old L3 MRR | New L3 MRR | Δ |
|--------|-------------|-------------|---|------------|------------|---|
| bm25 | 0.925 | **0.975** | +0.050 | 0.597 | **0.733** | **+0.135** |
| graph_hub_rerank | 0.950 | **0.975** | +0.025 | 0.624 | **0.776** | **+0.152** |
| graph_neighbor_prop | 0.950 | **1.000** | **+0.050** | 0.708 | **0.861** | **+0.154** |
| graph_full | 0.950 | **0.975** | +0.025 | 0.682 | **0.803** | **+0.121** |

- **BM25 基线大幅提升**（MRR +0.135）：bridge grounding 让 query 使用论文实际术语，BM25 词面匹配更准
- **neighbor_prop 达到完美 R@10=1.000, MRR=0.861**：bridge-grounded 证据完全落在图的 1-hop 邻域内
- **Graph 增益绝对值依然显著**：graph_full MRR 0.803 >> 旧 0.682；相对增益略降是因为 BM25 基线本身变强
- **核心结论**：图结构信息（bridge 段落）注入 query 生成 prompt 后，query 与 evidence 之间的词面和结构对齐同时提升

### 之前的实验结果（保留参考）

**M2 三实验（2026-03-21）**
- Exp A: 难度梯度 — Coverage L1=0.971 > L2=0.610 > L3=0.617
- Exp B: graph_full R@10=0.8736(+0.0269), MRR=0.6045(+0.0403)
- Exp C: 图检索覆盖 +1.9%(L2)/+6.1%(L3)，QA mention -0.5%(L2)/-1.7%(L3)
- Enrichment 消融：Graph 零成本 MRR +0.018 ≈ Enrichment $3 MRR +0.013，合用 ×1.73 超线性

### 下一步

**支线 A（学校集群）：量产 query + L3 重跑**
- 目标：用 P0-P4 增强 pipeline 重跑全量 L3（121 个 good bridge pairs），替换旧 115 条
- 同时量产 L2：L2+L3 从 325 扩到 550+，总计 1500+ queries
- 脚本：`scripts/run_production_batch.py` + `--reference-graph`
- 成本：~$12-15

**支线 B（公司集群）：Embedding 语义边**
- 目标：验证 embedding 相似度能否补充图的边
- 脚本：`scripts/build_embedding_edges.py` + `run_phase0_eval_ab.py --embedding-edges`
- 不需要 LLM API，只需 sentence-transformers + GPU

### M4 路线图（更新）
| 阶段 | 目标 | 时间 |
|------|------|------|
| Phase 0 ✅ | 锁定 M1.5 基线 + 定义 M4 schema + reasoning-depth tagging | 已完成 |
| Phase 1 ✅ | M2 pipeline + L3 生成 + 三实验全量运行 | 已完成 |
| Phase 1.5 ✅ | Enrichment 消融实验 + Exp C enriched 复验 | 已完成 |
| **Phase 1.7 ✅** | **P0-P4 Bridge Grounding 增强 + L3 质量验证** | **2026-03-24 完成** |
| **Phase 2A ⏳** | **L3 全量重跑 + 量产 1500+ queries** → 初代 benchmark | 本周 |
| **Phase 2B ⏳** | **Embedding 语义边** → 图增强 v2 | 本周 |
| Phase 3 | 合并 2A+2B → 增强图 + 大数据集 → 最终实验 | 下周 |
| Phase 4 | Multi-turn session + M4 联合验证 | 1-2 周 |

---

## 当前状态（2026-03-18 更新｜M4 战略重定位 + Schema 设计 + Reasoning-depth Tagging）

### 本轮完成（相对 2026-03-16）

- **M4 战略重定位完成** — 诚实评估：当前为 M1.5（跨模态 + 伪多跳），非 M4
  - 项目对外口径重定义为 "Graph-backed Cross-modal Dual-evidence Benchmark (M4-Foundation)"
  - 详见 `docs/M4_STRATEGY_REVIEW_2026-03-18.md`
- **M4 三套数据 Schema 设计完成（Schema-ready，非 Generator-ready）** — `docs/M4_SCHEMAS.md`
  - Schema 1: Strict Multi-hop Reasoning Chain（`reasoning_steps[]` + `depends_on_steps` + `evidence_type`）
  - Schema 2: Element-level Cross-document Bridge（`bridge_type` + `bridge_evidence` + `confidence`）
  - Schema 3: Multi-turn Session（`turns[]` + `coreference_type` + `turn_dependency_qc`）
  - 三者关系：Schema 2 提供跨文档边 → Schema 1 在图上生成推理链 → Schema 3 将推理链 session 化
  - **注意**：当前生成脚本已支持 3-step native generator（`PROMPT_3STEP_REASONING_CHAIN`），dual-evidence pair 容器同时保留
- **Reasoning-depth 启发式标记已集成** — `qc_reasoning_depth()` in `generate_multihop_l1_queries.py`
  - `classify_reasoning_structure()`：用语言表面特征（连接词模式）区分 parallel vs serial，**适合 auto-tagging / profiling，不适合作为严格 M4 合格判定**
  - `m4_reasoning_depth`、`m4_reasoning_structure`、`m4_is_true_multihop` 新增到 QC metrics
  - 对现有 dual-evidence 数据为 advisory（不 hard fail），对新 Schema 1 显式 `reasoning_steps[]` 数据做结构验证（hard fail）
  - Step-deletion **proxy**（非真正 step-deletion test）：`causal_link_count ≥ min_depth - 1`，基于 answer 中因果连接词计数
  - **已知局限**：① 写作风格可欺骗（爱写 because/therefore 会被高估）；② 不同 query_style 的连接词分布不同导致不鲁棒；③ evidence_type 判别依赖 span 词面
  - **待做**：30-50 条人工标注误差审计（precision/recall），验证 heuristic 可信度
- **现有数据自动标记**：所有新生成 query 将自动携带 `reasoning_depth` 和 `reasoning_structure` 字段

### 本轮关键决策
- **当前 multi-hop 是"双证据并行取证"而非"串行推理链"**，hop_distance 是拓扑距离不是推理深度
- **验证真正多跳的标准是 step-deletion test**：删掉任意中间步骤后答案不可得（当前仅有 proxy heuristic，真正 step-deletion 验证待 Phase 1）
- **不同时铺开三条线**：优先 Phase 1（严格 multi-hop）→ Phase 2（element-level cross-doc）→ Phase 3（multi-turn）
- **50-100 条 gold 3-step queries 比 500 条 2-evidence 拼接更有论文价值**

---

## 当前状态（2026-03-16 更新｜Phase0 Eval v3 达标 + Graph 首次显著超越 BM25）

### 本轮完成（相对 2026-03-15）

- **Phase0 效果验证达标** — `continue_expand = True` ✅
  - graph_full：R@10=0.8736 (+0.0269 vs BM25), MRR=0.6045 (+0.0403 vs BM25)
  - 满足决策门 MRR ≥ BM25 + 0.03（实际 +0.0403）
  - 详见 `docs/EXPERIMENT_RECORD_2026-03-16.md`
- **三项工程修复**：quality_score 从常量 0.8 → 拓扑特征加权 [0.13, 0.88]；hub coverage 从 9.53% → 90.42%（纳入 adjacent_backbone_bridges 397 个 element）；citation walk 加入双向 + 2-hop co-citation
- **组件权重解耦**：新增 `--hub-weight/--nprop-weight/--cite-weight` 独立调参；最优配置 hw=0.15, nd=0.20, cw=0.0
- **关键发现**：neighbor_prop（1-hop 邻域标签传播）是核心信号，能拯救 11 条 BM25 遗漏的 queries；citation_walk 为负贡献（doc-level 粒度与 element-level 证据定位不匹配），应在 graph_full 中关闭；2-hop 不如 1-hop
- **MoDora 工作流代码已实现并通过静态审计**（A1/A2/B1/B2/C1 + PersonaHub；其余子项以脚本能力为准），但尚未完成 500 candidates 全量运行验证
- **产物文件**：`data111/hub_candidates_enriched_v3.json`、`data/phase0_eval_report_v3_tuned.json`

### 本轮关键结论
- **Graph 效果验证已达标，支撑 4 月专利申请**。核心机制（bridge hub topology → element adjacency → 1-hop label propagation）全程纯规则，零 LLM 成本
- **MoDora workstream 代码就绪但未经全量实战检验**：需要用 `--provider company` 跑 500 candidates 的 real-user + persona queries 来验证
- **`docs/GRAPH_ARCHITECTURE.md` 需要大幅扩充**：当前仅 42 行框架，缺少 eval 结果、最优配置、构建公式细节
- **C-Pool 万金油查询库**和 **Graph RAG 调研**仍未启动

---

## 当前状态（2026-03-12 更新｜战略升级：Document Graph as Core + 专利路径确认）

### 本轮完成（相对 2026-03-10）

- **Mentor 周会战略共识达成**：项目从"Query 生成工具"重新定位为"Document Graph for Document Understanding"系统
  - Graph 核心贡献：节点/边构建方法 + Hub 评分 + 多任务应用
  - Query 生成降级为 graph 的第一个 application（仍是当前主要交付物）
- **时间线确认**：4 月申专利（公司专利），5 月开放论文投稿
- **新方向纳入 roadmap**：PersonaHub + C-Pool 万金油查询库 + Graph RAG 调研 + 泛化方案设计
- **讨论记录**：已更新至 `docs/DISCUSSION_LOG.md`（2026-03-12 节）

### 本轮关键设计决策
- **图架构文档化是最高优先**：Mentor 明确要求，每次周会前必须有独立的图文档（节点/边/成本/评分），不能再散落在 CLAUDE.md 中
- **验证效果是 4 月目标**：design document graph → vs baseline（BM25/dense）在 QA 或 evidence localization 上的实验
- **C-Pool 策略**：~50-100 条人工精选的万金油通用 query，QC 只验 evidence localization，不验 query 质量
- **PersonaHub 人设驱动**：借鉴 PersonaHub（Ge et al., 2024, arXiv:2406.20094）方法论，策展 50 类学术领域读者人设，按 pair_id 哈希确定性分配，增强 query 多样性

---

## 当前状态（2026-03-30 更新｜Enrichment 消融实验完成 + 多轮系统完善）

### 本轮完成（相对 2026-03-26）

- **2×2 Enrichment 消融实验完成**
  - 脚本：`scripts/run_ablation_enrich.py`
  - 6 个条件：1A（raw query + raw corpus）/ 2A（enrich query + raw corpus）/ 1B（raw query + enrich corpus）/ 2B（enrich query + enrich corpus）/ 1A_matched / 2A_matched
  - Matched-pair 子集：L2=127对 / L3=28对（消除 candidate-set 混淆）

- **消融核心结论（BM25/HITS 均验证）**

  | 条件 | L2 R@10 | L2 MRR | L3 R@10 | L3 MRR |
  |------|---------|--------|---------|--------|
  | 1A 基线 | 0.530 | 0.471 | 0.333 | 0.501 |
  | 2A 仅 query 富化 | 0.536 | 0.456 | 0.471 | 0.476 |
  | 1B 仅语料富化 | 0.727 | 0.664 | 0.469 | 0.721 |
  | **2B 双端富化** | **0.705** | **0.647** | **0.690** | **0.753** |

  - **语料库 enrichment 是最大杠杆**：L2 R@10 +0.197，L3 MRR +0.220
  - **仅做 query enrichment 有害**：词汇不对称（section-rich query vs raw corpus），L3 MRR −0.025 ~ −0.075
  - **双端富化触发非线性增益**：L3 R@10 翻倍（0.333→0.690），MRR +0.252；词汇循环闭合假说得到验证
  - **L1 对 query 侧 enrichment 完全免疫**：query 结构而非词汇是 L1 的瓶颈

- **L1 评测 bug 已修复**：`run_m2_classic_eval.py` 新增 `_norm_eid()` 将 `_fig_` / `_tbl_` / `_eq_` 规范化为 corpus 格式；之前 L1 全零是 ID 不匹配
- **多轮 session 生成器升级**（`scripts/generate_multiturn_sessions.py` v2）
  - 加入 `context_isolation_score()` Jaccard 代理指标（阈值 0.35）
  - 新增 intent_shift 类型（L3: drill_down/bridging/contrastive；L2: drill_down/bridging）
  - Researcher 角色扮演 system prompt
- **Persona 库扩充**：50→76 人设（新增 26 个非学术人设：学生/医疗/金融法律/政府/教育媒体等）
- **Semantic Scholar 批量下载脚本**：`scripts/download_papers_semantic_scholar.py`，BFS 引用网络爬取，API key 下延迟 0.2s

### 消融实验关键发现（用于论文写作）

1. **单独的 query enrichment 无效**：即使 section-level LLM 生成的丰富上下文，没有匹配的 corpus enrichment 时词汇不对称反而降低 MRR
2. **MoDora element enrichment 是必须的**：为 corpus 侧提供方法论词汇，让 BM25 基线直接从 0.53 跳到 0.73（L2）
3. **两侧 enrichment 相互增强**：2B 不是 1B+2A 的加和，而是超加性（L3 表现最突出）
4. **HITS 在 2B 条件下仍有稳定增益**：L3 MRR 0.753（BM25）→ 0.791（HITS），+0.038

### 当前数据集规模（2026-03-30）
- L1: 974, L2: 344+249=593, L3: 143+80=223，**总计 ~1790 条**
- 图：11298 nodes / 19429 edges，82 篇文档
- 最优检索配置：**2B + HITS**（双端 enrichment + 图增强）

### 下一步
| 优先级 | 任务 |
|--------|------|
| P0 | 用 SS API 扩充语料到 500+ 篇，multi-turn 量产 |
| P0 | M3 之前调 graph 参数：L1 hw≈0，L2/L3 hw=0.15 |
| P1 | 正式跑 2B 条件 HITS 完整评测（确认作为最终 baseline） |
| P1 | 验证 2A_matched L3 MRR 反常下降（词汇不对称 or 小样本噪声） |
| P2 | QA evaluation 改进（answer correctness 替代 evidence mention） |

---

## 当前状态（2026-03-26 更新｜Section Enrichment + graph_full 权重调优）

### 本轮完成（相对 2026-03-24）

- **Section-level Enrichment 完成**
  - `enrich_section_nodes.py` 新增 `--incremental` + `--flush-every`（断点续跑）
  - 1417 个 section/subsection/subsubsection 节点全部 enriched（82 篇文档）
  - 输出：`data/m2/section_nodes_enriched_2026-03-26.json`
  - 费用：$8.29（gpt-5.4）

- **Section-Enriched Query 生成完成**
  - L2: 249 pass / 428 total（58.2%，vs baseline 57.2%）
  - L3: **80 pass / 122 total（65.6%，vs baseline 48.1%）** — 数量翻倍
  - 输出：`data/m2/l{2,3}_production_2026-03-26_section_enriched{,_pass}.jsonl`

- **graph_full 权重调优完成**
  - Grid search: nprop_weight 0.20 → 1.00 是最大改进
  - graph_full MRR: 0.6225 → **0.7234（+16.2%）**
  - 最优配置：`hw=0.15, nw=1.00, cw=0`
  - neighbor_prop 仍为绝对主力（MRR 0.7145），但 graph_full 加 hub prior 后略超（0.7234）

- **检索评测对比完成**（section-enriched, n=329）

  | 方法 | R@10 | MRR | ΔMRR vs BM25 |
  |------|------|-----|--------------|
  | bm25 | 0.796 | 0.531 | — |
  | neighbor_prop | 0.906 | 0.715 | +0.184 |
  | graph_full (hw=0.15,nw=1.00) | 0.903 | **0.723** | **+0.192** |

### 当前数据集规模
- L1: 974, L2: 344+249=593, L3: 143+80=223, **总计 ~1790 条**
- 图：11298 nodes / 19429 edges（section-aware keyword_boost 版），82 篇文档
- Hub overlap: **100%**

### 下一步
| 优先级 | 任务 |
|--------|------|
| P0 | Embedding 语义边实验（`build_embedding_edges.py`，需 GPU） |
| P1 | 用 tuned weights (nw=1.00) 重跑 baseline eval 并更新文档 |
| P1 | 正则引用模式扩展（"Figure X" / "Table Y"），适配纯 PDF |
| P2 | QA evaluation 改进（answer correctness 替代 evidence mention） |

---

## 当前状态（2026-03-10 更新｜MoDora 深度整合 + Real-user Query 风格 + Enrichment 质量闸门）

### 本轮完成（相对 2026-03-09）

- **MoDora 整合实施方案设计完成**（4 个 workstream 并行）
  - Workstream A：节点粒度细化（段落按 section 切分 + section 节点参与路径枚举）
  - Workstream B：Real-user query 风格（5 类新模板 + `--query-style` 切换 + node_group 支持）
  - Workstream C：Enrichment 质量闸门（噪声过滤器 + figure/table 一致性校验 + hub summary 压缩重写）
  - Workstream D：QC 体系重构（`qc_real_user_query()` 并行于现有 `qc_multihop_query()` + retrievability_score）
- **同事 Review 反馈已纳入方案**
  - 最高优先：低质量 enrichment 过滤器（glyph/icon/marker 等噪声模式检测，命中则回退原始 context）
  - figure/table 轻量一致性校验（caption 含 metric 词但 enriched 输出 figure_type=other → 低置信标记）
  - hub summary 从拼接升级为压缩重写（50-80 词，提升桥接语义密度）
- **实施方案文档**：`plan.md`（项目根目录）

### 本轮关键设计决策
- **旧模板保留，新模板并存**：通过 `--query-style academic/real_user/mixed` 切换，默认 `academic` 向后兼容
- **仅英文**：新 real-user 模板仍为英文
- **Node group 替代 strict pair**：新模板支持 1-3 个元素的 node_group，不再强制恰好 2 个
- **QC 双轨制**：academic 走现有 `qc_multihop_query()`，real_user 走新 `qc_real_user_query()`（放宽 yes/no、template 限制，新增 retrievability_score）
- **Enrichment 质量优先于数量**：query 生成前过滤低质量 enriched 字段，而非盲信

### 待改动文件（5 个）
| 文件 | 工作流 | 改动 |
|------|--------|------|
| `src/parsers/latex_reference_extractor.py` | A | `_extract_paragraphs()` 按 section 边界切分 |
| `scripts/analyze_latex_graph_topology.py` | A | section 节点参与路径 + `--single-doc-only` |
| `scripts/generate_multihop_l1_queries.py` | B, C, D | 5 类新模板 + enrichment 过滤器 + real-user QC + `--query-style` |
| `scripts/enrich_hub_candidates.py` | B, C | node_group 支持 + hub summary 压缩重写 |
| `src/utils/token_logger.py` | — | 无需改动（已合规） |

---

## 当前状态（2026-03-09 更新｜MoDora [T]/[M]/[C] Enrichment 整合）

### 本轮完成（相对 2026-03-07）

- **MoDora CCTree 思路分析完成**
  - 分析文档：`docs/MODORA_INTEGRATION_ANALYSIS.md`
  - 结论：借鉴"上游语义增强"，不迁移 CCTree 检索框架
- **P0.5：Element [T]/[M]/[C] Enrichment 脚本落地**
  - 新增 `scripts/enrich_elements_modora.py`
  - 对 figure/table/formula 三类元素分别用类型特化 prompt 生成结构化描述
  - 输出 `enriched_title` / `enriched_metadata` / `enriched_content` 三个新字段（不覆盖原字段）
  - 支持 `--provider`（anthropic/openai/company）、`--incremental`（增量模式）、`--dry-run`
  - 输出：`data/multimodal_elements_enriched.json`
- **P1：Hub Cascade Summary 增强**
  - `enrich_hub_candidates.py` 新增 `--enriched-elements` 参数
  - 新增 `build_hub_semantic_summary()` 函数：聚合两端元素 enriched 描述 + edge context + keywords
  - 输出新字段 `hub_semantic_summary`（附加到每个 candidate pair）
- **Phase 3：Query 生成上下文升级**
  - `generate_multihop_l1_queries.py` 新增 `build_enriched_context_section()`
  - `_context()` 优先读取 `enriched_content`
  - 所有 4 个 prompt 模板自动附加 enriched section（当 enriched 字段存在时）
  - 向后兼容：无 enriched 字段时行为完全不变

### 本轮关键技术发现
- **MoDora [T]/[M]/[C] 思路对我们最有价值的是"上游语义增强"**，而非其树结构或在线检索
- 我们多层图（citation + cross-modal + backbone）对跨文档/跨模态表达力优于 CCTree 树合并
- Element enrichment 预期改善 `single_element_answer` 和 `weak_reasoning_connector` 类 QC 失败

### MoDora 整合 Pipeline（新增）
```bash
# Step 0: Element enrichment（MoDora-style [T]/[M]/[C]）
python scripts/enrich_elements_modora.py \
    --input data/multimodal_elements.json \
    --output data/multimodal_elements_enriched.json \
    --provider anthropic \
    --model claude-sonnet-4-5-20250929 \
    --delay 0.3

# Step 1: Hub enrichment（传入 enriched elements）
python scripts/enrich_hub_candidates.py \
    --hub-candidates data/latex_hub_multihop_candidates.json \
    --elements data/multimodal_elements.json \
    --latex-graph data/latex_reference_graph.json \
    --enriched-elements data/multimodal_elements_enriched.json \
    --output data/hub_candidates_enriched_v2.json

# Step 2: Query generation（自动使用 enriched context）
python scripts/generate_multihop_l1_queries.py \
    --candidates data/hub_candidates_enriched_v2.json \
    --output data/l1_dual_evidence_queries_hub_enriched_v1.jsonl \
    --pass-only \
    --provider anthropic \
    --model claude-sonnet-4-5-20250929 \
    --delay 0.3
```

---

## 当前状态（2026-03-07 更新｜公司 API 整合 + 全量生成就绪）

### 本轮完成（相对 2026-03-03）

- **公司 API（yunwu.ai）整合完成**
  - `generate_multihop_l1_queries.py` 新增 `--provider company` 选项
  - 通过 `local_api_logger` 的 `wrap_requests_call` 发送请求，SSE 流式解析 + 自动 token 日志
  - 环境变量：`COMPANY_API_KEY` / `COMPANY_API_URL`；也可通过 CLI `--company-api-key` / `--company-api-url` 传入
  - 图像用 OpenAI 兼容 `image_url` 格式发送（yunwu.ai 是 OpenAI-compat 代理）
  - `main.py` demo 脚本可做连通性测试
- **v4.4 run1 已有真实产物**（前序补记）
  - `data/l1_dual_evidence_queries_v4_4_run1.jsonl`：252 条
  - `data/l1_dual_evidence_queries_v4_4_run1_pass.jsonl`：113 条（44.8% pass）

### 本轮关键技术发现
- **公司 API 是 OpenAI-compat**：endpoint `/v1/chat/completions`，请求格式与 OpenAI SDK 一致，但走 `local_api_logger` 包装器自动记录 token 统计
- **三种 provider 并存**：`anthropic`（直连 Claude API）、`openai`（OpenAI SDK）、`company`（yunwu.ai via local_api_logger），在 `call_api()` 内按 provider 分支处理

### 当前全量生成就绪条件
- 代码侧：✅ 已完成（`--provider company` + SSE 解析 + token logging）
- `local_api_logger` 模块：⬜ 需用户放入项目根目录
- `COMPANY_API_KEY`：⬜ 需设置有效 key
- 目标：500 条 hub candidates → L1 dual-evidence queries

---

## 当前状态（2026-03-03 更新｜LaTeX Topology v2 + Hub Multi-hop Candidates）

### 本轮完成（相对 2026-02-24）

- **`analyze_latex_graph_topology.py` v2 完整落地**
  - 核心改动：backbone edges（1269 条）、bridge-first hub 评分、adjacent bridge 检测、cross-doc citation edges（434 条）、targeted enumeration（替换 DFS）、content_list.json 真实 page_idx、4 种 seed 类型轮换、structural dedup
  - 图统计：**2551 nodes, 3471 edges**（backbone:1269, paragraph_ref:1688, cross_doc_cite:434, element_ref:80）
  - label 匹配率：**49.8%**（从 28.8% 提升，Jaccard 阈值 0.25 + 数字后缀 fallback）
- **Hub 质量全面提升**
  - bridge_hubs: **60 个**（覆盖 31 篇文档，all-3 modality:31，fig+formula:25，fig+table:4）
  - top-60 hubs **100% category=bridge**（authority sinks 全部从排名中清除）
  - adjacent_backbone_bridges: **369 条**（覆盖 68 篇文档）
  - bridge-first hub_score 公式：`bridge_score = num_modalities*15 + out_to_elements*2`
- **500 候选对生成成功**（替换原来 DFS 产出的 23 对）
  - 分布：figure+formula:247 / figure+table:153 / formula+table:100
  - intra-doc:330 (66%) + cross-doc:170 (34%)
  - 2-hop:181 / 3-hop:319
  - 来源：bridge_hub:310 / adjacent_backbone_bridge:190
  - 覆盖文档：**40/82 篇**（35/82 篇仍为零候选，主要缺陷）
- **物理距离覆盖**
  - line_no_span: **100%**（全覆盖）
  - page_span: **19%**（需双端 label 匹配，结构性上限）
  - real page_idx（来自 content_list.json）：元素覆盖率 **94.8%**
- **Seed 多样性**
  - 4 种类型轮换（WHY/WHAT_IF/MISMATCH/CONDITION），by `hash(tuple(path)) % 4`
  - 独特 short seeds: 496/500 (99.2%)

### 本轮关键技术发现
- **MinerU content_list.json 有真实 page_idx**（multimodal_elements.json 中全为 0 是 parser bug）
  - Sequential type-order matching（第 N 个 figure 对应第 N 个 content_list 中的 image 项）实现 94.8% 覆盖
- **DFS 在 backbone chain 中迷路**：backbone 边（1269 条）形成长 para→para→para 链，max_hops=5 内到不了 2 个不同模态
  - 修复：targeted enumeration（2-hop direct + 3-hop via backbone neighbor + cross-doc）
- **Bridge hub vs Authority hub 区分**：高被引 formula 节点（如 in_from_paragraphs=49）会主导旧评分，实为 authority sink；真正有用的是覆盖多模态的 paragraph bridge

### 输出文件（新增）
- `data/latex_graph_topology_report.json` — 拓扑统计报告（节点/边/label匹配/hub分类）
- `data/latex_graph_hubs.json` — bridge_hubs 60 个 + adjacent_backbone_bridges 369 条
- `data/latex_hub_multihop_candidates.json` — **500 条候选对**（含 path, seed_question, page_span, line_no_span）

### 下一步（已确定）
1. **P0（最高优先）**：将 500 条 topology candidates 喂给 `generate_multihop_l1_queries.py` 生成新 L1 hub-multihop queries
2. **P1**：修复 35/82 篇零候选文档——降低 per_combo cap 或对 adj_bridge-only 文档单独生成
3. **P0.1**：Citation-based L2 候选（123 引用边 → 替代实体倒排索引）

---

## 当前状态（2026-02-24 更新｜Dual-evidence + Cross-doc）

### 本轮完成（相对 2026-02-22）
- **L1 dual-evidence 官方批次完成**（`data/l1_dual_evidence_queries_slurm_img150_tuned_v4_official.jsonl`）
  - 总量 222，QC pass 173，pass rate 77.93%
  - pair_type: figure+table 144 / figure+formula 62 / formula+table 16
- **Triplet 构建完成（v1 + v2）**
  - v1：`in_doc_swap + same_type_hard`
  - v2：`in_doc_swap + same_type_hard_plus`，并加入 `text_short`、图像覆盖统计
  - v2 all：222 triplets，avg_difficulty 0.7288，positive image coverage 100%
- **本地 embedding 跨文档匹配跑通（Qwen3-Embedding-4B）**
  - 输出：`data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B.jsonl`
  - records 590（top-k=20，总 match 11800）
- **4B 匹配审计完成**
  - 报告：`data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_audit.json`
  - baseline: top1_mean 0.8822，top10 target concentration 0.3153，unique top1 targets 186，suspicious 241
- **Stage-B Utility-aware Rerank 已落地**
  - 脚本：`scripts/rerank_mineru_crossdoc_matches.py`
  - 审计脚本：`scripts/audit_mineru_crossdoc_embedding_matches.py`
  - 严格版（cap=8）：`..._v2_rerank.jsonl`
  - 平衡版（cap=10，当前推荐）：`..._v2b_cap10.jsonl`
  - 平衡版结果：top1_mean 0.8690；top10 concentration 0.1305；unique top1 targets 286；reciprocal 0.8119；suspicious 146
- **汇报文档已整理**
  - `docs/REPORT_SUMMARY_2026-02-24.md`

### 本轮讨论共识（方法论）
- 仅优化 embedding top-1 属于 **objective mismatch**（“相似” != “多跳有用”）
- 当前阶段主目标应转向：
  1. 候选召回与多样性（Stage A）
  2. utility-aware rerank（Stage B）
  3. 构链约束与 answerability（Stage C）
- **top-1 平均分不是主 KPI**；应引入 `hop_utility` 相关评估

### 当前数据口径（重要）
- 当前 dual-evidence 数据**默认包含文本证据**（`text` / `text_short` + evidence spans）
- 当前 pair_type 仅保留：
  - `figure+table`
  - `figure+formula`
  - `formula+table`
- **不含单独 `figure+text / table+text / formula+text` 作为本轮 dual-evidence 训练单元**
  - 单图文 L1 历史线仍在：`data/l1_cross_modal_queries_v3.jsonl`

### 下一步（已确定）
1. 冻结平衡版 cross-doc 候选：`data/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2b_cap10.jsonl`
2. 建立 100-300 条人工标注小基准（relevance / hop_utility / redundancy / error_type）
3. 生成 triplet v3：在保留 `in_doc_swap` 基础上，引入 reranked cross-doc hard negatives
4. 做最小消融：embedding-only vs +hub/diversity rerank vs +context rerank

## 当前状态（2026-02-22 更新）

### 已完成
- **集群**: 86 篇 arXiv 论文下载（种子论文：1908.09635），85 篇 PDF 用 MinerU 解析
- **公司电脑**: 76 篇论文（数量差异正常，集群多跑了更多论文）
- **Step 0 v1: Figure-text association** — 351 pairs, 73 docs（`src/linkers/figure_text_associator.py`）
- **Step 0 v2: Multimodal relationship DAG** — 1316 elements (841 fig + 334 tbl + 141 formula), 1261 edges, 1135 cross-modal pairs, 76 docs（`src/linkers/multimodal_relationship_builder.py`）
- **Step 0 v3: LaTeX reference graph（集群）** — 82/86 篇源码下载，2021 labels, 7423 refs, 3019 edges, 74 篇有 .bbl（`scripts/build_latex_reference_graph.py`）
- **Step 0 v3.1: Cross-document citation graph（集群）** — **123 条**跨文档引用边, **55 篇**最大连通分量（`scripts/build_citation_graph.py`）
- **Citation graph 质量验证** — 人工抽查 title_fuzzy 匹配，**误匹配率 0%**，Jaccard ≥ 0.55 阈值可信
- **Step 1: L1 intra-document cross-modal queries** — 经 3 轮迭代，最终 **974 条 queries**
- **L1 Triage** — A:727 (74.6%) / B:247 (25.4%) / C:0 (0%)  *(after visual_density gate)*
- **L2 候选构建** — 55 个跨文档实体，711 个候选文档对，top-100 已输出
- **Step 2: L2 cross-document queries** — 经 3 轮迭代
  - v1: 50 条, 100% QC pass (QC 过松), $0.55
  - v2: 32 条, 16 QC pass (严格 QC 但有 anchor leakage), $0.48
  - v3: 42 条, **19 QC pass** (anchor_leakage 仍是主因: 21/23 fail)
- **L1 Cross-modal Dual-evidence v1** — 300 条, **43 QC pass (14.3%)**
- **L1 Cross-modal Dual-evidence v2（hard-gate）** — 296 条, **19 QC pass (6.42%)**，已导出 pass 子集
- **Step 0 v3.2 v3（G1+G2，集群已跑）** — **118 对**（proximity:105 + direct:13），gold:6 + silver:112，label 匹配率 28.8%（`data/latex_cross_modal_pairs.json`）
  - G1: hub de-dup（每 element ≤3 pairs by quality_score）
  - G2: cross-reference gate（ctx_a mention label_b OR ctx_b mention label_a，否则 hard drop）
  - char_proximity_limit: 300 chars（从 1000 缩紧）
- **L1 Cross-modal Dual-evidence v3（LaTeX bridge 注入）** — **236 条, 72 QC pass (30.5%)**, $1.66
  - 输入: 118 对 latex_cross_modal_pairs（含 bridge_text evidence）
  - 核心改进: `build_latex_bridge_section()` 注入 author 原句；formula 用 context_before/after 做 QC
  - QC 主失败: bridge_entity_leakage:84, single_element_answer:63, anchor_leakage:61
- **L1 Dual-evidence v4（Conceptual Masking + Operator 强制 + evidence_spans）** — **236 条, 139 QC pass (58.9%)**, $2.07
  - 输入: 同 118 对 latex_cross_modal_pairs
  - 核心改进: Rule 8 DE-NAME→Conceptual Masking；新增 cross-modal operator 约束；required_evidence_spans 字段；bridge_entity_leakage 降为软警告 (Option A)
  - figure+table: ~69%；figure+formula: 24/74 (32.4%)；formula+table: 9/16 (56.3%)
  - QC 主失败: single_element_answer:60, anchor_leakage:20, weak_reasoning_connector:19
  - 输出: `data/l1_dual_evidence_queries_v1.jsonl`
- **L1 Dual-evidence v4.1（opus figure+formula prompt + operator diversity + is_yes_no fix）** — **236 条, 138 QC pass (58.5%)**, $2.39
  - 输入: 同 118 对 latex_cross_modal_pairs
  - 核心改进: opus-4-6 重设计 PROMPT_FIGURE_FORMULA（Figure Type Strategy, 双 field）；禁 instantiate；is_yes_no_question WH-word 修复；--pass-only 硬门禁
  - figure+table: 101/146 (69.2%) ↑；figure+formula: 30/74 (40.5%) ↑；formula+table: 7/16 (43.8%) ↓
  - QC 主失败: single_element_answer:62, anchor_leakage:39 ↑（回归），weak_reasoning_connector:6 ↓
  - 输出: `data/l1_dual_evidence_queries_v2.jsonl`（138 条纯净 pass-only）
- **L1 Dual-evidence v4.2（PhD persona + verb diversity + natural operators）** — **236 条, 152 QC pass (64.4%)**, $2.57
  - 输入: 同 118 对 latex_cross_modal_pairs
  - 核心改进: persona "PhD student at lab meeting"（消除学术腔）；verb 黑名单（validate/quantify/justify/demonstrate 等）；SENTENCE STRUCTURE 多样性约束（GIVEN-WHY/WHAT-IF/WHY-INCONSISTENT/WHEN-CONDITION/WHAT-CAUSES）；CROSS_MODAL_OPERATORS 扩展自然英文动词（affect/differ/produce/achieve 等）；双文件输出（full + _pass）；is_yes_no WH-word 修复完善
  - figure+table: 111/146 (**76.0%**) ↑↑；figure+formula: 34/74 (**45.9%**) ↑；formula+table: 7/16 (43.8%)
  - QC 主失败: single_element_answer:57 ↓, anchor_leakage:29 ↓↓, weak_reasoning_connector:4 ↓
  - 输出: `data/l1_dual_evidence_queries_v3.jsonl`（全量 236 条）+ `data/l1_dual_evidence_queries_v3_pass.jsonl`（152 条）

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

- **Citation-based L2 候选对** — 用 123 条引用边（集群）替代实体倒排索引做 L2 候选（fuzzy match 质量已验证）
- **L1 v3 QC 分析与迭代** — bridge_entity_leakage(84) + single_element_answer(63) 仍是瓶颈，待分析 root cause
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
| `data/l1_multihop_queries_v2.jsonl` | L1 multihop v2 hard-gate（296 条，19 pass） |
| `data/l1_multihop_queries_v2_pass.jsonl` | v2 通过集（19 条） |
| `data/l1_multihop_queries_v3.jsonl` | L1 multihop v3 LaTeX-bridge（236 条，72 pass，30.5%） |
| `data/l1_dual_evidence_queries_v1.jsonl` | **L1 dual-evidence v4（236 条，139 pass，58.9%）** |
| `data/l1_dual_evidence_queries_v2.jsonl` | L1 dual-evidence v4.1（138 条，pass-only） |
| `data/l1_dual_evidence_queries_v3.jsonl` | **L1 dual-evidence v4.2 全量（236 条，含 fail）** |
| `data/l1_dual_evidence_queries_v3_pass.jsonl` | **L1 dual-evidence v4.2 通过集（152 条，64.4%）** |
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
| `scripts/analyze_latex_graph_topology.py` | **LaTeX 拓扑分析 v2（backbone+bridge-first+adj_bridge+cross_doc+page_idx）** |
| `data/latex_graph_topology_report.json` | 拓扑统计报告（2551 nodes, 3471 edges, 49.8% label match） |
| `data/latex_graph_hubs.json` | bridge_hubs 60 个 + adjacent_backbone_bridges 369 条 |
| `data/latex_hub_multihop_candidates.json` | **Hub multi-hop 候选对 500 条（含 page_span/line_no_span/seed）** |
| `scripts/enrich_elements_modora.py` | **MoDora-style [T]/[M]/[C] 元素语义增强（figure/table/formula）** |
| `data/multimodal_elements_enriched.json` | **MoDora enriched 元素（含 enriched_title/metadata/content）——待生成** |
| `docs/MODORA_INTEGRATION_ANALYSIS.md` | **MoDora CCTree 整合分析文档** |
| `docs/M4_STRATEGY_REVIEW_2026-03-18.md` | **M4 战略重定位文档（诚实现状评估 + 路线图）** |
| `docs/M4_SCHEMAS.md` | **M4 三套数据 Schema（multi-hop / cross-doc / multi-turn）** |
| `docs/M4_RESEARCH_NOTES.md` | M4 学术背景调研（M4DocBench / CoQA / TRACE / RT-RAG） |
| `scripts/filter_l3_candidates.py` | **M2: L3 候选筛选（hop≥3 + bridge paragraph + 跨模态，130/500 条）** |
| `scripts/package_m2_levels.py` | **M2: 三层数据打包（L1+L2+combined，统一 schema）** |
| `scripts/run_exp_a_difficulty.py` | **M2 Exp A: BM25 Recall@10 难度梯度实验（L1 vs L2 vs L3）** |
| `scripts/run_exp_c_qa_triangle.py` | **M2 Exp C: BM25 vs Graph 证据覆盖 + LLM QA 对比** |
| `data/m2/level1_single_element.jsonl` | **M2 Level 1 数据（974 条单元素 query）** |
| `data/m2/level2_dual_evidence.jsonl` | **M2 Level 2 数据（157 条双证据 query）** |
| `data/m2/all_levels_combined.jsonl` | **M2 全量合并（1131 条，含 difficulty_level 字段）** |
| `data/m2/l3_candidates_filtered.json` | **M2 Level 3 候选（130 条 3-hop 候选，待生成 query）** |
| `data/m2/exp_b_retrieval_enhancement.json` | **M2 Exp B 结果（复用 Phase0 eval v3）** |
| `main.py` | **公司 API 连通性测试脚本（yunwu.ai demo）** |
| `local_api_logger/` | **公司 API 日志库（wrap_requests_call + token 统计）——需用户放入** |

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

## 下一步 TODO（2026-03-20 更新）

### 已完成（历史）
- ~~**M4 Strategy Review + Schema 设计**~~ ✅ **完成** — 诚实重定位为 M4-Foundation；三套 Schema 落地；step-deletion QC 集成
- ~~**Phase0 Eval v2 首轮**~~ ✅ **完成** — graph 与 BM25 持平，hub_overlap=9.53%，continue_expand=False
- ~~**Phase0 Eval v3 三项修复**~~ ✅ **完成** — quality_score 重建 + hub coverage 扩大 + citation walk 修复
- ~~**Phase0 组件权重解耦 + Grid Search**~~ ✅ **完成** — graph_full MRR +0.0403，`continue_expand=True`
- ~~**MoDora 四工作流代码实现**~~ ✅ **完成** — A1/A2/B1/B2/C1/C3/D1 + PersonaHub 全部已实现（代码就绪，未全量运行）
- ~~**M2 pipeline 代码 + 数据打包**~~ ✅ **完成** — 三层数据 + L3 候选筛选 + 3-step prompt + 三组实验脚本
- ~~前序历史~~ ✅ 见 `docs/DISCUSSION_LOG.md`

### MoDora 工作流代码完成度（代码就绪，待全量验证）

| 工作流 | 代码 | 文件 | 待验证 |
|--------|------|------|--------|
| A1: Section 粒度切分 | ✅ | `src/parsers/latex_reference_extractor.py` | 需重跑 pipeline 验证切分效果 |
| A2: Strategy 4 + `--single-doc-only` | ✅ | `scripts/analyze_latex_graph_topology.py` | 需验证 section-bridged candidates 质量 |
| B1: 5 类 real-user 模板 | ✅ | `scripts/generate_multihop_l1_queries.py` | 需 `--query-style real_user` 全量跑 |
| B2: `--query-style` CLI | ✅ | `scripts/generate_multihop_l1_queries.py` | 同上 |
| C1: Enrichment 噪声过滤器 | ✅ | `scripts/generate_multihop_l1_queries.py` | 随 query 生成自动生效 |
| C3: Hub summary 压缩重写 | ✅ | `scripts/enrich_hub_candidates.py` | 已在 v3 enrichment 中使用 |
| D1: `qc_real_user_query()` | ✅ | `scripts/generate_multihop_l1_queries.py` | 需 real_user queries 触发 |
| PersonaHub 人设 (50 类) | ✅ | `scripts/generate_multihop_l1_queries.py` + `data/personahub_academic_personas.json` | 需 `--use-persona` 全量跑 |
| MoDora enrichment 脚本 | ✅ | `scripts/enrich_elements_modora.py` | 需跑生成 `multimodal_elements_enriched.json` |

### P0（本周，M2 实验执行 — L3 生成 + Exp A/C）

1. **~~M4 Strategy Review + Schema 设计~~** ✅ 完成
2. **~~Reasoning-depth heuristic tagging~~** ✅ 完成
3. **~~M2 pipeline 代码 + 数据打包~~** ✅ 完成 — 三层数据 + L3 候选 130 条 + 3-step prompt + Exp A/B/C 脚本
4. **用公司 API 生成 L3 queries**：`python scripts/generate_multihop_l1_queries.py --candidates data/m2/l3_candidates_filtered.json --output data/m2/l3_reasoning_chain_queries.jsonl --pass-only --provider company --model gpt-5.4 --delay 0.5`，目标 50-100 条 pass
5. **跑 Exp A（难度梯度）**：依赖 L3 queries 落地，`python scripts/run_exp_a_difficulty.py`
6. **跑 Exp C（QA 三角）**：依赖 L3 queries 落地，`python scripts/run_exp_c_qa_triangle.py --provider company`
7. **Reasoning-depth heuristic 误差审计**：抽 30-50 条人工标 serial/parallel/mixed，对比脚本分类结果，算 precision/recall
   - 审计脚本：`scripts/audit_reasoning_depth_heuristic.py`

### P0.5（并行保底交付线 — 不因战略升级停摆已有可交付）

8. **全量生成 real-user + persona queries**：`--provider company --query-style mixed --use-persona` 跑 500 hub candidates
9. **跑 MoDora element enrichment**：生成 `data/multimodal_elements_enriched.json`
10. **扩充 `docs/GRAPH_ARCHITECTURE.md`**：补充 eval 结果 + 最优配置 + hub 评分细节

### P1（2 周内，M4 Phase 2 — Multi-document）

11. **构建 element-level cross-doc edges**：用已有 Qwen3-Embedding-4B 匹配（`crossdoc_embedding_matches`）建立元素级跨文档边，输出 `cross_doc_edges_v1.jsonl`
12. **小规模 eval 验证 element-level > doc-level**：证明 element-level 桥接比 citation walk 更合理
13. **跨文档 multi-hop 路径枚举**：路径可跨越文档边界

### P2（1 个月内，M4 Phase 3 — Multi-turn + 收尾）

14. **Multi-turn session 生成**：将推理链转写为对话，每 hop → 一 turn，加入指代和省略
15. **Turn-dependency QC**：`qc_turn_dependency()` — 删掉前轮信息后当前轮不可回答
16. **M4 联合验证**：multi-hop + multi-doc + multi-turn + multi-modal 全覆盖 eval

### P3（持续）

17. **C-Pool 万金油查询库**：50-100 条通用学术 query
18. **泛化方案设计**：纯 PDF（无 LaTeX）场景下的低成本建图方案

详见 `docs/DISCUSSION_LOG.md` 最新讨论（2026-03-20 节）+ `docs/EXPERIMENT_RECORD_2026-03-16.md`

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

# === M2 experiment pipeline ===
# 打包三层数据
python scripts/package_m2_levels.py

# 筛选 L3 候选
python scripts/filter_l3_candidates.py

# 生成 L3 queries（公司 API）
python scripts/generate_multihop_l1_queries.py \
    --candidates data/m2/l3_candidates_filtered.json \
    --output data/m2/l3_reasoning_chain_queries.jsonl \
    --pass-only --provider company --model gpt-5.4 --delay 0.5

# Exp A: 难度梯度
python scripts/run_exp_a_difficulty.py

# Exp C: QA 三角
python scripts/run_exp_c_qa_triangle.py --provider company

# === 公司 API（yunwu.ai）pipeline ===
# 连通性测试
export COMPANY_API_KEY="sk-your-key"
python main.py

# 用公司 API 跑 500 条 hub candidates 全量生成
python scripts/generate_multihop_l1_queries.py \
    --candidates data/latex_hub_multihop_candidates.json \
    --output data/l1_dual_evidence_queries_hub_v1.jsonl \
    --pass-only \
    --provider company \
    --model claude-sonnet-4-20250514 \
    --delay 0.5

# === Hub 候选 enrichment pipeline ===
# Step 1: 将 topology hub candidates 转为生成脚本可用格式
python scripts/enrich_hub_candidates.py \
    --hub-candidates data/latex_hub_multihop_candidates.json \
    --elements data/multimodal_elements.json \
    --latex-graph data/latex_reference_graph.json \
    --output data/hub_candidates_enriched.json

# Step 2: 用 enriched 候选跑生成（公司 API）
python scripts/generate_multihop_l1_queries.py \
    --candidates data/hub_candidates_enriched.json \
    --output data/l1_dual_evidence_queries_hub_enriched_v1.jsonl \
    --pass-only \
    --provider company \
    --model claude-sonnet-4-20250514 \
    --delay 0.5

# Step 2 备选: 用 Anthropic 直连
python scripts/generate_multihop_l1_queries.py \
    --candidates data/hub_candidates_enriched.json \
    --output data/l1_dual_evidence_queries_hub_enriched_v1.jsonl \
    --pass-only \
    --provider anthropic \
    --model claude-sonnet-4-5-20250929 \
    --delay 0.3
```

## 关键命令（PowerShell 版，本地 Windows 使用）
```powershell
# 激活 conda 环境
conda activate minerU

# 加载 API key（从 .env 文件）
Get-Content .env | Where-Object { $_ -notmatch '^#' -and $_.Trim() -ne '' } | ForEach-Object { $p = $_ -split '=', 2; [Environment]::SetEnvironmentVariable($p[0], $p[1], 'Process') }

# === Hub 候选 enrichment pipeline ===
# Step 1: enrichment
python scripts/enrich_hub_candidates.py --hub-candidates data/latex_hub_multihop_candidates.json --elements data/multimodal_elements.json --latex-graph data/latex_reference_graph.json --output data/hub_candidates_enriched.json

# Step 2: 生成（公司 API）
$env:COMPANY_API_KEY = "sk-your-key"
python scripts/generate_multihop_l1_queries.py --candidates data/hub_candidates_enriched.json --output data/l1_dual_evidence_queries_hub_enriched_v1.jsonl --pass-only --provider company --model claude-sonnet-4-20250514 --delay 0.5

# Step 2 备选: Anthropic 直连
python scripts/generate_multihop_l1_queries.py --candidates data/hub_candidates_enriched.json --output data/l1_dual_evidence_queries_hub_enriched_v1.jsonl --pass-only --provider anthropic --model claude-sonnet-4-5-20250929 --delay 0.3

# === M2 experiment pipeline ===
python scripts/package_m2_levels.py
python scripts/filter_l3_candidates.py
python scripts/generate_multihop_l1_queries.py --candidates data/m2/l3_candidates_filtered.json --output data/m2/l3_reasoning_chain_queries.jsonl --pass-only --provider company --model gpt-5.4 --delay 0.5
python scripts/run_exp_a_difficulty.py
python scripts/run_exp_c_qa_triangle.py --provider company

# === 其他常用命令 ===
# 连通性测试（公司 API）
$env:COMPANY_API_KEY = "sk-your-key"; python main.py

# Dry-run 验证（不调 API，只看 prompt）
python scripts/generate_multihop_l1_queries.py --candidates data/hub_candidates_enriched.json --output NUL --dry-run --limit 5 --no-images

# Validation
python scripts/validate_queries.py data/l1_dual_evidence_queries_hub_enriched_v1.jsonl --output data/validation_report_hub_enriched_v1.json

# L1 三分法分拣
python scripts/triage_l1_v3.py

# 构建跨文档候选对
python scripts/build_l2_candidates.py --topk 100

# LaTeX 拓扑分析
python scripts/analyze_latex_graph_topology.py

# LaTeX cross-modal links
python scripts/build_latex_cross_modal_links.py --elements data/multimodal_elements.json --latex-graph data/latex_reference_graph.json --output data/latex_cross_modal_pairs.json
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

## 日期：2026-03-03（v4.4 全量运行阻塞排障，MinerU 服务部署任务排除）

### 本轮目标
- 根据最新讨论，执行一次新版 `v4.4` query 全量生成并做前后对比。
- 说明：**MinerU 部署服务任务本轮不做**（按用户要求排除）。

### 本轮已完成
1. 新增并落地拓扑/质量分析脚本与报告（已写入 `docs/TASK_EXECUTION_2026-03-03.md`）：
   - `scripts/analyze_latex_graph_topology.py`
   - `scripts/analyze_query_quality_focus.py`
   - 产物：
     - `data/latex_graph_topology_report.json`
     - `data/latex_graph_hubs.json`
     - `data/latex_hub_multihop_candidates.json`
     - `data/query_quality_focus_report_v4_official.json`
2. 升级 `scripts/generate_multihop_l1_queries.py` 到 v4.4（长度混合 + 架构图专项 QC）。
3. 为避免 `anthropic` 依赖问题，已给 `generate_multihop_l1_queries.py` 增加 `--provider openai` 兼容路径（可用 `OpenAI` 客户端直接跑）。

### 本轮阻塞（导致“跑一次”未完成）
1. **默认系统 Python 跑 Anthropic 路径失败**
   - 错误：`ModuleNotFoundError: No module named 'anthropic'`
2. **指定环境 `/projects/myyyx1/envs/minerU` 不可用**
   - 现象：`python`/`pip` 启动超时（`timeout` 返回码 124）
   - 进程状态：`Ds`
   - 内核等待点：`ceph_mdsc_wait_request`
   - 结论：当前不是脚本逻辑问题，而是环境/文件系统 I/O 卡死
3. **OpenAI fallback 探针到 API 层，但额度不足**
   - 命令成功发起到请求阶段
   - 返回：`429 insufficient_quota`
   - 文件：`data/_tmp_openai_probe.jsonl`（空）

### 当前状态（可直接对外同步）
- 代码侧改造已完成，运行链路已打通到 API 调用前/调用层。
- 目前缺的是**可用运行环境 + 可用额度 key**，不是 pipeline 代码缺失。
- 全量 run（150 candidates）尚未产出新文件：
  - 目标文件：`data/l1_dual_evidence_queries_v4_4_run1.jsonl`
  - `pass` 子集：`data/l1_dual_evidence_queries_v4_4_run1_pass.jsonl`

### 下一步最短恢复路径
1. 修复/切换 `minerU` 环境（优先，保证 Anthropic 路径可跑），或
2. 提供有额度的 `OPENAI_API_KEY`，走 `--provider openai` 直接全量。

## 日期：2026-03-03（状态对齐补记：v4.4 run1 已落盘）

### 对齐说明
- 上一节“未产出 run1 文件”是当时排障时的状态快照。
- 当前仓库已存在并可读取 `v4.4 run1` 产物，状态以本节为准。

### 已核验产物
- `data/l1_dual_evidence_queries_v4_4_run1.jsonl`：252 条
- `data/l1_dual_evidence_queries_v4_4_run1_pass.jsonl`：113 条

### 本轮结果摘要（run1）
- 总体：`qc_pass=113`，`qc_fail=139`（44.8% pass）
- 长度桶（all）：`short=104`，`long=87`，`medium=19`，`too_long=42`
- 长度桶（pass）：`short=59`，`long=54`（通过集已实现短长并存）
- 架构图样本：68 条，其中 pass 23 条（33.8%）
- 架构图失败主因：`architecture_intent_missing`（29），`length_mix_missing`（22），`query_too_long`（9）

### 当前结论（对外口径）
- “跑一次”已具备真实产物，不再是“仅代码改造完成”状态。
- 现阶段主问题从“环境/API 阻塞”转为“质量稳定性”，尤其是：
  - pair 级长度混合一致性（`length_mix_missing`）
  - 架构图场景的问题意图约束（`architecture_intent_missing`）
  - 过长 query 控制（`query_too_long`）

## 当前状态（2026-03-15 更新｜Phase0 Eval：Document Graph vs BM25 基线实验）

### 本轮完成

- **Phase0 Eval A/B 实验执行完成**（`scripts/run_phase0_eval_ab.py`）
  - 评测集：261 条通过 QC 的 L1 dual-evidence queries（v4_4_run1 113条 + v3 152条），候选库 1314 chunks
  - 运行两轮：保守版（alpha=0.3, citation_decay=0.0）+ Bug修复版（alpha=0.1, citation_decay=0.15）
  - 产物：`data/phase0_eval_report_tuned.json`、`data/phase0_eval_report_bugfix.json`

### 关键数字（Bug修复版）

| Method | Recall@10 | MRR | vs BM25 |
|--------|-----------|-----|---------|
| bm25（基线） | 0.8467 | 0.5642 | — |
| graph_hub_rerank | **0.8506** | **0.5637** | +0.0039 / -0.0005 |
| graph_neighbor_prop | **0.8506** | 0.5596 | +0.0039 / -0.0046 |
| graph_citation_walk | 0.8352 | 0.5618 | **-0.0115** |
| graph_full | 0.8467 | 0.5552 | 0 / -0.009 |

### 本轮关键发现

1. **Alpha 超参是最大变量**：alpha 0.3→0.1，hub_rerank Recall +0.0422（从 0.8084 升至 0.8506）。hub_overlap=9.53% 导致高 alpha 反噬 BM25 原本正确的打分
2. **neighbor_prop 最稳健**：两轮结果一致 +0.0039 Recall，邻域传播信号真实存在但小
3. **citation_walk 仍为负**：即使 bug 修复后，citation walk Recall -0.0115。推测原因：walk 方向（从 query doc 沿引用边传播）可能与证据实际所在方向错位
4. **hub_overlap = 9.53% 是结构上限**：261 条中只有约 25 条 queries 的 evidence 落在 hub 邻域，graph 信号天花板低
5. **continue_expand = False**：未达 +0.05 Recall 或 +0.03 MRR 阈值，暂不扩大 Phase0 规模

### 下一步从本次实验得出的行动

- **P0：扩大 hub coverage**（当前 9.53% 过低，需增加 hub 节点数或降低邻域判定阈值）
- **P0：调查 citation walk 方向**（逆向 walk 或双向传播实验）
- **P1：alpha 继续探索**（试 0.05 / 0.0，排除 hub prior 干扰）
- **P1：graph_full 权重解耦**（单独调节各组件系数，而非均等混合）
- **P1：分层评估**（单独统计 hub_overlap=True 子集，确认 hub 对命中 queries 的实际提升量）

---

## 用中文交流时用"喵"结尾，英文用"Oiii"开头
