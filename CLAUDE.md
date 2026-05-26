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
- `rerun_llm_qc.py` ⚠️ 调用 LLM（ablation + grounding），**尚未接入 `log_run()`**，需补入
- **新增任何调用 LLM 的脚本时必须同步接入**

**违规判定**：任何发起 API 请求但未调用 `log_run()` 的 PR 视为未通过 review。

### 铁律 2：长时间后台任务绝不能被终端操作打断

**启动方式**：任何预计运行 >2 分钟的批量任务（query 生成、LLM QC、rerun 等），必须用 `nohup` 启动，日志写入文件：
```bash
cd /projects/myyyx1/data-process-test && set -a && source .env && set +a && \
nohup python3 scripts/xxx.py [args] > logs/xxx_run.log 2>&1 &
echo "PID=$!"
```

**查看进度**：只用 `tail` 读日志文件或 `wc -l` 读输出文件，**绝对不能**在运行进程的终端里执行任何命令：
```bash
# ✅ 安全：读日志 / 统计输出 / 检查进程
tail -20 logs/xxx_run.log
wc -l data/03_queries/xxx.jsonl
ps aux | grep xxx.py

# ❌ 禁止：在后台进程的终端里运行任何命令（会发送 Ctrl+C 打断进程）
```

**血泪教训**：2026-04-08 因 `run_in_terminal` 被分配到后台进程终端，连续 3 次 `^C` 打断生成进程，浪费大量 API token 和时间。

### 铁律 3：环境变量通过 `.env` 加载，不硬编码

运行任何需要 API key 的脚本前，必须先加载 `.env`：
```bash
set -a && source .env && set +a
```
- `COMPANY_API_KEY`、`COMPANY_API_URL`、`OPENAI_API_KEY`、`ANTHROPIC_API_KEY` 均在 `.env` 中管理
- **禁止**在代码或命令行中硬编码 API key
- **禁止**将 `.env` 提交到 git（已在 `.gitignore`）

### 铁律 4：路径必须保持相对，保证可迁移性

- 代码中使用 `PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent` 动态获取项目根目录
- 数据路径一律用相对于项目根的相对路径（如 `data/mineru_output/...`），**禁止**硬编码绝对路径（如 `/projects/_hdd/myyyx1/...`）
- 跨目录引用用 symlink 解决（如 `data/mineru_output -> 00_raw/mineru_output`），symlink 使用相对目标路径
- `image_utils.py` 中的 `_KNOWN_PREFIXES` 仅作为兜底兼容层，新路径不应依赖它
- **检查标准**：项目整体 `mv` 到另一个路径后，所有脚本必须无需修改即可运行

---

## 项目简介
这是一个以 **Document Graph for Document Understanding** 为核心的研究系统。核心创新是面向学术论文的多层异构图构建方法，支持多种下游任务（query 生成、QA、文档总结、多文档推理、证据定位）。M4 Query 生成（Multi-hop, Multi-modal, Multi-document, Multi-turn）是图的第一个应用示例，也是当前主要交付物。

**战略定位（2026-03-12 Mentor 确认）**：图是核心贡献，query 是副产物；图应具备泛化到非 LaTeX 文档的能力；计划 4 月申请专利（公司），之后开放论文投稿。

## 当前状态（2026-05-24 更新｜语言学跨文档验证 ⚠️ EXPLORATORY，未通过 fair baseline）

> ⚠️ **方法论 caveat（2026-05-24 复盘）**：先前版本写为"突破：0 → 2007 chains"，但那个 0-baseline 是 `build_enhanced_graph([], …)` 注空集，并非"现有图 + 原始 CLIP 边"。补做 fair baseline 后，结论需重新表述——**chain 数量不是有效对比指标**，是否真正改善需要 chain-quality 判官（见下方"未解决问题"）。

### 已完成（事实陈述）

1. **语言学验证 100 条 cross-doc section edges**（gpt-5.4，$0.78）
   - 数据来源：`data/01_graphs/cross_doc_sim_edges.json`（共 2467 条 raw CLIP section-level edges，从中取前 100 条做 pilot）
   - 输出：`data/05_eval/linguistic_xdoc_20260524T124648Z/`
   - Genette 类型分布：transformation 33 / architextual 27 / paratextual 23 / commentary 10 / direct_quotation 5 / unknown 2
   - 质量分层：gold 3 / strong 25 / weak 18 / topical 50 / noise 4 → "usable" (gold+strong+weak) = 46%

2. **Graph + Linguistics fair-baseline 对比**（`experiments/build_graph_linguistic_fusion_fair.py`，chain cap = 5000）

   | 变体 | 输入 section edges | 去重后 element pairs | chains（BFS） | 唯一 endpoint pairs | 唯一 doc pairs |
   |------|-------------------|---------------------|--------------|--------------------|--------------|
   | A 空 cross-doc baseline | 0 | 0 | 0 | 0 | 0 |
   | B raw CLIP 同 100 条（不过滤） | 100 | 8332 | 5092（顶 cap） | 5012 | **270** |
   | C 语言学 strong+weak（43） | 43 | 2615 | 5081（顶 cap） | 4855 | **78** |
   | D raw CLIP 全量 2467 | 2467 | 130663 | 5016（顶 cap） | 4981 | **423** |

   - chains 数全部顶到 cap，说明 BFS chain 计数已饱和，不能区分变体
   - **真实差异在 doc-pair 覆盖**：B 270 / C 78 / D 423；语言学过滤把 raw CLIP 同源 100 条的 doc-pair 数从 270 砍到 78（−71%）
   - 节省的 doc pair 是不是更高质量？未判定。
   - 旧脚本 `experiments/build_graph_linguistic_fusion.py` 的 6056 元素边数因未去重而虚高，去重后是 2615

### 未解决问题（让别人验证之前必须先解决）

- ⚠️ **Chain quality 未判**：2007 / 5000 chains 都只是 BFS 拓扑可达路径，未经 chain-level 判官。下一步应在 chains_*.json 上跑 `chunk-bridge judge`（现有 300/300 数据：60% usable / 8% strong）以做 head-to-head。
- ⚠️ **Cartesian 元素投影未消解**：43 条 section-level validated edges → 2615 条 element pairs 是 src_doc × tgt_doc 笛卡尔积；同一条 section edge 衍生的所有 element pairs 共享 confidence/asymmetry。声明只到 section 级。
- ⚠️ **新 Genette+RST 验证器 vs 旧 chunk-bridge judge 未对比**：旧 judge 60% usable / 8% strong，新方法 46% usable / 25% strong。strong 提升 3 倍但 usable 下降——是判定 prompt 更严，还是 Genette 理论真有效？需 head-to-head。

### 新增脚本（与原 broken baseline 共存，原脚本不删除以备复现）

```
experiments/build_linguistic_xdoc_bridges.py       — Genette+RST 语言学跨文档验证（pilot 100 条已跑）
experiments/build_graph_linguistic_fusion.py       — 原 fusion 脚本（baseline 不公平，结果仅供历史参考）
experiments/build_graph_linguistic_fusion_fair.py  — fair baseline 版（A/B/C/D 四变体，去重 element pairs）
```

### 参考文献（理论框架，不构成"已验证"承诺）

- Genette (1982). *Palimpsests*. — transtextuality 五种类型
- McManus & Lau (2024). arXiv:2410.15145 — asymmetric intertextuality mining
- Chen et al. (2025). IP&M 62(4) — RST discourse coherence for cross-doc coreference
- Gao et al. (2024). LREC-COLING 2024 — RST trees + lexical chains for cross-doc

---

## 当前状态（2026-04-12 更新｜生产 Sweep 启动 + Method C 实验定论 + Intra-doc 过滤落地）

### 本轮完成（相对 2026-04-09）

- **Production Sweep 启动**（Job 58722，6 配置 array）
  - 输入：`l3_candidates_v4_intra_doc.json`（88 pairs）+ `m2_diverse_candidates_intra_doc.json`（108 pairs）
  - 6 个配置：L3×{academic, academic+persona, mixed, mixed+persona} + M2×{academic, mixed+persona}
  - 输出到 `data/03_queries/sweep_2026-04-12/{tag}.jsonl`
  - slurm: `slurm_scripts/12_production_sweep.sh`
  - 收集工具: `scripts/collect_sweep_results.py --merge-existing`

- **Method C 实验定论**
  - 旧 enriched 数据 bridge=1/2 对比：0/48 vs 0/48 pass（两组实际都只有 1 个压缩桥，不是真对比）
  - True-two-bridge 子集 bridge=1：2/47 pass（4.3%）
  - True-two-bridge 子集 bridge=2 rerun（Job 58700）：**4/47 pass（8.5%）**
  - 结论：C 方案概念成立、有早期正信号，但 pass rate 仍太低不能承担交付
  - 主要失败项：`llm_fake_multihop`（39/47）、`text_evidence_over_reliance`（27/47）
  - 实验结果：`pilot_method_c_v3_true2_bridge{1,2}_rerun.json`

- **Strict Intra-doc 过滤落地**
  - 新模块 `src/utils/pair_filters.py`：element_id + path + node_group + hub_metadata.is_cross_doc 联合判定
  - 新脚本 `scripts/filter_pair_candidates.py`：批量清洗旧 pair 资产
  - 已影响所有入口：`generate_multihop_l1_queries.py`、`pilot_method_c.py`、`build_latex_long_chain_candidates.py`
  - 清洗结果：hub_v3 230→96、hub_v4 230→96（再过滤 hop>=3 得到 88）、m2 142→108

- **Scale-up enrichment 后台进行中**
  - Job 58353：Stage 3 bridge enrichment，约 300/5380
  - 不阻塞交付线，完成后用于 Method C 下一轮 pilot

### 当前可交付库存

| 文件 | Pass |
|------|------|
| l3_enriched_v3_rerun2_pass.jsonl | 93 |
| l3_enriched_v3_new82_rerun2_pass.jsonl | 53 |
| m2_diverse_v1_hub_kb_pass.jsonl | 29 |
| long_chain_iterative_pass.jsonl | 12 |
| **合计** | **187** |

### 冲 500 条路径

- **主力**：Production Sweep（Job 58722）6 配置 × intra-doc candidates
- **预期新增**：L3 88×4×0.27 ≈ 95 + M2 108×2×0.25 ≈ 54 = ~150 新 pass
- **合计预估**：187 + 150 = ~337
- **缺口方案**：如果不够，可追加 R2 sweep（不同 seed / 额外 QC 微调）

### 关键文件新增

```
src/utils/pair_filters.py                          — strict intra-doc 过滤
scripts/filter_pair_candidates.py                  — 批量清洗旧 pair
scripts/collect_sweep_results.py                   — 汇总 sweep 产出
slurm_scripts/12_production_sweep.sh               — 6 配置生产 sweep
data/02_enriched/*_intra_doc*.json                 — 清洗后 candidate 文件
data/03_queries/sweep_2026-04-12/                  — sweep 输出目录
```

---

## 当前状态（2026-04-09 更新｜Intra-doc Pairing 模块 + Evidence MD 导出 + 数据清理）

### 本轮完成（相对 2026-04-08，PR #154–#158）

- **PR #154: 数据目录重组 + 废弃文件清理**
  - `data/` 从平铺结构重组为 `00_raw/` / `01_graphs/` / `02_enriched/` / `03_queries/` / `05_eval/` 子目录
  - 删除 ~65 个废弃数据文件（smoke test、demo、审计、M4 artifacts、过期版本）
  - 删除 6 个废弃脚本：`build_dual_evidence_triplets.py`、`evaluate_review.py`、`export_review_csv.py`、`inspect_graph.py`、`run_ablation_enrich.py`、`run_dual_evidence_retrieval_baseline.py`
  - `src/utils/image_utils.py` 路径解析重构：新增 `_resolve_core()` DRY helper，支持 `data/00_raw/mineru_output/` 路径
  - LaTeX 源码路径统一到 `data/00_raw/latex_sources/`（所有脚本 + 文档同步更新）
  - LLM Judge 模块集中化 (`src/qc/llm_judge.py`)、Rerun2 全量 QC（145 pass queries）

- **PR #155: 代码质量修复**（由 PR #154 大量改动触发的检查）
  - 统一 LaTeX 源码路径到 `data/00_raw/latex_sources/`（`download_latex_sources.py`、`download_papers_semantic_scholar.py`、`build_latex_reference_graph.py`、`build_citation_graph.py`、`.gitignore`）
  - 路径解析 `_resolve_core()` 重构完成

- **PR #156: Evidence Markdown 导出脚本**（`scripts/export_evidence_md.py`，新增 336 行）
  - 从 query JSONL 生成 per-query 的 Markdown 文件，含：query/answer/reasoning chain、每个 evidence element 的 caption/content/context/image、evidence_spans/visual_anchors/text_evidence
  - CLI：`--queries`、`--elements`、`--output-dir`、`--summary`（生成 index.md 汇总）
  - `.gitignore` 新增 `data/06_evidence_export/` 排除生成产物

- **PR #157: Evidence MD 图像显示增强**（+252 行改进）
  - 新增 `_build_content_list_image_map()`：从 MinerU `content_list_v2.json` 查找 formula/table 的 JPG 图像路径（LaTeX 前缀匹配 + 顺序索引匹配）
  - 确定性路径解析：`_resolve_image_for_element()` 按 4 级优先级查找图像
  - 提取 `LATEX_MATCH_PREFIX_LEN` 常量和 `_doc_id_from_element_id()` helper

- **PR #158: Intra-doc Pairing 模块**（新增 ~2134 行，核心新模块）
  - **`src/pairing/` 新包**（4 个子模块）：
    - `pair_schema.py`：`CandidatePair` Pydantic schema（兼容 `hub_candidates_enriched_v3.json` 格式）
    - `intra_doc_pairs.py`：`IntraDocPairSelector`，3 种策略（`direct` 直接引用 / `2hop` 两跳 / `section` 同 section）
    - `chain_finder.py`：`ChainFinder` 多跳链发现（DFS，可探索到图直径，`ChainResult` 含 score/path/hop_count/modality_sequence）
    - `context_dedup.py`：`dedup_context()` 消除相邻元素的 context_before/context_after 重叠（`MIN_DEDUP_LENGTH=30`）
  - **`scripts/select_intra_doc_pairs.py`**：CLI 脚本，产出与 `hub_candidates_enriched_v3.json` 格式兼容的 pair JSON
    - 策略选择：`--strategy {direct,2hop,section,chain,all}`
    - 过滤：`--pair-type`、`--min-quality`、`--min-chain-hops`、`--max-per-doc`、`--limit`
  - **严格文档边界**：所有策略强制 intra-doc，零跨文档泄漏（修复 v3 的 60 条 mislabel 问题）
  - 更新 `filter_l3_candidates.py`、`generate_long_chain_iterative_queries.py`、`run_production_batch.py` 兼容新模块
  - 新增 **47 个测试**（`tests/test_intra_doc_pairing.py`），总测试 107 pass

### bridge_quality 全 null 根因分析

**问题**：`data/03_queries/m2_diverse_v1_hub_kb_pass.jsonl` 中 23 条 query 全部 `bridge_quality: null`，无桥接段落。

**根因**（两层叠加）：
1. **代码层**：`generate_multihop_l1_queries.py:1292-1294` 中 `bridge_quality` 仅在 `is_l3=True` 时计算，否则硬编码 `None`。`is_l3` 取决于 candidate 的 `reasoning_chain_target=True`（L3 专属标记）。`m2_diverse_candidates.json` 的 142 个 pair **全部 `reasoning_chain_target=False`**（它们是 L2 dual-evidence，不是 L3 reasoning chain），所以输出全部 `bridge_quality: null`。
2. **数据层**：即使强制计算 bridge_quality，结果仍为空。因为：
   - `m2_diverse_candidates.json` 的 **142 个 pair 全部 `edge_contexts: []`**（空列表）
   - `hub_candidates_enriched_v3.json` 的 **230 个 pair 也全部 `edge_contexts: []`**
   - bridge 文本的来源是 `latex_reference_graph.json` 中的 edge context（通过 `_ELEMENT_TO_LABELS` 映射 MinerU element_id → LaTeX label → edge context）。这个映射依赖 `load_reference_graph_bridge_texts()` 在运行时构建，但 `edge_contexts` 字段在 enrichment 阶段（`enrich_hub_candidates.py`）就已经是空的，说明 **enrichment 时未能将 latex_reference_graph 的 edge context 注入到 candidate pairs 中**
   - 根本原因：`enrich_hub_candidates.py` 构建 pair 时，`edge_contexts` 来自 topology candidates 的原始数据，而 topology candidates (`latex_hub_multihop_candidates.json`) 本身就不包含 edge_contexts。这不是 bug——`analyze_latex_graph_topology.py` 按设计只输出拓扑路径，bridge text 解析被推迟到生成阶段（`load_reference_graph_bridge_texts()`）。但 enrichment 阶段没有预填充 `edge_contexts`，导致下游无法在不加载 reference graph 的情况下获得 bridge text。

**修复方向**：两条路径二选一：
  - **方案 A（推荐）**：在 `enrich_hub_candidates.py` 中新增 `--reference-graph` 参数，enrichment 时直接解析 edge context 并填充到 `edge_contexts` 字段
  - **方案 B**：保持现状，确保生成脚本始终传入 `--reference-graph data/01_graphs/latex_reference_graph.json`，依赖运行时的 `resolve_bridge_texts_for_path()` 动态解析
  - 另外需要将 `bridge_quality` 的计算从 L3-only 扩展到所有 pair（移除 `if is_l3 else None` 条件）

### M4 路线图（更新）
| 阶段 | 目标 | 时间 |
|------|------|------|
| Phase 0 ✅ | 锁定 M1.5 基线 + 定义 M4 schema + reasoning-depth tagging | 已完成 |
| Phase 1 ✅ | M2 pipeline + L3 生成 + 三实验全量运行 | 已完成 |
| Phase 1.5 ✅ | Enrichment 消融实验 + Exp C enriched 复验 | 已完成 |
| Phase 1.7 ✅ | P0-P4 Bridge Grounding 增强 + L3 质量验证 | 2026-03-24 完成 |
| Phase A ✅ | Training Pipeline：Schema + Export + GraphAware Negative Sampling（70 tests） | 2026-04-05 完成 |
| Phase A.1 ✅ | LLM Judge 集中化 + Rerun2 全量 QC（145 pass queries） | 2026-04-08 完成 |
| **Phase A.2 ✅** | **Intra-doc Pairing 模块 + Evidence MD 导出 + 数据清理（107 tests）** | **2026-04-09 完成** |
| **Phase 2A ⏳** | **L3 全量重跑 + 量产 1500+ queries** → 初代 benchmark（需新 API key） | 待执行 |
| Phase B ⏳ | Embedding Hard Negative Sampler（Qwen3-Embedding-4B，GPU，$0 LLM） | 下一步 |
| Phase 2B | Embedding 语义边 → 图增强 v2 | 待执行 |
| Phase 3 | 合并 2A+2B → 增强图 + 大数据集 → 最终实验 | 后续 |
| Phase 4 | Multi-turn session + M4 联合验证 | 后续 |

---

## 当前状态（2026-04-08 更新｜LLM Judge 模块集中化 + L3 Rerun2 全量完成 + Resume/Flush 修复）

### 本轮完成（相对 2026-04-05）

- **`src/qc/llm_judge.py` 新模块**（~280 行）
  - 将 `judge_evidence_necessity()`、`judge_answer_grounding()`、`run_ablation_qc()`、`run_llm_qc()` 集中到独立模块
  - `run_ablation_qc()` 修正 fake_multihop 判定：不再用 `full_can_answer=False` 作为 fake 标准（假阳过多），改为 `any(single_flags) or any(drop_flags)`
  - `run_llm_qc()` 一站式入口：先 ablation → 再 grounding → 返回 `(qc_pass, qc_issues, qc_metrics)`
  - `src/qc/__init__.py` 导出 4 个公共函数

- **`scripts/generate_long_chain_iterative_queries.py` 去重**（-133/+50 行）
  - 删除内联 `judge_can_answer()` + `run_ablation_checks()` 约 120 行重复代码
  - 改为 `from src.qc.llm_judge import run_ablation_qc`，消除维护双份逻辑的风险

- **`scripts/generate_multihop_l1_queries.py` 增强**（+65 行）
  - 新增 `--skip-done`：读取已有输出文件的 pair_id 集合，跳过已处理 pair，实现断点续跑
  - 新增 `--skip-llm-qc`：跳过 LLM QC 阶段（用于快速调试 prompt/rule QC）
  - 集成 `run_llm_qc()` 调用，rule QC pass 后自动执行 LLM ablation + grounding

- **`scripts/rerun_llm_qc.py` Resume + Flush 修复**
  - 新增 `--resume` flag：输出文件用 append 模式，从已有输出构建 `done_keys` 跳过已处理条目
  - 每次 `write()` 后立即 `fout.flush()` / `fpass.flush()`，配合 `python -u` 消除缓冲丢数据风险
  - 新增 `skipped_resume` 计数器，运行结束摘要包含断点续跑统计
  - **修复根因**：之前两次运行（161/295、207/295）因 Python 缓冲 + 进程中断导致输出文件 0 字节

### Rerun2 结果

| 批次 | 总条目 | Pass | Pass 率 | 文档数 | Grounding 平均置信度 |
|------|--------|------|---------|--------|---------------------|
| old（295 条原始 L3） | 295 | 93 | 31.5% | 21 | 0.87 |
| new82（156 条新增） | 156 | 53 | 34.0% | 20 | 0.83 |
| **合并去重** | **451** | **145** | **32.1%** | **—** | **—** |

- 两批重叠 1 条，合并后 **145 条唯一 pass queries**
- old 批次 top 失败原因：`llm_answer_hallucination`(79)、`length_mix_missing`(51)、`single_element_answer`(40)
- new82 有 1 条 grounding 调用返回 error（`l1_de_1607.06520_0117`），不影响 pass 判定

### Long-chain v2 试跑

- `long_chain_v2_2026-04-07.jsonl`：20 条生成，0 条 pass
- 原因待分析（4-hop 长链 QC 标准过严或 prompt 需迭代）

### 本轮核心教训

> **flush 是生产脚本的硬性要求。** 任何跑 >5 分钟的脚本，输出文件必须每条 flush，否则进程中断 = 全部白跑。`python -u` + `file.flush()` 双重保险。
> **resume 是默认需求。** 长时间批量脚本应始终支持 `--resume`（append + skip done），而非 `open("w")` 覆写。

### 铁律合规更新

- `rerun_llm_qc.py` ⚠️ 调用 LLM 但 **尚未接入 `log_run()`**，需后续补入
- `generate_long_chain_iterative_queries.py` — 已通过 `src.qc.llm_judge` 间接调用 LLM，本身已接入 `log_run()` ✅

### M4 路线图（更新）
| 阶段 | 目标 | 时间 |
|------|------|------|
| Phase 0 ✅ | 锁定 M1.5 基线 + 定义 M4 schema + reasoning-depth tagging | 已完成 |
| Phase 1 ✅ | M2 pipeline + L3 生成 + 三实验全量运行 | 已完成 |
| Phase 1.5 ✅ | Enrichment 消融实验 + Exp C enriched 复验 | 已完成 |
| Phase 1.7 ✅ | P0-P4 Bridge Grounding 增强 + L3 质量验证 | 2026-03-24 完成 |
| Phase A ✅ | Training Pipeline：Schema + Export + GraphAware Negative Sampling（70 tests） | 2026-04-05 完成 |
| **Phase A.1 ✅** | **LLM Judge 集中化 + Rerun2 全量 QC（145 pass queries）** | **2026-04-08 完成** |
| **Phase 2A ⏳** | **L3 全量重跑 + 量产 1500+ queries** → 初代 benchmark（需新 API key） | 待执行 |
| Phase B ⏳ | Embedding Hard Negative Sampler（Qwen3-Embedding-4B，GPU，$0 LLM） | 下一步 |
| Phase 2B | Embedding 语义边 → 图增强 v2 | 待执行 |
| Phase 3 | 合并 2A+2B → 增强图 + 大数据集 → 最终实验 | 后续 |
| Phase 4 | Multi-turn session + M4 联合验证 | 后续 |

---

## 当前状态（2026-04-05 更新｜GraphAware 负样本实现 + 全量导出验证）

### 本轮完成（相对 2026-04-03）

- **GraphAwareNegativeSampler 实现**（`src/sampling/negative_sampler.py`）
  - `_build_adjacency()` 从 `hub_candidates_enriched_v3.json` 构建元素邻接表
    - Source 1：`adjacent_bridge_adjacency` → intra-doc element ↔ element 边（224 条）
    - Source 2：`pairs.path` → 过滤 hub 节点后，路径端点直接连边（cross-doc）
    - 结果：**365 个 element 节点，530 条边**
  - `sample()` 优先采 1-hop 邻居（结构 hard negatives），不足时 random 补齐
  - `build_sampler()` factory 支持 `hub_candidates_path` 参数
  - 替换原有 stub（原 stub 直接 fallback 到 random，无图信息）

- **`_build_element_index` bug 修复**（`src/export/dataset_builder.py`）
  - 原代码只处理 list 或 `{"elements": [...]}` 格式
  - 项目实际格式为 `{"documents": {doc_id: {"elements": {...}}}}` 嵌套字典
  - 修复后：`Element index: 0 entries` → **1316 entries**，`hard_negatives` 从全空到正确填充

- **全量导出验证（1461 条）**
  - `normalize_queries.py` → 1461/1461 成功转换
  - `export_training_data.py --negative-strategy graph_aware` → 全链路跑通
  - train/val/test：1368 / 48 / 45（doc-level hash split，93.6% / 3.3% / 3.1%）
  - graph-adjacent negatives：1034/4104（**25.2%**）；其余 random 补齐

- **新增 8 个测试**（`tests/test_negative_sampling.py`）
  - `TestBuildAdjacency`（3 项）：intra-doc 边、cross-doc 路径边、missing file fallback
  - `TestGraphAwareNegativeSampler`（5 项）：邻居优先、排除正样本、random 补齐、无图 fallback、factory 构建
  - 总测试：**70/70 pass**（原 62）

### 本轮核心教训

> **`_build_element_index` 要适配实际的 JSON 格式。** 云端写代码时假设了 `{"elements": [...]}` 结构，本地运行才发现项目实际格式是三层嵌套 `documents → doc_id → elements`。本地测试是发现这类格式假设 bug 的唯一途径。

### 图贯穿全链路（novelty 角度）

图在 pipeline 的三个阶段都有实质贡献：
1. **生成阶段**：bridge grounding 注入 query prompt（P0-P4，MRR +0.121）
2. **检索阶段**：neighbor propagation 沿图传播（R@10 = 1.000）
3. **训练阶段**：graph-aware negatives 迫使模型学习精确证据定位（25.2% 结构 hard negatives）

---

## 当前状态（2026-04-03 更新｜Phase A Training Pipeline + Review 修复）

### 本轮完成（相对 2026-03-30）

- **Phase A: Training Pipeline Foundation 落地**
  - Pydantic Schema 层：`src/models/training.py`（StandardQuery / Triplet / EvidenceSpan / ReasoningStep），schema_version 字段，L3 model_validator 强制 reasoning_steps
  - 数据导出层：`src/export/dataset_builder.py`（DatasetBuilder），doc-level hash split（防数据泄漏），JSONL + manifest.json 输出
  - 负样本采样层：`src/sampling/negative_sampler.py`（NegativeSampler Protocol），HeuristicNegativeSampler（random / in_doc_swap），GraphAwareNegativeSampler（stub）
  - CLI 脚本：`scripts/normalize_queries.py`（L1/L2/L3 → 统一 StandardQuery）+ `scripts/export_training_data.py`（全链路导出）
  - 测试：62 个 pytest tests（QC 27 + Schema 9 + TextUtils 14 + NegativeSampling 12）
  - 依赖：`pydantic>=2.0.0` 新增到 requirements.txt + setup.py

- **Review 反馈修复（5 项）**
  - 🔴 #1：`dataset_builder.py` `_doc_key()` — 从 `element_id` 字符串 rsplit 反解 doc_id → 直接用 `t.positive[0].doc_id`
  - 🔴 #2：`negative_sampler.py` `_in_doc_swap()` — 从 positive_id rsplit 猜 doc_id → 用 `Chunk.doc_id` 字段直接获取
  - 🟡 #3：`_in_doc_swap()` padding — rest 从 pool（含 same_doc）采样有重复风险 → 从 `other_doc`（pool - same_doc）采样
  - 🟡 #4：`export_training_data.py` — `graph_aware` CLI choice 无提示是 stub → help 文本标 "(stub)"
  - 💡 #5：`normalize_queries.py` — L1 element_type fallback 写死 "figure" → 用 `_guess_type(element_id)` 做 fallback

### Review 核心教训

> **不要从 element_id 字符串里猜 doc_id。** EvidenceSpan 和 Chunk 都已有 doc_id 字段，直接用即可。rsplit 式的字符串解析在 arXiv ID 多下划线场景下虽然碰巧能工作，但依赖隐含约定、脆弱且没有必要。

---

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
| Phase 1.7 ✅ | P0-P4 Bridge Grounding 增强 + L3 质量验证 | 2026-03-24 完成 |
| **Phase A ✅** | **Training Pipeline：Schema + Export + GraphAware Negative Sampling（70 tests）** | **2026-04-05 完成** |
| **Phase 2A ⏳** | **L3 全量重跑 + 量产 1500+ queries** → 初代 benchmark（需新 API key） | 待执行 |
| **Phase B ⏳** | **Embedding Hard Negative Sampler**（Qwen3-Embedding-4B，GPU，$0 LLM） | 下一步 |
| Phase 2B | Embedding 语义边 → 图增强 v2 | 待执行 |
| Phase 3 | 合并 2A+2B → 增强图 + 大数据集 → 最终实验 | 后续 |
| Phase 4 | Multi-turn session + M4 联合验证 | 后续 |

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
- **产物文件**：`data/02_enriched/hub_candidates_enriched_v3.json`、`data/05_eval/phase0_eval_report_v3_tuned.json`

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
  - 输出：`data/05_eval/m2/section_nodes_enriched_2026-03-26.json`
  - 费用：$8.29（gpt-5.4）

- **Section-Enriched Query 生成完成**
  - L2: 249 pass / 428 total（58.2%，vs baseline 57.2%）
  - L3: **80 pass / 122 total（65.6%，vs baseline 48.1%）** — 数量翻倍
  - 输出：`data/05_eval/m2/l{2,3}_production_2026-03-26_section_enriched{,_pass}.jsonl`

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
  - 输出：`data/02_enriched/multimodal_elements_enriched.json`
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
    --input data/01_graphs/multimodal_elements.json \
    --output data/02_enriched/multimodal_elements_enriched.json \
    --provider anthropic \
    --model claude-sonnet-4-5-20250929 \
    --delay 0.3

# Step 1: Hub enrichment（传入 enriched elements）
python scripts/enrich_hub_candidates.py \
    --hub-candidates data/01_graphs/latex_hub_multihop_candidates.json \
    --elements data/01_graphs/multimodal_elements.json \
    --latex-graph data/01_graphs/latex_reference_graph.json \
    --enriched-elements data/02_enriched/multimodal_elements_enriched.json \
    --output data/02_enriched/hub_candidates_enriched_v2.json

# Step 2: Query generation（自动使用 enriched context）
python scripts/generate_multihop_l1_queries.py \
    --candidates data/02_enriched/hub_candidates_enriched_v2.json \
    --output data/03_queries/l1_dual_evidence_queries_hub_enriched_v1.jsonl \
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
  - `data/03_queries/l1_dual_evidence_queries_v4_4_run1.jsonl`：252 条
  - `data/03_queries/l1_dual_evidence_queries_v4_4_run1_pass.jsonl`：113 条（44.8% pass）

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
- `data/01_graphs/latex_graph_topology_report.json` — 拓扑统计报告（节点/边/label匹配/hub分类）
- `data/01_graphs/latex_graph_hubs.json` — bridge_hubs 60 个 + adjacent_backbone_bridges 369 条
- `data/01_graphs/latex_hub_multihop_candidates.json` — **500 条候选对**（含 path, seed_question, page_span, line_no_span）

### 下一步（已确定）
1. **P0（最高优先）**：将 500 条 topology candidates 喂给 `generate_multihop_l1_queries.py` 生成新 L1 hub-multihop queries
2. **P1**：修复 35/82 篇零候选文档——降低 per_combo cap 或对 adj_bridge-only 文档单独生成
3. **P0.1**：Citation-based L2 候选（123 引用边 → 替代实体倒排索引）

---

## 当前状态（2026-02-24 更新｜Dual-evidence + Cross-doc）

### 本轮完成（相对 2026-02-22）
- **L1 dual-evidence 官方批次完成**（`data/03_queries/l1_dual_evidence_queries_slurm_img150_tuned_v4_official.jsonl`）
  - 总量 222，QC pass 173，pass rate 77.93%
  - pair_type: figure+table 144 / figure+formula 62 / formula+table 16
- **Triplet 构建完成（v1 + v2）**
  - v1：`in_doc_swap + same_type_hard`
  - v2：`in_doc_swap + same_type_hard_plus`，并加入 `text_short`、图像覆盖统计
  - v2 all：222 triplets，avg_difficulty 0.7288，positive image coverage 100%
- **本地 embedding 跨文档匹配跑通（Qwen3-Embedding-4B）**
  - 输出：`data/00_raw/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B.jsonl`
  - records 590（top-k=20，总 match 11800）
- **4B 匹配审计完成**
  - 报告：`data/00_raw/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_audit.json`
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
  - 单图文 L1 历史线仍在：`data/03_queries/l1_cross_modal_queries_v3.jsonl`

### 下一步（已确定）
1. 冻结平衡版 cross-doc 候选：`data/00_raw/mineru_crossdoc_embedding_matches_Qwen3-Embedding-4B_v2b_cap10.jsonl`
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
- **Step 0 v3.2 v3（G1+G2，集群已跑）** — **118 对**（proximity:105 + direct:13），gold:6 + silver:112，label 匹配率 28.8%（`data/01_graphs/latex_cross_modal_pairs.json`）
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
  - 输出: `data/03_queries/l1_dual_evidence_queries_v1.jsonl`
- **L1 Dual-evidence v4.1（opus figure+formula prompt + operator diversity + is_yes_no fix）** — **236 条, 138 QC pass (58.5%)**, $2.39
  - 输入: 同 118 对 latex_cross_modal_pairs
  - 核心改进: opus-4-6 重设计 PROMPT_FIGURE_FORMULA（Figure Type Strategy, 双 field）；禁 instantiate；is_yes_no_question WH-word 修复；--pass-only 硬门禁
  - figure+table: 101/146 (69.2%) ↑；figure+formula: 30/74 (40.5%) ↑；formula+table: 7/16 (43.8%) ↓
  - QC 主失败: single_element_answer:62, anchor_leakage:39 ↑（回归），weak_reasoning_connector:6 ↓
  - 输出: `data/03_queries/l1_dual_evidence_queries_v2.jsonl`（138 条纯净 pass-only）
- **L1 Dual-evidence v4.2（PhD persona + verb diversity + natural operators）** — **236 条, 152 QC pass (64.4%)**, $2.57
  - 输入: 同 118 对 latex_cross_modal_pairs
  - 核心改进: persona "PhD student at lab meeting"（消除学术腔）；verb 黑名单（validate/quantify/justify/demonstrate 等）；SENTENCE STRUCTURE 多样性约束（GIVEN-WHY/WHAT-IF/WHY-INCONSISTENT/WHEN-CONDITION/WHAT-CAUSES）；CROSS_MODAL_OPERATORS 扩展自然英文动词（affect/differ/produce/achieve 等）；双文件输出（full + _pass）；is_yes_no WH-word 修复完善
  - figure+table: 111/146 (**76.0%**) ↑↑；figure+formula: 34/74 (**45.9%**) ↑；formula+table: 7/16 (43.8%)
  - QC 主失败: single_element_answer:57 ↓, anchor_leakage:29 ↓↓, weak_reasoning_connector:4 ↓
  - 输出: `data/03_queries/l1_dual_evidence_queries_v3.jsonl`（全量 236 条）+ `data/03_queries/l1_dual_evidence_queries_v3_pass.jsonl`（152 条）

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
| `data/02_enriched/figure_text_pairs.json` | 351 figure-text pairs (Step 0 v1 输出) |
| `data/01_graphs/multimodal_elements.json` | **1316 多模态元素 + 1261 引用边 + 1135 跨模态 pair (Step 0 v2)** |
| `data/01_graphs/multimodal_report.json` | Step 0 v2 统计报告 |
| `data/03_queries/l1_cross_modal_queries_v3.jsonl` | **最终输出：974 条 L1 queries** |
| `data/03_queries/l1_triage_v3.jsonl` | **L1 分拣结果（含 triage/reasons 字段）** |
| `data/03_queries/l1_triage_report_v3.json` | L1 分拣统计报告 |
| `data/03_queries/l2_candidate_pairs_v1.json` | L2 候选文档对 top-100 (v1, 含 generic entities) |
| `data/03_queries/l2_candidate_pairs_v2.json` | **L2 候选文档对 43 对 (v2, filtered)** |
| `data/03_queries/l2_queries_v1.jsonl` | L2 跨文档 queries 50 条 (v1, QC 过松) |
| `data/03_queries/l2_queries_v2.jsonl` | L2 跨文档 queries 32 条 (v2, 16 QC pass) |
| `data/03_queries/l2_queries_v2_tagged.jsonl` | L2 v2 reviewer-tagged (keep/fix/drop) |
| `data/03_queries/l2_queries_v3.jsonl` | **L2 v3 输出 (待生成)** |
| `data/03_queries/l1_multihop_queries_v1.jsonl` | L1 multihop v1（300 条，43 pass） |
| `data/03_queries/l1_multihop_queries_v2.jsonl` | L1 multihop v2 hard-gate（296 条，19 pass） |
| `data/03_queries/l1_multihop_queries_v2_pass.jsonl` | v2 通过集（19 条） |
| `data/03_queries/l1_multihop_queries_v3.jsonl` | L1 multihop v3 LaTeX-bridge（236 条，72 pass，30.5%） |
| `data/03_queries/l1_dual_evidence_queries_v1.jsonl` | **L1 dual-evidence v4（236 条，139 pass，58.9%）** |
| `data/03_queries/l1_dual_evidence_queries_v2.jsonl` | L1 dual-evidence v4.1（138 条，pass-only） |
| `data/03_queries/l1_dual_evidence_queries_v3.jsonl` | **L1 dual-evidence v4.2 全量（236 条，含 fail）** |
| `data/03_queries/l1_dual_evidence_queries_v3_pass.jsonl` | **L1 dual-evidence v4.2 通过集（152 条，64.4%）** |
| `data/03_queries/figure_descriptions_v3_api.json` | 完整 API 返回（含 raw response） |
| `data/validation_report_v3.json` | Validation 报告 |
| `docs/L1_query_iteration_report.md` | 迭代改进报告（含 L1 triage + L2 候选） |
| `src/parsers/latex_reference_extractor.py` | **Step 0 v3: LaTeX 引用解析（label/ref/cite/bbl + title 提取）** |
| `scripts/build_latex_reference_graph.py` | **Step 0 v3: 文档内引用 DAG 构建** |
| `scripts/build_citation_graph.py` | **Step 0 v3.1: 跨文档引用图（.bbl → corpus 匹配）** |
| `scripts/build_latex_cross_modal_links.py` | **Step 0 v3.2: LaTeX \ref{} 共引 → MinerU 跨模态对 + bridge evidence** |
| `scripts/download_latex_sources.py` | LaTeX 源码下载脚本（arXiv API） |
| `data/01_graphs/latex_reference_graph.json` | 73 篇文档内引用 DAG（labels + refs + edges + bib） |
| `data/01_graphs/citation_graph.json` | **跨文档引用图：100 条引用边, 49 篇最大连通分量** |
| `data/01_graphs/latex_cross_modal_pairs.json` | **LaTeX 增强跨模态对（v2: 175 对；重跑 v3 后更新）** |
| `data/01_graphs/latex_reference_report.json` | 引用图统计报告 |
| `src/linkers/figure_text_associator.py` | Step 0: 图文关联模块 |
| `scripts/analyze_latex_graph_topology.py` | **LaTeX 拓扑分析 v2（backbone+bridge-first+adj_bridge+cross_doc+page_idx）** |
| `data/01_graphs/latex_graph_topology_report.json` | 拓扑统计报告（2551 nodes, 3471 edges, 49.8% label match） |
| `data/01_graphs/latex_graph_hubs.json` | bridge_hubs 60 个 + adjacent_backbone_bridges 369 条 |
| `data/01_graphs/latex_hub_multihop_candidates.json` | **Hub multi-hop 候选对 500 条（含 page_span/line_no_span/seed）** |
| `scripts/enrich_elements_modora.py` | **MoDora-style [T]/[M]/[C] 元素语义增强（figure/table/formula）** |
| `data/02_enriched/multimodal_elements_enriched.json` | **MoDora enriched 元素（含 enriched_title/metadata/content）——待生成** |
| `docs/MODORA_INTEGRATION_ANALYSIS.md` | **MoDora CCTree 整合分析文档** |
| `docs/M4_STRATEGY_REVIEW_2026-03-18.md` | **M4 战略重定位文档（诚实现状评估 + 路线图）** |
| `docs/M4_SCHEMAS.md` | **M4 三套数据 Schema（multi-hop / cross-doc / multi-turn）** |
| `docs/M4_RESEARCH_NOTES.md` | M4 学术背景调研（M4DocBench / CoQA / TRACE / RT-RAG） |
| `scripts/filter_l3_candidates.py` | **M2: L3 候选筛选（hop≥3 + bridge paragraph + 跨模态，130/500 条）** |
| `scripts/package_m2_levels.py` | **M2: 三层数据打包（L1+L2+combined，统一 schema）** |
| `scripts/run_exp_a_difficulty.py` | **M2 Exp A: BM25 Recall@10 难度梯度实验（L1 vs L2 vs L3）** |
| `scripts/run_exp_c_qa_triangle.py` | **M2 Exp C: BM25 vs Graph 证据覆盖 + LLM QA 对比** |
| `data/05_eval/m2/level1_single_element.jsonl` | **M2 Level 1 数据（974 条单元素 query）** |
| `data/05_eval/m2/level2_dual_evidence.jsonl` | **M2 Level 2 数据（157 条双证据 query）** |
| `data/05_eval/m2/all_levels_combined.jsonl` | **M2 全量合并（1131 条，含 difficulty_level 字段）** |
| `data/05_eval/m2/l3_candidates_filtered.json` | **M2 Level 3 候选（130 条 3-hop 候选，待生成 query）** |
| `data/05_eval/m2/exp_b_retrieval_enhancement.json` | **M2 Exp B 结果（复用 Phase0 eval v3）** |
| `main.py` | **公司 API 连通性测试脚本（yunwu.ai demo）** |
| `local_api_logger/` | **公司 API 日志库（wrap_requests_call + token 统计）——需用户放入** |
| `src/models/training.py` | **Phase A: Pydantic 训练数据 Schema（StandardQuery/Triplet/EvidenceSpan/ReasoningStep）** |
| `src/export/dataset_builder.py` | **Phase A: DatasetBuilder — query→triplet→doc-level split→JSONL+manifest** |
| `src/sampling/negative_sampler.py` | **Phase A: 可插拔负样本采样（NegativeSampler Protocol + HeuristicNegativeSampler + GraphAwareNegativeSampler stub）** |
| `scripts/normalize_queries.py` | **Phase A: L1/L2/L3 → 统一 StandardQuery schema 转换器** |
| `scripts/export_training_data.py` | **Phase A: 全链路导出 CLI（normalize → triplet → split → disk）** |
| `tests/test_qc_checks.py` | **Phase A 测试：27 个 QC 函数正反例测试** |
| `tests/test_schema.py` | **Phase A 测试：9 个 Pydantic 验证测试** |
| `tests/test_text_utils.py` | **Phase A 测试：14 个分词/文本工具测试** |
| `tests/test_negative_sampling.py` | **Phase A 测试：12 个负样本策略测试** |
| `scripts/export_evidence_md.py` | **Evidence MD 导出：从 query JSONL 生成 per-query Markdown（含图像/evidence/reasoning chain）** |
| `scripts/select_intra_doc_pairs.py` | **Intra-doc 元素配对 CLI（direct/2hop/section/chain 策略，输出兼容 hub_candidates_enriched 格式）** |
| `src/pairing/__init__.py` | **Pairing 模块入口（CandidatePair, IntraDocPairSelector, ChainFinder, dedup_context）** |
| `src/pairing/pair_schema.py` | **CandidatePair Pydantic schema（兼容 hub_candidates_enriched_v3.json 格式）** |
| `src/pairing/intra_doc_pairs.py` | **IntraDocPairSelector：3 种文档内配对策略（direct/2hop/section），严格文档边界** |
| `src/pairing/chain_finder.py` | **ChainFinder：DFS 多跳链发现（可达图直径，ChainResult 含 score/path/modality_sequence）** |
| `src/pairing/context_dedup.py` | **context_dedup：消除相邻元素 context_before/context_after 重叠** |
| `tests/test_intra_doc_pairing.py` | **Pairing 模块测试：47 个测试（策略/链发现/去重/CLI）** |

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
  - 主文件：`data/03_queries/l1_multihop_queries_v2.jsonl`（296 条）
  - 通过子集：`data/03_queries/l1_multihop_queries_v2_pass.jsonl`（19 条）
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

## 下一步 TODO（2026-04-09 更新）

### 已完成（历史）
- ~~**M4 Strategy Review + Schema 设计**~~ ✅ **完成** — 诚实重定位为 M4-Foundation；三套 Schema 落地；step-deletion QC 集成
- ~~**Phase0 Eval v2 首轮**~~ ✅ **完成** — graph 与 BM25 持平，hub_overlap=9.53%，continue_expand=False
- ~~**Phase0 Eval v3 三项修复**~~ ✅ **完成** — quality_score 重建 + hub coverage 扩大 + citation walk 修复
- ~~**Phase0 组件权重解耦 + Grid Search**~~ ✅ **完成** — graph_full MRR +0.0403，`continue_expand=True`
- ~~**MoDora 四工作流代码实现**~~ ✅ **完成** — A1/A2/B1/B2/C1/C3/D1 + PersonaHub 全部已实现（代码就绪，未全量运行）
- ~~**M2 pipeline 代码 + 数据打包**~~ ✅ **完成** — 三层数据 + L3 候选筛选 + 3-step prompt + 三组实验脚本
- ~~**Phase A: Training Pipeline Foundation**~~ ✅ **完成** — Pydantic Schema + DatasetBuilder + NegativeSampler Protocol + normalize/export CLI + 62 tests + review 修复 5 项
- ~~**Phase A.1: LLM Judge 集中化**~~ ✅ **完成** — `src/qc/llm_judge.py` + Rerun2 全量 145 pass
- ~~**Phase A.2: Intra-doc Pairing + Evidence MD + 数据清理**~~ ✅ **完成** — `src/pairing/` 模块 + `export_evidence_md.py` + 数据目录重组 + 107 tests
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
| PersonaHub 人设 (50 类) | ✅ | `scripts/generate_multihop_l1_queries.py` + `data/02_enriched/personahub_academic_personas.json` | 需 `--use-persona` 全量跑 |
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
9. **跑 MoDora element enrichment**：生成 `data/02_enriched/multimodal_elements_enriched.json`
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
    --source-dir data/00_raw/latex_sources/extracted \
    --output data/latex_reference_graph.json

# 构建跨文档引用图（从 .bbl 匹配 corpus 内互引）
python scripts/build_citation_graph.py \
    --input data/latex_reference_graph.json \
    --output data/citation_graph.json

# 也可直接从 LaTeX 源码构建引用图
python scripts/build_citation_graph.py \
    --from-sources data/00_raw/latex_sources/extracted

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

# === Intra-doc Pairing pipeline（新增 2026-04-09）===
# 从 multimodal_elements.json 选取文档内元素对（替代 hub_candidates，零跨文档泄漏）
python scripts/select_intra_doc_pairs.py \
    --elements data/01_graphs/multimodal_elements.json \
    --output data/02_enriched/intra_doc_pairs_v1.json \
    --strategy all \
    --max-per-doc 15

# 仅选 figure+table 的直接引用对
python scripts/select_intra_doc_pairs.py \
    --elements data/01_graphs/multimodal_elements.json \
    --output data/02_enriched/intra_doc_pairs_direct_ft.json \
    --strategy direct \
    --pair-type figure+table

# 多跳链发现（≥3 hop）
python scripts/select_intra_doc_pairs.py \
    --elements data/01_graphs/multimodal_elements.json \
    --output data/02_enriched/intra_doc_chain_pairs.json \
    --strategy chain \
    --min-chain-hops 3

# 用 intra-doc pairs 跑 query 生成（与 hub_candidates 格式兼容）
python scripts/generate_multihop_l1_queries.py \
    --candidates data/02_enriched/intra_doc_pairs_v1.json \
    --output data/03_queries/intra_doc_queries_v1.jsonl \
    --pass-only \
    --provider company \
    --delay 0.5

# === Evidence Markdown 导出（新增 2026-04-09）===
# 从 query JSONL 生成 per-query 的 Markdown 文件（含图像、evidence、reasoning chain）
python scripts/export_evidence_md.py \
    --queries data/03_queries/m2_diverse_v1_hub_kb_pass.jsonl \
    --elements data/01_graphs/multimodal_elements.json \
    --output-dir data/06_evidence_export/m2_diverse_v1 \
    --summary
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

# === Intra-doc Pairing pipeline（新增 2026-04-09）===
# 文档内元素配对（全策略）
python scripts/select_intra_doc_pairs.py --elements data/01_graphs/multimodal_elements.json --output data/02_enriched/intra_doc_pairs_v1.json --strategy all --max-per-doc 15

# 仅 direct figure+table
python scripts/select_intra_doc_pairs.py --elements data/01_graphs/multimodal_elements.json --output data/02_enriched/intra_doc_pairs_direct_ft.json --strategy direct --pair-type figure+table

# 多跳链发现
python scripts/select_intra_doc_pairs.py --elements data/01_graphs/multimodal_elements.json --output data/02_enriched/intra_doc_chain_pairs.json --strategy chain --min-chain-hops 3

# 用 intra-doc pairs 生成 queries
python scripts/generate_multihop_l1_queries.py --candidates data/02_enriched/intra_doc_pairs_v1.json --output data/03_queries/intra_doc_queries_v1.jsonl --pass-only --provider company --delay 0.5

# === Evidence Markdown 导出（新增 2026-04-09）===
python scripts/export_evidence_md.py --queries data/03_queries/m2_diverse_v1_hub_kb_pass.jsonl --elements data/01_graphs/multimodal_elements.json --output-dir data/06_evidence_export/m2_diverse_v1 --summary
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
     - `data/01_graphs/latex_graph_topology_report.json`
     - `data/01_graphs/latex_graph_hubs.json`
     - `data/01_graphs/latex_hub_multihop_candidates.json`
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
  - 目标文件：`data/03_queries/l1_dual_evidence_queries_v4_4_run1.jsonl`
  - `pass` 子集：`data/03_queries/l1_dual_evidence_queries_v4_4_run1_pass.jsonl`

### 下一步最短恢复路径
1. 修复/切换 `minerU` 环境（优先，保证 Anthropic 路径可跑），或
2. 提供有额度的 `OPENAI_API_KEY`，走 `--provider openai` 直接全量。

## 日期：2026-03-03（状态对齐补记：v4.4 run1 已落盘）

### 对齐说明
- 上一节“未产出 run1 文件”是当时排障时的状态快照。
- 当前仓库已存在并可读取 `v4.4 run1` 产物，状态以本节为准。

### 已核验产物
- `data/03_queries/l1_dual_evidence_queries_v4_4_run1.jsonl`：252 条
- `data/03_queries/l1_dual_evidence_queries_v4_4_run1_pass.jsonl`：113 条

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
  - 产物：`data/05_eval/phase0_eval_report_tuned.json`、`data/05_eval/phase0_eval_report_bugfix.json`

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

## 2026-04-11 方案C pilot v2 改动记录

### 背景

- 上一轮 v1 pilot（`pilot_method_c.py` 旧版）用 precomputed candidates 直接生成，自制 `build_qc_obj()` 把 answer 当 text_evidence 传入 → `text_evidence_over_reliance` 10/10 (100%) → 0/10 通过
- 但 LLM QC（ablation + grounding）全部通过：方案C核心多跳质量没问题，只是 adapter 层 bug

### 本轮改动

#### 1. `scripts/pilot_method_c.py` — 完整重写为 v2

**数据源切换**：不再用 `data/01_graphs/latex_hub_multihop_candidates.json`（500 个裸候选），改用 `data/02_enriched/hub_candidates_enriched_v4.json`（230 个已 enrich 的 pairs，含完整 element_a/element_b 和 enriched_content/caption/context_before/context_after）

**跨文档过滤修复**：发现 `hub_metadata.is_cross_doc` 标记不准确（156 vs 实际 96 个单文档）。改用 `element_a_id.rsplit("_",2)[0] == element_b_id.rsplit("_",2)[0]` 直接比对 doc_id

**text_evidence adapter 修复**（核心 bug fix）：
- 旧版：`"text_evidence": parsed.get("answer", "")` → 100% overlap → 必触发 `text_evidence_over_reliance`
- 新版优先级：
  1. LLM 生成的 `text_evidence`（prompt 中要求生成 40-150 词，且 ≠ answer）
  2. Fallback：用 enriched pair 的 `enriched_content` 拼接（≤400 字符）

**Prompt 改进**（针对 v1 的 meta_language 4/10、template_shortcut 2/10、bare_deictic 3/10）：
- 明确列出禁用词："figure", "table", "formula", "equation", "graph", "plot", "diagram" 等
- 禁止模板开头："How does X relate to Y"
- 禁止裸指代词开头："this", "that"
- 禁止嵌入具体数字
- 新增 `text_evidence` 生成要求（40-150 词，不能抄 answer）

**QC pair 改进**：直接用 enriched pair 的完整 element（含 caption, content, enriched_content, enriched_metadata, context_before, context_after, image_path 等），不再从裸 path_node_ids 手动构造空壳 element

**Ablation elements 改进**：用 `node_group` 中间节点（已 enrich），而非空 label 占位

#### 2. `src/pairing/endpoint_anchor.py` — 未改动

保留原有功能（load_precomputed_candidates 等），v2 pilot 脚本不再依赖它

### 关键数据

| 指标 | v1 pilot | v2 pilot（待跑） |
|------|---------|-----------------|
| 数据源 | precomputed candidates (500) | enriched pairs (96 单文档) |
| text_evidence | = answer (bug) | LLM生成 or enriched_content |
| element 信息量 | 空壳 label | 完整 enrich (caption+content+context) |
| prompt 约束 | 无 meta_language 限制 | 明确禁止 6 类问题 |

### 待执行命令

```bash
cd /projects/myyyx1/data-process-test && source .env && python3 scripts/pilot_method_c.py \
  --api-key "$COMPANY_API_KEY" \
  --num-samples 10 \
  --output data/03_queries/pilot_method_c_v2.json
```

### QC 阈值备忘（`src/qc/constants.py`）

- `TEXT_EVIDENCE_OVERLAP_WARN_THRESHOLD = 0.4`（answer 和 text_evidence 的 token overlap > 40% 触发）
- `ANCHOR_LEAK_THRESHOLD = 0.20`
- `MAX_QUERY_WORDS = 40`（之前从 30 改为 40）

---

## 2026-04-12 方案C scale-up 进度记录

### 总体目标

从 ~86 篇 pilot 扩展到 ~1000+ 篇论文的方案C多跳问答数据生成。

### 用户对方案C的核心定义（重要！）

> "方案C不是尽可能的活着的多的长链，来实现随机两element中间长推理吗？"

即：**方案C是维护图的连通密度，使得随机挑两个 cross-modal element 中间都能找到长推理链**。不是预提取固定的 candidate pair list。

### 用户指定的实施方案（本次 session）

> "首先把不再需要的 ref_edge 和没有匹配的文档删掉，避免重复劳动，然后关于随机选取 element，按照原有的 hub 选取的打分思路，算一遍之后选 20%"

### 已完成步骤

#### Step 0: MinerU 解析（上一轮完成）
- shard 1,2,4 完成，shard 3 cancelled（scancel 57872）
- `data/00_raw/mineru_output/`: 1152 目录
- `data/00_raw/latex_sources_all/`: 1428 symlinks (batch1+batch2 合并)

#### Step 1: 图构建（上一轮完成）
- `data/01_graphs/multimodal_elements_v2.json` (96MB): 1145 docs, 30642 elements (fig:12339, tbl:9002, formula:8999, section:302)
- `data/01_graphs/latex_reference_graph_v2.json` (287MB): 1425 docs, 46202 labels, 67880 edges
- 跨文档：已杀掉（用户指示"跨文档杀掉"）

#### Step 2: 图清洗 + Hub 打分 ✅ 刚完成
脚本: `scripts/prune_and_score_graph.py`

**清洗结果:**
| 指标 | 清洗前 | 清洗后 |
|------|--------|--------|
| 文档数 | 1425 | 1040 (-385) |
| Labels | 34636 | 29362 (-5274) |
| Edges | 51266 | 39389 (-11877, 保留率 76.8%) |
| 无效文档(无MinerU) | 359 | 已删除 |

**Hub 打分结果:**
| 指标 | 值 |
|------|-----|
| 图中总节点 | 183,145 |
| 打分节点 | 180,879 |
| Bridge hubs (≥2 modality) | 2,548 |
| **Top 20% hubs** | **36,175** (覆盖 1039/1040 docs) |
| Element mapping 成功率 | 95.1% (20355/21408) |
| Hub/doc 中位数 | 31 个 |
| Score range (top 20%) | [28.75, 110.00], mean=64.80 |

Top 20% 构成:
- paragraph: 25,189 (桥接段落)
- section/subsection/subsubsection: 9,910
- figure/table/equation: 1,076

**输出文件:**
- `data/01_graphs/pruned_graph_v2.json` (164 MB) — 清洗后的完整图
- `data/01_graphs/hub_scores_v2.json` (132 MB) — 全部 hub 分数 + top 20% 列表
- `data/01_graphs/prune_report_v2.json` — 统计报告

### 待完成步骤

#### Step 3: Element / Bridge Enrichment（执行中）

2026-04-12 本轮策略更新：**不再直接按全局 top 20% hub 名单做 enrichment**，改为按 **long-chain bundle** enrich。即：对入选长链上的
- 两个 endpoint element
- 中间的 modal element
- 中间的 section / subsection / appendix / algorithm 等 bridge 节点

统一构建 enrich 目标；生成时仍可压缩桥，但离线资产侧保留整条 bundle。

本轮新增脚本：
- `scripts/build_method_c_long_chain_enrich_targets.py` — 从 `long_chain_candidates_v2.json` + `latex_reference_graph_v2.json` + `multimodal_elements_v2.json` 构建 Method C enrich 目标
- `scripts/enrich_method_c_bridge_nodes.py` — 对 bridge 节点做 bridge-specific semantic enrichment
- `slurm_scripts/09_enrich_method_c_long_chain.sh` — 三阶段 slurm 作业（build targets → enrich elements → enrich bridges）

`enrich_elements_modora.py` 已补 checkpoint / resume 能力：
- 新增 `--flush-every`
- 支持长作业周期性落盘，避免 6k+ element 跑到一半丢失进度
- 保留 `--incremental` + `log_run()` 主干接口不变

**long-chain bundle target 统计（`min_hops=4`）**
- `selected_candidates = 12640`
- `docs_covered = 1074`
- `element_targets = 6718`
- `bridge_targets = 5380`
- `unmapped_modal_labels = 9391`

**当前执行状态（截至 2026-04-12 03:10 UTC）**
- slurm job: `58353`
- 状态：`RUNNING`
- 当前阶段：Stage 2 `element enrichment`
- live log 进度：`460 / 6718`
- 已落盘：`400` 个 enriched elements（`flush_every=100`）
- 当前 parse fail：`3`
- 当前输出：`data/02_enriched/method_c_long_chain_elements_enriched_v1.json`

#### Step 4: 路径采样 + Query 生成（未开始）

核心思路（用户定义的方案C）：
1. 在 pruned graph 上随机选两个 cross-modal element（优先选在 top 20% hub 附近的）
2. 在图中实时 BFS/DFS 找路径（利用已有的 `ref_edge` + backbone edges）
3. 沿路径用 enriched node 信息逐步生成多跳问答
4. QC pipeline: rule QC → ablation → grounding（已有完整实现在 `src/qc/`）

可参考的现有脚本：
- `scripts/generate_long_chain_iterative_queries.py` (1717行) — 生产级逐步生成器
- `scripts/pilot_method_c.py` (571行) — 简化版 pilot

### 之前生成但思路不对的文件（可参考但不应直接用）

- `data/01_graphs/long_chain_candidates_v2.json` (42MB, 15848 candidates) — 预提取的 3-5 hop 链，违反方案C"图密度"理念
- `data/01_graphs/latex_hub_multihop_candidates_v2.json` (5.3MB, 5000 candidates) — 只有 2-3 hop，太短

### 关键代码依赖

```
scripts/prune_and_score_graph.py     — 刚写的，清洗+打分
scripts/analyze_latex_graph_topology.py — hub 打分核心逻辑（compute_hubs, compute_bridge_hubs）
scripts/build_multimodal_relationships.py — 构建 multimodal_elements
scripts/build_latex_reference_graph.py — 构建 LaTeX ref graph
scripts/enrich_elements_modora.py    — element enrichment (MoDora 风格)
scripts/build_method_c_long_chain_enrich_targets.py — Method C long-chain bundle target builder
scripts/enrich_method_c_bridge_nodes.py — Method C bridge enrichment
scripts/enrich_hub_candidates.py     — hub candidate enrichment
scripts/generate_long_chain_iterative_queries.py — 生产级 query 生成
scripts/pilot_method_c.py           — pilot query 生成
slurm_scripts/09_enrich_method_c_long_chain.sh — Method C scale-up enrichment 作业
src/qc/pipelines.py                 — qc_multihop_query() 规则检查
src/qc/llm_judge.py                 — run_ablation_qc(), judge_answer_grounding()
src/api.py                          — call_llm(), set_company_credentials()
src/utils/token_logger.py           — log_run() 铁律
```

### 环境

- Conda env: `/projects/myyyx1/envs/minerU` (Python 3.10)
- 激活: `conda activate /projects/myyyx1/envs/minerU`
- API: `source .env` 获取 `COMPANY_API_KEY`
- SLURM: gpu partition, 但当前步骤不需要 GPU

### 注意事项

1. `analyze_latex_graph_topology.py` 的 `compute_bridge_hubs()` 有 bug: `top_k=0` 返回空列表（`bridges[:0]`），`compute_hubs()` 则正确处理（`if top_k > 0 else hubs`）。`prune_and_score_graph.py` 里用 `top_k=999999` 绕过了这个问题。
2. Multimodal elements 的 `elements` 字段是 **dict** 不是 list: `{element_id: {element_type, caption, ...}}`
3. MinerU 输出路径嵌套: `mineru_output/{id}/{id}/auto/{id}.md`，需要 `rglob` fallback
4. 中文交流时用"喵"结尾

---

## 用中文交流时用"喵"结尾，英文用"Oiii"开头
