# 解耦计划 SOT — 2026-04-22

**状态**：Round 1 已落地（本 PR），Round 2 待跟进。

本文档是执行解耦重构的**单一真相源（Source of Truth）**。任何后续调整必须先更新本文件再动代码。

---

## 两套方案综合（Plan A + Plan B）

**骨架取 Plan B**（分层边界：`src/` = 库 / `experiments/` = 一次性 / `scripts/` = 薄壳）。
**吸收 Plan A 两点具体动作**：

1. `src/api/__init__.py` → 拆出 `src/api/llm.py`（本 PR Phase 2）
2. `src/retrieval/__init__.py` → 拆出 `src/retrieval/bm25.py + metrics.py`（本 PR Phase 4）

---

## 目录边界（判定代码归属的硬标准）

```
src/         可复用库：纯函数 / 轻状态、不依赖任何 data/ 路径常量、不带 CLI
experiments/ 一次性实验：pilot_*, run_exp_*, ablations、可写死路径
scripts/     生产 runner：argparse + 加载 src/ + 写产物，单文件 ≤ 300 行
slurm_scripts/ 不动
archive/     历史代码，只读
```

**判定一段代码归哪儿的三条硬规则**：

- 被 ≥2 个脚本 import 或 copy → `src/`
- 一次性 ablation / 论文实验 / 文件名带日期 / `pilot` / `exp` / `trial` / `gold` / `feasibility` → `experiments/`
- 上面两条都不是 → `scripts/`，且必须是薄壳

---

## 铁律（追加到 CLAUDE.md 之外的硬规则）

### 硬规则 R1 — `local_api_logger` 不得绕过

`company` provider 的实际 HTTP 请求**必须全部走** `local_api_logger.wrap_requests_call()`。
严格禁止：

- 在 `src/api/llm.py` 或任何脚本里直接 `requests.post` / `urllib.request` 调
  `yunwu.ai` 或任何公司代理 endpoint
- 用 `openai.OpenAI(base_url="yunwu.ai")` 之类的写法绕开 `local_api_logger`

违反 R1 = PR 自动不合格。

### 硬规则 R2 — `token_logger.log_run()` 是唯一的记账入口

`src/utils/token_usage_logger.py` 里的 `TokenUsageLogger` 只是 per-call JSONL
**补充**，不能替代 `log_run()`。任何调 LLM API 的脚本必须在结束时调用
`log_run()`（CLAUDE.md 铁律 1），不管有没有同时用 `TokenUsageLogger`。

### 硬规则 R3 — 每 Phase 一 commit + 必须跑 pytest

- 每个 Phase 结束必须跑 `pytest tests/`，绿了再进下一阶段
- 红了 revert，不要带病推进
- `commit message` 必须以 `Phase N: ` 开头
- 不允许一个 commit 塞多个 Phase

### 硬规则 R4 — 不允许顺手改业务

重构阶段发现 bug 写到 `docs/4.22-decoupling-notes.md` 留到下一 PR，**绝不边抽边改**。
例外：如果 bug 是测试/导入阻塞类的结构问题（比如本 PR 修的 `judge_evidence_necessity`
phantom re-export），可以作为"建立绿色基线"动作随 Phase 0 一起修。

### 硬规则 R5 — 路径必须相对

CLAUDE.md 铁律 4 延伸：`src/io/paths.py` 必须用 `Path(__file__).resolve().parents[2]`
而不是硬编码。项目整体 `mv` 到别的路径后所有脚本必须无需修改即可运行。

### 硬规则 R6 — god script 拆解严格三步走

1. 先在原文件里抽函数（纯粹 cut-paste，逻辑不变）
2. 跑原脚本验证产物与拆解前 byte 级一致（或 diff 只差时间戳）
3. 再 import 新模块，删掉原地的拷贝

**不允许"边抽边改逻辑"**。对 2019 行的 `analyze_latex_graph_topology.py` 和 1719/1388
行的 generate scripts 拆解时额外要求：冻结一个 golden input（如
`latex_hub_multihop_candidates.json`）做产物回归。

---

## 阶段表

### Round 1 — 本 PR 已完成 ✅

| Phase | 动作 | 文件 |
|------|------|------|
| 0 | 写 SOT 文档 + 修 pytest 基线（删 `judge_evidence_necessity` phantom export） | 本文件 + `src/qc/__init__.py` |
| 0.5 | 删 `src/pairing/endpoint_anchor.py`（576 行死代码，0 个外部 importer） | — |
| 1 | 新增 `src/io/{jsonl, paths}.py`；jsonl 委托到 `src/utils/file_utils` 保持后向兼容 | `src/io/` |
| 2a | `src/api/__init__.py`（273 行）→ `src/api/llm.py`；`__init__.py` 只 re-export | `src/api/` |
| 2b | 新增 `src/cli/common_args.py` 统一 `--provider/--model/--delay/--dry-run` 等 | `src/cli/` |
| 2c | `token_usage_logger.py` 加 deprecation 文档（仍保留，它是 per-call 补充不是替代） | `src/utils/token_usage_logger.py` |
| 4 | `src/retrieval/__init__.py`（92 行）→ `bm25.py` + `metrics.py`；re-export | `src/retrieval/` |
| 7 | 建 `experiments/__init__.py` 作 stub，文档化哪些脚本在 Round 2 移动 | `experiments/` |

### Round 2 — 待跟进 ⏳

| Phase | 动作 | 风险 | 依赖 |
|------|------|------|------|
| 1b | 迁移 24 个 `load_jsonl` / 7 个 `write_jsonl` 调用方到 `src.io.jsonl` | 中 | R3 绿测试 |
| 1c | 加 `src/io/element_loader.py` 统一 `load_multimodal_elements()` 返回 dict-of-dict 格式（**必须先贴实际 JSON schema**，文件是 `documents → doc_id → elements` 三层嵌套，memories 有 `_build_element_index` bug 教训） | 中 | 先验证 schema |
| 1d | 加 `src/io/graph_loader.py` 统一 `load_reference_graph()` / `load_citation_graph()` / `load_hub_scores()` | 中 | — |
| 2d | 让 `generate_*` / `enrich_*` 脚本用 `src.cli.add_llm_args` 注册 CLI | 低 | 灰度一次一个 |
| 2e | 补 `rerun_llm_qc.py` 的 `log_run()` 调用（CLAUDE.md 已标 "尚未接入"） | 低 | — |
| 7b | 物理移动实验脚本到 `experiments/`，**同步更新**：`slurm_scripts/` 中 10/11/23/37b/40 号作业，以及 `scripts/build_cpool_proxy_queries.py` / `validate_and_project_crossdoc.py` / `eval_chunk_graph_rerank.py` / `method_c_auto_followup.py` / `run_m2_classic_eval_oneclick.sh` / `run_phase0_grid.sh` / `run_exp_a_difficulty.py` / `eval_cpool_keyword_boost_graph.py` / `build_embedding_edges.py` 的路径 | 中 | atomic commit |
| 8 | 合并明显冗余（见下方） | 低 | — |

### Round 3 — 深耕，排期未定

- Phase 3：拆 `src/prompts/templates.py` 899 行 → `academic.py` + `real_user.py` + `builders.py`
- Phase 5：从 `analyze_latex_graph_topology.py` 2019 行抽 `src/graph/{topology, citation, scoring}.py`（**必须先冻结 golden input 做 byte 级回归**）
- Phase 6：从 `generate_long_chain_iterative_queries.py` 1719 行 + `generate_multihop_l1_queries.py` 1388 行抽 `src/generation/{iterator, checkpoint}.py`

---

## Round 2 Phase 8 — 明显冗余清单

按 R6 三步走规则处理，每项独立 commit：

- `scripts/build_context_aug_corpus.py` vs `build_context_aug_corpus_v2.py`：确认 v2 已完全替代，把 v1 移到 `archive/scripts/`
- `scripts/build_delivery_zip.py` (449) vs `build_full_delivery.py` (879) vs `package_delivery.py` (204)：抽公共骨架到 `src/export/delivery.py`，3 个脚本只保留差异
- `evaluate_evidence_localization.py` vs `evaluate_evidence_localization_stdlib.py`：保留 stdlib 版（无外部依赖更稳）
- `eval_dense_retrieval.py` vs `eval_dense_retrieval_hf.py`：合并到一个脚本 + `--backend {sentence-transformers,hf}`
- `src/linkers/multimodal_relationship_builder.py` 审一遍归属 —— `src/linkers/` 只剩这一个文件，要么并入 `src/pairing/`，要么并入未来的 `src/graph/`

---

## 验收清单

### Round 1（本 PR，必须过）

- [x] `pytest tests/ --ignore=tests/test_negative_sampling.py` 全绿（107 passed）
- [x] `src/api/llm.py` 存在，`src/api/__init__.py` 是 re-export shim
- [x] `src/retrieval/bm25.py` + `src/retrieval/metrics.py` 存在，`__init__.py` 是 re-export shim
- [x] `src/io/jsonl.py` + `src/io/paths.py` 存在
- [x] `src/cli/common_args.py` 存在
- [x] `src/pairing/endpoint_anchor.py` 已删
- [x] `experiments/__init__.py` 存在
- [x] `grep -rn "yunwu.ai" src/` 的结果里，所有命中都在 `local_api_logger.wrap_requests_call()` 的调用路径上
- [x] 既有的 `from src.api import call_llm` / `from src.retrieval import BM25Lite` / `from src.qc import run_llm_qc` 全部仍然可用

### Round 2

- [ ] `grep -rn "def load_jsonl" scripts/` 返回 0
- [ ] `grep -rn "Path(__file__).resolve().parent.parent" scripts/` 返回 0
- [ ] `src/utils/token_usage_logger.py` 合并或保留（取决于 Round 2 决策）
- [ ] slurm_scripts 与 experiments/ 新路径同步
- [ ] `rerun_llm_qc.py` 接入 `log_run()`
- [ ] `tests/test_negative_sampling.py` 的 `_build_adjacency` 引用修复（这是 Round 1 暴露的 pre-existing bug）

---

## 回滚方案

Round 1 全部是**纯结构调整 + re-export shim**，零行为变化。回滚方法：

```
git revert <commit-sha>
```

re-export shim 保证所有既有 `from src.api import ...` / `from src.retrieval import ...`
依然工作，没有任何脚本需要同步修改。如果某个 Round 1 动作引发问题，可以单独 revert
那一 commit 而不影响其他。
