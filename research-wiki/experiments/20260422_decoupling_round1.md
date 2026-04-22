# exp:20260422_decoupling_round1

**Date**: 2026-04-22
**Status**: ROUND 1 COMPLETE（结构落地，零行为变化）；Round 2 / Round 3 见 `docs/DECOUPLING_PLAN_2026-04-22.md`
**Track**: 基础设施 / 工程债，非检索实验
**Motivation**: 两位同事给了两份独立的解耦方案 A / B。A 精细到具体文件拆分，B 给出分层边界和硬规则。单独取其中任何一套都有瑕疵（A 没定层边界；B 的 `load_multimodal_elements` 对 JSON schema 判断错误、没注意 slurm 路径耦合）。需要做一次**综合执行**，并把决策固化成 SOT 文档。

---

## 任务判读

problem statement 本质是一份 meta-review，从中提炼出 6 条要执行的动作：

1. 以 Plan B 骨架为主（`src/` 库 / `experiments/` 一次性 / `scripts/` 薄壳三层分离）
2. 吸收 Plan A 的两处具体拆分：`src/api/__init__.py` → `src/api/llm.py`；`src/retrieval/__init__.py` → `bm25.py + metrics.py`
3. `endpoint_anchor.py`（576 行、0 个外部 importer）提前到 Phase 0.5 纯删
4. 新增硬规则：`company` provider 不得绕过 `local_api_logger.wrap_requests_call()`
5. 修正 Plan B 在 `load_multimodal_elements` 上的 schema 误判（实际是 `documents → doc_id → elements` 三层嵌套，不是 flat dict）
6. Phase 7 `experiments/` 移动必须与 slurm_scripts 路径同步，第一轮只建 stub

---

## 硬规则（已写入 `docs/DECOUPLING_PLAN_2026-04-22.md`）

| ID | 规则 | 违反后果 |
|----|------|---------|
| R1 | company provider 的 HTTP 请求必须全部走 `local_api_logger.wrap_requests_call()`；禁止裸 `requests.post` / `openai.OpenAI(base_url="yunwu.ai")` | PR 自动不合格 |
| R2 | `token_logger.log_run()` 是唯一的 per-run 记账入口；`TokenUsageLogger` 只是 per-call 补充，不替代它 | 违反 CLAUDE.md 铁律 1 |
| R3 | 每 Phase 一 commit，必须跑 `pytest` 绿了再推进；红了 revert 不带病推进 | — |
| R4 | 重构阶段不允许顺手改业务逻辑。例外：结构性阻塞（如 phantom re-export）可并入 Phase 0 绿基线 | — |
| R5 | `src/io/paths.py` 必须用 `Path(__file__).resolve().parents[2]`，禁止硬编码绝对路径 | 项目 mv 后会坏 |
| R6 | God script 拆解严格三步走：原地 cut-paste → byte 级产物回归 → 再提模块。拆 2019 行的 `analyze_latex_graph_topology.py` 等必须冻结 golden input | — |

---

## Round 1 执行路径

### Phase 0 — SOT + 绿基线
- 写 `docs/DECOUPLING_PLAN_2026-04-22.md`，固化 R1–R6 + 阶段表 + 回滚方案
- 修 `src/qc/__init__.py` 里的 phantom re-export：`judge_evidence_necessity` 已被重命名成 `judge_single_element_batch`，但 `__init__` 没同步，导致 `pytest` 直接 collect 失败。属于 R4 的结构性阻塞例外

### Phase 0.5 — 死代码
- `git rm src/pairing/endpoint_anchor.py` 575 行。`grep -rn "endpoint_anchor"` 只命中 docstring 内自引用，纯净收益

### Phase 1 — I/O 基元层
- 新建 `src/io/jsonl.py`：`load_jsonl` / `iter_jsonl` / `write_jsonl` / `append_jsonl`，默认 `flush_every=1`（继承 Phase A.1 教训：长作业被杀 = 全部白跑）；实现上委托到现有 `src/utils/file_utils.read_jsonl` 保持 24 处调用方零改动
- 新建 `src/io/paths.py`：`PROJECT_ROOT` 用 `Path(__file__).resolve().parents[2]`；导出 `DATA_DIR / LOGS_DIR / CONFIG_DIR / resolve_data_path()`，消除 40+ 处手写 `parent.parent`

### Phase 2a — `src/api/llm.py` 拆分
- `git mv` 把 273 行主体移到 `llm.py`（保留 git history），`__init__.py` 重写成 27 行 re-export shim
- R1 验证：`grep -rn "yunwu.ai" src/` 全部命中在 docstring 或 `wrap_requests_call(...)` 调用链上，零裸 HTTP

### Phase 2b — CLI 统一
- 新建 `src/cli/common_args.py`：`add_llm_args(parser)` + `build_llm_client(args)`，统一 10+ 个脚本重复的 `--provider/--model/--api-key/--delay/--dry-run`
- **关键设计**：company 分支不构造 SDK client，改调 `src.api.set_company_credentials()` 注入全局，实际 HTTP 仍走 `src/api/llm.py` 的 `wrap_requests_call()`——再次保证 R1 不被绕过

### Phase 2c — token logger 定位（否决合并建议）
- 读完两份代码发现不是双胞胎：`token_logger.log_run()` = per-run SQLite 记账（铁律 1 强制）；`TokenUsageLogger.log()` = per-call JSONL 审计（可选补充）
- 只改 docstring 明确分工，保留两个模块；决策写入 SOT，不在 commit message 里隐式带过

### Phase 4 — `src/retrieval/` 拆分
- `git mv __init__.py → _legacy.py`，读完后抽出 `BM25Lite` → `bm25.py`；`reciprocal_rank_binary / coverage_at_k / ndcg_at_k` → `metrics.py`
- 新 `__init__.py` re-export 所有符号；`git rm _legacy.py`

### Phase 7 — experiments stub
- 只建 `experiments/__init__.py`，docstring 里列出 Round 2 要移动的 8 个候选脚本（`pilot_method_c.py` / `run_exp_a_difficulty.py` / `build_trial57_*` / `build_crossdoc_gold57` / `method_c_auto_followup` 等）及其绑定的 5 个 slurm 作业（10 / 11 / 23 / 37b / 40）
- 物理移动推迟到 Round 2，必须和 slurm 路径同步做成 atomic commit

---

## 验证

1. `pytest tests/ --ignore=tests/test_negative_sampling.py` → **107 passed**，和重构前基线一致
2. 被忽略的 `test_negative_sampling.py` 是 pre-existing broken（模块级引用 `_build_adjacency`，但它是类方法），按 R4 不修，登记到 SOT
3. 导入 smoke：新老路径并存——`from src.api import call_llm`（老）+ `from src.api.llm import collect_company_stream`（新）都能用
4. R1 grep：`yunwu` 命中全部在 docstring / comment 或 `wrap_requests_call()` 调用链

---

## 交付盘点

```
新增（net +969 行）
  docs/DECOUPLING_PLAN_2026-04-22.md        SOT
  src/io/{__init__,jsonl,paths}.py
  src/cli/{__init__,common_args}.py
  src/retrieval/{bm25,metrics}.py
  src/api/llm.py                            （git mv，保留历史）
  experiments/__init__.py                   Round 2 stub

重写为 shim（零行为变化）
  src/api/__init__.py                        27 行 re-export
  src/retrieval/__init__.py                  16 行 re-export
  src/qc/__init__.py                         修 phantom export
  src/utils/token_usage_logger.py            加 deprecation 说明

删除（net -937 行）
  src/pairing/endpoint_anchor.py             575 行死代码
```

两个 commit：
1. `f7902f6` — Phase 0/0.5/1/2/4/7 结构落地
2. `c3427ba` — docs(cli): 根据 code review 补充 `--api-key` 凭据优先级和安全提示

---

## Code Review 反馈处理

| 评论 | 处理 | 原因 |
|------|------|------|
| `--api-key` CLI 安全文档 | ✅ 补 docstring（CLI 优先级 + 推荐用 `.env`） | 纯文档、直接收益 |
| JSONL 静默跳过坏行 | ❌ 保持 | 故意与 `file_utils.read_jsonl` 行为对齐，保证 back-compat |
| JSON decode 静默 | ❌ 保持 | 原 `__init__.py` cut-paste 来的，R6 禁止边抽边改 |
| company provider `verify=False` SSL | ❌ 保持 | 继承配置，yunwu.ai 是公司代理可能本来就是自签证书 |
| `PROJECT_ROOT` runtime assert | ❌ 保持 | scope creep，docstring 已说明 `parents[2]` 约定 |
| CodeQL `py/clear-text-logging-sensitive-data`（`scripts/generate_multihop_l1_queries.py:1067`、`scripts/rerun_llm_qc.py:94`） | ❌ 本 PR 不改 | 不在本次动的文件里，R4 禁止；登记到 SOT 待 Round 2 |

---

## Round 2 / Round 3 挂账

**Round 2**（待跟进）
- 迁移 24 个 `load_jsonl` / 7 个 `write_jsonl` 调用到 `src.io.jsonl`
- 加 `src/io/element_loader.py` 统一 `load_multimodal_elements()` 返回 dict-of-dict 格式——**必须先贴实际 schema**，文件是 `documents → doc_id → elements` 三层嵌套（见 memories 的 `_build_element_index` bug 教训）
- 加 `src/io/graph_loader.py` 统一 `load_reference_graph()` / `load_citation_graph()` / `load_hub_scores()`
- `generate_*` / `enrich_*` 脚本迁到 `src.cli.add_llm_args`
- 给 `rerun_llm_qc.py` 补 `log_run()`
- 物理移动 experiments + 同步 slurm 路径（atomic commit）
- 修 `tests/test_negative_sampling.py` 的 `_build_adjacency` 引用
- 处理 CodeQL clear-text-logging 两处告警

**Round 3**（深耕，排期未定）
- `src/prompts/templates.py` 899 行拆三份
- `analyze_latex_graph_topology.py` 2019 行抽 `src/graph/{topology, citation, scoring}.py`（冻结 golden input byte 级回归）
- `generate_long_chain_iterative_queries.py` 1719 + `generate_multihop_l1_queries.py` 1388 抽 `src/generation/{iterator, checkpoint}.py`

---

## 方法论沉淀

1. **永远先建绿色基线**：phantom re-export 不修 pytest 都跑不起来，后面每一步都没法定位问题——这条归入 R3
2. **re-export shim 是零风险重构的核心武器**：让 `git mv` + 拆分 = 纯结构动作，调用方零改动，可随时 `git revert` 单个 commit
3. **R4 例外要收敛**：phantom export 这种结构性阻塞归 Phase 0 是允许的，但业务 bug（如 CodeQL 的 clear-text logging）严格挡在下一轮——否则"边抽边改"会无限扩散
4. **Round 1 不产生功能价值，只立地基**：为 Round 2/3 的深度重构（拆 899 / 2019 / 1719 行的 god script）提供层边界、硬规则、SOT 文档、pytest 基线、shim 模式——这些就绪之后大拆分才有安全网

---

## 关联

- SOT 文档：`docs/DECOUPLING_PLAN_2026-04-22.md`
- 提交：`f7902f6`（Phase 0/0.5/1/2/4/7）、`c3427ba`（review 反馈）
- 依赖的既有 memory：CLAUDE.md 铁律 1（`log_run`）、铁律 4（相对路径）、Phase A.1 flush 教训、`_build_element_index` 三层嵌套坑
