---
type: experiment
node_id: exp:20260513_qc_prompt_v2_tightening
title: "Audit-driven QC + prompt v2: scrub Bridge vocabulary, add 7 hard-fail gates, merge copilot rewrite"
date: 2026-05-13
status: completed
verdict: 9_hard_fail_gates_wired_prompt_v2_pushed_expected_pass_rate_lift_53_to_80_85
related_experiments: []
related_claims: [C12]
---

# 目的

四 cell × ~1500 = **6111** 条新 query (`graph_max20k_four_cells_snapshot_20260513_utc`) 落地后，独立审计 query 质量并区分哪些问题靠 **prompt 改**能避免、哪些必须 **code QC 兜底**。在 `claude/review-triplet-learning-design-4dEas` 分支实施 + 合并 `copilot/design-triplet-construction` 的 prompt 重写。

# 数据来源

`data/03_queries/graph_max20k_four_cells_snapshot_20260513_utc/`
- 4 cell × 全量 jsonl (1500–1545 行/cell) + 4 个 `*_pass.jsonl`
- 总: **6111 generated, 3251 pass (53.2%)**, 覆盖 ~440 doc 每 cell

# 方法

1. **18 个 regex pattern + 大规模扫描**整 6111 行，归纳 prevalence
2. 抽 20 条 pass + 20 条 fail 人工读，找 regex 未覆盖的子模式
3. 按"治法可行性"分桶：prompt-fixable / code-fixable / data-bottleneck
4. 实施 9 个 hard-fail code QC + 19 条 prompt 规则（Rules 1-19）
5. 在 3251 pass 集上回溯审计新规则的覆盖率

# 结果

## 8 个高频污染模式（实测频率）

| # | 问题 | Pass 集出现率 | 治法分桶 |
|---|---|---|---|
| 1 | 答案出现 `the bridge X` 结构化引用 | **83.3% (2707)** | **PROMPT + CODE** |
| 2 | bridge 端点只连一边 (lexical) | **67.7% 上界** | **DATA + CODE** |
| 3 | bridge 是 metadata pointer (`we report results in...`) | 23.9% (776) | **PROMPT + CODE** |
| 4 | query 含 superlative + 答案直接 resolve | 17.8% (580) | **PROMPT + CODE** |
| 5 | 答案出现 `the premise/the conclusion` | 5.7% (186) | **PROMPT + CODE** |
| 6 | `bridge_quality < 0.20` 仍 pass | 3.4% (110) | **CODE 硬地板** |
| 7 | query 直接含 `the bridge` 引用 | 2.9% (95) | **CODE 硬 fail** |
| 8 | premise ≈ conclusion span | 0.2% (7) | **CODE Jaccard** |

## "the bridge X" 分布（Pass 集 3251 条）

| verb 形态 | 次数 |
|---|---|
| the bridge explains | 1726 |
| the bridge says | 318 |
| the bridge states | 199 |
| the bridge links | 154 |
| the bridge then | 126 |
| 其余 (`ties`, `notes`, `frames`, `connects`, ...) | 183 |
| **合计** | **2706** |

## Prompt 字面词频治理（最大杠杆）

| 词频 | 原始 snapshot | copilot v1 | 合并 v2 (本实验) |
|---|---|---|---|
| user-visible `bridge*` | 22 | 7 | **9** (全部 intentional) |
| `<placeholder>` 占位符 | 0 | 5 | **0** |
| canonical container 名 | "Bridge" | "Middle paragraph" 32 次 | **"Connecting paragraph" 30 次** |
| GOOD 示例含禁词 | n/a | 1 (`leading`) | **0** |

剩余 9 个 `bridge*` 分类：
- 2 个 format placeholder (`{bridge_text}`, `{bridge_quality_label}`) — 不可去
- 2 个 JSON schema 标识符 (`"bridge_paragraph"`) — 下游 QC key
- 5 个 Rule 17 / answer 字段说明里的禁词原文引用 — 教模型识别 alias

## 9 个 hard-fail code QC gate（pipeline 注册顺序）

| Gate ID | 检测内容 | 实施位置 |
|---|---|---|
| `bridge_meta_leak_in_query` | query 含 `\bthe bridge('s|s')?\b` | `has_bridge_meta_leak_in_query` |
| `bridge_narration_in_answer` | answer 含 `the bridge X` (20+ narration 动词) | `has_bridge_narration_in_answer` |
| `premise_conclusion_meta_in_answer` | answer 含 `the premise/conclusion` | `has_premise_conclusion_meta_in_answer` |
| `superlative_answer_spoiler` | query 超级 + answer resolver 动词 (含撇号 `method's stronger`) | `has_superlative_answer_spoiler` |
| `bridge_quality_too_low` | `bridge_quality < 0.20` 硬地板 | inline in `qc_multihop_query` |
| `bridge_one_sided` | bridge token 必须连 premise 和 conclusion 两端 | `check_bridge_one_sided` (port from copilot) |
| `premise_contains_answer` | step1 ∩ answer > step3 ∩ answer (≥4 tokens) | `check_premise_contains_answer` (port from copilot) |
| `premise_conclusion_paraphrase` | Jaccard(premise span, conclusion span) ≥ 0.55 | `premise_conclusion_paraphrase_score` |
| `bridge_meta_pointer` | bridge span 含 `we report|conduct|evaluate|...` | `has_bridge_meta_pointer` |

## 19 条 prompt 规则（合并版）

新增 Rules 13-19，结构性修改：
- Rule 13 禁词列表与 QC regex 完全对齐（含撇号变体 + `maximum/minimum` + 9 个 `most X`）
- Rule 17 禁词 alias 完整列出 `the bridge` / `the connecting paragraph` / `the middle paragraph` 三种容器引用
- 新增 ANTI-PATTERN block（5 BAD/GOOD 对照）
- 新增 OPENER VARIETY block（5 完整自然范例，无 `<placeholder>`）
- 新增 FORBIDDEN OPENERS block（5 模式黑名单）
- query_type enum 3 → 6 + 内联 `query_type_definitions` 词典

## 回溯审计（新 9 gate 在 3251 pass 上）

| 触发 issue | 次数 | 占比 |
|---|---|---|
| bridge_narration_in_answer | 2707 | 83.3% |
| bridge_one_sided | 2201 | 67.7% |
| bridge_meta_pointer | 776 | 23.9% |
| superlative_answer_spoiler | 580 | 17.8% |
| premise_conclusion_meta_in_answer | 186 | 5.7% |
| bridge_quality_too_low | 110 | 3.4% |
| bridge_meta_leak_in_query | 95 | 2.9% |
| premise_contains_answer | 36 | 1.1% |
| premise_conclusion_paraphrase | 7 | 0.2% |
| **任意 gate 触发** | **3149** | **96.9%** |
| **仍 pass** | **102** | **3.1%** |

# 治法分桶

| 问题 | Prompt 治 | Code QC 治 | 数据侧治 |
|---|---|---|---|
| bridge_narration_in_answer (83%) | ✅ 主要（22→9 词频治理 + canonical 改名）| ✅ regex 兜底 | — |
| superlative_spoiler (32% query) | ✅ Rule 13 + GOOD 对照 | ✅ regex | — |
| premise_conclusion_meta (5.7%) | ✅ Rule 17 forbidden alias | ✅ regex | — |
| bridge_meta_leak_in_query (2.9%) | ✅ Rule 13 + FORBIDDEN OPENERS | ✅ regex | — |
| bridge_meta_pointer (23.9%) | ✅ Rule 15 + BAD/GOOD 例 | ✅ regex | ⚠️ 部分需 enrich 侧过滤 |
| bridge_one_sided (67.7%) | ⚠️ Rule 16 软引导（模型常 fabricate） | ✅ check_bridge_one_sided | ✅ **主治：构造期 enrich 双向 token 闸门** |
| bridge_quality < 0.20 (3.4%) | — | ✅ 硬地板 | ✅ 构造期过滤 |
| premise ≈ conclusion (0.2%) | ✅ Rule 17 | ✅ Jaccard | — |
| Opening homogenization (56% `how does the` + `which X best`) | ✅ Rule 19 + OPENER VARIETY block + FORBIDDEN OPENERS | — | — |

# Verdict

9 个 hard-fail gate 全部 wire 通；prompt v2 把 user-visible 容器词暴露从 **22 降到 9**（全部 intentional），canonical 改名 `Connecting paragraph` (30 次)。回溯审计显示 96.9% 现 pass 集会被新 gate 砍掉——这反映老 gate 漏过的污染量级，**不是新 gate 太严**。下一代生成预期 pass 率 **53% → 80-85%**（受限于 bridge_one_sided 数据侧上限）。

# 影响范围 / Scope

- 影响所有未来 `--query-style academic` 或 `mixed` 跑 PROMPT_3STEP_REASONING_CHAIN 的 query 生成
- `qc_multihop_query` 和 `qc_real_user_query` 两 pipeline 都接入新 gate
- 现有 3251 pass 集需重跑 QC 标 issue（建议存为 `qc_v2_pass` 子集而非删除）
- 不影响 L1 / L2 / legacy dual-evidence pipeline（新 gate no-op on rows without `reasoning_steps`）

# Follow-ups / Next runs

| 优先级 | 任务 | 预估收益 |
|---|---|---|
| P0 | 跑 50-100 candidates 端到端冒烟，实测新 pass 率 | $0.5；验证 53%→80%+ |
| P0 | `enrich_hub_candidates.py` 构造期加双向 token 闸门 (修 bridge_one_sided 67.7%) | data-side 主治 |
| P1 | QC 解耦 refactor: `catalog.py` 单源 + `registry.py` 注册器 | 加新 check 改 3 文件 → 1 文件 |
| P1 | 重新跑 QC 标注 3251 pass 集为 v2 tier (warn/fail/pass) | 不丢数据 |
| P2 | 加 `step_deletion_proof` JSON 字段（R5，single_element 主治） | schema 变更需协调下游 |

# 关联文件 / Artifacts

```
src/prompts/templates.py                    PROMPT_3STEP_REASONING_CHAIN 重写
src/qc/checks.py                            +218 行（7 个新 check + 1 port）
src/qc/pipelines.py                         +134 行（接 wire 9 gate）
tests/test_qc_checks.py                     60 测试 pass（+13 新测试）
audit/qc_tightening_audit_2026-05-13.md     v1 报告
audit/qc_tightening_audit_2026-05-13_v2.md  v2 合并报告
```

Branch / commits:
```
claude/review-triplet-learning-design-4dEas
  1d65e80  qc + prompt: tighten 6 audit-driven gates
  1343f16  prompt v2: scrub Bridge vocabulary + merge copilot + 4 fixes
```

# 解耦讨论（已识别但本轮未实施）

当前 9 个 gate 是「写函数 → import → 两 pipeline 各 append」三文件耦合。建议下一步引入 `src/qc/catalog.py`（单源禁词常量）+ `src/qc/rules.py`（`Rule` dataclass with `check_fn` + `prompt_fragment`）+ `src/qc/registry.py`（注册器，pipelines 自动调度）。这样：
- 禁词列表只写一次，prompt 和 QC regex 自动同步
- 加新 check 只改 1 文件（注册装饰器）
- rule_id ↔ prompt Rule 编号 ↔ audit issue 字符串 三者绑定单源

详见会话讨论；本轮未动以避免本次 PR 同时改架构和加规则两件事冲突。
