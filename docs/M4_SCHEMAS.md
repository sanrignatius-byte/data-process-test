# M4 Data Schemas — Multi-hop / Multi-document / Multi-turn

> 本文档定义 M4 数据生成与评测所需的三套 schema。
> 每套 schema 附带 QC 验证标准和示例。

---

## Implementation Status: Schema-ready vs Generator-ready

| 状态 | 含义 | 当前位置 |
|------|------|---------|
| **Schema-ready** | 数据格式已定义，字段语义明确，可用于人工标注和验证 | ✅ 三套 Schema 均已完成 |
| **Tagger-ready** | 生成脚本可在输出中透传新字段 + 启发式 auto-tagging | ✅ `reasoning_depth` / `reasoning_structure` / `reasoning_steps` 已接入 |
| **Generator-ready** | 生成脚本原生支持 3+ element 路径枚举 + Schema-native 输出 | ❌ 待 Phase 1 实现 |

**当前状态**：所有 Schema 处于 **Schema-ready + Tagger-ready** 阶段。
生成脚本 `generate_multihop_l1_queries.py` 仍以 dual-evidence pair 为容器，
新增字段（`reasoning_steps`、`reasoning_depth`、`reasoning_structure`）通过透传挂载。

**Generator-ready 升级要点**（Phase 1 工程任务）：
- `element_ids` 从固定 2 元素 → 与 `reasoning_steps[].evidence_element_id` 一致的 N 元素
- 路径枚举从 2-3 hop 拓扑路径 → 3-4 节点因果路径
- LLM prompt 要求输出 Schema 1 格式的 `reasoning_steps[]`
- QC 从启发式 proxy → 真正 step-deletion 验证

---

## Schema 1: Strict Multi-hop Reasoning Chain（严格多跳推理链）

### 核心原则

**推理深度 ≠ 拓扑距离**。一条 query 的推理深度由"需要几步独立推理才能得出答案"决定，
而非图上经过了几条边。验证标准：**step-deletion test** — 删掉任意中间步骤后答案不可得。

### 数据格式

```jsonc
{
  // === 基础字段（兼容现有 dual-evidence 格式） ===
  "query_id": "l1_mh3_1904.03310_0001",
  "query": "Why does the fairness-accuracy tradeoff worsen when the feature distribution shifts, and what mathematical constraint governs this?",
  "answer": "Figure 3 shows accuracy drops for subgroup A under distribution shift. Table 5's ablation reveals feature F's absence causes this drop. Equation (7) proves F's distribution in subgroup A violates the bounded-loss inequality, making the tradeoff structurally inevitable.",
  "pair_type": "figure+table+formula",   // 3 modalities involved
  "element_ids": ["1904.03310::el::fig:3", "1904.03310::el::tab:5", "1904.03310::el::eq:7"],

  // === 新增：推理链字段 ===
  "reasoning_steps": [
    {
      "step_id": 1,
      "evidence_element_id": "1904.03310::el::fig:3",
      "evidence_type": "observation",       // observation | attribution | explanation | verification | prediction
      "evidence_span": "Figure 3 shows model X accuracy drops from 0.85 to 0.62 for subgroup A under shifted distribution",
      "reasoning_role": "premise",          // premise | intermediate | conclusion
      "produces_claim": "Model X suffers significant accuracy degradation for subgroup A under distribution shift"
    },
    {
      "step_id": 2,
      "evidence_element_id": "1904.03310::el::tab:5",
      "evidence_type": "attribution",
      "evidence_span": "Ablation row 3: removing feature F reduces subgroup A accuracy by 0.21, matching the Figure 3 drop",
      "reasoning_role": "intermediate",
      "depends_on_steps": [1],              // 必须有 step 1 的结论才能做 step 2 的推理
      "produces_claim": "Feature F's absence is the root cause of the accuracy drop observed in Figure 3"
    },
    {
      "step_id": 3,
      "evidence_element_id": "1904.03310::el::eq:7",
      "evidence_type": "explanation",
      "evidence_span": "Inequality (7): L(f,A) ≥ ||μ_A - μ_train||² / σ² when P(F|A) < τ",
      "reasoning_role": "conclusion",
      "depends_on_steps": [1, 2],
      "produces_claim": "The bounded-loss inequality proves the tradeoff is structurally inevitable when feature F is underrepresented in subgroup A"
    }
  ],

  // === 推理链元数据 ===
  "reasoning_depth": 3,                    // len(reasoning_steps)
  "reasoning_structure": "linear",         // linear | branching | converging
  "hop_types": ["observation", "attribution", "explanation"],
  "cross_modal_transitions": [
    {"from": "figure", "to": "table", "transition_type": "attribution"},   // 从观察到归因
    {"from": "table", "to": "formula", "transition_type": "formalization"}  // 从归因到数学解释
  ],

  // === Step-deletion 验证结果（Phase 1 目标格式） ===
  // 注意：当前代码中只有 proxy heuristic（因果连接词计数），
  // 真正的 step-deletion test（删 step 重判 answer derivability）待 Phase 1 实现
  "step_deletion_qc": {
    "passed": true,
    "results": [
      {"deleted_step": 1, "answer_still_derivable": false, "reason": "Without Figure 3 observation, no evidence of accuracy drop"},
      {"deleted_step": 2, "answer_still_derivable": false, "reason": "Without Table 5 ablation, cannot attribute drop to feature F"},
      {"deleted_step": 3, "answer_still_derivable": false, "reason": "Without Equation 7, cannot explain mathematical inevitability"}
    ]
  },

  // === 兼容现有字段 ===
  "path": ["1904.03310::el::fig:3", "1904.03310::p::00045", "1904.03310::el::tab:5", "1904.03310::p::00078", "1904.03310::el::eq:7"],
  "required_evidence_spans": [
    {"element_id": "1904.03310::el::fig:3", "span": "accuracy drops from 0.85 to 0.62 for subgroup A"},
    {"element_id": "1904.03310::el::tab:5", "span": "removing feature F reduces subgroup A accuracy by 0.21"},
    {"element_id": "1904.03310::el::eq:7", "span": "L(f,A) ≥ ||μ_A - μ_train||² / σ²"}
  ],
  "qc_pass": true,
  "qc_issues": []
}
```

### 推理链约束

| 约束 | 规则 | 验证方式 |
|------|------|---------|
| **最小深度** | `reasoning_depth ≥ 3` | 计数 |
| **串行依赖** | 除 step 1 外，每个 step 必须有 `depends_on_steps` | 结构检查 |
| **不同证据** | 每个 step 的 `evidence_element_id` 必须不同 | 集合去重 |
| **跨模态** | `element_ids` 至少包含 2 种不同模态 | 模态集合 |
| **step-deletion** | 删除任意 step 后答案不可导出 | LLM 或规则判定 |
| **因果递进** | `reasoning_role` 序列必须从 premise 到 conclusion | 顺序检查 |

### Evidence Type 定义

| 类型 | 说明 | 典型来源 |
|------|------|---------|
| `observation` | 从数据/图表中直接读出的事实 | figure, table |
| `attribution` | 将观察归因于特定原因 | table (ablation), paragraph |
| `explanation` | 用理论/公式解释因果机制 | formula, paragraph |
| `verification` | 用额外证据验证中间结论 | table, figure |
| `prediction` | 基于前序推理做出预测 | formula, paragraph |

---

## Schema 2: Element-level Cross-document Bridge（元素级跨文档桥接）

### 核心原则

跨文档链接必须在**元素级**而非文档级。"Paper A 引用了 Paper B" 不够，
必须知道 "Paper A 的 Figure 3 和 Paper B 的 Table 1 在讲同一件事"。

### 数据格式

```jsonc
{
  "edge_id": "xdoc_001",
  "source": {
    "doc_id": "1904.03310",
    "element_id": "1904.03310::el::fig:3",
    "element_type": "figure",
    "semantic_summary": "Accuracy-fairness tradeoff curve under distribution shift"
  },
  "target": {
    "doc_id": "1707.09457",
    "element_id": "1707.09457::el::tab:2",
    "element_type": "table",
    "semantic_summary": "Comparative results of fairness-constrained models on Adult dataset"
  },

  // === 桥接信息 ===
  "bridge_type": "shared_methodology",     // shared_methodology | shared_dataset | shared_metric |
                                           // extends_result | contradicts_result | applies_theory
  "bridge_evidence": "Both elements evaluate fairness-accuracy tradeoffs using demographic parity on the Adult dataset. Figure 3 (1904.03310) shows the tradeoff curve, while Table 2 (1707.09457) reports specific operating points on a similar curve.",
  "shared_entities": ["demographic parity", "Adult dataset", "accuracy-fairness tradeoff"],

  // === 置信度 ===
  "confidence": 0.82,
  "confidence_sources": {
    "embedding_similarity": 0.87,          // Qwen3-Embedding-4B cosine similarity
    "entity_overlap_jaccard": 0.45,        // shared entity Jaccard
    "citation_link_exists": true,          // doc-level citation edge exists
    "bridge_evidence_quality": "high"      // LLM-assessed or rule-based
  },

  // === 来源追踪 ===
  "discovery_method": "embedding+citation", // embedding | citation | entity_overlap | manual
  "embedding_match_rank": 3                // rank in cross-doc embedding results (if applicable)
}
```

### 桥接类型定义

| Bridge Type | 说明 | 跨文档推理价值 |
|-------------|------|--------------|
| `shared_methodology` | 两篇论文用相同方法，可对比 | 高 — "方法 M 在数据集 A vs B 上表现如何？" |
| `shared_dataset` | 两篇论文用相同数据集 | 中高 — "不同方法在同一数据上的效果差异" |
| `shared_metric` | 两篇论文度量相同指标 | 中 — "指标 X 在不同设置下的变化" |
| `extends_result` | B 的结论建立在 A 的发现之上 | 高 — "B 如何扩展/改进了 A 的发现？" |
| `contradicts_result` | B 的结果与 A 矛盾 | 很高 — "为什么 A 和 B 的结论不一致？" |
| `applies_theory` | B 应用了 A 提出的理论/公式 | 高 — "A 的理论在 B 的场景下是否成立？" |

### 构建来源优先级

1. **Embedding 相似度**（已有 Qwen3-4B 匹配）：cosine > 0.75 的 element 对
2. **共享实体**：两个元素上下文中出现相同的方法名/数据集名/指标名
3. **Citation + Element 共现**：doc-level 引用边存在 + 两端元素语义相关
4. **人工验证**：抽样验证 bridge 质量

---

## Schema 3: Multi-turn Session（多轮对话 Session）

### 核心原则

Multi-turn 是 multi-hop 的 **session 化外壳**。每个 turn 对应推理链中的一个 hop，
后续 turn 通过指代和省略依赖前序 turn 的上下文。

### 数据格式

```jsonc
{
  "session_id": "sess_1904.03310_001",
  "source_reasoning_chain_id": "l1_mh3_1904.03310_0001",  // 指向 Schema 1 的推理链

  // === 对话轮次 ===
  "turns": [
    {
      "turn_id": 1,
      "corresponds_to_step": 1,            // 对应 reasoning_steps[0]
      "query": "What trend does Figure 3 show for subgroup A's accuracy under distribution shift?",
      "answer": "Figure 3 shows that model X's accuracy drops from 0.85 to 0.62 for subgroup A when the feature distribution shifts.",
      "evidence_element_id": "1904.03310::el::fig:3",
      "depends_on_turns": [],              // 第一轮无依赖
      "coreference_type": null             // 无指代
    },
    {
      "turn_id": 2,
      "corresponds_to_step": 2,
      "query": "Which specific factor in Table 5's ablation study accounts for this drop?",
      "answer": "Removing feature F reduces subgroup A accuracy by 0.21, closely matching the 0.23 drop seen in the previous figure.",
      "evidence_element_id": "1904.03310::el::tab:5",
      "depends_on_turns": [1],
      "coreference_type": "explicit",      // "this drop" 指代 turn 1 的结论
      "coreference_tokens": ["this drop"]
    },
    {
      "turn_id": 3,
      "corresponds_to_step": 3,
      "query": "Does equation (7) explain why that relationship is mathematically inevitable?",
      "answer": "Yes — inequality (7) proves L(f,A) ≥ ||μ_A - μ_train||²/σ² when P(F|A) < τ. Since feature F is underrepresented in subgroup A (P(F|A) = 0.12 < τ = 0.3), the accuracy loss is bounded below, making the tradeoff structurally inevitable.",
      "evidence_element_id": "1904.03310::el::eq:7",
      "depends_on_turns": [1, 2],
      "coreference_type": "implicit",      // "that relationship" 需要理解 turn 1+2 的完整上下文
      "coreference_tokens": ["that relationship"]
    }
  ],

  // === Session 元数据 ===
  "num_turns": 3,
  "modalities_covered": ["figure", "table", "formula"],
  "is_cross_document": false,
  "coreference_density": 0.67,             // 有指代的 turn 比例 (2/3)

  // === Turn Dependency QC ===
  "turn_dependency_qc": {
    "passed": true,
    "results": [
      {
        "tested_turn": 2,
        "removed_context_from_turns": [1],
        "query_still_answerable": false,
        "reason": "'this drop' has no referent without Turn 1"
      },
      {
        "tested_turn": 3,
        "removed_context_from_turns": [1, 2],
        "query_still_answerable": false,
        "reason": "'that relationship' requires understanding both the observation and the attribution"
      }
    ]
  }
}
```

### Turn 生成策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| **hop-to-turn** | 推理链每步 → 一个 turn，加入指代词 | 3-step 推理链 → 3-turn 对话 |
| **zoom-in** | Turn 1 问全局趋势，Turn 2 追问具体数值，Turn 3 追问原因 | 单元素多层次探索 |
| **cross-doc pivot** | Turn 1 问 Doc A，Turn 2 跨文档问 Doc B，Turn 3 对比两者 | 跨文档推理 |

### 指代类型

| 类型 | 说明 | 示例 |
|------|------|------|
| `explicit` | 代词或明确指示词 | "this drop", "that model", "these results" |
| `implicit` | 需要推理才能理解的省略指代 | "the relationship"（未明说哪个关系） |
| `ellipsis` | 省略了前轮已提到的主语/宾语 | "And for subgroup B?"（省略了"what is the accuracy"） |

### QC 验证

| 检查项 | 规则 | 失败意味着 |
|--------|------|----------|
| **turn_dependency** | 删掉前轮上下文后当前轮不可回答 | 轮次间无真实依赖（伪多轮） |
| **coreference_resolution** | 指代词可以唯一消解到前轮内容 | 指代模糊 |
| **context_accumulation** | 最后一轮的答案依赖所有前轮的累积信息 | 对话深度不够 |
| **natural_flow** | 对话像真实交流，不像分割的独立问题 | 人工拼接感 |

---

## Schema 之间的关系

```
Schema 1: Multi-hop Reasoning Chain
    ↓ 推理链作为骨架
Schema 3: Multi-turn Session
    ↑ 指代 + 省略 = session 化

Schema 2: Cross-doc Bridge
    → 当推理链跨越文档边界时，
      path 中的 cross-doc 边来自 Schema 2
```

**生成顺序**：
1. 先用 Schema 2 构建 element-level 跨文档边，加入图中
2. 在图上枚举 3+ hop 因果路径，用 Schema 1 生成推理链 query
3. 将推理链转写为 Schema 3 的多轮对话

---

## 与现有数据的兼容性

| 现有字段 | Schema 1 | Schema 2 | Schema 3 |
|---------|----------|----------|----------|
| `query` / `answer` | ✅ 保留 | — | 拆分到 `turns[]` |
| `element_ids` | ✅ 扩展到 3+ | ✅ `source` / `target` | 每 turn 一个 |
| `path` | ✅ 保留 | — | 对应 turn 序列 |
| `required_evidence_spans` | ✅ 细化到每步 | — | 每 turn 一个 |
| `qc_pass` / `qc_issues` | ✅ + step-deletion | — | + turn-dependency |
| `reasoning_chain` (旧) | 升级为 `reasoning_steps[]` | — | — |
| `hop_distance` (旧) | 升级为 `reasoning_depth` | — | = `num_turns` |

**向后兼容**：所有现有 dual-evidence 数据可标记为 `reasoning_depth: 2, reasoning_structure: "parallel"`，
表示"并行取证"而非"串行推理"。新数据从 `reasoning_depth ≥ 3` 开始。
