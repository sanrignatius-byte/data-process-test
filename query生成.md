# Query与标注生成

> **核心代码**：`scripts/generate_multihop_l1_queries.py` · `src/qc/` · `src/prompts/`

---



## 一、反向query

### 1. 三层体系

- **L1**（单文档·双证据）：element+Hub
- ~~**L2**（跨文档）：每对产 2 条，双文档证据~~
- **L3**（多步推理链）：element+hub+element

### 2. 生成流水线

```
pair → style(academic/real_user) → template(模态+hop) → persona注入 → bridge+enriched上下文
     → MLM（产生query+answer+reasoning steps） → parse → QC → JSONL
```

- **模板**：fig+table(1hop/2hop) · fig+formula · formula+table · 3step_reasoning_chain · real_user×5
- **Persona**：76 人设，`MD5(pair_id)` 确定性分配

### 3. 标注生成

#### 3.1 Prompt约束

生成query/answer/reasoning step的过程中，通过提示词约束MLM进行如下操作：

**禁止**：Yes/No 问题 · 元语言(figure/table/equation) · 模板壳句 · 并行双问 · 数值泄漏 · 锚点拷贝

**必须** ：Observation Injection（具体观测描述） · 长度混合(短8-14词+长18-30词) · 实体豁免(F1/AUC等) · 因果连接词 · 双元素必要性

#### 3.2 额外知识

生成query/answer/reasoning step的过程中，把这些东西注入到提示词里：

- **Bridge 文本**：从 `latex_reference_graph.json` 提取 `\ref{}` 边 context，注入 prompt 让 query 使用论文原始术语
- **词汇表**（`src/qc/constants.py`）：`CROSS_MODAL_OPERATORS`(180+) · `ENTITY_AMNESTY_TERMS`(~50) · `RELATION_CONNECTORS`(~15) · `BAD_META_PATTERNS`

---



## 二、自然简短问题

（==4.20号讨论方案，确定后开始==）



### 1. query生成



### 2. evidence确定



### 3. answer生成



## 三、QC（==需要MLM-Judge==）

**阈值**：anchor_leak Jaccard ≤ 0.20 · answer_balance ≥ 0.20 · query ≤ 30 词 · evidence_overlap ≤ 0.40

**Academic QC**（24 项）：元语言 → yes/no → 数值泄漏 → 模板捷径 → 并行双问 → 语义错配 → reasoning_chain ≥ 40 字符 → answer ≥ 20 字符 → anchor_leak → 跨模态算子 → evidence_spans ≥ 2 元素 → single_element_answer → formula_symbol_hit → 因果连接词 → architecture_intent

**Real-user QC** 差异：取消模板检查 · yes/no 降为 advisory · 新增 retrievability_score + numeric_unsupported

**L3 QC**：依赖链非空 · 无重复 evidence · type 多样性 · 前提/结论弧不断裂



==MLM-as-Judge==：替代or增强纯正则的QC



