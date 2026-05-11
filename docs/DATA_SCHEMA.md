# Data Schema — delivery_v1

逐字段说明 `data/03_queries/delivery_v1_2026-04-13.jsonl` 中的 27 个字段，以及对应的训练 Triplet 字段（`data/07_training/delivery_v1/`）。

## 1. Delivery JSONL — 顶层字段

| 字段 | 类型 | 示例 | 说明 |
|------|------|------|------|
| `query_id` | str | `l1_de_1104.3913_0073` | 全局唯一 ID。前缀 `l1_de_*` / `l3_de_*` / `l1_de_lc_*` 反映来源批次，**不**代表当前难度等级 |
| `query` | str | "The image isolates one small boxed region..." | 用户提出的问题。要求覆盖两端证据，不能只看单一元素就能答 |
| `answer` | str | "The selective boxed region is justified because..." | 标准答案。需要组合两端证据 + 桥接逻辑 |
| `doc_id` | str | `1104.3913` | arXiv ID。本批次全部 intra-doc，两端 evidence 同属一篇 |
| `hop_distance` | int | 2 / 3 / 4 / 5 | 图上两端 element 之间的最短路径长度 |
| `query_style` | str | `academic` / `real_user` | 生成时使用的风格。`real_user` 是 PhD-student-at-lab-meeting persona，禁学术腔 |
| `query_type` | str | `hypothesis_verification` / `cross_application` / ... | 模板类型 |
| `cross_modal` | bool | `true` | 两端是否来自不同模态。本批次 473/473 = true |
| `element_ids` | list[str] | `["1104.3913_formula_2", "1104.3913_figure_2"]` | 两端 element ID（顺序无意义） |
| `element_a_type` / `element_b_type` | str | `figure` / `formula` / `table` | 两端模态 |
| `pair_type` | str | `figure+table` / `figure+formula` / `formula+table` / `figure+figure` | 标准化后的模态对（按字典序拼接） |
| `text_evidence` | str | "The expression computes the average distance..." | 来自桥接段落或 enriched_content 的文本证据。**不等于** answer |
| `visual_anchors` | list[obj] | `[{element_id, anchor}, ...]` | 每个 element 上的视觉锚点。`anchor` 是具体位置 / 标记 / 符号的描述，禁 generic 词 |
| `image_paths` | list[str] | `["data/mineru_output/.../images/xxx.jpg"]` | 该 query 涉及的图像文件路径（相对项目根） |
| `reasoning_chain` | str | "The visual isolates a single small boxed region..." | 自由文本：如何从两端证据 compose 出答案 |
| `reasoning_steps` | list[obj] | 见下方 | **可空**。仅 L3 严格推理链批次有内容；本批 69/473 条非空 |
| `reasoning_depth` | int | 0 / 2 / 3 | 启发式标记的推理深度（数答案中因果连接词） |
| `reasoning_structure` | str | `serial` / `parallel` / `mixed` / `unknown` | 推理结构。`serial` = 串行链，`parallel` = 并行取证 |
| `required_evidence_spans` | list[obj] | `[{element_id, span, evidence_type}, ...]` | 必要证据 span。`evidence_type` ∈ {observation, mechanism, ...} |
| `dual_evidence` | bool | `true` | 是否双证据。本批次 100% true |
| `persona` | str | `none` / `phd_ml_fairness` / ... | PersonaHub 人设。`none` 表示不开人设 |
| `quality_tier` | str | `unknown` / `gold` / `silver` | 留作后续手工 review 标注 |
| `qc_pass` | bool | `true` | 是否通过 QC。本交付包全部 `true` |
| `qc_metrics` | obj | 见 §2 | 所有 QC 维度的量化指标 |
| `qc_issues` | list[str] | `[]` | 触发的硬失败项 |
| `_source_batch` | str | `sweep_m2_academic` / `old_l3_v3` / ... | 来源批次。用于追溯哪个生成 run 产生的此条 |

### 1.1 `reasoning_steps[]` 子字段（L3 风格才填）

| 字段 | 说明 |
|------|------|
| `step_id` | int，从 0 开始 |
| `evidence_element_id` | 该 step 使用的 element |
| `evidence_type` | observation / mechanism / definition / ... |
| `evidence_span` | 引文片段 |
| `reasoning_role` | 该 step 在链中的角色（grounding / bridging / conclusion） |
| `depends_on_steps` | list[int]，依赖的前序 step_id（构成 DAG） |
| `produces_claim` | str，该 step 推出的子结论 |

### 1.2 `required_evidence_spans[]` 子字段

| 字段 | 说明 |
|------|------|
| `element_id` | 必须命中的 element |
| `span` | 在该 element 内必须看到的文字 / 视觉特征 |
| `evidence_type` | `observation` / `mechanism` / `definition` / `comparison` / ... |

### 1.3 `visual_anchors[]` 子字段

| 字段 | 说明 |
|------|------|
| `element_id` | 锚点所属 element |
| `anchor` | 具体位置 / 标记 / 符号。如 "row 3 column 2" / "central small outlined box" / "E(x) randomized output y" |

## 2. `qc_metrics` 子字段

3 类指标，全部为数值或布尔。生成器在 rule QC + LLM QC 阶段写入。

### 2.1 Rule QC（生成器内部计算）

| 字段 | 含义 |
|------|------|
| `query_word_count` | query 词数 |
| `query_length_bucket` | `short` / `medium` / `long` / `too_long` |
| `anchor_leak_jaccard` | query token vs anchor token Jaccard。>0.20 触发硬失败 |
| `anchor_token_copy_count` | query 直接抄 anchor 的 token 数 |
| `has_cross_modal_operator` | bool，是否含 affect/differ/produce/explain 等跨模态动词 |
| `anchor_count` | 显式视觉锚点数 |
| `answer_overlap_a` / `answer_overlap_b` | answer 与两端 evidence 的 token 重合 |
| `answer_balance` | overlap_a/(overlap_a+overlap_b) 偏离 0.5 的程度。极端值触发 `single_element_answer` |
| `text_evidence_overlap` | text_evidence 与 answer 的重合度。> 0.40 触发 `text_evidence_over_reliance` |
| `formula_symbol_term_count` | 公式中识别出的符号数 |
| `formula_symbol_grounded` | answer 是否真的用到这些符号 |
| `is_architecture_case` | 是否为模型架构图场景（触发额外 QC） |
| `opening_signature` | query 开头模式。`How does X relate` 等模板硬失败 |
| `pair_has_short_query` / `pair_has_long_query` | 同 pair 是否同时产出短 / 长 query（length mix） |

### 2.2 M4 推理深度 heuristic

| 字段 | 含义 |
|------|------|
| `m4_reasoning_depth` | answer 中因果连接词计数（because/therefore/leads to ...） |
| `m4_reasoning_structure` | `serial` / `parallel` / `mixed` 启发式分类 |
| `m4_is_true_multihop` | bool，严格判定（当前只有非常少量为 true） |
| `m4_causal_link_count` | 因果连接词总数 |
| `m4_step_deletion_proxy` | bool，causal_link_count ≥ depth-1 |
| `m4_depth_advisory_issues` | list[str]，advisory 而非 hard fail |

**注意**：M4 字段是 advisory tagging，不参与 qc_pass 判定。详见 [`M4_STRATEGY_REVIEW_2026-03-18.md`](./M4_STRATEGY_REVIEW_2026-03-18.md)。

### 2.3 LLM QC

```jsonc
"llm_ablation": {
    "full_can_answer": false,           // 给完整两端 evidence 能否答（应该=true 才说明 query 有效；
                                        //   注：此字段含义被反复迭代过，目前生成器以 ablation 综合判定 fake_multihop）
    "full_confidence": 0.9,
    "single_element_can_answer": [false, false],     // 单独给 a / 单独给 b 各自能否答
    "single_element_confidence": [0.96, 0.98],
    "drop_element_can_answer": [...],                // 删一个 step 后能否答（L3 才有）
    "drop_element_confidence": [...]
},
"llm_grounding": {
    "is_grounded": true,                // answer 是否真的能从 evidence 推出
    "confidence": 0.89,
    "hallucinations": [],               // 列出未被 evidence 支撑的 answer 片段
    "reason": "The answer is supported by combining ..."
}
```

## 3. 训练 Triplet 字段（`data/07_training/delivery_v1/{train,val,test}.jsonl`）

由 `scripts/export_training_data.py` 从 normalized StandardQuery 生成。

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema_version` | str | 当前 `"1.0.0"` |
| `query_id` | str | 与 delivery 一致 |
| `query_text` | str | = delivery.query |
| `difficulty_level` | str | `L2` / `L3`，按 `reasoning_steps` 是否非空划分 |
| `positive` | list[Chunk] | 两端正例 evidence chunk |
| `hard_negatives` | list[Chunk] | 负样本（默认 3 个） |
| `negative_strategy` | str | `random` / `in_doc_swap` / `same_type_hard` / `modal_mixed` / `graph_aware` |

### 3.1 `Chunk` 子字段

| 字段 | 说明 |
|------|------|
| `element_id` | 来自 multimodal_elements.json |
| `doc_id` | arXiv ID |
| `element_type` | `figure` / `table` / `formula` |
| `span_text` | 该 chunk 的文本（来自 caption + content + context）|
| `evidence_type` | 仅正例有 |
| `image_path` | 可选 |

### 3.2 数据划分

- **Doc-level hash split**：同一篇 `doc_id` 的所有 query 只进同一个 split，避免训练 / 验证泄漏
- **默认比例**：85% train / 7.5% val / 7.5% test（`scripts/export_training_data.py --train-ratio 0.85 --val-ratio 0.075`）

### 3.3 Negative sampling

- `graph_aware`（默认推荐）：从 `hub_candidates_enriched_v3.json` 构建 element 邻接表，优先采 1-hop 邻居作为结构 hard negative，不足时 random 补齐
- 历史经验：~25% 的 negative 来自图邻居，其余 random（取决于 query 是否覆盖图中节点）
- 详见 `src/sampling/negative_sampler.py` 与 CLAUDE.md「2026-04-05 GraphAware 负样本实现」节
