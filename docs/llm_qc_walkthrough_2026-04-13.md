# LLM QC 走查（精简版）

> 配套 4.13 交付：解释 `src/qc/llm_judge.py` 在做什么、为什么这么做、跑下来效果如何。

## 一句话

规则 QC 抓表面问题（长度、模板、anchor 泄漏），LLM QC 抓两类规则抓不到的核心缺陷：
1. **伪多跳** —— query 看着跨元素，其实一个元素就能答
2. **答案幻觉** —— answer 里有 evidence 撑不住的事实

只在规则 QC pass 之后跑，dry_run 时全部跳过。

## 两个 judge

| Judge | 干什么 | System prompt 关键约束 |
|---|---|---|
| **Necessity**（必要性） | 给一组 evidence，判断 query 能不能只靠这些证据答出来 | "Do not guess or use outside knowledge" |
| **Grounding**（落地） | 给完整 evidence + answer，判断 answer 是不是有依据 | "Allow reasonable synthesis and inference" — 只有直接矛盾或凭空捏造才算幻觉 |

两个 prompt 故意松紧不同：必要性用来做 step-deletion（要严），grounding 用来抓幻觉（不严就会冤枉合理推理）。

## 调用次数

Necessity judge 通过 `run_ablation_qc()` 重复跑：

- **Full set** × 1：完整 evidence 是否能答（参考用，**不**参与 fake 判定）
- **Single-element** × N：每个元素单独是否能答
- **Drop intermediate** × (N−2)：去掉每个中间节点是否还能答（仅 N≥3 时有意义）

Grounding judge 跑 1 次。

| Element 数 | Necessity 调用 | Grounding | 总计 |
|---|---|---|---|
| 2 | 1 + 2 = 3 | 1 | **4** |
| 3 | 1 + 3 + 1 = 5 | 1 | **6** |

## fake_multihop 的判定（2026-04-08 修复）

`src/qc/llm_judge.py:271`

```python
is_fake = any(single_flags) or any(drop_flags)
```

旧版还包含 `not full_can_answer` —— 但 judge 看到的 element 片段是被截断的（caption/content/context 各 300-800 字符），**拿不到生成时的完整 bridge 上下文**，导致 full_set 经常误判 False，假阳率高得离谱。

教训：fake multi-hop 的真正特征是"单元素就够答"，不是"全集答不了"。删掉 `not full_can` 之后判定才稳定。

## Grounding 的两条松绑

```
- is_grounded = false ONLY if the answer introduces specific numbers, names, or
  conclusions that directly contradict the evidence or have no basis in it whatsoever.
- Do NOT flag claims as hallucinations if they are reasonable inferences or
  syntheses from the provided evidence, even if not word-for-word present.
```

为什么松：multi-hop 答案天然要做跨元素综合（"图里的曲线下降 + 公式里的衰减项 → 答案：因为衰减项主导"）。如果要求字面对齐，所有合成型答案都会被打成幻觉。

## 主入口 `run_llm_qc()`

签名（`src/qc/llm_judge.py:277-287`）：

```python
def run_llm_qc(obj, pair, client, model, provider,
               images=None, dry_run=False,
               skip_ablation=False, skip_grounding=False):
```

流程：
1. 从 `pair` 收集 `element_a` + `intermediate_elements` + `element_b`
2. 跑 ablation → 若 fake，issues 追加 `llm_fake_multihop`
3. 跑 grounding → 若幻觉，issues 追加 `llm_answer_hallucination`
4. 返回 `(issues, metrics, in_tokens, out_tokens)`，issues 直接拼到规则 QC 的 issues 列表

任何一项抛异常都不会炸主流程，会写入 `metrics[...]["error"]`。

## 实战验证

**10 case 走查（4.13 交付集）**：10/10 都通过 LLM QC，其中 5 条 (Cases 3/4/6/8/9) 的 `full_can=false` —— 正是 2026-04-08 修复要解决的"高置信假阴"，新规则下不再被误杀。

**Rerun2 全量（295 条 L3 重跑）**：pass rate 31.5%，约 70% 被 LLM QC 挡掉，主要失败原因是 `llm_answer_hallucination`(79) 和（规则侧的）`length_mix_missing`(51)。证明这道关卡确实在过滤生成质量，不是橡皮图章。

## 已知缺口

- **rerun_llm_qc.py 还没接入 `log_run()`** —— 铁律 1 的合规缺口，下一轮必须补
- **judge 看不到完整 bridge 上下文** —— 截断片段让 full_set 判定不可信，所以才把 `not full_can` 移出 fake 判定。后续如果想恢复 full_set 信号，需要把 bridge_paragraph 一起塞进 element block
