# 方案C生成范式 v1

日期：2026-04-12

## 一句话定义

方案C不是“先枚举一批固定 candidate pair 再写多跳问题”，而是：

1. 在综合图中维持足够连通密度
2. 从长路径里发现真实可达的跨模态 endpoint 对
3. 生成时只保留 1-3 个关键桥
4. QC 时再用完整链验证这些桥是否真的必要

一句话记忆：

长链用于找路，短证据链用于出题，完整链用于验题。

## 三层视图

### 1. Discovery Path

作用：发现真实多跳可达性。

- 输入：综合图中的随机 cross-modal endpoints
- 方法：BFS / DFS / shortest-path / constrained longest-path
- 输出：`endpoint A -> ... -> endpoint B` 的完整图路径
- 特点：允许很长，允许包含 paragraph / section / modal element

这一层的目标不是直接喂模型，而是证明“图里真的有桥”。

### 2. Generation Chain

作用：作为 prompt 的实际输入。

- 从 discovery path 中压缩出 1-3 个关键 bridge/intermediate
- 优先保留：
  - 有 enriched_content 的中间 element
  - 跨模态转换处的桥节点
  - 带明确 bridge context 的 paragraph / section
- 不保留：
  - 仅拓扑上存在、但语义上不提供新信息的冗余节点
  - 连续同质、可互相替代的弱桥节点

生成时只让模型看到：

- Endpoint A
- 关键 bridge 1..k
- Endpoint B

而不是整条超长链。

### 3. QC Chain

作用：验证这题是不是真多跳。

- 规则 QC：继续看 query / answer / anchor / span 本身是否合法
- LLM QC：用完整链做 ablation / grounding
- 判定重点：
  - 删掉 bridge 后是否还能回答
  - 只保留单端点是否还能回答
  - answer 是否真的需要跨过 bridge 才成立

这一层必须比 generation chain 更完整。

## 生成规则

### Endpoint 采样

- 随机从图中采样 cross-modal endpoints
- 采样优先级向 top-hub 覆盖区域倾斜
- 但不要只从少数超密文档里采，必须做 per-doc diversity 控制

### Bridge 压缩

- 默认压缩到 1-3 个 bridge
- 如果 full path 只有一个核心桥，直接保留这个桥
- 如果 full path 很长，优先选“最能改变语义状态”的桥，而不是平均抽样

### Prompt 原则

- query 不暴露路径结构、paragraph id、path id
- answer 必须显式写成 `endpoint A -> bridge -> endpoint B`
- text_evidence 不能只抄 answer，要补上 bridge context

## 为什么不用整条长链直接生成

因为整条长链直接喂模型会带来三个问题：

1. prompt 太长，噪声太多
2. 模型容易把“路径存在”误写成“路径必要”
3. pass rate 会被假多跳和模板化表达拉低

所以方案C的正确姿势不是“超长链直接生成”，而是“超长链辅助发现 + 关键桥压缩生成”。

## 当前落地

`scripts/pilot_method_c.py` 已按这个范式改成 v3：

- `build_method_c_view()`：构造 discovery / generation / qc 三层视图
- `build_prompt()`：只使用 compressed generation chain
- `build_ablation_elements()`：LLM QC 使用完整链（含 synthetic bridge）
- `log_run()`：记录 token 消耗

## Scale-up 版本建议

后续从 pilot 扩到 1000+ 文档时，建议严格按这个顺序：

1. `pruned_graph_v2.json` 上做 endpoint 采样
2. 实时找 discovery path
3. 压缩成 generation chain
4. 只对 generation chain 覆盖到的关键 element 做 enrichment
5. 用完整链跑 QC

不要再回到“先离线产一大批固定 pair list，再统一生成”的旧路线。
