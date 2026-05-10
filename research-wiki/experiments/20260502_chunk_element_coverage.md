# Chunk→Element 覆盖分析 (2026-05-02)

## 动机

Mentor 录音 60 要求：在断言 chunk 适不适合做检索单元之前，先回答一个基本问题——**一个 chunk 平均覆盖几个 element？**

## 方法

`scripts/analyze_chunk_element_coverage.py`：
1. 加载 M4query_v1 的 `multimodal_elements.json`（57 docs，1873 elements）
2. 加载 `chunk_virtual_nodes_v2.json`（1147 docs，但只取 57 gold docs 的 964 chunks）
3. 用 `chunk_contains_element` 边 + position_idx 范围 + section 隶属关系，建立 chunk→element 映射
4. 逐 query 统计两个 evidence element 是否落在同一 chunk

## 结果

### Chunk→Element 分布

| 指标 | 数值 |
|------|------|
| 总 chunk 数（57 docs） | 964 |
| 含 ≥1 element 的 chunk | 964 (100%) |
| 平均 elements / chunk | 1.94 |
| 中位 elements / chunk | 2 |

| Elements per chunk | Chunk 数 | 占比 |
|-------------------|---------|------|
| 1 | 501 | 52.0% |
| 2 | 256 | 26.6% |
| 3 | 119 | 12.3% |
| 4+ | 88 | 9.1% |

### Element 类型分布

| 类型 | 数量 | 占比 |
|------|------|------|
| formula | 966 | 51.6% |
| figure | 604 | 32.2% |
| table | 187 | 10.0% |
| section | 116 | 6.2% |

### 双证据 Co-location（关键发现）

| 场景 | Query 数 | 占比 |
|------|---------|------|
| 两个 evidence 在**同一** chunk | 10 | 2.1% |
| 两个 evidence 在**不同** chunk | 357 | 75.5% |
| <2 elements mapped | 106 | 22.4% |

## 结论

1. **Chunk 检索天然劣势**：75% 的 query 需要命中 2 个不同 chunk 才能覆盖双证据。chunk 作为检索单元让 target 更稀疏（964 chunks vs 1798 elements），R@1 只能覆盖一个 element。

2. **Chunk 更适合做消费单元而非检索单元**：chunk 粒度在"检索精度"和"下游消费便利性"之间存在 trade-off。当前更合理的定位是 element 做检索、chunk 做 QA 消费。

3. **分离式检索的必要性**：figure/table 和 text chunk 的 embedding 不在同一语义空间，强制混在一起检索对多模态 query 不公平。

## Related

- [exp:20260502_split_modality] — 分离式检索实验
- [exp:20260421_chunk_as_retrieval_unit] — chunk 作检索单元的前置实验
