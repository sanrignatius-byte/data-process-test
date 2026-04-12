# Long-Chain Query Generation v2: Short-Chain Strategy

## 问题分析 (Job 57369)

### 失败率分析
- **总计**: 211 pairs → 12 pass (4.7%)
- **主要失败原因**:
  1. `query_too_long` (53.5%): MAX_QUERY_WORDS=30 过于严格
  2. `fake_long_chain` (38.3%): Ablation 检测到端点足够回答
  3. `text_evidence_over_reliance` (32.4%): 太依赖文本而非视觉
  4. `single_element_answer` (32.8%): 答案只需单元素
  5. `llm_answer_hallucination` (21.4%): 答案无法从证据推导

### 性能问题
- **运行时间**: 14h 30min
- **API成本**: $30.58
- **平均每pair LLM调用**: 8-10次 (step gen + final + ablation + repair)
- **原因**: 链太长 (hop_distance 3-11)，每个hop都要LLM调用

## 改进方案: 2-4 Element Short-Chain

### 核心思想
```
元素A ──bridge_1──> 元素B ──bridge_2──> 元素C
(start)            (hub)             (end)
```
- 最多 3-4 个可视元素
- 1-2 个 bridge hub
- 每个 bridge 是连接相邻元素的关键推理步骤

### 具体改动

1. **链长限制**
   - 原: hop_distance ≤ 11
   - 新: **hop_distance ≤ 4** (只接受 2-4 hops)
   - 效果: 减少 prompt 长度，减少 LLM 调用

2. **MAX_QUERY_WORDS 放宽**
   - 原: 30 words
   - 新: **40 words**
   - 效果: 减少 query_too_long 失败约 50%

3. **简化 Ablation QC**
   - 原: endpoints-only + drop-each-intermediate
   - 新: **仅 endpoints-only**
   - 效果: 减少 50% ablation 调用

4. **Bridge 选择优化**
   - 原: 全部中间节点都作为 step
   - 新: **选择 1-2 个最有信息量的 hub**
   - 选择标准:
     - 优先选有 enriched_content 的节点
     - 优先选与端点不同类型的节点 (e.g., figure→formula→table)
     - 跳过冗余同类型节点

5. **Prompt 优化**
   - 合并 step prompt + final prompt 为一次调用
   - 在 prompt 中直接提供 hub 的 evidence span
   - 减少 round-trip

### 预期效果

| 指标 | 原来 | 预期 |
|------|------|------|
| 通过率 | 4.7% | **15-25%** |
| 每 pair LLM调用 | 8-10 | **2-3** |
| 运行时间 (200 pairs) | 14h | **3-4h** |
| API成本 | $30 | **$8-12** |

## 建图后的流程

1. **MinerU 解析完成** (当前 1183/1422)
2. **提取 multimodal_elements.json**
   ```bash
   python scripts/extract_multimodal_elements.py \
     --input data/00_raw/mineru_output \
     --output data/01_graphs/multimodal_elements_batch2.json
   ```
3. **构建 LaTeX reference graph**
   ```bash
   python scripts/build_latex_reference_graph.py \
     --elements data/01_graphs/multimodal_elements_batch2.json \
     --latex data/00_raw/latex_sources_batch2/extracted \
     --output data/01_graphs/latex_reference_graph_batch2.json
   ```
4. **选择 short-chain pairs** (新策略)
   ```bash
   python scripts/select_intra_doc_pairs.py \
     --strategy short_chain \
     --min-hops 2 \
     --max-hops 4 \
     --max-per-doc 10 \
     --output data/03_queries/short_chain_candidates.json
   ```
5. **生成 queries** (改进版)
   ```bash
   python scripts/generate_short_chain_queries.py \
     --candidates data/03_queries/short_chain_candidates.json \
     --max-query-words 40 \
     --simplified-ablation \
     --output data/03_queries/short_chain_queries.jsonl
   ```

## 代码修改清单

1. `src/qc/constants.py`:
   - `MAX_QUERY_WORDS = 40`

2. `scripts/select_intra_doc_pairs.py`:
   - 添加 `--strategy short_chain` 选项
   - 实现 `select_short_chain_pairs()` 函数
   - 过滤 hop_distance > 4 的 pairs

3. `scripts/generate_long_chain_iterative_queries.py` 或新脚本:
   - 添加 `--max-hops` 参数 (default=4)
   - 添加 `--simplified-ablation` flag
   - 优化 bridge hub 选择逻辑
   - 合并 step + final prompts

## 时间线

1. [x] MinerU 解析 (预计 1h 内完成)
2. [ ] 提取 multimodal elements (30min)
3. [ ] 构建 reference graph (1h)
4. [ ] 选择 short-chain pairs (10min)
5. [ ] 生成 queries (4-6h)
