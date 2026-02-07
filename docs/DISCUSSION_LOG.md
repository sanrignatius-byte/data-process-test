# M4 Query 生成系统讨论记录

## 日期：2026-02-07

## 一、当前完成的工作

### 1. 下载参考文献
- 使用论文 `1908.09635` 作为种子，下载了 85 篇 arXiv 论文
- 添加了 `--arxiv-only` 参数，只下载有 arXiv ID 的论文
- 文件名使用 arXiv ID 命名
- 添加了 Semantic Scholar API 重试机制（处理 429 限速）

### 2. MinerU 解析
- 成功解析了 80/85 篇 PDF
- 输出在 `data/mineru_output/`

### 3. M4 Query 生成
- 生成了 50 条 queries（保存在 `data/m4_queries/queries.jsonl`）
- 但质量存在严重问题（见下方评价）

---

## 二、Query 质量评价（两位助手的毒舌点评）

### 核心问题

1. **实体提取垃圾**
   - 把 LaTeX token（`\frac`, `\cdot`, `\mathrm`, `\begin`, `\end`）当成"实体"
   - 导致 bridge 是表面符号匹配，不是语义关联

2. **模态假多样**
   - 只有 text + formula
   - 没有真正的 figure/table/image
   - "把公式当多模态来凑指标"

3. **文档覆盖窄**
   - 50 条 query 只用了 7 个 doc
   - 1607.06520 出现在全部 50 个 query 中
   - 同一段内容被复用 17-19 次

4. **Query 是作文题，不是检索查询**
   - 平均每句 ~19 词，2-3 轮对话堆叠
   - 缺少可定位锚点（变量符号、Figure 编号等）
   - 不需要检索就能回答

5. **Multi-hop 是假的**
   - 只是"强行拼接"，不是推理依赖
   - 模式固定：公式 → fairness/bias → reconcile

---

## 三、性能瓶颈分析

### 当前算法复杂度
```
O(D² × E²) = 80² × 500² = 2,650 万次比较
```

### 原因
1. 每个文档提取 500-2000 个"实体"（大部分是 LaTeX 垃圾）
2. 跨文档链接是全量两两比较
3. 每次比较还要算 n-gram 相似度

---

## 四、改进方案（按优先级）

### 第一阶段：减少实体数量（简单，1小时）

```python
# 1. 建立 LaTeX 黑名单
LATEX_BLACKLIST = {
    'frac', 'cdot', 'mathrm', 'mathbf', 'mathcal',
    'begin', 'end', 'left', 'right', 'array',
    'leq', 'geq', 'neq', 'approx', 'times',
    'sum', 'prod', 'int', 'sqrt', 'over',
    # ... 更多
}

# 2. 过滤条件
def is_valid_entity(name):
    normalized = name.lower().strip()
    if normalized in LATEX_BLACKLIST:
        return False
    if len(normalized) < 4:
        return False
    if normalized.isdigit():
        return False
    return True

# 3. 提高 min_entity_frequency 到 5
```

预期效果：实体数从 500/doc 降到 30-50/doc

### 第二阶段：优化比较算法

#### 方案 A：倒排索引
```python
# 只比较有共同 token 的实体对
index = defaultdict(list)
for entity in all_entities:
    for token in entity.name.split():
        index[token].append(entity)
```

#### 方案 B：Embedding + FAISS
```python
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([e.name for e in entities])
index = faiss.IndexFlatIP(384)
index.add(embeddings)
```

### 第三阶段：重新设计 Pipeline

考虑跳过实体链接，直接用 LLM 基于文档生成 Query：
```python
prompt = f"""
文档1: {doc1_summary}
文档1关键图表: {doc1_figures}
文档2: {doc2_summary}

生成一个必须结合两篇文档+图表才能回答的问题...
"""
```

---

## 五、关键文件位置

| 文件 | 说明 |
|------|------|
| `src/parsers/reference_pdf_collector.py` | 下载参考文献 |
| `src/linkers/cross_document_linker.py` | 实体提取 & 跨文档链接（需要重写） |
| `src/generators/m4_query_generator.py` | Query 生成（需要改进 prompt） |
| `scripts/generate_m4_queries.py` | 主脚本 |
| `slurm_scripts/03_generate_m4_queries.sh` | SLURM 任务脚本 |
| `data/m4_queries/queries.jsonl` | 生成的 50 条 query |

---

## 六、下次继续的 TODO

- [ ] 实现 LaTeX 黑名单过滤
- [ ] 提高 min_entity_frequency 到 5
- [ ] 只保留 >4 字符的实体
- [ ] 测试实体数量是否降到可接受范围
- [ ] 考虑用 embedding + FAISS 优化比较
- [ ] 加入真正的多模态（figure/table）
- [ ] 添加 final_query 字段用于检索

---

## 七、Git 提交记录

```
commit 9567404
feat: add arxiv-only download mode and improve M4 query generation
```
