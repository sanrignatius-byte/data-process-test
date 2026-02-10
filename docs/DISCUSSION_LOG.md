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

---

## 日期：2026-02-10（针对 L1_v3 前50条样本质疑的定向解析）

### 一、结论先行（对外沟通版本）

对方批评里有不少“语气过猛但技术点正确”的内容。基于当前样本特征，最关键不是立刻扩展 M2/M3/M4，而是先把 L1_v3 的**视觉必要性**和**证据闭合性**补齐。

- **成立的批评**
  1. `visual_anchor` 存在“文本化锚点”问题（OCR/词汇邻近描述多，几何/颜色/形状锚点少）。
  2. `ungrounded_why` 的确会让监督信号从“证据推理”退化成“词汇共现”。
  3. 截断/碎片化 `text_evidence` 会破坏 query-evidence 语义对齐。
- **过度绝对化的批评**
  1. “L1 完全不可用”不成立：仍有一批可救/可用样本，尤其是曲线趋势、结构图节点关系、表格对比类。
  2. “必须全盘废弃升级路径”不成立：更稳妥做法是分层筛选 + 约束生成，而非全量抛弃。

### 二、对三类核心问题的技术判读

#### 1) Visual Anchor 幻觉（高优先级）
- 问题本质：字段叫 `visual_anchor`，但很多只是在复述图中文字，导致“看图非必要”。
- 对训练影响：InfoNCE 会偏向文本可分性，图像分支梯度弱化，跨模态对齐失真。
- 处理原则：把“是否描述了几何/颜色/位置/形状关系”作为硬门禁；不满足则降级为 text-only 或剔除。

#### 2) Why 不可验证（高优先级）
- 问题本质：问“为什么”，证据只给“发生了什么”，缺因果桥接。
- 对训练影响：teacher 或模型会补写世界知识，制造伪推理监督。
- 处理原则：L1 中大幅降低 Why 比例（建议 ≤5%）；无法给出因果连接词与机制句的样本改写为 What/Which/How-many。

#### 3) Text Evidence 碎片化（中优先级）
- 问题本质：句段截断或上下文不足，导致证据语义边界不清。
- 对训练影响：相关性学习依赖噪声短语匹配，鲁棒性下降。
- 处理原则：证据最小单元提升到“完整句 + 必要前后文半句”；启用截断检测与拼写异常检测。

### 三、立刻执行的修复策略（不改大架构，先救数据）

1. **三分法分拣**：A(保留)/B(清洗)/C(剔除)，先冻结全量扩展。
2. **硬规则门禁**：
   - visual necessity test（遮图盲答可解 -> 降级）
   - why causality test（无因果桥接 -> 改写/剔除）
   - evidence completeness test（截断/低信息增益 -> 修复/剔除）
3. **用途分流**：
   - A 类用于正例监督与受约束 M4 生成；
   - B 类优先作为 hard negative 候选；
   - C 类彻底移出训练闭环。

### 四、对“是否继续做 M4”的明确立场

- 可以继续做 M4，但前提是：**M4 从 A 类证据单元受约束生成**，而不是从原始 L1 直接升级。
- 评估上必须新增两项：
  1. visual necessity pass rate（文本盲答失败率）；
  2. evidence closure rate（答案每个关键断言都能回指证据）。

### 五、对团队沟通建议（避免再次跑偏）

- 对外口径从“数量增长”改为“监督纯度优先”。
- 周报固定披露：Why 占比、OCR-anchor 占比、截断证据占比、A/B/C 分布。
- 在通过门禁前，暂停“自动扩大样本规模”的任务喵

---

## 日期：2026-02-10（第二轮修正：从“分析”切到“执行闭环”）

> 结论：上一版补充了方向判断，但仍不够“明天就能开工”。本节改为**生存优先路线**：先拿可打脸数字，再谈架构优雅。

### 0) 对三类意见的最终取舍

- **采纳（战术层）**：
  - 暂停过早工程化（85 篇阶段不用 FAISS/聚类/ANN）。
  - 先跑通 L2 小样本试产 + 最小检索评估闭环。
  - 先证明“数据对训练有增益”，再扩规模。
- **保留（方法层）**：
  - 不放弃方法论；但方法论必须绑定指标和可复现实验。
  - Dirty MVP 作为 baseline，不是终点。
- **拒绝（极端结论）**：
  - “L1 全废”与“只能模板堆量”都不成立。
  - 正确做法是：L1 先分级清洗，再进入约束式 L2/L3/M4。

### 1) 本周交付物（缺一不可）

1. **L1 健康化报告（v3）**
   - 输出 A/B/C 三分统计（保留/清洗/剔除）。
   - 关键率：OCR-anchor 占比、ungrounded-why 占比、截断 evidence 占比。
2. **L2 试产 50 条**
   - 基于 `entity -> doc_ids` 倒排（正则 + 规则即可）。
   - 每条答案必须显式引用两篇文档证据位（图锚点 + 文本锚点）。
3. **最小评估闭环**
   - 30 条人工测试集（10 单文档跨模态 + 10 跨文档 + 10 多跳倾向）。
   - 对比 BM25 与 embedding baseline 的 Recall@10 / MRR。

### 2) Day-1 命令清单（可直接执行）

```bash
# 1) 先跑现有 QC 基线
python scripts/validate_queries.py data/l1_cross_modal_queries_v3.jsonl \
  --output data/validation_report_v3_rerun.json

# 2) 生成 L1 三分法分拣（新增脚本）
python scripts/triage_l1_v3.py \
  --input data/l1_cross_modal_queries_v3.jsonl \
  --output data/l1_triage_v3.jsonl \
  --report data/l1_triage_report_v3.json

# 3) 从 triage=A 样本里构建跨文档候选对
python scripts/build_l2_candidates.py \
  --input data/l1_triage_v3.jsonl \
  --output data/l2_candidate_pairs_v1.json \
  --topk 50

# 4) 受约束生成 L2（小批）
python scripts/generate_l2_queries.py \
  --pairs data/l2_candidate_pairs_v1.json \
  --output data/l2_queries_v1.jsonl \
  --limit 50
```

### 3) 硬门禁（先于“多样性”）

- **Visual necessity gate**：遮蔽图片后仍可稳定回答 -> 降级 text-only 或剔除。
- **Why closure gate**：`why` 若无因果桥接证据（机制句/连接词）-> 改写为 `what/which/how` 或剔除。
- **Evidence completeness gate**：片段截断、明显拼写截断、信息增益不足 -> 修复或剔除。
- **Leakage gate**：query 出现直接答案型小数/具体指标值 -> 降级或重写。

### 4) 里程碑判断（两周后）

- 若 `L1+L2` 微调后 Recall@10 / MRR **不优于 BM25**：
  - 立即停止扩规模，回到 query 定义和门禁。
- 若有稳定增益：
  - 再引入自动化扩展（聚类/ANN/更复杂负采样）。

### 5) 复盘原则

- 不再以“生成了多少条”作为主 KPI。
- 只认三类数字：
  1. 可训练监督纯度（A 类占比与门禁通过率）
  2. 检索增益（vs BM25）
  3. 失败模式收敛速度（每周错误类型占比下降）

> 一句话：先用最小代价把“可证伪闭环”跑出来，再谈百万级优雅架构喵

---

## 日期：2026-02-10（第三轮定案：下一步先做什么）

### 结论（先后顺序）

基于当前进度，优先级应当是：

1. **先做 L2（跨文档链接）最小可用版**，不是继续空谈 L1→M4。
2. **并行做 MinerU 的 table 定向修补（轻量）**，但不应成为主线阻塞项。
3. **M4 只做小样验证，不做主战场扩产**，等 L2 + 评估闭环跑通后再扩。

> 简单说：**先把“跨文档可检索”做出来并测到指标，再决定是否重投解析链路**。

### 为什么不是先重做 MinerU？

- 现在最大不确定性是“现有监督能否带来检索增益”，不是“表格解析是否完美”。
- 如果先花 1-2 周重做解析，但最终 Recall@10 / MRR 仍不涨，会造成错误归因与时间浪费。
- table 解析应采用“边际收益驱动”：只修会直接影响候选证据构建与答案闭合的关键字段。

### 为什么不是继续深挖 L1→M4 设计？

- 现阶段继续设计会重复“方案正确、数字缺失”的问题。
- 你现在需要的是可证伪闭环：
  - L2 能否稳定产出（例如 50 条）
  - 质量门禁通过率如何
  - 相对 BM25 是否有可见增益

### 本周执行计划（建议）

- **70% 时间：L2 跨文档链接与 50 条试产**
  - `entity -> doc_ids` 倒排 + top-k 候选对
  - 约束生成 + 双文档证据位校验
- **20% 时间：最小评估闭环**
  - 30 条人工测试集
  - BM25 vs embedding（Recall@10, MRR）
- **10% 时间：MinerU table 热修**
  - 仅修“影响证据定位”的字段（caption/header/row-label 对齐）

### 决策闸门（下周复盘）

- 若 L2 试产质量高且指标优于 BM25：
  - 继续扩 L2/L3，并开始结构化引入 table。
- 若 L2 质量可控但指标无提升：
  - 优先回查监督定义与负例策略，再考虑解析升级。
- 若 L2 生成本身不稳定：
  - 先收缩任务到“高置信文档对 + 模板化约束”，暂停 M4 扩展。

> 最终答案：**先升级到 L2 并做评估闭环；MinerU 只做轻量并行优化；L1→M4 设计暂不继续扩讨论，先用数据定方向**喵

---

## 日期：2026-02-10（多轮深度讨论 + L2 pipeline 落地）

### 一、讨论背景

本次讨论涉及多方观点碰撞（用户、本 Claude 实例、另一位 Claude 助手、以及一位第三方"毒舌评审"）。核心议题：**L1 是否达标？下一步先做什么？如何应对百万级规模？**

---

### 二、L1 v3 进度判断（共识）

**结论：L1 v3 质量已达可用基线，不应继续在 L1 上无限打磨。**

关键指标回顾：
- 974 条 queries，覆盖 334 张图 / 73 篇论文
- Visual anchor 74.8%（v1 仅 36.6%）
- Comparison 类 41.9%（v1 仅 12%）
- Meta-language 0（QC 完全清除）
- Clean rate 84.3%

仍存在的问题（通过 triage 量化）：
- value_leakage 12.9%（query 含答案小数）
- ocr_only_anchor 10.4%（视觉锚点仅含 OCR 文字）
- ungrounded_why 2.7%
- evidence_truncated 0.2%

### 三、关于百万级规模的架构讨论

**用户核心担忧**：最终要处理百万级文档（~百 GB），当前设计的相似性计算会爆炸，且 L1→L2 逐层过滤会导致 yield 崩塌。

**讨论的两种路线**：

#### 路线 A：自底向上（Bottom-Up）
L1 → 找跨文档对 → L2 → 找多跳链 → L3 → 加 multi-turn → M4
- 优点：可审计、可控、复用 L1 资产
- 缺点：逐层 yield 衰减，O(D²) 复杂度

#### 路线 B：自顶向下（Top-Down）
文档聚类 → 选组合 → 一次性生成 M4
- 优点：零浪费，规模友好
- 缺点：一次性生成 M4 容易产生伪多跳/证据不闭合的合成噪声

**最终共识**：融合两种路线
- 底层用 A 的可审计性（L1 作为证据单元）
- 上层用 B 的选组思想（检索式候选生成，非枚举）
- 具体做法：L1 entities → 倒排索引 → 跨文档 pair → Claude API 生成 L2
- 85 篇规模用 dict 就够，不需要 FAISS/聚类（过早工程化被批评）

### 四、聚类对对比学习的影响（重要讨论）

**用户追问**：预聚类是否对对比学习不利？

**结论**：生成时聚类和训练时负采样是**解耦的**。
- 生成时在簇内选正例对（工程需要）
- 训练时负采样覆盖全语料（打破簇边界）
- 簇内非正例文档天然是 hard negative 来源
- 需要少量跨簇 bridge query（5-10%）防止 embedding 空间碎片化

### 五、第三方"毒舌评审"的批评要点

一位外部评审对所有助手进行了尖锐批评：

**被采纳的批评**：
1. "用讨论的激情掩盖执行的懒惰"——说得对，确实讨论太多、执行太少
2. "缺乏评估闭环"——没人提过 BM25 baseline / Recall@10 / MRR
3. "974 条连热身都不够"——对比学习训练需要几千到几万条
4. "在 85 篇上搞 FAISS 是杀鸡用牛刀"——正确

**被拒绝的批评**：
1. "用模板堆 5000 条发 workshop"——低估了项目学术目标
2. "L1 全废"——三分法分拣证明 77.1% 是 A 级
3. "聚类偏见讨论是纸上谈兵"——训练数据分布设计必须在生成前想清楚

### 六、另一位助手的补充分析（精华部分）

**被采纳的建议**：
1. **A/B/C 三分法分拣**——比"L1 够用"更严谨（已实现为 `triage_l1_v3.py`）
2. **B 类作为 hard negative 候选**——脏数据不是废物
3. **评估闭环方案**：30 条人工测试集 + BM25 baseline + Recall@10/MRR
4. **决策闸门**：预设退出条件（L2 不优于 BM25 → 止损回查）
5. **"监督纯度优先于数量"**——KPI 改为 A 类占比 + 检索增益 + 错误收敛

**被修正的建议**：
1. Why 占比 ≤5% 太激进——实际只需砍 ungrounded_why（2.7%），grounded why 保留
2. BM25 评估标准——不能以"绝对打过 BM25"为判断（数据量太少时 dense 打不过 BM25 是正常的），应看 scaling curve
3. MinerU table 热修 10% 时间——实际不需要，74 个 HTML table 已存在于 text context 中

### 七、本次实际交付（代码 + 数据）

#### 新增脚本

| 脚本 | 说明 |
|------|------|
| `scripts/triage_l1_v3.py` | L1 三分法分拣，4 个自动化门禁 |
| `scripts/build_l2_candidates.py` | 从 A-class L1 提取实体 → 倒排索引 → 跨文档 pair 排序 |
| `scripts/generate_l2_queries.py` | L2 query 生成（Claude API + QC + dry-run） |

#### Triage 结果

```
Grade A (keep):   751  (77.1%)
Grade B (clean):  223  (22.9%)
Grade C (drop):     0  (0.0%)

Reason breakdown:
  value_leakage         126  (12.9%)
  ocr_only_anchor       101  (10.4%)
  ungrounded_why         26  (2.7%)
  evidence_truncated      2  (0.2%)
```

#### L2 候选构建结果

```
Unique entities:       436
Cross-doc entities:     55 (出现在 2+ 篇文档)
Candidate doc pairs:   711
Top-100 已输出

Top bridge entities:
  fairness              33 docs
  accuracy              22 docs
  parity                10 docs
  logistic regression    7 docs
  COMPAS                 6 docs
  disparate impact       5 docs
  equalized odds         4 docs
  t-SNE                  3 docs
  PCA                    3 docs
  German Credit          3 docs
```

Top-1 pair: `1412.3756 × 1810.01943` (score=38.5)，共享 DI / German Credit / LR / fairness 等 10 个实体。

### 八、关键技术发现

1. **MinerU 的 table 不需要重新解析**：74 个 figure-text pair 的 text context 已含 HTML `<table>` 标签（21%），只是 L1 生成时 prompt 没有让模型关注它。L2 可以直接利用。

2. **旧的 `cross_document_linker.py` 应废弃**：它用正则提取实体，96.7% 都是 method 类型（diversity 极差），4785 个实体大部分是 LaTeX 垃圾。新方案从 L1 的 clean 字段提取实体，436 个实体质量远超旧方案。

3. **L1 的角色重新定义**：不是"必须升级成 M4 的半成品"，而是：
   - 训练数据的一部分（单文档跨模态检索信号）
   - L2/L3/M4 的证据缓存层（visual_anchor + text_evidence 不用重新生成）
   - 跨文档桥接的种子实体池（query/answer 中的方法名/数据集/指标）

### 九、下一步执行计划（给本地分身）

**优先级排序：先 L2 落地 → 评估闭环 → 再定扩展方向**

#### 立即执行

1. **跑 L2 生成**（~$2-5，一个下午）：
```bash
export $(grep -v '^#' .env | xargs)
python scripts/generate_l2_queries.py --limit 50 --delay 0.5
```

2. **人工写 30 条测试 query**：
   - 10 条单文档跨模态（L1 类型）
   - 10 条跨文档比较（L2 类型）
   - 10 条多跳推理倾向（L3 类型）

3. **最小评估闭环**：
   - BM25 baseline
   - 用 L1+L2 数据训练小 embedding
   - 指标：Recall@10, MRR
   - 评估标准：看 scaling curve（不期望绝对打过 BM25）

#### 决策闸门（一周后）

- L2 质量好 + 指标有上升趋势 → 扩产到全部 711 对
- L2 质量好 + 指标平 → 先扩量到 500 条再判
- L2 本身不稳定 → 收缩到高置信文档对 + 模板化约束

#### 后续路线图

- L3: 基于 L2 的 bridge entity graph 找 2-hop 路径
- Multi-turn: 把 L1+L2 query 拆解为 2-3 轮对话 + coreference
- Table 模态: 利用 74 个含 HTML table 的 pair，不需要重跑 MinerU
- 百万级扩展: 确认方法有效后，再引入 FAISS/聚类/ANN

### 十、核心原则（贯穿后续迭代）

1. **监督纯度优先于数量**——宁可少一点，也要是可训练的
2. **可证伪闭环**——每一步都有指标可查，不是"感觉质量好"
3. **不过早工程化**——85 篇用 dict，1 万篇用 FAISS，百万篇再上聚类
4. **L1 是资产不是废品**——它既是训练信号，也是 L2/L3/M4 的输入上下文

### 十一、Git 记录

```
commit 67b03d5
feat: L1 triage + L2 cross-document pipeline
- triage_l1_v3.py (A=751, B=223, C=0)
- build_l2_candidates.py (55 cross-doc entities, 711 pairs, top-100)
- generate_l2_queries.py (Claude API + QC, dry-run validated)
```
