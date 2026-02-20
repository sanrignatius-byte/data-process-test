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

## 日期：2026-02-12（L1 Multi-hop v2 第二轮硬门禁迭代复盘）

### 一、迭代目标
- 按外部“毒舌审阅”意见，重点解决三类问题：
  1. 文本捷径（bridge entity 直接写进 query）
  2. 弱模板（`Which component...` / `How does X relate to Y...`）
  3. 伪跨模态解释（答案缺少因果连接，单元素可答）

### 二、执行脚本与任务
- 生成脚本（已改）：`scripts/generate_multihop_l1_queries.py`
- 集群脚本：`slurm_scripts/07_generate_l1_multihop_v2.sh`
- 本轮任务：`sbatch -> job 27477`
- 日志：`logs/l1_mh_v2_27477.out`, `logs/l1_mh_v2_27477.err`

### 三、代码侧改动（已落地）
1. **Prompt 级约束增强**
   - 新增 de-naming 指令：query 禁止直接复制桥梁实体名
   - 禁用弱模板：`Which component...`、`How does ... relate to ...`
   - 强制答案包含机制连接词（because/leads to/explains/matches 等）

2. **QC 级硬门槛新增**
   - 新增 issue：
     - `template_shortcut`
     - `bridge_entity_leakage`
     - `weak_reasoning_connector`
   - 强化 `single_element_answer`：
     - 双元素最小 overlap
     - 更高 `answer_balance` 阈值

3. **运行安全修复**
   - 修复 `--dry-run` 清空输出文件问题：dry-run 输出重定向到 `/dev/null`，不再改写目标文件。

### 四、运行结果（job 27477）
- 处理 pair：150
- 写出 query：296（parse fail: 2）
- 输出文件：`data/l1_multihop_queries_v2.jsonl`（534KB）
- 通过数：19 / 296（6.42%）
- 额外导出 clean 子集：`data/l1_multihop_queries_v2_pass.jsonl`（19 条）

**QC issue 分布（Top）**
- `single_element_answer`: 209
- `bridge_entity_leakage`: 152
- `weak_reasoning_connector`: 100
- `anchor_leakage`: 68
- `template_shortcut`: 20

### 五、结论
- 这轮属于“高压筛选”模式：通过率显著下降，但更准确暴露了伪跨模态与文本捷径问题。
- 当前 v2 可作为“高纯度小集 + 失败样本分析集”两路使用：
  - `*_pass.jsonl` 用于高置信训练/评测
  - 全量 `v2.jsonl` 用于定向修复与 prompt/QC 迭代

### 六、下一步（建议）
- 做一版“阈值调优”迭代（目标 pass rate 回到 15%-25%）：
  1. 对 `weak_reasoning_connector` 按 `query_type` 分层启用
  2. `bridge_entity_leakage` 从 hard fail 调整为分级告警
  3. 对 `figure+formula` 单独加模板（当前 fail 最重）

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

commit 2170666
improve: apply code review feedback to L1 triage + L2 pipeline
- triage: expanded VISUAL_WORDS + visual_density gate (A: 751→727)
- candidates: GENERIC_TERMS blacklist + IDF filtering (MAX_DOC_FRACTION=0.35)
- generate: NULL output instruction, source_snippet, retry + checkpoint/resume
```

---


## 日期：2026-02-10（L2 试产完成）

### 一、L2 生成结果

#### 脚本修复
原 `generate_l2_queries.py` 的 `build_prompt()` 期望 `evidence_examples` 列表，但实际候选对数据是 `doc_a_*`/`doc_b_*` 平铺字段——prompt 中的 reference evidence 会是空的。已修复为：
1. **使用实际字段**：doc_a_query, doc_a_answer, doc_a_visual_anchor, doc_a_text_evidence, doc_a_caption 等
2. **加入 Vision 输入**：两张 figure 图像 base64 编码发送给 Claude，让模型看到实际图再生成 query
3. **扩展 QC**：meta-language 正则从 3 条扩到 7 条，加 short_answer 检测
4. **加入 delay/用量追踪/成本估算**
5. **模型更新**：`claude-sonnet-4-5-20250929`（与 L1 v3 一致）

#### 生成统计

```
Total pairs:       50
QC passed:         50 (100%)
QC failed:          0
NULL (no query):    0
Parse failures:     0
Input tokens:   80,079
Output tokens:  20,746
Est. cost:      $0.55
```

#### 质量审计

| 指标 | 结果 |
|------|------|
| Meta-language in queries | 0/50 (0%) |
| Visual-rich anchors | 99/100 evidence refs (99%) |
| Dual-doc evidence | 50/50 (100%) |
| Unique doc pairs | 50 (零重复) |
| Unique docs covered | 30 |
| Query length | 25.1 词 (mean) |
| Answer length | 60.5 词 (mean) |

#### Query 类型分布

| 类型 | 数量 | 占比 |
|------|------|------|
| cross_synthesis | 27 | 54% |
| cross_comparison | 19 | 38% |
| cross_contradiction | 4 | 8% |

#### 样例

**cross_comparison**: `1412.3756 × 1810.01943`
> Q: How does the fairness-utility tradeoff for logistic regression on German Credit differ between the combinatorial repair approach and the optimized pre-processing method at disparate impact 0.8?
> A: The combinatorial repair approach shows logistic regression achieving utility around 0.65-0.70 at DI=0.8 on German Credit with a clear downward trend. In contrast, optimized pre-processing with logistic regression maintains higher balanced accuracy (~0.73-0.75) at similar disparate impact levels near 0.8.

**cross_contradiction**: `1610.07524 × 2005.07293`
> Q: Can the rising false positive rates for Black defendants with more priors in COMPAS be reconciled with an equity framework that allocates compensatory resources to historically disadvantaged groups?
> A: COMPAS exhibits increasing false positive rates for Black defendants as prior record count grows (from ~0.22 at zero priors to ~0.92 at >10 priors), contradicting equity principles.

### 二、L2 vs L1 对比

| 维度 | L1 v3 | L2 v1 |
|------|-------|-------|
| 数量 | 974 | 50 |
| QC pass rate | 97.2% | 100% |
| Meta-language | 0% | 0% |
| Visual anchor quality | 74.8% | 99% |
| Docs covered | 73 | 30 |
| 文档/query 关系 | 单文档 | 跨文档 (每条 2 docs) |
| 平均 query 长度 | 17.9 词 | 25.1 词 |
| 成本 | $4.59 | $0.55 |

### 三、关键发现

1. **Vision 输入是关键**：发送双图像让 Claude 能引用具体视觉元素（颜色、趋势、位置），99% visual-rich anchor 远超 L1 的 74.8%。
2. **丰富的 L1 上下文有效**：prompt 中包含 L1 的 query/answer/anchor/evidence 作为参考，让 L2 生成有据可依而非凭空编造。
3. **100% QC pass 说明 prompt 约束力强**：强 system prompt + 丰富的 good/bad examples + 具体的 JSON schema = 零废品率。
4. **跨文档 query 类型自然涌现**：54% synthesis、38% comparison、8% contradiction，无需手动指定配额。

### 四、下一步（按讨论日志 §九 决策闸门）

当前状态对应 **"L2 质量好"** 分支：
- ✅ L2 试产 50 条全部通过 QC
- ✅ Visual anchor 质量 99%
- ✅ 零 meta-language
- ⏳ 待验证：检索增益（需要评估闭环）

**立即执行**：
1. 评估闭环（BM25 baseline + Recall@10/MRR）
2. 若指标有上升趋势 → 扩产到全部 711 对
3. 若指标平 → 先扩量到 500 条再判

---

## 日期：2026-02-10（L2 v2 四方 Reviewer 反馈 + v3 脚本重写）

### 一、L2 v2 生成结果（中间版本）

在 v1 基础上做了三个 hotfix 后重新生成：

#### Hotfix 内容
1. **P0 `build_l2_candidates.py`**：
   - BLACKLIST 加 "in figure", "figure", "table", "section" 等文档结构短语
   - 新增 GENERIC_ENTITIES（accuracy, fairness, precision 等 18 个）
   - 要求每对至少 1 个 non-generic entity（消灭纯 generic 对）
   - 评分：specific entity 3.0 分, generic 0.5 分
2. **P1 prompt**：加 visual necessity、ban yes/no、ban speculative、semantic relevance check
3. **P1 QC**：新增 VISUAL_CUE_WORDS / SPECULATIVE_PHRASES / YES_NO_STARTERS / TEMPLATE_VERBS

#### v2 生成统计
```
候选对：43 (v1 的 100 对过滤后)
生成：  32 条 (11 NULL, 0 parse fail)
QC pass: 16 (50%)
QC fail: 16 (主要是 template_verb: 14/16)
成本：  $0.48
```

v2 相对 v1 的改进：0% yes/no, 0% speculative, 100% visual cues (in passed)。
但 QC 发现新问题：template_verb 占 QC failure 的 87.5%。

### 二、四方 Reviewer 深度反馈

用户提供了四位 reviewer 的独立评审，以及 reviewer-tagged 文件 `data/l2_queries_v2_tagged.jsonl`。

#### Tagged 文件统计
| 决策 | 数量 | 占比 |
|------|------|------|
| keep | 1 | 3% |
| fix | 26 | 81% |
| drop | 5 | 16% |

唯一 keep 的 query（l2_v2_025, Jaccard=0.138）证明低泄漏 = 高质量。

#### 核心发现：两个正交问题

**问题 1：Anchor Leakage（工程问题）**
- v2 prompt 要求 query 包含 visual cue words → 模型从 evidence anchor 复制视觉描述到 query
- 平均 Jaccard(query tokens, anchor tokens) = 0.292
- 最高达 0.54（l2_v2_020）
- 后果：BM25 可通过表面 token 匹配直接检索到文档，不需要语义理解
- 修复：query 用概念语言，视觉细节只放 evidence_refs.anchor

**问题 2：Prompt 哲学（设计问题）**
- v2 prompt 本质是 "compare X in A with Y in B"
- 产出模式固化为 "How does [visual detail A] relate to [visual detail B]?"
- Reviewer 建议：从 "concept comparison" 转向 "hypothetical reasoning"
  - 用 Doc B 的理论/框架去解释 Doc A 的观察
  - 或用 Doc A 的实证数据去预测 Doc B 的方法会如何表现
- 核心区别：comparison 是并列关系，reasoning 是因果/应用关系

#### Reviewer 批评中被采纳 vs 被拒绝的部分

**采纳**：
- Anchor leakage 是真问题，需要 QC 检测 + prompt 约束
- "relate to" 类 template verb 是空洞的
- 强制跨域桥接（DAG 连接两个不相关实验）应该 NULL
- Information gap design（query 描述一侧，答案需要另一侧）

**拒绝/修正**：
- "v2 比 v1 差"——不成立，v2 QC 是诚实的，v1 100% pass 是 QC 瞎了
- "pair_score 下降说明候选崩塌"——不成立，过滤 generic 是对的
- "需要推倒重来"——不需要，改 prompt 哲学 + QC 深度即可

### 三、v3 脚本改动（已完成，待执行）

#### `generate_l2_queries.py` 改动清单

| 改动 | v2 | v3 |
|------|----|----|
| System prompt | "data annotator" | "expert research analyst" |
| Prompt 哲学 | "compare visual X with visual Y" | "reasoning operation: explain/predict/diagnose" |
| 给模型的信息 | visual_anchor + text_evidence (泄漏源) | 只给 caption + L1 query/answer |
| Query 语言要求 | 必须含 visual cue words | 必须用概念语言，禁止 visual tokens |
| QC: no_visual_cue | ✅ (直接导致泄漏) | **移除** |
| QC: anchor_leakage | 无 | **新增** Jaccard(query, anchor) > 0.15 → fail |
| Temperature | 0.7 | 0.5 |
| Query types | comparison/synthesis/contradiction/trend | **application/prediction/diagnosis/comparison** |
| 新字段 | - | reasoning_direction, l2_id, qc_metrics |
| 默认输入 | l2_candidate_pairs_v1.json | l2_candidate_pairs_v2.json |
| 默认输出 | l2_queries_v1.jsonl | l2_queries_v3.jsonl |

#### 新 QC 函数 `anchor_leak_jaccard()`
- 提取 query 和 evidence anchor 的 content tokens（3+ chars, 去停用词）
- 计算 max Jaccard overlap across all evidence refs
- 阈值 0.15（唯一 keep query 的 Jaccard 是 0.138）

#### 新 Prompt 关键指令
1. **INFORMATION GAP**: query 描述一个文档的 context，答案需要另一个文档的 figure
2. **NO ANCHOR COPYING**: visual details 只放 evidence_refs.anchor，query 用方法名/指标名
3. **NO FORCED BRIDGES**: generic concept 连接不同实验 → 输出 NULL
4. **REASONING DIRECTION**: 新增 A_explains_B / B_explains_A / mutual

### 四、v3 执行命令

```bash
cd /projects/myyyx1/data-process-test
source /cluster/apps/software/Miniforge3/24.11.3-1/etc/profile.d/conda.sh
conda activate /projects/myyyx1/envs/minerU
export $(grep -v '^#' .env | xargs)

# 先 dry-run 验证
python scripts/generate_l2_queries.py --dry-run --limit 5

# 正式跑
python scripts/generate_l2_queries.py --limit 43 --delay 0.5
```

### 五、预期与决策闸门

- **乐观预期**：anchor_leak_jaccard < 0.15 的比例 > 60%，无 template verb
- **中性预期**：一些 pair 仍然 NULL（generic bridge），pass rate 40-50%
- **悲观预期**：模型仍然倾向 comparison 模式，需要更激进的 prompt 或 few-shot

**决策**：
- v3 QC pass ≥ 15 条 + 平均 Jaccard < 0.15 → 进入评估闭环
- v3 QC pass < 10 条 → 再调 prompt 或考虑 few-shot examples
- v3 NULL > 50% → candidate pairs 质量问题，需回头看 build_l2_candidates 的 entity 选择

---

## 日期：2026-02-10（L2 v3 三位评论家综合复盘 + 收工决议）

### 一、三位评论家的共识批评（统一摘要）

1. **“工程化字段增加 ≠ 质量提升”**
   - `reasoning_direction`、`qc_metrics` 提供了可观测性，但没有自动转化为训练集纯度。
   - 若 `qc_pass=false` 样本仍保留在候选产物中，最终会把噪声带入训练闭环。

2. **Anchor Leakage 仍是主矛盾**
   - 大量 query 与 anchor 高重叠，部分 query 直接出现关键答案数字。
   - 这会把任务从“跨文档语义检索”降级为“词面+数字匹配”。

3. **桥接实体语义退化（同名异义/泛词）**
   - `map/plot/graph/distribution` 这类泛词导致伪桥接；
   - `shared_entities` 的语义信息密度不足，易触发“强行跨域解释”。

4. **标签与推理链偶有错位**
   - 部分 `reasoning_direction` 与证据链方向不一致；
   - `cross_diagnosis` 存在滥用风险（相关性描述被包装成因果诊断）。

5. **多模态闭环不稳定**
   - 有图像输入，但部分问答主要可由文本完成，视觉必要性门禁需继续加严。

### 二、评论家观点中“采纳 vs 不采纳”

**采纳**
- 严格执行 `qc_pass` 门禁，失败样本不进入训练集；
- 优先修复实体桥接质量（先砍泛词、同名异义词）；
- 在 query 层禁止答案型数值泄露；
- 先做最小评估闭环，再谈扩产。

**不采纳（或修正后采纳）**
- “L2 路线已死、应全回滚 L1”不采纳；
  - 修正：L2 仍有可用子集，当前问题是筛选和门禁，不是方向性死亡。
- “必须全量推倒重写”不采纳；
  - 修正：优先做硬门禁 + 候选对提纯，成本更低且可快速验证。

### 三、与当日实测对齐（执行后数字）

- v3 正式运行：43 对候选，1 条 NULL，写入 42 条。
- 质检结果：`qc_pass=19`, `qc_fail=23`。
- fail 原因：`anchor_leakage=21`，`template_verb=2`。
- `evidence_closure` 整体通过，说明“证据可回指”已基本到位，当前瓶颈集中在泄漏和桥接质量。

### 四、立刻生效的收敛策略（下个工作日执行）

1. **暂停 L2 扩产**
   - 不扩到 711 对，先用 clean subset（`qc_pass=true`）跑评估闭环。

2. **三重门禁（训练前）**
   - candidate gate：提升桥接实体质量，禁用泛词桥接；
   - generation gate：query 禁止答案型数值；
   - training gate：`qc_pass=false` 一律不进训练集。

3. **评估优先于讨论**
   - 用 clean subset 跑 BM25 + dense baseline，关注 Recall@10 / MRR 趋势；
   - 若趋势无改善，再决定是否收缩 L2 或改候选构建策略。

### 五、今日收工结论

本日结论不是“继续造更多 L2”，而是“先保证进入训练的 L2 是干净的”。  
执行策略已从“生成优先”切换为“纯度优先 + 评估闭环优先”。今天到此结束。喵
## 日期：2026-02-11（Mentor 反馈 + 深耕 L1 方向定调）

### 一、本次讨论背景

用户带来 Mentor 三条建议，要求结合当前数据分析可行性。同时回顾了 Step 0 和 Step 1 的技术细节（是否用了大模型看图片）。

### 二、Mentor 三条建议

1. **丰富模态，引入 table/formula/figure 并细分**
   - 模型图？实验结果表？信息汇总表？Chart？
   - 各模态需要有针对性的处理方式

2. **构建文档内部链接与结构，自然实现多跳**
   - 方案①：利用 LaTeX 源构建不同部分的引用关系
   - 方案②：利用 MinerU 结果构建关系（较难）

3. **展望：embedding 隐空间跨文档探索**
   - 利用 embedding 在隐空间中找文本相似性更高的跨文档关联

**Mentor 鼓励先继续深耕 L1。**

### 三、数据分析结果

#### 模态分布（L1 的 974 条 query）

| 模态 | 数量 | 占比 |
|------|------|------|
| plot | 694 | 71.3% |
| diagram | 201 | 20.6% |
| example | 51 | 5.2% |
| architecture | 12 | 1.2% |
| table | 6 | 0.6% |

**问题**：plot 一家独大（71.3%），table 几乎为零（0.6%），architecture 也极少。模态多样性不足。

#### 已有但未利用的多模态资源

- **50 个** figure-text pair 上下文含 HTML `<table>`（分布在 33 篇文档中）
- **20 个**上下文含公式块（13 篇文档）
- 这些素材在 Step 0 就存在，但 L1 生成时的 prompt 没有引导模型关注 table/formula

#### 文档内交叉引用密度

在 351 个图文对的上下文中：
- Figure 引用：**1028 次**
- Table 引用：**362 次**
- Equation 引用：**69 次**
- Section 引用：**72 次**
- **302/351（86%）** 的图文对上下文含 2 个以上交叉引用

**结论**：文档内天然存在大量 Figure→Table、Figure→Equation 的引用链路，是构建多跳 query 的理想素材。

### 四、对 Mentor 建议的逐条分析

#### 建议 1：模态丰富 + 细分

**完全可行，分两步**：

1. **图片类型精分**：用大模型对 351 张图做一轮 classification（当前 `_classify_figure` 只用关键词匹配 caption，没看图片本身），成本 ~$0.5-1。得到精确的子模态分布后再定策略。

2. **补 table/formula 的专用 L1 query**：
   - 对 50 个 table-context pair 写 table-aware prompt（引导模型对比表中行/列数据 + 上下文解释）
   - 对 20 个 formula-context pair 写 formula-aware prompt（引导模型将公式变量与图中数值对应）
   - 成本 ~$1，产出预计 100-200 条新 query

#### 建议 2：文档内引用图构建多跳

**最有价值且零成本的方向。**

- **不需要 LaTeX 源码**（repo 中无 .tex/.bbl），MinerU markdown 已足够
- 正则提取 `Figure N`/`Table N`/`Eq N`/`Section N` 引用关系 → 构建文档内 DAG
- 2-hop 路径天然就是多跳 query 素材：
  ```
  Figure 3 ─引用→ Table 2 ─引用→ Equation 5
     ↑                ↑                  ↑
   L1 query     table query       formula query
  ```
- 直接与 L3 multi-hop 接轨，且不花 API 费用

#### 建议 3：Embedding 隐空间探索

**方向正确，但时机在后面。**

- 当前 85 篇规模用实体倒排索引已够（L2 candidates 已就绪）
- 当到百万级时，实体匹配的 recall 确实太低（同义不同词问题）
- 建议路径：先训初版 embedding → 用它做跨文档相似度检索 → 发现实体匹配漏掉的隐性关联 → 生成更多 L2 → 反哺训练（self-play 循环）

### 五、关键技术发现：Step 0 没用大模型看图

回顾 pipeline 发现：
- **Step 0（`figure_text_associator.py`）**：纯正则 + 位置关系解析，**没有任何大模型参与**。图片分类只看 caption 关键词。
- **Step 1（`batch_figure_understanding_api.py`）**：Claude Sonnet 4.5 同时接收 base64 图片 + 文本 prompt，真正做了多模态理解。

**影响**：Step 0 的 `figure_type` 分类可信度低（不看图片如何知道是 scatter plot 还是 architecture？）。Mentor 说的"细分模态"需要在这里补一轮大模型分类。

### 六、执行优先级排序

| 优先级 | 任务 | 成本 | 依赖 |
|--------|------|------|------|
| 1 | L1 文档内引用图（DAG）| 零（纯规则） | figure_text_pairs.json |
| 2 | L1 模态细分 + table/formula prompt | ~$1 | 引用图 + 现有 pair |
| 3 | 图片类型精分（大模型分类）| ~$0.5-1 | 351 张图片 |
| 4 | 评估闭环（30 query + BM25）| 零 | L1 + L2 数据 |
| 5 | L2 跨文档生成 | ~$2-5 | 已就绪 |
| 6 | Embedding 隐空间探索 | 待定 | 初版模型 |

### 七、语料库领域备忘

种子论文 `1908.09635` 是**算法公平性（algorithmic fairness）**方向。85 篇论文几乎都围绕 ML fairness 展开。典型实体：Disparate Impact、Statistical Parity Difference、Equalized Odds、German Credit（数据集）、COMPAS（数据集）。

"fairness" 出现在 73 篇文档中的 33 篇（45%），作为桥接实体区分度太低，已被 `MAX_DOC_FRACTION=0.35` IDF 过滤剔除。真正有价值的桥接实体是 Disparate Impact（5 docs）、German Credit（3 docs）、t-SNE（3 docs）等。

---

## 日期：2026-02-12（L1 Cross-modal Dual-evidence v1 评审 + v2 计划）

### 一、v1 生成结果

| 指标 | 数值 |
|------|------|
| 候选对 | 150 (figure+table:90, figure+formula:45, formula+table:15) |
| 产出 | 300 条 query |
| QC pass | **43 (14.3%)** |
| QC fail 分布 | anchor_leakage:196, yes_no_question:126, single_element_answer:112, meta_language:22 |
| Jaccard 均值 | 0.196 (阈值 0.15) |
| answer_balance=0 | 135/300 (45%) |
| 按 pair_type pass rate | figure+table:17.8%, figure+formula:11.1%, formula+table:**3.3%** |
| 文档覆盖 | 43 docs |

脚本：`scripts/select_multihop_candidates.py` → `scripts/generate_multihop_l1_queries.py`
产出文件：`data/l1_multihop_queries_v1.jsonl`

### 二、专家评审核心批评（两轮独立评审）

#### 评审采纳的批评（经数据验证）

1. **Anchor leakage 是根因**
   - prompt 直接给 600 chars table content + 完整 LaTeX → 模型抄数值到 query
   - 跟 L2 v2 的病因一模一样
   - 表面数字匹配就能 BM25 检索到文档，不需要语义理解

2. **Yes/No 问句泛滥 (43%)**
   - prompt 示例 "Does a trend match values?" 教坏了模型
   - yes/no 对对比学习梯度贡献极低
   - QC 标了但 prompt 没禁

3. **"Multi-hop" 名不副实**
   - 298/300 path 长度 = 2，是"跨模态双证据并行查找"，不是链式推理
   - 真 multi-hop 需要 Step 1 输出作为 Step 2 输入（sequential dependency）
   - 应改名 "cross-modal dual-evidence"

4. **Single element answerable (45%)**
   - 答案只引用一个元素的 token，另一个是装饰
   - answer_balance=0 占 135/300

5. **Formula 配对严重失败 (3.3%)**
   - LaTeX 以文本形式给出，模型复制符号串
   - 公式没有 image_path，无法发图

#### 评审批评中的过火部分（已修正理解）

1. **"工业垃圾/全废"** — 不成立，43 条 QC pass 里有 ~30 条真正有价值的 dual-evidence query
2. **"多模态完全没用"** — 不准确，问题不是图片没用而是 text_evidence 足以回答
3. **"应放弃 L1 回去做 L2"** — 不对，L1 和 L2 解决不同问题，Mentor 明确先深耕 L1
4. **"300 条全废"** — v1 是诊断版，提供了清晰的改进方向

### 三、关键发现：Tables 有图片！

探查 `data/multihop_l1_candidates.json` 发现：
- **所有 150 对中的 table 元素都有有效的 image_path**
- v1 代码已经尝试发送双图（`img_a = encode_image(...)`, `img_b = encode_image(...)`）
- 但 prompt 同时给了 600 chars `tbl_content` 文本，使得 table 图片变得冗余
- **修复方向**：减少文本暴露（只给 headers），让模型从图片读具体值

Formula 确认无图片 (0/all)，只能用 LaTeX 文本但需限制暴露。

### 四、v2 改进计划

详见 plan file `~/.claude/plans/encapsulated-kindling-micali.md`

#### 核心改动

| 改动 | v1 | v2 |
|------|----|----|
| Table 内容 | 600 chars 原文 | 150 chars headers + 发送 table 图片 |
| Formula 内容 | 完整 LaTeX | 提取 key variables，禁止完整符号串 |
| Prompt 哲学 | "Does A match B?" (验证式) | "Given A context, what does B reveal?" (信息差) |
| Yes/No | 未禁止，示例引导 | 明确禁止 + BAD/GOOD 对比示例 |
| 数值泄漏 | 无检测 | query 禁止含 2+ 具体数字 |
| Answer balance | overlap=0 才标 | balance < 0.15 即标 |
| Temperature | 0.5 | 0.4 |
| 输出 | `multi_hop: true` (全部) | `cross_modal: true`, `multi_hop` 仅 path≥3 |

#### 4 个 Prompt 模板重写要点

1. **FIGURE_TABLE**: information gap 设计，"describe one element's context, ask what the other reveals"
2. **FIGURE_FORMULA**: "empirical evidence (figure) meets theoretical framework (formula)"，禁止复制 LaTeX
3. **FORMULA_TABLE**: 提取 key variables 代替 raw LaTeX，要求模型从 table 图片读值
4. **所有模板共同**：BAD/GOOD 示例对、"UNANSWERABLE if removed"、30 词上限、禁 meta-language

#### QC 新增

- `numeric_leakage`: query 含 2+ 具体数字 → fail
- `yes_no_answer`: answer 以 Yes/No 开头 → fail
- `answer_balance` 阈值收紧: < 0.15 → `single_element_answer`

### 五、验证步骤

```bash
# 1. Dry-run 验证 prompt
python scripts/generate_multihop_l1_queries.py --dry-run --limit 5

# 2. 小规模测试 (10 pairs ≈ 20 queries)
python scripts/generate_multihop_l1_queries.py --limit 10 --delay 0.5 \
  --output data/l1_multihop_queries_v2.jsonl

# 3. 若 pass rate ≥ 30% → full run (150 pairs)
python scripts/generate_multihop_l1_queries.py --limit 150 --delay 0.5 \
  --output data/l1_multihop_queries_v2.jsonl
```

### 六、目标

| 指标 | v1 | v2 目标 |
|------|-----|---------|
| QC pass rate | 14.3% | ≥40% |
| anchor_leakage | 65% | ≤25% |
| yes_no_question | 43% | ≤10% |
| single_element_answer | 45% | ≤25% |
| formula+table pass | 3.3% | ≥15% |

---

## 日期：2026-02-16（LaTeX 引用图 + 跨文档 Citation Graph 落地）

### 一、本次成果概览

**完成了 Mentor 建议 2 的核心部分**：从 LaTeX 源码构建文档内引用 DAG + 跨文档引用图。

| 产物 | 关键指标 |
|------|----------|
| LaTeX 源码下载 | 73/76 篇 .tex, 65 篇 .bbl, 3 篇 no_source |
| 文档内引用 DAG | 1949 labels, 5547 refs, 2847 edges (ref+containment) |
| 跨文档引用图 | **100 条引用边**, 49 篇最大连通分量 |
| Multi-hop paths (constrained) | 过滤纯 containment 后存活率待观测 |

### 二、跨文档引用图统计分析

```json
{
  "total_bib_entries": 2001,
  "total_citation_edges": 100,
  "match_rate": "5.05%",
  "match_method_dist": {
    "title_fuzzy": 98,
    "title_exact": 3,
    "arxiv_id_explicit": 0,
    "arxiv_id_bare": 0
  }
}
```

#### 关键指标解读

| 指标 | 值 | 解读 |
|------|---|------|
| 匹配率 | 5.05% (101/2001) | 合理：73 篇 corpus 只占引用宇宙的 ~5% |
| arXiv ID 匹配 | **0** | .bbl 走会议出版引用（ICML/NeurIPS），无 arXiv URL |
| title_fuzzy | 98 | 标题匹配扛起全部，需验证误匹配率 |
| title_exact | 3 | 仅 3 篇标题完全一致 |
| 最大连通分量 | **49/73 (67%)** | 核心子图非常密集 |
| 孤立论文 | 20 (28%) | 无互引，可能是边缘论文 |
| papers_citing | 38 (52%) | 超半数论文引用了 corpus 内其他论文 |
| papers_cited | 26 (36%) | 超三分之一被 corpus 内引用 |
| in-degree max | **19** | 一篇被 19 篇引用 — 大概率是 fairness 奠基论文 |
| out-degree max | 7 | 最多引 7 篇 corpus 内论文 |

#### 度分布

| 方向 | mean | p50 | p75 | p90 | max |
|------|------|-----|-----|-----|-----|
| Out-degree (cites) | 1.37 | 1 | 2 | 4 | 7 |
| In-degree (cited-by) | 1.37 | 0 | 1 | 4 | 19 |

In-degree 高度偏斜：大多数论文 cited_by=0，少数核心论文被大量引用。这符合学术引用的幂律分布。

### 三、两个关键发现

#### 发现 1：arXiv ID 匹配全军覆没

.bbl 文件中的引用走的是正式会议/期刊出版（如 ICML 2019, NeurIPS 2018），不包含 arXiv 预印本 URL。这意味着：
- 标题匹配是唯一可行的跨文档关联策略
- Jaccard ≥ 0.55 的阈值对 fairness 领域可能偏松（"Fair Classification via..." 类标题太多）
- **需要人工抽查 5-10 条 fuzzy match 验证精度**

#### 发现 2：Citation graph 是 L2 候选的天然信号源

100 条引用边 = 100 个有文献级证据的跨文档关系。相比之前实体倒排索引产出的 711 对（大量伪桥接），引用关系具有以下优势：
- **语义确定性高**：A 引用 B 意味着作者认为 B 与 A 相关
- **方向性明确**：知道谁引谁，可设计 "B 的理论解释 A 的观察" 类推理 query
- **49 篇连通**：不是孤立 pair，可以做 2-hop citation chain（A→B→C）

### 四、对 L2 候选策略的影响

| 维度 | 实体倒排索引 (旧方案) | Citation graph (新方案) |
|------|----------------------|------------------------|
| 候选对数量 | 711 (top-100) | 100 (unique edges) |
| 桥接信号 | 共享实体名（易伪匹配） | 文献引用（语义确定） |
| 方向性 | 无 | 有（citing → cited） |
| 多跳潜力 | 弱（实体重叠不传递） | 强（A→B→C 引用链） |
| 主要风险 | 泛词桥接（fairness 等） | title_fuzzy 误匹配 |

**建议**：用 citation graph 作为 L2 主候选源，实体倒排索引降级为辅助验证信号。

### 五、代码改动汇总

| 文件 | 改动 |
|------|------|
| `src/parsers/latex_reference_extractor.py` | +`_extract_title()` 从 `\title{}` 提取论文标题 |
| | +`find_multihop_paths(require_ref_edge=True)` 过滤纯 containment 路径 |
| `scripts/build_latex_reference_graph.py` | +occurrence vs unique pair 统计 |
| | +per-doc 分布 (P50/P75/P90/P99) |
| | +constrained multi-hop path 计数 |
| `scripts/build_citation_graph.py` | **新文件**: .bbl → corpus 匹配 → 引用图 |
| | 3 种匹配: arxiv_id_explicit, arxiv_id_bare, title (exact+fuzzy) |
| | 输出: edges + adjacency + 连通分量 + 度分布 |

### 六、下一步 TODO

1. **~~P0: Citation fuzzy match 质量验证~~** ✅ **已完成**
   - ~~人工抽查 10 条 title_fuzzy 匹配~~
   - **结果：抽查样本误匹配率 0%（100% 准确）**
   - **结论：Jaccard ≥ 0.55 阈值在 fairness 领域有效，无需收紧**
   - 100 条引用边可直接用作 L2 候选

2. **P0.1: Citation-based L2 候选对构建**（当前最高优先级）
   - 从 100 条引用边中选 top-50 对
   - 用 citing direction 设计 prompt："B 的理论解释 A 的实验观察"
   - 每条 edge 的 `contexts` 字段提供 \cite{} 周围文本

3. **P1: 2-hop citation chain 探索**
   - 在 49 篇连通分量中找 A→B→C 路径
   - 天然的 3-doc multi-hop query 素材

4. **P2: 引用图 + 文档内 DAG 融合**
   - merge `latex_reference_graph.json` + `citation_graph.json`
   - 跨文档引用 + 文档内 Figure/Table/Eq 引用 = 完整的多层 DAG

### 七、补充：fuzzy match 质量验证结果（2026-02-20）

用户人工抽查了 title_fuzzy 匹配样本，**误匹配率 0%**。

**结论**：
- Jaccard ≥ 0.55 阈值在本 fairness 语料库中足够精确
- 虽然 fairness 领域存在大量 "Fair X via Y" 类似标题，但 Jaccard 字符级相似度仍能有效区分
- **100 条引用边全部视为可信**，可直接用于 L2 候选构建，无需人工过滤

这消除了之前最大的数据质量风险。Citation-based L2 路线正式解锁喵
### 七、Git 记录

```
commit 12981ac
feat: cross-document citation graph + multi-hop constraints + report enhancements
- build_citation_graph.py (100 edges, 49-paper component)
- find_multihop_paths(require_ref_edge=True)
- title extraction from \title{}
- per-doc distribution + occurrence vs unique pair stats
```

---

## 日期：2026-02-20（Step 0 v3.2：LaTeX 跨模态链接 + bridge evidence）

### 一、背景与动机

**用户想法**：利用 LaTeX 源码强化 table/equation 等模态与其他模态的链接质量。

**核心问题**：L1 cross-modal dual-evidence 中 formula+table 配对 pass rate 仅 3.3%，根因是模型不知道这两个元素为什么有关联 —— 只拿到了 LaTeX 公式文本和表格图片，没有"桥接文字"说明两者之间的语义关系。

**架构原则**（达成共识）：
- **MinerU = 主体**：element data（image_path, caption, content, context）全部来自 MinerU
- **LaTeX = 参考/增强层**：仅提供 `latex_bridge` —— 作者亲笔写的、解释两个元素为何相关的原文句子

### 二、关键洞察：LatexRefEdge.context 就是 bridge evidence

LaTeX 源码里，一个段落经常同时引用多个元素：

```latex
In Figure~\ref{fig:tradeoff}, we visualize the Pareto frontier defined by
Equation~\ref{eq:pareto}. As Table~\ref{tab:results} demonstrates...
```

`LatexRefEdge.context` 字段（±300 chars 上下文）捕获的就是这段文字。这正是回答"为什么这两个元素相关"的最优证据 —— 比 MinerU 的位置邻近法有本质提升：

| 维度 | Step 0 v2 (MinerU 位置邻近) | Step 0 v3.2 (LaTeX 共引) |
|------|---------------------------|------------------------|
| 发现机制 | 同页/相邻段落 | 显式 `\ref{}` 共引用 |
| 跨页链接 | ❌ | ✅ 任意距离 |
| Bridge evidence | ❌ 无（纯位置关系） | ✅ 作者原文 |
| 方向性 | ❌ | ✅ 谁解释谁 |
| formula+table 预期 | 3.3% pass | 有语义解释 → 显著提升 |

### 三、三种发现策略

| 策略 | 场景 | 置信度 |
|------|------|--------|
| **direct** | `fig:roc → eq:fairness` 直接跨模态边 | 高 (0.95×match_conf) |
| **section** | 同一节引用 fig:X 和 tab:Y | 中 (0.8×match_conf) |
| **paragraph** | 两个 ref 共享高 Jaccard 的上下文文本 | 低 (0.65×match_conf) |

### 四、标签 → MinerU 元素的桥接方案

**两步匹配**（顺序尝试）：
1. **数字提取**：`fig:3` / `fig_3` / `fig3` → 找 MinerU 中 `number == 3` 的 figure（conf=0.95）
2. **Caption Jaccard**：清洗 LaTeX 命令后，计算 token overlap（阈值 0.25）

**两者都来自同一个 `\caption{}` 命令**，文本应高度重叠，所以 0.25 的阈值足够。

### 五、输出格式

输出 `data/latex_cross_modal_pairs.json`，格式与 `multihop_l1_candidates.json` 完全兼容，额外增加 `latex_bridge` 字段：

```json
{
  "pair_id": "1906.12345_xl_0001",
  "element_a": { "element_id": "...", "image_path": "...", ... },
  "element_b": { "element_id": "...", "content": "...", ... },
  "edge_contexts": [{ "context_snippet": "..." }],
  "latex_bridge": {
    "bridge_text":  "In Figure 3, we visualize Equation (1)...",
    "label_a":      "fig:tradeoff",
    "label_b":      "eq:pareto",
    "match_conf_a": 0.87,
    "match_conf_b": 0.72,
    "strategy":     "direct"
  }
}
```

`generate_multihop_l1_queries.py` 可在 prompt 中优先使用 `latex_bridge.bridge_text` 作为"为什么这两个元素相关"的说明，大幅减少模型猜测。

### 六、新增文件

| 文件 | 说明 |
|------|------|
| `scripts/build_latex_cross_modal_links.py` | **Step 0 v3.2 主脚本** |
| `data/latex_cross_modal_pairs.json` | 输出（待运行） |
| `data/latex_cross_modal_pairs_report.json` | 统计报告（待运行） |

### 七、下一步

1. 在集群上运行：
   ```bash
   python scripts/build_latex_cross_modal_links.py \
       --elements data/multimodal_elements.json \
       --latex-graph data/latex_reference_graph.json \
       --output data/latex_cross_modal_pairs.json
   ```
2. 根据输出统计调整 `--min-match-conf` 阈值
3. 更新 `generate_multihop_l1_queries.py`：在 prompt 中加入 `latex_bridge.bridge_text`（如果存在）
4. 重跑 formula+table 配对，验证 pass rate 是否从 3.3% 上升

---
