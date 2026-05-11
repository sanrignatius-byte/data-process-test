# 周报 5/10 — 数据图组（DPT）

汇报人：你（草稿，自己审一遍再发）
覆盖区间：2026-05-04 周 → 2026-05-10
上次周报对应：5/2 mentor 录音 60 那次会议（18 条 todo）

---

## 零、对照上次给的 4 条 to-do 状态

| # | To-do | 状态 | 一句话结论 |
|---|---|---|---|
| 1 | 修改 enrich 方式，模仿 summary+细节 格式参与找回 | ⚠️ Blocked | 公司 API 自 4/21 401 至今 19 天；同时 `claim:C8` 已证 MODORA 式视觉 enrich 对 text-style retrieval 净负，方案要重设计 |
| 2 | 用 VL-Embedding 让图片参与 dense retrieval | ❌ 跑过两轮全负 + 发现 corpus 端 bug | 详见下方 #1，**表格图片在 corpus 构建时被吃掉了**，不是 VL 模型本身的问题 |
| 3 | 统计 chunk 平均含几个 element、查 R@10 低的根因 | ✅ 完成并升级为 claim | 964 chunk 平均 1.94 element；75% 双证据 query 跨 chunk；K=1 zero rate 71%；已成 `claim:C9` |
| 4 | 原有 md 文档更新，保证可读性 + 说人话 | ✅ 完成 | `文档建图.md` 5/10 重写：术语统一（element 四类平级）、图加权改大白话、第十节重名修复、第十一节 verdict 加 modality 选择性 |

完成率 2/4，1 项被 API 卡住，1 项发现新 bug（见 #1）。

---

## 一、本周做完的事（按重要度）

### 1. 公式检索瓶颈被推开了一道口子（这周最重要的事）

之前 10 个配置（图拓扑改、切片重建、把段落上下文注入公式、把 LaTeX 表面归一化、换 reranker 家族）公式题的 R@10 全都卡在 0.56。

这次换了思路：**把公式段落单独用一个数学专精的模型重新编码**（Qwen2.5-Math-7B，挑它是因为它在 LaTeX 源码上专门训练过），然后跟原来 Qwen3-Embedding-4B 的检索排名做倒数排名融合。

**结果**

| 指标 | 之前 | 这次 (k=60) | 差 |
|---|---|---|---|
| 公式题 R@10（179 道，全集） | 0.5600 | **0.6313** | **+7.3pp** |
| 公式题 R@10（25 道，平衡冒烟集） | 0.5600 | 0.6000 | +4.0pp |
| 公式题 R@100（全集） | 0.8636 | 0.9441 | +8.0pp |

这是 4/17 之后第一个真正推动公式天花板的信号。原来"模型本身就编码不了 LaTeX"这个结论被部分推翻了。

**但有代价**：整体 R@10 跌了 9.7pp（从 dense 0.6195 跌到 0.5222）。原因是融合权重做得太简单：公式段落同时拿到 Qwen3 和 Math 两票，图/表段落只有 Qwen3 一票，所以图/表的相对位置被压下来了。

**下周怎么修**：只在 query 看起来是公式题时才触发 Math 编码器（query 里有数学符号 / 公式关键词）。这样图/表题目走原来的路径不动，公式题走融合路径。

### 2. 50 道平衡冒烟测试集跑完了（5/2 mentor 提的）

按 mentor 5/2 录音里要求的"10 文本 / 10 图 / 10 表 / 10 公式"做。当前 query 集合里没有纯文本题，所以实际比例是 17 fig / 17 tab / 16 formula。

**结论**：之前 0.6913 那个上限不是数据偏置造出来的（比如 figure 占 56%），冒烟集上图增益 +1.87pp 在误差范围内，上限是真的。

**但**：图增益不是均匀的——fig +10.3pp / tab +8.3pp / **formula 0pp**。意味着论文里 C1 / C5 / C7 三个 claim 都要明确写"对图、表有效；对公式无效"，不能笼统地说图增益。

### 3. mineru → latex 元素级匹配率审计（B2）

之前一直没有具体数字，只有"大概 50%"和"92%"两种说法。这次手动核完：

- figure: 49.7% 元素级匹配
- table: 67.3% 元素级匹配
- formula: **0%** 元素级匹配

92% 那个数字是文档级覆盖率（56/57 = 98.2%），跟元素级是两回事。

**这件事解释了 #2 里 formula 为什么没图增益**：mineru 给的 formula label 跟 LaTeX 源里的 label 完全没打通，图上根本没有边连到 formula 节点上去，图信号传不进来。

### 4. 用 LaTeX 行号重建 chunk-element 边（B1）

按 mentor 要求把字符串模糊匹配换成 LaTeX 行号。Phase 1 用内容 jaccard 绕过 label key 通道，把 formula 的元素匹配从 0% 救到 41.2%。Phase 2 重建 chunk-element 边，新边和老边几乎完全不重合（修了 1130 条，删了 529 条）。

**实测**：拓扑虽然改了，但 explicit-only 这个图配置下 R@10 完全不动。原因是这个配置走桥接节点（hub）这条路，不经过 chunk-element 这条边。

**结论**：B1 的工程价值在 QA / SFT 数据合成那边（对齐准了，证据定位不容易出错），不在 retrieval ceiling。retrieval ceiling 卡在公式编码上（已经被 #1 部分推开）。

### 5. 文档建图.md 重写

按你 5/2 发的版本改了。统一术语：

- 元素（element）：text / figure / table / formula 四种平级，inline 公式不算独立元素
- 切片（chunk）：合并连续文本元素得到的检索单元
- 图加权用大白话写：出入度、阅读顺序分数传播、引用边分数传播、汇总加权
- 投影边：元素层引用边映射到切片层

第十节标题之前重复了（跟第七节同名），改了。第十一节 verdict 加了模态选择性的说明。

### 6. 内部杂项

- chunk 在双证据题上稀释信号这件事坐实了（claim:C9 入库）：双证据题里 75% 两个证据落在不同切片，K=1 时 71% query 一个证据都召不到
- BCD 阶段执行整体完成度从 32% 推到 76%

### 7. 新发现：VL embedding 在 table 上归零的真正根因不是模型

本周复盘 to-do #2 时挖到的 bug，比之前归因更严重：

| 层级 | figure 带图比例 | table 带图比例 |
|---|---:|---:|
| mineru 原始输出 | 100% | **100%**（163/163 抽样） |
| `multimodal_elements.json` 图层 | 841/841 | 237/334（97 个是 inline HTML，本来就无图） |
| `corpus_v1_enriched.jsonl` **检索 corpus** | 842/842 ✅ | **0/2 ❌** |

也就是说 **mineru 给的 table crop 图片在 corpus 构建那一步几乎全被丢了**：1798 条 passage 里只剩 2 条 table 类型，且都不带 `image_path`。VL embedding 脚本里 `resolve_image_path(table)` 必然返回 None → 走文本编码 → table R@10 归零。

之前我把"split_vl2b_t5 table R@10 = 0.0278"归因为"split allocation 不适合 table"，**这个归因错了**。修复方向：

- 不是去改 VL 脚本
- 而是改 corpus 构建：把 `multimodal_elements.json` 里 237 个 table-with-image 显式作为独立 passage 注入 corpus，带 `image_path` 字段
- 修完之后再跑一次 VL split，table lane 才有公平对比的可能

预计这个 corpus 修复 + 重跑 VL 大约 1 天 + 30 min A6000。下周可以排进去。

---

## 二、卡住的事

### B4 全量 element enrich

公司 API endpoint `az.gptplus5.com` 自 4/21 起返回 401 unauthorized，到今天 19 天没恢复。当前覆盖 10988 / 27209 = 40.4%，剩下的元素一个都加不了。

OpenAI 直连可以 fallback 但成本是公司 API 的约 10 倍。**等师兄拍板**是否切。

### C5 多粒度 enrich（DocResearcher 风格）

同样卡在 API 上。另外这条线跟 [claim:C8](../research-wiki/claims/C8_modora_visual_enrichment_net_negative.md)（视觉描述注入会反向）有冲突，不是 API 通了就能直接做，需要重新设计方案。

---

## 三、下周打算（5/13–5/17）

| 优先级 | 事 | 估时 |
|---|---|---|
| P0 | F-formula Phase 2a：query 感知路由（query 含数学符号才触发 Math 编码器） | 1-2 天 + ~30 min A6000 |
| P0 | **corpus 端补 table-with-image passage**，重跑 VL split（对应 to-do #2 修复） | 1 天 + 30 min A6000 |
| P0 备用 | F-formula Phase 2b：两阶段——dense 取 top-100 后只对公式候选重排 | 同上 |
| P1 | 给 C1 / C5 / C7 论文 claim 加 modality scope（配合 C10） | 半天 |
| P2 | 拿 jina-embeddings-v3 跑一次独立家族对照（排除"是不是 Qwen 系列偏置"的疑问） | 半天 + ~20 min A6000 |

---

## 四、要你拍板的事

1. **API 切不切 OpenAI 直连**？切了 B4 / C5（to-do #1）才能继续推。不切的话这两条线继续挂着。
2. **F-formula 整体 R@10 −9.7pp 的代价能不能接受**？我倾向 Phase 2a 把它修回去（保住整体的同时留住公式增益）。如果你觉得公式增益本身就够发论文，那 Phase 2 可以省掉直接收工。
3. **VL split 还要不要继续推（to-do #2）**？现在已知 corpus 端有 bug 把 table 图丢了。修完 corpus 后值得再给 VL 一次机会，但 smoke50 上 figure VL 0.33 vs graph 0.82 这个 49pp 的差距摆在那，**就算 table 修好，VL 路线大概率还是输给 graph rerank**。我倾向"修一次 + 跑一次"作为收尾，结论清楚后归档，不再继续推。
4. **下次跟你同步**：还按 mentor 录音 60 之前的节奏（每两周一次大讨论），还是改成每周？

---

## 附：所有材料的位置

- 实验记录在 `research-wiki/experiments/20260510_*`
- 决策报告在 `refine-logs/*_20260510.md`
- 数据 / ranking / qrels 在 `data/05_eval/dense_retrieval/qwen25math_formula_routing/`
- claim 定义在 `research-wiki/claims/C{9,10,11}*.md`
- BCD 主计划在 `refine-logs/BCD_PHASED_PLAN_20260510.md`

GitHub commits 这周：`fa29c84`（5/10 上午 BCD 执行）→ `cbd3a01` → `4d859d0` → `904189d`（5/10 傍晚 F-formula 结果）。
