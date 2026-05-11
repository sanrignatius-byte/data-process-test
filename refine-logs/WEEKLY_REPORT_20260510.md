# 周报 5/10 — 数据图组（DPT）

汇报人：你（草稿，自己审一遍再发）
覆盖区间：2026-05-02 → 2026-05-10
对应议程：5/2 mentor 录音 60 给的 4 条 to-do + 自走线（reranker / smoke50 / F-formula / 文档建图）

---

## 零、一句话总结

4 条 to-do 完成 2 条（#3 #4），1 条证伪（#2，且发现 corpus 端把 mineru 的 table 抠图全丢了），1 条卡 API 加被 C8 反证（#1）。本周期外的最大收益是 F-formula 那条线 —— Qwen2.5-Math-7B 第一次把公式 R@10 天花板从 0.56 抬到 0.6313 (+7.3pp)。

---

## 一、对照 5/2 给的 4 条 to-do

### To-do #1 — 修改 enrich 方式，模仿 summary+细节 格式参与找回

**状态**：⚠️ Blocked + 与已有 claim 冲突，本周没继续推

**做了什么**：
- `exp:20260503_corpus_enrich_fix` — 修了两个独立 bug（`load_enriched_index` 漏读 MODORA 格式 1285 条；`build_element_text` 优先级 OR 在 enrich 命中时跳过 graph context）。两个 corpus 变体都试了：
  - `fix_v1`（replace）dense R@10 = **0.5106 (−10.9pp)**
  - `fix_v2`（additive，平均 figure passage 长度 683）dense R@10 = **0.5888 (−3.1pp)** / R@100 = 0.8436 (−2.0pp) / graph_static_plus_neighbor R@10 = 0.6860 (−0.5pp)
  - **D5 verdict：不晋升**，DEFAULT_ENRICHED_FILES 已回滚

**为什么 blocked**：
1. 公司 API endpoint `az.gptplus5.com` 自 4/21 起 401 unauthorized，本周仍未恢复（19 天）。当前 element enrich 覆盖 10988/27209 = 40.4%
2. 5/3 三条独立证据收敛到 `claim:C8`：MODORA 风格视觉/语义 enrich 在 text-style scientific QA retrieval 上 **净负**（corpus replace −10.9pp / additive −3.1pp / graph rerank −0.5pp）。"summary + 细节"是同一类思路，**不是 API 通了就能直接做**，方案要重设计

**这条线下一步要的不是 API，是先想清楚为什么 enrich 反向**。

---

### To-do #2 — 用 VL-Embedding 让图片参与 dense retrieval

**状态**：❌ 跑过两轮全负 + 本周复盘挖到 corpus 端 bug

**做了什么**：
- `exp:20260502_split_modality` — 0.6B/4B text 模型按 modality split 检索，4B mixed-index R@10 = 0.4767 vs unified 0.6195，**split 反而掉**（H2 成立）；text-only 对 figure/table 的 `[Image: xxx]` 占位符无语义（H3 成立）
- `exp:20260502_split_modality_vl_failed` — Qwen3-VL-Embedding-2B 跑出 R@10 = **0.0021**（准随机），当时归因为"checkpoint 缺权重"
- `exp:20260503_split_modality_vl_t5_rerun` — 复查根因：transformers 4.57 vs 模型要 5.2+，权重多了一层 `model.` 前缀没被剥离 → **language 塔被随机初始化**。用 transformers 5 overlay 修好后 625/625 weights clean load，R@10 升到 **0.2579**（仍弱于 split_4B_text 0.4767 和 unified 4B 0.6195）。Per-modality：figure +13pp 涨到 0.54，**table 归 0，formula 归 0**
- `exp:20260505_smoke50_balanced_audit` — 50 道平衡冒烟集上 split_vl2b_t5 figure R@10 = 0.33 vs **graph rerank figure R@10 = 0.82**（−49pp），Phase C VL fusion 触发条件失败

**本周复盘新发现（需要重新归因）**：

之前以为 table 在 VL 下归零是"split allocation 不适合 table"。本周对照 to-do 时核数据，发现真正根因在 **corpus 端**：

| 层级 | figure 带图 | table 带图 |
|---|---:|---:|
| mineru 原始输出 | 100% | **100%**（163/163 抽样） |
| `multimodal_elements.json` 图层 | 841/841 | 237/334（97 个 inline HTML 本就无图） |
| `corpus_v1_enriched.jsonl` **检索 corpus** | 842/842 ✅ | **0/2 ❌** |

**1798 条 passage 里只剩 2 条 table 类型，且都不带 `image_path`**。VL 脚本里 `resolve_image_path(table)` 必然返回 None → table 走文本编码 → R@10 归零。

mineru 抠的 table .jpg 文件好端端在磁盘上，是 corpus 构建管道把 table 合并/丢弃了。要修这条线，先修 corpus，不是改模型。

---

### To-do #3 — 统计 chunk 平均含几个 element，查 R@10 低根因

**状态**：✅ 完成并升级为 paper claim

**做了什么**：
- `exp:20260502_chunk_element_coverage` — 57 docs / 964 chunks 上：
  - 平均 elements/chunk = **1.94**（中位 2，52% 只含 1 个）
  - element 类型分布：formula 51.6% / figure 32.2% / table 10.0% / section 6.2%
  - **双证据 query：75.5% 两个 evidence 落在不同 chunk**，只 2.1% 同 chunk
- `exp:20260503_chunk_query_element_recall` — per-query 视角验证：n500 partial-overlay lane 上 chunk R@10 = 0.678 vs **elem R@10 = 0.530（15pp gap）**，K=1 zero-rate **71%**
- `exp:20260503_failure_profiling` — 121 个 partial+zero query 的 rank-of-missed 分布：69% 漏掉的 qrel 在 rank (100, 500]（→ 触发 R2 cross-encoder 路线），formula 中 form_high=0.016 否决了"encoder 在 formula 上崩塌"假说
- `exp:20260510_b1_phase2_lineno` — 顺手修了 chunk_contains_element 边的 P1 bug（之前与 eval qrels chunk_id **0% 一致**），用 LaTeX 行号重建（kept 20 / added 1130 / removed 529）

**5/10 升级为 `claim:C9` "chunk dilutes double evidence signal"**，scope = M4query_v1 elem-level qrels。Mentor C2 todo（重新审视 chunk 是不是噪声）从 ⚠️ 升 ✅。

---

### To-do #4 — 在原有 md 文档上更新，保证可读性 + 说人话

**状态**：✅ 完成

**做了什么**：
- `文档建图.md` 5/10 重写：
  - 术语统一：元素四类（text / figure / table / formula）平级，inline 公式不算独立元素
  - chunk 定义：合并连续文本元素得到的检索单元
  - 图加权改大白话：出入度 / 阅读顺序分数传播 / 引用边分数传播 / 汇总加权 / 投影边
  - 第十节标题重名修复（之前跟第七节同名），第十一节加 modality 选择性 verdict
- 新建 `research-wiki/reference/multimodal_element_taxonomy.md`（B3 todo 配套，含 modality 分布快照）
- `exp:20260503_mentor_recording60_full_todo` — 把录音 60 拆成 18 条 todo 入 wiki，便于跟踪

---

## 二、to-do 之外，本周也做完的事（按重要度）

### 1. 公式检索瓶颈被推开了一道口子（这周最重要）

之前 10 个配置（图拓扑改、切片重建、把段落上下文注入公式、把 LaTeX 表面归一化、换 reranker 家族）公式题 R@10 全卡在 **0.56**。

`exp:20260510_f_formula_qwen25math_routing` —— 把公式段落单独用 **Qwen2.5-Math-7B**（在 LaTeX 源码上专门训练过）重新编码，跟 Qwen3-Embedding-4B 的检索排名做 RRF：

| 指标 | 之前 | 这次 (k=60) | 差 |
|---|---|---|---|
| 公式题 R@10（179 道，全集） | 0.5600 | **0.6313** | **+7.3pp** |
| 公式题 R@10（25 道，平衡冒烟集） | 0.5600 | 0.6000 | +4.0pp |
| 公式题 R@100（全集） | 0.8636 | 0.9441 | +8.0pp |

4/17 之后第一个真正撬动公式天花板的信号。`claim:C11`（formula ceiling is dense-encoder bound）部分证伪 —— math-aware encoder + fusion 是第一根有效的撬棍。

**代价**：整体 R@10 跌 **9.7pp**（dense 0.6195 → 0.5222），融合权重做得太简单（公式有两票，图/表只有一票，相对位置被压下去）。下周 Phase 2a 修：query 看起来是公式题才触发 Math 编码器。

### 2. F-formula 之前两轮失败也跑完了，可以收尾

- `exp:20260510_f_formula_caption` — 把 mineru `context_before` 注入公式 passage，重编码。Dense R@10 0.6195 → **0.5825 (−3.7pp)**，graph 0.6913 → 0.6691 (−2.2pp)，formula 桶跌 16pp。**HD verdict** —— 文本增强救不了 LaTeX
- `exp:20260510_f_formula_math_norm` — LaTeX 表面归一化，10 configs 全没破 0.56。**HD FAIL**

两轮一起强化了 C11，也把 #1 那个 math encoder 路线证成"唯一剩下的可行方向"。

### 3. 50 道平衡冒烟测试集跑完了

`exp:20260505_smoke50_balanced_audit` — 按 mentor 5/2 录音"10 文 / 10 图 / 10 表 / 10 公式"做。M4query_v1 没有 text qrel，实际只能 17 fig / 17 tab / 16 formula。

**verdict：S2 命中** —— ceiling 0.6913 是真的，不是 figure-heavy artifact（smoke50 graph 0.71 vs full 0.69，偏差 +1.87pp 在误差内）。

但 graph 增益不均匀：figure +10.3pp / table +8.3pp / **formula 0pp**。Paper claim C1/C5/C7 都要加 modality scope。`claim:C10` 已入库。

### 4. 三轮 reranker 全部证伪（4/21 起的剩余收尾）

- `exp:20260503_ce_rerank_bge` — BGE-reranker-v2-m3，R@10 跌到 **0.4482 (−17pp)**，MRR 跌 24pp。原因：严重 text-bias，top-1 modality {text 348, figure 87, table 29, formula 9}。RRF 救回 R@100 +2.3pp 但 R@10 仍 0.6258 没破 ceiling
- `exp:20260503_qwen3_rerank_fusion` — Qwen3-Reranker-4B 各种 fusion，全负。BGE pilot 那个唯一正向信号 R@100 +2.3pp 在 Qwen3 上没复现
- 关键发现：BGE 偏 text，Qwen3 偏 formula，**模态偏置完全相反**。"换 reranker 家族"路线整体证伪

### 5. mineru → latex 元素级匹配率审计（B2）

之前一直只有"50%"和"92%"两个口径在打架。本周手动核完：figure **49.7%** / table **67.3%** / formula **0%**。92% 是文档级覆盖率（56/57 = 98.2%），跟元素级是两回事。

**这件事解释了 graph rerank 在 formula 上为什么 0pp 增益** —— mineru 给的 formula label 跟 LaTeX label 完全没打通，图上根本没边连到 formula 节点，graph 信号传不进来。

### 6. 用 LaTeX 行号重建 chunk-element 边（B1）

按 mentor 要求把字符串模糊匹配换成 LaTeX 行号。Phase 1 用内容 jaccard 把 formula 元素匹配从 0% 救到 41.2%。Phase 2 重建 chunk-element 边，新边和老边几乎完全不重合（修了 1130 条，删了 529 条）。

**实测**：拓扑虽然改了，但 explicit-only 这个图配置下 R@10 完全不动。原因是这个配置走桥接节点（hub）这条路，不经过 chunk-element 这条边。

**结论**：B1 的工程价值在 QA / SFT 数据合成那边（对齐准了，证据定位不容易出错），不在 retrieval ceiling。

### 7. 内部杂项

- BCD 阶段执行整体完成度从 32% 推到 76%
- mentor 18 条 todo 草稿（D1）写好待 user 审

---

## 三、卡住的事

### B4 全量 element enrich

公司 API endpoint `az.gptplus5.com` 自 4/21 起 401 unauthorized，到今天 19 天没恢复。当前覆盖 10988 / 27209 = 40.4%，剩下的元素一个都加不了。OpenAI 直连可以 fallback 但成本约 10×。**等师兄拍板**。

### C5 多粒度 enrich（DocResearcher 风格）

同样卡 API。另外这条线跟 `claim:C8`（视觉/语义 enrich 反向）有冲突，不是 API 通了就能直接做，需要重新设计方案。

---

## 四、下周打算（5/13–5/17）

| 优先级 | 事 | 估时 |
|---|---|---|
| P0 | F-formula Phase 2a：query 感知路由（query 含数学符号才触发 Math 编码器） | 1-2 天 + ~30 min A6000 |
| P0 | **修 corpus 端 table-with-image passage**，重跑 VL split（对应 to-do #2 收尾） | 1 天 + 30 min A6000 |
| P0 备用 | F-formula Phase 2b：两阶段——dense 取 top-100 后只对公式候选重排 | 同上 |
| P1 | 给 C1 / C5 / C7 论文 claim 加 modality scope（配合 C10） | 半天 |
| P2 | 拿 jina-embeddings-v3 跑一次独立家族对照（排除"是不是 Qwen 系列偏置"的疑问） | 半天 + ~20 min A6000 |

---

## 五、要你拍板的事

1. **API 切不切 OpenAI 直连**？切了 to-do #1 (B4 / C5) 才能继续推。不切的话这两条线继续挂着
2. **F-formula 整体 R@10 −9.7pp 的代价能不能接受**？我倾向 Phase 2a 把它修回去（保住整体的同时留住公式增益）。如果你觉得公式增益本身就够发论文，那 Phase 2 可以省掉直接收工
3. **VL split 还要不要继续推（to-do #2）**？现在已知 corpus 端 bug 把 table 图丢了。修完 corpus 后值得再给 VL 一次机会，但 smoke50 上 figure VL 0.33 vs graph 0.82 这个 49pp 的差距摆在那，**就算 table 修好，VL 路线大概率还是输给 graph rerank**。我倾向"修一次 + 跑一次"作为收尾，结论清楚后归档，不再继续推
4. **下次同步节奏**：还按 mentor 录音 60 之前的两周一次，还是改成每周？

---

## 附：所有材料的位置

- 实验记录在 `research-wiki/experiments/2026050{2,3,5}_*` + `research-wiki/experiments/20260510_*`
- 决策报告在 `refine-logs/*_2026050{3,5,10}.md`
- 数据 / ranking / qrels 在 `data/05_eval/dense_retrieval/qwen25math_formula_routing/`、`data/05_eval/smoke50/`
- claim 定义在 `research-wiki/claims/C{8,9,10,11}*.md`
- BCD 主计划在 `refine-logs/BCD_PHASED_PLAN_20260510.md`
- 文档建图 `文档建图.md`（根目录），多模态术语 `research-wiki/reference/multimodal_element_taxonomy.md`

GitHub commits 这周：`fa29c84`（5/10 上午 BCD 执行）→ `cbd3a01` → `4d859d0` → `904189d`（5/10 傍晚 F-formula 结果）。后续提交进 PR #167（branch `claude/check-project-progress-WPUht`）。
