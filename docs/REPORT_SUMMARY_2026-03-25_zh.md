# 周报 2026-03-17 ～ 2026-03-25

### 本周完成

- **图架构专利化**：GRAPH_ARCHITECTURE.md v3→v4，完成术语定义表、技术问题法定化表述、LaTeX 抽象、PRF 对比、引证补充，可直接作为专利素材底稿
- **M2 三实验全量完成**：Exp A 难度梯度（Coverage L1=97.1% > L2=61.0% ≈ L3=61.7%）、Exp B 图增强（graph_full MRR +4.0% vs BM25）、Exp C QA 三角（L3 检索覆盖 +6.1%，Graph 核心价值在检索层）
- **Enrichment 消融实验**：Graph 零成本 MRR +0.018 ≈ 花 $3 的 Enrichment +0.013，合用超线性 ×1.73
- **Section 粒度图重建**：引入 section 节点后 hub_overlap 从 9.53% 升至 **90.95%**，解决了图信号覆盖不足的根本问题
- **L2/L3 数据量产**：keyword boost pipeline 重跑，L2=210→344 条、L3=115→143 条（新批次），总数据集 1461 条
- **检索调参实验**：四组超参对比，锁定最优配置（nprop=1.0, decay=0.5, 1-hop, cite=0），graph_full **R@10=0.8585（+8.59%）、MRR=0.6339（+11.65%）**，`continue_expand=True`
- **Evidence packaging**：新 L3 生成 10 条 demo 样本，`package_l3_demo_evidence.py` 完成 content_list_v2 fallback 修复，8/10 element 图片 OK

### 卡点 / 需要帮助

- **新 L3 reasoning_steps 全空**：37 条 pass query 的 `reasoning_steps` 字段全为空、`reasoning_structure` 100% parallel，serial chain prompt 未生效，原因未查明（是解析 bug 还是模型输出问题）
- **L3 QC 梯度待修**：L2→L3 Evidence Coverage 持平（61% vs 62%），难度梯度不明显，L3 数据质量需进一步提升

### 下周计划

- 查明 reasoning_steps 为空的根因，修复后重跑 L3 生成，产出有完整推理链的 L3 数据（目标 50 条 serial chain pass）
- 补充 Exp A/C 对比表格（raw vs enriched 两轮），完成论文 M2 实验章节草稿
- 启动 embedding 语义边实验（`build_embedding_edges.py`），验证 embedding 边是否能补充图结构
