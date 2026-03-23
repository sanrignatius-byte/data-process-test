# 十条推理逻辑最强、桥接最显著且不跨文档的 query

数据源：`data/m2/l3_reasoning_chain_queries_pass.jsonl`
筛选原则：
1. 两个证据元素均来自同一篇文档；
2. `qc_pass = true`；
3. 优先选择“前提 → 桥接机制 → 结论”三段链条最完整、桥接语义最明确的样本；
4. 优先选择关键 evidence 可直接支撑结论、且不存在明显跨文档依赖的 query。

---

## 1. l3_de_1703.06856_0012
**Query**：How does the race-swap arrest change motivate using counterfactual fairness, and what predictor form satisfies it by excluding descendants of A?

**Reasoning chain**
- ☑ **Evidence 1（前提）**：`the overall number of arrests decreases (from 5659 to 3722)`
- ☑ **Bridge**：`A predictor \hat Y is counterfactually fair if ... the prediction is the same ... where A had been different`
- ☑ **Evidence 2（结论）**：`P(\hat Y_{A\leftarrow a}(U)=y\mid X=x,A=a)=P(\hat Y_{A\leftarrow a'}(U)=y\mid X=x,A=a)`

**中文链路概括**：
仅交换种族就让 arrest 总量从 5659 降到 3722，说明模型输出受保护属性 `A` 影响；因此需要引入 **counterfactual fairness** 作为桥接原则；最终满足该原则的 predictor 形式，就是在反事实干预下对 `A` 保持预测不变、从而排除 `A` 后代路径影响的预测器。

---

## 2. l3_de_1804.06876_0024
**Query**：Which linkage in the physician–secretary pair is anti-stereotypical, and why must pronoun gender not affect that decision?

**Reasoning chain**
- ☑ **Evidence 1（前提）**：`physician 38 ... secretary 95`
- ☑ **Bridge**：`When woman dominate profession ... female pronoun is pro-stereotypical, and male pronoun is anti-stereotypical`
- ☑ **Evidence 2（结论）**：`the gender of the pronominal reference is irrelevant for the co-reference decision`

**中文链路概括**：
`secretary` 的女性占比显著更高，而 `physician` 没那么女性化，因此若把 `secretary` 与男性代词相连，就是 **anti-stereotypical**；但共指决策本身不应被代词性别左右，所以即使是 anti-stereotypical 版本，也仍应按语义解析，而不是被性别线索带偏。

---

## 3. l3_de_1904.03035_0028
**Query**：Which λ best balances low Ppl. with bias reduction, through what regularization mechanism, and where does that mechanism act in the language model?

**Reasoning chain**
- ☑ **Evidence 1（前提）**：`0.5  0.34  0.48  0.14  0.79  1.31  0.20  116.19`
- ☑ **Bridge**：`Cross Entropy Loss + λ(N.B)`
- ☑ **Evidence 2（结论）**：`λ controls the importance of minimizing bias in the embedding matrix`

**中文链路概括**：
表中 `λ = 0.5` 同时给出最低 perplexity 且偏置指标较低，因此是最优平衡点；其桥接机制是训练目标由 `Cross Entropy Loss + λ(N.B)` 构成，也就是把偏置惩罚项显式加入语言模型训练；而这一机制具体作用在 **embedding matrix** 上，因此能在保留语言建模性能的同时压低性别偏差。

---

## 4. l3_de_1907.12059_0037
**Query**：Which fairness metric most directly trades off with Err-Exp for the Bank set under Wass-1 Penalty DB, and what mechanism links that metric to rising error?

**Reasoning chain**
- ☑ **Evidence 1（前提）**：`Wass-1 Penalty DB  .131  .001  .006  .018`
- ☑ **Bridge**：`ensure that the output of a classification system does not depend on sensitive information using the Wasserstein-1 distance`
- ☑ **Evidence 2（结论）**：`as the learning model moves towards the fairness goal of SDP, model accuracy decreases (Err-Exp increases)`

**中文链路概括**：
在 Bank 数据集上，Wass-1 Penalty DB 对应极低的 SDD / SPDD，这说明它直接优化的是“输出对敏感属性独立”的公平目标；桥接点在于这种独立性是通过 **Wasserstein-1 distance** 约束实现的；因此当模型更逼近公平目标时，误差 `Err-Exp` 会同步上升，形成清晰的 fairness–accuracy tradeoff。

---

## 5. l3_de_1905.03674_0050
**Query**：How does the triangle-inequality bound in the second case support proportionality by ruling out a blocking coalition of size \(\lceil n/k\rceil\)?

**Reasoning chain**
- ☑ **Evidence 1（前提）**：`By the triangle inequality ... d(i*,x) <= r_y + d(i,x) + d(i,y)`
- ☑ **Bridge**：`the representative center i* witnesses that agents near y cannot all strictly improve relative to X`
- ☑ **Evidence 2（结论）**：`S is a blocking coalition against X if |S| >= ceil(n/k) and ... for all i in S, d(i,y) < D_i(X)`

**中文链路概括**：
三角不等式先给出一个上界，把候选中心 `y` 与现有配置 `X` 的距离关系约束住；桥接步骤说明这个 bound 意味着靠近 `y` 的代理人不可能“全体都严格改善”；于是就无法满足 blocking coalition 的严格定义，也就证明了该情形下 proportionality 成立。

---

## 6. l3_de_1607.06520_0116
**Query**：Which debiasing method best preserves analogy quality while increasing appropriate analogies, and how is that consistency reflected in RG, WS, and MSR-analogy?

**Reasoning chain**
- ☑ **Evidence 1（前提）**：`Number of stereotypical (Left) and appropriate (Right) analogies generated ... before and after debiasing`
- ☑ **Bridge**：`nurse is moved to be equally male and female in the direction g`
- ☑ **Evidence 2（结论）**：`The results show that the performance does not degrade after debiasing`

**中文链路概括**：
图中可以看到 debiasing 后 stereotype analogies 减少、appropriate analogies 增加；桥接机制是把诸如 `nurse` 这样的职业词沿性别方向 `g` 做中和处理；因此最优方法是 **hard-debiased** ——它既提升了更合适的 analogy，又没有在 RG、WS、MSR-analogy 上造成性能退化。

---

## 7. l3_de_1610.07524_0120
**Query**：How does the total variation bound on disparate impact connect to the observed Black-White COMPAS histogram separation under the high-risk cutoff?

**Reasoning chain**
- ☑ **Evidence 1（前提）**：`disparate impact Δ is related to ... percent non-overlap measure`
- ☑ **Bridge**：`under the MinMax policy, total variation distance ... determines a sharp bound on Δ`
- ☑ **Evidence 2（结论）**：`COMPAS decile score histograms for Black and White defendants ... d = 0.60, non-overlap`

**中文链路概括**：
首先，论文把 disparate impact `Δ` 与分布不重叠程度联系起来；桥接处进一步指出，在 MinMax policy 下，这个联系可被 **total variation distance** 精确界定；因此当 COMPAS 中 Black/White 的 decile score histogram 已出现明显分离且 non-overlap 较大时，就能推出高风险阈值下存在相应的 disparate impact。

---

## 8. l3_de_1610.08452_0123
**Query**：When the covariance-based constraint is applied to both false positive and false negative mistakes, what accuracy and error-rate gaps result, and why?

**Reasoning chain**
- ☑ **Evidence 1（前提）**：`Cov(z, d_θ(x))`
- ☑ **Bridge**：`as the covariance threshold is decreased, the false positive rates ... become closer, but the accuracy also decreases`
- ☑ **Evidence 2（结论）**：`0.645  -0.01  -0.01`

**中文链路概括**：
方法本身通过约束 `Cov(z, d_θ(x))` 控制敏感属性与决策边界距离的相关性；桥接结论指出，一旦该 covariance threshold 被压低，FPR/FNR gap 会缩小，但准确率会下降；最终在同时约束 FP 与 FN 的设置下，得到的结果就是 **accuracy = 0.645，两个 error-rate gap 约为 -0.01 / -0.01**。

---

## 9. l3_de_1611.07509_0125
**Query**：Which mediator change links the path-specific loan approval shift under treatment as white to the recanting witness example’s disadvantage-group outcome?

**Reasoning chain**
- ☑ **Evidence 1（前提）**：`expected change in loan approval ... when the bank is instructed to treat the applicants as from the advantage group`
- ☑ **Bridge**：`the path-specific effect can be estimated from the observational data, known as the recanting witness criterion`
- ☑ **Evidence 2（结论）**：`same racial makeups shown in the Zip code as the advantage group`

**中文链路概括**：
前提是：银行把 disadvantaged group “按 white 对待”时，贷款批准率会发生路径特异性的变化；桥接机制是 **recanting witness criterion**，它说明这种 path-specific effect 可以通过观察数据识别；由此落到具体 mediator 上，就是把 disadvantage group 的 **Zip code racial makeup** 调整为与 advantage group 相同。

---

## 10. l3_de_1706.02744_0129
**Query**：Which criterion rules out the indirect A→X→R influence in admissions by requiring intervention-based invariance over the proxy path?

**Reasoning chain**
- ☑ **Evidence 1（前提）**：`R does not only directly depend on gender A, but also on department choice X, which in turn is also affected by gender A`
- ☑ **Bridge**：`The proxy P`
- ☑ **Evidence 2（结论）**：`connect our interventional approach to individual fairness and other proposed criteria`

**中文链路概括**：
录取结果 `R` 不仅受性别 `A` 直接影响，还会经由 `A→X→R` 这条间接路径被影响；桥接点是把中间变量视为 **proxy**；因此对应的最佳 criterion 就是要求对这条 proxy path 施加 **intervention-based invariance**，从而排除代理歧视。

---

## 简短结论
这 10 条里，桥接最明显的共同模式都是：
- 先由一个 **观测事实 / 图表数值 / 公式约束** 给出前提；
- 再通过一个 **机制定义**（如 counterfactual fairness、Wasserstein-1、path-specific effect、covariance constraint、proxy discrimination）把前提“翻译”为可推理的中间层；
- 最后再落到 **具体结论**（最优 λ、最优 debiasing 方法、具体 fairness tradeoff、具体 mediator、具体禁止的路径）。

如果你愿意，我下一步可以继续帮你把这 10 条整理成：
1. **更适合标注的数据表格版**；或
2. **只保留 query + 中文 reasoning chain 的精简版**；或
3. **导出成新的 JSON/JSONL 文件**。
