# l3_de_1703.06856_0021

- **pair_type**: figure+table
- **cross_doc**: True
- **hop_distance**: 3
- **reasoning_depth**: 3
- **element_ids**: ['1703.06856_table_1', '1607.06520_figure_5']

## Query

Which debiasing approach best mirrors Fair K and Fair Add by sacrificing less biased behavior for some performance cost, and what outcome appears as generated analogies increase?

## Answer

Full and Unaware achieve the highest logistic-regression accuracy, while Fair K and Fair Add accept a small accuracy drop to ensure counterfactual fairness. The bridge links this trade-off to debiasing under the shared themes of logistic regression, counterfactual fairness, and accuracy. In the embedding setting, hard debiasing best matches that fairness-first strategy, because as generated analogies increase it keeps stereotypic analogies far lower than before or soft debiasing, remaining below about 10 even near 150.
