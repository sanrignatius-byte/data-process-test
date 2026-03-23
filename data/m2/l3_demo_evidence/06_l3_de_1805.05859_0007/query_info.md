# l3_de_1805.05859_0007

- **pair_type**: figure+formula
- **cross_doc**: True
- **hop_distance**: 3
- **reasoning_depth**: 3
- **element_ids**: ['1805.05859_formula_2', '1607.06520_figure_5']

## Query

Which debiasing method best minimizes the analogue of the disturbance term’s unwanted contribution to Y, when treating stereotypic analogies as the outcome affected by embedding changes?

## Answer

Y is modeled as a linear combination of inputs plus an exogenous disturbance term, so unwanted variation can be viewed as contribution from unmodeled factors. The bridge links that structural-equation view of an outcome to the debiasing setting where stereotypic analogies are the measured outcome under embedding changes. In that setting, hard-debiased embeddings keep stereotypic analogies near 0 until about 60 generated analogies and below 10 even near 150, so hard debiasing most effectively minimizes the unwanted contribution.
