# Reasoning Chain

## Step 1: premise

- **evidence_element**: 1703.06856_table_1
- **evidence_type**: observation
- **depends_on**: []

### Evidence Span

we must sacrifice a small amount of accuracy to ensuring counterfactually fair prediction (Fair K , Fair Add), versus the models that use unfair features: GPA, LSAT, race, sex (Full, Unaware)

### Produces Claim

The fair logistic-regression variants prioritize reduced bias over maximum accuracy, unlike Full and Unaware.

## Step 2: intermediate

- **evidence_element**: bridge_paragraph
- **evidence_type**: attribution
- **depends_on**: [1]

### Evidence Span

Logistic regression performance trade-off between accuracy and counterfactual fairness

### Produces Claim

This frames the relevant mechanism as a general debiasing trade-off, letting the fairness-first pattern from logistic regression guide the choice of analogous embedding method.

## Step 3: conclusion

- **evidence_element**: 1607.06520_figure_5
- **evidence_type**: explanation
- **depends_on**: [1, 2]

### Evidence Span

Hard-debiased embeddings suppress stereotypic analogies much more strongly, staying near 0 until roughly ~60 analogies and remaining below ~10 even at ~150 generated analogies

### Produces Claim

Hard debiasing is the embedding approach that most closely matches the fairness-first sacrifice identified earlier, yielding the lowest stereotypic-analogy counts as generation increases.
