# Reasoning Chain

## Step 1: premise

- **evidence_element**: 1703.06856_table_1
- **evidence_type**: observation
- **depends_on**: []

### Evidence Span

we must sacrifice a small amount of accuracy to ensuring counterfactually fair prediction (Fair K, Fair Add)

### Produces Claim

Fair K achieves counterfactual fairness at the cost of slightly worse RMSE than unfair models.

## Step 2: intermediate

- **evidence_element**: bridge_paragraph
- **evidence_type**: attribution
- **depends_on**: [1]

### Evidence Span

Domain adaptation can also be cast as learning representations

### Produces Claim

The observed accuracy cost is attributed to learning a fair representation rather than using unfair predictive features directly.

## Step 3: conclusion

- **evidence_element**: 1511.00830_formula_2
- **evidence_type**: explanation
- **depends_on**: [1, 2]

### Evidence Span

returns 1 or 0, while p(z_k = 1 | x_i, s = 1) returns values between values 0 and 1, then the penalty could still be satisfied, but information could still leak through

### Produces Claim

A fair representation must stop leakage in both binary decisions and continuous latent probabilities, explaining why Fair K accepts some RMSE loss.
