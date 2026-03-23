# Reasoning Chain

## Step 1: premise

- **evidence_element**: 1805.05859_formula_2
- **evidence_type**: observation
- **depends_on**: []

### Evidence Span

Y as a linear combination of two inputs, A and Z, plus an exogenous disturbance term

### Produces Claim

The outcome Y includes both modeled inputs and an unwanted exogenous component that can be interpreted as residual noise affecting the outcome.

## Step 2: intermediate

- **evidence_element**: bridge_paragraph
- **evidence_type**: attribution
- **depends_on**: [1]

### Evidence Span

structural equation, linear model, outcome, coefficients, latent noise

### Produces Claim

This connects the residual-noise interpretation from the structural equation to the debiasing scenario by framing stereotypic analogies as an outcome influenced by controllable inputs and latent noise.

## Step 3: conclusion

- **evidence_element**: 1607.06520_figure_5
- **evidence_type**: explanation
- **depends_on**: [1, 2]

### Evidence Span

Hard-debiased embeddings suppress stereotypic analogies much more strongly, staying near 0 until roughly ~60 analogies and remaining below ~10 even at ~150 generated analogies

### Produces Claim

Given the outcome-plus-noise framing, hard debiasing is the method that most strongly reduces the unwanted stereotypic outcome relative to baseline and soft debiasing.
