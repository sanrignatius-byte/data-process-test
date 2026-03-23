# Reasoning Chain

## Step 1: premise

- **evidence_element**: 1602.05352_figure_1
- **evidence_type**: observation
- **depends_on**: []

### Evidence Span

horror lovers” who rate all horror movies 5 and all romance movies 1... “romance lovers” ... opposite way... both groups rate dramas as 3

### Produces Claim

The ratings are systematically structured by user preference group and movie genre rather than being homogeneous across entries.

## Step 2: intermediate

- **evidence_element**: bridge_paragraph
- **evidence_type**: attribution
- **depends_on**: [1]

### Evidence Span

the observation indicator depends on propensity, creating selection bias in which ratings are observed

### Produces Claim

Because observation is governed by propensity over the structured ratings from step 1, the held-out set becomes selectively sampled instead of representative.

## Step 3: conclusion

- **evidence_element**: 1602.05352_formula_1
- **evidence_type**: explanation
- **depends_on**: [1, 2]

### Evidence Span

Denote with u ∈ {1, . . . , U} the users and with i ∈ {1, . . . , I} the movies

### Produces Claim

The biased evaluation setup induced by selective observation is formalized over user index u and movie index i, completing the toy-example mechanism.
