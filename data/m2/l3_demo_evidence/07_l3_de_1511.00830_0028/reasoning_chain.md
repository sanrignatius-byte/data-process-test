# Reasoning Chain

## Step 1: premise

- **evidence_element**: 1511.00830_formula_1
- **evidence_type**: observation
- **depends_on**: []

### Evidence Span

draw a latent representation z from a chosen prior, then generate an observed sample x from a model distribution conditioned on both z and an additional variable s

### Produces Claim

The model generates x from z while explicitly conditioning on s, establishing the relevant generative dependency.

## Step 2: intermediate

- **evidence_element**: bridge_paragraph
- **evidence_type**: attribution
- **depends_on**: [1]

### Evidence Span

latent variable, prior, conditional likelihood, generator, nuisance/sensitive attribute

### Produces Claim

This frames the dependency from step 1 as VFAE's generator using conditional likelihood with s as a nuisance/sensitive attribute, connecting the formula to the representation mechanism.

## Step 3: conclusion

- **evidence_element**: 1511.00830_figure_5
- **evidence_type**: explanation
- **depends_on**: [1, 2]

### Evidence Span

Examples with the same ID tend to appear in local neighborhoods, forming loose clusters that reflect identity structure in the latent representation

### Produces Claim

Given the conditional generative mechanism in VFAE, the z1 map shows the resulting latent space preserves person identity through local clustering.
