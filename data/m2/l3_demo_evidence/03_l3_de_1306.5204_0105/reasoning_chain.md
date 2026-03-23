# Reasoning Chain

## Step 1: premise

- **evidence_element**: 1306.5204_figure_14
- **evidence_type**: observation
- **depends_on**: []

### Evidence Span

Median. S = 0.018 \hat{\mu} = 0.013, \hat{\sigma} = 0.001, z = 5.000

### Produces Claim

The observed median Jensen-Shannon divergence is significantly above the null distribution, indicating a meaningful single-day discrepancy.

## Step 2: intermediate

- **evidence_element**: bridge_paragraph
- **evidence_type**: attribution
- **depends_on**: [1]

### Evidence Span

identifying ~50% key-players correctly for a single day is reasonable, and accuracy can be increased by using longer observation periods. Even more, the Potential Reach metrics are quite stable for some days in the aggregated data.

### Produces Claim

The single-day discrepancy is attributed to limited daily accuracy, so longer observation periods favor stable measures such as Potential Reach.

## Step 3: conclusion

- **evidence_element**: 1306.5204_table_2
- **evidence_type**: verification
- **depends_on**: [1, 2]

### Evidence Span

Potential Reach 100 59.2 (32–83) 80

### Produces Claim

Among the centrality measures, Potential Reach is validated as the best choice for longer observation periods because its aggregated accuracy reaches 80.
