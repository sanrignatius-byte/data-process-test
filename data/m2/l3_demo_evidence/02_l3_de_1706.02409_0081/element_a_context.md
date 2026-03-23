# Element: 1610.07524_figure_2

- **type**: figure
- **doc_id**: 1610.07524
- **label**: Figure 2
- **page_idx**: 0

## Caption

Figure 2: False positive rates across prior record count for defendants charged with a Misdemeanor offense. Plot is based on assessing a defendant as “high-risk” if their COMPAS decile score is $> 4$ . Error bars represent 95% confidence intervals. Figure 3: COMPAS decile score histograms for Black and White defendants. Cohen’s $d = 0 . 6 0$ , non-overlap $d _ { \mathrm { T V } } ( f _ { b } , f _ { w } ) = 2 4 . 5 \%$ .

## Content

Figure 2: False positive rates across prior record count for defendants charged with a Misdemeanor offense. Plot is based on assessing a defendant as “high-risk” if their COMPAS decile score is $> 4$ . Error bars represent 95% confidence intervals. Figure 3: COMPAS decile score histograms for Black and White defendants. Cohen’s $d = 0 . 6 0$ , non-overlap $d _ { \mathrm { T V } } ( f _ { b } , f _ { w } ) = 2 4 . 5 \%$ .

## Context Before

(无)

## Context After

is equivalent to the total variation distance. Letting $f _ { r , y } ( s )$ denote the score distribution for race $r$ and recidivism outcome $y$ , one can establish the following sharp bound on $\Delta$ .

Proposition 3.2 (Percent overlap bound). Under the MinMax policy,

$$ \Delta \leq (t _ {H} - t _ {L}) d _ {\mathrm {T V}} (f _ {b, y}, f _ {w, y}). $$

One might expect that differences in false positive rates are largely attributable to the subset of defendants who are charged with more serious offenses and who have a larger number of prior arrests/convictions. While it is true that the false positive rates within both racial groups are higher for defendants with worse criminal histories, considerable between-group differences in these error rates persist across low prior count subgroups. Figure 2 shows a plot of false positive rates across di

Our analysis indicates that there are risk assessment use cases in which it is desirable to balance error rates across different groups, even though this will generally result in risk assessments that are not free from predictive bias. However, balancing error rates overall may not be sufficient, as this does not guarantee balance at finer levels of granularity. That is, even if $\mathrm { F P R } _ { b } = \mathrm { F P R } _ { w }$ , we may still see differences in error rates within prior rec

## Referring Paragraphs

1. One might expect that differences in false positive rates are largely attributable to the subset of defendants who are charged with more serious offenses and who have a larger number of prior arrests/convictions. While it is true that the false positive rates within both racial groups are higher for defendants with worse criminal histories, considerable between-group differences in these error rates persist across low prior count subgroups. Figure 2 shows a plot of false positive rates across di

2. Figure 2 shows a plot of false positive rates across different ranges of prior count for defendants charged with a misdemeanor offense, which is the lowest severity criminal offense category.

3. Figure 2: False positive rates across prior record count for defendants charged with a Misdemeanor offense.

4. That is, even if $\mathrm { F P R } _ { b } = \mathrm { F P R } _ { w }$ , we may still see differences in error rates within prior record score categories (see e.g., Figure 2).
