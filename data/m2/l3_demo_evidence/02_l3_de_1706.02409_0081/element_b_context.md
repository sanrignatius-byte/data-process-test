# Element: 1706.02409_table_1

- **type**: table
- **doc_id**: 1706.02409
- **label**: Table 1
- **page_idx**: 0

## Caption

Table 1: Summary of datasets. Type indicates whether regression is logistic or linear; $n$ is total number of data points; $d$ is dimensionality; Minority $n$ is the number of data points in the smaller population; Protected indicates which feature is protected or fairness-sensitive.

## Content

Table 1: Summary of datasets. Type indicates whether regression is logistic or linear; $n$ is total number of data points; $d$ is dimensionality; Minority $n$ is the number of data points in the smaller population; Protected indicates which feature is protected or fairness-sensitive.

## Context Before

5We only used the data in Adult.data in our experiments.

6http://www2.law.ucla.edu/sander/Systemic/Data.htm

goal is to predict the sentence length given by the judge based on factors such as previous criminal records and the crimes for which the conviction was obtained. The protected attribute is gender.

## Context After

4.1 Accuracy-Fairness Efficient Frontiers

We begin by examining the efficient frontier of accuracy vs. fairness for the six datasets. These curves are shown in Figure 1, and are obtained by varying the weight $\lambda$ on the fairness regularizer, and for each value of $\lambda$ finding the model which minimizes the associated regularized loss function. For the logistic regression cases, we extract probabilities from the learned model w as ${ \mathrm { P r } } [ y _ { i } = 1 ] =$ $\exp ( \mathbf { w } \cdot x _ { i } ) / ( 1 + \exp ( \mathbf { w } \cdot x _ { i } ) )$ and evaluate these probabilities as predictions for the binary labels using MSE. 7 In all of the datasets, as $\lambda$ increases, the models converge to the best constant predictor, which minimizes the fairness penalties.

Perhaps the most striking aspect of Figure 1 is the great diversity of tradeoffs across different datasets and different fairness regularizers. For instance, if we examine the individual fairness regularizer, on four of the datasets (Adult, Communities and Crime, Law School and Sentencing), the curvature is relatively mild and constant — there is an approximately fixed rate at which fairness can be traded for accuracy. In contrast, on COMPAS and Default, fairness loss can be reduced almost for “free” until some small threshold value, at which point the accuracy cost increases dramatically. Similar comments can be made regarding hybrid fairness in the logistic regression cases.

The datasets

## Referring Paragraphs

1. The datasets themselves are summarized in Table 1, where we specify the size and dimensionality of each, along with the “protected” feature (race or gender) that thus defines the subgroups across which we apply our fairness criteria (see Appendix A.3 for more details). The datasets vary considerably in the number of observations, their dimensionality, and the relative size of the minority subgroup.

2. While the fairness losses 1, 2, and 3 are defined using all the $n _ { 1 } \times n _ { 2 }$ cross pairs in the dataset, in our experiments we only used $2 \times$ Minority $n$ random cross pairs where Minority $n = \operatorname* { m i n } \{ n _ { 1 } , n _ { 2 } \}$ (see Table 1). This is because: (1) using more cross pairs did not substantially improve the efficiency curves in Figure 1, (2) the CVXPY solver for binary-valued problems would become unstable when using individual fairness if we

3. The datasets themselves are summarized in Table 1, where we specify the size and dimensionality of each, along with the “protected” feature (race or gender) that thus defines the subgroups across which we apply our fairness criteria (see Appendix A.3 for more details).

4. The datasets themselves are summarized in Table 1, where we specify the size and dimensionality of each, along with the “protected” feature (race or gender) that thus defines the subgroups across which we apply our fairness criteria (see Appendix A.3 for more details). The datasets vary considerably in the number of observations, their dimensionality, and the relative size of the minority subgroup.

5. While the fairness losses 1, 2, and 3 are defined using all the $n _ { 1 } \times n _ { 2 }$ cross pairs in the dataset, in our experiments we only used $2 \times$ Minority $n$ random cross pairs where Minority $n = \operatorname* { m i n } \{ n _ { 1 } , n _ { 2 } \}$ (see Table 1). This is because: (1) using more cross pairs did not substantially improve the efficiency curves in Figure 1, (2) the CVXPY solver for binary-valued problems would become unstable when using individual fairness if we

6. The datasets themselves are summarized in Table 1, where we specify the size and dimensionality of each, along with the “protected” feature (race or gender) that thus defines the subgroups across which we apply our fairness criteria (see Appendix A.3 for more details).
