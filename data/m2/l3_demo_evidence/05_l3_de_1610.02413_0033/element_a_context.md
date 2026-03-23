# Element: 1610.02413_figure_5

- **type**: figure
- **doc_id**: 1610.02413
- **label**: Figure 5
- **page_idx**: 0

## Caption

Figure 5: Graphical model for Scenario II.

## Content

Figure 5: Graphical model for Scenario II.

## Context Before

Scenario I Consider the dependency structure depicted in Figure 4. Here, $X _ { 1 }$ is a feature highly (even deterministically) correlated with the protected attribute $A$ , but independent of the target Y given A. For example, $X _ { 1 }$ might be “languages spoken at home” or “great great grandfather’s profession”. The target Y has a statistical correlation with the protected attribute. There’s a second real-valued feature $X _ { 2 }$ correlated with Y , but only related to A through Y . For

Scenario I Consider the dependency structure depicted in Figure 4.

the feature $X _ { 2 }$ through the score $\widetilde { R } = X _ { 2 } ^ { { \mathbf { \Upsilon } } }$ . The score $\widetilde { R }$ satisfies equalized odds, since $X _ { 2 }$ and $A$ are independent conditional on Y . Because of the statistical correlation between $A$ and $Y$ , a better statistical predictor, with greater power, can be obtained by taking into account also the protected attribute $A$ , or perhaps its surrogate $X _ { 1 }$ . The statistically optimal predictor would have the form $R ^ { * } = r _ { I } ^ { * } ( X _ { 2 } , X _ { 1 } )$ , biasing the score according to the protected attribute $A$ . The score $R ^ { * }$ does not satisfy equalized odds, and in a sense seems to be “profiling” based on $A$ .

Scenario II Now consider the dependency structure depicted in Figure 5. Here $X _ { 3 }$ is a feature, e.g. “wealth” or “annual income”, correlated with the protected attribute $A$ and directly p

## Context After

6.1 Unidentifiability

The above two scenarios seem rather different. The optimal score $R ^ { * }$ is in one case based directly on $A$ or its surrogate, and in another only on a directly predictive feature, but this is not apparent by considering the equalized odds criterion, suggesting a possible shortcoming of equalized odds. In fact, as we will now see, the two scenarios are indistinguishable using any oblivious test. That is, no test based only on the target labels, the protected attribute and the score would give different indications for the optimal score $R ^ { * }$ in the two scenarios. If it were judged unfair in one scenario, it would also be judged unfair in the other.

We will show this by constructing specific instantiations of the two scenarios where the joint distributions over $( Y , \overbrace { A } , R ^ { * } , \widetilde { R } )$ are identical. The scenarios are thus unidentifiable based only on these joint distributions.

Scenario II Now consider the dependency structure depicted in Figure 5. Here $X _ { 3 }$ is a feature, e.g. “wealth” or “annual income”, correlated with the protected attribute $A$ and directly predictive of the target Y . That is, in this model, the probability of paying back of a loan is just a function of an individual’s wealth, independent of their race. Using $X _ { 3 }$ on its own as a predictor, e.g. using the score $R ^ { * } = X _ { 3 }$ , does not naturally seem directly discriminatory

Scenario II Now consider the dependency

## Referring Paragraphs

1. might capture an applicant’s driving record if applying for insurance, financial activity if applying for a loan, or criminal history in criminal justice situations. An intuitively “fair” predictor here is to use only

the feature $X _ { 2 }$ through the score $\widetilde { R } = X _ { 2 } ^ { { \mathbf { \Upsilon } } }$ . The score $\widetilde { R }$ satisfies equalized odds, since $X _ { 2 }$ and $A$ are independent conditional on Y . Because of the statistical correlation between $A$ and $Y$ 

2. the feature $X _ { 2 }$ through the score $\widetilde { R } = X _ { 2 } ^ { { \mathbf { \Upsilon } } }$ . The score $\widetilde { R }$ satisfies equalized odds, since $X _ { 2 }$ and $A$ are independent conditional on Y . Because of the statistical correlation between $A$ and $Y$ , a better statistical predictor, with greater power, can be obtained by taking into account also the protected attribute $A$ , or perhaps its surrogate $X _ { 1 }$ . The statistically optimal predictor would have the f

3. Scenario II Now consider the dependency structure depicted in Figure 5. Here $X _ { 3 }$ is a feature, e.g. “wealth” or “annual income”, correlated with the protected attribute $A$ and directly predictive of the target Y . That is, in this model, the probability of paying back of a loan is just a function of an individual’s wealth, independent of their race. Using $X _ { 3 }$ on its own as a predictor, e.g. using the score $R ^ { * } = X _ { 3 }$ , does not naturally seem directly discriminatory

4. Scenario II Now consider the dependency structure depicted in Figure 5.

5. Figure 5: Graphical model for Scenario II.
