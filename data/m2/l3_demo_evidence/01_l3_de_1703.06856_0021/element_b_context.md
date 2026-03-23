# Element: 1607.06520_figure_5

- **type**: figure
- **doc_id**: 1607.06520
- **label**: 
- **page_idx**: 0

## Caption

To reduce the bias in an embedding, we change the embeddings of gender neutral words, by removing

## Content

To reduce the bias in an embedding, we change the embeddings of gender neutral words, by removing

## Context Before

Direct Bias. First we used the same analogy generation task as before: for both the hard-debiased and the soft-debiased embeddings, we automatically generated pairs of words that are analogous to she-he and asked crowd-workers to evaluate whether these pairs reflect gender stereotypes. Figure 8 shows the results. On the initial w2vNEWS embedding, 19% of the top 150 analogies were judged as showing gender stereotypes by a majority of the ten workers. After applying our hard debiasing algorithm, only 6% of the new embedding were judged as stereotypical. As an example, consider the analogy puzzle, he to doctor is as she to $X$ . The original embedding returns $X = n u r s e$ while the hard-debiased embedding finds X = physician. Moreover the hard-debiasing algorithm preserved gender appropriate analogies such as she to ovarian cancer is as he to prostate cancer. This demonstrates that the hard-debiasing has effectively reduced the gender stereotypes in the word embedding. Figure 8 also shows that the number of appropriate analogies remains similar as in the original embedding after executing hard-debiasing. This demonstrates that that the quality of the embeddings is preserved. The details results are in Appendix G. Soft-debiasing was less effective in removing gender bias.

To further confirms the quality of embeddings after debiasing, we tested the debiased embedding on several standard benchmarks that measure whether related words have similar embeddings as well as how well t

## Context After

(无)
