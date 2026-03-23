# Element: 1511.00830_formula_1

- **type**: formula
- **doc_id**: 1511.00830
- **label**: Formula
- **page_idx**: 0

## Caption

(无)

## Content

$$\mathbf {z} \sim p (\mathbf {z}); \qquad \mathbf {x} \sim p _ {\theta} (\mathbf {x} | \mathbf {z}, \mathbf {s})$$

## Context Before

(无)

## Context After

As for the Health dataset; this dataset is extremely imbalanced, with only $15 \%$ of the patients being admitted to a hospital. Therefore, each of the classifiers seems to predict the majority class as the label y for every point. For the invariance against s however, the results were more interesting. On the one hand, the VAE model on this dataset did maintain some sensitive information, which could be identified both linearly and non-linearly. On the other hand, VFAE and the LFR methods were able to retain less information in their latent representation, since only Random Forest was able to achieve higher than random chance accuracy. This further justifies our choice for including the MMD penalty in the lower bound of the VAE. .

In order to further assess the nature of our new representations, we visualized two dimensional Barnes-Hut SNE (van der Maaten, 2013) embeddings of the $\mathbf { z } _ { 1 }$ representations, obtained from the model trained on the Adult dataset, in Figure 4. As we can see, the nuisance/sensitive variables s can be identified both on the original representation x and on a latent representation $\mathbf { z } _ { 1 }$ that does not have the MMD penalty and the independence properties between $\mathbf { z } _ { 1 }$ and s in the prior. By

[Section: Published as a conference paper at ICLR 2016]
