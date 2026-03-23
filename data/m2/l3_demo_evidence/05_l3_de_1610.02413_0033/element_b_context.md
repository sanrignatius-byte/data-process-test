# Element: 1511.00830_formula_2

- **type**: formula
- **doc_id**: 1511.00830
- **label**: Formula
- **page_idx**: 0

## Caption

(无)

## Content

$$\mathrm {P A D} (\epsilon) = 2 (1 - 2 \epsilon)$$

## Context Before

returns 1 or 0, while $p ( z _ { k } = 1 | x _ { i } , s = 1 )$ returns values between values 0 and 1, then the penalty could still be satisfied, but information could still leak through. We addressed both of these issues in this paper.

Domain adaptation can also be cast as learning representations that are “invariant” with respect to a discrete variable s, the domain. Most similar to our work are neural network approaches which try to match the feature distributions between the domains. This was performed in an unsupervised way with mSDA (Chen et al., 2012) by training denoising autoencoders jointly on all domains, thus implicitly obtaining a representation general enough to explain both the domain and the data. This is in contrast to our approach where we instead try to learn representations that explicitly remove domain information during the learning process. For the latter we find more similarities with “domain-regularized” supervised approaches that simultaneously try to predict the label for a data point and remove domain specific information. This is done with either MMD (Long & Wang, 2015; Tzeng et al., 2014) or adversarial (Ganin et al., 2015) penalties at the hidden layers of the network. In our model however the main “domain-regularizer” stems from the independence properties of the prior over the domain and latent representations. We also employ MMD on our model but from a different perspective since we consider a slightly more difficult case where the domain s 

## Context After

(无)
