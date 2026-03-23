# Element: 1409.0575_figure_7

- **type**: figure
- **doc_id**: 1409.0575
- **label**: 
- **page_idx**: 0

## Caption

Image classification Ground truth

## Content

Image classification Ground truth

## Context Before

We elaborate further on these and other more minor challenges with large-scale evaluation. Appendix F describes the submission protocol and other details of running the competition itself.

4.1 Image classification

The scale of ILSVRC classification task (1000 categories and more than a million of images) makes it very expensive to label every instance of every object in every image. Therefore, on this dataset only one object category is labeled in each image. This creates ambiguity in evaluation. For example, an image might be labeled as a “strawberry” but contain both a strawberry and an apple. Then an algorithm would not know which one of the two objects to name. For the image classification task we allowed an algorithm to identify multiple (up to 5) objects in an image and not be penalized as long as one of the objects indeed corresponded to the ground truth label. Figure 7(top row) shows some examples.

Here $d ( b _ { i j } , B _ { i k } )$ is the error of localization, defined as 0 if the area of intersection of boxes $b _ { i j }$ and $B _ { i k }$ divided by the areas of their union is greater than 0.5, and 1 otherwise. (Everingham et al., 2010) The error of an algorithm is computed as in Eq. 1.

Evaluating localization is inherently difficult in some images. Consider a picture of a bunch of bananas or a carton of apples. It is easy to classify these images as containing bananas or apples, and even possible to localize a few instances of each fruit. However, in orde

## Context After

(无)
