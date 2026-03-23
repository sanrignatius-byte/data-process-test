# l3_de_1511.00830_0028

- **pair_type**: figure+formula
- **cross_doc**: False
- **hop_distance**: 3
- **reasoning_depth**: 3
- **element_ids**: ['1511.00830_formula_1', '1511.00830_figure_5']

## Query

How does conditioning VFAE generation on s lead to the identity neighborhoods seen in z1 for Extended Yale B?

## Answer

The generative process samples z from p(z) and then draws x from pθ(x|z,s), so x depends jointly on latent factors and the attribute s. The bridge identifies this as latent-variable generation with a conditional likelihood involving a nuisance/sensitive attribute, linking the formula to how VFAE structures its latent space. In the Extended Yale B z1 embedding, examples with the same person ID cluster in local neighborhoods, indicating that this representation preserves identity structure despite conditioning on s.
