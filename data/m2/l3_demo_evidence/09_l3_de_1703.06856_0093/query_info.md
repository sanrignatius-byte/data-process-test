# l3_de_1703.06856_0093

- **pair_type**: formula+table
- **cross_doc**: True
- **hop_distance**: 3
- **reasoning_depth**: 3
- **element_ids**: ['1703.06856_table_1', '1511.00830_formula_2']

## Query

Which fairness mechanism explains why Fair K sacrifices RMSE yet still prevents leakage when outputs are binary but latent probabilities remain continuous?

## Answer

Full has lower RMSE than Fair K, so counterfactually fair prediction sacrifices a small amount of accuracy. The bridge attributes this tradeoff to enforcing fairness through a representation or prediction mechanism that blocks sensitive information. Formula 2’s context explains why preventing leakage must address both binary outputs and continuous latent probabilities, so Fair K’s sacrifice supports a leakage-resistant fair representation.
