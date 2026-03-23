# l3_de_1602.05352_0113

- **pair_type**: figure+formula
- **cross_doc**: False
- **hop_distance**: 3
- **reasoning_depth**: 3
- **element_ids**: ['1602.05352_figure_1', '1602.05352_formula_1']

## Query

Which ratings make conventional held-out evaluation disastrously biased, through what observation mechanism, and which user and movie variables formalize that setup?

## Answer

The toy setup has horror lovers rating horror 5 and romance 1, romance lovers doing the opposite, and both groups rating dramas 3. The bridge explains that observation depends on propensity, so ratings are not missing uniformly but are selected by user–item exposure. That selection mechanism is why held-out evaluation becomes disastrously biased, and the setup is formalized with users indexed by u in {1,...,U} and movies by i in {1,...,I}.
