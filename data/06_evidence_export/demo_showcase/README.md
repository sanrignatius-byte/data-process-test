# Delivery v1 — Demo Showcase (10 cases)

Curated for the Tuesday agent group meeting. Selected for: cross-modal reasoning depth, balanced pair_type / style / hop, LLM grounding confidence ≥ 0.85.

| # | query_id | pair_type | style | hop | source batch |
|---|----------|-----------|-------|-----|---------------|
| 1 | [l1_de_1703.06856_0064](./l1_de_1703.06856_0064.md) | formula+table | academic | 3 | old_l3_v3 |
| 2 | [l1_de_1804.06876_0202](./l1_de_1804.06876_0202.md) | figure+table | academic | 3 | old_l3_v3 |
| 3 | [l1_de_1610.02413_0122](./l1_de_1610.02413_0122.md) | figure+formula | academic | 3 | old_l3_v3 |
| 4 | [l3_de_1610.07524_0103](./l3_de_1610.07524_0103.md) | figure+formula | real_user | 3 | sweep_l3_mixed |
| 5 | [l1_de_1802.08139_0169](./l1_de_1802.08139_0169.md) | figure+formula | academic | 3 | sweep_m2_academic |
| 6 | [l1_de_1904.03310_0153](./l1_de_1904.03310_0153.md) | figure+table | academic | 3 | old_l3_v3 |
| 7 | [l1_de_1904.03035_0209](./l1_de_1904.03035_0209.md) | figure+table | academic | 3 | old_l3_v3 |
| 8 | [l3_de_1610.08452_0116](./l3_de_1610.08452_0116.md) | formula+table | real_user | 3 | sweep_l3_mixed |
| 9 | [l3_de_1707.09457_0053](./l3_de_1707.09457_0053.md) | formula+table | real_user | 3 | sweep_l3_mixed |
| 10 | [l1_de_lc_2005.07293_0179](./l1_de_lc_2005.07293_0179.md) | figure+figure | real_user | 5 | old_long_chain |

## How to read each card

Each MD file shows:
- **Query** + **Answer** (the user-facing pair)
- **Reasoning chain** (free-form trace of how the two evidence pieces compose)
- **Required evidence spans** with `evidence_type` ∈ {observation, mechanism, ...}
- **Visual anchors** — element-specific phrases pointing to the pixel-level evidence
- **Per-element panel** — caption / content / context / inline image

All 473 cases live in `../delivery_v1/`; `index.md` there is the full list.
