# Delivery v1 — Demo Showcase (10 cases, intra-doc only)

Curated for the Tuesday agent group meeting.

**Strict intra-doc constraint**: every evidence element shares the same `doc_id`.
Selection criteria: hop ≥ 3 preferred, high LLM-grounding confidence, m4 step-deletion proxy, balanced across `pair_type` and `query_style`.

| # | query_id | pair_type | style | hop | source batch |
|---|----------|-----------|-------|-----|---------------|
| 1 | [l3_de_1610.07524_0103](./l3_de_1610.07524_0103.md) | figure+formula | real_user | 3 | sweep_l3_mixed |
| 2 | [l1_de_1707.09457_0002](./l1_de_1707.09457_0002.md) | figure+formula | academic | 3 | old_l3_v3 |
| 3 | [l1_de_1802.08139_0169](./l1_de_1802.08139_0169.md) | figure+formula | academic | 3 | sweep_m2_academic |
| 4 | [l1_de_1904.03035_0209](./l1_de_1904.03035_0209.md) | figure+table | academic | 3 | old_l3_v3 |
| 5 | [l1_de_1901.10436_0217](./l1_de_1901.10436_0217.md) | figure+table | academic | 3 | old_l3_v3 |
| 6 | [l1_de_1706.02409_0270](./l1_de_1706.02409_0270.md) | figure+table | academic | 3 | old_l3_v3 |
| 7 | [l3_de_1610.08452_0116](./l3_de_1610.08452_0116.md) | formula+table | real_user | 3 | sweep_l3_mixed |
| 8 | [l3_de_1511.00830_0000](./l3_de_1511.00830_0000.md) | formula+table | academic | 3 | sweep_l3_academic_persona |
| 9 | [l3_de_1607.06520_0089](./l3_de_1607.06520_0089.md) | formula+table | real_user | 3 | sweep_l3_mixed |
| 10 | [l1_de_lc_1805.03677_0041](./l1_de_lc_1805.03677_0041.md) | figure+figure | real_user | 5 | old_long_chain |

## How to read each card

- **Query** + **Answer** (the user-facing pair)
- **Reasoning chain** (free-form trace)
- **Required evidence spans** with `evidence_type`
- **Visual anchors** — element-specific phrases
- **Per-element panel** — caption / content / context / image

Full 473-case index lives at `../delivery_v1/index.md`.
