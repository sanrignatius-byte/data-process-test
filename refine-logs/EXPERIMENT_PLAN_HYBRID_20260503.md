# Experiment Plan — Modality Routing Ablation with Rank Fusion

**Problem**: Pure modality split (all non-text→VL or all non-text→4B) kills the non-target modalities (table/formula R@10→0). But per-modality evidence shows the "best encoder" per modality is not what was initially assumed — 4B text is surprisingly strong on figure, and VL only has a marginal edge on formula.
**Method Thesis**: Find the optimal per-modality encoder assignment through routing ablation, then merge lanes via reciprocal rank fusion. Do not assume figure/table→VL.
**Date**: 2026-05-03 (revised 2026-05-03T07:00:00Z)

---

## Per-Modality Evidence (corrected)

Actual per-modality R@10 from completed experiments:

| System | figure R@10 | table R@10 | formula R@10 | text R@10 |
|--------|------------:|-----------:|-------------:|----------:|
| `unified_4B` mixed | — | — | — | — |
| `split_4B_text` mixed | **0.5307** | **0.4985** | 0.3017 | — |
| `split_4B_text` split | **0.7128** | 0.0000 | 0.0000 | — |
| `split_VL_2B_t5` mixed | 0.4102 | 0.0236 | **0.3352** | — |
| `split_VL_2B_t5` split | 0.5390 | 0.0000 | 0.0000 | — |

Key observations:
1. **4B text beats VL on figure** (0.7128 split > 0.5390 split; 0.5307 mixed > 0.4102 mixed) — 4B gets enough signal from figure captions/context
2. **4B text destroys VL on table** (0.4985 mixed >> 0.0236 mixed) — VL image encoding of tables is currently broken
3. **VL marginally beats 4B on formula in mixed mode** (0.3352 vs 0.3017) — but both are killed in split mode
4. **Split mode kills non-target modalities for BOTH encoders** — table/formula R@10→0 in every split config

The hypothesis "figure/table→VL, formula/text→4B" is NOT supported by evidence. Instead, the question is: **can routing each modality to its individually-best encoder, with RRF merge, beat unified 4B (0.6195)?**

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|-----------------------------|---------------|
| C_R1 | Optimal per-modality routing + RRF beats split_4B_text mixed (0.4767) | routing R@10 ≥ 0.50 | B1, B2 |
| C_R2 | 4B text is the strong mainline — VL only adds marginal value on formula | figure/table→4B routing ≥ figure/table→VL routing | B2 |
| C_R3 | Split indexes are harmful for sparse modalities (table/formula killed) | mixed-index routing ≥ split-index routing | B3 |

---

## Current Baselines (M4query_v1, 473 queries, 2809 passages)

| Config | R@1 | R@5 | R@10 | R@100 | MRR | Notes |
|--------|----:|----:|-----:|------:|----:|-------|
| `unified_4B` | 0.2336 | 0.5275 | **0.6195** | 0.8636 | 0.6121 | All modalities, one dense space |
| `split_4B_text` mixed | 0.1934 | 0.3964 | **0.4767** | 0.7526 | 0.4995 | Text 4B, single mixed index |
| `split_4B_text` split | 0.1744 | 0.2801 | **0.3235** | 0.8044 | 0.4476 | Text 4B, per-modality indexes |
| `split_VL_2B_t5` mixed | 0.1205 | 0.2326 | **0.2579** | 0.4123 | 0.3217 | VL 2B, single mixed index |
| `split_VL_2B_t5` split | 0.1406 | 0.2209 | **0.2442** | 0.6723 | 0.3633 | VL 2B, per-modality indexes |

---

## Paper Storyline

**Main paper must prove:**
- Multimodal retrieval on scientific documents benefits from modality-aware routing — but 4B text is the strong baseline, not VL
- Per-modality encoder assignment is an empirical question, not an architectural assumption
- RRF provides principled cross-space merge without score calibration

**Appendix can support:**
- Per-modality encoder comparison table (which encoder wins per modality)
- RRF k-parameter sensitivity
- Why split indexes kill sparse modalities (table/formula have too few passages)

**Experiments intentionally cut:**
- Linear projection to align VL/text spaces (complex, fragile)
- Late-interaction / ColBERT-style fusion (too heavy for current infra)

---

## Experiment Blocks

### Block 1: Routing Ablation — Fixed Assignments [MUST-RUN]

- **Claim tested**: C_R1, C_R2
- **Why this block exists**: Test each plausible modality→encoder assignment. Evidence already shows 4B text is strong on figure/table; this block measures all combinations.
- **Dataset**: M4query_v1, 473 queries, 2809 passages, element-level corpus
- **Compared systems** (all use mixed index per encoder + RRF k=60):

  | Config | figure | table | formula | text | Hypothesis |
  |--------|--------|-------|---------|------|------------|
  | `r_4b_all` | 4B | 4B | 4B | 4B | Baseline: all-4B = split_4B_text mixed (0.4767) |
  | `r_vl_fig_tab` | VL | VL | 4B | 4B | Original (flawed) hypothesis |
  | `r_4b_fig_tab` | 4B | 4B | VL | 4B | **New best guess**: 4B for visual, VL only for formula |
  | `r_vl_formula_only` | 4B | 4B | VL | 4B | Minimal VL: only formula gets VL |
  | `r_vl_all_nontext` | VL | VL | VL | 4B | Pure split baseline (VL for all non-text) |

- **Metrics**: R@1, R@5, R@10, R@100, MRR; per-modality R@10
- **Setup details**:
  - Each encoder builds its own faiss index over its assigned passage subset
  - Query encoded by both encoders
  - Per-encoder top-100 merged via RRF k=60
  - VL encoder: `PYTHONPATH=/projects/myyyx1/envs/qwen3vl_tf5_overlay:$PYTHONPATH`
- **Success criterion**: Best routing R@10 ≥ 0.50 (beats split_4B_text 0.4767)
- **Failure interpretation**: If all routings < 0.48, modality splitting itself is harmful — unified 4B is the right answer
- **Table / figure target**: Main paper Table 1
- **Priority**: MUST-RUN

### Block 2: Per-Modality Encoder Comparison [MUST-RUN]

- **Claim tested**: C_R2
- **Why this block exists**: Settle "which encoder wins per modality" — evidence already shows 4B > VL on figure/table, VL > 4B on formula (mixed). Verify in RRF context.
- **Dataset**: Same as B1
- **Compared systems**: Post-hoc from B1 — compare per-modality R@10 across routing configs
- **Metrics**: Per-modality R@10 (figure/table/formula/text), #queries per modality
- **Setup details**: Filter rankings by gold element type
- **Success criterion**:
  - figure/table→4B R@10 ≥ figure/table→VL R@10 (confirm 4B advantage)
  - formula→VL R@10 ≥ formula→4B R@10 (confirm VL marginal edge)
- **Failure interpretation**: If VL beats 4B on figure (contradicting current evidence), revisit routing
- **Table / figure target**: Main paper Table 2
- **Priority**: MUST-RUN

### Block 3: Mixed vs Split Index [NICE-TO-HAVE]

- **Claim tested**: C_R3
- **Why this block exists**: Split indexes kill table/formula (R@10→0). Test whether mixed-index routing recovers them.
- **Dataset**: Same as B1
- **Compared systems**:
  1. Best routing from B1 (mixed index per encoder)
  2. Same routing, split indexes (per-modality separate indexes, merge via RRF)
- **Metrics**: Per-modality R@10 for table/formula, overall R@10
- **Setup details**: Same encoder assignment, different index strategy
- **Success criterion**: Mixed index R@10 > split index R@10, especially for table/formula
- **Failure interpretation**: If split beats mixed, current eval infra has a bug
- **Table / figure target**: Appendix
- **Priority**: NICE-TO-HAVE

### Block 4: RRF k Sensitivity [NICE-TO-HAVE]

- **Claim tested**: k parameter robustness
- **Why this block exists**: Show RRF is not sensitive to k choice
- **Dataset**: Same as B1
- **Compared systems**: Best routing from B1, k ∈ {30, 60, 100}
- **Metrics**: R@10 across k values
- **Setup details**: Same as B1, vary k only
- **Success criterion**: R@10 variation ≤ 2pp across k values
- **Failure interpretation**: If k-sensitive, document optimal k and recommend it
- **Table / figure target**: Appendix
- **Priority**: NICE-TO-HAVE

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Est. GPU-min | Risk |
|-----------|------|------|---------------|-------------|------|
| **M0** | Sanity: both encoders load, 10q RRF works | 1 dry-run | Non-random rankings | 5 min | VL overlay env |
| **M1** | B1 best-guess config first: `r_4b_fig_tab` | 1 run × 473q | R@10 > 0.45 | ~15 min | Routing logic bug |
| **M2** | B1 remaining 4 routing configs | 4 runs | Find best routing | ~60 min | Some configs may crash |
| **M3** | B2 per-modality breakdown | Post-hoc from M1-M2 | Settle per-modality encoder choice | 0 min | — |
| **M4** | Decision: if best routing R@10 ≥ 0.50, proceed | — | Go/no-go for B3/B4 | — | — |
| **M5** | B3 mixed vs split index | 1 run | Confirm mixed > split | ~15 min | — |
| **M6** | B4 k sensitivity (k=30,100) | 2 runs | k-insensitive | ~30 min | — |

**Total estimated GPU-minutes: ~125 min** (must-run: ~80 min, nice-to-have: ~45 min)

---

## Compute and Data Budget

- **Total estimated GPU-time**: ~2 hours on A6000
- **Data preparation needs**:
  - Routing logic: read `element_type` field from corpus.jsonl, route to encoder
  - RRF merge implementation (~30 lines Python)
  - Per-modality metric computation (post-hoc from rankings + qrels)
- **Human evaluation needs**: None
- **Biggest bottleneck**: VL encoder dependency chain (transformers 5 overlay) — fragile

---

## Risks and Mitigations

- **Risk**: No routing beats split_4B_text mixed (0.4767) — modality splitting is net negative
  - **Mitigation**: That's a valid negative result — document that unified 4B is optimal for this corpus size
- **Risk**: Per-modality numbers from previous experiments don't reproduce under RRF
  - **Mitigation**: B2 directly measures per-modality R@10 from RRF rankings, not relying on old numbers
- **Risk**: VL overlay environment breaks
  - **Mitigation**: M0 sanity check validates both encoders before full run

---

## Final Checklist

- [x] Main paper tables are covered (B1, B2)
- [x] Evidence drives routing choice (C_R2: 4B wins figure/table, VL marginal on formula)
- [x] Simplicity is defended (B3: mixed > split index)
- [x] Frontier contribution is justified (VL only if it adds value beyond 4B)
- [x] Nice-to-have runs separated from must-run
