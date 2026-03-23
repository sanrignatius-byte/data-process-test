# Evidence Showcase: Co-reference Bridge Paragraph Examples

> **Purpose**: 5 curated examples demonstrating how a single bridge paragraph co-references two multimodal elements, producing high-quality cross-modal queries.
> **Date**: 2026-03-23
> **Source**: `latex_cross_modal_pairs.json` (17 G2-gated co-ref pairs) → `l1_dual_evidence_queries_slurm_img150_tuned_v4_official.jsonl`

---

## Summary

| # | Paper | Type | Bridge Type | Pair Type | Tier | Quality |
|---|-------|------|-------------|-----------|------|---------|
| 1 | 1603.07025 | **Body co-ref** | Same paragraph, direct `\ref{}` | figure+table | gold | 0.97 |
| 2 | 2005.07293 | **Body co-ref** | Same paragraph, proximity 39 chars | figure+table | silver | 0.93 |
| 3 | 1809.02208 | **Caption co-ref** | Figure caption references Table | figure+table | gold | 1.00 |
| 4 | 1611.07509 | **Body co-ref** | Results paragraph, direct `\ref{}` | figure+table | gold | 0.90 |
| 5 | 1802.08139 | **Caption co-ref** | Figure caption references Equation | figure+formula | silver | 0.58 |

---

## Example 1: Simpson's Paradox in Comment Length [BODY CO-REF]

**Paper**: 1603.07025 — *"Evolution of Reddit: From the Front Page of the Internet to a Self-Referential Community?"*

### Elements

| Role | Element | Caption |
|------|---------|---------|
| **Element A** | `1603.07025_figure_6` (figure) | Figure 6: Figure (a) shows the average comment length over clock time and Figure (b) from the user-referential time. Both figures show the cohorted trends. |
| **Element B** | `1603.07025_table_1` (table) | Table 1: Evolution of the average throughout the years for each cohort. Each column is one cohort and each line is one year in time. |

### Bridge Paragraph (LaTeX source)

> **Table\~\\ref{tab:simpson}** provides some clues to what might be going on. When we move down the rows, we observe an increasing tendency in each cohort column. It means that the average comment length increases for these users. However, when we move right through the columns, people in later cohorts tend to write shorter comments...

**Co-reference**: Same sentence block simultaneously references `tab:simpson` (Table 1) and discusses the trend visible in `fig:comment_length` (Figure 6). Strategy = `direct`, char_dist = 0.

### Generated Query

> **Q**: Why does overall average comment length decline over calendar years despite each cohort showing increasing comment length as they mature?

> **A**: The overall average declines from 220 characters in 2007 to 176 in 2014 because newer cohorts start with progressively lower baseline lengths (2007 cohort starts at 220, 2014 at 176), even though within each cohort, comment length increases over time (e.g., 2007 cohort grows from 220 to 246). This Simpson's paradox occurs because the compositional shift toward newer, shorter-writing cohorts dominates the aggregate trend.

### Evidence

| # | Element | Evidence Type | Span |
|---|---------|---------------|------|
| 1 | `figure_6` | observation | *"overall comment length in Reddit over time (the darker line)"* |
| 2 | `table_1` | result | *"When we move down the rows, we observe an increasing tendency in each cohort column"* |

**Why this is a good example**: The query requires **reading the declining trend from the figure** AND **understanding the per-cohort increasing pattern from the table** to identify the Simpson's paradox. Neither element alone answers the question.

---

## Example 2: Equity vs Parity Human Preferences [BODY CO-REF]

**Paper**: 2005.07293 — *"Fairness is not Static: Deeper Understanding of Long Term Fairness via Simulation Studies"*

### Elements

| Role | Element | Caption |
|------|---------|---------|
| **Element A** | `2005.07293_figure_4` (figure) | Figure 4: Human ratings of equity and parity notions of fairness in different scenarios. |
| **Element B** | `2005.07293_table_5` (table) | Table 5: Number of people preferring solutions provided by the equity vs solutions provided by the parity notions of fairness in different scenarios. |

### Bridge Paragraph (LaTeX source)

> The statistics of ratings for each of the 4 scenarios is shown in **Figure\~\\ref{mturk\_violin}**. In addition, **Table\~\\ref{votes\_stats}** depicts the number of mechanical turk workers who preferred a certain solution following a fairness definition in each of the scenarios. Similar to findings in \cite{saxena2019...}

**Co-reference**: Two consecutive sentences in the same paragraph, `\ref{mturk_violin}` (Figure 4) and `\ref{votes_stats}` (Table 5), char_dist = 39 characters.

### Generated Query

> **Q**: How do preference counts in the first two scenarios explain the upper-range density observed in the Equity rating distribution?

> **A**: Scenario 1 (134 Equity preferences) and Scenario 2 (115 Equity preferences) contribute to the upper-range density in the Equity violin plot, with ratings extending to 4, because these scenarios show overwhelming preference for equity-based solutions compared to parity (16 and 35 respectively).

### Evidence

| # | Element | Evidence Type | Span |
|---|---------|---------------|------|
| 1 | `figure_4` | observation | *"Equity ratings extending to upper range near 4"* |
| 2 | `table_5` | result | *"Scenario 1: 134 Equity vs 16 Parity, Scenario 2: 115 Equity vs 35 Parity"* |

**Why this is a good example**: The violin plot (figure) shows **distribution shape**, the table shows **discrete preference counts**. The query forces the reader to connect count dominance (table) with density concentration (figure).

---

## Example 3: Gender Pronoun Translation Bias [CAPTION CO-REF]

**Paper**: 1809.02208 — *"Gender Bias in Machine Translation"*

### Elements

| Role | Element | Caption |
|------|---------|---------|
| **Element A** | `1809.02208_figure_2` (figure) | Figure 2: The data for the number of translated female pronouns per merged occupation category... STEM fields are nearly exclusively concentrated at X = 0. |
| **Element B** | `1809.02208_table_7` (table) | Table 7: Percentage of female, male and neutral gender pronouns obtained for each of the merged occupation category. |

### Bridge Paragraph (LaTeX source — Caption)

> \\caption{The data for the number of translated female pronouns per merged occupation category totaled among languages suggests an inverse distribution. STEM fields are nearly exclusively concentrated at $X = 0$, while more evenly distributed in fields such as production and healthcare **(See Table \\ref{tab:gender-by-category-grouped})**}

**Co-reference**: Figure 2's caption explicitly says "See Table" — the figure's own caption directs the reader to Table 7 for the numeric breakdown. Strategy = `direct`, char_dist = 0.

### Generated Query

> **Q**: What percentage of female pronouns in STEM occupations explains the near-exclusive concentration at zero translated female pronouns across languages?

> **A**: STEM occupations show only 4.219% female pronouns, which directly explains the histogram's near-exclusive concentration at X=0. This contrasts sharply with Healthcare (23.305%) and Education (23.485%), which show more distributed patterns in the figure because their higher female pronoun percentages spread across multiple histogram bins.

### Evidence

| # | Element | Evidence Type | Span |
|---|---------|---------------|------|
| 1 | `figure_2` | observation | *"STEM fields are nearly exclusively concentrated at X = 0"* |
| 2 | `table_7` | result | *"STEM Female (%) 4.219"* |

**Why this is a good example**: The histogram shows a **visual pattern** (concentration at zero); the table provides the **exact percentage** (4.219%) that explains it. The caption's "See Table" is the author's own explicit cross-reference.

---

## Example 4: Discrimination Pathway Analysis [BODY CO-REF]

**Paper**: 1611.07509 — *"Discrimination-aware Data Mining with Causal Modeling"*

### Elements

| Role | Element | Caption |
|------|---------|---------|
| **Element A** | `1611.07509_figure_3` (figure) | Figure 3: Causal network for Adult dataset: the green path represents the direct path, and the blue paths represent the indirect paths passing through marital status. |
| **Element B** | `1611.07509_table_1` (table) | Table 1: Direct/indirect discriminatory effects in the modified training data and predictions for the testing data. Values violating the discrimination criterion are marked in bold. |

### Bridge Paragraph (LaTeX source)

> The results are shown in **Table \\ref{tab:rd}**. As shown in the column "**PSE-DD**", both the modified training data and the predictions for the testing data contain no direct and indirect discrimination. In addition, **PSE-DD** produces relatively small data utility loss in term of $\chi^2$ and the corresponding **Figure \\ref{fig:dutch2}** shows the causal paths...

**Co-reference**: Results paragraph discusses Table metrics AND references the causal network Figure in the same block. Strategy = `direct`, char_dist = 0.

### Generated Query

> **Q**: Why does PSE-DD achieve lower indirect discrimination metrics than LMSG and LPS when marital status mediates the indirect pathways?

> **A**: PSE-DD achieves indirect discrimination of 0.049 in training and 0.041–0.042 in prediction, whereas LMSG shows 0.288 and LPS shows 0.174. This is because PSE-DD targets both the direct path (green) and the indirect paths (blue) passing through marital status, while LMSG and LPS focus primarily on direct discrimination removal.

### Evidence

| # | Element | Evidence Type | Span |
|---|---------|---------------|------|
| 1 | `figure_3` | observation | *"blue paths represent the indirect paths passing through marital status"* |
| 2 | `table_1` | result | *"Indirect discrimination values for PSE-DD, LMSG, and LPS in Train row"* |

**Why this is a good example**: The figure defines the **causal pathway structure** (direct green vs indirect blue); the table shows the **numeric discrimination metrics**. The query requires mapping "blue path = marital status mediation" (figure) to "indirect discrimination = 0.049 vs 0.288" (table).

---

## Example 5: Causal Graph + Error Distribution Equation [CAPTION CO-REF]

**Paper**: 1802.08139 — *"Fairness through Causal Awareness: Path-Specific Counterfactual Fairness"*

### Elements

| Role | Element | Caption |
|------|---------|---------|
| **Element A** | `1802.08139_figure_3` (figure) | Figure 3: (a) Empirical distribution of ε_m for the case in which m is generated by Eq. (3) with an extra non-linear term f(A,C). (b) Modification of the GCM. |
| **Element B** | `1802.08139_formula_3` (formula) | Equation (3): The linear data-generation model for mediator M. |

### Bridge Paragraph (LaTeX source — Caption)

> \\caption{(a): Empirical distribution of $\\epsilon^n\_m$ for the case in which $m^n$ is generated by **\\eqref{eq:lm}** with an extra non-linear term $f(A,C)$ (continuous lines). Histograms of $\\tilde p(H\_m|A)$ (crossed lines), see (b). (b): Modification of the GCM corresponding to **\\eqref{eq:lm}**.}

**Co-reference**: Figure 3's caption references Equation (3) twice — once for the data-generation process, once for the GCM correspondence. Strategy = `direct`, char_dist = 0.

### Generated Query

> **Q**: Why does introducing the latent parent node between the sensitive attribute and mediator variable reveal distributional mismatch in the error term?

> **A**: The latent variable H_m in the modified causal graph explicitly captures hidden confounding between A and M that the original data-generation process assumes is absorbed into ε_m. When the true process includes a non-linear term f(A,C) not represented in the learned model, the error distribution deviates from the assumed form because the latent structure exposes the mismatch.

### Evidence

| # | Element | Evidence Type | Span |
|---|---------|---------------|------|
| 1 | `figure_3` | observation | *"latent variable H_m as parent node of M in the modified causal graph"* |
| 2 | `formula_3` | mechanism | *"error term epsilon_m in the data-generation process for M"* |

**Why this is a good example**: This is a **figure+formula** pair (the only non figure+table example). The figure shows the **graphical model structure**; the equation defines the **algebraic relationship**. The query requires connecting visual graph topology to mathematical error terms.

---

## Key Takeaways for Presentation

1. **Bridge paragraphs are author-created reasoning links**: When a paragraph co-references two elements, the author intended them to be read together. This is stronger evidence than proximity alone.

2. **Two bridge types**:
   - **Body co-ref** (Examples 1, 2, 4): A results/discussion paragraph simultaneously cites Figure X and Table Y — *"The results are shown in Table \ref{...}... the corresponding Figure \ref{...} shows..."*
   - **Caption co-ref** (Examples 3, 5): A figure's own caption says "See Table \ref{...}" or "corresponding to \eqref{...}" — the author explicitly linked the two elements.

3. **All 5 queries are genuinely dual-evidence**: Removing either element makes the query unanswerable. The figure provides visual/structural insight; the table/formula provides quantitative/formal precision.

4. **Detection is zero-cost**: These co-references are extracted from LaTeX `\ref{}` patterns — no LLM calls needed.
