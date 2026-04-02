"""Prompt templates for cross-modal dual-evidence query generation.

Extracted from generate_multihop_l1_queries.py to enable reuse by other
scripts and modules.
"""
from __future__ import annotations
from typing import Dict

SYSTEM_PROMPT = (
    "You are a data annotator creating cross-modal retrieval training data "
    "for multimodal academic documents. "
    "Output valid JSON only, no other text, no markdown fences."
)

# ──────────────────────────────────────────────────────────────
# Prompt templates per modality combination
# ──────────────────────────────────────────────────────────────

PROMPT_FIGURE_TABLE_1HOP = """You are a PhD student presenting experimental results at a group meeting. Ask direct, empirically grounded questions the way scientists actually discuss data — specific and precise, not convoluted.

Generate 2 cross-modal retrieval queries requiring BOTH a figure and a table from the same document.

## Figure ({fig_id})
Caption: {fig_caption}
Context: {fig_context}
[Figure image is provided above]

## Table ({tbl_id})
Caption: {tbl_caption}
Column/row headers: {tbl_headers}
Context: {tbl_context}
[Table image is provided above; read specific values from the table image]

## Textual connection
{edge_context}
{latex_bridge}

## YOUR TASK
Create queries that REQUIRE BOTH elements. The query should create an INFORMATION GAP:
describe context from one element, ask a question answerable only by the other.

## STRICT RULES
1. Query MUST be UNANSWERABLE if either the figure or the table is removed.
2. NEVER start with Do/Does/Did/Is/Are/Can/Has/Will/Would; NO yes/no questions.
3. NEVER start with "Given that" or "What causes" (avoid template collapse).
4. NEVER put specific numbers, percentages, or exact values in the query.
5. NEVER use meta-language: "figure", "table", "the text", "according to", "as shown in".
6. Max 30 words per query. Answer max 3 sentences with specific values from BOTH elements.
7. OBSERVATION INJECTION (MANDATORY): Each query MUST describe in natural language what you concretely observe in one element (a trend, drop, gap, plateau, cluster, contrast). Use phrases like "the score stays flat while...", "the curve drops sharply above...", "X outperforms Y on minority but not aggregate". FORBIDDEN: meta-language ("figure"/"table"/"as shown in") and verbatim anchor coords. The visual_anchors field contains orientation coordinates for evaluation only — paraphrase the observation, never copy the anchor string.
8. The two queries must use DIFFERENT aspects of the data.
9. ENTITY AMNESTY: you MUST use exact paper terminology (method names, metric names,
   dataset names, variable names like "F1 score" or "p-value") when needed.
   Do NOT replace concrete terms with vague descriptors.
10. CAUSAL TOPOLOGY: each query must test one explicit relationship:
    (a) table value explains a visual trend, or (b) visual anomaly is supported/refuted by
    table values. Do NOT stitch unrelated facts.
11. Avoid weak templates: "Which component..." "How does X relate to Y..."
12. Answer must include a relationship connector: because / due to / consistent with /
    constrained by / compared with / whereas / despite / under.

## STYLE DIVERSITY — MANDATORY
- The 2 queries MUST use different opening bigrams (first two words).
- At least one query MUST NOT start with "Why" or "Under what".
- LENGTH MIX (STRUCTURAL, not just word count): queries[0] = SHORT compressed causal question (8-14 words, e.g. "The X drops after Y — does Z explain this?"); queries[1] = LONG I-notice + why/how + context (18-30 words, e.g. "X stays flat while Y rises across all conditions — given the constraint in the formula, what mechanism prevents X from scaling?"). Count words BEFORE finalizing.
- DO NOT use template shells: "Under what condition does..." or "Why is A different from B...".
- Do NOT create parallel dual asks using "..., and which ...". Each query should ask one causal/comparative question.
- Use natural research wording; explicit terms like "F1 score", "p-value", and "regularization strength" are allowed.

## BAD vs GOOD examples
BAD: "Did the red line peak at 90,000 and match the keyword set?" — yes/no, visual coords
BAD: "Which configuration best validates the theoretical optimum?" — banned verb "validates"
BAD: "How does the high-recall setting relate to the fairness tradeoff?" — banned "relate", vague
BAD: "How does fixing shared background nodes satisfy the identity requirement?" — abstract, no observation

GOOD (I-NOTICE + WHY, short): "The streaming volume drops after mid-December — does the sampling cap explain this?"
GOOD (I-NOTICE + WHY, long): "The minority-group gain stays flat above 1k samples while majority-group accuracy keeps rising — does the regularization term bound this asymmetry?"
GOOD (COUNTERINTUITIVE CONTRAST): "Why does direct discrimination approach zero while indirect pathway scores stay elevated despite the same fairness objective?"
GOOD (HOW-DISCREPANCY): "How does the drop in session frequency after later positions correspond to the retrieval-window setting that yields the highest scores?"
GOOD (WHY-INCONSISTENT): "Why is the ensemble gain larger on the minority subgroup than on the aggregate benchmark under the same evaluation regime?"

## Output format (JSON only):
CRITICAL: Generate EXACTLY 2 queries. queries[0] MUST be SHORT (8-14 words). queries[1] MUST be LONG (18-30 words). Count words carefully before writing.
{{
  "queries": [
    {{
      "query_length_bucket": "short",
      "reasoning_chain": "Max 3 sentences: (1) concrete figure observation, (2) concrete table metric/value pattern, (3) causal link that requires both",
      "query": "SHORT question (8-14 words ONLY), based on reasoning_chain, NO specific values",
      "answer": "factual answer citing specific values from BOTH elements, max 3 sentences, with connector",
      "query_type": "causal_explanation|discrepancy_analysis|hypothesis_verification",
      "required_evidence_spans": [
        {{"element_id": "{fig_id}", "span": "short extractive phrase from figure caption/content (semantic concept)", "evidence_type": "observation"}},
        {{"element_id": "{tbl_id}", "span": "short extractive phrase from table headers/content (semantic concept)", "evidence_type": "result"}}
      ],
      "visual_anchors": [
        {{"element_id": "{fig_id}", "anchor": "physical coords only: color/position/curve (evaluation only — NOT in query)"}},
        {{"element_id": "{tbl_id}", "anchor": "physical coords only: row/column/cell (evaluation only — NOT in query)"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }},
    {{
      "query_length_bucket": "long",
      "reasoning_chain": "Max 3 sentences: (1) concrete figure observation, (2) concrete table metric/value pattern, (3) causal link that requires both",
      "query": "LONG question (18-30 words), based on reasoning_chain, NO specific values",
      "answer": "factual answer citing specific values from BOTH elements, max 3 sentences, with connector",
      "query_type": "causal_explanation|discrepancy_analysis|hypothesis_verification",
      "required_evidence_spans": [
        {{"element_id": "{fig_id}", "span": "short extractive phrase from figure caption/content (semantic concept)", "evidence_type": "observation"}},
        {{"element_id": "{tbl_id}", "span": "short extractive phrase from table headers/content (semantic concept)", "evidence_type": "result"}}
      ],
      "visual_anchors": [
        {{"element_id": "{fig_id}", "anchor": "physical coords only: color/position/curve (evaluation only — NOT in query)"}},
        {{"element_id": "{tbl_id}", "anchor": "physical coords only: row/column/cell (evaluation only — NOT in query)"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }}
  ]
}}"""

PROMPT_FIGURE_TABLE_2HOP = """You are a PhD student presenting experimental results at a group meeting. Ask direct, empirically grounded questions the way scientists actually discuss data — specific and precise, not convoluted.

Generate 2 cross-modal retrieval queries requiring BOTH a figure and a table with CHAIN REASONING.

## Figure ({fig_id})
Caption: {fig_caption}
Context: {fig_context}
[Figure image is provided above]

## Table ({tbl_id})
Caption: {tbl_caption}
Column/row headers: {tbl_headers}
Context: {tbl_context}
[Table image is provided above; read specific values from the table image]

## Connection chain (2-hop)
{edge_context}
Intermediate element(s): {intermediate_info}
{latex_bridge}

## YOUR TASK
Create queries that REQUIRE BOTH elements with CHAIN REASONING.
The intermediate element is the bridge — use it as a cognitive stepping stone.

## STRICT RULES
1. Query MUST be UNANSWERABLE if either endpoint or bridge is removed.
2. NEVER start with Do/Does/Did/Is/Are/Can/Has/Will/Would; NO yes/no questions.
3. NEVER start with "Given that" or "What causes" (avoid template collapse).
4. NEVER put specific numbers or exact values in the query.
5. NEVER use meta-language: "figure", "table", "the text", "according to", "as shown in".
6. Max 30 words per query; answer max 3 sentences with values from BOTH endpoints.
7. OBSERVATION INJECTION (MANDATORY): Each query MUST describe in natural language what you concretely observe in one element (a trend, drop, gap, plateau, cluster, contrast). Use phrases like "the score stays flat while...", "the curve drops sharply above...", "X outperforms Y on minority but not aggregate". FORBIDDEN: meta-language ("figure"/"table"/"as shown in") and verbatim anchor coords. The visual_anchors field contains orientation coordinates for evaluation only — paraphrase the observation, never copy the anchor string.
8. ENTITY AMNESTY: you MUST use exact paper terminology when needed.
9. CAUSAL TOPOLOGY: the query must require a chain:
   figure observation -> bridge mechanism -> table metric (or the reverse), not parallel lookup.
10. Avoid weak templates: "Which component..." "How does X relate to Y..."
11. Answer must include a relationship connector: because / due to / consistent with /
    constrained by / compared with / whereas / despite / under.

## STYLE DIVERSITY — MANDATORY
- The 2 queries MUST use different opening bigrams (first two words).
- At least one query MUST NOT start with "Why" or "Under what".
- LENGTH MIX (STRUCTURAL, not just word count): queries[0] = SHORT compressed causal question (8-14 words, e.g. "The X drops after Y — does Z explain this?"); queries[1] = LONG I-notice + why/how + context (18-30 words, e.g. "X stays flat while Y rises across all conditions — given the constraint in the formula, what mechanism prevents X from scaling?"). Count words BEFORE finalizing.
- DO NOT use template shells: "Under what condition does..." or "Why is A different from B...".
- Do NOT create parallel dual asks using "..., and which ...". Each query should ask one causal/comparative question.
- Use natural research wording; explicit terms like "F1 score", "p-value", and "regularization strength" are allowed.

## Output format (JSON only):
CRITICAL: Generate EXACTLY 2 queries. queries[0] MUST be SHORT (8-14 words). queries[1] MUST be LONG (18-30 words). Count words carefully before writing.
{{
  "queries": [
    {{
      "query_length_bucket": "short",
      "reasoning_chain": "Max 3 sentences: endpoint A observation -> bridge property -> endpoint B metric/conclusion",
      "query": "SHORT chain-reasoning question (8-14 words ONLY), NO specific values",
      "answer": "factual answer citing values from both elements, max 3 sentences, with connector",
      "query_type": "causal_explanation|discrepancy_analysis|hypothesis_verification",
      "required_evidence_spans": [
        {{"element_id": "{fig_id}", "span": "short extractive phrase from figure (semantic concept)", "evidence_type": "observation"}},
        {{"element_id": "{tbl_id}", "span": "short extractive phrase from table (semantic concept)", "evidence_type": "result"}}
      ],
      "bridge": {{
        "element_id": "{intermediate_info}",
        "anchor": "key property of the bridge element used in the chain",
        "evidence_span": "extractive phrase connecting bridge to both endpoints"
      }},
      "visual_anchors": [
        {{"element_id": "{fig_id}", "anchor": "physical coords only — evaluation only, NOT in query"}},
        {{"element_id": "{tbl_id}", "anchor": "physical coords only — evaluation only, NOT in query"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }},
    {{
      "query_length_bucket": "long",
      "reasoning_chain": "Max 3 sentences: endpoint A observation -> bridge property -> endpoint B metric/conclusion",
      "query": "LONG chain-reasoning question (18-30 words), NO specific values",
      "answer": "factual answer citing values from both elements, max 3 sentences, with connector",
      "query_type": "causal_explanation|discrepancy_analysis|hypothesis_verification",
      "required_evidence_spans": [
        {{"element_id": "{fig_id}", "span": "short extractive phrase from figure (semantic concept)", "evidence_type": "observation"}},
        {{"element_id": "{tbl_id}", "span": "short extractive phrase from table (semantic concept)", "evidence_type": "result"}}
      ],
      "bridge": {{
        "element_id": "{intermediate_info}",
        "anchor": "key property of the bridge element used in the chain",
        "evidence_span": "extractive phrase connecting bridge to both endpoints"
      }},
      "visual_anchors": [
        {{"element_id": "{fig_id}", "anchor": "physical coords only — evaluation only, NOT in query"}},
        {{"element_id": "{tbl_id}", "anchor": "physical coords only — evaluation only, NOT in query"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }}
  ]
}}"""

PROMPT_FIGURE_FORMULA = """You are a PhD student presenting experimental results at a group meeting. Ask questions that connect what you see in experiment plots or model diagrams to what the math predicts or requires.

Generate 2 cross-modal retrieval queries connecting a figure with a mathematical formula.

## Figure ({fig_id})
Caption: {fig_caption}
Context: {fig_context}
[Figure image is provided above]

## Formula ({formula_id})
Key variables: {formula_variables}
Context: {formula_context}

## Textual connection
{edge_context}
{latex_bridge}

## CORE PRINCIPLE: DUAL-EVIDENCE INFORMATION GAP

Your query must create an information gap that ONLY closes when evidence from BOTH the figure AND the formula is combined. Think of it as two puzzle halves:

**Half A (Figure):** Something OBSERVABLE in the figure — a trend, a structural choice, a comparison, a spatial arrangement, a quantitative pattern. Identify a SPECIFIC, CONCRETE observation.
**Half B (Formula):** Something DERIVABLE from the formula — a mathematical constraint, an optimization objective, a theoretical bound, a functional relationship. Identify a SPECIFIC mathematical mechanism.

The query asks WHY or HOW Half A is explained / constrained / justified by Half B. The answer MUST cite concrete details from BOTH halves.

## FIGURE TYPE STRATEGY (CRITICAL)

**If the figure shows QUANTITATIVE results** (curves, bar charts, heatmaps, scatter plots):
- Half A = a specific trend, peak, crossing point, gap, plateau, or comparison between conditions.

**If the figure shows STRUCTURAL/ARCHITECTURAL diagrams** (network topology, causal graphs, pipeline flowcharts, encoder-decoder):
- Half A = a specific STRUCTURAL CHOICE: number of branches, where a loss is applied, which components share weights, direction of information flow, separation of pathways.
- Name the structural choice concretely (e.g., "two separate encoder branches feeding a bottleneck", "the adversarial path after the feature extractor").
- Ask: how does this specific structural choice satisfy / enforce / follow from the mathematical constraint in the formula?

{architecture_guidance}

## ENTITY AMNESTY
Use exact paper terminology (method names, metric names, variable names) when needed.
Avoid vague substitutions like "the best-performing method" when a concrete name exists.

## STRICT RULES
1. Query MUST be UNANSWERABLE without BOTH the figure AND the formula. Test: if you remove the figure, can you still answer from the formula alone? If yes — REJECT and rewrite.
2. NEVER start with Do/Does/Did/Is/Are/Can/Has/Will/Would; NO yes/no questions.
3. NEVER start with "Given that" or "What causes" (avoid template collapse).
4. NEVER copy raw LaTeX strings or raw variable symbols into the query.
5. You MAY use at most ONE anchor value in the query only if needed to preserve directionality.
6. NEVER use meta-language: "equation", "formula", "figure", "as shown in", "diagram", "architecture".
7. Max 30 words per query.
8. OBSERVATION INJECTION (MANDATORY): Each query MUST describe in natural language what you concretely observe in one element (a trend, drop, gap, plateau, cluster, contrast). Use phrases like "the score stays flat while...", "the curve drops sharply above...", "X outperforms Y on minority but not aggregate". FORBIDDEN: meta-language ("figure"/"table"/"as shown in") and verbatim anchor coords. The visual_anchors field contains orientation coordinates for evaluation only — paraphrase the observation, never copy the anchor string.
9. CAUSAL TOPOLOGY: query must connect one concrete visual phenomenon and one concrete
   mathematical mechanism; no unrelated stitching.
10. Avoid weak templates: "Which component..." "How does X relate to Y..." "What role does..."
11. Answer must include a relationship connector (because / due to / consistent with /
    constrained by / compared with / whereas / despite / under).
12. IRON RULE (symbolic grounding): if Element B is mathematical, the answer MUST explicitly quote at least one specific variable/function/constraint term from Element B (e.g., lambda, theta, f(A,C)) and explain how it maps to the observed visual topology in Element A. Avoid generic phrases like "the mathematical structure".

## SELF-CHECK before outputting each query

Ask yourself:
1. Does the answer cite a SPECIFIC, CONCRETE observation from the figure (trend/structure/comparison)?
2. Does the answer cite a SPECIFIC mathematical mechanism from the formula (constraint/bound/objective)?
3. If I removed either piece of evidence, would the answer become incomplete or wrong?

If any answer is NO — rewrite.

## STYLE DIVERSITY — MANDATORY
- The 2 queries MUST use different opening bigrams (first two words).
- At least one query MUST NOT start with "Why" or "Under what".
- LENGTH MIX (STRUCTURAL, not just word count): queries[0] = SHORT compressed causal question (8-14 words, e.g. "The X drops after Y — does Z explain this?"); queries[1] = LONG I-notice + why/how + context (18-30 words, e.g. "X stays flat while Y rises across all conditions — given the constraint in the formula, what mechanism prevents X from scaling?"). Count words BEFORE finalizing.
- DO NOT use template shells: "Under what condition does..." or "Why is A different from B...".
- Do NOT create parallel dual asks using "..., and which ...". Each query should ask one causal/comparative question.
- Use natural research wording; explicit terms like "F1 score", "p-value", and "regularization strength" are allowed.

## BAD vs GOOD examples

BAD: "How does the two-branch encoder instantiate the disentanglement penalty?" — banned "instantiate"
BAD: "Why does the adversarial discriminator enforce the mutual exclusivity constraint?" — banned "enforce", banned "constraint"
BAD: "What role does the regularization term play in the convergence pattern?" — vague, no operator

GOOD (HOW-DISCREPANCY, quantitative figure):
"How does the plateau in selection rates correspond to the sharper utility drop in the minority group under the same threshold region?"
→ Figure half: minority vs majority utility curves diverge at a specific selection rate.
→ Formula half: the fairness constraint bounds minority selection independently of the aggregate rate.

GOOD (WHAT-CONSTRAINT, architectural figure):
"What constraint in the penalty term limits correlation between the two encoder-path representations?"
→ Figure half: two parallel encoder branches with no shared weights (specific structural choice).
→ Formula half: the penalty minimizes mutual information between branch outputs.

## Output format (JSON only):
CRITICAL: Generate EXACTLY 2 queries. queries[0] MUST be SHORT (8-14 words). queries[1] MUST be LONG (18-30 words). Count words carefully before writing.
{{
  "queries": [
    {{
      "query_length_bucket": "short",
      "reasoning_chain": "Max 3 sentences: concrete figure observation -> concrete formula mechanism -> explicit causality between them",
      "query": "SHORT question (8-14 words ONLY), based on reasoning_chain, NO LaTeX/values/letters/meta-language",
      "answer": "max 4 sentences with connector. Must cite both halves explicitly.",
      "answer_figure_evidence": "1-2 sentences: what specific observation from the figure is cited (trend/structure/comparison). Start with the concrete observable.",
      "answer_formula_evidence": "1-2 sentences: what specific mathematical mechanism from the formula is cited (constraint/bound/objective). Start with the mechanism.",
      "query_type": "causal_explanation|discrepancy_analysis|hypothesis_verification",
      "required_evidence_spans": [
        {{"element_id": "{fig_id}", "span": "specific observable feature: name the trend/structure/comparison concretely", "evidence_type": "observation"}},
        {{"element_id": "{formula_id}", "span": "specific mathematical mechanism: name the constraint/term/property (NOT generic 'the formula')", "evidence_type": "mechanism"}}
      ],
      "visual_anchors": [
        {{"element_id": "{fig_id}", "anchor": "physical coords only: color/position/curve/branch/edge — NOT in query"}},
        {{"element_id": "{formula_id}", "anchor": "specific term or variable — NOT standalone letter in query"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }},
    {{
      "query_length_bucket": "long",
      "reasoning_chain": "Max 3 sentences: concrete figure observation -> concrete formula mechanism -> explicit causality between them",
      "query": "LONG question (18-30 words), based on reasoning_chain, NO LaTeX/values/letters/meta-language",
      "answer": "max 4 sentences with connector. Must cite both halves explicitly.",
      "answer_figure_evidence": "1-2 sentences: what specific observation from the figure is cited (trend/structure/comparison). Start with the concrete observable.",
      "answer_formula_evidence": "1-2 sentences: what specific mathematical mechanism from the formula is cited (constraint/bound/objective). Start with the mechanism.",
      "query_type": "causal_explanation|discrepancy_analysis|hypothesis_verification",
      "required_evidence_spans": [
        {{"element_id": "{fig_id}", "span": "specific observable feature: name the trend/structure/comparison concretely", "evidence_type": "observation"}},
        {{"element_id": "{formula_id}", "span": "specific mathematical mechanism: name the constraint/term/property (NOT generic 'the formula')", "evidence_type": "mechanism"}}
      ],
      "visual_anchors": [
        {{"element_id": "{fig_id}", "anchor": "physical coords only: color/position/curve/branch/edge — NOT in query"}},
        {{"element_id": "{formula_id}", "anchor": "specific term or variable — NOT standalone letter in query"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }}
  ]
}}"""

PROMPT_FORMULA_TABLE = """You are a PhD student presenting experimental results at a group meeting. Ask questions that connect what the theory (formula) predicts to what the numbers (table) actually show.

Generate 2 cross-modal retrieval queries connecting a formula with a table from the same document.

## Formula ({formula_id})
Key variables: {formula_variables}
Context: {formula_context}

## Table ({tbl_id})
Caption: {tbl_caption}
Column/row headers: {tbl_headers}
Context: {tbl_context}
[Table image is provided above; read specific values from the table image]

## Textual connection
{edge_context}
{latex_bridge}

## YOUR TASK
Create queries that connect mathematical relationships with tabular evidence.
The query must require BOTH the formula's theoretical structure AND the table's data.

## STRICT RULES
1. Query MUST be UNANSWERABLE without both the formula and the table.
2. NEVER start with Do/Does/Did/Is/Are/Can/Has/Will/Would; NO yes/no questions.
3. NEVER start with "Given that" or "What causes" (avoid template collapse).
4. NEVER copy raw LaTeX strings or raw variable symbols into the query.
5. You MAY use at most ONE anchor value in the query only if needed to preserve directionality.
6. NEVER use meta-language: "table", "equation", "formula", "the text".
7. Max 30 words per query; answer max 3 sentences with specific values from the table.
8. OBSERVATION INJECTION (MANDATORY): Each query MUST describe in natural language what you concretely observe in one element (a trend, drop, gap, plateau, cluster, contrast). Use phrases like "the score stays flat while...", "the curve drops sharply above...", "X outperforms Y on minority but not aggregate". FORBIDDEN: meta-language ("figure"/"table"/"as shown in") and verbatim anchor coords. The visual_anchors field contains orientation coordinates for evaluation only — paraphrase the observation, never copy the anchor string.
9. ENTITY AMNESTY: you MUST use exact paper terminology (method/metric/variable names)
   when needed; avoid vague substitutions.
10. CAUSAL TOPOLOGY: question must connect one concrete mathematical mechanism and one
    concrete tabular result through a causal/comparative claim.
11. Avoid weak templates: "Which component..." "How does X relate to Y..."
12. Answer must include a relationship connector: because / due to / consistent with /
    constrained by / compared with / whereas / despite / under.

## STYLE DIVERSITY — MANDATORY
- The 2 queries MUST use different opening bigrams (first two words).
- At least one query MUST NOT start with "Why" or "Under what".
- LENGTH MIX (STRUCTURAL, not just word count): queries[0] = SHORT compressed causal question (8-14 words, e.g. "The X drops after Y — does Z explain this?"); queries[1] = LONG I-notice + why/how + context (18-30 words, e.g. "X stays flat while Y rises across all conditions — given the constraint in the formula, what mechanism prevents X from scaling?"). Count words BEFORE finalizing.
- DO NOT use template shells: "Under what condition does..." or "Why is A different from B...".
- Do NOT create parallel dual asks using "..., and which ...". Each query should ask one causal/comparative question.
- Use natural research wording; explicit terms like "F1 score", "p-value", and "regularization strength" are allowed.

## BAD vs GOOD examples
BAD: "Does beta=0.3 in row 4 satisfy Eq. 2?" — yes/no, numbers, meta-language
BAD: "What does N represent?" — standalone letter, vague
BAD: "Which configuration best instantiates the theoretical threshold?" — banned "instantiate"

GOOD (HOW-DISCREPANCY): "How does the smoothing-driven sparsity regime correspond to the steeper accuracy drop under low-resource conditions?"
GOOD (WHY-INCONSISTENT): "Why is the highest-regularization setting better than the unregularized baseline in out-of-domain evaluation despite overfitting risk?"

## Output format (JSON only):
CRITICAL: Generate EXACTLY 2 queries. queries[0] MUST be SHORT (8-14 words). queries[1] MUST be LONG (18-30 words). Count words carefully before writing.
{{
  "queries": [
    {{
      "query_length_bucket": "short",
      "reasoning_chain": "Max 3 sentences: formula mechanism -> table metric pattern -> explicit causality/comparison",
      "query": "SHORT formula-data question (8-14 words ONLY), based on reasoning_chain, NO LaTeX/values/letters",
      "answer": "factual answer with specific values from both, max 3 sentences, with connector",
      "query_type": "causal_explanation|discrepancy_analysis|hypothesis_verification",
      "required_evidence_spans": [
        {{"element_id": "{formula_id}", "span": "short extractive phrase describing the formula's role or constraint (NOT raw LaTeX)", "evidence_type": "constraint"}},
        {{"element_id": "{tbl_id}", "span": "short extractive phrase from table headers/content (semantic concept)", "evidence_type": "result"}}
      ],
      "visual_anchors": [
        {{"element_id": "{formula_id}", "anchor": "specific term or variable — NOT standalone letter in query"}},
        {{"element_id": "{tbl_id}", "anchor": "physical coords only: row/column/cell — NOT in query"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }},
    {{
      "query_length_bucket": "long",
      "reasoning_chain": "Max 3 sentences: formula mechanism -> table metric pattern -> explicit causality/comparison",
      "query": "LONG formula-data question (18-30 words), based on reasoning_chain, NO LaTeX/values/letters",
      "answer": "factual answer with specific values from both, max 3 sentences, with connector",
      "query_type": "causal_explanation|discrepancy_analysis|hypothesis_verification",
      "required_evidence_spans": [
        {{"element_id": "{formula_id}", "span": "short extractive phrase describing the formula's role or constraint (NOT raw LaTeX)", "evidence_type": "constraint"}},
        {{"element_id": "{tbl_id}", "span": "short extractive phrase from table headers/content (semantic concept)", "evidence_type": "result"}}
      ],
      "visual_anchors": [
        {{"element_id": "{formula_id}", "anchor": "specific term or variable — NOT standalone letter in query"}},
        {{"element_id": "{tbl_id}", "anchor": "physical coords only: row/column/cell — NOT in query"}}
      ],
      "text_evidence": "direct quote from context, min 40 chars"
    }}
  ]
}}"""


# ──────────────────────────────────────────────────────────────
# Level 3: 3-step reasoning chain prompt (M4 Schema 1)
# ──────────────────────────────────────────────────────────────

PROMPT_3STEP_REASONING_CHAIN = """You are building a benchmark that tests whether a system can follow a multi-step reasoning chain across different evidence sources. The chain MUST have exactly 3 steps, each grounded in a different piece of evidence.

## Graph-Grounded Evidence Path

The document graph connects these 3 nodes via explicit \\ref{{}} links in the LaTeX source. The bridge paragraph is the ACTUAL text the authors wrote to connect the two endpoint elements.

### Node 1 (Premise): {elem_a_type} ({elem_a_id})
Caption: {elem_a_caption}
Context: {elem_a_context}
{elem_a_image_note}

### Node 2 (Bridge): paragraph — author's own connecting text
{bridge_text}
Bridge quality: {bridge_quality_label}

### Node 3 (Conclusion): {elem_b_type} ({elem_b_id})
Caption: {elem_b_caption}
Context: {elem_b_context}
{elem_b_image_note}

### Graph path: {graph_path_description}

## YOUR TASK

Generate 1 query that REQUIRES all 3 evidence nodes IN SEQUENCE. The bridge paragraph is the key — it explains WHY Node 1's observation leads to Node 3's conclusion.

### SERIAL CHAIN PATTERN (REQUIRED)
Node 1 observation → Bridge paragraph mechanism/explanation → Node 3 confirmation/outcome

Example of VALID serial chain:
  "Node 1 shows accuracy drops for minority groups"
  → Bridge: "As shown in [fig:3], the accuracy drops significantly for minority groups. The exact numbers are in [tab:2]."
  → "Node 3 provides per-group accuracy values confirming the 12% gap"
  Query: "What numerical precision confirms the minority group accuracy drop visible in the performance trend?"

### PARALLEL IS FORBIDDEN
"Node 1 says A, Node 3 says B, therefore A+B" — this is just two independent lookups. The bridge MUST provide a causal/explanatory link, not just co-occurrence.

### STEP-DELETION TEST (self-check before outputting)
For each step, ask: "If I remove THIS step's evidence, can I still derive the answer?"
- If YES for any step → your chain is too weak, rewrite it
- If NO for all 3 steps → your chain is valid

### BRIDGE GROUNDING RULE
Your answer MUST quote or paraphrase specific content from the bridge paragraph text above. If the bridge text says "X leads to Y", your answer must use that causal link. Do NOT invent connections not present in the bridge.

## STRICT RULES
1. Query MUST require ALL 3 evidence nodes. Removing ANY node makes the answer underivable.
2. NEVER start with Do/Does/Did/Is/Are/Can/Has/Will/Would; NO yes/no questions.
3. NEVER use meta-language: "figure", "table", "equation", "the text", "according to", "as shown in".
4. Max 30 words for the query. Answer max 4 sentences.
5. ENTITY AMNESTY: use exact paper terminology (method names, metrics, variables).
6. Each reasoning_step MUST have a DIFFERENT evidence_type from: observation, attribution, explanation, verification, prediction.
7. The role arc MUST follow: premise → intermediate → conclusion.
8. Visual anchors MUST specify physical location: row/column for tables, axis region/color/marker for figures, specific variable/term for formulas. Generic anchors like "the table" or "the figure" will be rejected.

## Output format (JSON only):
{{
  "queries": [
    {{
      "query": "A question requiring serial 3-step reasoning (max 30 words)",
      "answer": "Answer citing specific evidence from all 3 nodes, referencing bridge text explicitly, max 4 sentences",
      "query_type": "causal_chain|mechanism_trace|conditional_prediction",
      "reasoning_steps": [
        {{
          "step_id": 1,
          "evidence_element_id": "{elem_a_id}",
          "evidence_type": "observation",
          "evidence_span": "extractive phrase from Node 1",
          "reasoning_role": "premise",
          "depends_on_steps": [],
          "produces_claim": "What this step establishes (1 sentence)"
        }},
        {{
          "step_id": 2,
          "evidence_element_id": "bridge_paragraph",
          "evidence_type": "attribution",
          "evidence_span": "extractive phrase copied from the bridge paragraph text above",
          "reasoning_role": "intermediate",
          "depends_on_steps": [1],
          "produces_claim": "How the bridge connects step 1's observation to a mechanism (1 sentence)"
        }},
        {{
          "step_id": 3,
          "evidence_element_id": "{elem_b_id}",
          "evidence_type": "explanation",
          "evidence_span": "extractive phrase from Node 3",
          "reasoning_role": "conclusion",
          "depends_on_steps": [1, 2],
          "produces_claim": "What final conclusion requires both prior steps (1 sentence)"
        }}
      ],
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "extractive phrase from Node 1", "evidence_type": "observation"}},
        {{"element_id": "bridge_paragraph", "span": "extractive phrase from bridge paragraph", "evidence_type": "attribution", "content": "verbatim or close-paraphrase from bridge text above, min 40 chars"}},
        {{"element_id": "{elem_b_id}", "span": "extractive phrase from Node 3", "evidence_type": "explanation"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "specific physical location: row X col Y / axis region / marker color / variable name"}},
        {{"element_id": "{elem_b_id}", "anchor": "specific physical location: row X col Y / axis region / marker color / variable name"}}
      ],
      "text_evidence": "direct quote from bridge paragraph context, min 40 chars"
    }}
  ]
}}"""


# ──────────────────────────────────────────────────────────────
# Real-user query style prompt templates (B1)
# ──────────────────────────────────────────────────────────────
# These templates generate naturally phrased queries as a curious
# reader / practitioner would ask, rather than the structured
# PhD dual-evidence academic style.  Activated via --query-style
# real_user or mixed.  Both elements are still required to answer.

PROMPT_REAL_USER_FACTUAL = """A researcher is skimming an AI/ML paper and wants quick factual clarity.
Generate 2 natural lookup queries that require BOTH the figure and the table/formula below.

## Element A ({elem_a_id}, {elem_a_type})
Caption: {elem_a_caption}
Context: {elem_a_context}

## Element B ({elem_b_id}, {elem_b_type})
Caption: {elem_b_caption}
Context: {elem_b_context}

## Connection
{edge_context}
{latex_bridge}

## YOUR TASK
Write 2 questions a researcher would genuinely ask while reading this paper.
Style: factual_lookup — "What is X?", "What value does Y take when Z?", "What does X show about Y?"

## RULES
1. Both elements must be needed to fully answer the question.
2. Questions must be in natural English — no academic formalism.
3. NEVER use meta-language: "figure", "table", "equation", "as shown in".
4. Max 25 words per question. Answer max 2 sentences.
5. Do NOT copy raw LaTeX, variable names as standalone letters, or numeric values into the question.
6. Each question must have a different opening word.

## Output (JSON only):
{{
  "queries": [
    {{
      "query": "natural factual question requiring both elements (max 25 words)",
      "answer": "direct factual answer (max 2 sentences, cite values from both elements)",
      "query_type": "factual_lookup",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "key concept or value from element A", "evidence_type": "data"}},
        {{"element_id": "{elem_b_id}", "span": "key concept or value from element B", "evidence_type": "data"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "specific region or term in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "specific region or term in element B"}}
      ],
      "text_evidence": "direct quote from context supporting the answer (min 30 chars)"
    }},
    {{
      "query": "second natural factual question, different opening word (max 25 words)",
      "answer": "direct factual answer (max 2 sentences)",
      "query_type": "factual_lookup",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "key concept from element A", "evidence_type": "data"}},
        {{"element_id": "{elem_b_id}", "span": "key concept from element B", "evidence_type": "data"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "specific region or term in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "specific region or term in element B"}}
      ],
      "text_evidence": "direct quote from context (min 30 chars)"
    }}
  ]
}}"""

PROMPT_REAL_USER_SUMMARY = """A reader wants a concise synthesis of what two pieces of evidence mean together.
Generate 2 natural summary questions requiring BOTH elements below.

## Element A ({elem_a_id}, {elem_a_type})
Caption: {elem_a_caption}
Context: {elem_a_context}

## Element B ({elem_b_id}, {elem_b_type})
Caption: {elem_b_caption}
Context: {elem_b_context}

## Connection
{edge_context}
{latex_bridge}

## YOUR TASK
Write 2 questions asking for a combined summary of both elements.
Style: summary — "Summarize how...", "What does the combination of X and Y suggest?", "What can we conclude from X and Y together?"

## RULES
1. Both elements must be needed — a summary of only one element is wrong.
2. Natural English. No academic formalism.
3. NEVER use meta-language: "figure", "table", "equation", "as shown in".
4. Max 25 words per question. Answer max 3 sentences.
5. Each question must have a different opening word.
6. Do NOT use yes/no questions.

## Output (JSON only):
{{
  "queries": [
    {{
      "query": "natural summary question requiring both elements (max 25 words)",
      "answer": "synthesized answer covering both elements (max 3 sentences)",
      "query_type": "summary",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "key insight from element A", "evidence_type": "observation"}},
        {{"element_id": "{elem_b_id}", "span": "key insight from element B", "evidence_type": "observation"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "main visual or conceptual anchor in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "main visual or conceptual anchor in element B"}}
      ],
      "text_evidence": "relevant context quote supporting the synthesis (min 30 chars)"
    }},
    {{
      "query": "second summary question, different opening word (max 25 words)",
      "answer": "synthesized answer (max 3 sentences)",
      "query_type": "summary",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "key insight from element A", "evidence_type": "observation"}},
        {{"element_id": "{elem_b_id}", "span": "key insight from element B", "evidence_type": "observation"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "main anchor in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "main anchor in element B"}}
      ],
      "text_evidence": "context quote (min 30 chars)"
    }}
  ]
}}"""

PROMPT_REAL_USER_COMPARISON = """A reader wants to understand how two results or components compare.
Generate 2 natural comparison questions requiring BOTH elements below.

## Element A ({elem_a_id}, {elem_a_type})
Caption: {elem_a_caption}
Context: {elem_a_context}

## Element B ({elem_b_id}, {elem_b_type})
Caption: {elem_b_caption}
Context: {elem_b_context}

## Connection
{edge_context}
{latex_bridge}

## YOUR TASK
Write 2 questions comparing something across the two elements.
Style: comparison — "How does X in [element A] compare to Y in [element B]?", "Which is better at..., and why?", "What differences exist between X and Y?"

## RULES
1. The comparison must be grounded in both elements — not inferable from one alone.
2. Natural English, conversational tone.
3. NEVER use meta-language: "figure", "table", "equation".
4. Max 25 words per question. Answer max 3 sentences.
5. Do NOT copy raw LaTeX or standalone variable letters.
6. Each question must have a different opening word.
7. Do NOT use yes/no questions.

## Output (JSON only):
{{
  "queries": [
    {{
      "query": "natural comparison question requiring both elements (max 25 words)",
      "answer": "comparative answer with specific values from both (max 3 sentences)",
      "query_type": "comparison",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "compared aspect from element A", "evidence_type": "observation"}},
        {{"element_id": "{elem_b_id}", "span": "compared aspect from element B", "evidence_type": "observation"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "compared feature in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "compared feature in element B"}}
      ],
      "text_evidence": "context quote supporting the comparison (min 30 chars)"
    }},
    {{
      "query": "second comparison question, different opening word (max 25 words)",
      "answer": "comparative answer (max 3 sentences)",
      "query_type": "comparison",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "compared aspect from element A", "evidence_type": "observation"}},
        {{"element_id": "{elem_b_id}", "span": "compared aspect from element B", "evidence_type": "observation"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "compared feature in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "compared feature in element B"}}
      ],
      "text_evidence": "context quote (min 30 chars)"
    }}
  ]
}}"""

PROMPT_REAL_USER_HOW_WORKS = """A practitioner wants to understand the mechanism connecting two results.
Generate 2 natural how/why questions requiring BOTH elements below.

## Element A ({elem_a_id}, {elem_a_type})
Caption: {elem_a_caption}
Context: {elem_a_context}

## Element B ({elem_b_id}, {elem_b_type})
Caption: {elem_b_caption}
Context: {elem_b_context}

## Connection
{edge_context}
{latex_bridge}

## YOUR TASK
Write 2 mechanistic questions about how/why the two pieces of evidence are related.
Style: how_works — "How does X lead to Y?", "Why does X affect Y?", "What mechanism connects X and Y?"

## RULES
1. Both elements must be needed to explain the mechanism.
2. Natural English — not overly academic.
3. NEVER use meta-language: "figure", "table", "equation", "as shown in".
4. Max 25 words per question. Answer max 3 sentences.
5. Do NOT copy raw LaTeX or standalone variable letters.
6. Each question must have a different opening word.
7. Do NOT use yes/no questions.

## Output (JSON only):
{{
  "queries": [
    {{
      "query": "natural how/why question connecting both elements (max 25 words)",
      "answer": "mechanistic answer drawing on both elements (max 3 sentences)",
      "query_type": "how_works",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "mechanism aspect from element A", "evidence_type": "mechanism"}},
        {{"element_id": "{elem_b_id}", "span": "mechanism aspect from element B", "evidence_type": "mechanism"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "key mechanism feature in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "key mechanism feature in element B"}}
      ],
      "text_evidence": "context quote supporting the mechanism (min 30 chars)"
    }},
    {{
      "query": "second how/why question, different opening word (max 25 words)",
      "answer": "mechanistic answer (max 3 sentences)",
      "query_type": "how_works",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "mechanism aspect from element A", "evidence_type": "mechanism"}},
        {{"element_id": "{elem_b_id}", "span": "mechanism aspect from element B", "evidence_type": "mechanism"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "key feature in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "key feature in element B"}}
      ],
      "text_evidence": "context quote (min 30 chars)"
    }}
  ]
}}"""

PROMPT_REAL_USER_WHAT_IF = """A researcher wants to reason about hypothetical changes to a method or result.
Generate 2 natural what-if questions requiring BOTH elements below.

## Element A ({elem_a_id}, {elem_a_type})
Caption: {elem_a_caption}
Context: {elem_a_context}

## Element B ({elem_b_id}, {elem_b_type})
Caption: {elem_b_caption}
Context: {elem_b_context}

## Connection
{edge_context}
{latex_bridge}

## YOUR TASK
Write 2 hypothetical / counterfactual questions that require both elements to reason about.
Style: what_if — "What would happen to X if Y changed?", "If X were different, how would Y be affected?", "What would change in [result] if [condition] were altered?"

## RULES
1. Both elements must be needed to reason about the hypothetical.
2. Natural English — conversational speculation.
3. NEVER use meta-language: "figure", "table", "equation".
4. Max 25 words per question. Answer max 3 sentences.
5. Do NOT copy raw LaTeX or standalone variable letters.
6. Each question must have a different opening word.
7. Do NOT use yes/no questions.

## Output (JSON only):
{{
  "queries": [
    {{
      "query": "natural what-if question connecting both elements (max 25 words)",
      "answer": "reasoned counterfactual answer drawing on both elements (max 3 sentences)",
      "query_type": "what_if",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "condition or variable from element A", "evidence_type": "condition"}},
        {{"element_id": "{elem_b_id}", "span": "affected result from element B", "evidence_type": "result"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "hypothetical feature in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "affected feature in element B"}}
      ],
      "text_evidence": "context quote supporting the reasoning (min 30 chars)"
    }},
    {{
      "query": "second what-if question, different opening word (max 25 words)",
      "answer": "reasoned counterfactual answer (max 3 sentences)",
      "query_type": "what_if",
      "required_evidence_spans": [
        {{"element_id": "{elem_a_id}", "span": "condition from element A", "evidence_type": "condition"}},
        {{"element_id": "{elem_b_id}", "span": "affected result from element B", "evidence_type": "result"}}
      ],
      "visual_anchors": [
        {{"element_id": "{elem_a_id}", "anchor": "feature in element A"}},
        {{"element_id": "{elem_b_id}", "anchor": "feature in element B"}}
      ],
      "text_evidence": "context quote (min 30 chars)"
    }}
  ]
}}"""

# Mapping used by select_template() for real_user style
REAL_USER_TEMPLATES: Dict[str, str] = {
    "factual_lookup": PROMPT_REAL_USER_FACTUAL,
    "summary": PROMPT_REAL_USER_SUMMARY,
    "comparison": PROMPT_REAL_USER_COMPARISON,
    "how_works": PROMPT_REAL_USER_HOW_WORKS,
    "what_if": PROMPT_REAL_USER_WHAT_IF,
}
REAL_USER_STYLE_CYCLE = list(REAL_USER_TEMPLATES.keys())
