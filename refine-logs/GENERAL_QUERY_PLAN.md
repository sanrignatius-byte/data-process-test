# General Query Plan

**Date**: 2026-04-18
**Context**: This plan condenses the repeated discussion threads around `general` / universal queries, answer generation, and evaluation so they do not stay scattered across meeting notes.

## Why This Exists

The project has repeatedly converged on the same point:

- dual-evidence and graph-grounded queries are good for retrieval training and graph validation
- but we also need a `general` query track that looks like what real readers or agents ask
- this track should not be forced into the same hard multihop QC regime as graph-synthesis queries

This requirement appears in:

- `4.16.md`: short natural questions, more real-user input, answer design still open
- `CLAUDE.md`: C-Pool / universal query library, evidence-localization-first QC
- `DISCUSSION_LOG.md`: 50-100 high-frequency universal queries and lightweight evaluation

## Proposed Split

Treat query work as two separate tracks:

### Track A: Graph-synthesis queries

- Source:
  - intra-doc or long-chain candidate pairs from the graph
- Goal:
  - train retrieval and validate graph value
- Query type:
  - dual-evidence, multihop, graph-grounded
- QC:
  - existing `qc_multihop_query()` or long-chain QC
- Main metric:
  - evidence localization and retrieval uplift

### Track B: General queries

- Source:
  - human-written seeds
  - C-Pool templates
  - LLM paraphrases based on a curated reader-question pool
- Goal:
  - simulate what a real reader or agent asks about any academic paper
- Query type:
  - summary, motivation, method, contribution, setup, comparison, figure intent, limitation, citation relation
- QC:
  - do **not** hard-fail on multihop structure
  - only verify answerability and evidence localization
- Main metric:
  - can retrieval find the supporting evidence
  - can answer generation stay grounded

## Proposed General Query Taxonomy

- `paper_summary`: What is the paper about?
- `core_contribution`: What is the main contribution?
- `problem_motivation`: Why is this problem important?
- `method_overview`: How does the method work at a high level?
- `experiment_setup`: What datasets or settings are used?
- `result_comparison`: Which baseline or variant performs better?
- `figure_intent`: What does Figure X show?
- `table_lookup`: Which method or condition has the best score?
- `limitation_or_failure`: What limitation or failure mode is discussed?
- `cross_paper_relation`: How does this connect to another paper or cited work?

## Answer Generation Strategy

General queries should separate retrieval from answering:

1. retrieve evidence first
2. generate answer from top evidence
3. evaluate grounding against evidence, not against prompt cleverness

This means the answer pipeline should record:

- retrieved evidence ids
- answer text
- answer grounding score
- whether the answer mentions unsupported numbers or claims

## Evaluation Strategy

### For the query itself

Use lightweight evaluation only:

- query is understandable
- query is document-relevant
- query can be localized to evidence

Do not require:

- multihop reasoning depth
- dual-evidence necessity
- graph-path dependence

### For the answer

Use grounding-focused evaluation:

- `answer_supported`: is the answer fully supported by the retrieved evidence?
- `number_grounded`: are numbers traceable to evidence?
- `citation_grounded`: if cross-paper, are both sides actually grounded?
- `hallucination_warn`: does the answer add unsupported interpretation?

### For the whole system

Use two-level metrics:

- retrieval:
  - Recall@k / MRR on evidence ids
- answer:
  - evidence mention / grounding pass rate / hallucination rate

## Immediate Recommendation

Do not merge general queries into the current M2 production sweep.

Instead:

1. keep current full-doc generation focused on graph-synthesis queries
2. build a separate `general_queries_v1.jsonl` / `general_answers_v1.jsonl`
3. start from a curated 50-100 query C-Pool
4. add answer-grounding evaluation as a separate experiment block

## Open Next Step

Create a dedicated CPU-side script for:

- sampling documents
- prompting general queries from section / figure / table context
- generating grounded answers from retrieved evidence
- scoring answer grounding with lightweight judge logic

