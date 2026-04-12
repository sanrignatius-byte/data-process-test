# Long-Chain Query Generation Pipeline — Technical Handoff

**Date**: 2026-04-09  
**Purpose**: Complete technical reference for the long-chain multi-hop query generation pipeline. Covers data flow, image resolution, chain construction, chain decomposition, and all pending work.

---

## 1. End-to-End Pipeline Overview

```
┌──────────────────┐     ┌───────────────────────┐     ┌──────────────────────────────┐
│  MinerU PDF      │     │  multimodal_elements   │     │  select_intra_doc_pairs.py   │
│  Parse Output    │────▶│  .json (76 docs,       │────▶│  --strategy chain            │
│  (content_list)  │     │  1316 elems, 1261 edges)│     │  --min-chain-hops 3          │
└──────────────────┘     └───────────────────────┘     └──────────────┬───────────────┘
                                                                      │
                                                        CandidatePair JSON
                                                        (hub_candidates_enriched format)
                                                                      │
                                                                      ▼
                                                       ┌──────────────────────────────┐
                                                       │  generate_long_chain_        │
                                                       │  iterative_queries.py        │
                                                       │  --candidates <pairs.json>   │
                                                       └──────────────┬───────────────┘
                                                                      │
                                                                  JSONL output
                                                                      │
                                                                      ▼
                                                       ┌──────────────────────────────┐
                                                       │  data/03_queries/*.jsonl     │
                                                       │  (pass + fail queries)       │
                                                       └──────────────────────────────┘
```

---

## 2. How Images / Tables / Formulas Are Found

### 2.1 Origin: MinerU Parse → `multimodal_elements.json`

The graph builder `src/linkers/multimodal_relationship_builder.py` reads MinerU's `content_list.json` (or `content_list_v2.json`) for each document:

```
data/00_raw/mineru_output/{doc_id}/{doc_id}/hybrid_auto/{doc_id}_content_list.json
```

For each item of type `image`, `figure`, or `table`, it extracts:
- **`image_path`**: from `item.img_path` → `item.content.image_source.path` → `item.image_path` (in priority order)
- **`caption`**: from `table_caption` (tables) or `image_caption` (figures)
- **`content`**: raw text/LaTeX content (especially important for formulas and tables)

These get stored in `data/01_graphs/multimodal_elements.json` under `documents.{doc_id}.elements.{element_id}`.

### 2.2 Image Coverage in multimodal_elements.json

| Element Type | Total | With `image_path` | Coverage |
|-------------|-------|-------------------|----------|
| figure      | 841   | 841               | **100%** |
| table       | 334   | 237               | **71%**  |
| formula     | 141   | 0                 | **0%**   |

- **Figures**: Always have images (MinerU extracts them as JPG)
- **Tables**: ~71% have rendered images; the rest have only text/HTML content
- **Formulas**: Never have images in `multimodal_elements.json`; they only have LaTeX content text

### 2.3 `content_list_v2` — Where It Is and Isn't Used

**NOT used in query generation pipeline.** Neither `image_utils.py`, `generate_long_chain_iterative_queries.py`, nor `generate_multihop_l1_queries.py` touches `content_list_v2`.

**Only used in** `scripts/export_evidence_md.py` for the Evidence Markdown export feature:
- `_build_content_list_image_map()` scans `content_list_v2.json` to find formula/table rendered images as a fallback when `image_path` is empty in `multimodal_elements.json`
- This is a **presentation-only** feature, not part of query generation

### 2.4 Image Path Resolution at Runtime

When a script calls `encode_image(path)` (from `src/utils/image_utils.py`), the path goes through `_resolve_core()` with 4 strategies:

```
Strategy 0: data/mineru_output/... → data/00_raw/mineru_output/...  (auto-rewrite)
Strategy 1: Strip known prefixes (/projects/_hdd/myyyx1/...) → relative → PROJECT_ROOT/relative
Strategy 2: Extract suffix after /data/mineru_output/ marker → PROJECT_ROOT/data/00_raw/mineru_output/suffix
Strategy 3: Generic /data/ re-root → PROJECT_ROOT/data/...
```

**Symlink**: `data/mineru_output -> 00_raw/mineru_output` ensures both old and new paths work.

**Verified working**: absolute paths like `/projects/_hdd/myyyx1/data-process-test/data/mineru_output/1306.5204/...` resolve correctly via Strategy 1 + symlink.

---

## 3. How Long Chains Are Constructed

### 3.1 ChainFinder Algorithm (`src/pairing/chain_finder.py`)

**Input**: One document's elements + edges from `multimodal_elements.json`

**Graph construction**:
1. Keep only modal elements (figure, table, formula) — skip paragraphs, sections, etc.
2. Build **undirected** adjacency from edges (cross-reference links parsed from paper text)
3. Only edges where BOTH endpoints are modal elements are kept

**DFS enumeration** (`find_chains()`):
1. Start DFS from every modal element
2. Maintain `visited` set → simple paths only (no node revisited)
3. Collect path when: cannot extend further (all neighbors visited) OR reached `max_length=12`
4. Only keep paths with `len >= min_length` (default 3 = at least 4 nodes = 3 hops)
5. Deduplicate by `frozenset(path)` — undirected graph means A→B→C = C→B→A
6. Safety: cap at `MAX_RESULTS=2000` chains per document

**Scoring** (`_score_chain()`):
```
score = length_bonus * 0.4 + cross_modal * 0.4 + diversity * 0.2

length_bonus = min(len(path) / 8, 1.0)           # 8-node chain = max
cross_modal  = min(transitions * 0.1, 0.4)       # each type change +0.1
diversity    = (unique_modality_types / 3) * 0.2  # all 3 types = max
```

### 3.2 What Edges Mean

Each edge in `multimodal_elements.json` is a **cross-reference** found in the paper text. Example:

```json
{
  "source_id": "1809.02208_table_1",
  "target_id": "1809.02208_figure_1",
  "ref_text": "Figure 1 e",
  "context_snippet": "As Figure 1 exemplifies, this approach produces results consistent with..."
}
```

This means a paragraph near Table 1 references Figure 1. The `context_snippet` is the surrounding text.

### 3.3 From Chains to CandidatePairs

`IntraDocPairSelector._chain_pairs()` converts `ChainResult` → `CandidatePair`:

- `element_a` = first node of chain (full ElementDetail with caption, content, context, image_path)
- `element_b` = last node of chain
- `path` = full ordered list of element_ids in the chain
- **`node_group`** = **ALL elements in the chain** (not just endpoints) — each with full metadata
- `edge_contexts` = the cross-reference text for each consecutive pair in the chain
- `hop_distance` = number of edges = len(path) - 1
- `strategy = "chain"`
- `hub_metadata` = `{chain_hops, modality_sequence, cross_modal_transitions, unique_modalities}`

### 3.4 Real Data: Chain Statistics

```
76 documents → 20,459 chains (≥3 hops)
Max chain length: 11 hops (12 nodes)
Dense documents: 1809.02208, 1809.10083, 2005.07293 have 11-hop chains with score 1.0

After endpoint dedup + max_per_doc=5:
  206 chain pairs selected
  Hop distribution: 3-11 hops
  45 documents covered
```

---

## 4. Can Chains Be Decomposed? Yes.

A chain of length N can be decomposed into (N-2) overlapping sub-chains of length 3:

**Example**: 5-hop chain through doc `1809.02208`:
```
Full chain: figure_1 → table_1 → table_3 → figure_2 → table_6 → figure_6

Sub-chains (sliding window of 3):
  [0]: figure_1 → table_1 → table_3    (figure→table→table)
  [1]: table_1  → table_3 → figure_2   (table→table→figure)
  [2]: table_3  → figure_2 → table_6   (table→figure→table)
  [3]: figure_2 → table_6 → figure_6   (figure→table→figure)
```

Each sub-chain is itself a valid 2-hop reasoning triplet. The `generate_long_chain_iterative_queries.py` script already does this implicitly — it generates **hop-by-hop bridge steps** for each consecutive pair:

```
Stage 1: Generate subquery for figure_1 → table_1 (hop 1)
Stage 2: Generate subquery for table_1 → table_3 (hop 2)
Stage 3: Generate subquery for table_3 → figure_2 (hop 3)
...
Stage N: Generate FINAL query that requires ALL bridge facts from previous hops
```

The iterative architecture ensures each hop is grounded in its own evidence, and the ablation QC (step-deletion test) verifies that removing any intermediate node makes the final query unanswerable.

---

## 5. How the Long-Chain Script Works (`generate_long_chain_iterative_queries.py`)

### 5.1 Architecture: Iterative Bridge-Step Generation

Unlike `generate_multihop_l1_queries.py` (one-shot prompt), the long-chain script uses a **multi-stage** approach:

```
For each pair:
  1. get_path_nodes(pair) → resolve all elements from node_group
  2. For each intermediate node (hop 1..N-1):
     - build_step_prompt(source, target, prior_steps)
     - LLM call → extract {subquery, anchor, evidence_span, step_answer}
     - Accumulate bridge_steps[]
  3. build_final_prompt(start, end, bridge_steps)
     - LLM call → {final_query, final_answer, query_type, spans, anchors, text_evidence}
  4. refine_query_answer_locally() → deterministic cleanup (remove numbers, template rewrites, inject bridge terms)
  5. qc_multihop_query() → rule-based QC
  6. run_ablation_qc() → LLM step-deletion test
  7. judge_answer_grounding() → LLM hallucination check
  8. maybe_repair_candidate() → one focused repair attempt if QC fails
```

### 5.2 get_path_nodes() — How Elements Are Loaded

This function resolves all chain nodes from the pair dict. It checks three sources in order:

1. `element_a` / `element_b` (authoritative endpoint data — always used)
2. `intermediate_elements` (old format, only intermediates)
3. **`node_group`** (new format from `src/pairing/`, all elements in chain)

Elements from `element_a`/`element_b` take priority (never overwritten by `node_group`). Returns ordered list matching `pair["path"]`, or `None` if any node is missing.

### 5.3 Image Usage in Long-Chain Script

```python
# In hop-step generation (build_step_prompt):
imgs = [encode_image(source.get("image_path")), encode_image(target.get("image_path"))]

# In final query generation (build_final_prompt):
imgs = [encode_image(start.get("image_path")), encode_image(end.get("image_path"))]

# In repair:
imgs = [encode_image(start_elem.get("image_path")), encode_image(end_elem.get("image_path"))]
```

- Hop steps: images of source + target nodes for that hop
- Final query: images of chain endpoints only
- Formulas: `encode_image(None)` returns `None`, which is filtered out — no image sent for formula elements (they use LaTeX `content` text instead)

---

## 6. Critical Gaps in the Long-Chain Script

### 6.1 Missing from `generate_multihop_l1_queries.py` (MUST FIX)

| Feature | Multihop Script | Long-Chain Script | Impact |
|---------|----------------|-------------------|--------|
| **`--skip-done` (resume)** | ✅ Loads done pair_ids, skips | ❌ Not implemented | Process crash = restart from zero |
| **`flush()` after write** | ✅ Every write flushed | ❌ Not implemented | Process kill = 0 bytes output |
| **Append mode** | ✅ `open("a")` with skip-done | ❌ Always `open("w")` | Resume overwrites existing data |
| **Persona injection** | ✅ `--use-persona`, 76 PersonaHub personas | ❌ None | All queries have identical voice |
| **Query style diversity** | ✅ `--query-style mixed` (academic/real_user 50/50) | ❌ Single hardcoded prompt | Zero style variation |
| **Template library** | ✅ 11 templates from `src/prompts/templates.py` | ❌ 1 hardcoded `build_final_prompt` | Monotonous output |
| **Enriched context** | ✅ `_with_enriched()` appends enriched sections | ❌ Only caption/content/context_before/after | Less context for LLM |
| **Bridge text from reference graph** | ✅ `--reference-graph`, `resolve_bridge_texts_for_path()` | ❌ Relies entirely on LLM-extracted bridge | Missing author's actual words |

### 6.2 New QC Checks (IMPLEMENTED but not yet committed/tested)

Two new checks from the discussion record review have been implemented in `src/qc/checks.py` + `src/qc/pipelines.py`:

1. **`has_conditional_hedge_overload(answer, threshold=3)`** → issue `underdetermined_query`
   - Detects answers with ≥3 conditional hedges (if/assuming/suppose)
   - Indicates the query is under-specified
   
2. **`has_bridge_overclaim_signal(query, answer)`** → issue `bridge_overclaim`
   - Query uses strong causal language ("causes", "leads to") but answer hedges ("may", "might")
   - Indicates bridge is weaker than query claims

These are already wired into BOTH `qc_multihop_query()` and `qc_real_user_query()` pipelines. **Status: code changed, not committed, not tested.**

### 6.3 NOT Implemented Yet

| Feature | Description | Priority |
|---------|-------------|----------|
| Per-pair query cap | Same pair_id should produce max 2 pass queries | High |
| Semantic dedup | Token-level Jaccard > 0.4 between queries from same pair → reject | High |
| `qc_summary_label` | Human-readable label from `can_answer` × `confidence` combos | Medium |
| Slurm script | Reliable execution on compute nodes instead of login node nohup | **Critical** |

---

## 7. Uncommitted Changes (git diff as of 2026-04-09)

```
src/qc/checks.py    | +70 lines  (2 new check functions + docstring translation zh→en)
src/qc/pipelines.py | +53 lines  (wire new checks + docstring translation zh→en)
```

**What changed**:
- `checks.py`: Added `has_conditional_hedge_overload()` and `has_bridge_overclaim_signal()`, translated all Chinese docstrings to English
- `pipelines.py`: Added checks #11 (underdetermined_query) and #12 (bridge_overclaim) to both `qc_multihop_query()` and `qc_real_user_query()`, translated Chinese docstrings to English

---

## 8. Key File Locations

### Core Pipeline Files
| File | Purpose |
|------|---------|
| `data/01_graphs/multimodal_elements.json` | Source graph: 76 docs, 1316 elements, 1261 edges |
| `src/pairing/chain_finder.py` | ChainFinder: DFS-based multi-hop chain discovery |
| `src/pairing/intra_doc_pairs.py` | IntraDocPairSelector: 4 strategies (direct/2hop/section/chain) |
| `src/pairing/context_dedup.py` | Dedup overlapping context_before/context_after |
| `src/pairing/pair_schema.py` | CandidatePair Pydantic schema |
| `scripts/select_intra_doc_pairs.py` | CLI: select pairs → hub_candidates_enriched format JSON |
| `scripts/generate_long_chain_iterative_queries.py` | Iterative hop-by-hop query generation (1093 lines) |
| `scripts/generate_multihop_l1_queries.py` | One-shot dual-evidence query generation (1379 lines) |
| `src/utils/image_utils.py` | Image path resolution: `resolve_image_path()`, `encode_image()` |
| `src/linkers/multimodal_relationship_builder.py` | Builds multimodal_elements.json from MinerU output |

### Prompt & Style System (used by multihop, NOT by long-chain)
| File | Purpose |
|------|---------|
| `src/prompts/templates.py` | 11 prompt templates (6 academic + 5 real_user) |
| `src/prompts/personas.py` | 76 PersonaHub personas, `resolve_persona()`, `inject_persona_prefix()` |
| `src/prompts/styles.py` | `select_template()`, `resolve_query_style()` (academic/real_user/mixed router) |

### QC System
| File | Purpose |
|------|---------|
| `src/qc/checks.py` | 25+ atomic check functions |
| `src/qc/pipelines.py` | `qc_multihop_query()` (strict) + `qc_real_user_query()` (relaxed) |
| `src/qc/llm_judge.py` | `run_ablation_qc()`, `judge_answer_grounding()`, `run_llm_qc()` |

### Slurm Infrastructure
| File | Purpose |
|------|---------|
| `slurm_scripts/01_fetch_references.sh` | Existing template: `cluster02` partition, 4h, 4cpu, 12G |
| `slurm_scripts/submit_all.sh` | Batch submission wrapper |

---

## 9. Discussion Record: Review Findings & Improvement Roadmap

### Review Consensus (from external reviewers)

**Problems identified in query quality**:
1. **Pseudo-multihop (parallel assembly)**: Queries that use "together explain" but answer decomposes into two independent single-element lookups. LLM Judge correctly catches this as `llm_fake_multihop`.
2. **Meta-language**: Queries discuss "how to explain" rather than asking substantive questions. e.g., "How do X and Y together shape a fairness explanation?"
3. **Same-pair over-extraction**: Same pair_id producing multiple near-identical queries (semantic echo)
4. **Underdetermined queries**: Answers require 3+ if/assuming clauses to survive → query is under-specified
5. **Bridge overclaim**: Query uses strong causal language but answer only hedges → bridge is weaker than claimed
6. **Consulting tone**: "How should a product team..." — inappropriate for academic benchmark

### Agreed Improvement Actions

| # | Action | Scope | Status |
|---|--------|-------|--------|
| 1 | Per-pair query cap (max 2 pass per pair_id) | Pipeline | ❌ Not implemented |
| 2 | Semantic dedup (Jaccard > 0.4 → reject) | Pipeline | ❌ Not implemented |
| 3 | Conditional hedge check | QC atomic check | ✅ Implemented (uncommitted) |
| 4 | Bridge overclaim check | QC atomic check | ✅ Implemented (uncommitted) |
| 5 | `qc_summary_label` field | Output format | ❌ Not implemented |
| 6 | Persona + query-style for long-chain | Prompt system | ❌ Not implemented |
| 7 | Flush + skip-done for long-chain | Robustness | ❌ Not implemented |
| 8 | Slurm script for reliable execution | Infrastructure | ❌ Not implemented |

---

## 10. Immediate Next Steps (Priority Order)

1. **Write slurm script** for long-chain generation (cluster02, 4h, reference: `slurm_scripts/01_fetch_references.sh`)
2. **Add flush + skip-done + append** to `generate_long_chain_iterative_queries.py`
3. **Wire persona + query-style** into long-chain script's `build_final_prompt`
4. **Run tests** to verify QC changes don't break existing 107 tests
5. **Commit all changes** and push
6. **Submit slurm job**: `select_intra_doc_pairs.py --strategy chain` → `generate_long_chain_iterative_queries.py`
7. After completion: merge with existing pass queries, run coverage analysis

---

## 11. Code Changes — 2026-04-09 / 2026-04-10 Session

All changes are in `scripts/generate_long_chain_iterative_queries.py` unless noted.

### Fix 1: `reasoning_chain` synthesis from `bridge_steps`

**Problem**: `evaluate_current()` never populated the `reasoning_chain` field in `qc_obj`, but `has_min_reasoning_chain` in `src/qc/checks.py` checks `obj.get("reasoning_chain", "")`. Result: **100% false-positive** `missing_reasoning_chain` on all entries.

**Root cause**: The long-chain pipeline stores reasoning data in `bridge_steps` (structured list of `{hop_index, subquery, step_answer, ...}`), not in a flat `reasoning_chain` string. This mismatch existed since the pipeline was created.

**Fix v1 (2026-04-09)**: Synthesize `reasoning_chain_text` from `bridge_steps[].step_answer` only, then inject into `qc_obj["reasoning_chain"]`.

**Residual issue**: step_answers can be extremely terse (single variable names like `"A"`, `"M"`, or bare numbers like `"134.21"`). When all step_answers in a chain are short, the joined text is < 40 chars and still fails.

**Fix v2 (2026-04-10)**: Include `subquery` (bridge question) in the synthesis, format: `"{subquery} -> {step_answer}"`. A reasoning chain = question -> answer -> question -> answer.

**Impact**:
- Old entries (pre-fix): 41% hit rate -> N/A (not re-evaluated, `--skip-done`)
- New entries (fix v1): 6% (3/47 still failing)
- New entries (fix v2): Expected ~0% -- all 3 residual failures now produce 278-339 chars

**Location**: Lines ~838-857

### Fix 2: `steps_txt` prompt format — hide answers (2026-04-09)

**Problem**: `build_final_prompt()` exposed `anchor=X; span=Y; fact=Z` in the steps summary, which the LLM copied verbatim into queries -> inflated word count -> `query_too_long`.

**Fix**: Changed steps_txt from `anchor/span/fact` format to question-oriented format using `subquery`. Changed prompt label from "Bridge facts extracted from intermediate nodes:" to "Bridge questions this chain must resolve (do NOT reveal these answers in the query):".

**Impact**: Partial -- `query_too_long` still 53% (many queries inherently 28-33 words, boundary effect with `MAX_QUERY_WORDS=30`).

**Location**: Lines ~513-544

### Fix 3: QC divergence reporting (2026-04-09)

**Problem**: Rule QC and LLM QC systematically disagreed, but there was no way to see where or how much.

**Fix**: Added `compute_qc_divergence()` function + per-entry `qc_divergence` field + end-of-run summary classifying each issue as `agree_pass | agree_fail | rule_only_fail | llm_only_fail`.

**Location**: Lines ~1100-1148 (function), ~1555+1588 (per-entry), ~1599-1681 (stats + summary)

### Summary Table

| # | Fix | Lines Changed | Effect |
|---|-----|--------------|--------|
| 1 | `reasoning_chain` synthesis v1->v2 (subquery+answer) | ~838-857 | `missing_reasoning_chain`: 100%->41%->6%->~0% |
| 2 | `steps_txt` question-oriented, hide answers | ~513-544 | `query_too_long`: reduced but still ~53% |
| 3 | QC divergence reporting | ~1100-1681 | New diagnostic: rule vs LLM disagreement tracking |

### Known Remaining Issues (post-fix)

| Issue | Hit Rate | Root Cause | Proposed Fix |
|-------|----------|-----------|-------------|
| `query_too_long` | 53% | `MAX_QUERY_WORDS=30` too tight for long-chain (median ~30 words) | Raise to 35-40, or make warning |
| `fake_long_chain` | 47% | LLM ablation finds intermediate hops skippable | Structural: foreground/background architecture |
| `single_element_answer` | 37% | Token-overlap rule disagrees with LLM ablation | Downgrade to warning for long-chain |
| `text_evidence_over_reliance` | 28% | Queries rely too heavily on text evidence | Prompt improvement |
| `missing_reasoning_chain` | 26% total (mostly old data) | Historical entries from before fix | Will disappear in clean re-run |

---

## 12. Immediate Next Steps (Priority Order)

1. **Wait for Job 57369** to complete (~8h remaining, 43/211 done)
2. **Analyze full run results**: issue distribution, divergence stats, pass rate
3. **Decide on `MAX_QUERY_WORDS`**: raise 30->35-40 if query_too_long remains dominant
4. **Decide on `single_element_answer`**: downgrade to warning for long-chain if divergence persists
5. **Consider foreground/background architecture** if pass rate still < 5%
6. **Clean re-run** without old entries to get accurate stats
7. **Commit all changes** and push
