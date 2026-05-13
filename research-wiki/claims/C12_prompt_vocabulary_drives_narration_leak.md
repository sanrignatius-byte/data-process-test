---
type: claim
node_id: claim:C12
title: "Prompt vocabulary density drives container-narration leakage in generated answers"
status: supported
date: 2026-05-13
updated: 2026-05-13 (initial — supported by 3251-row audit on graph_max20k snapshot)
related_experiments: [exp:20260513_qc_prompt_v2_tightening]
related_claims: []
---

# Statement

When a generation prompt uses a label word (e.g. "Bridge") to mark a structural
slot in its own scaffolding **N times in user-visible prose**, the LLM will
echo that label as a noun reference in the generated answer (`"the bridge
explains why..."`) at a frequency proportional to N — regardless of explicit
forbidden-phrase rules issued elsewhere in the same prompt. The fix is not
stricter rules but **reducing N in the prompt itself**.

# Evidence

Direct test on `data/03_queries/graph_max20k_four_cells_snapshot_20260513_utc/`
(4 cells × ~1500 = 6111 generated, 3251 pass).

## Prompt vs. output correlation

| Prompt version | user-visible `bridge*` occurrences | "the bridge X" in pass-set answers | "the bridge" in queries |
|---|---:|---:|---:|
| Snapshot prompt (in production at generation time) | 22 | **2707 / 3251 (83.3%)** | 95 / 3251 (2.9%) |
| Prompt v2 after structural scrub (this branch) | 9 (all intentional: format / schema / forbidden-quote) | not yet measured end-to-end | not yet measured end-to-end |

The snapshot prompt explicitly told the model (Rule 17 in copilot v1; absent in
the original) NOT to write "the bridge" in the answer — yet 83.3% of answers
did. The instruction-strength gradient (22 positive mentions vs 1 negative
mention) overwhelmed the rule.

## Verb-level breakdown of leaked references in answers (pass set, n=3251)

| Continuation after "the bridge" | Count |
|---|---:|
| explains | 1726 |
| says | 318 |
| states | 199 |
| links | 154 |
| then | 126 |
| ties / notes / frames / connects / maps / ... | 183 |
| **Total** | **2706** |

## Symmetric finding for "premise/conclusion" role labels

| Role label in user-visible prose | Pass-set answers using `the premise/conclusion` as noun |
|---|---:|
| Role labels used in JSON schema + 1 instructional sentence | 186 / 3251 (5.7%) |

Lower than the bridge case because the role labels appeared fewer times in the
prompt's prose — consistent with the density hypothesis.

# Mechanism

LLMs are sensitive to the frequency of a token in their context window. When a
word appears repeatedly in the system / user message — especially as a label
attached to a content block the model is asked to use — the model learns
"this word is a legitimate way to refer to this content." A single late-prompt
rule saying "do not use this word in the answer" is too weak a signal against
the cumulative association built by 22 prior mentions.

The same mechanism explains why supplying a worked example whose answer says
`"As shown in the figure, ..."` produces meta-language failures even when the
prompt's Rule 3 explicitly bans `"as shown in"`.

# Implications

1. **Rename, don't restrict.** When a structural slot must be discussed in the
   prompt (a connecting paragraph, a worked example, a role label), pick a
   user-visible name that is awkward or generic enough that the model is
   unlikely to copy it as a referent. Reserve identifier-style names
   (`bridge_paragraph`, `step_id`) for the JSON schema only.
2. **Worked example is the strongest teacher.** Whatever phrasing the GOOD
   example uses, the model will use. The GOOD example's answer text is more
   instructive than any Rule N statement.
3. **Forbidden-phrase rules need code enforcement.** Even after density
   reduction, ~5-10% of generations will still leak. Regex hard-fail in QC is
   the only reliable backstop.
4. **Audit by counting**, not by sampling. Per-pattern prevalence on a multi-
   thousand-row corpus is needed to spot density-driven leakage — small-N
   reviews miss the pattern entirely.

# Predictions to validate

| Prediction | Expected metric on next 50-candidate smoke run |
|---|---|
| Prompt v2 (9 visible `bridge*`) will drop "the bridge X" in answers below 10% | `bridge_narration_in_answer < 10%` |
| Prompt v2 will not introduce a comparable "the connecting paragraph X" leak (canonical name is awkward enough) | `< 5% of answers contain "the connecting paragraph X"` |
| Removing `<placeholder>` syntax from OPENER VARIETY will eliminate literal `<observation>` echoes | 0 outputs with `<...>` brackets |
| Listing apostrophe superlative forms verbatim in Rule 13 will not by itself eliminate them (model still slips) — QC regex hard-fail is required | apostrophe form rate < 1% only after combined prompt+QC; prompt-only ~5% |

# Scope

- Holds for instruction-tuned LLMs (Claude, GPT-4, etc.) used in zero-shot
  generation tasks. Not tested on smaller models or fine-tuned ones.
- The density threshold (around what N triggers the leak) is not measured —
  only the qualitative direction (22 → 83%, 9 → expected <10%).
- Independent of language: tested on English generation; the mechanism is
  token-frequency-based and should generalize.

# Counter-evidence to look for

- A prompt with N=5+ user-visible mentions of a label word where the model
  still produces <5% leakage → would weaken the density hypothesis.
- A prompt where renaming + reducing density does NOT lift downstream QC pass
  rate after re-generation → would suggest other factors dominate.

The 50-candidate smoke run (Follow-up P0 in
[exp:20260513_qc_prompt_v2_tightening](../experiments/20260513_qc_prompt_v2_tightening.md))
is the immediate validation.
