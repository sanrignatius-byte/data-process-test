"""Tests for src/qc/checks.py — the data quality gate.

Coverage strategy: each check function gets ≥2 test cases (one pass, one
fail) so regressions are caught immediately.
"""

from __future__ import annotations

import pytest

from src.qc.checks import (
    anchor_leak_jaccard,
    check_bridge_one_sided,
    check_evidence_spans,
    has_bridge_meta_leak_in_query,
    has_bridge_meta_pointer,
    has_bridge_narration_in_answer,
    has_min_reasoning_chain,
    has_no_cross_modal_operator,
    has_numeric_leakage,
    has_premise_conclusion_meta_in_answer,
    has_relationship_connector,
    has_shortcut_template,
    has_superlative_answer_spoiler,
    has_templated_opening,
    is_noisy_enrichment,
    is_yes_no_question,
    premise_conclusion_paraphrase_score,
    query_length_bucket,
    query_word_count,
)


# ── has_numeric_leakage ──────────────────────────────────────────────────────

class TestHasNumericLeakage:
    def test_no_leakage(self):
        assert has_numeric_leakage("What causes the trend in the graph?") is False

    def test_year_is_exempt(self):
        assert has_numeric_leakage("What changed in 2024?") is False

    def test_float_leaks(self):
        assert has_numeric_leakage("The accuracy is 0.95, why?") is True

    def test_zero_and_one_exempt(self):
        assert has_numeric_leakage("Why is the value 0 or 1?") is False


# ── is_yes_no_question ───────────────────────────────────────────────────────

class TestIsYesNo:
    def test_wh_question(self):
        assert is_yes_no_question("What causes the drop in performance?") is False

    def test_yes_no_is(self):
        assert is_yes_no_question("Is this method better?") is True

    def test_yes_no_does(self):
        assert is_yes_no_question("Does the model converge?") is True

    def test_how_question(self):
        assert is_yes_no_question("How does training loss change?") is False


# ── has_no_cross_modal_operator ──────────────────────────────────────────────

class TestCrossModalOperator:
    def test_with_operator(self):
        # has "relate to" → operator present → returns False
        assert has_no_cross_modal_operator(
            "How does the trend in Figure 1 relate to Table 2?"
        ) is False

    def test_without_operator(self):
        assert has_no_cross_modal_operator(
            "What is shown in the figure?"
        ) is True


# ── has_shortcut_template ────────────────────────────────────────────────────

class TestShortcutTemplate:
    def test_clean(self):
        assert has_shortcut_template(
            "What mechanism explains the accuracy gap between models A and B?"
        ) is False

    def test_template_how_does_x_relate(self):
        assert has_shortcut_template("How does X relate to Y?") is True


# ── has_templated_opening ────────────────────────────────────────────────────

class TestTemplatedOpening:
    def test_clean_opening(self):
        assert has_templated_opening("Why does the model fail on long sequences?") is False

    def test_given_that_opening(self):
        # "given that" IS in TEMPLATED_QUERY_OPENINGS
        assert has_templated_opening("Given that the data shows a trend, what follows?") is True

    def test_what_causes_opening(self):
        # "what causes" IS in TEMPLATED_QUERY_OPENINGS
        assert has_templated_opening("What causes the divergence in performance?") is True


# ── query_word_count + query_length_bucket ───────────────────────────────────

class TestQueryLength:
    def test_word_count(self):
        # query_word_count uses [A-Za-z0-9][A-Za-z0-9_-]* — "?" is not a token
        assert query_word_count("How many tokens are there") == 5

    def test_bucket_too_short(self):
        # SHORT_QUERY_MIN_WORDS = 8, so < 8 words → "too_short"
        assert query_length_bucket("Short query here") == "too_short"

    def test_bucket_short(self):
        # 8-14 words → "short"
        q = "What does the model show about the training process trends"  # 10 words
        assert query_length_bucket(q) == "short"

    def test_bucket_long(self):
        # >= LONG_QUERY_MIN_WORDS (18) → "long"
        q = " ".join(["word"] * 20)
        assert query_length_bucket(q) == "long"


# ── check_evidence_spans ─────────────────────────────────────────────────────

class TestCheckEvidenceSpans:
    def test_valid_spans(self):
        obj = {
            "required_evidence_spans": [
                {"element_id": "doc_fig_1", "span": "The model shows convergence after 50 epochs of training"},
                {"element_id": "doc_tbl_1", "span": "Accuracy reaches 95.3% on the validation dataset"},
            ],
        }
        pair = {"element_a_id": "doc_fig_1", "element_b_id": "doc_tbl_1"}
        assert check_evidence_spans(obj, pair) is True

    def test_missing_element(self):
        obj = {
            "required_evidence_spans": [
                {"element_id": "doc_fig_1", "span": "some long span here about the model"},
            ],
        }
        pair = {"element_a_id": "doc_fig_1", "element_b_id": "doc_tbl_1"}
        assert check_evidence_spans(obj, pair) is False


# ── anchor_leak_jaccard ──────────────────────────────────────────────────────

class TestAnchorLeakJaccard:
    def test_no_leak(self):
        score = anchor_leak_jaccard(
            "What causes performance degradation?",
            [{"anchor": "model accuracy drops significantly after epoch 100"}],
        )
        assert score < 0.5

    def test_high_overlap(self):
        score = anchor_leak_jaccard(
            "the model accuracy metric",
            [{"anchor": "the model accuracy metric"}],
        )
        assert score > 0.5


# ── has_min_reasoning_chain ──────────────────────────────────────────────────

class TestMinReasoningChain:
    def test_with_chain(self):
        obj = {
            "reasoning_chain": (
                "First the model observes the pattern in the data. "
                "Then it correlates this with the theoretical prediction; "
                "leading to the final conclusion about convergence."
            ),
        }
        assert has_min_reasoning_chain(obj) is True

    def test_no_chain(self):
        obj = {"reasoning_chain": "Short."}
        assert has_min_reasoning_chain(obj) is False

    def test_empty(self):
        assert has_min_reasoning_chain({}) is False


# ── has_relationship_connector ───────────────────────────────────────────────

class TestRelationshipConnector:
    def test_with_connector(self):
        assert has_relationship_connector(
            "The improvement is because of the new architecture, "
            "which leads to better convergence"
        ) is True

    def test_without_connector(self):
        assert has_relationship_connector("The result is shown") is False


# ── is_noisy_enrichment ──────────────────────────────────────────────────────

class TestNoisyEnrichment:
    def test_clean_text(self):
        assert is_noisy_enrichment(
            "This figure shows the training loss curve over 100 epochs"
        ) is False

    def test_glyph_noise(self):
        assert is_noisy_enrichment("⊕ ⊗ ▲ ● □ ◆ ◇") is True


# ── has_bridge_narration_in_answer ───────────────────────────────────────────

class TestBridgeNarrationInAnswer:
    def test_bridge_explains(self):
        assert has_bridge_narration_in_answer(
            "FairBoost reweights margins. The bridge explains why."
        ) is True

    def test_bridge_says(self):
        assert has_bridge_narration_in_answer(
            "The bridge says reweighting overcorrects."
        ) is True

    def test_bridge_links(self):
        assert has_bridge_narration_in_answer(
            "Then the bridge links the figure observation to the table."
        ) is True

    def test_bare_the_bridge(self):
        assert has_bridge_narration_in_answer(
            "The training loss drops. And then the bridge."
        ) is True

    def test_clean_mechanism_answer(self):
        # Names the actual mechanism, no "bridge" reference at all
        assert has_bridge_narration_in_answer(
            "FairBoost reweights majority margins, which shrinks minority confidence."
        ) is False

    def test_empty_answer_does_not_fail(self):
        assert has_bridge_narration_in_answer("") is False


# ── has_premise_conclusion_meta_in_answer ────────────────────────────────────

class TestPremiseConclusionMetaInAnswer:
    def test_the_premise(self):
        assert has_premise_conclusion_meta_in_answer(
            "FairBoost reweights margins. The premise establishes this."
        ) is True

    def test_the_conclusion(self):
        assert has_premise_conclusion_meta_in_answer(
            "Margins shrink. The conclusion shows the gap."
        ) is True

    def test_clean(self):
        assert has_premise_conclusion_meta_in_answer(
            "FairBoost reweights majority margins, producing a minority confidence gap."
        ) is False


# ── has_bridge_meta_leak_in_query ────────────────────────────────────────────

class TestBridgeMetaLeakInQuery:
    def test_after_the_bridge_notes(self):
        assert has_bridge_meta_leak_in_query(
            "Which method aligns with X after the bridge notes Y?"
        ) is True

    def test_the_bridges_claim(self):
        assert has_bridge_meta_leak_in_query(
            "Which choice matches the bridge's claim about Z?"
        ) is True

    def test_clean_mechanism_query(self):
        assert has_bridge_meta_leak_in_query(
            "Why does FairBoost reweighting shrink minority confidence?"
        ) is False


# ── has_superlative_answer_spoiler ───────────────────────────────────────────

class TestSuperlativeAnswerSpoiler:
    def test_strongest_with_achieves(self):
        assert has_superlative_answer_spoiler(
            "Which method achieves the strongest accuracy on COCO?",
            "Mask2Former achieves the highest accuracy on COCO."
        ) is True

    def test_apostrophe_form(self):
        assert has_superlative_answer_spoiler(
            "Which method's stronger result emerges from late fusion?",
            "X outperforms all baselines."
        ) is True

    def test_no_resolver_in_answer(self):
        # Query has superlative but answer doesn't claim "X achieves/outperforms"
        assert has_superlative_answer_spoiler(
            "Which method's strongest behavior follows from X?",
            "The behavior arises from reweighting, which compresses minority margins."
        ) is False

    def test_no_superlative_in_query(self):
        assert has_superlative_answer_spoiler(
            "Why does FairBoost overcorrect minority margins?",
            "It achieves the lowest variance under reweighting."
        ) is False


# ── has_bridge_meta_pointer ──────────────────────────────────────────────────

class TestBridgeMetaPointer:
    def test_we_report_results(self):
        assert has_bridge_meta_pointer(
            "We report results in Table 5 for the ablation."
        ) is True

    def test_detailed_in_section(self):
        assert has_bridge_meta_pointer(
            "The full numbers are detailed in Section 4.2."
        ) is True

    def test_we_conducted_ablation(self):
        assert has_bridge_meta_pointer(
            "We conducted an ablation study on the embedding dimension."
        ) is True

    def test_mechanism_bridge(self):
        # Real mechanism sentence — not a pointer
        assert has_bridge_meta_pointer(
            "FairBoost reweights majority margins, which overcorrects minority confidence."
        ) is False


# ── premise_conclusion_paraphrase_score ──────────────────────────────────────

class TestPremiseConclusionParaphrase:
    def test_identical_returns_high(self):
        s = premise_conclusion_paraphrase_score(
            "fairboost reweights majority margins",
            "fairboost reweights majority margins",
        )
        assert s == 1.0

    def test_disjoint_returns_zero(self):
        s = premise_conclusion_paraphrase_score(
            "fairboost reweights majority margins",
            "the encoder uses sinusoidal position embedding",
        )
        assert s == 0.0

    def test_partial_overlap_below_threshold(self):
        s = premise_conclusion_paraphrase_score(
            "fairboost reweights majority margins on COCO splits",
            "training stability drops on the minority cohort",
        )
        assert s < 0.55

    def test_empty_returns_zero(self):
        assert premise_conclusion_paraphrase_score("", "anything") == 0.0


# ── check_bridge_one_sided ───────────────────────────────────────────────────

class TestBridgeOneSided:
    def _obj(self, premise_span, bridge_span, conclusion_span):
        return {
            "reasoning_steps": [
                {"step_id": 1, "reasoning_role": "premise",
                 "evidence_element_id": "e_a", "evidence_span": premise_span},
                {"step_id": 2, "reasoning_role": "intermediate",
                 "evidence_element_id": "bridge_paragraph", "evidence_span": bridge_span},
                {"step_id": 3, "reasoning_role": "conclusion",
                 "evidence_element_id": "e_b", "evidence_span": conclusion_span},
            ]
        }

    def test_bridge_connects_both(self):
        obj = self._obj(
            premise_span="fairboost reweights majority cohort margins",
            bridge_span="reweights overcorrect majority margins which shrinks minority confidence",
            conclusion_span="minority cohort confidence drops sharply on the test split",
        )
        fail, _ = check_bridge_one_sided(obj, {})
        # bridge shares "reweights/majority/margins" with premise and
        # "minority/confidence" with conclusion → connects both sides
        assert fail is False

    def test_bridge_only_premise_side(self):
        obj = self._obj(
            premise_span="fairboost reweights majority cohort margins",
            bridge_span="fairboost reweights majority cohort under different lambda values",
            conclusion_span="memory consumption decreases by quantization",  # zero overlap with bridge
        )
        fail, _ = check_bridge_one_sided(obj, {})
        assert fail is True

    def test_no_bridge_step_is_skipped(self):
        obj = {"reasoning_steps": [
            {"step_id": 1, "evidence_element_id": "e_a", "evidence_span": "foo"},
            {"step_id": 2, "evidence_element_id": "e_b", "evidence_span": "bar"},
        ]}
        fail, _ = check_bridge_one_sided(obj, {})
        assert fail is False
