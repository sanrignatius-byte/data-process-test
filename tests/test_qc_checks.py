"""Tests for src/qc/checks.py — the data quality gate.

Coverage strategy: each check function gets ≥2 test cases (one pass, one
fail) so regressions are caught immediately.
"""

from __future__ import annotations

import pytest

from src.qc.checks import (
    anchor_leak_jaccard,
    check_evidence_spans,
    has_min_reasoning_chain,
    has_no_cross_modal_operator,
    has_numeric_leakage,
    has_relationship_connector,
    has_shortcut_template,
    has_templated_opening,
    is_noisy_enrichment,
    is_yes_no_question,
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
