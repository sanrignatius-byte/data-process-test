"""Tests for src/models/training.py — Pydantic schema validation.

Ensures that the StandardQuery / Triplet models enforce constraints
correctly, preventing bad data from entering the training pipeline.
"""

from __future__ import annotations

import json

import pytest

from src.models.training import (
    DifficultyLevel,
    EvidenceSpan,
    ReasoningStep,
    StandardQuery,
    Triplet,
)


# ── EvidenceSpan ──────────────────────────────────────────────────────────────

class TestEvidenceSpan:
    def test_valid(self):
        es = EvidenceSpan(element_id="doc_fig_1", doc_id="doc", span_text="hello")
        assert es.element_id == "doc_fig_1"

    def test_empty_element_id_rejected(self):
        with pytest.raises(Exception):
            EvidenceSpan(element_id="  ", doc_id="doc")


# ── StandardQuery ─────────────────────────────────────────────────────────────

class TestStandardQuery:
    def _make_minimal(self, **overrides) -> dict:
        base = {
            "query_id": "q1",
            "query_text": "What does the figure show?",
            "answer_text": "The training curve.",
            "difficulty_level": "L1",
            "evidence_spans": [
                {"element_id": "doc_fig_1", "doc_id": "doc", "span_text": "curve"}
            ],
        }
        base.update(overrides)
        return base

    def test_valid_l1(self):
        sq = StandardQuery(**self._make_minimal())
        assert sq.difficulty_level == DifficultyLevel.L1
        assert len(sq.evidence_spans) == 1

    def test_empty_evidence_rejected(self):
        with pytest.raises(Exception):
            StandardQuery(**self._make_minimal(evidence_spans=[]))

    def test_l3_requires_reasoning_steps(self):
        with pytest.raises(Exception):
            StandardQuery(**self._make_minimal(
                difficulty_level="L3",
                reasoning_steps=[],
            ))

    def test_l3_with_steps(self):
        sq = StandardQuery(**self._make_minimal(
            difficulty_level="L3",
            reasoning_steps=[{
                "step_id": 1,
                "evidence_element_id": "doc_fig_1",
                "reasoning_role": "premise",
                "produces_claim": "Something",
            }],
        ))
        assert sq.difficulty_level == DifficultyLevel.L3
        assert len(sq.reasoning_steps) == 1

    def test_round_trip_json(self):
        sq = StandardQuery(**self._make_minimal())
        js = sq.model_dump_json()
        sq2 = StandardQuery.model_validate_json(js)
        assert sq.query_id == sq2.query_id
        assert sq.evidence_spans[0].element_id == sq2.evidence_spans[0].element_id

    def test_extra_fields_preserved(self):
        sq = StandardQuery(**self._make_minimal(
            extra={"bridge_quality": "high", "custom_field": 42}
        ))
        assert sq.extra["bridge_quality"] == "high"


# ── Triplet ───────────────────────────────────────────────────────────────────

class TestTriplet:
    def test_valid(self):
        t = Triplet(
            query_id="q1",
            query_text="What does the figure show?",
            difficulty_level="L1",
            positive=[EvidenceSpan(element_id="doc_fig_1", span_text="curve")],
            hard_negatives=[EvidenceSpan(element_id="doc_tbl_1", span_text="table")],
            negative_strategy="random",
        )
        assert len(t.positive) == 1
        assert len(t.hard_negatives) == 1

    def test_empty_positive_rejected(self):
        with pytest.raises(Exception):
            Triplet(
                query_id="q1",
                query_text="test",
                difficulty_level="L1",
                positive=[],
            )

    def test_round_trip_json(self):
        t = Triplet(
            query_id="q1",
            query_text="What does the figure show?",
            difficulty_level="L1",
            positive=[EvidenceSpan(element_id="doc_fig_1", span_text="curve")],
        )
        js = t.model_dump_json()
        t2 = Triplet.model_validate_json(js)
        assert t.query_id == t2.query_id
