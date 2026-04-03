"""Tests for src/utils/text_utils.py — tokenisation and similarity helpers."""

from __future__ import annotations

from src.utils.text_utils import (
    jaccard,
    overlap_ratio,
    tokenize,
    tokenize_for_retrieval,
    tokenize_words,
)


# ── tokenize ─────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_basic(self):
        tokens = tokenize("Hello World foo")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens

    def test_short_words_excluded(self):
        tokens = tokenize("I am a cat in the hat")
        assert "am" not in tokens
        assert "cat" in tokens
        assert "hat" in tokens

    def test_empty(self):
        assert tokenize("") == set()
        assert tokenize(None) == set()


# ── tokenize_words ───────────────────────────────────────────────────────────

class TestTokenizeWords:
    def test_includes_digits(self):
        tokens = tokenize_words("accuracy 0.95 epoch 100")
        assert "accuracy" in tokens
        assert "100" in tokens

    def test_empty(self):
        assert tokenize_words("") == set()


# ── tokenize_for_retrieval ───────────────────────────────────────────────────

class TestTokenizeForRetrieval:
    def test_returns_list(self):
        result = tokenize_for_retrieval("model training loss")
        assert isinstance(result, list)
        assert "model" in result

    def test_excludes_short(self):
        result = tokenize_for_retrieval("a b cd efg")
        assert "efg" in result
        # single-char tokens excluded by TOKEN_RE pattern
        assert "a" not in result


# ── jaccard ──────────────────────────────────────────────────────────────────

class TestJaccard:
    def test_identical_sets(self):
        assert jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        assert jaccard({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        score = jaccard({"a", "b", "c"}, {"b", "c", "d"})
        assert 0.4 < score < 0.6  # 2/4 = 0.5

    def test_empty_sets(self):
        assert jaccard(set(), set()) == 0.0


# ── overlap_ratio ────────────────────────────────────────────────────────────

class TestOverlapRatio:
    def test_full_overlap(self):
        assert overlap_ratio({"a", "b"}, {"a", "b", "c"}) == 1.0

    def test_no_overlap(self):
        assert overlap_ratio({"a"}, {"b"}) == 0.0

    def test_empty(self):
        assert overlap_ratio(set(), {"a"}) == 0.0
