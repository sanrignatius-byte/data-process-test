"""Tests for src/sampling/negative_sampler.py — negative sampling strategies."""

from __future__ import annotations

from src.models import Chunk
from src.sampling.negative_sampler import (
    HeuristicNegativeSampler,
    build_sampler,
)


def _make_chunks(n: int, doc_prefix: str = "doc") -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"{doc_prefix}_{i}",
            doc_id=doc_prefix,
            text=f"content of chunk {i}",
        )
        for i in range(n)
    ]


class TestHeuristicSampler:
    def test_random_returns_requested_count(self):
        sampler = HeuristicNegativeSampler(strategy="random", seed=42)
        chunks = _make_chunks(10)
        result = sampler.sample("test query", ["doc_0"], chunks, 3)
        assert len(result) == 3
        assert all(c.chunk_id != "doc_0" for c in result)

    def test_random_excludes_positives(self):
        sampler = HeuristicNegativeSampler(strategy="random", seed=42)
        chunks = _make_chunks(5)
        result = sampler.sample("test", ["doc_0", "doc_1"], chunks, 10)
        ids = {c.chunk_id for c in result}
        assert "doc_0" not in ids
        assert "doc_1" not in ids

    def test_in_doc_swap_prefers_same_doc(self):
        sampler = HeuristicNegativeSampler(strategy="in_doc_swap", seed=42)
        chunks = _make_chunks(5, "docA") + _make_chunks(5, "docB")
        result = sampler.sample("test", ["docA_0"], chunks, 3)
        assert len(result) == 3

    def test_empty_pool(self):
        sampler = HeuristicNegativeSampler(strategy="random", seed=42)
        result = sampler.sample("test", [], [], 3)
        assert result == []

    def test_seed_reproducibility(self):
        s1 = HeuristicNegativeSampler(strategy="random", seed=123)
        s2 = HeuristicNegativeSampler(strategy="random", seed=123)
        chunks = _make_chunks(20)
        r1 = s1.sample("test", ["doc_0"], chunks, 5)
        r2 = s2.sample("test", ["doc_0"], chunks, 5)
        assert [c.chunk_id for c in r1] == [c.chunk_id for c in r2]


class TestBuildSampler:
    def test_default_random(self):
        sampler = build_sampler({})
        assert isinstance(sampler, HeuristicNegativeSampler)

    def test_from_config(self):
        sampler = build_sampler({
            "strategy": "in_doc_swap",
            "seed": 99,
        })
        assert isinstance(sampler, HeuristicNegativeSampler)
        assert sampler.strategy == "in_doc_swap"
