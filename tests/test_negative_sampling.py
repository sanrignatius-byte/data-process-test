"""Tests for src/sampling/negative_sampler.py — negative sampling strategies."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.models import Chunk
from src.sampling.negative_sampler import (
    GraphAwareNegativeSampler,
    HeuristicNegativeSampler,
    _build_adjacency,
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
        # All 3 should come from docA (4 remaining same-doc chunks >= n=3)
        assert all(c.doc_id == "docA" for c in result)

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


# ── Helpers for graph-aware tests ─────────────────────────────────────────────

def _hub_candidates_fixture(tmp_path: Path) -> Path:
    """Write a minimal hub_candidates JSON with known adjacency."""
    data = {
        "metadata": {},
        "summary": {},
        "pairs": [
            {
                "pair_id": "p1",
                "element_a_id": "1234.56789_figure_1",
                "element_b_id": "9876.54321_table_2",
                "path": ["1234.56789_figure_1", "1234.56789::p::00001", "9876.54321_table_2"],
            }
        ],
        "adjacent_bridge_elements": [],
        "adjacent_bridge_adjacency": [
            {
                "doc_id": "1234.56789",
                "elements_i": ["1234.56789_figure_1"],
                "elements_j": ["1234.56789_formula_3"],
            }
        ],
    }
    p = tmp_path / "hub_candidates.json"
    p.write_text(json.dumps(data))
    return p


def _make_graph_chunks() -> list[Chunk]:
    """Chunks that match the fixture's element IDs."""
    ids = [
        ("1234.56789_figure_1", "1234.56789"),
        ("1234.56789_formula_3", "1234.56789"),   # intra-doc neighbour
        ("9876.54321_table_2", "9876.54321"),      # cross-doc neighbour
        ("9999.11111_figure_5", "9999.11111"),     # unrelated
        ("9999.11111_table_1", "9999.11111"),      # unrelated
    ]
    return [Chunk(chunk_id=cid, doc_id=did, text=f"text {cid}") for cid, did in ids]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBuildAdjacency:
    def test_intra_doc_edge(self, tmp_path):
        path = _hub_candidates_fixture(tmp_path)
        adj = _build_adjacency(path)
        assert "1234.56789_formula_3" in adj["1234.56789_figure_1"]
        assert "1234.56789_figure_1" in adj["1234.56789_formula_3"]

    def test_cross_doc_edge_from_path(self, tmp_path):
        path = _hub_candidates_fixture(tmp_path)
        adj = _build_adjacency(path)
        # Hub node is filtered out; consecutive element nodes in the filtered
        # list become direct edges: figure_1 ↔ table_2.
        # This is intentional — path endpoints make good cross-doc hard negatives.
        assert "9876.54321_table_2" in adj.get("1234.56789_figure_1", set())
        assert "1234.56789_figure_1" in adj.get("9876.54321_table_2", set())

    def test_missing_file_returns_empty(self, tmp_path):
        adj = _build_adjacency(tmp_path / "nonexistent.json")
        assert adj == {}


class TestGraphAwareNegativeSampler:
    def test_prefers_graph_neighbours(self, tmp_path):
        path = _hub_candidates_fixture(tmp_path)
        sampler = GraphAwareNegativeSampler(hub_candidates_path=path, seed=0)
        chunks = _make_graph_chunks()
        # positive = figure_1; graph neighbours = formula_3 (intra-doc) + table_2 (cross-doc via path)
        result = sampler.sample("q", ["1234.56789_figure_1"], chunks, 2)
        assert len(result) == 2
        result_ids = {c.chunk_id for c in result}
        # Both graph neighbours should be selected (no need for random padding)
        assert result_ids == {"1234.56789_formula_3", "9876.54321_table_2"}

    def test_excludes_positives(self, tmp_path):
        path = _hub_candidates_fixture(tmp_path)
        sampler = GraphAwareNegativeSampler(hub_candidates_path=path, seed=0)
        chunks = _make_graph_chunks()
        result = sampler.sample("q", ["1234.56789_figure_1"], chunks, 4)
        ids = {c.chunk_id for c in result}
        assert "1234.56789_figure_1" not in ids

    def test_pads_with_random_when_not_enough_neighbours(self, tmp_path):
        path = _hub_candidates_fixture(tmp_path)
        sampler = GraphAwareNegativeSampler(hub_candidates_path=path, seed=0)
        chunks = _make_graph_chunks()
        # Only 1 graph neighbour (formula_3), request 3 → should pad with randoms
        result = sampler.sample("q", ["1234.56789_figure_1"], chunks, 3)
        assert len(result) == 3

    def test_fallback_no_graph(self, tmp_path):
        """No hub candidates file → falls back to random."""
        sampler = GraphAwareNegativeSampler(hub_candidates_path=tmp_path / "missing.json", seed=0)
        assert not sampler.has_graph
        chunks = _make_graph_chunks()
        result = sampler.sample("q", ["1234.56789_figure_1"], chunks, 2)
        assert len(result) == 2

    def test_build_sampler_graph_aware(self, tmp_path):
        path = _hub_candidates_fixture(tmp_path)
        sampler = build_sampler({"strategy": "graph_aware", "hub_candidates_path": str(path)})
        assert isinstance(sampler, GraphAwareNegativeSampler)
        assert sampler.has_graph
