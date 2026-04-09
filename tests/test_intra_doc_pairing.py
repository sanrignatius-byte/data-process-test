"""Tests for src.pairing — intra-doc pair selection."""

from __future__ import annotations

import pytest

from src.pairing.pair_schema import CandidatePair, ElementDetail
from src.pairing.intra_doc_pairs import (
    IntraDocPairSelector,
    _make_pair_type,
    _doc_id_from_element_id,
    _build_hub_summary,
    MODAL_TYPES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_element(doc_id: str, etype: str, num: int, **kw) -> dict:
    eid = f"{doc_id}_{etype}_{num}"
    return {
        "element_id": eid,
        "doc_id": doc_id,
        "element_type": etype,
        "caption": kw.get("caption", f"{etype.title()} {num}"),
        "content": kw.get("content", ""),
        "image_path": kw.get("image_path", ""),
        "context_before": kw.get("context_before", ""),
        "context_after": kw.get("context_after", ""),
    }


def _make_edge(src: str, tgt: str, ref: str = "") -> dict:
    return {
        "source_id": src,
        "target_id": tgt,
        "source_type": "",
        "target_type": "",
        "ref_text": ref,
        "context_snippet": "",
    }


def _single_doc_data() -> dict:
    """One document with 3 elements (fig, tbl, formula) and 2 edges."""
    doc_id = "1234.5678"
    fig = _make_element(doc_id, "figure", 1, caption="Figure 1: results")
    tbl = _make_element(doc_id, "table", 1, caption="Table 1: metrics")
    formula = _make_element(doc_id, "formula", 1, caption="Eq 1")
    edges = [
        _make_edge(f"{doc_id}_figure_1", f"{doc_id}_table_1", ref="see Table 1"),
        _make_edge(f"{doc_id}_table_1", f"{doc_id}_formula_1", ref="from Eq 1"),
    ]
    return {
        doc_id: {
            "doc_id": doc_id,
            "elements": {
                f"{doc_id}_figure_1": fig,
                f"{doc_id}_table_1": tbl,
                f"{doc_id}_formula_1": formula,
            },
            "edges": edges,
            "multimodal_pairs": [
                {
                    "pair_id": f"{doc_id}_pair_1",
                    "doc_id": doc_id,
                    "element_a_id": f"{doc_id}_figure_1",
                    "element_b_id": f"{doc_id}_table_1",
                    "element_a_type": "figure",
                    "element_b_type": "table",
                    "hop_distance": 1,
                    "path": [f"{doc_id}_figure_1", f"{doc_id}_table_1"],
                    "relationship": "direct_reference",
                    "quality_score": 1.0,
                    "metadata": {},
                },
            ],
        }
    }


def _two_doc_data() -> dict:
    """Two documents — verifies no cross-doc pairs leak."""
    d1 = "1111.1111"
    d2 = "2222.2222"
    return {
        d1: {
            "doc_id": d1,
            "elements": {
                f"{d1}_figure_1": _make_element(d1, "figure", 1),
                f"{d1}_table_1": _make_element(d1, "table", 1),
            },
            "edges": [_make_edge(f"{d1}_figure_1", f"{d1}_table_1", "Fig 1 vs Table 1")],
            "multimodal_pairs": [],
        },
        d2: {
            "doc_id": d2,
            "elements": {
                f"{d2}_figure_1": _make_element(d2, "figure", 1),
                f"{d2}_formula_1": _make_element(d2, "formula", 1),
            },
            "edges": [_make_edge(f"{d2}_figure_1", f"{d2}_formula_1", "Fig 1 and Eq 1")],
            "multimodal_pairs": [],
        },
    }


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_make_pair_type_sorted(self):
        assert _make_pair_type("table", "figure") == "figure+table"
        assert _make_pair_type("formula", "figure") == "figure+formula"

    def test_doc_id_from_element_id(self):
        assert _doc_id_from_element_id("1306.5204_figure_2") == "1306.5204"
        assert _doc_id_from_element_id("1802.08139_table_1") == "1802.08139"
        assert _doc_id_from_element_id("1802.08139_formula_3") == "1802.08139"
        assert _doc_id_from_element_id("unknown_format") == ""

    def test_build_hub_summary(self):
        el_a = {"caption": "Figure 1", "element_type": "figure"}
        el_b = {"caption": "Table 1", "element_type": "table"}
        s = _build_hub_summary(el_a, el_b, "see Table 1")
        assert "FIGURE A" in s
        assert "TABLE B" in s
        assert "BRIDGE" in s


# ---------------------------------------------------------------------------
# Unit tests: CandidatePair schema
# ---------------------------------------------------------------------------

class TestCandidatePairSchema:
    def test_pair_type_auto_sorted(self):
        p = CandidatePair(
            pair_id="test_1",
            doc_id="doc",
            element_a_id="a",
            element_b_id="b",
            element_a_type="table",
            element_b_type="figure",
            pair_type="table+figure",
        )
        assert p.pair_type == "figure+table"

    def test_to_dict_roundtrip(self):
        p = CandidatePair(
            pair_id="test_1",
            doc_id="doc",
            element_a_id="a",
            element_b_id="b",
            element_a_type="figure",
            element_b_type="table",
            pair_type="figure+table",
            quality_score=0.9,
        )
        d = p.to_dict()
        assert d["pair_id"] == "test_1"
        assert d["quality_score"] == 0.9

    def test_element_detail(self):
        ed = ElementDetail(element_id="x", element_type="figure")
        assert ed.caption == ""
        assert ed.enriched_title == ""


# ---------------------------------------------------------------------------
# Integration tests: IntraDocPairSelector
# ---------------------------------------------------------------------------

class TestDirectStrategy:
    def test_finds_direct_pairs(self):
        sel = IntraDocPairSelector(_single_doc_data())
        pairs = sel.select(strategy="direct")
        assert len(pairs) >= 2  # fig→tbl and tbl→formula
        for p in pairs:
            assert p.hop_distance == 1
            assert p.strategy == "direct"
            assert p.hub_metadata.get("is_cross_doc") is False

    def test_cross_modal_only(self):
        """Direct pairs must be cross-modal (different element types)."""
        sel = IntraDocPairSelector(_single_doc_data())
        pairs = sel.select(strategy="direct")
        for p in pairs:
            assert p.element_a_type != p.element_b_type

    def test_no_cross_doc_leak(self):
        """Two separate docs should never produce cross-doc pairs."""
        sel = IntraDocPairSelector(_two_doc_data())
        pairs = sel.select(strategy="direct")
        for p in pairs:
            da = _doc_id_from_element_id(p.element_a_id)
            db = _doc_id_from_element_id(p.element_b_id)
            assert da == db, f"Cross-doc leak: {p.element_a_id} vs {p.element_b_id}"


class TestTwoHopStrategy:
    def test_finds_2hop_pairs(self):
        sel = IntraDocPairSelector(_single_doc_data())
        pairs = sel.select(strategy="2hop")
        # fig→tbl→formula gives figure↔formula via table bridge
        assert len(pairs) >= 1
        for p in pairs:
            assert p.hop_distance == 2
            assert p.strategy == "2hop"

    def test_dedup(self):
        sel = IntraDocPairSelector(_single_doc_data())
        pairs = sel.select(strategy="2hop")
        keys = [frozenset([p.element_a_id, p.element_b_id]) for p in pairs]
        assert len(keys) == len(set(keys)), "Duplicate pairs found"


class TestSectionStrategy:
    def test_uses_multimodal_pairs(self):
        sel = IntraDocPairSelector(_single_doc_data())
        pairs = sel.select(strategy="section")
        # The fixture has 1 multimodal_pair
        assert len(pairs) >= 1

    def test_quality_score_set(self):
        sel = IntraDocPairSelector(_single_doc_data())
        pairs = sel.select(strategy="section")
        for p in pairs:
            assert 0 < p.quality_score <= 1.0


class TestAllStrategy:
    def test_all_combines_strategies(self):
        sel = IntraDocPairSelector(_single_doc_data())
        pairs = sel.select(strategy="all")
        strategies = {p.strategy for p in pairs}
        assert "direct" in strategies
        # 2hop and section may or may not produce unique pairs

    def test_max_per_doc(self):
        sel = IntraDocPairSelector(_single_doc_data())
        pairs = sel.select(strategy="all", max_per_doc=2)
        assert len(pairs) <= 2

    def test_pair_type_filter(self):
        sel = IntraDocPairSelector(_single_doc_data())
        pairs = sel.select(strategy="all", pair_types={"figure+table"})
        for p in pairs:
            assert p.pair_type == "figure+table"

    def test_multi_doc_isolation(self):
        """Pairs from different docs are independent."""
        sel = IntraDocPairSelector(_two_doc_data())
        pairs = sel.select(strategy="all")
        docs = {p.doc_id for p in pairs}
        assert len(docs) == 2

    def test_never_cross_doc(self):
        """Absolute guarantee: no pair ever crosses document boundaries."""
        sel = IntraDocPairSelector(_two_doc_data())
        pairs = sel.select(strategy="all")
        for p in pairs:
            da = _doc_id_from_element_id(p.element_a_id)
            db = _doc_id_from_element_id(p.element_b_id)
            assert da == db


class TestStats:
    def test_stats_structure(self):
        sel = IntraDocPairSelector(_single_doc_data())
        s = sel.stats()
        assert s["documents"] == 1
        assert s["elements"] == 3
        assert s["edges"] == 2
        assert "figure" in s["type_distribution"]


class TestRealData:
    """Test with actual multimodal_elements.json if available."""

    @pytest.fixture
    def real_selector(self):
        import os
        path = "data/01_graphs/multimodal_elements.json"
        if not os.path.exists(path):
            pytest.skip("multimodal_elements.json not available")
        return IntraDocPairSelector.from_file(path)

    def test_no_cross_doc_in_real_data(self, real_selector):
        pairs = real_selector.select(strategy="all", max_per_doc=5)
        for p in pairs:
            da = _doc_id_from_element_id(p.element_a_id)
            db = _doc_id_from_element_id(p.element_b_id)
            assert da == db, f"Cross-doc leak: {p.element_a_id} vs {p.element_b_id}"

    def test_real_data_produces_pairs(self, real_selector):
        pairs = real_selector.select(strategy="direct", max_per_doc=10)
        assert len(pairs) > 0, "Expected at least some direct pairs from real data"


# ---------------------------------------------------------------------------
# ChainFinder tests
# ---------------------------------------------------------------------------

from src.pairing.chain_finder import ChainFinder, ChainResult, _score_chain


def _chain_doc_data() -> dict:
    """Document with 5 elements forming a chain: fig1→tbl1→fig2→formula1→tbl2."""
    doc_id = "5555.5555"
    elems = {
        f"{doc_id}_figure_1": _make_element(doc_id, "figure", 1),
        f"{doc_id}_table_1": _make_element(doc_id, "table", 1),
        f"{doc_id}_figure_2": _make_element(doc_id, "figure", 2),
        f"{doc_id}_formula_1": _make_element(doc_id, "formula", 1),
        f"{doc_id}_table_2": _make_element(doc_id, "table", 2),
    }
    edges = [
        _make_edge(f"{doc_id}_figure_1", f"{doc_id}_table_1", "see Table 1"),
        _make_edge(f"{doc_id}_table_1", f"{doc_id}_figure_2", "cf. Figure 2"),
        _make_edge(f"{doc_id}_figure_2", f"{doc_id}_formula_1", "using Eq 1"),
        _make_edge(f"{doc_id}_formula_1", f"{doc_id}_table_2", "results in Table 2"),
    ]
    return {
        "doc_id": doc_id,
        "elements": elems,
        "edges": edges,
        "multimodal_pairs": [],
    }


class TestChainFinder:
    def test_finds_chains(self):
        finder = ChainFinder.from_doc(_chain_doc_data())
        chains = finder.find_chains(min_length=3)
        assert len(chains) > 0
        for c in chains:
            assert c.hop_count >= 2

    def test_finds_longest_chain(self):
        finder = ChainFinder.from_doc(_chain_doc_data())
        chains = finder.find_longest(top_k=1)
        assert len(chains) == 1
        assert chains[0].hop_count == 4  # 5 nodes = 4 hops

    def test_longest_chain_path(self):
        finder = ChainFinder.from_doc(_chain_doc_data())
        chains = finder.find_longest(top_k=1)
        chain = chains[0]
        assert len(chain.path) == 5
        assert len(chain.modality_sequence) == 5

    def test_cross_modal_only(self):
        finder = ChainFinder.from_doc(_chain_doc_data())
        chains = finder.find_endpoint_pairs(min_hops=2, cross_modal_only=True)
        for c in chains:
            assert c.modality_sequence[0] != c.modality_sequence[-1], \
                f"Same modality endpoints: {c.modality_sequence}"

    def test_endpoint_dedup(self):
        finder = ChainFinder.from_doc(_chain_doc_data())
        chains = finder.find_endpoint_pairs(min_hops=1)
        endpoints = [frozenset(c.endpoints) for c in chains]
        assert len(endpoints) == len(set(endpoints)), "Duplicate endpoint pairs"

    def test_score_increases_with_length(self):
        s1 = _score_chain(["a", "b"], ["figure", "table"])
        s2 = _score_chain(["a", "b", "c", "d"], ["figure", "table", "formula", "figure"])
        assert s2 > s1

    def test_score_rewards_diversity(self):
        s1 = _score_chain(["a", "b", "c"], ["figure", "figure", "figure"])
        s2 = _score_chain(["a", "b", "c"], ["figure", "table", "formula"])
        assert s2 > s1

    def test_chain_result_ordering(self):
        c1 = ChainResult(score=0.5, path=("a", "b"), doc_id="x",
                         hop_count=1, modality_sequence=("f", "t"),
                         cross_modal_transitions=1, unique_modalities=2)
        c2 = ChainResult(score=0.8, path=("a", "b", "c"), doc_id="x",
                         hop_count=2, modality_sequence=("f", "t", "f"),
                         cross_modal_transitions=2, unique_modalities=2)
        assert c1 < c2  # score-based ordering

    def test_stats(self):
        finder = ChainFinder.from_doc(_chain_doc_data())
        s = finder.stats()
        assert s["elements"] == 5
        assert s["edges"] == 4
        assert s["connected_components"] == 1

    def test_disconnected_graph(self):
        """Two disconnected components should not form cross-component chains."""
        doc_id = "6666.6666"
        elems = {
            f"{doc_id}_figure_1": _make_element(doc_id, "figure", 1),
            f"{doc_id}_table_1": _make_element(doc_id, "table", 1),
            f"{doc_id}_figure_2": _make_element(doc_id, "figure", 2),
            f"{doc_id}_formula_1": _make_element(doc_id, "formula", 1),
        }
        edges = [
            _make_edge(f"{doc_id}_figure_1", f"{doc_id}_table_1"),
            _make_edge(f"{doc_id}_figure_2", f"{doc_id}_formula_1"),
        ]
        doc = {
            "doc_id": doc_id, "elements": elems,
            "edges": edges, "multimodal_pairs": [],
        }
        finder = ChainFinder.from_doc(doc)
        assert finder.stats()["connected_components"] == 2
        chains = finder.find_chains(min_length=3)
        assert len(chains) == 0  # no 3-node chain possible


# ---------------------------------------------------------------------------
# Context dedup tests
# ---------------------------------------------------------------------------

from src.pairing.context_dedup import dedup_context, _common_prefix_length, _fast_similarity


class TestContextDedup:
    def test_identical_strings(self):
        text = "This is a paragraph about machine learning. " * 5
        text = text.strip()  # dedup_context strips inputs
        before, after = dedup_context(text, text)
        # One should be cleared
        assert (before == "" or after == "")
        assert (before == text or after == text)

    def test_prefix_overlap(self):
        shared = "This is shared context about fairness. " * 3
        unique_tail = " This part is unique to after context only."
        combined = shared + unique_tail
        before, after = dedup_context(shared.strip(), combined.strip())
        # The shared prefix should be detected; either before is kept and
        # after trimmed, or entire before is absorbed into after.
        total_len = len(before) + len(after)
        assert total_len < len(shared.strip()) + len(combined.strip()), \
            "Expected dedup to reduce total text"

    def test_no_overlap(self):
        before = "Methods section: We use gradient descent for optimization."
        after = "Results section: The model achieves 95% accuracy on test set."
        b, a = dedup_context(before, after)
        assert b == before
        assert a == after

    def test_short_strings_untouched(self):
        before, after = dedup_context("short", "text")
        assert before == "short"
        assert after == "text"

    def test_empty_strings(self):
        assert dedup_context("", "") == ("", "")
        assert dedup_context("hello", "") == ("hello", "")
        assert dedup_context("", "world") == ("", "world")

    def test_common_prefix_length(self):
        assert _common_prefix_length("abcdef", "abcxyz") == 3
        assert _common_prefix_length("hello", "hello world") == 5
        assert _common_prefix_length("abc", "xyz") == 0

    def test_fast_similarity_identical(self):
        text = "The quick brown fox jumps over the lazy dog"
        assert _fast_similarity(text, text) > 0.99

    def test_fast_similarity_different(self):
        assert _fast_similarity("aaaa", "zzzz") < 0.1

    def test_real_duplicate_pattern(self):
        """Simulate the 1904.03035 pattern: consecutive tables share context."""
        shared = ("The tables show how the scores vary for the training text "
                  "and the generated text for different values of the parameter. "
                  "We observe consistent improvements across all metrics when "
                  "the parameter is increased. This validates our hypothesis.")
        before, after = dedup_context(shared, shared)
        assert before == shared or after == shared
        assert before == "" or after == ""


# ---------------------------------------------------------------------------
# Chain strategy integration tests
# ---------------------------------------------------------------------------

class TestChainStrategy:
    def test_chain_strategy_finds_pairs(self):
        docs = {"5555.5555": _chain_doc_data()}
        sel = IntraDocPairSelector(docs)
        pairs = sel.select(strategy="chain")
        assert len(pairs) > 0
        for p in pairs:
            assert p.strategy == "chain"
            assert p.hop_distance >= 2

    def test_chain_has_full_path(self):
        docs = {"5555.5555": _chain_doc_data()}
        sel = IntraDocPairSelector(docs)
        pairs = sel.select(strategy="chain")
        for p in pairs:
            assert len(p.path) == p.hop_distance + 1
            assert len(p.node_group) >= 2

    def test_chain_endpoints_cross_modal(self):
        docs = {"5555.5555": _chain_doc_data()}
        sel = IntraDocPairSelector(docs)
        pairs = sel.select(strategy="chain")
        for p in pairs:
            assert p.element_a_type != p.element_b_type

    def test_chain_min_hops_filter(self):
        docs = {"5555.5555": _chain_doc_data()}
        sel = IntraDocPairSelector(docs)
        pairs = sel.select(strategy="chain", min_chain_hops=3)
        for p in pairs:
            assert p.hop_distance >= 3

    def test_chain_metadata(self):
        docs = {"5555.5555": _chain_doc_data()}
        sel = IntraDocPairSelector(docs)
        pairs = sel.select(strategy="chain")
        for p in pairs:
            assert "chain_hops" in p.hub_metadata
            assert "modality_sequence" in p.hub_metadata
            assert p.hub_metadata["is_cross_doc"] is False

    def test_all_includes_chains(self):
        docs = {"5555.5555": _chain_doc_data()}
        sel = IntraDocPairSelector(docs)
        pairs = sel.select(strategy="all")
        strategies = {p.strategy for p in pairs}
        assert "chain" in strategies

    def test_real_data_chains(self):
        import os
        path = "data/01_graphs/multimodal_elements.json"
        if not os.path.exists(path):
            pytest.skip("multimodal_elements.json not available")
        sel = IntraDocPairSelector.from_file(path)
        pairs = sel.select(strategy="chain", max_per_doc=5, min_chain_hops=3)
        assert len(pairs) > 0, "Expected chain pairs from real data"
        max_hops = max(p.hop_distance for p in pairs)
        assert max_hops >= 3, f"Expected chains with ≥3 hops, got max {max_hops}"
