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
