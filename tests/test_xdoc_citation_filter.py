"""Tests for strict G11 filtering of MinerU xdoc citation edges."""

from __future__ import annotations

from src.pairing.xdoc_citation_filter import CitationFilterConfig, should_keep_edge


def _edge(**kw):
    edge = {
        "source_doc": "1111.1111",
        "target_doc": "2222.2222",
        "section_title": "2 Related Work",
        "chunk_text": "Prior work compares to the target method.",
        "probability": 0.91,
        "features": {
            "cite_pattern": 0.0,
            "title_match": 0.0,
            "text_sim": 0.62,
            "position": 0.4,
        },
    }
    for key, value in kw.items():
        if key == "features":
            edge["features"].update(value)
        else:
            edge[key] = value
    return edge


def test_keeps_structural_title_match_body_edge():
    keep, reason, meta = should_keep_edge(_edge(features={"title_match": 0.35}))
    assert keep is True
    assert reason == "structural_citation"
    assert meta["section_bucket"] == "body"
    assert meta["structural_evidence"] is True


def test_drops_references_by_default():
    keep, reason, _ = should_keep_edge(
        _edge(section_title="References", features={"title_match": 0.8})
    )
    assert keep is False
    assert reason == "references_section"


def test_can_keep_references_when_requested():
    config = CitationFilterConfig(body_only=False)
    keep, reason, _ = should_keep_edge(
        _edge(section_title="References", features={"title_match": 0.8}),
        config,
    )
    assert keep is True
    assert reason == "structural_citation"


def test_drops_acknowledgement_edges_even_with_high_probability():
    keep, reason, _ = should_keep_edge(
        _edge(section_title="Acknowledgements", probability=0.99, features={"title_match": 0.8})
    )
    assert keep is False
    assert reason == "noisy_section"


def test_drops_semantic_neighbor_without_citation_evidence():
    keep, reason, _ = should_keep_edge(_edge(features={"text_sim": 0.68}))
    assert keep is False
    assert reason == "semantic_without_citation_evidence"


def test_semantic_high_conf_is_opt_in():
    edge = _edge(probability=0.99, features={"text_sim": 0.82})
    keep, reason, _ = should_keep_edge(edge)
    assert keep is False
    assert reason == "semantic_without_citation_evidence"

    config = CitationFilterConfig(keep_semantic_high_conf=True)
    keep, reason, meta = should_keep_edge(edge, config)
    assert keep is True
    assert reason == "semantic_high_conf"
    assert meta["structural_evidence"] is False
