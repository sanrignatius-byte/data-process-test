"""Intra-document element pairing for multi-hop query generation.

Provides reusable tools for selecting pairs of multimodal elements
within a single document, connected by cross-reference edges or
shared section membership. All strategies enforce strict document
boundary checks — no cross-document pairs are ever produced.

Usage::

    from src.pairing import IntraDocPairSelector, CandidatePair

    selector = IntraDocPairSelector.from_file("data/01_graphs/multimodal_elements.json")
    pairs = selector.select(strategy="all", max_per_doc=10)
"""

from src.pairing.pair_schema import CandidatePair  # noqa: F401
from src.pairing.intra_doc_pairs import IntraDocPairSelector  # noqa: F401
