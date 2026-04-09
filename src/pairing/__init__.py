"""Intra-document element pairing for multi-hop query generation.

Provides reusable tools for selecting pairs of multimodal elements
within a single document, connected by cross-reference edges or
multi-hop chains.  All strategies enforce strict document boundary
checks — no cross-document pairs are ever produced.

Components:

- **CandidatePair** — Pydantic schema for a pair of elements.
- **IntraDocPairSelector** — selector with direct / 2hop / section / chain
  strategies.
- **ChainFinder** — graph traversal for discovering long multi-hop chains.
- **dedup_context** — removes overlap between context_before / context_after.

Usage::

    from src.pairing import IntraDocPairSelector, CandidatePair, ChainFinder

    selector = IntraDocPairSelector.from_file("data/01_graphs/multimodal_elements.json")
    pairs = selector.select(strategy="all", max_per_doc=10)
"""

from src.pairing.pair_schema import CandidatePair  # noqa: F401
from src.pairing.intra_doc_pairs import IntraDocPairSelector  # noqa: F401
from src.pairing.chain_finder import ChainFinder, ChainResult  # noqa: F401
from src.pairing.context_dedup import dedup_context  # noqa: F401
