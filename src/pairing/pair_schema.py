"""Pydantic schema for candidate element pairs.

CandidatePair is the single source of truth for pair data passed between
pairing selectors, enrichment scripts, and query generators.  It replaces
the ad-hoc dict format previously used in hub_candidates_enriched_v3.json.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator


class ElementDetail(BaseModel):
    """Detail blob attached to each endpoint element."""

    element_id: str
    element_type: str  # "figure" | "table" | "formula"
    caption: str = ""
    content: str = ""
    image_path: str = ""
    context_before: str = ""
    context_after: str = ""
    enriched_title: str = ""
    enriched_content: str = ""
    enriched_metadata: Dict[str, Any] = {}


class CandidatePair(BaseModel):
    """A pair of multimodal elements suitable for query generation.

    All fields mirror the enriched pair format so downstream consumers
    (``generate_multihop_l1_queries.py``) can use this without changes.
    """

    pair_id: str
    doc_id: str
    element_a_id: str
    element_b_id: str
    element_a_type: str
    element_b_type: str
    pair_type: str  # sorted, e.g. "figure+table"

    hop_distance: int = 1
    path: List[str] = []
    quality_score: float = 0.5

    element_a: Optional[ElementDetail] = None
    element_b: Optional[ElementDetail] = None
    node_group: List[ElementDetail] = []

    edge_contexts: List[Dict[str, Any]] = []
    hub_semantic_summary: str = ""

    strategy: str = ""  # which strategy produced this pair

    hub_metadata: Dict[str, Any] = {}

    @field_validator("pair_type", mode="before")
    @classmethod
    def _sort_pair_type(cls, v: str) -> str:
        parts = v.split("+")
        return "+".join(sorted(parts))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict compatible with enriched pair format."""
        d = self.model_dump()
        # Convert ElementDetail to plain dicts
        for key in ("element_a", "element_b"):
            if d[key] is not None:
                d[key] = dict(d[key])
        d["node_group"] = [dict(e) for e in d["node_group"]]
        return d
