"""Unified training data models for the contrastive learning pipeline.

These Pydantic models enforce schema consistency across all difficulty levels
(L1/L2/L3) and serve as the single source of truth for:
  - query normalisation  (scripts/normalize_queries.py)
  - triplet construction (scripts/build_dual_evidence_triplets.py)
  - dataset export       (scripts/export_training_data.py)

Schema version is bumped whenever fields are added/removed so that downstream
consumers can detect and handle format changes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Constants ─────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "1.0.0"


# ── Enums ─────────────────────────────────────────────────────────────────────

class DifficultyLevel(str, Enum):
    """Difficulty level of a query."""

    L1 = "L1"  # single-element intra-doc
    L2 = "L2"  # dual-evidence / cross-doc
    L3 = "L3"  # multi-hop reasoning chain


# ── Evidence ──────────────────────────────────────────────────────────────────

class EvidenceSpan(BaseModel):
    """A single piece of evidence grounding a query answer."""

    element_id: str
    doc_id: str = ""
    element_type: str = ""          # "figure" / "table" / "formula" / "text"
    span_text: str = ""             # the textual excerpt used as evidence
    evidence_type: str = ""         # "observation" / "data" / "attribution" / …
    image_path: Optional[str] = None

    @field_validator("element_id")
    @classmethod
    def element_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("element_id must be a non-empty string")
        return v


# ── Reasoning Step ────────────────────────────────────────────────────────────

class ReasoningStep(BaseModel):
    """A single step in a multi-hop reasoning chain (L3 queries)."""

    step_id: int
    evidence_element_id: str = ""
    evidence_type: str = ""
    evidence_span: str = ""
    reasoning_role: str = ""        # "premise" / "bridge" / "conclusion"
    depends_on_steps: List[int] = Field(default_factory=list)
    produces_claim: str = ""


# ── Standard Query ────────────────────────────────────────────────────────────

class StandardQuery(BaseModel):
    """Unified query schema shared by L1 / L2 / L3.

    Every query produced or consumed by the training pipeline MUST be
    representable as a ``StandardQuery``.  Legacy JSONL files are converted
    via ``scripts/normalize_queries.py``.
    """

    schema_version: str = SCHEMA_VERSION

    # ── identity
    query_id: str
    query_text: str
    answer_text: str

    # ── difficulty
    difficulty_level: DifficultyLevel

    # ── evidence
    evidence_spans: List[EvidenceSpan] = Field(default_factory=list)

    # ── multi-hop (L3)
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list)

    # ── generation metadata (optional, preserved from upstream)
    doc_id: str = ""
    pair_id: str = ""
    pair_type: str = ""             # "figure+table", "figure+formula", …
    element_ids: List[str] = Field(default_factory=list)
    image_paths: List[str] = Field(default_factory=list)
    query_style: str = "academic"
    query_type: str = ""
    persona: str = ""
    hop_distance: int = 0

    # ── QC metadata
    qc_pass: bool = True
    qc_issues: List[str] = Field(default_factory=list)

    # ── catch-all for extra upstream fields
    extra: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_spans")
    @classmethod
    def at_least_one_evidence(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("must have ≥1 evidence span")
        return v

    @model_validator(mode="after")
    def reasoning_steps_required_for_l3(self) -> "StandardQuery":
        if self.difficulty_level == DifficultyLevel.L3 and not self.reasoning_steps:
            raise ValueError("L3 queries must have at least one reasoning step")
        return self


# ── Triplet ───────────────────────────────────────────────────────────────────

class Triplet(BaseModel):
    """A training triplet for contrastive learning.

    ``positive`` and ``hard_negatives`` are lists of ``EvidenceSpan``-like
    chunk references so the downstream DataLoader can resolve texts/images.
    """

    schema_version: str = SCHEMA_VERSION

    query_id: str
    query_text: str
    difficulty_level: DifficultyLevel

    positive: List[EvidenceSpan]
    hard_negatives: List[EvidenceSpan] = Field(default_factory=list)

    negative_strategy: str = ""     # "in_doc_swap" / "same_type_hard" / …
    difficulty_score: float = 0.0

    # ── provenance
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("positive")
    @classmethod
    def at_least_one_positive(cls, v: list) -> list:
        if len(v) == 0:
            raise ValueError("triplet must have ≥1 positive evidence")
        return v
