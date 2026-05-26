"""Filters for MinerU cross-document citation edges.

The C18 predictor is intentionally recall-oriented: it emits useful citation
candidates, but also keeps references-list matches and semantic neighbours with
no citation evidence.  M4 chain construction needs a stricter paragraph-bridge
backbone, so this module applies the G11 noise filters before downstream
pairing consumes those edges.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, Iterator, Tuple


REFERENCE_SECTION_RE = re.compile(r"\b(references?|bibliography|works cited)\b", re.I)
NOISY_SECTION_RE = re.compile(
    r"\b(acknowledg(?:e)?ments?|funding|financial support|author contributions?|"
    r"affiliations?|contributors?)\b",
    re.I,
)
AUTHOR_LIST_HINT_RE = re.compile(
    r"(@|university|department|institute|laborator(?:y|ies)|school of|"
    r"\b[a-z]+@[a-z0-9.-]+\.[a-z]{2,}\b)",
    re.I,
)


@dataclass(frozen=True)
class CitationFilterConfig:
    """Configuration for strict cross-doc citation filtering."""

    min_probability: float = 0.5
    min_title_match: float = 0.2
    min_semantic_text_sim: float = 0.75
    min_semantic_probability: float = 0.95
    body_only: bool = True
    keep_semantic_high_conf: bool = False


def _features(edge: Dict[str, Any]) -> Dict[str, Any]:
    raw = edge.get("features")
    return raw if isinstance(raw, dict) else {}


def section_bucket(edge: Dict[str, Any]) -> str:
    """Classify the edge's source section for filtering and reporting."""
    section_title = str(edge.get("section_title") or edge.get("source_section") or "")
    chunk_text = str(edge.get("chunk_text") or edge.get("source_text") or "")
    head_text = chunk_text[:400]
    position = float((_features(edge).get("position") or 0.0))

    if REFERENCE_SECTION_RE.search(section_title) or REFERENCE_SECTION_RE.search(head_text[:160]):
        return "references"
    if NOISY_SECTION_RE.search(section_title) or NOISY_SECTION_RE.search(head_text):
        return "noisy_section"
    if position <= 0.04 and AUTHOR_LIST_HINT_RE.search(head_text):
        return "author_list"
    return "body"


def structural_evidence(edge: Dict[str, Any], config: CitationFilterConfig) -> bool:
    """Return True when the edge has direct citation/title evidence."""
    f = _features(edge)
    cite_pattern = float(f.get("cite_pattern") or 0.0)
    title_match = float(f.get("title_match") or 0.0)
    return cite_pattern > 0.0 or title_match >= config.min_title_match


def should_keep_edge(
    edge: Dict[str, Any],
    config: CitationFilterConfig = CitationFilterConfig(),
) -> Tuple[bool, str, Dict[str, Any]]:
    """Decide whether an inferred citation edge is chain-backbone quality.

    Returns ``(keep, reason, metadata)``.  ``reason`` is a drop reason when
    ``keep`` is false, and an evidence tier when ``keep`` is true.
    """
    source_doc = str(edge.get("source_doc") or "")
    target_doc = str(edge.get("target_doc") or "")
    if not source_doc or not target_doc:
        return False, "missing_doc_id", {}
    if source_doc == target_doc:
        return False, "self_edge", {}

    probability = float(edge.get("probability") or 0.0)
    if probability < config.min_probability:
        return False, "low_probability", {"probability": probability}

    bucket = section_bucket(edge)
    if bucket in {"noisy_section", "author_list"}:
        return False, bucket, {"section_bucket": bucket}
    if config.body_only and bucket == "references":
        return False, "references_section", {"section_bucket": bucket}

    f = _features(edge)
    text_sim = float(f.get("text_sim") or 0.0)
    has_structure = structural_evidence(edge, config)
    metadata = {
        "filter_version": "g11_v1",
        "section_bucket": bucket,
        "structural_evidence": has_structure,
    }

    if has_structure:
        return True, "structural_citation", metadata

    if (
        config.keep_semantic_high_conf
        and text_sim >= config.min_semantic_text_sim
        and probability >= config.min_semantic_probability
    ):
        metadata["semantic_text_sim"] = text_sim
        return True, "semantic_high_conf", metadata

    return False, "semantic_without_citation_evidence", metadata


def filter_edges(
    edges: Iterable[Dict[str, Any]],
    config: CitationFilterConfig = CitationFilterConfig(),
) -> Tuple[Iterator[Dict[str, Any]], Counter, Counter]:
    """Yield kept edges and collect kept/drop counters.

    The returned iterator closes over the counters.  Consume it before reading
    the final counter values.
    """
    kept_by_tier: Counter = Counter()
    dropped_by_reason: Counter = Counter()

    def _iter() -> Iterator[Dict[str, Any]]:
        for edge in edges:
            keep, reason, metadata = should_keep_edge(edge, config)
            if not keep:
                dropped_by_reason[reason] += 1
                continue
            kept_by_tier[reason] += 1
            out = dict(edge)
            out["filter_metadata"] = metadata | {"evidence_tier": reason}
            yield out

    return _iter(), kept_by_tier, dropped_by_reason
