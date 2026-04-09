"""Context deduplication for multimodal elements.

When elements are physically adjacent in a document (e.g. three
consecutive tables), their ``context_before`` and ``context_after``
fields can be near-identical because the same surrounding paragraphs
are captured for both directions.

This module detects and removes such overlap, keeping only the unique
portion of each context field.

Typical usage::

    from src.pairing.context_dedup import dedup_context

    before, after = dedup_context(el["context_before"], el["context_after"])
"""

from __future__ import annotations


def dedup_context(
    context_before: str,
    context_after: str,
    *,
    similarity_threshold: float = 0.5,
) -> tuple[str, str]:
    """Remove overlap between context_before and context_after.

    Three overlap patterns are handled:

    1. **Prefix overlap**: ``context_after`` starts with the same text
       as ``context_before`` — strip the shared prefix from
       ``context_after``.
    2. **Full duplicate**: the two are (near-)identical — keep
       ``context_before`` and clear ``context_after``.
    3. **Partial substring**: ``context_before`` is a substring of
       ``context_after`` or vice-versa — keep the longer one and
       clear the shorter.

    Parameters
    ----------
    context_before, context_after : str
        Raw context strings from multimodal_elements.json.
    similarity_threshold : float
        Minimum character-level overlap ratio to trigger dedup.
        Default 0.5 (50%).

    Returns
    -------
    tuple[str, str]
        ``(cleaned_before, cleaned_after)``
    """
    before = (context_before or "").strip()
    after = (context_after or "").strip()

    if not before or not after:
        return before, after

    min_len = min(len(before), len(after))
    if min_len < 30:
        return before, after

    # Pattern 1: Prefix overlap — context_after starts same as context_before
    prefix_match = _common_prefix_length(before, after)
    prefix_ratio = prefix_match / min_len

    if prefix_ratio >= similarity_threshold:
        # They share a long prefix → after is a superset or duplicate
        if prefix_match >= len(after):
            # context_after is entirely contained in context_before
            return before, ""
        if prefix_match >= len(before):
            # context_before is entirely contained in context_after
            return "", after
        # Partial prefix: trim the shared prefix from context_after
        trimmed_after = after[prefix_match:].strip()
        return before, trimmed_after

    # Pattern 2: One is a substring of the other
    if before in after:
        return "", after
    if after in before:
        return before, ""

    # Pattern 3: General similarity check (expensive, only for moderate-length)
    if min_len <= 2000:
        ratio = _fast_similarity(before, after)
        if ratio >= similarity_threshold:
            # Keep the longer one
            if len(before) >= len(after):
                return before, ""
            else:
                return "", after

    return before, after


def _common_prefix_length(a: str, b: str) -> int:
    """Count characters matching at the start of both strings."""
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def _fast_similarity(a: str, b: str) -> float:
    """Cheap bigram-overlap similarity (Dice coefficient).

    Much faster than ``SequenceMatcher`` for moderate strings.
    """
    if not a or not b:
        return 0.0

    def _bigrams(s: str) -> dict[str, int]:
        bg: dict[str, int] = {}
        for i in range(len(s) - 1):
            pair = s[i:i + 2]
            bg[pair] = bg.get(pair, 0) + 1
        return bg

    bg_a = _bigrams(a)
    bg_b = _bigrams(b)
    overlap = 0
    for pair, count in bg_a.items():
        overlap += min(count, bg_b.get(pair, 0))
    total = (len(a) - 1) + (len(b) - 1)
    if total == 0:
        return 0.0
    return (2.0 * overlap) / total
