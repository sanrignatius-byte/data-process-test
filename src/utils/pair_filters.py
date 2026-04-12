"""Utilities for strict intra-document filtering of pair-style candidates."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


_ELEMENT_ID_RE = re.compile(r"^([^_]+)_(figure|table|formula)_\d+$")


def extract_doc_id_from_node(node_id: Any) -> str:
    """Extract doc_id from candidate node IDs used in pair/path records."""
    text = str(node_id or "").strip()
    if not text:
        return ""
    if "::" in text:
        return text.split("::", 1)[0]
    match = _ELEMENT_ID_RE.match(text)
    if match:
        return match.group(1)
    return ""


def collect_pair_doc_ids(pair: Dict[str, Any]) -> Set[str]:
    """Collect all doc ids that appear anywhere in a pair/path record."""
    docs: Set[str] = set()

    for value in (
        pair.get("doc_id"),
        pair.get("element_a_id"),
        pair.get("element_b_id"),
        pair.get("element_a", {}).get("doc_id"),
        pair.get("element_b", {}).get("doc_id"),
    ):
        doc_id = extract_doc_id_from_node(value) if value != pair.get("doc_id") else str(value or "").strip()
        if doc_id:
            docs.add(doc_id)

    for node_id in pair.get("path", []) or []:
        doc_id = extract_doc_id_from_node(node_id)
        if doc_id:
            docs.add(doc_id)

    for elem in pair.get("node_group", []) or []:
        if not isinstance(elem, dict):
            continue
        doc_id = extract_doc_id_from_node(elem.get("element_id")) or str(elem.get("doc_id", "") or "").strip()
        if doc_id:
            docs.add(doc_id)

    return docs


def is_intra_doc_pair(pair: Dict[str, Any]) -> bool:
    """Return True only when pair metadata and actual ids/path agree on one doc."""
    if pair.get("is_cross_doc") or pair.get("hub_metadata", {}).get("is_cross_doc", False):
        return False
    docs = collect_pair_doc_ids(pair)
    return len(docs) == 1


def filter_intra_doc_pairs(
    pairs: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Drop every cross-doc or mixed-doc pair conservatively."""
    kept: List[Dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for pair in pairs:
        docs = collect_pair_doc_ids(pair)
        flag_cross = bool(pair.get("is_cross_doc") or pair.get("hub_metadata", {}).get("is_cross_doc", False))

        if flag_cross:
            stats["drop_cross_doc_flag"] += 1
            continue
        if not docs:
            stats["drop_missing_doc_ids"] += 1
            continue
        if len(docs) > 1:
            stats["drop_mixed_doc_ids"] += 1
            continue

        kept.append(pair)
        stats["kept"] += 1

    stats["input"] = len(pairs)
    stats["dropped"] = len(pairs) - len(kept)
    return kept, dict(stats)


def pair_doc_id(pair: Dict[str, Any]) -> str:
    """Return the resolved single doc id for a strict intra-doc pair."""
    docs = sorted(collect_pair_doc_ids(pair))
    return docs[0] if len(docs) == 1 else ""
