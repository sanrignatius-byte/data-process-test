"""Bridge text resolution utilities.

从 generate_multihop_l1_queries.py 提取的共享函数，用于解析 LaTeX 引用图
中的 bridge 段落文本。

核心链路: element_id → LaTeX label → ref graph edge → context

使用示例:
    from src.utils.bridge_utils import (
        load_reference_graph_bridge_texts,
        resolve_bridge_texts_for_path,
    )
    
    # 初始化缓存
    load_reference_graph_bridge_texts(
        ref_graph_path="data/01_graphs/latex_reference_graph.json",
        topology_candidates_path="data/01_graphs/latex_topology_candidates.json",
    )
    
    # 为某个 pair 获取 bridge 文本
    bridge_texts = resolve_bridge_texts_for_path(pair)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# ── Module-level caches ──────────────────────────────────────────────────────

# {doc_id: {latex_label: "bridge paragraph text"}}
_BRIDGE_TEXT_CACHE: Dict[str, Dict[str, str]] = {}

# {element_id: [latex_label1, latex_label2, ...]}
_ELEMENT_TO_LABELS: Dict[str, List[str]] = {}


# ── Public API ───────────────────────────────────────────────────────────────

def load_reference_graph_bridge_texts(
    ref_graph_path: str,
    topology_candidates_path: str = "",
) -> None:
    """Pre-load paragraph contexts from latex_reference_graph.json.

    Also loads topology candidates to build element_id → LaTeX label mapping,
    so we can resolve MinerU element IDs to the LaTeX labels used in edge contexts.

    Bridge text is found by: element_id → LaTeX label → edges referencing that label
    → edge context = the bridge paragraph text.
    """
    if not ref_graph_path or not Path(ref_graph_path).exists():
        return
    data = json.loads(Path(ref_graph_path).read_text(encoding="utf-8"))
    docs = data.get("documents", {})
    for doc_id, doc in docs.items():
        ctx_by_label: Dict[str, List[str]] = defaultdict(list)

        # Index edge contexts by target label (the element being referenced)
        for edge in doc.get("edges", []):
            ctx = (edge.get("context", "") or "").strip()
            if len(ctx) < 20:
                continue
            # Skip containment edges ("fig:X is within sec:Y")
            if " is within " in ctx:
                continue
            ctx_clean = _clean_latex_bridge(ctx)
            if len(ctx_clean) < 20:
                continue
            tgt = edge.get("target_label", "")
            if tgt:
                ctx_by_label[tgt].append(ctx_clean)

        _BRIDGE_TEXT_CACHE[doc_id] = {
            k: " | ".join(dict.fromkeys(vs[:3]))  # dedup while preserving order
            for k, vs in ctx_by_label.items()
        }

    # Build element_id → LaTeX label mapping from topology candidates
    _build_element_label_map_from_topology(topology_candidates_path)
    _build_element_label_map_from_ref_graph(data)


def resolve_bridge_texts_for_path(pair: Dict[str, Any]) -> List[str]:
    """Given a candidate pair with path, resolve actual bridge paragraph texts.

    Strategy: map MinerU element_ids → LaTeX labels (via ordinal mapping),
    then look up edge contexts referencing those labels in the reference graph.
    The edge context IS the bridge paragraph text — the sentence where the
    author connects two elements via \\ref{}.

    Returns a list of bridge paragraph texts (cleaned, max 3).
    """
    elem_a_id = pair.get("element_a_id", "")
    elem_b_id = pair.get("element_b_id", "")

    bridge_texts: List[str] = []
    seen: Set[str] = set()

    for eid in [elem_a_id, elem_b_id]:
        # Extract doc_id from element_id: "1709.02012_figure_4" → "1709.02012"
        parts = eid.rsplit("_", 2)
        if len(parts) < 3:
            continue
        eid_doc = parts[0]

        cache = _BRIDGE_TEXT_CACHE.get(eid_doc, {})
        if not cache:
            continue

        # Get LaTeX labels for this element_id from the mapping
        latex_labels = _ELEMENT_TO_LABELS.get(eid, [])
        for label in latex_labels:
            if label in cache:
                text = cache[label]
                if text not in seen:
                    seen.add(text)
                    bridge_texts.append(text)

    # Fallback: use bridge_contexts from topology (if stored by P0 enhancement)
    if not bridge_texts:
        for bc in pair.get("bridge_contexts", []):
            text = (bc.get("text", "") or "").strip()
            if text and text not in seen:
                seen.add(text)
                bridge_texts.append(text)

    return bridge_texts[:3]  # Cap at 3 bridge segments


def get_bridge_text_cache() -> Dict[str, Dict[str, str]]:
    """Return the bridge text cache for inspection."""
    return _BRIDGE_TEXT_CACHE


def get_element_to_labels() -> Dict[str, List[str]]:
    """Return the element → labels mapping for inspection."""
    return _ELEMENT_TO_LABELS


# ── Internal helpers ─────────────────────────────────────────────────────────

def _clean_latex_bridge(text: str) -> str:
    """Strip LaTeX commands from bridge text while preserving semantic content."""
    text = re.sub(r'\\includegraphics[^}]*\}', '', text)
    text = re.sub(r'\\(?:ref|eqref|autoref|cref|Cref)\{([^}]*)\}', r'[\1]', text)
    text = re.sub(r'\\cite\{([^}]*)\}', r'[cite:\1]', text)
    text = re.sub(r'\\[a-zA-Z]+\*?\s*(?:\[[^\]]*\])?\{([^}]{0,120})\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+\*?', ' ', text)
    text = re.sub(r'[${}]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _build_element_label_map_from_topology(topo_path: str) -> None:
    """Build element_id → label mapping from topology candidates."""
    if not topo_path or not Path(topo_path).exists():
        return
    topo = json.loads(Path(topo_path).read_text(encoding="utf-8"))
    for cand in topo.get("candidates", []):
        path_ids = cand.get("path_node_ids", [])
        path_types = cand.get("path_node_types", [])
        for nid, ntype in zip(path_ids, path_types):
            if ntype in ("figure", "table", "formula", "equation"):
                # Extract LaTeX label from node_id: "doc::el::fig:cooking" → "fig:cooking"
                if "::el::" in nid:
                    latex_label = nid.split("::el::")[-1]
                    doc_id = nid.split("::")[0]
                    _ELEMENT_TO_LABELS.setdefault(f"{doc_id}::{ntype}", []).append(latex_label)


def _build_element_label_map_from_ref_graph(data: Dict[str, Any]) -> None:
    """Build element_id → label mapping from reference graph labels.

    Uses type+line_no ordering: MinerU assigns element IDs by ordinal within type,
    so we sort LaTeX labels by line_no and map ordinals to MinerU-style IDs.
    """
    for doc_id, doc in data.get("documents", {}).items():
        labels = doc.get("labels", {}) or {}
        # Group labels by type, sorted by line_no
        by_type: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for label_key, info in labels.items():
            lt = (info.get("label_type", "") or "").lower()
            # Normalize type
            if "fig" in lt:
                etype = "figure"
            elif "tab" in lt:
                etype = "table"
            elif "eq" in lt or "formula" in lt:
                etype = "formula"
            else:
                continue
            line_no = int(info.get("line_no", 0)) if isinstance(info.get("line_no"), int) else 0
            by_type[etype].append((line_no, label_key))

        # Sort by line_no and assign ordinal (1-based to match MinerU numbering)
        for etype, items in by_type.items():
            items.sort()
            for ordinal, (_, label_key) in enumerate(items, start=1):
                element_id = f"{doc_id}_{etype}_{ordinal}"
                if element_id not in _ELEMENT_TO_LABELS:
                    _ELEMENT_TO_LABELS[element_id] = []
                _ELEMENT_TO_LABELS[element_id].append(label_key)
