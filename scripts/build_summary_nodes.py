#!/usr/bin/env python3
"""
build_summary_nodes.py
======================
Emit lightweight **summary virtual nodes** for sections and chunks, as
an add-on layer on top of ``chunk_virtual_nodes.json``.

This is the *non-LLM prototype* of the summary layer called out in the
4.16 plan (虚拟节点 summary). It is deliberately zero-cost so we can
wire summary nodes into the graph and measure retrieval impact before
committing to an LLM-based summarizer.

Summary strategy (prototype v1, heuristic, no LLM):
    - section_summary: section title + first 1-2 sentences of the
      section's first paragraph, capped at ``--max-words`` tokens.
    - chunk_summary:   first 1-2 sentences of the chunk, same cap.

If/when this proves useful, swap ``_heuristic_summary()`` with an
LLM-backed implementation. All node IDs and edge shapes stay stable.

Input:
    data/01_graphs/chunk_virtual_nodes.json  (from build_chunk_virtual_nodes.py)

Output:
    data/01_graphs/summary_virtual_nodes.json
        {
          "metadata": {..., "strategy": "first_n_sentences"},
          "documents": {
            doc_id: {
              "section_summaries": { summary_id: {...} },
              "chunk_summaries":   { summary_id: {...} },
              "edges": [ {source, target, relation: "summary_of"} ]
            }
          },
          "stats": {...}
        }

Usage:
    python scripts/build_summary_nodes.py
    python scripts/build_summary_nodes.py --max-words 30 --sentences 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_INPUT = "data/01_graphs/chunk_virtual_nodes.json"
DEFAULT_OUTPUT = "data/01_graphs/summary_virtual_nodes.json"
DEFAULT_MAX_WORDS = 40
DEFAULT_SENTENCES = 2

# ── Sentence splitter (shared with build_chunk_virtual_nodes) ───────────────

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(])")


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def _word_cap(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:") + "…"


# ── Summary strategies ──────────────────────────────────────────────────────

def _heuristic_summary(
    text: str,
    *,
    sentences: int,
    max_words: int,
    prefix: str = "",
) -> str:
    """First-N-sentences summary, capped at max_words.

    Strategy v1: take the first N sentences of the text; if they exceed
    max_words, truncate on word boundary.

    Returns an empty string if the input text is empty.
    """
    sents = split_sentences(text)
    if not sents:
        return prefix.strip()
    picked = " ".join(sents[: max(1, sentences)])
    if prefix:
        picked = f"{prefix.strip()} — {picked}"
    return _word_cap(picked, max_words)


# ── Core emission ───────────────────────────────────────────────────────────

def build_summaries_for_doc(
    doc_id: str,
    doc: Dict[str, Any],
    *,
    max_words: int,
    sentences: int,
) -> Tuple[Dict[str, dict], Dict[str, dict], List[dict]]:
    """Return (section_summaries, chunk_summaries, edges) for a single doc."""
    section_summaries: Dict[str, dict] = OrderedDict()
    chunk_summaries: Dict[str, dict] = OrderedDict()
    edges: List[dict] = []

    chunk_nodes: Dict[str, dict] = doc.get("nodes", {}) or {}
    paragraph_nodes: Dict[str, dict] = doc.get("paragraph_nodes", {}) or {}

    # ── Section summaries ─────────────────────────────────────────
    # Group paragraphs by section_title, keep the first one per section.
    seen_sections: Dict[str, dict] = OrderedDict()
    for para_id, para in paragraph_nodes.items():
        sec_title = para.get("section_title", "")
        if not sec_title or sec_title in seen_sections:
            continue
        seen_sections[sec_title] = para

    for sec_idx, (sec_title, first_para) in enumerate(seen_sections.items()):
        summary_text = _heuristic_summary(
            first_para.get("text", ""),
            sentences=sentences,
            max_words=max_words,
            prefix=sec_title,
        )
        if not summary_text:
            continue
        summary_id = f"{doc_id}_secsum_{sec_idx}"
        section_summaries[summary_id] = {
            "summary_id": summary_id,
            "doc_id": doc_id,
            "scope": "section",
            "source_section": sec_title,
            "text": summary_text,
            "method": "first_n_sentences",
        }
        edges.append({
            "source": summary_id,
            "target": sec_title,  # matches build_chunk_virtual_nodes convention
            "source_type": "summary",
            "target_type": "section",
            "relation": "summary_of",
        })

    # ── Chunk summaries ───────────────────────────────────────────
    for chunk_id, chunk in chunk_nodes.items():
        chunk_text = chunk.get("text", "")
        summary_text = _heuristic_summary(
            chunk_text,
            sentences=sentences,
            max_words=max_words,
            prefix="",
        )
        if not summary_text:
            continue
        summary_id = f"{chunk_id}_sum"
        chunk_summaries[summary_id] = {
            "summary_id": summary_id,
            "doc_id": doc_id,
            "scope": "chunk",
            "source_chunk": chunk_id,
            "section_title": chunk.get("section_title", ""),
            "text": summary_text,
            "method": "first_n_sentences",
        }
        edges.append({
            "source": summary_id,
            "target": chunk_id,
            "source_type": "summary",
            "target_type": "chunk",
            "relation": "summary_of",
        })

    return section_summaries, chunk_summaries, edges


def process(
    input_path: str,
    output_path: str,
    *,
    max_words: int,
    sentences: int,
) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)

    documents = chunk_data.get("documents", {})

    result: Dict[str, Any] = {
        "metadata": {
            "source_file": input_path,
            "script": "scripts/build_summary_nodes.py",
            "strategy": "first_n_sentences",
            "max_words": max_words,
            "sentences": sentences,
            "node_types": ["summary"],
            "edge_types": ["summary_of"],
            "notes": (
                "Prototype heuristic summary (non-LLM). Swap "
                "_heuristic_summary() with an LLM-backed implementation "
                "once retrieval impact is measured."
            ),
        },
        "documents": OrderedDict(),
        "stats": {
            "total_docs": 0,
            "total_section_summaries": 0,
            "total_chunk_summaries": 0,
            "total_edges": 0,
        },
    }

    for doc_id, doc in documents.items():
        sec_sums, chunk_sums, edges = build_summaries_for_doc(
            doc_id, doc, max_words=max_words, sentences=sentences
        )
        if not sec_sums and not chunk_sums:
            continue
        result["documents"][doc_id] = {
            "doc_id": doc_id,
            "section_summaries": sec_sums,
            "chunk_summaries": chunk_sums,
            "edges": edges,
            "num_section_summaries": len(sec_sums),
            "num_chunk_summaries": len(chunk_sums),
            "num_edges": len(edges),
        }
        result["stats"]["total_docs"] += 1
        result["stats"]["total_section_summaries"] += len(sec_sums)
        result["stats"]["total_chunk_summaries"] += len(chunk_sums)
        result["stats"]["total_edges"] += len(edges)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    stats = result["stats"]
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("── Summary virtual nodes ──")
    print(f"  Docs processed:           {stats['total_docs']}")
    print(f"  Section summaries:        {stats['total_section_summaries']}")
    print(f"  Chunk summaries:          {stats['total_chunk_summaries']}")
    print(f"  summary_of edges:         {stats['total_edges']}")
    print(f"  Saved to {output_path} ({size_mb:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Emit heuristic summary virtual nodes for section / chunk"
    )
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help=f"chunk_virtual_nodes.json path (default: {DEFAULT_INPUT})")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help=f"output path (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS,
                    help=f"max words per summary (default: {DEFAULT_MAX_WORDS})")
    ap.add_argument("--sentences", type=int, default=DEFAULT_SENTENCES,
                    help=f"number of leading sentences to keep (default: {DEFAULT_SENTENCES})")
    args = ap.parse_args()

    if not Path(args.input).exists():
        print(f"[fatal] input not found: {args.input}", file=sys.stderr)
        print(
            "        run scripts/build_chunk_virtual_nodes.py first",
            file=sys.stderr,
        )
        sys.exit(1)

    process(
        args.input,
        args.output,
        max_words=args.max_words,
        sentences=args.sentences,
    )


if __name__ == "__main__":
    main()
