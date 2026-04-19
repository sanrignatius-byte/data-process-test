#!/usr/bin/env python3
"""
build_paragraph_chunks.py
=========================
Re-merge pre-parsed paragraph nodes from an existing chunk graph
(chunk_virtual_nodes_v2.json) into fixed-size chunks with a new
target word count (e.g., 400 or 500 tokens).

Outputs a new JSON in the same format as chunk_virtual_nodes_v2.json,
which can then be used as input to build_graph_augmented_corpus.py via
--paragraph-chunks to build corpus v3.

Usage:
    python scripts/build_paragraph_chunks.py \
        --input  data/01_graphs/chunk_virtual_nodes_v2.json \
        --chunk-size 400 \
        --output data/01_graphs/paragraph_chunks_n400.json

    python scripts/build_paragraph_chunks.py \
        --chunk-size 500 \
        --output data/01_graphs/paragraph_chunks_n500.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = "data/01_graphs/chunk_virtual_nodes_v2.json"
DEFAULT_MIN_WORDS = 80


# ── Word count ───────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


# ── Chunking ─────────────────────────────────────────────────────────────────

def merge_paragraphs(
    para_nodes: list[dict],
    target_words: int,
    min_words: int,
) -> list[dict]:
    """Greedy merge of paragraph nodes into chunks of ~target_words.

    Para nodes must be in reading order (para_idx ascending).
    Returns list of chunk dicts (no IDs yet; caller adds doc_id + chunk_idx).
    """
    if not para_nodes:
        return []

    chunks: list[dict] = []
    buf: list[dict] = []
    buf_words = 0

    def flush():
        if not buf:
            return
        primary_sec = buf[0].get("section_title", "")
        primary_level = buf[0].get("section_level", 0)
        sec_titles = list(dict.fromkeys(p.get("section_title", "") for p in buf))
        chunks.append({
            "section_title": primary_sec,
            "section_level": primary_level,
            "section_titles": sec_titles,
            "text": "\n\n".join(p["text"] for p in buf),
            "word_count": buf_words,
            "paragraph_indices": [p["para_idx"] for p in buf],
        })

    for para in para_nodes:
        wc = para.get("word_count") or word_count(para.get("text", ""))
        if not para.get("text", "").strip():
            continue

        # Major section boundary: flush if buffer is big enough
        is_major_boundary = (
            buf
            and para.get("section_title") != buf[-1].get("section_title")
            and para.get("section_level", 0) <= 1
            and buf_words >= min_words
        )
        if is_major_boundary:
            flush()
            buf = []
            buf_words = 0

        buf.append({**para, "word_count": wc})
        buf_words += wc

        if buf_words >= target_words:
            flush()
            buf = []
            buf_words = 0

    flush()

    # Merge trailing runts
    merged: list[dict] = []
    for chunk in chunks:
        if merged and chunk["word_count"] < min_words:
            prev = merged[-1]
            prev["text"] += "\n\n" + chunk["text"]
            prev["word_count"] += chunk["word_count"]
            prev["paragraph_indices"].extend(chunk["paragraph_indices"])
            prev["section_titles"] = list(
                dict.fromkeys(prev["section_titles"] + chunk["section_titles"])
            )
        else:
            merged.append(chunk)

    return merged


def build_edges(doc_id: str, chunks: list[dict], para_nodes: list[dict]) -> list[dict]:
    edges: list[dict] = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        for sec in chunk.get("section_titles", [chunk["section_title"]]):
            edges.append({
                "source": sec, "target": chunk_id,
                "source_type": "section", "target_type": "chunk",
                "relation": "section_contains_chunk",
            })

    for i in range(len(chunks) - 1):
        edges.append({
            "source": f"{doc_id}_chunk_{i}",
            "target": f"{doc_id}_chunk_{i + 1}",
            "source_type": "chunk", "target_type": "chunk",
            "relation": "chunk_sequence",
        })

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        for pidx in chunk["paragraph_indices"]:
            edges.append({
                "source": chunk_id, "target": f"{doc_id}_para_{pidx}",
                "source_type": "chunk", "target_type": "paragraph",
                "relation": "chunk_contains_paragraph",
            })

    for para in para_nodes:
        edges.append({
            "source": para.get("section_title", ""),
            "target": f"{doc_id}_para_{para['para_idx']}",
            "source_type": "section", "target_type": "paragraph",
            "relation": "section_contains_paragraph",
        })

    return edges


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Re-chunk paragraph nodes to fixed target size")
    ap.add_argument("--input", default=DEFAULT_INPUT,
                    help=f"Source chunk graph JSON (default: {DEFAULT_INPUT})")
    ap.add_argument("--chunk-size", type=int, required=True,
                    help="Target words per chunk (e.g. 400 or 500)")
    ap.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS,
                    help=f"Minimum words for a standalone chunk (default: {DEFAULT_MIN_WORDS})")
    ap.add_argument("--output", required=True,
                    help="Output JSON path")
    args = ap.parse_args()

    input_path = (PROJECT_ROOT / args.input
                  if not Path(args.input).is_absolute() else Path(args.input))
    output_path = (PROJECT_ROOT / args.output
                   if not Path(args.output).is_absolute() else Path(args.output))

    print(f"Loading {input_path} ...", flush=True)
    source = json.loads(input_path.read_bytes())

    result = {
        "metadata": {
            "source": str(input_path),
            "target_words": args.chunk_size,
            "min_words": args.min_words,
            "script": "scripts/build_paragraph_chunks.py",
        },
        "documents": {},
        "stats": {
            "total_docs": 0,
            "docs_with_chunks": 0,
            "total_chunks": 0,
            "total_paragraphs": 0,
            "total_edges": 0,
            "avg_words_per_chunk": 0.0,
        },
    }

    total_words = 0

    for doc_id, doc in source.get("documents", {}).items():
        # Collect paragraph nodes in reading order
        raw_paras = doc.get("paragraph_nodes", {})
        if isinstance(raw_paras, dict):
            para_list = sorted(raw_paras.values(), key=lambda p: p.get("para_idx", 0))
        else:
            para_list = sorted(raw_paras, key=lambda p: p.get("para_idx", 0))

        if not para_list:
            continue

        result["stats"]["total_docs"] += 1

        chunks = merge_paragraphs(para_list, args.chunk_size, args.min_words)
        if not chunks:
            continue

        result["stats"]["docs_with_chunks"] += 1

        nodes = OrderedDict()
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            nodes[chunk_id] = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_idx": i,
                "section_title": chunk["section_title"],
                "section_titles": chunk["section_titles"],
                "section_level": chunk["section_level"],
                "text": chunk["text"],
                "word_count": chunk["word_count"],
                "paragraph_indices": chunk["paragraph_indices"],
            }

        paragraph_nodes = OrderedDict()
        for para in para_list:
            para_id = f"{doc_id}_para_{para['para_idx']}"
            paragraph_nodes[para_id] = {
                "paragraph_id": para_id,
                "doc_id": doc_id,
                "para_idx": para["para_idx"],
                "section_title": para.get("section_title", ""),
                "section_level": para.get("section_level", 0),
                "text": para.get("text", ""),
                "word_count": para.get("word_count") or word_count(para.get("text", "")),
            }

        edges = build_edges(doc_id, chunks, para_list)

        result["documents"][doc_id] = {
            "doc_id": doc_id,
            "source_markdown": doc.get("source_markdown", ""),
            "num_sections": doc.get("num_sections", 0),
            "num_chunks": len(chunks),
            "num_paragraphs": len(paragraph_nodes),
            "num_edges": len(edges),
            "total_words": sum(c["word_count"] for c in chunks),
            "nodes": nodes,
            "paragraph_nodes": paragraph_nodes,
            "edges": edges,
        }

        result["stats"]["total_chunks"] += len(chunks)
        result["stats"]["total_paragraphs"] += len(paragraph_nodes)
        result["stats"]["total_edges"] += len(edges)
        total_words += sum(c["word_count"] for c in chunks)

    if result["stats"]["total_chunks"] > 0:
        result["stats"]["avg_words_per_chunk"] = round(
            total_words / result["stats"]["total_chunks"], 1
        )

    s = result["stats"]
    print(f"  Docs processed:   {s['total_docs']}")
    print(f"  Docs with chunks: {s['docs_with_chunks']}")
    print(f"  Total chunks:     {s['total_chunks']}")
    print(f"  Total paragraphs: {s['total_paragraphs']}")
    print(f"  Avg words/chunk:  {s['avg_words_per_chunk']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
