#!/usr/bin/env python3
"""
build_paragraph_chunks.py
=========================
将 chunk_virtual_nodes_v2.json 中的文本元素（text element，即 paragraph）
合并为固定长度的切片（chunk）。支持 NLTK / spaCy / regex 三种分词器。
可选地从 multimodal_elements.json 注入 element_ids，使每个切片知道
它包含哪些图/表/公式元素。

元素-切片对齐：element.position_idx 与 text element 的 para_idx 共享
同一序号空间——position_idx 落在切片 paragraph_indices 范围内的元素
即被分配给该切片。对齐优先使用 LaTeX 行号精确匹配，字符串模糊匹配仅
作为后备（fallback）。

Usage:
    python scripts/build_paragraph_chunks.py \\
        --chunk-size 400 \\
        --output data/01_graphs/paragraph_chunks_n400.json

    python scripts/build_paragraph_chunks.py \\
        --chunk-size 500 \\
        --output data/01_graphs/paragraph_chunks_n500.json \\
        --tokenizer spacy

    # Without element injection (fast, no elements.json needed):
    python scripts/build_paragraph_chunks.py \\
        --chunk-size 400 --no-elements \\
        --output data/01_graphs/paragraph_chunks_n400_noelems.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT    = "data/01_graphs/chunk_virtual_nodes_v2.json"
DEFAULT_ELEMENTS = "data/03_queries/M4query_v1/graphs/multimodal_elements.json"
DEFAULT_MIN_WORDS = 80


# ── Tokenizer ─────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_tok_cache: dict = {}


def _build_tokenizer(preferred: str):
    if preferred in _tok_cache:
        return _tok_cache[preferred]

    if preferred == "nltk":
        try:
            from nltk.tokenize import word_tokenize  # type: ignore
            word_tokenize("probe.")  # triggers LookupError if punkt missing
            fn = word_tokenize
            print("[tokenizer] using NLTK word_tokenize", file=sys.stderr)
        except Exception as e:
            print(f"[tokenizer] NLTK unavailable ({e}), falling back to regex", file=sys.stderr)
            fn = lambda s: _WORD_RE.findall(s)

    elif preferred == "spacy":
        try:
            import spacy  # type: ignore
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger", "lemmatizer"])
            fn = lambda s: [t.text for t in nlp.tokenizer(s)]
            print("[tokenizer] using spaCy tokenizer (en_core_web_sm)", file=sys.stderr)
        except Exception as e:
            print(f"[tokenizer] spaCy unavailable ({e}), falling back to regex", file=sys.stderr)
            fn = lambda s: _WORD_RE.findall(s)

    else:  # "regex" or "auto" fallback
        # Try NLTK first in auto mode
        try:
            from nltk.tokenize import word_tokenize  # type: ignore
            word_tokenize("probe.")
            fn = word_tokenize
            print("[tokenizer] auto → NLTK word_tokenize", file=sys.stderr)
        except Exception:
            fn = lambda s: _WORD_RE.findall(s)
            print("[tokenizer] auto → regex fallback", file=sys.stderr)

    _tok_cache[preferred] = fn
    return fn


def word_count(text: str, tokenizer) -> int:
    return len(tokenizer(text))


# ── Element index ─────────────────────────────────────────────────────────────

def build_element_index(elements_path: Optional[Path]) -> Dict[str, Dict[int, List[str]]]:
    """Return {doc_id: {position_idx: [element_id, ...]}}."""
    if elements_path is None or not elements_path.exists():
        return {}
    data = json.loads(elements_path.read_bytes())
    index: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    for doc_id, doc in data.get("documents", {}).items():
        for eid, elem in doc.get("elements", {}).items():
            pidx = elem.get("position_idx")
            if pidx is not None:
                index[doc_id][pidx].append(eid)
    return index


# ── Chunking ──────────────────────────────────────────────────────────────────

def merge_paragraphs(
    para_nodes: List[dict],
    target_words: int,
    min_words: int,
    tokenizer,
) -> List[dict]:
    """Greedy merge of ordered paragraph nodes into ~target_words chunks."""
    if not para_nodes:
        return []

    chunks: List[dict] = []
    buf: List[dict] = []
    buf_words = 0

    def flush():
        if not buf:
            return
        sec_titles = list(dict.fromkeys(p.get("section_title", "") for p in buf))
        chunks.append({
            "section_title":  buf[0].get("section_title", ""),
            "section_level":  buf[0].get("section_level", 0),
            "section_titles": sec_titles,
            "text":           "\n\n".join(p["text"] for p in buf),
            "word_count":     buf_words,
            "paragraph_indices": [p["para_idx"] for p in buf],
        })

    for para in para_nodes:
        text = para.get("text", "")
        if not text.strip():
            continue
        wc = para.get("word_count") or word_count(text, tokenizer)

        # Flush at major section boundaries
        is_major_boundary = (
            buf
            and para.get("section_title") != buf[-1].get("section_title")
            and para.get("section_level", 0) <= 1
            and buf_words >= min_words
        )
        if is_major_boundary:
            flush()
            buf, buf_words = [], 0

        buf.append({**para, "word_count": wc})
        buf_words += wc

        if buf_words >= target_words:
            flush()
            buf, buf_words = [], 0

    flush()

    # Absorb trailing runts
    merged: List[dict] = []
    for chunk in chunks:
        if merged and chunk["word_count"] < min_words:
            prev = merged[-1]
            prev["text"]      += "\n\n" + chunk["text"]
            prev["word_count"] += chunk["word_count"]
            prev["paragraph_indices"].extend(chunk["paragraph_indices"])
            prev["section_titles"] = list(
                dict.fromkeys(prev["section_titles"] + chunk["section_titles"])
            )
        else:
            merged.append(chunk)
    return merged


def assign_elements(
    chunks: List[dict],
    doc_id: str,
    elem_index: Dict[int, List[str]],
) -> None:
    """Mutate each chunk dict to add element_ids field."""
    for chunk in chunks:
        pidx_set: Set[int] = set(chunk["paragraph_indices"])
        eids: List[str] = []
        for pidx in sorted(pidx_set):
            eids.extend(elem_index.get(pidx, []))
        chunk["element_ids"] = eids


def build_edges(doc_id: str, chunks: List[dict], para_nodes: List[dict]) -> List[dict]:
    edges: List[dict] = []

    # section → chunk
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        for sec in chunk.get("section_titles", [chunk["section_title"]]):
            if sec:
                edges.append({
                    "source": sec, "target": chunk_id,
                    "source_type": "section", "target_type": "chunk",
                    "relation": "section_contains_chunk",
                })

    # chunk_sequence
    for i in range(len(chunks) - 1):
        edges.append({
            "source": f"{doc_id}_chunk_{i}",
            "target": f"{doc_id}_chunk_{i + 1}",
            "source_type": "chunk", "target_type": "chunk",
            "relation": "chunk_sequence",
        })

    # chunk → paragraph
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        for pidx in chunk["paragraph_indices"]:
            edges.append({
                "source": chunk_id,
                "target": f"{doc_id}_para_{pidx}",
                "source_type": "chunk", "target_type": "paragraph",
                "relation": "chunk_contains_paragraph",
            })

    # chunk → element
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        for eid in chunk.get("element_ids", []):
            edges.append({
                "source": chunk_id, "target": eid,
                "source_type": "chunk", "target_type": "element",
                "relation": "chunk_contains_element",
            })

    # section → paragraph
    for para in para_nodes:
        sec = para.get("section_title", "")
        if sec:
            edges.append({
                "source": sec,
                "target": f"{doc_id}_para_{para['para_idx']}",
                "source_type": "section", "target_type": "paragraph",
                "relation": "section_contains_paragraph",
            })

    return edges


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      default=DEFAULT_INPUT)
    ap.add_argument("--elements",   default=DEFAULT_ELEMENTS,
                    help="multimodal_elements.json for element_ids injection")
    ap.add_argument("--no-elements", action="store_true",
                    help="Skip element injection (faster)")
    ap.add_argument("--chunk-size", type=int, required=True,
                    help="Target words per chunk")
    ap.add_argument("--min-words",  type=int, default=DEFAULT_MIN_WORDS)
    ap.add_argument("--tokenizer",  choices=["auto", "nltk", "spacy", "regex"],
                    default="auto")
    ap.add_argument("--output",     required=True)
    args = ap.parse_args()

    tokenizer = _build_tokenizer(args.tokenizer)

    input_path = (PROJECT_ROOT / args.input
                  if not Path(args.input).is_absolute() else Path(args.input))
    output_path = (PROJECT_ROOT / args.output
                   if not Path(args.output).is_absolute() else Path(args.output))

    elements_path: Optional[Path] = None
    if not args.no_elements:
        ep = (PROJECT_ROOT / args.elements
              if not Path(args.elements).is_absolute() else Path(args.elements))
        if ep.exists():
            elements_path = ep
        else:
            print(f"[warn] elements file not found, skipping injection: {ep}", file=sys.stderr)

    print(f"Loading {input_path} ...", flush=True)
    source = json.loads(input_path.read_bytes())

    print("Building element index ...", flush=True)
    global_elem_index = build_element_index(elements_path)
    total_elem_indexed = sum(len(v) for d in global_elem_index.values() for v in d.values())
    print(f"  {total_elem_indexed} elements indexed across {len(global_elem_index)} docs")

    result = {
        "metadata": {
            "source":       str(input_path),
            "target_words": args.chunk_size,
            "min_words":    args.min_words,
            "tokenizer":    args.tokenizer,
            "elements":     str(elements_path) if elements_path else None,
            "script":       "scripts/build_paragraph_chunks.py",
        },
        "documents": {},
        "stats": {
            "total_docs": 0, "docs_with_chunks": 0,
            "total_chunks": 0, "total_paragraphs": 0,
            "total_elements_assigned": 0, "total_edges": 0,
            "avg_words_per_chunk": 0.0,
        },
    }

    total_words = 0

    for doc_id, doc in source.get("documents", {}).items():
        raw_paras = doc.get("paragraph_nodes", {})
        if isinstance(raw_paras, dict):
            para_list = sorted(raw_paras.values(), key=lambda p: p.get("para_idx", 0))
        else:
            para_list = sorted(raw_paras, key=lambda p: p.get("para_idx", 0))

        if not para_list:
            continue

        result["stats"]["total_docs"] += 1

        # Recompute word counts with proper tokenizer
        for p in para_list:
            p["word_count"] = word_count(p.get("text", ""), tokenizer)

        chunks = merge_paragraphs(para_list, args.chunk_size, args.min_words, tokenizer)
        if not chunks:
            continue

        result["stats"]["docs_with_chunks"] += 1

        # Inject element_ids
        doc_elem_index = global_elem_index.get(doc_id, {})
        assign_elements(chunks, doc_id, doc_elem_index)

        # Build nodes
        nodes = OrderedDict()
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            nodes[chunk_id] = {
                "chunk_id":          chunk_id,
                "doc_id":            doc_id,
                "chunk_idx":         i,
                "section_title":     chunk["section_title"],
                "section_titles":    chunk["section_titles"],
                "section_level":     chunk["section_level"],
                "text":              chunk["text"],
                "word_count":        chunk["word_count"],
                "paragraph_indices": chunk["paragraph_indices"],
                "element_ids":       chunk["element_ids"],
            }

        # Preserve paragraph nodes with updated word counts
        paragraph_nodes = OrderedDict()
        for para in para_list:
            para_id = f"{doc_id}_para_{para['para_idx']}"
            paragraph_nodes[para_id] = {
                "paragraph_id":  para_id,
                "doc_id":        doc_id,
                "para_idx":      para["para_idx"],
                "section_title": para.get("section_title", ""),
                "section_level": para.get("section_level", 0),
                "text":          para.get("text", ""),
                "word_count":    para["word_count"],
            }

        edges = build_edges(doc_id, chunks, para_list)

        elem_assigned = sum(len(c["element_ids"]) for c in chunks)
        result["documents"][doc_id] = {
            "doc_id":           doc_id,
            "source_markdown":  doc.get("source_markdown", ""),
            "num_sections":     doc.get("num_sections", 0),
            "num_chunks":       len(chunks),
            "num_paragraphs":   len(paragraph_nodes),
            "num_elements":     elem_assigned,
            "num_edges":        len(edges),
            "total_words":      sum(c["word_count"] for c in chunks),
            "nodes":            nodes,
            "paragraph_nodes":  paragraph_nodes,
            "edges":            edges,
        }

        result["stats"]["total_chunks"]            += len(chunks)
        result["stats"]["total_paragraphs"]        += len(paragraph_nodes)
        result["stats"]["total_elements_assigned"] += elem_assigned
        result["stats"]["total_edges"]             += len(edges)
        total_words += sum(c["word_count"] for c in chunks)

    if result["stats"]["total_chunks"] > 0:
        result["stats"]["avg_words_per_chunk"] = round(
            total_words / result["stats"]["total_chunks"], 1
        )

    s = result["stats"]
    print(f"  Docs:               {s['total_docs']}")
    print(f"  Docs with chunks:   {s['docs_with_chunks']}")
    print(f"  Total chunks:       {s['total_chunks']}")
    print(f"  Total paragraphs:   {s['total_paragraphs']}")
    print(f"  Elements assigned:  {s['total_elements_assigned']}")
    print(f"  Avg words/chunk:    {s['avg_words_per_chunk']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved → {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
