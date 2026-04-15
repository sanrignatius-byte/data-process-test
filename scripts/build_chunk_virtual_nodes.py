#!/usr/bin/env python3
"""
build_chunk_virtual_nodes.py
============================
Parse MinerU markdown outputs, group paragraphs by section headers,
merge adjacent paragraphs into ~400-500 word chunks, and produce an
augmented graph JSON with chunk virtual nodes + edges.

Output: data/01_graphs/chunk_virtual_nodes.json

Schema (per document):
  nodes:  { chunk_id: { doc_id, chunk_idx, section_title, section_level,
                        text, word_count, page_range, paragraph_indices } }
  edges:  [ { source, target, source_type, target_type, relation } ]
          relation ∈ {section_contains_chunk, chunk_sequence, chunk_overlaps_section}

Usage:
    python scripts/build_chunk_virtual_nodes.py [--target-words 450] [--min-words 100]
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

MINERU_BASE = "data/00_raw/mineru_output"
OUTPUT_PATH = "data/01_graphs/chunk_virtual_nodes.json"
DEFAULT_TARGET_WORDS = 450
DEFAULT_MIN_WORDS = 100  # minimum words for a standalone chunk


# ── Word tokenization ────────────────────────────────────────────────────────
#
# Priority: nltk.word_tokenize > regex fallback.
# We avoid spacy because spacy models add hundreds of MB of weights that
# aren't justified for simple word counting. nltk falls back to a Treebank-
# style regex if the `punkt` data is missing, so this function never crashes.

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_TOKENIZER_STATE: dict = {"engine": None, "callable": None}


def _resolve_tokenizer(preferred: str = "auto"):
    """Pick a tokenizer once and cache it in _TOKENIZER_STATE.

    preferred:
        "auto"       — try nltk, then regex
        "nltk"       — force nltk (errors if unavailable)
        "regex"      — force regex fallback
    """
    if _TOKENIZER_STATE["callable"] is not None:
        return _TOKENIZER_STATE["engine"], _TOKENIZER_STATE["callable"]

    def _regex_tok(s: str) -> list[str]:
        return _WORD_RE.findall(s)

    if preferred == "regex":
        _TOKENIZER_STATE.update({"engine": "regex", "callable": _regex_tok})
        return "regex", _regex_tok

    try:
        from nltk.tokenize import word_tokenize  # type: ignore

        # Sanity-check: word_tokenize requires the `punkt` resource. If
        # missing, nltk raises LookupError on first call. Probe once here.
        word_tokenize("probe sentence.")
        _TOKENIZER_STATE.update({"engine": "nltk", "callable": word_tokenize})
        return "nltk", word_tokenize
    except Exception as e:
        if preferred == "nltk":
            raise RuntimeError(
                "nltk tokenizer requested but unavailable. "
                "Install nltk and run nltk.download('punkt') first."
            ) from e
        # auto mode: silently fall back
        _TOKENIZER_STATE.update({"engine": "regex", "callable": _regex_tok})
        return "regex", _regex_tok


def tokenize_words(text: str) -> list[str]:
    """Split text into word-like tokens using the configured tokenizer."""
    _, tok = _resolve_tokenizer()
    return tok(text)


# ── Sentence splitting (for paragraph preview / first-sentence hooks) ───────

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(])")


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter. Good enough for academic English prose."""
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


# ── Section-level inference ──────────────────────────────────────────────────

def infer_section_level(title: str) -> int:
    """Infer section depth from numbering pattern.
    
    Examples:
        "1 INTRODUCTION"        → 1
        "2.1 UNSUPERVISED"      → 2
        "2.3.1 MAXIMUM MEAN"    → 3
        "A DISCRIMINATION"      → 1  (appendix)
        "ABSTRACT"              → 0  (unnumbered)
    """
    # Appendix: A, A.1, B.2.3
    m = re.match(r'^([A-Z](?:\.\d+)*)\s', title)
    if m:
        return len(m.group(1).split('.'))
    # Regular: 1, 2.1, 2.3.1
    m = re.match(r'^(\d+(?:\.\d+)*)\s', title)
    if m:
        return len(m.group(1).split('.'))
    return 0  # Title, Abstract, References, etc.


# ── Markdown section parser ─────────────────────────────────────────────────

def parse_markdown_sections(md_text: str) -> list[dict]:
    """Split markdown into sections based on # headers.
    
    Returns list of {title, level, text, paragraphs: [{text, word_count}]}
    """
    # MinerU outputs all headers as single # (level 1 in markdown)
    # We infer actual depth from numbering.
    parts = re.split(r'^(#{1,6}\s+.+)$', md_text, flags=re.MULTILINE)
    
    sections = []
    current_title = "(preamble)"
    current_text = ""
    
    for part in parts:
        header_match = re.match(r'^#{1,6}\s+(.+)$', part)
        if header_match:
            if current_text.strip():
                sections.append(_make_section(current_title, current_text))
            current_title = header_match.group(1).strip()
            current_text = ""
        else:
            current_text += part
    
    if current_text.strip():
        sections.append(_make_section(current_title, current_text))
    
    return sections


def _make_section(title: str, text: str) -> dict:
    """Build a section dict from title and raw text."""
    # Split text into paragraphs (double newline separated)
    raw_paras = re.split(r'\n\s*\n', text.strip())
    paragraphs = []
    for p in raw_paras:
        p = p.strip()
        if not p:
            continue
        words = tokenize_words(p)
        paragraphs.append({
            "text": p,
            "word_count": len(words),
        })
    
    total_words = sum(p["word_count"] for p in paragraphs)
    return {
        "title": title,
        "level": infer_section_level(title),
        "text": text.strip(),
        "word_count": total_words,
        "paragraphs": paragraphs,
    }


# ── Chunking algorithm ──────────────────────────────────────────────────────

def merge_paragraphs_into_chunks(
    sections: list[dict],
    target_words: int = 450,
    min_words: int = 100,
) -> tuple[list[dict], list[dict]]:
    """Merge paragraphs into ~target_words chunks using a two-pass strategy.

    Pass 1: Flatten all paragraphs (keeping section metadata), then greedily
            accumulate until target_words. Prefer breaking at section
            boundaries but allow merging short sections together.
    Pass 2: Merge any trailing runt chunk (< min_words) into its predecessor.

    Returns:
        (chunks, flat_paras) — flat_paras is the per-doc global paragraph
        list, caller uses it to emit paragraph virtual nodes keyed by
        para_idx.
    """
    # ── Pass 0: flatten paragraphs with section metadata ──
    flat_paras = []  # [{text, word_count, section_title, section_level, para_idx}]
    global_para_idx = 0
    for sec in sections:
        for para in sec["paragraphs"]:
            flat_paras.append({
                "text": para["text"],
                "word_count": para["word_count"],
                "section_title": sec["title"],
                "section_level": sec["level"],
                "para_idx": global_para_idx,
            })
            global_para_idx += 1
    
    if not flat_paras:
        return [], []

    # ── Pass 1: greedy chunking ──
    chunks = []
    buf_paras = []
    buf_words = 0
    buf_sections = set()
    
    for i, fp in enumerate(flat_paras):
        # Check if adding this paragraph crosses a major section boundary
        # (level 0 or 1) AND the buffer already has content
        is_major_boundary = (
            buf_paras
            and fp["section_title"] != buf_paras[-1]["section_title"]
            and fp["section_level"] <= 1
            and buf_words >= min_words
        )
        
        if is_major_boundary:
            # Emit current buffer before starting new major section
            _emit_chunk(chunks, buf_paras, buf_words, buf_sections)
            buf_paras = []
            buf_words = 0
            buf_sections = set()
        
        buf_paras.append(fp)
        buf_words += fp["word_count"]
        buf_sections.add(fp["section_title"])
        
        # Emit if we've reached target
        if buf_words >= target_words:
            _emit_chunk(chunks, buf_paras, buf_words, buf_sections)
            buf_paras = []
            buf_words = 0
            buf_sections = set()
    
    # Remaining buffer
    if buf_paras:
        _emit_chunk(chunks, buf_paras, buf_words, buf_sections)
    
    # ── Pass 2: merge runts ──
    merged = []
    for chunk in chunks:
        if (
            merged
            and chunk["word_count"] < min_words
        ):
            # Merge into previous
            prev = merged[-1]
            prev["text"] += "\n\n" + chunk["text"]
            prev["word_count"] += chunk["word_count"]
            prev["paragraph_indices"].extend(chunk["paragraph_indices"])
            prev["section_titles"] = list(
                dict.fromkeys(prev["section_titles"] + chunk["section_titles"])
            )
        else:
            merged.append(chunk)

    return merged, flat_paras


def _emit_chunk(
    chunks: list, buf_paras: list, buf_words: int, buf_sections: set
):
    """Helper to create a chunk dict from buffered paragraphs."""
    # Primary section = section of the first paragraph
    primary_section = buf_paras[0]["section_title"]
    primary_level = buf_paras[0]["section_level"]
    # All sections spanned (ordered by first appearance)
    section_titles = list(dict.fromkeys(fp["section_title"] for fp in buf_paras))
    
    chunks.append({
        "section_title": primary_section,
        "section_level": primary_level,
        "section_titles": section_titles,
        "text": "\n\n".join(fp["text"] for fp in buf_paras),
        "word_count": buf_words,
        "paragraph_indices": [fp["para_idx"] for fp in buf_paras],
    })


# ── Build edges ──────────────────────────────────────────────────────────────

def build_edges(
    doc_id: str,
    chunks: list[dict],
    flat_paras: list[dict],
) -> list[dict]:
    """Build edges between chunks, sections, paragraphs, and sequential neighbors.

    Edge types emitted:
        section_contains_chunk      section → chunk   (existing)
        chunk_sequence              chunk_i → chunk_i+1 (existing)
        chunk_contains_paragraph    chunk → paragraph (new)
        section_contains_paragraph  section → paragraph (new)
    """
    edges: list[dict] = []

    # 1. section_contains_chunk (one section → many chunks if chunk spans)
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        for sec_title in chunk.get("section_titles", [chunk["section_title"]]):
            edges.append({
                "source": sec_title,
                "target": chunk_id,
                "source_type": "section",
                "target_type": "chunk",
                "relation": "section_contains_chunk",
            })

    # 2. chunk_sequence: reading order between adjacent chunks
    for i in range(len(chunks) - 1):
        edges.append({
            "source": f"{doc_id}_chunk_{i}",
            "target": f"{doc_id}_chunk_{i + 1}",
            "source_type": "chunk",
            "target_type": "chunk",
            "relation": "chunk_sequence",
        })

    # 3. chunk_contains_paragraph: chunk → each of its constituent paragraphs
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{i}"
        for para_idx in chunk.get("paragraph_indices", []):
            edges.append({
                "source": chunk_id,
                "target": f"{doc_id}_para_{para_idx}",
                "source_type": "chunk",
                "target_type": "paragraph",
                "relation": "chunk_contains_paragraph",
            })

    # 4. section_contains_paragraph: section → each paragraph directly
    for fp in flat_paras:
        edges.append({
            "source": fp["section_title"],
            "target": f"{doc_id}_para_{fp['para_idx']}",
            "source_type": "section",
            "target_type": "paragraph",
            "relation": "section_contains_paragraph",
        })

    return edges


# ── Find markdown file for a doc ────────────────────────────────────────────

def find_markdown_file(doc_dir: str) -> str | None:
    """Find the best markdown file in a MinerU doc directory.
    
    Prefer hybrid_auto/ subdir, then any .md that isn't formulas*.md.
    """
    candidates = []
    for root, dirs, files in os.walk(doc_dir):
        for f in files:
            if f.endswith('.md') and not f.startswith('formulas'):
                path = os.path.join(root, f)
                # Prefer hybrid path
                if 'hybrid' in root:
                    return path
                candidates.append(path)
    
    return candidates[0] if candidates else None


# ── Main processing ─────────────────────────────────────────────────────────

def process_all_docs(
    mineru_base: str,
    target_words: int,
    min_words: int,
) -> dict:
    """Process all MinerU documents and build chunk + paragraph virtual nodes."""
    engine, _ = _resolve_tokenizer()
    result = {
        "metadata": {
            "source": "MinerU markdown outputs",
            "target_words": target_words,
            "min_words": min_words,
            "tokenizer": engine,
            "script": "scripts/build_chunk_virtual_nodes.py",
            "node_types": ["chunk", "paragraph"],
            "edge_types": [
                "section_contains_chunk",
                "chunk_sequence",
                "chunk_contains_paragraph",
                "section_contains_paragraph",
            ],
        },
        "documents": {},
        "stats": {
            "total_docs_scanned": 0,
            "docs_with_markdown": 0,
            "docs_with_chunks": 0,
            "total_chunks": 0,
            "total_paragraphs": 0,
            "total_edges": 0,
            "total_words_in_chunks": 0,
            "avg_words_per_chunk": 0,
            "avg_paragraphs_per_chunk": 0,
        },
    }
    
    doc_dirs = sorted([
        d for d in os.listdir(mineru_base)
        if not d.startswith('.') and os.path.isdir(os.path.join(mineru_base, d))
    ])
    
    result["stats"]["total_docs_scanned"] = len(doc_dirs)
    
    for doc_id in doc_dirs:
        doc_path = os.path.join(mineru_base, doc_id)
        md_file = find_markdown_file(doc_path)
        
        if not md_file:
            continue
        
        result["stats"]["docs_with_markdown"] += 1
        
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                md_text = f.read()
        except Exception as e:
            print(f"  WARN: Cannot read {md_file}: {e}", file=sys.stderr)
            continue
        
        if len(md_text.strip()) < 50:
            continue
        
        # Parse sections
        sections = parse_markdown_sections(md_text)

        # Merge into chunks (also returns the flat paragraph list)
        chunks, flat_paras = merge_paragraphs_into_chunks(
            sections, target_words, min_words
        )

        if not chunks:
            continue

        result["stats"]["docs_with_chunks"] += 1

        # Build chunk nodes
        nodes = OrderedDict()
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            nodes[chunk_id] = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_idx": i,
                "section_title": chunk["section_title"],
                "section_titles": chunk.get("section_titles", [chunk["section_title"]]),
                "section_level": chunk["section_level"],
                "text": chunk["text"],
                "word_count": chunk["word_count"],
                "paragraph_indices": chunk["paragraph_indices"],
            }

        # Build paragraph virtual nodes (one per flat_paras entry)
        paragraph_nodes = OrderedDict()
        for fp in flat_paras:
            para_id = f"{doc_id}_para_{fp['para_idx']}"
            paragraph_nodes[para_id] = {
                "paragraph_id": para_id,
                "doc_id": doc_id,
                "para_idx": fp["para_idx"],
                "section_title": fp["section_title"],
                "section_level": fp["section_level"],
                "text": fp["text"],
                "word_count": fp["word_count"],
            }

        # Build edges
        edges = build_edges(doc_id, chunks, flat_paras)

        result["documents"][doc_id] = {
            "doc_id": doc_id,
            "source_markdown": os.path.relpath(md_file, "."),
            "num_sections": len(sections),
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
        result["stats"]["total_words_in_chunks"] += sum(
            c["word_count"] for c in chunks
        )
    
    # Compute averages
    if result["stats"]["total_chunks"] > 0:
        result["stats"]["avg_words_per_chunk"] = round(
            result["stats"]["total_words_in_chunks"]
            / result["stats"]["total_chunks"],
            1,
        )
        result["stats"]["avg_paragraphs_per_chunk"] = round(
            result["stats"]["total_paragraphs"]
            / result["stats"]["total_chunks"],
            2,
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build chunk virtual nodes from MinerU markdown outputs"
    )
    parser.add_argument(
        "--target-words", type=int, default=DEFAULT_TARGET_WORDS,
        help=f"Target words per chunk (default: {DEFAULT_TARGET_WORDS})"
    )
    parser.add_argument(
        "--min-words", type=int, default=DEFAULT_MIN_WORDS,
        help=f"Minimum words for standalone chunk (default: {DEFAULT_MIN_WORDS})"
    )
    parser.add_argument(
        "--mineru-base", type=str, default=MINERU_BASE,
        help=f"MinerU output directory (default: {MINERU_BASE})"
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_PATH,
        help=f"Output JSON path (default: {OUTPUT_PATH})"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="auto",
        choices=["auto", "nltk", "regex"],
        help="Word tokenizer backend (default: auto = nltk → regex fallback)"
    )
    args = parser.parse_args()

    # Lock tokenizer choice early so word counts are consistent across docs
    _resolve_tokenizer(args.tokenizer)
    engine, _ = _resolve_tokenizer()

    print(f"Building chunk virtual nodes...")
    print(f"  MinerU base:  {args.mineru_base}")
    print(f"  Target words: {args.target_words}")
    print(f"  Min words:    {args.min_words}")
    print(f"  Tokenizer:    {engine}")
    print(f"  Output:       {args.output}")
    print()

    result = process_all_docs(args.mineru_base, args.target_words, args.min_words)

    # Print stats
    stats = result["stats"]
    print(f"── Results ──")
    print(f"  Docs scanned:       {stats['total_docs_scanned']}")
    print(f"  Docs with markdown: {stats['docs_with_markdown']}")
    print(f"  Docs with chunks:   {stats['docs_with_chunks']}")
    print(f"  Total chunks:       {stats['total_chunks']}")
    print(f"  Total paragraphs:   {stats['total_paragraphs']}")
    print(f"  Total edges:        {stats['total_edges']}")
    print(f"  Avg words/chunk:    {stats['avg_words_per_chunk']}")
    print(f"  Avg paras/chunk:    {stats['avg_paragraphs_per_chunk']}")
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    file_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n  Saved to {args.output} ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()
