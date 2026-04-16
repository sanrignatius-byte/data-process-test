#!/usr/bin/env python3
"""
build_graph_augmented_corpus.py
================================
Build a graph-augmented retrieval corpus for the M4query_v1 dataset by enriching
element text and injecting chunk / summary virtual nodes.

WHY DENSE RETRIEVAL BASELINE IS SO LOW (Recall@100 ≈ 9.5%):
  - 899 formula passages contain only raw LaTeX math (e.g. "D(Mx,My) ≤ d(x,y)")
  - 842 figure passages contain only "[Image: xxx.jpg]" — zero semantic text
  - enriched_content from MODORA only covers 30/1798 passages
  - Queries are natural-language multi-hop; positives are opaque math/images
  → Dense model has almost no text signal to match on

STRATEGY (three corpus variants for ablation):
  v1_enriched_elements:
    Keep original passage_ids; enrich text with context_before + caption +
    content + context_after + enriched_content (where available).
    Qrels unchanged — same passage_ids.

  v2_with_chunks:
    v1 + add chunk virtual nodes from chunk_virtual_nodes.json as extra passages.
    Qrels extended: for each positive element e, chunks in the same doc whose
    section_title overlaps with the element's section are added as soft positives
    (relevance = 1, same as originals).

  v3_with_summaries:
    v2 + add LLM summary nodes from llm_summary_virtual_nodes.json.
    Qrels extended: section summaries from the same section as a positive
    element are added as soft positives.

Outputs (in --output-dir):
  corpus_v1_enriched.jsonl
  corpus_v2_chunks.jsonl
  corpus_v3_summaries.jsonl
  qrels_v1.jsonl              (same as original, renamed for clarity)
  qrels_v2_chunks.jsonl
  qrels_v3_summaries.jsonl
  build_stats.json

Usage:
  python scripts/build_graph_augmented_corpus.py
  python scripts/build_graph_augmented_corpus.py --output-dir data/05_eval/dense_retrieval/augmented
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_M4_DIR         = "data/03_queries/M4query_v1"
DEFAULT_CHUNK_FILE     = "data/01_graphs/chunk_virtual_nodes.json"
DEFAULT_SUMMARY_FILE   = "data/01_graphs/llm_summary_virtual_nodes.json"
DEFAULT_ENRICHED_FILES = [
    "data/02_enriched/hub_candidates_enriched_v4_intra_doc.json",
    "data/02_enriched/hub_candidates_enriched_v3_intra_doc.json",
    "data/02_enriched/m2_diverse_candidates_intra_doc.json",
]
DEFAULT_OUTPUT_DIR = "data/05_eval/dense_retrieval/augmented"

MAX_CONTEXT_CHARS = 400   # context_before / context_after truncation


# ── IO helpers ────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] {path}:{ln} bad json: {e}", file=sys.stderr)
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── Build enriched-element text ───────────────────────────────────────────────

def load_enriched_index(enriched_files: List[str]) -> Dict[str, Dict[str, str]]:
    """Build element_id → {enriched_title, enriched_content} from all enriched files."""
    index: Dict[str, Dict[str, str]] = {}
    for fname in enriched_files:
        p = PROJECT_ROOT / fname
        if not p.exists():
            print(f"[info] enriched file not found, skipping: {p}")
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            print(f"[warn] failed to load {p}: {e}")
            continue
        pairs = d.get("pairs", d.get("candidates", []))
        for pair in pairs:
            for key in ("element_a", "element_b"):
                e = pair.get(key, {})
                eid = e.get("element_id", "")
                ec = (e.get("enriched_content") or "").strip()
                if eid and ec and eid not in index:
                    index[eid] = {
                        "enriched_title":   (e.get("enriched_title") or "").strip(),
                        "enriched_content": ec,
                    }
    print(f"[info] enriched index: {len(index)} elements")
    return index


def load_graph_element_index(m4_graphs_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load multimodal_elements.json → element_id → full element dict."""
    p = m4_graphs_dir / "multimodal_elements.json"
    if not p.exists():
        print(f"[warn] multimodal_elements.json not found: {p}")
        return {}
    d = json.load(open(p, encoding="utf-8"))
    index: Dict[str, Dict[str, Any]] = {}
    for doc_id, doc in d.get("documents", {}).items():
        for eid, e in doc.get("elements", {}).items():
            index[eid] = e
    print(f"[info] graph element index: {len(index)} elements")
    return index


def build_element_text(
    corpus_row: Dict[str, Any],
    graph_elem: Optional[Dict[str, Any]],
    enrich_entry: Optional[Dict[str, str]],
) -> str:
    """Compose the richest possible text for a corpus element.

    Priority stack (richest first):
      1. enriched_title + enriched_content (from MODORA enrichment)
      2. caption + content + context_before + context_after (from graph element)
      3. original corpus text (last resort)
    """
    parts: List[str] = []

    if enrich_entry:
        title = enrich_entry.get("enriched_title", "").strip()
        content = enrich_entry.get("enriched_content", "").strip()
        if title:
            parts.append(title)
        if content:
            parts.append(content)

    if graph_elem and not parts:
        caption = (graph_elem.get("caption") or "").strip()
        content_raw = (graph_elem.get("content") or "").strip()
        ctx_before = (graph_elem.get("context_before") or "").strip()[:MAX_CONTEXT_CHARS]
        ctx_after  = (graph_elem.get("context_after") or "").strip()[:MAX_CONTEXT_CHARS]
        if caption:
            parts.append(caption)
        if content_raw:
            parts.append(content_raw)
        if ctx_before:
            parts.append(ctx_before)
        if ctx_after:
            parts.append(ctx_after)

    if not parts:
        # Fall back to whatever raw text is already in the corpus row
        fallback = (corpus_row.get("text") or "").strip()
        # For [Image: ...] entries, try context from graph_elem
        if graph_elem and (fallback.startswith("[Image:") or not fallback):
            ctx_before = (graph_elem.get("context_before") or "").strip()[:MAX_CONTEXT_CHARS]
            ctx_after  = (graph_elem.get("context_after") or "").strip()[:MAX_CONTEXT_CHARS]
            label = graph_elem.get("label", "")
            if ctx_before or ctx_after:
                parts.extend(filter(None, [label, ctx_before, ctx_after]))
        if not parts:
            parts.append(fallback)

    return " ".join(parts).strip()


# ── Enrich corpus (v1) ────────────────────────────────────────────────────────

def build_v1_enriched(
    corpus_rows: List[Dict[str, Any]],
    graph_elem_index: Dict[str, Dict[str, Any]],
    enrich_index: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Return (enriched_corpus_rows, pid_to_section_title)."""
    enriched: List[Dict[str, Any]] = []
    pid_to_section: Dict[str, str] = {}
    n_enriched = n_context = n_fallback = 0

    for row in corpus_rows:
        pid = row["passage_id"]
        doc_id = row.get("doc_id", "")

        # Map corpus passage_id → graph element_id by stripping duplicate doc prefix
        short_eid = pid[len(doc_id) + 1:] if pid.startswith(doc_id + "_") else pid
        graph_elem = graph_elem_index.get(short_eid)
        enrich_entry = enrich_index.get(short_eid)

        new_text = build_element_text(row, graph_elem, enrich_entry)

        if enrich_entry:
            n_enriched += 1
        elif graph_elem and (
            graph_elem.get("context_before") or graph_elem.get("context_after")
        ):
            n_context += 1
        else:
            n_fallback += 1

        # Track section for qrels extension
        section = ""
        if graph_elem:
            # context_before often starts with a section header
            cb = (graph_elem.get("context_before") or "")
            first_line = cb.strip().split("\n")[0].strip()
            if first_line and len(first_line) < 120:
                section = first_line
        pid_to_section[pid] = section

        enriched.append({
            **row,
            "text": new_text,
            "text_source": (
                "enriched_content" if enrich_entry
                else ("context" if n_context > 0 else "raw")
            ),
        })

    print(
        f"[v1] enriched: {n_enriched}, context-augmented: {n_context}, "
        f"raw fallback: {n_fallback} / {len(corpus_rows)} total"
    )
    return enriched, pid_to_section


# ── Add chunk nodes (v2) ──────────────────────────────────────────────────────

def load_chunk_virtual_nodes(
    chunk_file: Path,
    doc_ids: Set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Load chunk nodes for the target doc_ids.

    Returns (chunk_corpus_rows, chunk_id_to_section).
    """
    if not chunk_file.exists():
        print(f"[warn] chunk_virtual_nodes not found: {chunk_file}")
        return [], {}

    data = json.load(open(chunk_file, encoding="utf-8"))
    documents = data.get("documents", {})

    chunk_rows: List[Dict[str, Any]] = []
    chunk_id_to_section: Dict[str, str] = {}

    for doc_id in sorted(doc_ids):
        doc = documents.get(doc_id)
        if not doc:
            continue
        for chunk_id, chunk in doc.get("nodes", {}).items():
            text = chunk.get("text", "").strip()
            if not text:
                continue
            sec = chunk.get("section_title", "")
            chunk_id_to_section[chunk_id] = sec
            chunk_rows.append({
                "passage_id": chunk_id,
                "doc_id": doc_id,
                "text": text,
                "type": "chunk",
                "section": sec,
                "section_level": chunk.get("section_level"),
                "word_count": chunk.get("word_count"),
            })

    print(f"[v2] loaded {len(chunk_rows)} chunk passages from {len(doc_ids)} docs")
    return chunk_rows, chunk_id_to_section


# ── Add summary nodes (v3) ────────────────────────────────────────────────────

def load_summary_nodes(
    summary_file: Path,
    doc_ids: Set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Load LLM summary nodes for the target doc_ids.

    Returns (summary_corpus_rows, summary_id_to_section).
    """
    if not summary_file.exists():
        print(f"[warn] llm_summary_virtual_nodes not found: {summary_file}")
        return [], {}

    data = json.load(open(summary_file, encoding="utf-8"))
    documents = data.get("documents", {})

    summary_rows: List[Dict[str, Any]] = []
    summary_id_to_section: Dict[str, str] = {}

    for doc_id in sorted(doc_ids):
        doc = documents.get(doc_id)
        if not doc:
            continue
        for sum_id, s in doc.get("section_summaries", {}).items():
            text = s.get("text", "").strip()
            sec = s.get("source_section", "")
            if not text:
                continue
            summary_id_to_section[sum_id] = sec
            summary_rows.append({
                "passage_id": sum_id,
                "doc_id": doc_id,
                "text": text,
                "type": "section_summary",
                "section": sec,
            })
        for sum_id, s in doc.get("chunk_summaries", {}).items():
            text = s.get("text", "").strip()
            sec = s.get("section_title", "")
            if not text:
                continue
            summary_id_to_section[sum_id] = sec
            summary_rows.append({
                "passage_id": sum_id,
                "doc_id": doc_id,
                "text": text,
                "type": "chunk_summary",
                "section": sec,
            })

    print(f"[v3] loaded {len(summary_rows)} summary passages from {len(doc_ids)} docs")
    return summary_rows, summary_id_to_section


# ── Extend qrels ──────────────────────────────────────────────────────────────

def extend_qrels(
    original_qrels: List[Dict[str, Any]],
    corpus_rows: List[Dict[str, Any]],
    pid_to_section: Dict[str, str],
    extra_id_to_section: Dict[str, str],
    doc_id_of_extra: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Add soft positives: extra passages from the same doc+section as positives.

    For each (query_id, positive_pid) in qrels:
      - find doc_id and section of positive_pid
      - find extra passages with same doc_id AND overlapping section
      - add them as additional qrels entries (relevance=1)

    Overlap rule: section A and section B overlap if one is a prefix of the other
    (handles sub-sections) OR if A == B.
    """
    # Build lookup: (doc_id, section_title) → list of extra passage_ids
    section_to_extras: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for extra_id, sec in extra_id_to_section.items():
        doc_id = doc_id_of_extra.get(extra_id, "")
        if doc_id and sec:
            section_to_extras[(doc_id, sec)].append(extra_id)

    # doc_id lookup for original corpus passages
    pid_to_docid: Dict[str, str] = {}
    for row in corpus_rows:
        pid_to_docid[row["passage_id"]] = row.get("doc_id", "")

    def sections_overlap(s1: str, s2: str) -> bool:
        if not s1 or not s2:
            return False
        s1l, s2l = s1.lower(), s2.lower()
        return s1l == s2l or s1l.startswith(s2l) or s2l.startswith(s1l)

    extended = list(original_qrels)
    existing: Set[Tuple[str, str]] = {
        (r["query_id"], r["passage_id"]) for r in original_qrels
    }

    added = 0
    for row in original_qrels:
        qid = row["query_id"]
        pid = row["passage_id"]
        doc_id = pid_to_docid.get(pid, "")
        pos_section = pid_to_section.get(pid, "")

        if not doc_id:
            continue

        for (edoc, esec), eids in section_to_extras.items():
            if edoc != doc_id:
                continue
            if not sections_overlap(pos_section, esec):
                continue
            for eid in eids:
                key = (qid, eid)
                if key not in existing:
                    existing.add(key)
                    extended.append({
                        "query_id": qid,
                        "passage_id": eid,
                        "relevance": 1,
                        "is_soft_positive": True,
                    })
                    added += 1

    print(f"[qrels] added {added} soft positives → {len(extended)} total")
    return extended


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Build graph-augmented corpus variants")
    ap.add_argument("--m4-dir", default=DEFAULT_M4_DIR)
    ap.add_argument("--chunk-file", default=DEFAULT_CHUNK_FILE)
    ap.add_argument("--summary-file", default=DEFAULT_SUMMARY_FILE)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument(
        "--enriched-files",
        nargs="*",
        default=DEFAULT_ENRICHED_FILES,
        help="Enriched candidate JSON files",
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        choices=["v1", "v2", "v3"],
        default=["v1", "v2", "v3"],
        help="Which corpus variants to build",
    )
    args = ap.parse_args()

    m4_dir = PROJECT_ROOT / args.m4_dir
    chunk_file = PROJECT_ROOT / args.chunk_file
    summary_file = PROJECT_ROOT / args.summary_file
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load base data ──
    corpus_rows = load_jsonl(m4_dir / "corpus.jsonl")
    orig_qrels   = load_jsonl(m4_dir / "qrels.jsonl")

    # doc_ids in the M4 dataset
    doc_ids: Set[str] = {r["doc_id"] for r in corpus_rows if r.get("doc_id")}
    print(f"[info] {len(corpus_rows)} corpus passages, {len(orig_qrels)} qrels, "
          f"{len(doc_ids)} docs")

    # ── Load enrichment sources ──
    enrich_index    = load_enriched_index([str(PROJECT_ROOT / f) for f in args.enriched_files])
    graph_elem_idx  = load_graph_element_index(m4_dir / "graphs")

    stats: Dict[str, Any] = {
        "base_corpus": len(corpus_rows),
        "base_qrels": len(orig_qrels),
        "enriched_element_count": len(enrich_index),
        "graph_element_count": len(graph_elem_idx),
    }

    # ── v1: enriched elements ──
    if "v1" in args.variants:
        print("\n── Building v1 (enriched element text) ──")
        v1_corpus, pid_to_section = build_v1_enriched(
            corpus_rows, graph_elem_idx, enrich_index
        )
        write_jsonl(v1_corpus, out_dir / "corpus_v1_enriched.jsonl")
        # qrels unchanged for v1 (same passage_ids)
        write_jsonl(orig_qrels, out_dir / "qrels_v1.jsonl")
        stats["v1_corpus"] = len(v1_corpus)
        stats["v1_qrels"] = len(orig_qrels)
        print(f"  → corpus_v1_enriched.jsonl ({len(v1_corpus)} passages)")
    else:
        v1_corpus, pid_to_section = build_v1_enriched(
            corpus_rows, graph_elem_idx, enrich_index
        )

    # ── v2: + chunk virtual nodes ──
    if "v2" in args.variants:
        print("\n── Building v2 (+ chunk nodes) ──")
        chunk_rows, chunk_to_section = load_chunk_virtual_nodes(chunk_file, doc_ids)
        chunk_to_docid = {
            cid: r["doc_id"] for cid, r in zip(
                [r["passage_id"] for r in chunk_rows], chunk_rows
            )
        }
        v2_corpus = v1_corpus + chunk_rows
        write_jsonl(v2_corpus, out_dir / "corpus_v2_chunks.jsonl")

        v2_qrels = extend_qrels(
            orig_qrels,
            corpus_rows,
            pid_to_section,
            extra_id_to_section=chunk_to_section,
            doc_id_of_extra=chunk_to_docid,
        )
        write_jsonl(v2_qrels, out_dir / "qrels_v2_chunks.jsonl")
        stats["v2_corpus"] = len(v2_corpus)
        stats["v2_qrels"] = len(v2_qrels)
        print(f"  → corpus_v2_chunks.jsonl ({len(v2_corpus)} passages)")
    else:
        chunk_rows, chunk_to_section = [], {}
        chunk_to_docid = {}

    # ── v3: + summary nodes ──
    if "v3" in args.variants:
        print("\n── Building v3 (+ summary nodes) ──")
        sum_rows, sum_to_section = load_summary_nodes(summary_file, doc_ids)
        sum_to_docid = {r["passage_id"]: r["doc_id"] for r in sum_rows}

        v3_base_corpus = v1_corpus + chunk_rows + sum_rows
        write_jsonl(v3_base_corpus, out_dir / "corpus_v3_summaries.jsonl")

        # Extend from v2 qrels (or original if v2 not built)
        base_qrels_for_v3 = v2_qrels if "v2" in args.variants else orig_qrels
        v3_qrels = extend_qrels(
            base_qrels_for_v3,
            corpus_rows,
            pid_to_section,
            extra_id_to_section=sum_to_section,
            doc_id_of_extra=sum_to_docid,
        )
        write_jsonl(v3_qrels, out_dir / "qrels_v3_summaries.jsonl")
        stats["v3_corpus"] = len(v3_base_corpus)
        stats["v3_qrels"] = len(v3_qrels)
        print(f"  → corpus_v3_summaries.jsonl ({len(v3_base_corpus)} passages)")

    # Copy queries (unchanged across all variants)
    import shutil
    shutil.copy(m4_dir / "queries.jsonl", out_dir / "queries.jsonl")

    # Save stats
    stats_path = out_dir / "build_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n── Summary ──")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nOutputs in: {out_dir}")


if __name__ == "__main__":
    main()
