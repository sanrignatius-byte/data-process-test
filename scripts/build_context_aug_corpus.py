#!/usr/bin/env python3
"""Build a section-context-augmented corpus (S2 / 4.20 deliverable).

Concept (orthogonal to S1 shadow boost):
    Each passage in the corpus is prepended with its parent section's
    LLM-generated summary (truncated). This injects high-level semantic context
    directly into the embedding without changing the candidate pool size.

    text' = "[Section: {section_title}] {section_summary[:N words]}\n\n{original_text}"

Why prepend (not append):
    Sentence-Transformers truncates at max_seq_length; the prefix carries the
    discriminative high-level topic so cosine similarity benefits even when the
    original passage is long.

Why section summary (not chunk summary):
    Chunk summaries cover ~1 chunk = a few hundred words, which often overlaps
    with the passage itself. Section summaries (1 per top-level section) provide
    *complementary* abstract context.

Usage:
    python scripts/build_context_aug_corpus.py \
        --summary-graph data/01_graphs/llm_summary_virtual_nodes.json \
        --chunk-graph   data/01_graphs/chunk_virtual_nodes.json \
        --elements-graph data/01_graphs/multimodal_elements.json \
        --in-corpus  data/05_eval/dense_retrieval/rebuilt_20260417/augmented/corpus_v1_enriched.jsonl \
        --out-corpus data/05_eval/dense_retrieval/augmented_v2/corpus_v1_sec_context.jsonl \
        --max-summary-words 60
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def build_pid_to_section(
    chunk_data: Dict[str, Any],
    elem_data: Dict[str, Any],
    valid_pids: Set[str],
) -> Tuple[Dict[str, Tuple[str, str]], Dict[str, int]]:
    """Return (pid -> (doc_id, section_title), stats)."""
    # doc -> position_idx -> chunk_id
    doc_pos_to_chunk: Dict[str, Dict[int, str]] = {}
    # doc -> chunk_id -> section_title (use the chunk node itself which has section_title)
    doc_chunk_to_section: Dict[str, Dict[str, str]] = {}
    for doc_id, doc in chunk_data.get("documents", {}).items():
        pos_to_chunk: Dict[int, str] = {}
        chunk_to_sec: Dict[str, str] = {}
        for nid, node in doc.get("nodes", {}).items():
            chunk_to_sec[nid] = node.get("section_title", "") or ""
            for pidx in node.get("paragraph_indices", []):
                pos_to_chunk[pidx] = nid
        doc_pos_to_chunk[doc_id] = pos_to_chunk
        doc_chunk_to_section[doc_id] = chunk_to_sec

    doc_type0_to_pos: Dict[str, Dict[Tuple[str, int], int]] = {}
    doc_elemid_to_pos: Dict[str, Dict[str, int]] = {}
    for doc_id, doc in elem_data.get("documents", {}).items():
        type0_map: Dict[Tuple[str, int], int] = {}
        eid_map: Dict[str, int] = {}
        for elem_id, elem in doc.get("elements", {}).items():
            pidx = elem.get("position_idx")
            if pidx is None:
                continue
            eid_map[elem_id] = pidx
            parts = elem_id.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                one_idx = int(parts[1])
                prefix = parts[0]
                etype = (
                    prefix[len(doc_id) + 1:]
                    if prefix.startswith(doc_id + "_")
                    else prefix.rsplit("_", 1)[-1]
                )
                type0_map[(etype, one_idx - 1)] = pidx
        doc_type0_to_pos[doc_id] = type0_map
        doc_elemid_to_pos[doc_id] = eid_map

    type_alias = {"fig": "figure"}
    pid_to_section: Dict[str, Tuple[str, str]] = {}
    for pid in valid_pids:
        for doc_id in doc_type0_to_pos:
            if not pid.startswith(doc_id + "_"):
                continue
            rest = pid[len(doc_id) + 1:]
            if rest.startswith(doc_id + "_"):
                rest = rest[len(doc_id) + 1:]
            parts = rest.rsplit("_", 1)
            chunk_id = None
            if len(parts) == 2 and parts[1].isdigit():
                raw_type = parts[0]
                n = int(parts[1])
                etype = type_alias.get(raw_type, raw_type)
                pidx = doc_type0_to_pos[doc_id].get((etype, n))
                if pidx is not None:
                    chunk_id = doc_pos_to_chunk.get(doc_id, {}).get(pidx)
                if chunk_id is None:
                    elem_id = f"{doc_id}_{raw_type}_{n}"
                    pidx2 = doc_elemid_to_pos[doc_id].get(elem_id)
                    if pidx2 is not None:
                        chunk_id = doc_pos_to_chunk.get(doc_id, {}).get(pidx2)
            if chunk_id:
                sec = doc_chunk_to_section.get(doc_id, {}).get(chunk_id, "")
                pid_to_section[pid] = (doc_id, sec)
            break
    stats = {
        "valid_pids": len(valid_pids),
        "pid_with_section": len(pid_to_section),
    }
    return pid_to_section, stats


def build_section_summary_lookup(
    summary_data: Dict[str, Any],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """(doc_id, section_title) -> section_summary record."""
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for doc_id, doc in summary_data.get("documents", {}).items():
        for sid, sm in (doc.get("section_summaries") or {}).items():
            sec = sm.get("source_section", "")
            if sec:
                # If multiple summaries map to same section pick the longest text
                cur = lookup.get((doc_id, sec))
                if cur is None or len(sm.get("text", "")) > len(cur.get("text", "")):
                    lookup[(doc_id, sec)] = sm
    return lookup


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-graph", default="data/01_graphs/llm_summary_virtual_nodes.json")
    ap.add_argument("--chunk-graph", default="data/01_graphs/chunk_virtual_nodes.json")
    ap.add_argument("--elements-graph", default="data/01_graphs/multimodal_elements.json")
    ap.add_argument("--in-corpus", required=True)
    ap.add_argument("--out-corpus", required=True)
    ap.add_argument("--max-summary-words", type=int, default=60)
    ap.add_argument("--include-keywords", action="store_true",
                    help="Append top section keywords to the prefix")
    args = ap.parse_args()

    def resolve(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p

    in_path = resolve(args.in_corpus)
    out_path = resolve(args.out_corpus)
    sum_path = resolve(args.summary_graph)
    chunk_path = resolve(args.chunk_graph)
    elem_path = resolve(args.elements_graph)

    print(f"[info] loading corpus  {in_path}", flush=True)
    rows = load_jsonl(in_path)
    valid_pids = {str(r["passage_id"]) for r in rows}
    print(f"[info] passages: {len(rows)}", flush=True)

    print(f"[info] loading summary graph: {sum_path}", flush=True)
    summary_data = json.loads(sum_path.read_bytes())
    print(f"[info] loading chunk graph:   {chunk_path}", flush=True)
    chunk_data = json.loads(chunk_path.read_bytes())
    print(f"[info] loading elements:      {elem_path}", flush=True)
    elem_data = json.loads(elem_path.read_bytes())

    pid2sec, pidstats = build_pid_to_section(chunk_data, elem_data, valid_pids)
    print(f"[info] pid->section: {pidstats}", flush=True)
    sec_lookup = build_section_summary_lookup(summary_data)
    print(f"[info] section_summaries indexed: {len(sec_lookup)}", flush=True)

    out_rows: List[Dict[str, Any]] = []
    n_aug = n_no_section = n_no_summary = 0
    for r in rows:
        pid = str(r["passage_id"])
        original_text = r.get("text", "") or ""
        info = pid2sec.get(pid)
        if info is None:
            out_rows.append(r)
            n_no_section += 1
            continue
        doc_id, sec_title = info
        sm = sec_lookup.get((doc_id, sec_title))
        if sm is None or not sec_title:
            out_rows.append(r)
            n_no_summary += 1
            continue
        sec_text = truncate_words(sm.get("text", ""), args.max_summary_words)
        prefix = f"[Section: {sec_title}] {sec_text}"
        if args.include_keywords:
            kws = [k for k in (sm.get("keywords") or []) if k][:5]
            if kws:
                prefix += " | Keywords: " + ", ".join(kws)
        new_text = prefix + "\n\n" + original_text
        nr = dict(r)
        nr["text"] = new_text
        nr["section"] = sec_title
        nr["text_source"] = (r.get("text_source", "") or "raw") + "+sec_ctx"
        out_rows.append(nr)
        n_aug += 1

    write_jsonl(out_path, out_rows)
    stats = {
        "in_corpus": str(in_path),
        "out_corpus": str(out_path),
        "total_passages": len(rows),
        "augmented_passages": n_aug,
        "skipped_no_section_mapping": n_no_section,
        "skipped_no_section_summary": n_no_summary,
        "augmented_pct": round(n_aug / max(1, len(rows)) * 100, 2),
        "max_summary_words": args.max_summary_words,
    }
    stats_path = out_path.with_suffix(out_path.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print("[done]", json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
