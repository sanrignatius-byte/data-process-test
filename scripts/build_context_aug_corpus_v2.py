#!/usr/bin/env python3
"""Build a fine-grained context-augmented corpus (S2 v2 / 4.20 deliverable).

Key design principles (fix over v1):
    1. Fine-grained: use *chunk* summary (1:1-4 children) instead of section
       summary (1:10-30 children) to avoid within-section homogenization.
    2. Protect short elements: only inject on passages whose original text has
       at least --min-orig-words words. Figure captions / formulas stay raw.
    3. Concise prefix: total prefix capped to --max-prefix-words (default 25);
       avoids drowning original signal.
    4. Per-passage disambiguation: the first retrieval_hint of the chunk
       summary is added, not the full summary text, keeping siblings slightly
       different at least in keyword emphasis.

    text' = "[Ctx {section_title}: {short_hint_or_summary}] {original_text}"

Usage:
    python scripts/build_context_aug_corpus_v2.py \\
        --summary-graph data/01_graphs/llm_summary_virtual_nodes.json \\
        --chunk-graph   data/01_graphs/chunk_virtual_nodes.json \\
        --elements-graph data/01_graphs/multimodal_elements.json \\
        --in-corpus  .../corpus_v1_enriched.jsonl \\
        --out-corpus .../corpus_v1_chunk_ctx.jsonl \\
        --scope chunk \\
        --min-orig-words 80 \\
        --max-prefix-words 25
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------- I/O ----------
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


# ---------- pid -> (chunk_id, section_title) ----------
def build_pid_context_mapping(
    chunk_data: Dict[str, Any],
    elem_data: Dict[str, Any],
    valid_pids: Set[str],
) -> Tuple[Dict[str, Tuple[str, str, str]], Dict[str, int]]:
    """Return pid -> (doc_id, chunk_id, section_title)."""
    doc_pos_to_chunk: Dict[str, Dict[int, str]] = {}
    doc_chunk_to_section: Dict[str, Dict[str, str]] = {}
    for doc_id, doc in chunk_data.get("documents", {}).items():
        pos_to_chunk: Dict[int, str] = {}
        c2s: Dict[str, str] = {}
        for nid, node in doc.get("nodes", {}).items():
            c2s[nid] = node.get("section_title", "") or ""
            for pidx in node.get("paragraph_indices", []):
                pos_to_chunk[pidx] = nid
        doc_pos_to_chunk[doc_id] = pos_to_chunk
        doc_chunk_to_section[doc_id] = c2s

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
    pid_ctx: Dict[str, Tuple[str, str, str]] = {}
    for pid in valid_pids:
        for doc_id in doc_type0_to_pos:
            if not pid.startswith(doc_id + "_"):
                continue
            rest = pid[len(doc_id) + 1:]
            if rest.startswith(doc_id + "_"):
                rest = rest[len(doc_id) + 1:]
            parts = rest.rsplit("_", 1)
            chunk_id: Optional[str] = None
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
                pid_ctx[pid] = (doc_id, chunk_id, sec)
            break
    stats = {"valid_pids": len(valid_pids), "pid_with_context": len(pid_ctx)}
    return pid_ctx, stats


# ---------- summary lookup ----------
def build_summary_lookups(summary_data: Dict[str, Any]) -> Tuple[
    Dict[Tuple[str, str], Dict[str, Any]],
    Dict[Tuple[str, str], Dict[str, Any]],
]:
    """Returns:
        chunk_lookup:   (doc_id, chunk_id)      -> chunk_summary record
        section_lookup: (doc_id, section_title) -> section_summary record
    """
    chunk_lu: Dict[Tuple[str, str], Dict[str, Any]] = {}
    section_lu: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for doc_id, doc in summary_data.get("documents", {}).items():
        for sid, sm in (doc.get("chunk_summaries") or {}).items():
            cid = sm.get("source_chunk", "")
            if cid:
                cur = chunk_lu.get((doc_id, cid))
                if cur is None or len(sm.get("text", "")) > len(cur.get("text", "")):
                    chunk_lu[(doc_id, cid)] = sm
        for sid, sm in (doc.get("section_summaries") or {}).items():
            sec = sm.get("source_section", "")
            if sec:
                cur = section_lu.get((doc_id, sec))
                if cur is None or len(sm.get("text", "")) > len(cur.get("text", "")):
                    section_lu[(doc_id, sec)] = sm
    return chunk_lu, section_lu


def truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return ""
    w = text.split()
    if len(w) <= max_words:
        return text
    return " ".join(w[:max_words]) + "..."


def make_prefix(
    rec: Optional[Dict[str, Any]],
    section_title: str,
    max_prefix_words: int,
    style: str,
) -> str:
    if rec is None and not section_title:
        return ""
    sec = (section_title or "").strip() or "ctx"
    if style == "hint":
        # shortest: just 1 retrieval hint (or fallback to a few summary words)
        hints = [h for h in (rec.get("retrieval_hints") or []) if h] if rec else []
        if hints:
            body = hints[0]
        else:
            body = (rec.get("text") if rec else "") or ""
        body = truncate_words(body, max_prefix_words)
        return f"[Ctx {sec}: {body}]" if body else f"[Ctx {sec}]"
    # style == "summary": short summary text only
    body = (rec.get("text") if rec else "") or ""
    body = truncate_words(body, max_prefix_words)
    return f"[Ctx {sec}: {body}]" if body else f"[Ctx {sec}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-graph", default="data/01_graphs/llm_summary_virtual_nodes.json")
    ap.add_argument("--chunk-graph", default="data/01_graphs/chunk_virtual_nodes.json")
    ap.add_argument("--elements-graph", default="data/01_graphs/multimodal_elements.json")
    ap.add_argument("--in-corpus", required=True)
    ap.add_argument("--out-corpus", required=True)
    ap.add_argument("--scope", choices=["chunk", "section"], default="chunk",
                    help="Granularity of summary used. chunk is finer (1:~3 children).")
    ap.add_argument("--style", choices=["hint", "summary"], default="hint",
                    help="hint = one retrieval_hint (most discriminative); summary = short summary text")
    ap.add_argument("--min-orig-words", type=int, default=80,
                    help="Only inject on passages whose raw text has at least this many words")
    ap.add_argument("--max-prefix-words", type=int, default=25,
                    help="Truncate prefix body to this many words")
    args = ap.parse_args()

    def resolve(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p

    in_path = resolve(args.in_corpus)
    out_path = resolve(args.out_corpus)

    print(f"[info] loading corpus {in_path}", flush=True)
    rows = load_jsonl(in_path)
    valid_pids = {str(r["passage_id"]) for r in rows}
    print(f"[info] passages: {len(rows)}", flush=True)

    summary_data = json.loads(resolve(args.summary_graph).read_bytes())
    chunk_data = json.loads(resolve(args.chunk_graph).read_bytes())
    elem_data = json.loads(resolve(args.elements_graph).read_bytes())

    pid_ctx, pidstats = build_pid_context_mapping(chunk_data, elem_data, valid_pids)
    chunk_lu, section_lu = build_summary_lookups(summary_data)
    print(f"[info] pid->context: {pidstats} "
          f"chunk_summaries={len(chunk_lu)} section_summaries={len(section_lu)}", flush=True)

    # audit: group sizes for the chosen scope
    if args.scope == "chunk":
        group_of: Dict[str, int] = defaultdict(int)
        for pid, (doc_id, cid, _) in pid_ctx.items():
            if (doc_id, cid) in chunk_lu:
                group_of[(doc_id, cid)] += 1  # type: ignore[index]
        sizes = sorted(group_of.values())
        if sizes:
            print(f"[info] chunk group sizes (passages per chunk): "
                  f"min={sizes[0]} p50={sizes[len(sizes)//2]} p95={sizes[int(len(sizes)*0.95)]} max={sizes[-1]} mean={sum(sizes)/len(sizes):.2f}",
                  flush=True)
    else:
        group_of = defaultdict(int)
        for pid, (doc_id, _, sec) in pid_ctx.items():
            if (doc_id, sec) in section_lu:
                group_of[(doc_id, sec)] += 1  # type: ignore[index]
        sizes = sorted(group_of.values())
        if sizes:
            print(f"[info] section group sizes: "
                  f"min={sizes[0]} p50={sizes[len(sizes)//2]} p95={sizes[int(len(sizes)*0.95)]} max={sizes[-1]} mean={sum(sizes)/len(sizes):.2f}",
                  flush=True)

    out_rows: List[Dict[str, Any]] = []
    n_aug = n_short = n_no_ctx = n_no_summary = 0
    for r in rows:
        pid = str(r["passage_id"])
        orig = (r.get("text") or "").strip()
        orig_wc = len(orig.split())
        if orig_wc < args.min_orig_words:
            out_rows.append(r)
            n_short += 1
            continue
        info = pid_ctx.get(pid)
        if info is None:
            out_rows.append(r)
            n_no_ctx += 1
            continue
        doc_id, cid, sec = info
        rec = None
        if args.scope == "chunk":
            rec = chunk_lu.get((doc_id, cid))
        else:
            rec = section_lu.get((doc_id, sec))
        if rec is None:
            out_rows.append(r)
            n_no_summary += 1
            continue
        prefix = make_prefix(rec, sec, args.max_prefix_words, args.style)
        if not prefix:
            out_rows.append(r)
            n_no_summary += 1
            continue
        nr = dict(r)
        nr["text"] = prefix + " " + orig
        nr["section"] = sec
        nr["text_source"] = (r.get("text_source", "") or "raw") + f"+{args.scope}_ctx"
        out_rows.append(nr)
        n_aug += 1

    write_jsonl(out_path, out_rows)
    stats = {
        "in_corpus": str(in_path),
        "out_corpus": str(out_path),
        "total_passages": len(rows),
        "augmented_passages": n_aug,
        "skipped_short_passage": n_short,
        "skipped_no_context_mapping": n_no_ctx,
        "skipped_no_summary": n_no_summary,
        "augmented_pct": round(n_aug / max(1, len(rows)) * 100, 2),
        "config": {
            "scope": args.scope,
            "style": args.style,
            "min_orig_words": args.min_orig_words,
            "max_prefix_words": args.max_prefix_words,
        },
    }
    stats_path = out_path.with_suffix(out_path.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print("[done]", json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
