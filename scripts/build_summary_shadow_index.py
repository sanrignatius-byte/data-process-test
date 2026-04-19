#!/usr/bin/env python3
"""Build a *shadow* rerank index from LLM-generated section/chunk summaries.

Motivation (4.18):
    The 4.16 v3_summaries experiment failed because summaries were INSERTED
    into the corpus, diluting the candidate pool and crashing R@1 (0.0782 -> 0.0272).
    True virtual nodes (LLM summaries) should not appear as candidates; instead
    they act as a side index that *boosts* their child passages when the query
    semantically matches the summary.

This script:
    1. Reads ``llm_summary_virtual_nodes.json`` (1076 section + 1691 chunk summaries).
    2. Builds ``summary_id -> [child_pid, ...]`` using the same chunk/section -> pid
       resolution as ``build_typed_adjacency`` in ``eval_graph_topk_rerank.py``.
    3. Encodes each summary text (+ retrieval_hints) with the embedding model.
    4. Encodes the query set with the same model (query prompt).
    5. For each query keeps the top-K most similar summaries (cosine).
    6. Dumps everything into a single JSON usable by the eval rerank script.

The summary itself never enters the candidate pool. Only its children pids are
boosted at rerank time, gated by query<->summary similarity.

Usage example:
    python scripts/build_summary_shadow_index.py \
        --summary-graph data/01_graphs/llm_summary_virtual_nodes.json \
        --chunk-graph   data/01_graphs/chunk_virtual_nodes.json \
        --elements-graph data/01_graphs/multimodal_elements.json \
        --corpus  data/05_eval/dense_retrieval/rebuilt_20260417/augmented/corpus_v1_enriched.jsonl \
        --queries data/05_eval/dense_retrieval/rebuilt_20260417/augmented/queries.jsonl \
        --model-name Qwen/Qwen3-Embedding-0.6B \
        --download-root models \
        --top-k-summaries 30 \
        --output data/01_graphs/summary_shadow_index.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


# ---------------------------------------------------------------------------
# pid <-> chunk / section mapping (mirror of build_typed_adjacency)
# ---------------------------------------------------------------------------
def build_pid_resolution(
    chunk_data: Dict[str, Any],
    elem_data: Dict[str, Any],
    valid_pids: Set[str],
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
    """Return (pid->chunk_id, chunk_id->[pid], doc_id->{section_title->[chunk_id]})."""
    doc_pos_to_chunk: Dict[str, Dict[int, str]] = {}
    doc_section_to_chunks: Dict[str, Dict[str, List[str]]] = {}
    for doc_id, doc in chunk_data.get("documents", {}).items():
        pos_to_chunk: Dict[int, str] = {}
        sec_to_chunks: Dict[str, List[str]] = defaultdict(list)
        for nid, node in doc.get("nodes", {}).items():
            for pidx in node.get("paragraph_indices", []):
                pos_to_chunk[pidx] = nid
        for edge in doc.get("edges", []):
            if edge.get("relation") == "section_contains_chunk":
                sec_to_chunks[edge["source"]].append(edge["target"])
        doc_pos_to_chunk[doc_id] = pos_to_chunk
        doc_section_to_chunks[doc_id] = sec_to_chunks

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
    pid_to_chunk: Dict[str, str] = {}
    for pid in valid_pids:
        for doc_id in doc_type0_to_pos:
            if not pid.startswith(doc_id + "_"):
                continue
            rest = pid[len(doc_id) + 1:]
            if rest.startswith(doc_id + "_"):
                rest = rest[len(doc_id) + 1:]
            parts = rest.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                raw_type = parts[0]
                n = int(parts[1])
                etype = type_alias.get(raw_type, raw_type)
                pidx = doc_type0_to_pos[doc_id].get((etype, n))
                if pidx is not None:
                    chunk_id = doc_pos_to_chunk.get(doc_id, {}).get(pidx)
                    if chunk_id:
                        pid_to_chunk[pid] = chunk_id
                if pid not in pid_to_chunk:
                    elem_id = f"{doc_id}_{raw_type}_{n}"
                    pidx2 = doc_elemid_to_pos[doc_id].get(elem_id)
                    if pidx2 is not None:
                        chunk_id2 = doc_pos_to_chunk.get(doc_id, {}).get(pidx2)
                        if chunk_id2:
                            pid_to_chunk[pid] = chunk_id2
            break

    chunk_to_pids: Dict[str, List[str]] = defaultdict(list)
    for pid, cid in pid_to_chunk.items():
        chunk_to_pids[cid].append(pid)

    return pid_to_chunk, dict(chunk_to_pids), doc_section_to_chunks


# ---------------------------------------------------------------------------
# summary -> pids
# ---------------------------------------------------------------------------
def build_summary_to_pids(
    summary_data: Dict[str, Any],
    chunk_to_pids: Dict[str, List[str]],
    doc_section_to_chunks: Dict[str, Dict[str, List[str]]],
) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]], Dict[str, int]]:
    """Return (summary_id -> pids, summary_records (list, ordered), stats)."""
    sum_to_pids: Dict[str, List[str]] = {}
    records: List[Dict[str, Any]] = []
    n_sec_mapped = n_sec_unmapped = n_chunk_mapped = n_chunk_unmapped = 0

    for doc_id, doc in summary_data.get("documents", {}).items():
        sec2chunks = doc_section_to_chunks.get(doc_id, {})
        # ---- section summaries ----
        for sid, sm in (doc.get("section_summaries") or {}).items():
            sec_title = sm.get("source_section", "")
            chunks = sec2chunks.get(sec_title, [])
            pids: List[str] = []
            for cid in chunks:
                pids.extend(chunk_to_pids.get(cid, []))
            pids = sorted(set(pids))
            if pids:
                sum_to_pids[sid] = pids
                n_sec_mapped += 1
            else:
                n_sec_unmapped += 1
            records.append({
                "summary_id": sid,
                "doc_id": doc_id,
                "scope": "section",
                "source": sec_title,
                "text": sm.get("text", ""),
                "keywords": sm.get("keywords") or [],
                "retrieval_hints": sm.get("retrieval_hints") or [],
                "num_pids": len(pids),
            })
        # ---- chunk summaries ----
        for sid, sm in (doc.get("chunk_summaries") or {}).items():
            cid = sm.get("source_chunk", "")
            pids = sorted(chunk_to_pids.get(cid, []))
            if pids:
                sum_to_pids[sid] = pids
                n_chunk_mapped += 1
            else:
                n_chunk_unmapped += 1
            records.append({
                "summary_id": sid,
                "doc_id": doc_id,
                "scope": "chunk",
                "source": cid,
                "text": sm.get("text", ""),
                "keywords": sm.get("keywords") or [],
                "retrieval_hints": sm.get("retrieval_hints") or [],
                "num_pids": len(pids),
            })

    stats = {
        "section_summaries_total": n_sec_mapped + n_sec_unmapped,
        "section_summaries_mapped": n_sec_mapped,
        "section_summaries_unmapped": n_sec_unmapped,
        "chunk_summaries_total": n_chunk_mapped + n_chunk_unmapped,
        "chunk_summaries_mapped": n_chunk_mapped,
        "chunk_summaries_unmapped": n_chunk_unmapped,
        "summaries_with_pids": len(sum_to_pids),
    }
    return sum_to_pids, records, stats


# ---------------------------------------------------------------------------
# encoding (mirror of eval_dense_retrieval.encode_texts)
# ---------------------------------------------------------------------------
def resolve_model_path(model_name: str, model_path: str | None, download_root: str | None) -> str:
    if model_path:
        p = Path(model_path).expanduser()
        if not p.is_dir():
            sys.exit(f"[fatal] --model-path {p} is not a directory")
        return str(p)
    from modelscope import snapshot_download
    kwargs: Dict[str, Any] = {}
    if download_root:
        kwargs["cache_dir"] = download_root
    print(f"[info] downloading {model_name} via modelscope ...", flush=True)
    return snapshot_download(model_name, **kwargs)


def load_model(model_dir: str, device: str):
    from sentence_transformers import SentenceTransformer
    print(f"[info] loading SentenceTransformer from {model_dir}", flush=True)
    return SentenceTransformer(model_dir, device=device, trust_remote_code=True)


def encode(model, texts: List[str], *, is_query: bool, batch_size: int, max_length: int) -> np.ndarray:
    try:
        model.max_seq_length = max_length
    except Exception:
        pass
    kwargs = dict(batch_size=batch_size, convert_to_numpy=True,
                  normalize_embeddings=True, show_progress_bar=True)
    if is_query:
        try:
            return model.encode(texts, prompt_name="query", **kwargs)
        except (ValueError, KeyError):
            print("[warn] no 'query' prompt, encoding raw", file=sys.stderr)
    return model.encode(texts, **kwargs)


def make_summary_text(rec: Dict[str, Any]) -> str:
    """Compose the text we feed the encoder for a summary record.

    Combines summary text with retrieval_hints (semantic anchors), since hints
    are crafted to be query-like.
    """
    parts: List[str] = []
    txt = (rec.get("text") or "").strip()
    if txt:
        parts.append(txt)
    hints = [h for h in (rec.get("retrieval_hints") or []) if h]
    if hints:
        parts.append("Topics: " + "; ".join(hints))
    kws = [k for k in (rec.get("keywords") or []) if k]
    if kws:
        parts.append("Keywords: " + ", ".join(kws))
    return "\n".join(parts) if parts else (rec.get("source") or rec["summary_id"])


def extract_query_text(row: Dict[str, Any]) -> str:
    return row.get("query") or row.get("question") or row.get("text") or ""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-graph", default="data/01_graphs/llm_summary_virtual_nodes.json")
    ap.add_argument("--chunk-graph", default="data/01_graphs/chunk_virtual_nodes.json")
    ap.add_argument("--elements-graph", default="data/01_graphs/multimodal_elements.json")
    ap.add_argument("--corpus", required=True, help="Corpus jsonl (defines valid pids)")
    ap.add_argument("--queries", required=True, help="Queries jsonl")
    ap.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--model-path", default="")
    ap.add_argument("--download-root", default="models")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--top-k-summaries", type=int, default=30,
                    help="Per-query: keep this many top summaries by cosine")
    ap.add_argument("--min-score", type=float, default=0.20,
                    help="Drop summaries below this cosine score")
    ap.add_argument("--scope", choices=["both", "section", "chunk"], default="both",
                    help="Which summary scope to include. 'chunk' gives finer granularity")
    ap.add_argument("--max-children", type=int, default=0,
                    help="Drop summaries whose child-pid count exceeds this (0=no filter). Protects fine-grained ranking by excluding large sibling groups.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    def resolve(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p

    sum_path = resolve(args.summary_graph)
    chunk_path = resolve(args.chunk_graph)
    elem_path = resolve(args.elements_graph)
    corpus_path = resolve(args.corpus)
    queries_path = resolve(args.queries)
    out_path = resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -- load graphs --
    print(f"[info] loading summary graph: {sum_path}", flush=True)
    summary_data = json.loads(sum_path.read_bytes())
    print(f"[info] loading chunk graph:   {chunk_path}", flush=True)
    chunk_data = json.loads(chunk_path.read_bytes())
    print(f"[info] loading elements graph: {elem_path}", flush=True)
    elem_data = json.loads(elem_path.read_bytes())

    # -- build pid mapping --
    corpus_rows = load_jsonl(corpus_path)
    valid_pids = {str(r["passage_id"]) for r in corpus_rows}
    print(f"[info] valid pids: {len(valid_pids)}", flush=True)
    pid_to_chunk, chunk_to_pids, doc_section_to_chunks = build_pid_resolution(
        chunk_data, elem_data, valid_pids
    )
    print(f"[info] pid->chunk mapped: {len(pid_to_chunk)} ({len(pid_to_chunk)/max(1,len(valid_pids))*100:.1f}%)", flush=True)

    # -- build summary -> pids --
    sum_to_pids, records, sum_stats = build_summary_to_pids(
        summary_data, chunk_to_pids, doc_section_to_chunks
    )
    print(f"[info] summary stats: {json.dumps(sum_stats)}", flush=True)
    if not sum_to_pids:
        sys.exit("[fatal] no summary mapped to any pid; check graph alignment")

    # Scope + max-children filter (fine-grained control)
    before = len(records)
    if args.scope != "both":
        records = [r for r in records if r["scope"] == args.scope]
        keep_ids = {r["summary_id"] for r in records}
        sum_to_pids = {k: v for k, v in sum_to_pids.items() if k in keep_ids}
    if args.max_children and args.max_children > 0:
        records = [r for r in records
                   if r["summary_id"] in sum_to_pids
                   and len(sum_to_pids[r["summary_id"]]) <= args.max_children]
        keep_ids = {r["summary_id"] for r in records}
        sum_to_pids = {k: v for k, v in sum_to_pids.items() if k in keep_ids}
    print(f"[info] scope={args.scope} max_children={args.max_children} -> {len(records)} records (from {before})", flush=True)

    # Filter records to those that have at least one mapped pid (no point encoding orphans)
    records = [r for r in records if r["summary_id"] in sum_to_pids]
    print(f"[info] active summaries to encode: {len(records)}", flush=True)

    # -- load queries --
    query_rows = load_jsonl(queries_path)
    qids = [str(r.get("query_id") or r.get("qid") or "") for r in query_rows]
    qtexts = [extract_query_text(r) for r in query_rows]
    print(f"[info] queries: {len(qids)}", flush=True)

    # -- encode --
    model_dir = resolve_model_path(args.model_name, args.model_path or None,
                                   args.download_root or None)
    model = load_model(model_dir, args.device)

    sum_texts = [make_summary_text(r) for r in records]
    t0 = time.time()
    sum_emb = encode(model, sum_texts, is_query=False,
                     batch_size=args.batch_size, max_length=args.max_length)
    print(f"[info] encoded {len(sum_texts)} summaries in {time.time()-t0:.1f}s, shape={sum_emb.shape}", flush=True)
    t0 = time.time()
    q_emb = encode(model, qtexts, is_query=True,
                   batch_size=args.batch_size, max_length=args.max_length)
    print(f"[info] encoded {len(qtexts)} queries in {time.time()-t0:.1f}s, shape={q_emb.shape}", flush=True)

    # -- per-query top-K summaries --
    print(f"[info] computing top-{args.top_k_summaries} summaries per query ...", flush=True)
    sims = q_emb @ sum_emb.T  # (Q, S), already normalized
    K = min(args.top_k_summaries, sims.shape[1])
    topk_idx = np.argpartition(-sims, K - 1, axis=1)[:, :K]
    # sort within
    rows = np.arange(sims.shape[0])[:, None]
    topk_scores = sims[rows, topk_idx]
    order = np.argsort(-topk_scores, axis=1)
    topk_idx = topk_idx[rows, order]
    topk_scores = topk_scores[rows, order]

    sid_list = [r["summary_id"] for r in records]
    query_summary_scores: Dict[str, List[List[Any]]] = {}
    score_min = float(args.min_score)
    for i, qid in enumerate(qids):
        pairs: List[List[Any]] = []
        for j in range(K):
            sc = float(topk_scores[i, j])
            if sc < score_min:
                break
            pairs.append([sid_list[int(topk_idx[i, j])], round(sc, 6)])
        query_summary_scores[qid] = pairs

    # -- dump --
    out = {
        "config": {
            "summary_graph": str(sum_path),
            "chunk_graph": str(chunk_path),
            "elements_graph": str(elem_path),
            "corpus": str(corpus_path),
            "queries": str(queries_path),
            "model_name": args.model_name,
            "model_path": args.model_path,
            "top_k_summaries": args.top_k_summaries,
            "min_score": args.min_score,
            "scope": args.scope,
            "max_children": args.max_children,
            "max_length": args.max_length,
        },
        "summary_stats": {
            **sum_stats,
            "active_summaries": len(records),
            "avg_pids_per_summary": round(
                sum(len(v) for v in sum_to_pids.values()) / max(1, len(sum_to_pids)), 3
            ),
            "num_queries": len(qids),
            "avg_summaries_per_query": round(
                sum(len(v) for v in query_summary_scores.values()) / max(1, len(qids)), 3
            ),
        },
        "summary_to_pids": sum_to_pids,
        "summary_meta": {r["summary_id"]: {"doc_id": r["doc_id"], "scope": r["scope"]}
                         for r in records},
        "query_summary_scores": query_summary_scores,
    }
    print(f"[info] writing index to {out_path}", flush=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False))
    print("[done]", json.dumps(out["summary_stats"], indent=2), flush=True)


if __name__ == "__main__":
    main()
