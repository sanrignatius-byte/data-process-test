#!/usr/bin/env python3
"""Split-modality retrieval with genuine multimodal embeddings.

Unlike the text-only version (which encodes "[Image: xxx.jpg]" placeholders
as if they were text), this script uses Qwen3-VL-Embedding-2B to encode:
  - Figures: actual image files via vision encoder
  - Tables: text (caption + data) via text encoder
  - Formulas: LaTeX text via text encoder
  - Text: plain text via text encoder
  - Queries: text via text encoder

All embeddings live in a unified 2048-dim multimodal space, so a text
query can retrieve visual content through cross-modal alignment.

Usage:
  python scripts/eval_split_modality_vl.py \
    --data-dir data/03_queries/M4query_v1 \
    --vl-model models/Qwen3-VL-Embedding-2B \
    --output-dir data/05_eval/dense_retrieval/split_modality_vl
"""

from __future__ import annotations

import argparse, json, math, os, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_image_path(passage: dict, docs_dir: Path) -> str | None:
    """Resolve image_path from corpus to actual file on disk.

    Tries multiple path patterns:
      1. documents/{doc_id}/mineru/{image_path} (figures)
      2. documents/{doc_id}/latex/{image_path} (latex source)
      3. Strip "data/00_raw/mineru_output/{doc_id}/" prefix (tables)
      4. Search by filename in mineru/ and latex/ dirs
    """
    ip = passage.get("image_path")
    doc_id = passage.get("doc_id", "")
    if not ip:
        return None

    candidates = [
        docs_dir / doc_id / "mineru" / ip,
        docs_dir / doc_id / "latex" / ip,
    ]

    # Table image_paths have a "data/00_raw/mineru_output/{doc_id}/" prefix
    prefix = f"data/00_raw/mineru_output/{doc_id}/"
    if ip.startswith(prefix):
        stripped = ip[len(prefix):]
        candidates.append(docs_dir / doc_id / "mineru" / stripped)
        candidates.append(docs_dir / doc_id / "latex" / stripped)

    for c in candidates:
        if c.exists():
            return str(c)

    # Fallback: search by filename in mineru/latex dirs
    fname = os.path.basename(ip)
    for src in ("mineru", "latex"):
        src_dir = docs_dir / doc_id / src
        if not src_dir.exists():
            continue
        for root, dirs, files in os.walk(str(src_dir)):
            if fname in files:
                return os.path.join(root, fname)

    return None


def encode_text_batch(model, texts: list[str], batch_size: int = 32, desc: str = "") -> np.ndarray:
    embs = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embs)


def encode_corpus_vl(
    model, corpus: list[dict], docs_dir: Path,
    batch_size: int = 32,
    text_max_length: int = 512,
    image_max_length: int = 4096,
) -> tuple[np.ndarray, dict[str, int]]:
    """Encode corpus with VL-awareness.

    Returns (embeddings, pid_to_index).
    Figures with valid images → image encode; everything else → text encode.
    """
    # Separate passages by encoding strategy
    img_passages = []   # (passage, image_path)
    txt_passages = []   # passage

    for p in corpus:
        ptype = p.get("type", "text")
        if ptype in ("figure", "table"):
            impath = resolve_image_path(p, docs_dir)
            if impath:
                img_passages.append((p, impath))
                continue
        txt_passages.append(p)

    n_img = len(img_passages)
    n_txt = len(txt_passages)
    n_total = len(corpus)
    print(f"  Image-encode: {n_img} (figures + tables)")
    print(f"  Text-encode:  {n_txt} passages")

    embs = np.empty((n_total, 2048), dtype=np.float32)  # VL model output dim
    pid_to_idx: dict[str, int] = {}

    # 1. Encode images
    if img_passages:
        print(f"\n  Encoding {n_img} images...")
        images = []
        img_passage_list = []
        for p, impath in img_passages:
            try:
                img = Image.open(impath).convert("RGB")
                images.append(img)
                img_passage_list.append(p)
            except Exception as e:
                # Fallback: encode as text
                print(f"    WARN: cannot load {impath}: {e}, falling back to text")
                txt_passages.append(p)

        if images:
            t0 = time.time()
            # Qwen3-VL preprocessing emits a per-image vision-token block whose
            # length depends on image resolution; truncating below that block
            # causes input_ids ≠ text image-token count and the processor raises.
            # Bump max_seq_length only for the image pass; restore afterwards.
            saved_max = model.max_seq_length
            model.max_seq_length = image_max_length
            try:
                img_embs = model.encode(images, batch_size=1, show_progress_bar=True,
                                        normalize_embeddings=True)
            finally:
                model.max_seq_length = saved_max
            print(f"    {len(images)} images in {time.time()-t0:.1f}s "
                  f"({(time.time()-t0)/max(len(images),1):.2f}s/img)")

            for i, p in enumerate(img_passage_list):
                idx = len(pid_to_idx)
                embs[idx] = img_embs[i].astype(np.float32)
                pid_to_idx[p["passage_id"]] = idx

    # 2. Encode remaining as text
    if txt_passages:
        print(f"\n  Encoding {len(txt_passages)} text passages...")
        texts = [p.get("text", "") for p in txt_passages]
        txt_embs = encode_text_batch(model, texts, batch_size, "corpus/text")

        for i, p in enumerate(txt_passages):
            idx = len(pid_to_idx)
            embs[idx] = txt_embs[i].astype(np.float32)
            pid_to_idx[p["passage_id"]] = idx

    assert len(pid_to_idx) == n_total, f"Mismatch: {len(pid_to_idx)} != {n_total}"
    return embs, pid_to_idx


def build_faiss_index(embeddings: np.ndarray) -> Any:
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def search_index(index, query_emb: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    return index.search(query_emb.astype(np.float32), top_k)


def evaluate(qrels_by_qid: dict, ranking_by_qid: dict, ks: list[int] = [1, 5, 10, 100]):
    metrics = {}
    for k in ks:
        recall_sum = 0.0
        for qid, qrels in qrels_by_qid.items():
            if qid not in ranking_by_qid:
                continue
            top_k = set(ranking_by_qid[qid][:k])
            hits = sum(1 for qr in qrels if qr["passage_id"] in top_k)
            recall_sum += hits / len(qrels) if qrels else 0
        metrics[f"recall@{k}"] = recall_sum / max(len(qrels_by_qid), 1)

    mrr_sum = 0.0
    for qid, qrels in qrels_by_qid.items():
        if qid not in ranking_by_qid:
            continue
        qrel_pids = {qr["passage_id"] for qr in qrels}
        for rank, pid in enumerate(ranking_by_qid[qid], 1):
            if pid in qrel_pids:
                mrr_sum += 1.0 / rank
                break
    metrics["mrr"] = mrr_sum / max(len(qrels_by_qid), 1)
    metrics["num_queries"] = len(qrels_by_qid)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/03_queries/M4query_v1")
    parser.add_argument("--vl-model", default="models/Qwen3-VL-Embedding-2B")
    parser.add_argument("--output-dir", default="data/05_eval/dense_retrieval/split_modality_vl")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--text-max-length", type=int, default=512,
                        help="Max sequence length for text encoding (queries + text passages).")
    parser.add_argument("--image-max-length", type=int, default=4096,
                        help="Max sequence length for image encoding. Qwen3-VL emits "
                             "~1000-2000 vision tokens per image; truncating below that "
                             "causes processor token-count mismatch.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-queries", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = data_dir / "documents"

    # Load data
    print("Loading data...")
    corpus = load_jsonl(data_dir / "corpus.jsonl")
    queries = load_jsonl(data_dir / "queries.jsonl")
    qrels = load_jsonl(data_dir / "qrels.jsonl")

    if args.limit_queries > 0:
        queries = queries[:args.limit_queries]
        qids_keep = {q["query_id"] for q in queries}
        qrels = [qr for qr in qrels if qr["query_id"] in qids_keep]

    qrels_by_qid = defaultdict(list)
    for qr in qrels:
        qrels_by_qid[qr["query_id"]].append(qr)

    print(f"  Corpus: {len(corpus)} passages")
    print(f"  Queries: {len(queries)}")
    print(f"  Qrels: {len(qrels)}")

    # Split corpus by type (for reporting and per-type indices)
    type_passages: dict[str, list[dict]] = defaultdict(list)
    for p in corpus:
        ptype = p.get("type", "text")
        type_passages[ptype].append(p)

    print("\nCorpus by type:")
    for t, plist in sorted(type_passages.items()):
        print(f"  {t:<10s}: {len(plist):>5d} passages")

    # Load VL model
    print(f"\nLoading VL model: {args.vl_model}")
    model = SentenceTransformer(args.vl_model, device=args.device, trust_remote_code=True)
    model.max_seq_length = args.text_max_length
    print(f"  text_max_length:  {args.text_max_length}")
    print(f"  image_max_length: {args.image_max_length}")

    # Encode queries (text only)
    print("\nEncoding queries...")
    query_texts = []
    for q in queries:
        qt = q.get("query", q.get("question", q.get("text", "")))
        query_texts.append(qt)
    query_embs = encode_text_batch(model, query_texts, args.batch_size, "queries")
    print(f"  Query embeddings: {query_embs.shape}")

    # Encode corpus with VL awareness
    print("\nEncoding corpus (VL-aware)...")
    all_embs, pid_to_idx = encode_corpus_vl(
        model, corpus, docs_dir, args.batch_size,
        text_max_length=args.text_max_length,
        image_max_length=args.image_max_length,
    )
    print(f"  Corpus embeddings: {all_embs.shape}")
    all_ids = list(pid_to_idx.keys())
    # Verify ordering: idx → passage_id
    idx_to_pid = {v: k for k, v in pid_to_idx.items()}

    # =========================================================
    # A. BASELINE — Single Mixed Index
    # =========================================================
    print("\n" + "=" * 60)
    print("A. BASELINE — Single Mixed Index (VL)")
    print("=" * 60)

    mixed_index = build_faiss_index(all_embs)

    t0 = time.time()
    scores, indices = search_index(mixed_index, query_embs, args.top_k)
    elapsed = time.time() - t0

    baseline_ranking = {}
    for qi, q in enumerate(queries):
        qid = q["query_id"]
        ranked = [idx_to_pid[idx] for idx in indices[qi] if idx >= 0 and idx in idx_to_pid]
        baseline_ranking[qid] = ranked

    baseline_metrics = evaluate(qrels_by_qid, baseline_ranking)
    print(f"  Search: {elapsed:.2f}s")
    for k, v in baseline_metrics.items():
        if k.startswith("recall") or k == "mrr":
            print(f"  {k}: {v:.4f}")

    # Save baseline ranking
    with open(out_dir / "ranking_baseline.jsonl", "w") as f:
        for q in queries:
            qid = q["query_id"]
            f.write(json.dumps({"query_id": qid, "top_k": baseline_ranking.get(qid, [])}) + "\n")

    # =========================================================
    # B. SPLIT-INDEX — Per-Modality Indices
    # =========================================================
    print("\n" + "=" * 60)
    print("B. SPLIT-INDEX — Per-Modality Indices (VL)")
    print("=" * 60)

    type_indices: dict[str, tuple[Any, list[str]]] = {}
    for ptype, plist in type_passages.items():
        pids = {p["passage_id"] for p in plist}
        # Extract embeddings for this type
        indices_in_full = [pid_to_idx[pid] for pid in pids if pid in pid_to_idx]
        type_embs = all_embs[indices_in_full]
        type_ids = [idx_to_pid[i] for i in indices_in_full]
        idx = build_faiss_index(type_embs)
        type_indices[ptype] = (idx, type_ids)
        print(f"  {ptype}: {len(type_ids)} passages, dim={type_embs.shape[1]}")

    # Merge strategies
    merge_configs = [
        ("equal_split", {t: 1.0 / len(type_indices) for t in type_indices}),
        ("prop_to_corpus", {t: len(pl) / len(corpus) for t, pl in type_passages.items()}),
        ("boost_formula", {"figure": 0.30, "table": 0.25, "formula": 0.35, "text": 0.10}),
        ("boost_figure", {"figure": 0.40, "table": 0.25, "formula": 0.25, "text": 0.10}),
    ]

    all_merge_results = {}

    for merge_name, allocation in merge_configs:
        print(f"\n  Merge: {merge_name}")
        print(f"    Allocation: {allocation}")

        ranking = {}
        for qi, q in enumerate(queries):
            qid = q["query_id"]
            q_emb = query_embs[qi:qi + 1]

            merged: list[str] = []
            seen: set[str] = set()

            for ptype, (idx, ids) in type_indices.items():
                budget = max(1, int(args.top_k * allocation.get(ptype, 0.25)))
                _, idxs = search_index(idx, q_emb, budget)
                for i in idxs[0]:
                    if i >= 0 and i < len(ids):
                        pid = ids[i]
                        if pid not in seen:
                            seen.add(pid)
                            merged.append(pid)

            if len(merged) < args.top_k:
                for ptype, (idx, ids) in type_indices.items():
                    if len(merged) >= args.top_k:
                        break
                    _, idxs = search_index(idx, q_emb, args.top_k)
                    for i in idxs[0]:
                        if i >= 0 and i < len(ids):
                            pid = ids[i]
                            if pid not in seen:
                                seen.add(pid)
                                merged.append(pid)
                                if len(merged) >= args.top_k:
                                    break

            ranking[qid] = merged[:args.top_k]

        merge_metrics = evaluate(qrels_by_qid, ranking)
        all_merge_results[merge_name] = merge_metrics

        for k, v in merge_metrics.items():
            if k.startswith("recall") or k == "mrr":
                delta = v - baseline_metrics.get(k, 0)
                print(f"    {k}: {v:.4f}  (Δ={delta:+.4f})")

        with open(out_dir / f"ranking_{merge_name}.jsonl", "w") as f:
            for q in queries:
                qid = q["query_id"]
                f.write(json.dumps({"query_id": qid, "top_k": ranking.get(qid, [])}) + "\n")

    # =========================================================
    # C. Per-Modality Recall Comparison
    # =========================================================
    print("\n" + "=" * 60)
    print("C. Per-Modality Recall Comparison")
    print("=" * 60)

    best_name = max(all_merge_results, key=lambda n: all_merge_results[n].get("mrr", 0))
    best_ranking = {}
    with open(out_dir / f"ranking_{best_name}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            best_ranking[r["query_id"]] = r["top_k"]

    pid2type = {p["passage_id"]: p.get("type", "?") for p in corpus}

    print(f"  Best merge: {best_name}")
    print(f"  {'Type':<10s} {'Baseline R@10':>15s} {'Split R@10':>15s} {'Delta':>10s}")
    print("  " + "-" * 52)

    for ptype in ["figure", "table", "formula"]:
        base_hit, base_total = 0, 0
        split_hit, split_total = 0, 0
        for qid, qrs in qrels_by_qid.items():
            base_top = baseline_ranking.get(qid, [])[:10]
            split_top = best_ranking.get(qid, [])[:10]
            for qr in qrs:
                if pid2type.get(qr["passage_id"]) == ptype:
                    base_total += 1
                    split_total += 1
                    if qr["passage_id"] in base_top:
                        base_hit += 1
                    if qr["passage_id"] in split_top:
                        split_hit += 1
        base_r = base_hit / max(base_total, 1)
        split_r = split_hit / max(split_total, 1)
        delta = split_r - base_r
        print(f"  {ptype:<10s} {base_r:>15.4f} {split_r:>15.4f} {delta:>+10.4f}")

    # =========================================================
    # D. Save Summary
    # =========================================================
    report = {
        "baseline": baseline_metrics,
        "splits": all_merge_results,
        "best_merge": best_name,
        "vl_config": {
            "vl_model": args.vl_model,
            "data_dir": str(data_dir),
            "top_k": args.top_k,
            "num_queries": len(queries),
            "num_passages": len(corpus),
            "num_img_encoded": sum(1 for p in corpus
                                   if p.get("type") in ("figure", "table")
                                   and resolve_image_path(p, docs_dir)),
        },
    }
    with open(out_dir / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nResults saved to {out_dir}")
    print(f"  Baseline MRR: {baseline_metrics['mrr']:.4f}  R@10: {baseline_metrics['recall@10']:.4f}")
    best = all_merge_results[best_name]
    print(f"  Best ({best_name}): MRR: {best['mrr']:.4f}  R@10: {best['recall@10']:.4f}")


if __name__ == "__main__":
    main()
