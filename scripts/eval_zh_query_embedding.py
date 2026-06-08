#!/usr/bin/env python3
"""Embedding-based cross-lingual retrieval validation for ZH-translated queries.

Mirrors `eval_zh_query_retrieval.py` (which uses BM25) but ranks passages by
cosine similarity in a multilingual embedding space. Uses
`Qwen/Qwen3-Embedding-0.6B` — a small text-only multilingual retriever already
cached locally.

For each translated triplet we score three query variants against the same
English corpus:
  EN     : original English query
  ZH     : Mandarin Chinese query (true cross-lingual test)
  ZH→EN  : round-trip back-translation
"""

import argparse, json, os, re
from pathlib import Path

# Restrict to GPU 1 per project convention before importing torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Qwen3-Embedding-0.6B uses a manual instruction-style query encoding:
#   "Instruct: <task>\nQuery: <query>"
# Qwen3-VL-Embedding-* has a built-in "default" prompt template that ST applies
# automatically — passing the same instruction manually would double-wrap.
QUERY_INSTRUCTION = (
    "Given a research question, retrieve passages that answer it. "
    "Treat Chinese and English as semantically equivalent."
)


def fmt_query(q: str, use_instruction: bool) -> str:
    if use_instruction:
        return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {q}"
    return q


def passage_units(t: dict, corpus_lookup: dict | None = None):
    """Yield (positives, negatives) as lists of (uid, text).

    Handles three schemas:
      - delivery_v1 : positive=list of dicts with span_text + element_id
      - triplet_v2  : positive=dict with text
      - M4query     : positive_passages = list of passage_id strings
                      (texts resolved via `corpus_lookup`)
    """
    if isinstance(t.get("positive"), list):
        pos = [(p.get("element_id") or f"pos_{i}", p.get("span_text", ""))
               for i, p in enumerate(t["positive"])]
        neg = [(n.get("element_id") or f"neg_{i}", n.get("span_text", ""))
               for i, n in enumerate(t.get("hard_negatives", []) or [])]
    elif isinstance(t.get("positive"), dict):
        pf = t["positive"]
        pos = [(pf.get("element_id") or "pos_0", pf.get("text", ""))]
        neg = [(n.get("element_id") or f"neg_{i}", n.get("text", ""))
               for i, n in enumerate(t.get("negatives", []) or [])]
    elif "positive_passages" in t:  # M4query: passages are ID strings
        pids_pos = t.get("positive_passages") or []
        pids_neg = t.get("hard_negative_passages") or []
        if corpus_lookup is None:
            return [], []
        def _resolve(pids):
            out = []
            for pid in pids:
                if isinstance(pid, str):
                    txt = corpus_lookup.get(pid, "")
                    if txt:
                        out.append((pid, txt))
                elif isinstance(pid, dict):
                    out.append((pid.get("passage_id") or pid.get("docid") or "",
                                pid.get("text", "")))
            return out
        pos = _resolve(pids_pos)
        neg = _resolve(pids_neg)
    else:
        return [], []
    pos = [(u, x) for u, x in pos if x]
    neg = [(u, x) for u, x in neg if x]
    return pos, neg


def load_corpus_file(path: Path) -> tuple[dict, list, list]:
    """Load M4query-style corpus.jsonl. Returns (id→text dict, ids list, texts list)."""
    id_to_text: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        pid = d.get("passage_id") or d.get("docid") or d.get("id")
        if not pid:
            continue
        # Combine text + caption + description for richest retrieval signal
        text = (d.get("text") or "").strip()
        cap = (d.get("caption") or "").strip()
        desc = (d.get("description") or "").strip()
        parts = []
        if cap and cap != text:
            parts.append(cap)
        if text:
            parts.append(text)
        if desc and desc not in (cap, text):
            parts.append(desc)
        merged = " ".join(parts).strip()
        if merged:
            id_to_text[pid] = merged
    ids = list(id_to_text.keys())
    texts = [id_to_text[i] for i in ids]
    return id_to_text, ids, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/05_eval/zh_retrieval_embedding_report.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--no-instruction-prefix", action="store_true",
                    help="Skip manual 'Instruct: …' wrap (use for VL-Embedding models that have built-in default prompt)")
    ap.add_argument("--corpus-file", default="",
                    help="External corpus.jsonl (M4query schema). When set, positives "
                         "are resolved from passage_id strings in the triplet against this file.")
    ap.add_argument("--corpus-cache", default="",
                    help="Optional path to cache encoded corpus embeddings.")
    args = ap.parse_args()

    triplets = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    print(f"Loaded {len(triplets)} triplets")

    # ── Build corpus ─────────────────────────────────────────────────────
    corpus_lookup = None
    if args.corpus_file:
        print(f"Loading external corpus from {args.corpus_file} …")
        corpus_lookup, corpus_uids, corpus_texts = load_corpus_file(Path(args.corpus_file))
        uid_to_idx = {u: i for i, u in enumerate(corpus_uids)}
        gold_per_q = []
        unresolved = 0
        for t in triplets:
            pos, _ = passage_units(t, corpus_lookup)
            g = {uid_to_idx[u] for u, _x in pos if u in uid_to_idx}
            if not g:
                unresolved += 1
            gold_per_q.append(g)
        print(f"Corpus from file: {len(corpus_texts)} unique passages "
              f"({unresolved} triplets with no resolvable gold)")
    else:
        uid_to_idx: dict[str, int] = {}
        corpus_texts: list[str] = []
        def _add(uid: str, text: str) -> int:
            if uid in uid_to_idx:
                return uid_to_idx[uid]
            uid_to_idx[uid] = len(corpus_texts)
            corpus_texts.append(text)
            return uid_to_idx[uid]
        gold_per_q = []
        for t in triplets:
            pos, neg = passage_units(t)
            g = set()
            for u, x in pos:
                g.add(_add(u, x))
            for u, x in neg:
                _add(u, x)
            gold_per_q.append(g)
        print(f"Corpus built from triplets: {len(corpus_texts)} unique passages")

    print(f"Loading {args.model} on {'cuda' if torch.cuda.is_available() else 'cpu'} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')})…")
    model = SentenceTransformer(args.model, trust_remote_code=True)
    model.max_seq_length = args.max_len
    print(f"  embedding dim: {model.get_sentence_embedding_dimension()}")

    cache_path = Path(args.corpus_cache) if args.corpus_cache else None
    if cache_path and cache_path.exists():
        print(f"Loading cached corpus embeddings from {cache_path} …")
        corpus_emb = torch.load(cache_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=True)
        if corpus_emb.shape[0] != len(corpus_texts):
            print(f"  ⚠ cached corpus size {corpus_emb.shape[0]} ≠ current {len(corpus_texts)}; re-encoding")
            corpus_emb = None
        else:
            print(f"  loaded {corpus_emb.shape}")
    else:
        corpus_emb = None
    if corpus_emb is None:
        print("Encoding corpus…")
        corpus_emb = model.encode(
            corpus_texts, batch_size=args.batch_size,
            normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True,
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(corpus_emb, cache_path)
            print(f"  Cached → {cache_path}")

    def eval_variant(label: str, field: str) -> dict:
        queries = [(t.get(field) or "").strip() for t in triplets]
        kept = [(i, q) for i, q in enumerate(queries) if q and gold_per_q[i]]
        if not kept:
            return {"label": label, "n": 0}
        idx_list, q_list = zip(*kept)
        q_in = [fmt_query(q, use_instruction=not args.no_instruction_prefix) for q in q_list]
        print(f"  encoding {len(q_in)} queries [{label}]…")
        q_emb = model.encode(
            list(q_in), batch_size=args.batch_size,
            normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False,
        )
        scores = (q_emb @ corpus_emb.T).float().cpu().numpy()  # [Q, C]
        N = len(corpus_texts)
        import numpy as np
        order = (-scores).argsort(axis=1)  # [Q, C], descending
        ranks_first = []
        avg_gold_rank = []
        for row_i, q_idx in enumerate(idx_list):
            golds = gold_per_q[q_idx]
            ranks = []
            for g in golds:
                rank = int(np.where(order[row_i] == g)[0][0]) + 1
                ranks.append(rank)
            ranks_first.append(min(ranks))
            avg_gold_rank.append(sum(ranks) / len(ranks))
        n = len(ranks_first)
        r1 = sum(1 for r in ranks_first if r <= 1) / n
        r5 = sum(1 for r in ranks_first if r <= 5) / n
        r10 = sum(1 for r in ranks_first if r <= 10) / n
        r50 = sum(1 for r in ranks_first if r <= 50) / n
        mrr = sum(1.0 / r for r in ranks_first) / n
        median = sorted(ranks_first)[n // 2]
        return {
            "label": label,
            "n": n,
            "R@1": round(r1, 4),
            "R@5": round(r5, 4),
            "R@10": round(r10, 4),
            "R@50": round(r50, 4),
            "MRR": round(mrr, 4),
            "median_first_gold_rank": median,
            "mean_avg_gold_rank": round(sum(avg_gold_rank) / n, 2),
        }

    variants = [
        eval_variant("EN (original)", "query_en"),
        eval_variant("ZH (cross-lingual)", "query_zh"),
        eval_variant("ZH→EN (round-trip)", "query_zh2en"),
    ]
    report = {
        "input": args.input,
        "model": args.model,
        "corpus_size": len(corpus_texts),
        "n_triplets": len(triplets),
        "variants": variants,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    cols = ["n", "R@1", "R@5", "R@10", "R@50", "MRR", "median_first_gold_rank"]
    print(f"\n{'variant':22} " + " ".join(f"{c:>12}" for c in cols))
    print("-" * 110)
    for v in variants:
        row = [v.get(c, "—") for c in cols]
        print(f"{v['label']:22} " + " ".join(f"{str(x):>12}" for x in row))
    print(f"\nReport → {out_path}")


if __name__ == "__main__":
    main()
