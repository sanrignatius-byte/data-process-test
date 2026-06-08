#!/usr/bin/env python3
"""Pre-encode an M4query corpus.jsonl with Qwen3-Embedding-0.6B and cache.

Run this once per corpus; subsequent eval_zh_query_embedding.py runs that
point at --corpus-cache will skip the encoding step.
"""

import argparse, json, os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
from sentence_transformers import SentenceTransformer


def load_corpus(path: Path) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        pid = d.get("passage_id") or d.get("docid") or d.get("id")
        if not pid:
            continue
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
            ids.append(pid)
            texts.append(merged)
    return ids, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ids-out", default="",
                    help="Where to dump the passage_id list parallel to the embeddings")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    print(f"Loading corpus {args.corpus}")
    ids, texts = load_corpus(Path(args.corpus))
    print(f"  passages with non-empty text: {len(ids)}")

    print(f"Loading {args.model} on cuda:{os.environ.get('CUDA_VISIBLE_DEVICES','')}")
    model = SentenceTransformer(args.model, trust_remote_code=True)
    model.max_seq_length = args.max_len
    print(f"  dim: {model.get_embedding_dimension()}")

    emb = model.encode(
        texts, batch_size=args.batch_size,
        normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True,
    )
    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, cache)
    print(f"  saved embeddings → {cache} (shape {emb.shape}, dtype {emb.dtype})")

    if args.ids_out:
        Path(args.ids_out).write_text("\n".join(ids))
        print(f"  saved ids → {args.ids_out}")


if __name__ == "__main__":
    main()
