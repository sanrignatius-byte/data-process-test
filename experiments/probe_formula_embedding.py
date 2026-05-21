"""Probe candidate formula embedding backends on a small sample.

Compares score distributions and qualitative top pairs between:
  - clip_text (current baseline, ViT-B/32 openai)
  - math_similarity (math-similarity/Bert-MLM_arXiv-MP-class_arXiv)

Goal: confirm the math model spreads similarity out instead of CLIP's
p50=0.966 collapse, before wiring it into build_mineru_vl_edges.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import statistics

import numpy as np

GRAPH = Path("/projects/myyyx1/data-process-test/data/05_eval/mineru_topology_graph_v1_latest/mineru_topology_graph_v1.json")
OUT_DIR = Path("/projects/myyyx1/data-process-test/data/05_eval/probe_formula_embedding")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_formulas(limit: int | None = None) -> list[dict]:
    g = json.loads(GRAPH.read_text())
    formulas = [n for n in g["nodes"] if n.get("node_type") == "formula"]
    if limit:
        formulas = formulas[:limit]
    return formulas


def cos_matrix(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n[n == 0] = 1.0
    e = emb / n
    return e @ e.T


def score_stats(name: str, sim: np.ndarray) -> dict:
    iu = np.triu_indices_from(sim, k=1)
    vals = sim[iu]
    qs = np.quantile(vals, [0.5, 0.75, 0.9, 0.95, 0.99])
    return {
        "backend": name,
        "n_pairs": int(vals.size),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "p50": float(qs[0]),
        "p75": float(qs[1]),
        "p90": float(qs[2]),
        "p95": float(qs[3]),
        "p99": float(qs[4]),
    }


def top_pairs(sim: np.ndarray, formulas, k: int = 5):
    iu = np.triu_indices_from(sim, k=1)
    vals = sim[iu]
    order = np.argsort(-vals)[:k]
    out = []
    for o in order:
        i, j = iu[0][o], iu[1][o]
        out.append({
            "score": float(vals[o]),
            "i": int(i),
            "j": int(j),
            "doc_i": formulas[i]["doc_id"],
            "doc_j": formulas[j]["doc_id"],
            "snippet_i": formulas[i]["text_snippet"][:200],
            "snippet_j": formulas[j]["text_snippet"][:200],
        })
    return out


def low_pairs(sim: np.ndarray, formulas, k: int = 3):
    iu = np.triu_indices_from(sim, k=1)
    vals = sim[iu]
    order = np.argsort(vals)[:k]
    out = []
    for o in order:
        i, j = iu[0][o], iu[1][o]
        out.append({
            "score": float(vals[o]),
            "snippet_i": formulas[i]["text_snippet"][:120],
            "snippet_j": formulas[j]["text_snippet"][:120],
        })
    return out


def embed_clip_text(texts: list[str]) -> np.ndarray:
    import torch
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            tokens = tokenizer(batch)
            feat = model.encode_text(tokens)
            embs.append(feat.cpu().numpy())
    return np.vstack(embs).astype(np.float32)


def embed_math_similarity(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("math-similarity/Bert-MLM_arXiv-MP-class_arXiv")
    emb = model.encode(texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True)
    return emb.astype(np.float32)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="formulas to sample")
    ap.add_argument("--backends", nargs="+", default=["clip_text", "math_similarity"])
    args = ap.parse_args()

    formulas = load_formulas(limit=args.limit)
    texts = [f["text_snippet"] for f in formulas]
    print(f"Loaded {len(formulas)} formulas", file=sys.stderr)

    report = {"sample_size": len(formulas), "backends": []}

    for backend in args.backends:
        print(f"\n== {backend} ==", file=sys.stderr)
        t0 = time.time()
        if backend == "clip_text":
            emb = embed_clip_text(texts)
        elif backend == "math_similarity":
            emb = embed_math_similarity(texts)
        else:
            print(f"unknown backend: {backend}", file=sys.stderr)
            continue
        dt = time.time() - t0
        print(f"  embed shape: {emb.shape}, took {dt:.1f}s", file=sys.stderr)

        sim = cos_matrix(emb)
        stats = score_stats(backend, sim)
        stats["embed_seconds"] = round(dt, 2)
        stats["top_pairs"] = top_pairs(sim, formulas, k=5)
        stats["low_pairs"] = low_pairs(sim, formulas, k=3)
        report["backends"].append(stats)

        out_path = OUT_DIR / f"sim_{backend}.npy"
        np.save(out_path, sim.astype(np.float16))
        print(f"  saved sim matrix -> {out_path}", file=sys.stderr)

    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "backends"}, ensure_ascii=False))
    for b in report["backends"]:
        print(f"\n[{b['backend']}] p50={b['p50']:.3f} p90={b['p90']:.3f} p99={b['p99']:.3f} std={b['std']:.3f}")


if __name__ == "__main__":
    main()
