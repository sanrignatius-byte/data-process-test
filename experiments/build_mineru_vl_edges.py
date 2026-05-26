#!/usr/bin/env python3
"""Build VL-derived MinerU graph edges.

Preferred backend: CLIP ViT-B/32 via ``open_clip_torch`` on CPU.  A TF-IDF
proxy backend is also available for fast pipeline validation when torch/open_clip
are not installed; the output schema is identical and the backend is recorded in
metadata.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V1_DIR = ROOT / "data/05_eval/mineru_only_graph_v1_latest"
DEFAULT_TOPOLOGY_DIR = ROOT / "data/05_eval/mineru_topology_graph_v1_latest"


def compact_text(value: Any, limit: int = 800) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    return text[:limit]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr.astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


def load_element_to_node(topology_path: Path | None) -> dict[str, str]:
    if not topology_path or not topology_path.exists():
        return {}
    topo = read_json(topology_path)
    mapping = topo.get("element_to_node")
    if isinstance(mapping, dict):
        return {str(k): str(v) for k, v in mapping.items()}
    out: dict[str, str] = {}
    for node in topo.get("nodes", []) or []:
        element_id = node.get("element_id") or node.get("mapped_element_id")
        node_id = node.get("node_id")
        if element_id and node_id:
            out[str(element_id)] = str(node_id)
    return out


def collect_items(v1_graph: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    visual_items: list[dict[str, Any]] = []
    text_items: list[dict[str, Any]] = []
    formula_items: list[dict[str, Any]] = []

    for doc_id, doc in sorted((v1_graph.get("documents") or {}).items()):
        for element_id, elem in sorted((doc.get("elements") or {}).items(), key=lambda kv: (
            int(kv[1].get("page_idx") or 0),
            int(kv[1].get("position_idx") or 0),
            kv[0],
        )):
            etype = str(elem.get("element_type") or "")
            base = {
                "element_id": element_id,
                "doc_id": doc_id,
                "element_type": etype,
                "label": compact_text(elem.get("label"), 160),
                "caption": compact_text(elem.get("caption"), 500),
                "page_idx": elem.get("page_idx"),
                "position_idx": elem.get("position_idx"),
                "number": elem.get("number"),
            }
            if etype in {"figure", "table"}:
                image_path = str(elem.get("image_path") or "")
                if image_path and Path(image_path).exists():
                    proxy_text = " ".join(
                        part for part in [
                            base["label"],
                            base["caption"],
                            compact_text(elem.get("content"), 500),
                            compact_text(elem.get("context_before"), 700),
                            compact_text(elem.get("context_after"), 700),
                        ]
                        if part
                    )
                    visual_items.append({
                        **base,
                        "image_path": image_path,
                        "proxy_text": proxy_text,
                    })
            elif etype == "text":
                text = compact_text(elem.get("content"), 1000)
                if len(text) >= 40:
                    text_items.append({
                        **base,
                        "text": text,
                    })
            elif etype == "formula":
                metadata = elem.get("metadata") if isinstance(elem.get("metadata"), dict) else {}
                latex = ""
                if isinstance(metadata, dict):
                    latex = compact_text(metadata.get("latex"), 1000)
                text = latex or compact_text(elem.get("content"), 1000)
                if len(text) >= 8:
                    formula_items.append({
                        **base,
                        "text": text,
                    })

    return visual_items, text_items, formula_items


def choose_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    has_clip = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("open_clip") is not None
    if has_clip:
        return "open_clip"
    print("[warn] torch/open_clip not installed; falling back to TF-IDF proxy backend")
    return "tfidf"


def embed_with_open_clip(
    visual_items: list[dict[str, Any]],
    text_items: list[dict[str, Any]],
    formula_items: list[dict[str, Any]],
    model_name: str,
    pretrained: str,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    import open_clip
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    image_features: list[np.ndarray] = []
    image_failures: list[dict[str, str]] = []
    with torch.no_grad():
        done_images = 0
        for batch in batched(visual_items, batch_size):
            tensors = []
            kept = []
            for item in batch:
                try:
                    with Image.open(item["image_path"]) as im:
                        tensors.append(preprocess(im.convert("RGB")))
                    kept.append(item)
                except Exception as exc:  # pragma: no cover - depends on local image corruption
                    image_failures.append({"element_id": item["element_id"], "error": str(exc)})
            if not tensors:
                continue
            batch_tensor = torch.stack(tensors).to(device)
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            arr = feats.cpu().numpy().astype(np.float32)
            image_features.append(arr)
            done_images += len(kept)
            print(f"[open_clip] embedded images {done_images}/{len(visual_items)}", flush=True)
            if len(kept) != len(batch):
                raise RuntimeError("OpenCLIP image failures currently require rerunning after removing bad images")

        def encode_texts(texts: list[str]) -> np.ndarray:
            chunks: list[np.ndarray] = []
            done_texts = 0
            for text_batch in batched(texts, batch_size):
                tokens = tokenizer(text_batch).to(device)
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                chunks.append(feats.cpu().numpy().astype(np.float32))
                done_texts += len(text_batch)
                print(f"[open_clip] embedded texts {done_texts}/{len(texts)}", flush=True)
            if not chunks:
                return np.zeros((0, 512), dtype=np.float32)
            return np.vstack(chunks).astype(np.float32)

        text_arr = encode_texts([item["text"] for item in text_items])
        formula_arr = encode_texts([item["text"] for item in formula_items])

    if image_features:
        visual_arr = np.vstack(image_features).astype(np.float32)
    else:
        visual_arr = np.zeros((0, text_arr.shape[1] if text_arr.size else 512), dtype=np.float32)

    meta = {
        "backend": "open_clip",
        "model_name": model_name,
        "pretrained": pretrained,
        "device": device,
        "image_failures": image_failures,
    }
    return visual_arr, text_arr, formula_arr, meta


def embed_with_tfidf(
    visual_items: list[dict[str, Any]],
    text_items: list[dict[str, Any]],
    formula_items: list[dict[str, Any]],
    max_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    visual_texts = [item.get("proxy_text") or item.get("caption") or item.get("label") or "" for item in visual_items]
    paragraph_texts = [item["text"] for item in text_items]
    formula_texts = [item["text"] for item in formula_items]
    all_texts = visual_texts + paragraph_texts + formula_texts
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english",
        norm="l2",
    )
    matrix = vectorizer.fit_transform(all_texts).astype(np.float32)
    visual_end = len(visual_texts)
    text_end = visual_end + len(paragraph_texts)
    visual_arr = matrix[:visual_end].toarray().astype(np.float32)
    text_arr = matrix[visual_end:text_end].toarray().astype(np.float32)
    formula_arr = matrix[text_end:].toarray().astype(np.float32)
    meta = {
        "backend": "tfidf",
        "max_features": max_features,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "note": "Text proxy backend: visual items are represented by caption/context text, not pixels.",
    }
    return visual_arr, text_arr, formula_arr, meta


def embed_formula_with_math_similarity(
    formula_items: list[dict[str, Any]],
    model_name: str,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Embed formula LaTeX text with a math-aware sentence encoder.

    CLIP's text encoder collapses formula similarity (probe: p50=0.966,
    std=0.027), so it cannot separate genuinely-related formulas from
    coincidental token overlap.  ``math-similarity/Bert-MLM_arXiv-MP-class_arXiv``
    spreads the distribution out (probe: p50=0.817, std=0.172, min=0.036),
    which is what makes a top-k + threshold cut meaningful.
    """
    from sentence_transformers import SentenceTransformer

    if not formula_items:
        return np.zeros((0, 768), dtype=np.float32), {
            "formula_backend": "math_similarity",
            "model_name": model_name,
            "num_formulas": 0,
        }
    model = SentenceTransformer(model_name, device=device)
    texts = [item["text"] for item in formula_items]
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    ).astype(np.float32)
    meta = {
        "formula_backend": "math_similarity",
        "model_name": model_name,
        "device": device,
        "num_formulas": len(formula_items),
        "embedding_dim": int(emb.shape[1]) if emb.size else 0,
    }
    return emb, meta


def node_id(item: dict[str, Any], element_to_node: dict[str, str]) -> str:
    return element_to_node.get(item["element_id"], item["element_id"])


def edge_doc_id(src_doc: str, tgt_doc: str) -> str:
    return src_doc if src_doc == tgt_doc else f"{src_doc}->{tgt_doc}"


def add_edge(
    edges: list[dict[str, Any]],
    seen: set[tuple[str, str, str]],
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float,
    doc_id: str,
    metadata: dict[str, Any],
) -> None:
    if source_id == target_id:
        return
    key = (source_id, target_id, edge_type)
    if key in seen:
        return
    seen.add(key)
    edges.append({
        "source_id": source_id,
        "target_id": target_id,
        "doc_id": doc_id,
        "edge_type": edge_type,
        "weight": round(float(weight), 6),
        "metadata": metadata,
    })


def top_indices(scores: np.ndarray, top_k: int, threshold: float, exclude: set[int] | None = None) -> list[int]:
    if scores.size == 0 or top_k <= 0:
        return []
    exclude = exclude or set()
    order = np.argsort(-scores)
    out: list[int] = []
    for idx in order:
        i = int(idx)
        if i in exclude:
            continue
        if float(scores[i]) < threshold:
            break
        out.append(i)
        if len(out) >= top_k:
            break
    return out


def build_similarity_edges(
    visual_items: list[dict[str, Any]],
    text_items: list[dict[str, Any]],
    formula_items: list[dict[str, Any]],
    visual_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    formula_embeddings: np.ndarray,
    element_to_node: dict[str, str],
    visual_top_k: int,
    visual_threshold: float,
    cross_doc_top_k: int,
    cross_doc_threshold: float,
    text_top_k: int,
    text_threshold: float,
    formula_top_k: int,
    formula_threshold: float,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    if len(visual_items) and visual_embeddings.size:
        vv = visual_embeddings @ visual_embeddings.T
        for i, src in enumerate(visual_items):
            same_or_cross = top_indices(vv[i], visual_top_k, visual_threshold, exclude={i})
            src_id = node_id(src, element_to_node)
            for j in same_or_cross:
                tgt = visual_items[j]
                tgt_id = node_id(tgt, element_to_node)
                sim = float(vv[i, j])
                etype = "cross_doc_visual_sim" if src["doc_id"] != tgt["doc_id"] else "visual_similarity"
                add_edge(
                    edges,
                    seen,
                    src_id,
                    tgt_id,
                    etype,
                    sim,
                    edge_doc_id(src["doc_id"], tgt["doc_id"]),
                    {
                        "similarity": round(sim, 6),
                        "source_element_id": src["element_id"],
                        "target_element_id": tgt["element_id"],
                        "source_visual_type": src["element_type"],
                        "target_visual_type": tgt["element_type"],
                        "rank_scope": "visual_top_k",
                    },
                )

            cross_scores = vv[i].copy()
            for j, tgt in enumerate(visual_items):
                if tgt["doc_id"] == src["doc_id"]:
                    cross_scores[j] = -np.inf
            for j in top_indices(cross_scores, cross_doc_top_k, cross_doc_threshold):
                tgt = visual_items[j]
                tgt_id = node_id(tgt, element_to_node)
                sim = float(vv[i, j])
                add_edge(
                    edges,
                    seen,
                    src_id,
                    tgt_id,
                    "cross_doc_visual_sim",
                    sim,
                    edge_doc_id(src["doc_id"], tgt["doc_id"]),
                    {
                        "similarity": round(sim, 6),
                        "source_element_id": src["element_id"],
                        "target_element_id": tgt["element_id"],
                        "source_visual_type": src["element_type"],
                        "target_visual_type": tgt["element_type"],
                        "rank_scope": "cross_doc_top_k",
                        "passes_visual_threshold": bool(sim >= visual_threshold),
                    },
                )

    if len(text_items) and len(visual_items) and text_embeddings.size and visual_embeddings.size:
        texts_by_doc: dict[str, list[int]] = defaultdict(list)
        for i, item in enumerate(text_items):
            texts_by_doc[item["doc_id"]].append(i)
        tv = text_embeddings @ visual_embeddings.T
        for fig_idx, fig in enumerate(visual_items):
            candidate_texts = texts_by_doc.get(fig["doc_id"], [])
            if not candidate_texts:
                continue
            scores = np.array([tv[text_idx, fig_idx] for text_idx in candidate_texts], dtype=np.float32)
            local_top = top_indices(scores, text_top_k, text_threshold)
            fig_id = node_id(fig, element_to_node)
            for rank, local_idx in enumerate(local_top, start=1):
                text_idx = candidate_texts[local_idx]
                para = text_items[text_idx]
                para_id = node_id(para, element_to_node)
                sim = float(tv[text_idx, fig_idx])
                richness = min(1.25, 1.0 + math.log1p(len(para["text"])) / 25.0)
                weight = sim * richness
                meta = {
                    "similarity": round(sim, 6),
                    "richness_bonus": round(richness, 6),
                    "rank_for_figure": rank,
                    "source_element_id": para["element_id"],
                    "target_element_id": fig["element_id"],
                    "target_visual_type": fig["element_type"],
                    "paragraph_preview": compact_text(para["text"], 240),
                    "caption": fig.get("caption", ""),
                }
                add_edge(edges, seen, para_id, fig_id, "text_describes_figure", weight, fig["doc_id"], meta)
                add_edge(edges, seen, fig_id, para_id, "figure_described_by_text", weight, fig["doc_id"], meta)

    if len(formula_items) and formula_embeddings.size:
        ff = formula_embeddings @ formula_embeddings.T
        for i, src in enumerate(formula_items):
            src_id = node_id(src, element_to_node)
            for j in top_indices(ff[i], formula_top_k, formula_threshold, exclude={i}):
                tgt = formula_items[j]
                tgt_id = node_id(tgt, element_to_node)
                sim = float(ff[i, j])
                add_edge(
                    edges,
                    seen,
                    src_id,
                    tgt_id,
                    "formula_similarity",
                    sim,
                    edge_doc_id(src["doc_id"], tgt["doc_id"]),
                    {
                        "similarity": round(sim, 6),
                        "source_element_id": src["element_id"],
                        "target_element_id": tgt["element_id"],
                        "rank_scope": "formula_top_k",
                    },
                )

    return edges


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MinerU VL Edges v1",
        "",
        f"- backend: **{summary['backend']}**",
        f"- visual items: **{summary['num_visual_items']}**",
        f"- text items: **{summary['num_text_items']}**",
        f"- formula items: **{summary['num_formula_items']}**",
        f"- edges: **{summary['total_edges']}** `{summary['edge_type_counts']}`",
        "",
        "## Embeddings",
        f"- visual shape: `{summary['embedding_shapes']['visual']}`",
        f"- text shape: `{summary['embedding_shapes']['text']}`",
        f"- formula shape: `{summary['embedding_shapes']['formula']}`",
    ]
    if summary["backend"] == "tfidf":
        lines.extend([
            "",
            "## Caveat",
            "- TF-IDF backend is a schema/pipeline fallback; use `--backend open_clip` after installing torch/open_clip for pixel-level visual embeddings.",
        ])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_latest_symlink(out_dir: Path) -> None:
    latest = ROOT / "data/05_eval/mineru_vl_edges_v1_latest"
    try:
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        if not latest.exists():
            latest.symlink_to(out_dir.resolve())
    except OSError as exc:
        print(f"[warn] could not update latest symlink: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MinerU VL edges v1")
    parser.add_argument("--v1-dir", default=str(DEFAULT_V1_DIR))
    parser.add_argument("--elements", default="")
    parser.add_argument("--topology", default=str(DEFAULT_TOPOLOGY_DIR / "mineru_topology_graph_v1.json"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--backend", choices=["auto", "open_clip", "tfidf"], default="auto")
    parser.add_argument(
        "--formula-backend",
        choices=["same", "math_similarity"],
        default="same",
        help="'same' reuses the main backend for formulas; 'math_similarity' uses a math-aware sentence encoder (recommended -- CLIP collapses formula similarity).",
    )
    parser.add_argument(
        "--formula-model-name",
        default="math-similarity/Bert-MLM_arXiv-MP-class_arXiv",
        help="SentenceTransformer model used when --formula-backend math_similarity.",
    )
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tfidf-max-features", type=int, default=2048)
    parser.add_argument("--visual-top-k", type=int, default=5)
    parser.add_argument("--visual-threshold", type=float, default=0.70)
    parser.add_argument("--cross-doc-top-k", type=int, default=3)
    parser.add_argument("--cross-doc-threshold", type=float, default=0.0)
    parser.add_argument("--text-top-k", type=int, default=3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--formula-top-k", type=int, default=5)
    parser.add_argument(
        "--formula-threshold",
        type=float,
        default=None,
        help="Floor for formula_similarity edges. Default depends on backend: 0.45 for clip/tfidf text, 0.85 for math_similarity (whose scores spread far wider).",
    )
    args = parser.parse_args()

    if args.formula_threshold is None:
        args.formula_threshold = 0.85 if args.formula_backend == "math_similarity" else 0.45

    v1_dir = Path(args.v1_dir)
    elements_path = Path(args.elements) if args.elements else v1_dir / "mineru_elements_v1.json"
    topology_path = Path(args.topology) if args.topology else None
    if not elements_path.exists():
        raise FileNotFoundError(elements_path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / f"data/05_eval/mineru_vl_edges_v1_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    emb_dir = out_dir / "mineru_vl_embeddings_v1"
    emb_dir.mkdir(parents=True, exist_ok=True)

    v1_graph = read_json(elements_path)
    element_to_node = load_element_to_node(topology_path)
    visual_items, text_items, formula_items = collect_items(v1_graph)
    backend = choose_backend(args.backend)

    if backend == "open_clip":
        visual_embeddings, text_embeddings, formula_embeddings, backend_meta = embed_with_open_clip(
            visual_items,
            text_items,
            formula_items,
            args.model_name,
            args.pretrained,
            args.batch_size,
            args.device,
        )
    else:
        visual_embeddings, text_embeddings, formula_embeddings, backend_meta = embed_with_tfidf(
            visual_items,
            text_items,
            formula_items,
            args.tfidf_max_features,
        )

    if args.formula_backend == "math_similarity":
        formula_embeddings, formula_meta = embed_formula_with_math_similarity(
            formula_items,
            args.formula_model_name,
            args.batch_size,
            args.device,
        )
        backend_meta["formula_override"] = formula_meta

    visual_embeddings = l2_normalize(visual_embeddings)
    text_embeddings = l2_normalize(text_embeddings)
    formula_embeddings = l2_normalize(formula_embeddings)

    np.save(emb_dir / "visual_embeddings.npy", visual_embeddings)
    np.save(emb_dir / "text_embeddings.npy", text_embeddings)
    np.save(emb_dir / "formula_embeddings.npy", formula_embeddings)
    (emb_dir / "visual_items.json").write_text(json.dumps(visual_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (emb_dir / "text_items.json").write_text(json.dumps(text_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (emb_dir / "formula_items.json").write_text(json.dumps(formula_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    edges = build_similarity_edges(
        visual_items,
        text_items,
        formula_items,
        visual_embeddings,
        text_embeddings,
        formula_embeddings,
        element_to_node,
        args.visual_top_k,
        args.visual_threshold,
        args.cross_doc_top_k,
        args.cross_doc_threshold,
        args.text_top_k,
        args.text_threshold,
        args.formula_top_k,
        args.formula_threshold,
    )

    summary = {
        "builder": "mineru_vl_edges_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": backend,
        "formula_backend": args.formula_backend if args.formula_backend != "same" else backend,
        "backend_metadata": backend_meta,
        "source_elements": str(elements_path),
        "source_topology": str(topology_path) if topology_path else "",
        "num_visual_items": len(visual_items),
        "num_text_items": len(text_items),
        "num_formula_items": len(formula_items),
        "total_edges": len(edges),
        "edge_type_counts": dict(Counter(edge["edge_type"] for edge in edges)),
        "embedding_shapes": {
            "visual": list(visual_embeddings.shape),
            "text": list(text_embeddings.shape),
            "formula": list(formula_embeddings.shape),
        },
        "thresholds": {
            "visual_threshold": args.visual_threshold,
            "cross_doc_threshold": args.cross_doc_threshold,
            "text_threshold": args.text_threshold,
            "formula_threshold": args.formula_threshold,
        },
    }

    write_jsonl(out_dir / "mineru_vl_edges_v1.jsonl", edges)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(out_dir, summary)
    update_latest_symlink(out_dir)

    print(f"[ok] wrote {out_dir / 'mineru_vl_edges_v1.jsonl'}")
    print(f"backend={backend} edges={len(edges)} counts={summary['edge_type_counts']}")


if __name__ == "__main__":
    main()
