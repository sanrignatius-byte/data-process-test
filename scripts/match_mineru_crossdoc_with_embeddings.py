#!/usr/bin/env python3
"""
Cross-document MinerU element matching with embedding similarity.

Backends:
- openai: OpenAI-compatible embeddings API
- local: HuggingFace local embedding model (e.g. Qwen/Qwen3-Embedding-8B)

Output:
- JSONL with per-source top-k cross-doc matches (same modality)
- JSON report with probe/config/stats
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


REPO_ROOTS = (
    "/projects/_hdd/myyyx1/data-process-test/",
    "/projects/myyyx1/data-process-test/",
)


@dataclass
class ElementItem:
    element_id: str
    doc_id: str
    element_type: str
    text: str
    image_path: Optional[str]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


def normalize_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = str(path).strip().replace("\\", "/")
    if not p:
        return None
    for root in REPO_ROOTS:
        if p.startswith(root):
            return p[len(root):]
    # Generic fallback: find '/data/' and keep relative path
    idx = p.find("/data/")
    if idx >= 0:
        return p[idx + 1:]
    return p


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def truncate(text: str, n: int) -> str:
    if n <= 0:
        return ""
    if len(text) <= n:
        return text
    return text[: n - 3].rstrip() + "..."


def build_element_text(elem: Dict[str, Any], max_chars: int) -> str:
    etype = str(elem.get("element_type", "unknown"))
    caption = clean_text(str(elem.get("caption", "")))
    content = clean_text(str(elem.get("content", "")))
    ctx_before = clean_text(str(elem.get("context_before", "")))
    ctx_after = clean_text(str(elem.get("context_after", "")))

    parts = [f"Type: {etype}."]
    if caption:
        parts.append(f"Caption: {caption}")
    if content:
        parts.append(f"Content: {content}")
    if ctx_before:
        parts.append(f"Context before: {ctx_before}")
    if ctx_after:
        parts.append(f"Context after: {ctx_after}")
    return truncate("\n".join(parts), max_chars)


def load_elements(
    multimodal_path: Path,
    modalities: List[str],
    max_text_chars: int,
    min_text_chars: int,
    max_elements: int,
    doc_ids: Optional[set],
) -> List[ElementItem]:
    data = json.loads(multimodal_path.read_text(encoding="utf-8"))
    docs = data.get("documents", {})
    out: List[ElementItem] = []

    for doc_id, doc in docs.items():
        if doc_ids is not None and doc_id not in doc_ids:
            continue
        elems = doc.get("elements", {})
        for eid, elem in elems.items():
            etype = str(elem.get("element_type", "unknown")).lower()
            if etype not in modalities:
                continue
            txt = build_element_text(elem, max_chars=max_text_chars)
            if len(txt) < min_text_chars:
                continue
            out.append(
                ElementItem(
                    element_id=eid,
                    doc_id=doc_id,
                    element_type=etype,
                    text=txt,
                    image_path=normalize_path(elem.get("image_path")),
                )
            )
            if max_elements > 0 and len(out) >= max_elements:
                return out

    return out


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def build_doc_filter_from_queries(path: Path) -> set:
    out = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            row = json.loads(s)
            doc_id = str(row.get("doc_id", "")).strip()
            if doc_id:
                out.add(doc_id)
    return out


# ----------------------------
# OpenAI-compatible backend
# ----------------------------
def probe_models_openai(client: Any, models: List[str]) -> List[Dict[str, Any]]:
    out = []
    for model in models:
        item: Dict[str, Any] = {"model": model, "ok": False, "error": "", "dim": 0}
        try:
            r = client.embeddings.create(model=model, input="embedding probe")
            dim = len(r.data[0].embedding) if r.data else 0
            item["ok"] = True
            item["dim"] = dim
        except Exception as e:
            item["error"] = str(e).replace("\n", " ")[:300]
        out.append(item)
    return out


def embed_texts_openai(client: Any, model: str, texts: List[str], batch_size: int) -> np.ndarray:
    vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        data = sorted(resp.data, key=lambda x: x.index)
        vecs.extend([d.embedding for d in data])
    return np.asarray(vecs, dtype=np.float32)


# ----------------------------
# Local HF backend
# ----------------------------
def embed_texts_local(
    model_path_or_id: str,
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: str,
    dtype: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype_map = {
        "auto": torch.bfloat16 if device.startswith("cuda") else torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, dtype_map["auto"])

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path_or_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.eval()
    model.to(device)

    vecs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(**enc)
            if hasattr(out, "last_hidden_state"):
                hidden = out.last_hidden_state  # [B, T, D]
            elif isinstance(out, tuple) and len(out) > 0:
                hidden = out[0]
            else:
                raise RuntimeError("Unsupported model output for embedding extraction")

            mask = enc.get("attention_mask")
            if mask is None:
                emb = hidden.mean(dim=1)
            else:
                mask = mask.unsqueeze(-1).to(hidden.dtype)
                summed = (hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-6)
                emb = summed / denom

            vecs.append(emb.detach().float().cpu().numpy())

    arr = np.concatenate(vecs, axis=0).astype(np.float32)
    meta = {
        "device": device,
        "dtype": str(torch_dtype).replace("torch.", ""),
        "max_length": max_length,
    }
    return arr, meta


def run_matching(
    elements: List[ElementItem],
    embeddings: np.ndarray,
    top_k: int,
    normalize: bool,
    selected_model: str,
    output_path: Path,
) -> int:
    if normalize:
        embeddings = l2_normalize(embeddings)

    idx_by_type: Dict[str, List[int]] = {}
    for i, e in enumerate(elements):
        idx_by_type.setdefault(e.element_type, []).append(i)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for etype, idxs in idx_by_type.items():
            if not idxs:
                continue
            mat = embeddings[np.asarray(idxs)]
            sim = cosine_matrix(mat, mat)

            for local_i, src_global_idx in enumerate(idxs):
                src = elements[src_global_idx]
                row = sim[local_i].copy()

                # mask self + same doc
                for local_j, tgt_global_idx in enumerate(idxs):
                    tgt = elements[tgt_global_idx]
                    if tgt_global_idx == src_global_idx or tgt.doc_id == src.doc_id:
                        row[local_j] = -1e9

                k = min(top_k, len(idxs))
                if k <= 0:
                    top_local = []
                else:
                    top_local = np.argpartition(-row, kth=max(0, k - 1))[:k]
                    top_local = sorted(top_local, key=lambda x: row[x], reverse=True)

                matches = []
                for li in top_local:
                    if row[li] <= -1e8:
                        continue
                    tgt = elements[idxs[li]]
                    matches.append(
                        {
                            "target_element_id": tgt.element_id,
                            "target_doc_id": tgt.doc_id,
                            "target_type": tgt.element_type,
                            "score": round(float(row[li]), 6),
                            "target_image_path": tgt.image_path,
                        }
                    )

                rec = {
                    "source_element_id": src.element_id,
                    "source_doc_id": src.doc_id,
                    "source_type": src.element_type,
                    "source_image_path": src.image_path,
                    "model": selected_model,
                    "matches": matches,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-document MinerU matching with embeddings")
    ap.add_argument("--backend", choices=["openai", "local"], default="openai")

    ap.add_argument("--elements", default="data/multimodal_elements.json")
    ap.add_argument("--output", default="data/mineru_crossdoc_embedding_matches_v1.jsonl")
    ap.add_argument("--report", default="data/mineru_crossdoc_embedding_matches_v1_report.json")
    ap.add_argument("--queries", default="", help="Optional query JSONL to restrict doc_ids")

    # OpenAI-compatible args
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--model", default="qwen3-embedding")
    ap.add_argument("--probe-models", nargs="*", default=[
        "qwen3-embedding",
        "Qwen/Qwen3-Embedding-0.6B",
        "Qwen/Qwen3-Embedding-8B",
        "text-embedding-3-small",
    ])
    ap.add_argument("--probe-only", action="store_true")

    # Local backend args
    ap.add_argument("--local-model-path", default="")
    ap.add_argument("--local-device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--local-dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--local-max-length", type=int, default=512)

    # Shared matching args
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-elements", type=int, default=0)
    ap.add_argument("--max-text-chars", type=int, default=900)
    ap.add_argument("--min-text-chars", type=int, default=40)
    ap.add_argument("--modalities", nargs="*", default=["figure", "table", "formula"])
    ap.add_argument("--normalize", action="store_true", default=True)
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = project_root / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    elem_path = Path(args.elements)
    if not elem_path.is_absolute():
        elem_path = project_root / elem_path

    doc_ids = None
    if args.queries:
        qpath = Path(args.queries)
        if not qpath.is_absolute():
            qpath = project_root / qpath
        if qpath.exists():
            doc_ids = build_doc_filter_from_queries(qpath)

    report: Dict[str, Any] = {
        "backend": args.backend,
        "selected_model": None,
        "elements_loaded": 0,
        "matches_written": 0,
        "probe": [],
    }

    elements = load_elements(
        multimodal_path=elem_path,
        modalities=[m.lower() for m in args.modalities],
        max_text_chars=args.max_text_chars,
        min_text_chars=args.min_text_chars,
        max_elements=args.max_elements,
        doc_ids=doc_ids,
    )
    report["elements_loaded"] = len(elements)

    if not elements:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("No elements loaded. Report:", report_path)
        return

    texts = [e.text for e in elements]

    if args.backend == "openai":
        from openai import OpenAI

        env_path = Path(args.env_file)
        if not env_path.is_absolute():
            env_path = project_root / env_path
        load_env(env_path)

        api_key = os.getenv(args.api_key_env, "")
        if not api_key:
            raise SystemExit(f"ERROR: missing {args.api_key_env}")

        base_url = args.base_url.strip() or os.getenv("OPENAI_BASE_URL", "").strip()
        report["base_url"] = base_url or "default_openai"

        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        probes = probe_models_openai(client, args.probe_models)
        report["probe"] = probes

        ok_models = [p for p in probes if p.get("ok")]
        selected_model = args.model
        if not any(p["ok"] and p["model"] == selected_model for p in probes):
            if ok_models:
                selected_model = ok_models[0]["model"]
            else:
                report["selected_model"] = None
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print("No usable embedding model found. See report:", report_path)
                return

        report["selected_model"] = selected_model
        if args.probe_only:
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print("Probe complete. Report:", report_path)
            return

        embs = embed_texts_openai(client=client, model=selected_model, texts=texts, batch_size=args.batch_size)

    else:
        # local backend
        local_model = args.local_model_path.strip() or args.model
        report["selected_model"] = local_model
        if args.probe_only:
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print("Probe complete (local mode, no API probe). Report:", report_path)
            return

        embs, local_meta = embed_texts_local(
            model_path_or_id=local_model,
            texts=texts,
            batch_size=args.batch_size,
            max_length=args.local_max_length,
            device=args.local_device,
            dtype=args.local_dtype,
        )
        report["local_runtime"] = local_meta

    written = run_matching(
        elements=elements,
        embeddings=embs,
        top_k=args.top_k,
        normalize=args.normalize,
        selected_model=str(report["selected_model"]),
        output_path=output_path,
    )

    report["matches_written"] = written
    report["output"] = str(output_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("MinerU Cross-Doc Embedding Match Summary")
    print("=" * 60)
    print(f"Backend:  {args.backend}")
    if args.backend == "openai":
        print(f"Base URL: {report.get('base_url', 'default_openai')}")
    print(f"Model:    {report['selected_model']}")
    print(f"Elements: {len(elements)}")
    print(f"Records:  {written}")
    print(f"Output:   {output_path}")
    print(f"Report:   {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
