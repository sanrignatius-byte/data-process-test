#!/usr/bin/env python3
"""Sanity test: load Qwen3-VL-Embedding-2B and verify language_model weights are real (not random).

Usage:
  python scripts/sanity_test_qwen3vl_emb_load.py

Pass criteria:
  1. transformers version >= 5.2
  2. SentenceTransformer loads with NO "newly initialized" warnings on language_model.*
  3. Encoding two semantically-different texts produces well-separated embeddings
     (cosine similarity < 0.95 — random encoder would give ~0.99 for any pair)
  4. Encoding same text twice is deterministic (cosine == 1.0)
"""

from __future__ import annotations

import sys
import warnings
import logging
import io
import contextlib

import numpy as np


def main():
    import transformers
    import sentence_transformers

    print(f"transformers       = {transformers.__version__}")
    print(f"sentence_transformers = {sentence_transformers.__version__}")

    v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if v < (5, 2):
        print(f"FAIL: transformers {transformers.__version__} < 5.2")
        sys.exit(1)

    # Capture loader warnings
    log_buf = io.StringIO()
    log_handler = logging.StreamHandler(log_buf)
    log_handler.setLevel(logging.WARNING)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.addHandler(log_handler)
    transformers_logger.setLevel(logging.WARNING)

    print(f"\nLoading Qwen3-VL-Embedding-2B...")
    from sentence_transformers import SentenceTransformer
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = SentenceTransformer(
            "/projects/myyyx1/models/Qwen3-VL-Embedding-2B",
            device="cuda" if _cuda_available() else "cpu",
            trust_remote_code=True,
        )

    captured = log_buf.getvalue()
    print(f"\n--- Loader warnings ({len(captured.split(chr(10)))} lines) ---")
    bad_phrases = ["newly initialized", "not initialized from the model"]
    found_bad = [p for p in bad_phrases if p in captured]
    if found_bad:
        print(f"FAIL: loader emitted bad warnings:")
        for p in found_bad:
            for line in captured.split("\n"):
                if p in line:
                    print(f"  {line[:200]}")
        sys.exit(2)
    print("OK: no 'newly initialized' / 'not initialized' warnings on load.")

    # Encoding sanity
    print(f"\nTesting query encoding...")
    texts = [
        "Image classification with deep convolutional networks.",
        "The Shanghai stock exchange closed up 1.5% today.",
    ]
    e0 = model.encode(texts[0], normalize_embeddings=True)
    e1 = model.encode(texts[1], normalize_embeddings=True)
    e0b = model.encode(texts[0], normalize_embeddings=True)

    cos_diff = float(np.dot(e0, e1))
    cos_same = float(np.dot(e0, e0b))
    print(f"  cos(text_A, text_B) = {cos_diff:.4f}  (expect < 0.95)")
    print(f"  cos(text_A, text_A) = {cos_same:.4f}  (expect ≈ 1.0)")

    if cos_same < 0.999:
        print(f"FAIL: same-text cosine {cos_same:.4f} < 0.999 → encoding is non-deterministic")
        sys.exit(3)
    if cos_diff > 0.95:
        print(f"FAIL: different-text cosine {cos_diff:.4f} > 0.95 → encoder may be random")
        sys.exit(4)

    print(f"\nOK: model loads cleanly and produces meaningful text embeddings.")
    print(f"Ready to submit slurm 41b.")


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
