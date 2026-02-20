#!/usr/bin/env python3
"""
Export a stratified random sample of citation edges to CSV for manual review.

Fixes over the previous version (your friend's script):
  1. Correct field paths: edge["bib_title"] / edge["contexts"] / edge["match_method"]
     (NOT edge["metadata"]["bib_entry"]["title"] / edge["metadata"]["context"])
  2. Random sampling instead of list slicing -- avoids ordering bias.
  3. Strict sample_size cutoff -- output is always exactly SAMPLE_SIZE rows (or fewer
     if the graph has fewer fuzzy edges than requested).
  4. exact/fuzzy split uses edge["match_method"] directly, not conf==1.0 heuristic.

Stratification:
  - 50% from HIGH-risk bucket  (confidence < 0.65)
  - ~33% from MID-risk bucket  (0.65 <= confidence < 0.75)
  - remainder from LOW-risk    (confidence >= 0.75, fuzzy only)
  - exact-match edges (arxiv_id_explicit / arxiv_id_bare / title_exact) are skipped
    unless the graph has fewer fuzzy edges than SAMPLE_SIZE.

Usage:
    python scripts/export_review_csv.py
    python scripts/export_review_csv.py --input data/citation_graph.json \\
        --output data/citation_review.csv --sample 20
"""

import argparse
import csv
import json
import random
from pathlib import Path

SAMPLE_SIZE = 20
RANDOM_SEED = 42

# Exact-match methods we skip during normal review (reliable enough)
EXACT_METHODS = {"arxiv_id_explicit", "arxiv_id_bare", "title_exact"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_ctx(v, max_len: int = 220) -> str:
    """Flatten a context value (str or list) to a single truncated string."""
    if v is None:
        return ""
    if isinstance(v, list):
        joined = " || ".join(str(x).replace("\n", " ").strip() for x in v[:3])
    else:
        joined = str(v).replace("\n", " ").strip()
    return (joined[:max_len] + "...") if len(joined) > max_len else joined


def _extract_row(edge: dict) -> dict:
    """
    Extract review columns from a single edge dict.

    Real citation_graph.json schema (as built by build_citation_graph.py):
        edge["source"]        str  -- citing arxiv ID
        edge["target"]        str  -- cited  arxiv ID
        edge["bib_title"]     str  -- title from .bbl entry  ← correct path
        edge["contexts"]      list -- cite-context snippets  ← correct path
        edge["match_method"]  str  -- how we matched
        edge["confidence"]    float
    """
    conf   = float(edge.get("confidence", 0.0))
    method = str(edge.get("match_method", "unknown"))
    src    = str(edge.get("source", ""))
    tgt    = str(edge.get("target", ""))

    # Title: top-level bib_title (not nested in metadata)
    title = str(edge.get("bib_title") or "N/A").strip() or "N/A"

    # Context: top-level contexts list (not metadata.context)
    ctx = _normalize_ctx(edge.get("contexts"))

    return {
        "Review_Label":   "",
        "Confidence":     f"{conf:.3f}",
        "Match_Method":   method,
        "Source_ID":      src,
        "Target_ID":      tgt,
        "Matched_Title":  title,
        "Context_Snippet": ctx,
        "Notes":          "",
    }


def _sample(lst: list, k: int, rng: random.Random) -> list:
    if not lst or k <= 0:
        return []
    return rng.sample(lst, min(k, len(lst)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_path: str, output_path: str, sample_size: int, seed: int) -> None:
    in_p  = Path(input_path)
    out_p = Path(output_path)

    if not in_p.exists():
        raise FileNotFoundError(f"Input not found: {in_p.resolve()}")

    with open(in_p, encoding="utf-8") as f:
        data = json.load(f)

    edges = data.get("edges", [])
    if not edges:
        raise RuntimeError(
            "No 'edges' found in the JSON. "
            "Run inspect_graph.py to check the actual top-level keys."
        )

    print(f"Total edges: {len(edges)}")

    rng = random.Random(seed)

    # --- Separate exact vs fuzzy; bucket fuzzy by confidence ---
    high, mid, low, exact_edges = [], [], [], []
    for e in edges:
        if not isinstance(e, dict):
            continue
        method = str(e.get("match_method", "")).lower()
        conf   = float(e.get("confidence", 0.0))

        if method in EXACT_METHODS or conf >= 1.0:
            exact_edges.append(e)
            continue

        if conf < 0.65:
            high.append(e)
        elif conf < 0.75:
            mid.append(e)
        else:
            low.append(e)

    fuzzy_total = len(high) + len(mid) + len(low)
    print(
        f"Fuzzy edges: {fuzzy_total}  "
        f"(high<0.65: {len(high)}, mid 0.65-0.75: {len(mid)}, low>=0.75: {len(low)})"
    )
    print(f"Exact edges (skipped): {len(exact_edges)}")

    # --- Stratified random sample ---
    n_high = sample_size // 2                         # 50% high-risk
    n_mid  = sample_size // 3                         # ~33% mid-risk
    n_low  = sample_size - n_high - n_mid             # rest low-risk

    pick = []
    pick += _sample(high, n_high, rng)
    pick += _sample(mid,  n_mid,  rng)
    pick += _sample(low,  n_low,  rng)

    # Back-fill from other buckets if we couldn't fill from the primary ones
    if len(pick) < sample_size:
        already = set(id(e) for e in pick)
        pool = [e for e in (high + mid + low) if id(e) not in already]
        rng.shuffle(pool)
        pick += pool[: sample_size - len(pick)]

    # If still short, allow exact edges as last resort (good ground-truth contrast)
    if len(pick) < sample_size:
        already = set(id(e) for e in pick)
        pool = [e for e in exact_edges if id(e) not in already]
        rng.shuffle(pool)
        pick += pool[: sample_size - len(pick)]

    # Strict cap
    pick = pick[:sample_size]
    rng.shuffle(pick)  # shuffle so reviewers don't see all high-risk first

    print(f"Sampled: {len(pick)} edges for review")

    # --- Write CSV ---
    headers = [
        "Review_Label", "Confidence", "Match_Method",
        "Source_ID", "Target_ID", "Matched_Title",
        "Context_Snippet", "Notes",
    ]
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for e in pick:
            w.writerow(_extract_row(e))

    print(f"Wrote: {out_p.resolve()}")
    print()
    print("Next steps:")
    print("  1. Open the CSV and fill Review_Label:  1=accept  0=reject  ?=borderline")
    print("  2. Run:  python scripts/evaluate_review.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Export stratified random citation sample for manual review"
    )
    ap.add_argument(
        "--input",  default="data/citation_graph.json",
        help="Path to citation_graph.json",
    )
    ap.add_argument(
        "--output", default="data/citation_review.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--sample", type=int, default=SAMPLE_SIZE,
        help=f"Number of edges to sample (default: {SAMPLE_SIZE})",
    )
    ap.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )
    args = ap.parse_args()
    main(args.input, args.output, args.sample, args.seed)
