"""Prep delivery sweep chunks (2026-05-12) — split candidate pools for 7-cell array.

Input pools:
  - method_c_true2_candidates_2026-04-12T050859Z.json (817 intra-doc pairs)
  - hub_candidates_enriched_v4_intra_doc.json (96 pairs)
  - hub_candidates_enriched_v4_intra_doc_long_seed.json (88 pairs)
  - m2_diverse_candidates_intra_doc.json (108 pairs, after dedup ≈76 unconsumed)

Output: data/02_enriched/delivery_chunks_20260512/
  cell_0_methodc_a.json   ← method_c_true2[0:205]
  cell_1_methodc_b.json   ← method_c_true2[205:410]
  cell_2_methodc_c.json   ← method_c_true2[410:615]
  cell_3_methodc_d.json   ← method_c_true2[615:817]
  cell_4_hub_v4.json      ← hub_v4_intra_doc full (96)
  cell_5_m2_diverse.json  ← m2_diverse_intra full (108)
  cell_6_long_seed.json   ← hub_v4_long_seed full (88)

Idempotent — running twice produces identical files.
Wraps each output in the same {metadata, pairs} schema as upstream files.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUTS = {
    "method_c": PROJECT_ROOT / "data/03_queries/method_c_true2_candidates_2026-04-12T050859Z.json",
    "hub_v4":   PROJECT_ROOT / "data/02_enriched/hub_candidates_enriched_v4_intra_doc.json",
    "m2_div":   PROJECT_ROOT / "data/02_enriched/m2_diverse_candidates_intra_doc.json",
    "long_seed": PROJECT_ROOT / "data/02_enriched/hub_candidates_enriched_v4_intra_doc_long_seed.json",
}

OUT_DIR = PROJECT_ROOT / "data/02_enriched/delivery_chunks_20260512"

# method_c_true2: 817 pairs → 4 chunks of ~205 each
METHOD_C_SLICES = [(0, 205), (205, 410), (410, 615), (615, 817)]


def _load_pairs(fp: Path) -> tuple[dict, list]:
    """Return (metadata-dict, pairs-list) from a candidate file."""
    with open(fp, encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, list):
        return ({}, d)
    pairs = d.get("pairs") or d.get("candidates") or []
    meta = {k: v for k, v in d.items() if k not in ("pairs", "candidates")}
    return meta, pairs


def _write_chunk(out_fp: Path, pairs: list, source_tag: str, source_meta: dict) -> None:
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "prep_script": "scripts/prep_delivery_chunks.py",
            "source_tag": source_tag,
            "pair_count": len(pairs),
            "source_metadata": source_meta,
        },
        "summary": {"pair_count": len(pairs)},
        "pairs": pairs,
    }
    with open(out_fp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  wrote {len(pairs):4d} pairs → {out_fp.relative_to(PROJECT_ROOT)}")


def main() -> None:
    print(f"Output dir: {OUT_DIR.relative_to(PROJECT_ROOT)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # method_c_true2 → 4 chunks
    mc_meta, mc_pairs = _load_pairs(INPUTS["method_c"])
    print(f"\nMethod-C true2: {len(mc_pairs)} total pairs")
    chunk_names = ["a", "b", "c", "d"]
    for idx, (lo, hi) in enumerate(METHOD_C_SLICES):
        out = OUT_DIR / f"cell_{idx}_methodc_{chunk_names[idx]}.json"
        _write_chunk(out, mc_pairs[lo:hi], f"method_c_true2[{lo}:{hi}]", mc_meta)

    # hub_v4_intra_doc → cell 4
    hub_meta, hub_pairs = _load_pairs(INPUTS["hub_v4"])
    print(f"\nhub_v4_intra_doc: {len(hub_pairs)} pairs")
    _write_chunk(OUT_DIR / "cell_4_hub_v4.json", hub_pairs, "hub_v4_intra_doc", hub_meta)

    # m2_diverse_intra → cell 5
    m2_meta, m2_pairs = _load_pairs(INPUTS["m2_div"])
    print(f"\nm2_diverse_intra: {len(m2_pairs)} pairs")
    _write_chunk(OUT_DIR / "cell_5_m2_diverse.json", m2_pairs, "m2_diverse_intra", m2_meta)

    # long_seed → cell 6 (L3 mode)
    ls_meta, ls_pairs = _load_pairs(INPUTS["long_seed"])
    print(f"\nhub_v4_long_seed: {len(ls_pairs)} pairs")
    _write_chunk(OUT_DIR / "cell_6_long_seed.json", ls_pairs, "hub_v4_long_seed", ls_meta)

    print(f"\nDone. 7 chunk files in {OUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
