#!/usr/bin/env python3
"""Filter pair-style candidate assets to strict intra-document subsets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.pair_filters import filter_intra_doc_pairs


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Input candidate JSON with top-level pairs[]")
    ap.add_argument("--output", required=True, help="Filtered output JSON")
    ap.add_argument("--min-hop", type=int, default=0, help="Optional min hop filter after intra-doc cleanup")
    ap.add_argument("--max-hop", type=int, default=0, help="Optional max hop filter after intra-doc cleanup")
    args = ap.parse_args()

    in_path = PROJECT_ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    out_path = PROJECT_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)

    data = _load_json(in_path)
    raw_pairs: List[Dict[str, Any]] = data.get("pairs", []) or []
    intra_pairs, strict_stats = filter_intra_doc_pairs(raw_pairs)

    hop_filtered = []
    hop_drop = 0
    for pair in intra_pairs:
        hop = int(pair.get("hop_distance", 0))
        if args.min_hop > 0 and hop < args.min_hop:
            hop_drop += 1
            continue
        if args.max_hop > 0 and hop > args.max_hop:
            hop_drop += 1
            continue
        hop_filtered.append(pair)

    by_type = Counter(str(pair.get("pair_type", "")) for pair in hop_filtered)
    by_hop = Counter(str(int(pair.get("hop_distance", 0))) for pair in hop_filtered)
    docs_covered = len({str(pair.get("doc_id", "") or "").strip() for pair in hop_filtered if str(pair.get("doc_id", "") or "").strip()})

    data["pairs"] = hop_filtered
    data.setdefault("metadata", {})
    data["metadata"]["strict_intra_doc_filter"] = {
        "source": "filter_pair_candidates.py",
        "input_pairs": len(raw_pairs),
        "strict_stats": strict_stats,
        "min_hop": args.min_hop,
        "max_hop": args.max_hop,
        "hop_filtered_out": hop_drop,
        "output_pairs": len(hop_filtered),
    }
    data["summary"] = {
        "total_selected": len(hop_filtered),
        "by_type": dict(by_type),
        "by_hop": dict(by_hop),
        "docs_covered": docs_covered,
        "strict_intra_doc_pairs": len(hop_filtered),
        "cross_doc": 0,
        "intra_doc": len(hop_filtered),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[done] strict intra-doc filtering complete")
    print(f"  input pairs   : {len(raw_pairs)}")
    print(f"  intra kept    : {strict_stats.get('kept', 0)}")
    print(f"  dropped flag  : {strict_stats.get('drop_cross_doc_flag', 0)}")
    print(f"  dropped mixed : {strict_stats.get('drop_mixed_doc_ids', 0)}")
    print(f"  dropped nodoc : {strict_stats.get('drop_missing_doc_ids', 0)}")
    print(f"  hop filtered  : {hop_drop}")
    print(f"  output pairs  : {len(hop_filtered)}")
    print(f"  output file   : {out_path}")


if __name__ == "__main__":
    main()
