#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize phase0 grid reports")
    ap.add_argument("--dir", type=Path, required=True, help="Directory with report_*.json files")
    args = ap.parse_args()

    files = sorted(args.dir.glob("report_a*_n*.json"))
    if not files:
        raise SystemExit(f"No grid reports found in {args.dir}")

    rows = []
    for p in files:
        obj = json.loads(p.read_text(encoding="utf-8"))
        cfg = obj.get("config", {})
        m_b = obj["metrics"]["bm25"]
        m_g = obj["metrics"]["graph_hub_rerank"]
        rows.append(
            {
                "file": str(p),
                "alpha": float(cfg.get("graph_alpha", 0.0)),
                "topn": int(cfg.get("graph_rerank_topn", 0)),
                "bm25_r10": float(m_b["recall_at_10"]),
                "bm25_mrr": float(m_b["mrr"]),
                "graph_r10": float(m_g["recall_at_10"]),
                "graph_mrr": float(m_g["mrr"]),
                "d_r10": float(m_g["recall_at_10"]) - float(m_b["recall_at_10"]),
                "d_mrr": float(m_g["mrr"]) - float(m_b["mrr"]),
            }
        )

    # primary rank by delta recall then delta mrr
    rows_sorted = sorted(rows, key=lambda x: (x["d_r10"], x["d_mrr"]), reverse=True)

    print("Top configs (by ΔRecall@10 then ΔMRR):")
    for r in rows_sorted[:10]:
        print(
            f"alpha={r['alpha']:.1f} topn={r['topn']:>3d} | "
            f"graph R10={r['graph_r10']:.4f} MRR={r['graph_mrr']:.4f} | "
            f"ΔR10={r['d_r10']:+.4f} ΔMRR={r['d_mrr']:+.4f}"
        )

    out_json = args.dir / "summary.json"
    out_json.write_text(json.dumps({"rows": rows_sorted}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {out_json}")


if __name__ == "__main__":
    main()
