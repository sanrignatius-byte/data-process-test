#!/usr/bin/env python3
"""Threshold portability check for the cross-doc rerank tiers.

Goal (todo): the rerank thresholds (min_text_sim, min_combined_score) and the
formula/visual cuts were chosen on this corpus.  When we move to PDF-only
documents we will NOT have LaTeX silver to re-tune them, so we need to know the
thresholds are not over-fit to a particular slice of documents.

Caveat, stated plainly: this 53-doc corpus has only ONE doc without LaTeX
(1805.03677), so a real LaTeX-vs-PDF-only split is not statistically meaningful.
Instead this does a *split-half stability* test, which is the right proxy: if
the score distributions and resulting tier proportions are stable across two
disjoint halves of the documents, then a threshold tuned on one set of docs
transfers to unseen docs (the PDF-only future case) without re-tuning.

We split documents into halves A/B by a hash of the doc_id (deterministic,
roughly balanced), bucket every reranked cross-doc edge by the source doc, and
compare per-half:
  - score quantiles (visual / caption / context / enriched / combined)
  - tier proportions
A small max tier-proportion delta and similar quantiles => portable thresholds.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RERANK = ROOT / "data/05_eval/mineru_crossdoc_text_rerank_v1_latest/mineru_crossdoc_text_rerank_edges_v1.jsonl"
OUT_DIR = ROOT / f"data/05_eval/rerank_threshold_portability_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

SCORE_FIELDS = ["visual_score", "caption_sim", "context_sim", "enriched_sim", "all_text_sim", "combined_score"]


def doc_of(node_id: str) -> str:
    return node_id.split("::", 1)[0] if "::" in node_id else node_id.split("_", 1)[0]


def half(doc_id: str) -> str:
    h = int(hashlib.md5(doc_id.encode()).hexdigest(), 16)
    return "A" if h % 2 == 0 else "B"


def quantiles(vals: list[float]) -> dict:
    if not vals:
        return {}
    vs = sorted(vals)
    def q(p):
        return round(vs[min(len(vs) - 1, int(round((len(vs) - 1) * p)))], 4)
    return {"n": len(vs), "p10": q(.1), "p50": q(.5), "p90": q(.9),
            "mean": round(statistics.mean(vs), 4)}


def main():
    rows = [json.loads(l) for l in RERANK.read_text().splitlines() if l.strip()]
    buckets: dict[str, list[dict]] = defaultdict(list)
    docs_in_half: dict[str, set] = defaultdict(set)
    for r in rows:
        d = doc_of(r["source_id"])
        hlf = half(d)
        buckets[hlf].append(r)
        docs_in_half[hlf].add(d)

    per_half = {}
    for hlf in ("A", "B"):
        b = buckets[hlf]
        tier_counts = Counter(r.get("support_tier") for r in b)
        total = max(1, len(b))
        per_half[hlf] = {
            "docs": len(docs_in_half[hlf]),
            "edges": len(b),
            "score_quantiles": {f: quantiles([r.get(f, 0.0) for r in b]) for f in SCORE_FIELDS},
            "tier_proportions": {k: round(v / total, 4) for k, v in tier_counts.items()},
            "tier_counts": dict(tier_counts),
        }

    # stability deltas
    all_tiers = set(per_half["A"]["tier_proportions"]) | set(per_half["B"]["tier_proportions"])
    tier_deltas = {
        t: round(abs(per_half["A"]["tier_proportions"].get(t, 0.0)
                     - per_half["B"]["tier_proportions"].get(t, 0.0)), 4)
        for t in all_tiers
    }
    quant_deltas = {
        f: round(abs(per_half["A"]["score_quantiles"][f].get("p50", 0.0)
                     - per_half["B"]["score_quantiles"][f].get("p50", 0.0)), 4)
        for f in SCORE_FIELDS
    }
    max_tier_delta = max(tier_deltas.values()) if tier_deltas else 0.0
    max_quant_delta = max(quant_deltas.values()) if quant_deltas else 0.0

    summary = {
        "builder": "rerank_threshold_portability",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_rerank": str(RERANK),
        "caveat": "Only 1/53 docs is PDF-only (1805.03677); this is a split-half stability proxy, not a real LaTeX-vs-PDF split.",
        "total_edges": len(rows),
        "per_half": per_half,
        "tier_proportion_deltas": tier_deltas,
        "median_score_deltas": quant_deltas,
        "max_tier_proportion_delta": max_tier_delta,
        "max_median_score_delta": max_quant_delta,
        "verdict": (
            "portable" if max_tier_delta <= 0.08 and max_quant_delta <= 0.05
            else "marginal" if max_tier_delta <= 0.15 else "unstable"
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# Rerank Threshold Portability (split-half stability)",
        "",
        f"> {summary['caveat']}",
        "",
        f"**Verdict: `{summary['verdict']}`**  (max tier-proportion delta = {max_tier_delta}, max median-score delta = {max_quant_delta})",
        "",
        "| Half | docs | edges |",
        "|---|---:|---:|",
        f"| A | {per_half['A']['docs']} | {per_half['A']['edges']} |",
        f"| B | {per_half['B']['docs']} | {per_half['B']['edges']} |",
        "",
        "## Tier proportions (A vs B, delta)",
        "",
        "| tier | A | B | |delta| |",
        "|---|---:|---:|---:|",
    ]
    for t in sorted(all_tiers, key=lambda x: -tier_deltas[x]):
        a = per_half["A"]["tier_proportions"].get(t, 0.0)
        b = per_half["B"]["tier_proportions"].get(t, 0.0)
        lines.append(f"| `{t}` | {a} | {b} | {tier_deltas[t]} |")
    lines += ["", "## Median score by field (A vs B, delta)", "",
              "| field | A p50 | B p50 | |delta| |", "|---|---:|---:|---:|"]
    for f in SCORE_FIELDS:
        a = per_half["A"]["score_quantiles"][f].get("p50", 0.0)
        b = per_half["B"]["score_quantiles"][f].get("p50", 0.0)
        lines.append(f"| {f} | {a} | {b} | {quant_deltas[f]} |")
    lines += [
        "",
        "## Reading",
        "",
        "- `portable`: tier proportions and median scores barely move between two disjoint document sets, so a threshold tuned on one set transfers to unseen (PDF-only) docs.",
        "- `marginal` / `unstable`: thresholds depend on which docs you calibrated on; expect to re-check tier proportions when onboarding a new PDF batch.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n")

    latest = ROOT / "data/05_eval/rerank_threshold_portability_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(OUT_DIR.resolve())

    print(f"[ok] {OUT_DIR/'report.md'}")
    print(f"verdict={summary['verdict']} max_tier_delta={max_tier_delta} max_score_delta={max_quant_delta}")


if __name__ == "__main__":
    main()
