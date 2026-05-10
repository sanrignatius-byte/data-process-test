#!/usr/bin/env python3
"""Phase B post-processor for CEILING_PROFILING_PLAN_20260503.

Given a full-corpus ranking (top-k = corpus size) and the qrels, compute the
true rank of every missed qrel for the 121 partial+zero query subset, attach
modality from the multimodal-elements graph, and emit decision tables T6+T7.
"""
from __future__ import annotations
import argparse, json, collections
from pathlib import Path

RANK_BUCKETS = [
    ("(100, 500]", 100, 500),
    ("(500, 2000]", 500, 2000),
    ("(2000, inf)", 2000, float("inf")),
]


def bucket_of(rank: int) -> str:
    for label, lo, hi in RANK_BUCKETS:
        if lo < rank <= hi:
            return label
    return "<=100"  # should not happen for missed qrels


def modality_of_pid(pid: str, elem_meta: dict) -> str:
    """Infer modality from pid suffix or element graph metadata."""
    p = pid.lower()
    for k in ("formula", "figure", "table"):
        if f"_{k}_" in p or p.endswith(f"_{k}"):
            return k
    # fallback: lookup in elements graph
    meta = elem_meta.get(pid)
    if meta:
        return (meta.get("type") or meta.get("element_type") or "text").lower()
    return "text"


def load_qrels(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = collections.defaultdict(set)
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("relevance", 1) >= 1:
                out[r["query_id"]].add(r["passage_id"])
    return out


def load_full_ranking(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["query_id"]] = r["top_k"]
    return out


def load_elements_meta(path: Path) -> dict[str, dict]:
    """Element-id keyed metadata (best-effort; only modality is needed)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: dict[str, dict] = {}
    # graph layout varies; flatten any leaf dicts that have an element id
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict):
                if "type" in v or "element_type" in v:
                    out[k] = v
                else:
                    for kk, vv in v.items():
                        if isinstance(vv, dict) and ("type" in vv or "element_type" in vv):
                            out[kk] = vv
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-ranking", required=True, type=Path)
    ap.add_argument("--qrels", required=True, type=Path)
    ap.add_argument("--target-qids", required=True, type=Path)
    ap.add_argument("--elements-graph", required=True, type=Path)
    ap.add_argument("--out-ranks", required=True, type=Path)
    ap.add_argument("--out-tables", required=True, type=Path)
    args = ap.parse_args()

    qrels = load_qrels(args.qrels)
    rankings = load_full_ranking(args.full_ranking)
    target_qids = json.loads(args.target_qids.read_text())
    elem_meta = load_elements_meta(args.elements_graph)

    missed: dict[str, list[dict]] = {}
    for qid in target_qids:
        gold = qrels.get(qid, set())
        rank_list = rankings.get(qid, [])
        pid_to_rank = {pid: i + 1 for i, pid in enumerate(rank_list)}
        entries = []
        for pid in sorted(gold):
            r = pid_to_rank.get(pid)
            if r is None or r > 100:
                entries.append(
                    {
                        "qrel_pid": pid,
                        "modality": modality_of_pid(pid, elem_meta),
                        "rank": r if r is not None else -1,
                    }
                )
        if entries:
            missed[qid] = entries

    args.out_ranks.parent.mkdir(parents=True, exist_ok=True)
    args.out_ranks.write_text(json.dumps(missed, indent=2))

    # ---- Aggregate T6 + T7 ----
    flat = [e for entries in missed.values() for e in entries]
    n = len(flat)

    # Treat rank == -1 (not in top-2809; should not happen but be safe) as inf
    def rank_for_bucket(r: int) -> int:
        return r if r > 0 else 10**9

    # T6: rank-bucket distribution
    bucket_counts: dict[str, int] = collections.Counter()
    for e in flat:
        bucket_counts[bucket_of(rank_for_bucket(e["rank"]))] += 1

    # T7: rank-bucket × modality cross-tab
    modalities = ["formula", "figure", "table", "text"]
    cross: dict[tuple[str, str], int] = collections.Counter()
    mod_totals: dict[str, int] = collections.Counter()
    for e in flat:
        b = bucket_of(rank_for_bucket(e["rank"]))
        m = e["modality"] if e["modality"] in modalities else "text"
        cross[(m, b)] += 1
        mod_totals[m] += 1

    # ---- Render markdown ----
    lines: list[str] = []
    lines.append("# Decision Tables — Failure Profiling 2026-05-03")
    lines.append("")
    lines.append(f"**Target queries (partial+zero)**: {len(target_qids)}")
    lines.append(f"**Total missed qrels**: {n}")
    lines.append("")
    lines.append("## T6 — Missed qrel rank-bucket distribution")
    lines.append("")
    lines.append("| Bucket | Count | % |")
    lines.append("|---|---:|---:|")
    for label, _, _ in RANK_BUCKETS:
        c = bucket_counts.get(label, 0)
        pct = 100.0 * c / n if n else 0.0
        lines.append(f"| {label} | {c} | {pct:.1f}% |")
    lines.append("")

    lines.append("## T7 — Rank-bucket × modality cross-tab")
    lines.append("")
    header = "| modality | " + " | ".join(b for b, _, _ in RANK_BUCKETS) + " | total | share |"
    sep = "|---|" + "---:|" * (len(RANK_BUCKETS) + 2)
    lines.append(header)
    lines.append(sep)
    for m in modalities:
        if mod_totals.get(m, 0) == 0:
            continue
        row = [m]
        for b, _, _ in RANK_BUCKETS:
            row.append(str(cross.get((m, b), 0)))
        tot = mod_totals[m]
        share = 100.0 * tot / n if n else 0.0
        row.append(str(tot))
        row.append(f"{share:.1f}%")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ---- Trigger variables for decision rules ----
    m_form = mod_totals.get("formula", 0) / n if n else 0.0
    m_fig = mod_totals.get("figure", 0) / n if n else 0.0
    m_tab = mod_totals.get("table", 0) / n if n else 0.0
    r_low = bucket_counts.get("(100, 500]", 0) / n if n else 0.0
    r_mid = bucket_counts.get("(500, 2000]", 0) / n if n else 0.0
    r_high = bucket_counts.get("(2000, inf)", 0) / n if n else 0.0
    form_total = mod_totals.get("formula", 0)
    form_high = (cross.get(("formula", "(2000, inf)"), 0) / form_total) if form_total else 0.0

    lines.append("## Decision-rule trigger variables")
    lines.append("")
    lines.append("| var | value |")
    lines.append("|---|---:|")
    lines.append(f"| m_form | {m_form:.3f} |")
    lines.append(f"| m_fig | {m_fig:.3f} |")
    lines.append(f"| m_tab | {m_tab:.3f} |")
    lines.append(f"| r_low | {r_low:.3f} |")
    lines.append(f"| r_mid | {r_mid:.3f} |")
    lines.append(f"| r_high | {r_high:.3f} |")
    lines.append(f"| form_high | {form_high:.3f} |")
    lines.append("")

    # ---- Apply decision rules in order ----
    if m_form >= 0.40 and form_high >= 0.50:
        rule = "R1 — math-aware encoder swap OR formula query expansion"
    elif r_low >= 0.60:
        rule = "R2 — cross-encoder rerank on dense top-500"
    elif (m_fig + m_tab) >= 0.50 and r_mid >= 0.40:
        rule = "R3 — fix corpus enrichment injection bug FIRST"
    elif r_mid >= 0.40:
        rule = "R4 — HyDE query rewriting"
    else:
        rule = "R5 — mixed signal; open three small parallel pilots"
    lines.append(f"**Matched rule**: {rule}")
    lines.append("")

    args.out_tables.write_text("\n".join(lines))
    print(f"Wrote {args.out_ranks} ({len(missed)} queries, {n} missed qrels)")
    print(f"Wrote {args.out_tables}")
    print(f"Matched rule: {rule}")


if __name__ == "__main__":
    main()
