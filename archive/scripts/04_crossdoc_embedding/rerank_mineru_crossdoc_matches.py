#!/usr/bin/env python3
"""Utility-aware reranking for MinerU cross-doc embedding matches.

Goal:
- Keep high semantic similarity.
- Suppress hub targets.
- Improve candidate diversity for downstream multi-hop construction.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarize_top1(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    top1_scores: List[float] = []
    top1_target = Counter()
    top1_doc = Counter()
    by_type = defaultdict(list)

    for r in rows:
        ms = r.get("matches", []) or []
        if not ms:
            continue
        m = ms[0]
        s = float(m.get("score", 0.0))
        top1_scores.append(s)
        tid = str(m.get("target_element_id", ""))
        tdoc = str(m.get("target_doc_id", ""))
        top1_target[tid] += 1
        top1_doc[tdoc] += 1
        by_type[str(r.get("source_type", "unknown"))].append(s)

    n = len(top1_scores)
    if n == 0:
        return {
            "n": 0,
            "top1_mean_score": 0.0,
            "unique_top1_targets": 0,
            "top10_target_concentration": 0.0,
            "max_top1_target_indegree": 0,
            "unique_top1_docs": 0,
            "top5_doc_concentration": 0.0,
            "max_top1_doc_indegree": 0,
            "top1_mean_score_by_source_type": {},
            "top1_target_top10": [],
            "top1_doc_top10": [],
        }

    return {
        "n": n,
        "top1_mean_score": round(sum(top1_scores) / n, 6),
        "unique_top1_targets": len(top1_target),
        "top10_target_concentration": round(
            sum(c for _, c in top1_target.most_common(10)) / n, 6
        ),
        "max_top1_target_indegree": max(top1_target.values()) if top1_target else 0,
        "unique_top1_docs": len(top1_doc),
        "top5_doc_concentration": round(
            sum(c for _, c in top1_doc.most_common(5)) / n, 6
        ),
        "max_top1_doc_indegree": max(top1_doc.values()) if top1_doc else 0,
        "top1_mean_score_by_source_type": {
            t: round(sum(v) / len(v), 6) for t, v in sorted(by_type.items()) if v
        },
        "top1_target_top10": top1_target.most_common(10),
        "top1_doc_top10": top1_doc.most_common(10),
    }


def local_rerank_candidates(
    matches: List[Dict[str, Any]],
    target_freq: Counter,
    target_doc_freq: Counter,
    hub_lambda: float,
    doc_lambda: float,
    diversity_lambda: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    if not matches:
        return []

    candidates: List[Dict[str, Any]] = []
    for idx, m in enumerate(matches):
        score = float(m.get("score", 0.0))
        tid = str(m.get("target_element_id", ""))
        tdoc = str(m.get("target_doc_id", ""))
        hub_pen = hub_lambda * math.log1p(target_freq[tid])
        doc_pen = doc_lambda * math.log1p(target_doc_freq[tdoc])
        adj = score - hub_pen - doc_pen
        item = dict(m)
        item["_orig_rank"] = idx + 1
        item["_adj_score"] = adj
        candidates.append(item)

    selected: List[Dict[str, Any]] = []
    doc_sel = Counter()
    pool = candidates.copy()

    k = min(top_k, len(pool))
    while pool and len(selected) < k:
        best_i = -1
        best_s = -1e18
        for i, cand in enumerate(pool):
            tdoc = str(cand.get("target_doc_id", ""))
            s = float(cand["_adj_score"]) - diversity_lambda * float(doc_sel[tdoc])
            if s > best_s:
                best_s = s
                best_i = i
        chosen = pool.pop(best_i)
        selected.append(chosen)
        doc_sel[str(chosen.get("target_doc_id", ""))] += 1

    return selected


def rerank(
    rows: List[Dict[str, Any]],
    top_k: int,
    hub_lambda: float,
    doc_lambda: float,
    diversity_lambda: float,
    top1_target_cap: int,
    top1_doc_cap: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # popularity priors from baseline top-k pool
    target_freq = Counter()
    target_doc_freq = Counter()
    for r in rows:
        for m in (r.get("matches", []) or []):
            target_freq[str(m.get("target_element_id", ""))] += 1
            target_doc_freq[str(m.get("target_doc_id", ""))] += 1

    # local rerank per source
    local_orders: List[List[Dict[str, Any]]] = []
    for r in rows:
        local_orders.append(
            local_rerank_candidates(
                matches=r.get("matches", []) or [],
                target_freq=target_freq,
                target_doc_freq=target_doc_freq,
                hub_lambda=hub_lambda,
                doc_lambda=doc_lambda,
                diversity_lambda=diversity_lambda,
                top_k=top_k,
            )
        )

    # global caps on top1 selection
    used_target = Counter()
    used_doc = Counter()
    selected_rank_counter = Counter()
    cap_fallback_count = 0
    top1_assignments: List[Dict[str, Any]] = []

    for order in local_orders:
        pick = None
        for cand in order:
            tid = str(cand.get("target_element_id", ""))
            tdoc = str(cand.get("target_doc_id", ""))
            ok_target = used_target[tid] < top1_target_cap if top1_target_cap > 0 else True
            ok_doc = used_doc[tdoc] < top1_doc_cap if top1_doc_cap > 0 else True
            if ok_target and ok_doc:
                pick = cand
                break
        if pick is None:
            if order:
                pick = order[0]
                cap_fallback_count += 1
            else:
                top1_assignments.append({})
                continue
        tid = str(pick.get("target_element_id", ""))
        tdoc = str(pick.get("target_doc_id", ""))
        used_target[tid] += 1
        used_doc[tdoc] += 1
        selected_rank_counter[int(pick.get("_orig_rank", 0))] += 1
        top1_assignments.append(pick)

    # build output rows
    out_rows: List[Dict[str, Any]] = []
    for r, order, top1 in zip(rows, local_orders, top1_assignments):
        rec = dict(r)
        if not order:
            rec["matches"] = []
            out_rows.append(rec)
            continue

        top1_id = str(top1.get("target_element_id", ""))
        top1_doc = str(top1.get("target_doc_id", ""))

        # keep top1 + local order remainder
        rest = [
            m for m in order
            if not (str(m.get("target_element_id", "")) == top1_id and str(m.get("target_doc_id", "")) == top1_doc)
        ]
        merged = [top1] + rest
        merged = merged[:top_k]

        clean_matches: List[Dict[str, Any]] = []
        for m in merged:
            x = {k: v for k, v in m.items() if not k.startswith("_")}
            clean_matches.append(x)
        rec["matches"] = clean_matches
        out_rows.append(rec)

    meta = {
        "target_freq_unique": len(target_freq),
        "target_doc_freq_unique": len(target_doc_freq),
        "selected_top1_original_rank_distribution": {
            str(k): v for k, v in sorted(selected_rank_counter.items())
        },
        "cap_fallback_count": cap_fallback_count,
        "cap_fallback_rate": round(cap_fallback_count / len(rows), 6) if rows else 0.0,
        "final_top1_target_max_indegree": max(used_target.values()) if used_target else 0,
        "final_top1_doc_max_indegree": max(used_doc.values()) if used_doc else 0,
    }
    return out_rows, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Utility-aware rerank for cross-doc embedding matches")
    ap.add_argument("--input", required=True, help="Input matches JSONL")
    ap.add_argument("--output", required=True, help="Output reranked JSONL")
    ap.add_argument("--report", required=True, help="Output comparison report JSON")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--hub-lambda", type=float, default=0.03)
    ap.add_argument("--doc-lambda", type=float, default=0.01)
    ap.add_argument("--diversity-lambda", type=float, default=0.02)
    ap.add_argument("--top1-target-cap", type=int, default=8)
    ap.add_argument("--top1-doc-cap", type=int, default=0, help="0 disables doc cap")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    report_path = Path(args.report)

    rows = load_jsonl(in_path)
    before = summarize_top1(rows)
    out_rows, meta = rerank(
        rows=rows,
        top_k=args.top_k,
        hub_lambda=args.hub_lambda,
        doc_lambda=args.doc_lambda,
        diversity_lambda=args.diversity_lambda,
        top1_target_cap=args.top1_target_cap,
        top1_doc_cap=args.top1_doc_cap,
    )
    after = summarize_top1(out_rows)

    write_jsonl(out_path, out_rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "input": str(in_path),
        "output": str(out_path),
        "config": {
            "top_k": args.top_k,
            "hub_lambda": args.hub_lambda,
            "doc_lambda": args.doc_lambda,
            "diversity_lambda": args.diversity_lambda,
            "top1_target_cap": args.top1_target_cap,
            "top1_doc_cap": args.top1_doc_cap,
        },
        "before": before,
        "after": after,
        "delta": {
            "top1_mean_score": round(after["top1_mean_score"] - before["top1_mean_score"], 6),
            "unique_top1_targets": after["unique_top1_targets"] - before["unique_top1_targets"],
            "top10_target_concentration": round(
                after["top10_target_concentration"] - before["top10_target_concentration"], 6
            ),
            "max_top1_target_indegree": after["max_top1_target_indegree"] - before["max_top1_target_indegree"],
            "top5_doc_concentration": round(
                after["top5_doc_concentration"] - before["top5_doc_concentration"], 6
            ),
        },
        "rerank_meta": meta,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("Rerank Summary")
    print("=" * 60)
    print(f"Input:  {in_path}")
    print(f"Output: {out_path}")
    print(f"Report: {report_path}")
    print("-" * 60)
    print(f"Top1 mean score: {before['top1_mean_score']:.6f} -> {after['top1_mean_score']:.6f}")
    print(
        f"Top10 target concentration: {before['top10_target_concentration']:.6f} "
        f"-> {after['top10_target_concentration']:.6f}"
    )
    print(
        f"Unique top1 targets: {before['unique_top1_targets']} "
        f"-> {after['unique_top1_targets']}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
