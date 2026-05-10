#!/usr/bin/env python3
"""Post-hoc fusion and modality-bias report for DPT reranker pilots."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


KS = (1, 5, 10, 100)
NON_TEXT = ("figure", "table", "formula")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rankings(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in load_jsonl(path):
        out[row["query_id"]] = row.get("top_k") or row.get("ranking") or []
    return out


def load_qrels(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for row in load_jsonl(path):
        if row.get("relevance", 1) >= 1:
            out[row["query_id"]].add(row["passage_id"])
    return out


def load_pid_types(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in load_jsonl(path):
        pid = row.get("passage_id") or row.get("pid") or row.get("id")
        out[pid] = normalize_modality(row.get("type") or row.get("modality") or "text")
    return out


def normalize_modality(value: str) -> str:
    value = (value or "text").lower()
    return value if value in NON_TEXT else "text"


def compute_metrics(rankings: dict[str, list[str]], qrels: dict[str, set[str]]) -> dict:
    recall_at = {k: [] for k in KS}
    ndcg_at = {k: [] for k in KS}
    hit_at = {k: [] for k in KS}
    rrs: list[float] = []
    for qid, gold in qrels.items():
        ranked = rankings.get(qid, [])
        rr = 0.0
        for idx, pid in enumerate(ranked, start=1):
            if pid in gold:
                rr = 1.0 / idx
                break
        rrs.append(rr)
        for k in KS:
            topk = ranked[:k]
            hits = sum(1 for pid in topk if pid in gold)
            recall_at[k].append(hits / len(gold) if gold else 0.0)
            hit_at[k].append(1.0 if hits else 0.0)
            dcg = sum(1.0 / math.log2(i + 2) for i, pid in enumerate(topk) if pid in gold)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
            ndcg_at[k].append(dcg / idcg if idcg else 0.0)

    def avg(xs: Iterable[float]) -> float:
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0

    out = {"num_queries": len(qrels), "mrr": avg(rrs)}
    for k in KS:
        out[f"recall@{k}"] = avg(recall_at[k])
        out[f"hit@{k}"] = avg(hit_at[k])
        out[f"ndcg@{k}"] = avg(ndcg_at[k])
    return out


def rrf(rank_lists: list[list[str]], k: int, max_rank: int = 1000) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in rank_lists:
        for rank, pid in enumerate(ranked[:max_rank], start=1):
            scores[pid] += 1.0 / (k + rank)
    return [pid for pid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def fuse_all(inputs: dict[str, dict[str, list[str]]], qids: list[str],
             rrf_ks: list[int]) -> dict[str, dict[str, list[str]]]:
    dense = inputs["dense"]
    graph = inputs["graph"]
    ce = inputs["ce"]
    out: dict[str, dict[str, list[str]]] = {}
    for k in rrf_ks:
        out[f"rrf_dense_ce_k{k}"] = {
            qid: rrf([dense.get(qid, []), ce.get(qid, [])], k) for qid in qids
        }
        out[f"rrf_graph_ce_k{k}"] = {
            qid: rrf([graph.get(qid, []), ce.get(qid, [])], k) for qid in qids
        }
        out[f"rrf_dense_graph_k{k}"] = {
            qid: rrf([dense.get(qid, []), graph.get(qid, [])], k) for qid in qids
        }
        out[f"rrf_dense_graph_ce_k{k}"] = {
            qid: rrf([dense.get(qid, []), graph.get(qid, []), ce.get(qid, [])], k)
            for qid in qids
        }
    return out


def modality_report(rankings: dict[str, list[str]], qrels: dict[str, set[str]],
                    pid_types: dict[str, str], top_k: int = 10) -> dict:
    top1 = Counter()
    per_mod = defaultdict(lambda: {"hit": 0, "total": 0})
    for qid, ranked in rankings.items():
        if ranked:
            top1[pid_types.get(ranked[0], "text")] += 1
    for qid, gold in qrels.items():
        top = set(rankings.get(qid, [])[:top_k])
        for pid in gold:
            mod = pid_types.get(pid, "text")
            per_mod[mod]["total"] += 1
            if pid in top:
                per_mod[mod]["hit"] += 1
    per_mod_out = {}
    for mod in ("text", "figure", "table", "formula"):
        s = per_mod.get(mod, {"hit": 0, "total": 0})
        per_mod_out[mod] = {
            "hit": s["hit"],
            "total": s["total"],
            "recall@10": (s["hit"] / s["total"]) if s["total"] else 0.0,
        }
    return {"top1_distribution": dict(top1), "per_modality_recall@10": per_mod_out}


def metric_row(name: str, metrics: dict, dense_r10: float, graph_r10: float) -> str:
    r10 = metrics["recall@10"]
    return (
        f"| {name} | {metrics['recall@1']:.4f} | {metrics['recall@5']:.4f} | "
        f"{r10:.4f} | {metrics['recall@100']:.4f} | {metrics['mrr']:.4f} | "
        f"{(r10 - dense_r10) * 100:+.1f} | {(r10 - graph_r10) * 100:+.1f} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense-ranking", required=True, type=Path)
    ap.add_argument("--graph-ranking", required=True, type=Path)
    ap.add_argument("--ce-ranking", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--qrels", required=True, type=Path)
    ap.add_argument("--rrf-ks", default="10,20,60")
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    args = ap.parse_args()

    qrels = load_qrels(args.qrels)
    qids = list(qrels)
    pid_types = load_pid_types(args.corpus)
    rankings = {
        "dense": load_rankings(args.dense_ranking),
        "graph": load_rankings(args.graph_ranking),
        "ce": load_rankings(args.ce_ranking),
    }
    rrf_ks = [int(x) for x in args.rrf_ks.split(",") if x.strip()]
    rankings.update(fuse_all(rankings, qids, rrf_ks))

    metrics = {name: compute_metrics(ranking, qrels) for name, ranking in rankings.items()}
    modality = {
        name: modality_report(ranking, qrels, pid_types)
        for name, ranking in rankings.items()
    }

    dense_r10 = metrics["dense"]["recall@10"]
    graph_r10 = metrics["graph"]["recall@10"]
    best_name = max(metrics, key=lambda n: metrics[n]["recall@10"])
    best = metrics[best_name]

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "metrics": metrics,
        "modality": modality,
        "best_by_recall@10": {"name": best_name, **best},
        "baselines": {
            "dense_recall@10": dense_r10,
            "graph_recall@10": graph_r10,
        },
    }, indent=2))

    lines = [
        "# Reranker Fusion Report",
        "",
        "## Metrics",
        "",
        "| config | R@1 | R@5 | R@10 | R@100 | MRR | ΔR@10 vs dense pp | ΔR@10 vs graph pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    preferred_order = [
        "dense", "graph", "ce",
        *[f"rrf_dense_ce_k{k}" for k in rrf_ks],
        *[f"rrf_graph_ce_k{k}" for k in rrf_ks],
        *[f"rrf_dense_graph_k{k}" for k in rrf_ks],
        *[f"rrf_dense_graph_ce_k{k}" for k in rrf_ks],
    ]
    for name in preferred_order:
        if name in metrics:
            lines.append(metric_row(name, metrics[name], dense_r10, graph_r10))

    lines.extend([
        "",
        "## Modality Guard",
        "",
        "| config | top1 text | top1 figure | top1 table | top1 formula | fig R@10 | table R@10 | formula R@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in preferred_order:
        if name not in modality:
            continue
        m = modality[name]
        top1 = Counter(m["top1_distribution"])
        pm = m["per_modality_recall@10"]
        lines.append(
            f"| {name} | {top1.get('text', 0)} | {top1.get('figure', 0)} | "
            f"{top1.get('table', 0)} | {top1.get('formula', 0)} | "
            f"{pm['figure']['recall@10']:.4f} | {pm['table']['recall@10']:.4f} | "
            f"{pm['formula']['recall@10']:.4f} |"
        )

    best_r10 = best["recall@10"]
    if best_r10 >= 0.72:
        decision = "SUCCESS: primary bar met (R@10 >= 0.72)."
    elif best_r10 > graph_r10:
        decision = "PARTIAL SUCCESS: graph ceiling broken, below primary bar."
    elif metrics["ce"]["recall@10"] < dense_r10 and modality["ce"]["top1_distribution"].get("text", 0) > 200:
        decision = "FAIL: CE-alone regressed and shows a text-skewed top1 distribution."
    else:
        decision = "FAIL: no fusion beats the graph ceiling."

    lines.extend([
        "",
        "## Decision",
        "",
        f"Best R@10 config: `{best_name}` = {best_r10:.4f}.",
        "",
        decision,
        "",
    ])
    args.out_md.write_text("\n".join(lines))
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(f"best {best_name}: R@10={best_r10:.4f} R@100={best['recall@100']:.4f}")
    print(decision)


if __name__ == "__main__":
    main()
