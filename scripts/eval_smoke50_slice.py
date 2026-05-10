#!/usr/bin/env python3
"""Slice existing M4query_v1 ranking files to smoke50 query_ids and compute
per-system × per-modality metrics.

Reads:
  data/03_queries/M4query_smoke50/{queries.jsonl, qrels.jsonl}
  data/03_queries/M4query_v1/graphs/multimodal_elements.json (for elem_type)
  Each ranking file's `top_k` field

Writes:
  data/05_eval/smoke50/per_system_per_modality.json
  data/05_eval/smoke50/per_system_per_modality.md
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
SMOKE_DIR = ROOT / "data/03_queries/M4query_smoke50"
ELEMENTS_FILE = ROOT / "data/03_queries/M4query_v1/graphs/multimodal_elements.json"
OUT_DIR = ROOT / "data/05_eval/smoke50"

# system_label -> ranking file
SYSTEMS = {
    "dense_v1_enriched": "data/05_eval/dense_retrieval/rebuilt_20260417/ranking_v1_enriched.jsonl",
    "graph_explicit_only_baseline": "data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_only_fixed/ranking_graph_static_plus_neighbor.jsonl",
    "dense_formula_caption": "data/05_eval/dense_retrieval/rebuilt_20260417/ranking_v1_formula_caption.jsonl",
    "graph_formula_caption": "data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_only_formula_caption/ranking_graph_static_plus_neighbor.jsonl",
    "graph_explicit_lineno": "data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_only_lineno_v1chunk/ranking_graph_static_plus_neighbor.jsonl",
    "graph_explicit_virtual_origpos": "data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_plus_virtual_origpos/ranking_graph_static_plus_neighbor.jsonl",
    "graph_explicit_virtual_lineno": "data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_plus_virtual_lineno/ranking_graph_static_plus_neighbor.jsonl",
    "graph_static_prior_baseline": "data/05_eval/dense_retrieval/rebuilt_20260417/graph_explicit_only_fixed/ranking_graph_static_prior.jsonl",
    "bge_ce": "data/05_eval/cross_encoder_rerank/bge_v2m3_top500/ranking_ce_bge_v2m3.jsonl",
    "qwen3_ce": "data/05_eval/cross_encoder_rerank/qwen3_reranker_4b_transformers_anchor_top500/ranking_ce_qwen3_tf.jsonl",
    "split_4b_text_baseline": "data/05_eval/dense_retrieval/split_modality/4B/ranking_baseline.jsonl",
    "split_vl2b_t5_baseline": "data/05_eval/dense_retrieval/split_modality_vl_t5/ranking_baseline.jsonl",
    "math_norm_dense": "data/05_eval/dense_retrieval/rebuilt_20260417/graph_math_norm/ranking_baseline.jsonl",
    "math_norm_graph": "data/05_eval/dense_retrieval/rebuilt_20260417/graph_math_norm/ranking_graph_static_plus_neighbor.jsonl",
}
KS = (1, 5, 10, 100)
MODALITIES = ("figure", "formula", "table", "all")


def load_elem_types() -> dict[str, str]:
    data = json.load(open(ELEMENTS_FILE))
    out = {}
    for doc in data["documents"].values():
        for eid, e in doc["elements"].items():
            out[eid] = e["element_type"]
    return out


def load_qrels(path: Path) -> dict[str, list[str]]:
    qrels: dict[str, list[str]] = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r.get("relevance", 1) > 0:
            qrels[r["query_id"]].append(r["passage_id"])
    return qrels


def load_ranking(path: Path, qids: set[str]) -> dict[str, list[str]]:
    """Returns {query_id: top_k passage_ids}, restricted to qids."""
    out: dict[str, list[str]] = {}
    if not path.exists():
        return out
    for line in open(path):
        r = json.loads(line)
        qid = r["query_id"]
        if qid in qids:
            out[qid] = r["top_k"]
    return out


def compute_metrics(
    rankings: dict[str, list[str]],
    qrels: dict[str, list[str]],
    elem_type: dict[str, str],
    modality: str,
) -> dict:
    """Per-modality slice: only qrels of given modality count.
    For modality='all', use all qrels."""
    total_relevant = 0
    recall_at: dict[int, int] = {k: 0 for k in KS}
    rr_sum = 0.0
    queries_evaluated = 0
    for qid, gold in qrels.items():
        if modality != "all":
            gold = [g for g in gold if elem_type.get(g) == modality]
        if not gold:
            continue
        ranked = rankings.get(qid, [])
        if not ranked:
            queries_evaluated += 1
            total_relevant += len(gold)
            continue
        gold_set = set(gold)
        # Recall@K: fraction of gold in top-K
        for k in KS:
            hits_k = sum(1 for p in ranked[:k] if p in gold_set)
            recall_at[k] += hits_k
        # MRR: 1 / first-rank-of-any-gold
        rr = 0.0
        for i, p in enumerate(ranked, 1):
            if p in gold_set:
                rr = 1.0 / i
                break
        rr_sum += rr
        queries_evaluated += 1
        total_relevant += len(gold)
    if queries_evaluated == 0 or total_relevant == 0:
        return {"queries": 0, "qrels": 0}
    return {
        "queries": queries_evaluated,
        "qrels": total_relevant,
        **{f"R@{k}": recall_at[k] / total_relevant for k in KS},
        "MRR": rr_sum / queries_evaluated,
    }


def main() -> None:
    elem_type = load_elem_types()
    qrels = load_qrels(SMOKE_DIR / "qrels.jsonl")
    qids = set(qrels.keys())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {"smoke50_size": len(qids), "systems": {}}

    for sys_name, rel_path in SYSTEMS.items():
        path = ROOT / rel_path
        ranking = load_ranking(path, qids)
        if not ranking:
            print(f"[SKIP] {sys_name}: ranking missing or empty ({rel_path})")
            results["systems"][sys_name] = {"status": "missing", "path": rel_path}
            continue
        per_modality = {}
        for m in MODALITIES:
            per_modality[m] = compute_metrics(ranking, qrels, elem_type, m)
        results["systems"][sys_name] = {
            "path": rel_path,
            "queries_in_ranking": len(ranking),
            "per_modality": per_modality,
        }
        print(f"[OK] {sys_name}: {len(ranking)}/{len(qids)} queries")

    (OUT_DIR / "per_system_per_modality.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    # Markdown table
    lines = [
        "# Smoke50 — per-system × per-modality metrics",
        "",
        f"Built from: data/03_queries/M4query_smoke50 ({len(qids)} queries)",
        "",
        "## R@10 by modality",
        "",
        "| System | figure (39 qrels) | formula (25 qrels) | table (36 qrels) | all (100 qrels) |",
        "|---|---:|---:|---:|---:|",
    ]
    for sys_name, info in results["systems"].items():
        if info.get("status") == "missing":
            lines.append(f"| {sys_name} | — | — | — | — _(missing)_ |")
            continue
        pm = info["per_modality"]
        cells = []
        for m in ("figure", "formula", "table", "all"):
            v = pm.get(m, {}).get("R@10")
            cells.append(f"{v:.4f}" if v is not None else "—")
        lines.append(f"| {sys_name} | {' | '.join(cells)} |")

    lines += ["", "## MRR by modality", "",
              "| System | figure | formula | table | all |",
              "|---|---:|---:|---:|---:|"]
    for sys_name, info in results["systems"].items():
        if info.get("status") == "missing":
            lines.append(f"| {sys_name} | — | — | — | — |")
            continue
        pm = info["per_modality"]
        cells = []
        for m in ("figure", "formula", "table", "all"):
            v = pm.get(m, {}).get("MRR")
            cells.append(f"{v:.4f}" if v is not None else "—")
        lines.append(f"| {sys_name} | {' | '.join(cells)} |")

    lines += ["", "## R@1 / R@5 / R@100 (overall, smoke50 all qrels)", "",
              "| System | R@1 | R@5 | R@10 | R@100 | MRR |",
              "|---|---:|---:|---:|---:|---:|"]
    for sys_name, info in results["systems"].items():
        if info.get("status") == "missing":
            continue
        pm = info["per_modality"].get("all", {})
        cells = [f"{pm.get(f'R@{k}', 0):.4f}" for k in KS]
        cells.append(f"{pm.get('MRR', 0):.4f}")
        lines.append(f"| {sys_name} | {' | '.join(cells)} |")

    (OUT_DIR / "per_system_per_modality.md").write_text("\n".join(lines))
    print(f"\nWrote {OUT_DIR}/per_system_per_modality.{{json,md}}")


if __name__ == "__main__":
    main()
