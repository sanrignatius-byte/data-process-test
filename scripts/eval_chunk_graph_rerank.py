#!/usr/bin/env python3
"""
eval_chunk_graph_rerank.py
==========================
Graph rerank for chunk-based corpus (passage_id = chunk_id).

原有 eval_graph_topk_rerank.py 假设 passage 是 element_id，
无法处理 {doc_id}_chunk_{i} 格式的 passage。
本脚本专门针对 chunk corpus，构建两层图信号：

  Layer 1 — chunk_sequence（段落顺序）：
    chunk_i ↔ chunk_{i±1}（同文档相邻 chunk）

  Layer 2 — explicit_projected（bridge 图投影）：
    hub_candidates 里的 element pair (a, b)
    → 通过 elem→chunk 映射 → (chunk_a, chunk_b)
    = 跨段落的语义桥接信号

两种 rerank 方法（与原脚本一致）：
  static_prior:        score += w_static * log(1 + degree)
  static_plus_neighbor: 同上 + 邻居的 dense score 加权传播

Usage:
    python scripts/eval_chunk_graph_rerank.py \
        --ranking  data/05_eval/chunk_corpus_n400/ranking.jsonl \
        --corpus   data/05_eval/chunk_corpus_n400/corpus.jsonl \
        --qrels    data/05_eval/chunk_corpus_n400/qrels.jsonl \
        --chunk-graph  data/01_graphs/paragraph_chunks_n400_v2.json \
        --hub-candidates data111/hub_candidates_enriched_v3.json \
        --output-dir   data/05_eval/chunk_corpus_n400/graph_rerank
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np


# ── IO ────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── 图构建 ─────────────────────────────────────────────────────────────────────

def build_chunk_graphs(
    chunk_graph_path: Path,
    hub_candidates_path: Path,
    valid_pids: Set[str],
    allow_missing_hub_candidates: bool,
) -> Tuple[Dict[str, List[str]], Dict[str, List[Tuple[str, float]]], dict]:
    """
    返回：
      seq_adj   : {chunk_id: [neighbor_chunk_id, ...]}   (chunk_sequence 边)
      exp_adj   : {chunk_id: [(neighbor_chunk_id, weight), ...]}  (explicit 投影边)
      stats     : 统计信息
    """
    # ── 1. 加载 chunk 图，构建 chunk_sequence + elem→chunk 映射 ──
    print(f"[info] loading chunk graph: {chunk_graph_path}", flush=True)
    chunk_data = json.loads(chunk_graph_path.read_bytes())

    seq_adj: Dict[str, Set[str]] = defaultdict(set)
    elem_to_chunk: Dict[str, str] = {}     # element_id → chunk_id
    chunk_to_elems: Dict[str, List[str]] = defaultdict(list)  # chunk_id → [element_ids]

    n_seq_edges = 0
    for doc_id, doc in chunk_data.get("documents", {}).items():
        nodes = doc.get("nodes", {})
        edges = doc.get("edges", [])

        # chunk_sequence
        for edge in edges:
            if edge.get("relation") == "chunk_sequence":
                src, tgt = edge["source"], edge["target"]
                if src in valid_pids and tgt in valid_pids:
                    seq_adj[src].add(tgt)
                    seq_adj[tgt].add(src)
                    n_seq_edges += 1

            # chunk → element 映射
            elif edge.get("relation") == "chunk_contains_element":
                chunk_id = edge["source"]
                elem_id  = edge["target"]
                elem_to_chunk[elem_id] = chunk_id
                chunk_to_elems[chunk_id].append(elem_id)

    # fallback: 如果 chunk 文件没有 chunk_contains_element 边，从 element_ids 字段读
    if not elem_to_chunk:
        for doc_id, doc in chunk_data.get("documents", {}).items():
            for chunk_id, node in doc.get("nodes", {}).items():
                for eid in (node.get("element_ids") or []):
                    elem_to_chunk[eid] = chunk_id
                    chunk_to_elems[chunk_id].append(eid)

    print(f"  chunk_sequence edges (both directions): {n_seq_edges*2}")
    print(f"  elem→chunk mappings: {len(elem_to_chunk)}")

    # ── 2. 加载 hub_candidates，投影到 chunk 空间 ──
    exp_adj_raw: Dict[str, Dict[str, float]] = defaultdict(dict)
    n_elem_pairs = n_chunk_pairs = 0

    if hub_candidates_path and hub_candidates_path.exists():
        print(f"[info] loading hub candidates: {hub_candidates_path}", flush=True)
        obj = json.loads(hub_candidates_path.read_bytes())

        def add_pair(a_elem: str, b_elem: str, weight: float) -> None:
            nonlocal n_chunk_pairs
            ca = elem_to_chunk.get(a_elem)
            cb = elem_to_chunk.get(b_elem)
            if ca and cb and ca != cb and ca in valid_pids and cb in valid_pids:
                old = exp_adj_raw[ca].get(cb, 0.0)
                exp_adj_raw[ca][cb] = max(old, weight)
                exp_adj_raw[cb][ca] = max(exp_adj_raw[cb].get(ca, 0.0), weight)
                n_chunk_pairs += 1

        for pair in (obj.get("pairs") or []):
            a = str(pair.get("element_a_id", "") or "").strip()
            b = str(pair.get("element_b_id", "") or "").strip()
            if a and b and a != b:
                n_elem_pairs += 1
                add_pair(a, b, float(pair.get("quality_score", 1.0) or 1.0))

        for bridge in (obj.get("adjacent_bridge_adjacency") or []):
            left  = [str(x).strip() for x in (bridge.get("elements_i") or []) if str(x).strip()]
            right = [str(x).strip() for x in (bridge.get("elements_j") or []) if str(x).strip()]
            for a in left:
                for b in right:
                    if a != b:
                        n_elem_pairs += 1
                        add_pair(a, b, 0.6)

        print(f"  element pairs in hub: {n_elem_pairs}, projected chunk pairs: {n_chunk_pairs}")
    elif allow_missing_hub_candidates:
        print("[warn] hub candidates not found, skipping explicit layer", file=sys.stderr)
    else:
        raise FileNotFoundError(
            f"hub candidates not found: {hub_candidates_path}. "
            "Refusing to run chunk graph rerank with an empty explicit layer. "
            "Pass --allow-missing-hub-candidates only if you intentionally want seq-only ablations."
        )

    # 转成 list 格式
    seq_adj_list  = {k: list(v) for k, v in seq_adj.items()}
    exp_adj_list  = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)
                     for k, v in exp_adj_raw.items()}

    stats = {
        "seq_edges_directed":    sum(len(v) for v in seq_adj_list.values()),
        "exp_edges_directed":    sum(len(v) for v in exp_adj_list.values()),
        "elem_to_chunk_mapped":  len(elem_to_chunk),
        "pids_with_seq_edges":   len(seq_adj_list),
        "pids_with_exp_edges":   len(exp_adj_list),
    }
    return seq_adj_list, exp_adj_list, stats


def load_crossdoc_chunk_edges(
    crossdoc_path: Path,
    valid_pids: Set[str],
    weight: float = 0.3,
    use_cite_boost: bool = True,
    cite_boost_extra: float = 0.1,
) -> Tuple[Dict[str, List[Tuple[str, float]]], dict]:
    """Load chunk-type cross-doc edges from build_crossdoc_gold57.py output.

    Returns crossdoc_adj: {chunk_id: [(neighbor_chunk_id, weight), ...]}
    """
    if not crossdoc_path.exists():
        return {}, {"crossdoc_edges_loaded": 0}

    obj = json.loads(crossdoc_path.read_bytes())
    edges = obj.get("edges", [])

    crossdoc_raw: Dict[str, Dict[str, float]] = defaultdict(dict)
    n_loaded = n_chunk = n_cite = 0
    for e in edges:
        if e.get("type") != "chunk":
            continue
        src, tgt = str(e["source"]), str(e["target"])
        if src not in valid_pids or tgt not in valid_pids:
            continue
        sim = float(e.get("boosted_similarity", e.get("similarity", weight)))
        w = sim * weight
        if use_cite_boost and e.get("cite_boosted"):
            w += cite_boost_extra
            n_cite += 1
        crossdoc_raw[src][tgt] = max(crossdoc_raw[src].get(tgt, 0.0), w)
        crossdoc_raw[tgt][src] = max(crossdoc_raw[tgt].get(src, 0.0), w)
        n_chunk += 1
        n_loaded += 1

    crossdoc_adj = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)
                    for k, v in crossdoc_raw.items()}
    stats = {
        "crossdoc_edges_loaded": n_loaded,
        "crossdoc_chunk_pairs": n_chunk,
        "crossdoc_cite_boosted": n_cite,
        "crossdoc_pids_covered": len(crossdoc_adj),
    }
    print(f"[info] crossdoc chunk edges: {n_loaded} loaded, {n_chunk} chunk pairs, "
          f"{n_cite} cite-boosted, {len(crossdoc_adj)} pids covered", flush=True)
    return crossdoc_adj, stats


# ── 评估工具 ───────────────────────────────────────────────────────────────────

def compute_metrics(
    ranking: Dict[str, List[str]],
    qrels: Dict[str, Set[str]],
    ks: List[int] = (1, 5, 10, 100),
) -> Dict[str, float]:
    recalls, ndcgs, mrrs = defaultdict(list), defaultdict(list), []

    for qid, gold in qrels.items():
        if not gold:
            continue
        pids = ranking.get(qid, [])

        mrr_val = 0.0
        for k in ks:
            top = pids[:k]
            hits = sum(1 for p in top if p in gold)
            recalls[k].append(hits / len(gold))
            dcg = sum(1.0 / math.log2(i + 2) for i, p in enumerate(top) if p in gold)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
            ndcgs[k].append(dcg / idcg if idcg > 0 else 0.0)
        for i, p in enumerate(pids):
            if p in gold:
                mrr_val = 1.0 / (i + 1)
                break
        mrrs.append(mrr_val)

    out: Dict[str, float] = {"mrr": float(np.mean(mrrs)) if mrrs else 0.0}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        out[f"ndcg@{k}"]   = float(np.mean(ndcgs[k]))  if ndcgs[k]  else 0.0
    return out


def rerank_static_prior(
    ranking: Dict[str, List[str]],
    seq_adj: Dict[str, List[str]],
    exp_adj: Dict[str, List[Tuple[str, float]]],
    dense_scores: Dict[str, Dict[str, float]],
    static_weight: float,
    use_exp: bool,
    use_seq: bool,
) -> Dict[str, List[str]]:
    """static_prior: boost by log(1 + degree) of combined graph."""
    # Build combined degree
    degree: Dict[str, int] = defaultdict(int)
    if use_seq:
        for pid, nbs in seq_adj.items():
            degree[pid] += len(nbs)
    if use_exp:
        for pid, nbs in exp_adj.items():
            degree[pid] += len(nbs)

    new_ranking: Dict[str, List[str]] = {}
    for qid, pids in ranking.items():
        scores = {p: dense_scores[qid].get(p, 0.0) for p in pids}
        for p in pids:
            scores[p] += static_weight * math.log1p(degree.get(p, 0))
        new_ranking[qid] = sorted(pids, key=lambda p: scores[p], reverse=True)
    return new_ranking


def rerank_static_plus_neighbor(
    ranking: Dict[str, List[str]],
    seq_adj: Dict[str, List[str]],
    exp_adj: Dict[str, List[Tuple[str, float]]],
    dense_scores: Dict[str, Dict[str, float]],
    static_weight: float,
    neighbor_decay: float,
    boost_cap: float,
    use_exp: bool,
    use_seq: bool,
) -> Dict[str, List[str]]:
    """static_prior + neighbor propagation: neighbors in top-K boost each other.
    boost_cap 限制单个 passage 能获得的最大邻居 boost，防止高度数 hub chunk 过度膨胀。
    """
    degree: Dict[str, int] = defaultdict(int)
    if use_seq:
        for pid, nbs in seq_adj.items():
            degree[pid] += len(nbs)
    if use_exp:
        for pid, nbs in exp_adj.items():
            degree[pid] += len(nbs)

    new_ranking: Dict[str, List[str]] = {}
    for qid, pids in ranking.items():
        pid_set = set(pids)
        scores = {p: dense_scores[qid].get(p, 0.0) for p in pids}

        # static
        for p in pids:
            scores[p] += static_weight * math.log1p(degree.get(p, 0))

        # neighbor propagation with cap (within top-K only)
        delta: Dict[str, float] = defaultdict(float)
        if use_seq:
            for p in pids:
                for nb in seq_adj.get(p, []):
                    if nb in pid_set:
                        delta[nb] += neighbor_decay * scores[p]
        if use_exp:
            for p in pids:
                for nb, w in exp_adj.get(p, []):
                    if nb in pid_set:
                        delta[nb] += neighbor_decay * w * scores[p]

        for p, d in delta.items():
            scores[p] += min(d, boost_cap)  # 上限保护
        new_ranking[qid] = sorted(pids, key=lambda p: scores[p], reverse=True)
    return new_ranking


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking",         required=True, help="dense retrieval ranking.jsonl (chunk_id passages)")
    ap.add_argument("--corpus",          required=True, help="chunk corpus.jsonl")
    ap.add_argument("--qrels",           required=True, help="remapped qrels.jsonl")
    ap.add_argument("--chunk-graph",     required=True, help="paragraph_chunks_nXXX_v2.json (with element_ids)")
    ap.add_argument("--hub-candidates",  default="data111/hub_candidates_enriched_v3.json")
    ap.add_argument("--allow-missing-hub-candidates", action="store_true",
                    help="允许 hub candidates 缺失时退化成无 explicit layer 的实验")
    ap.add_argument("--output-dir",      required=True)
    ap.add_argument("--static-weight",   type=float, default=0.15)
    ap.add_argument("--neighbor-decay",  type=float, default=0.20)
    ap.add_argument("--boost-cap",       type=float, default=0.20,
                    help="单个 passage 的最大邻居 boost 上限（防止 hub chunk 过度膨胀）")
    ap.add_argument("--top-k",           type=int,   default=100)
    ap.add_argument("--crossdoc-edges",  default=None,
                    help="build_crossdoc_gold57.py 输出的 cross-doc chunk 边文件")
    ap.add_argument("--crossdoc-weight", type=float, default=0.3,
                    help="cross-doc 边的 base weight（乘以 similarity 分数）")
    ap.add_argument("--crossdoc-cite-boost", type=float, default=0.1,
                    help="BBL citation 额外 boost")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 ranking
    # eval_dense_retrieval.py 输出格式: {"query_id": ..., "top_k": [...], "scores": [...]}
    print("Loading ranking ...", flush=True)
    ranking_rows = load_jsonl(Path(args.ranking))
    dense_scores: Dict[str, Dict[str, float]] = {}
    ranking_raw: Dict[str, List[str]] = {}
    n_with_scores = 0
    for row in ranking_rows:
        qid      = str(row["query_id"])
        pid_list = row.get("top_k") or row.get("passage_ids") or []
        pid_list = [str(p) for p in pid_list][:args.top_k]
        sc_list  = row.get("scores") or []
        sc_list  = [float(s) for s in sc_list][:args.top_k]
        if sc_list and len(sc_list) == len(pid_list):
            n_with_scores += 1
        else:
            # 无 cosine score 时用倒序 rank（会在日志中警告）
            n = len(pid_list)
            sc_list = [n - i for i in range(n)]
        ranking_raw[qid]  = pid_list
        dense_scores[qid] = {p: s for p, s in zip(pid_list, sc_list)}
    if n_with_scores < len(ranking_raw):
        print(f"[warn] {len(ranking_raw) - n_with_scores}/{len(ranking_raw)} queries missing cosine scores, "
              f"falling back to rank-based scores — graph rerank results will be approximate",
              file=sys.stderr)
    print(f"  {len(ranking_raw)} queries, top-{args.top_k}, {n_with_scores} with cosine scores")

    # 2. 加载 qrels
    qrels_rows = load_jsonl(Path(args.qrels))
    qrels: Dict[str, Set[str]] = defaultdict(set)
    for row in qrels_rows:
        qrels[str(row["query_id"])].add(str(row["passage_id"]))
    print(f"  {len(qrels)} queries with qrels")

    # 3. 有效 passage 集合
    valid_pids: Set[str] = set()
    for pids in ranking_raw.values():
        valid_pids.update(pids)

    # 4. 构建图
    seq_adj, exp_adj, graph_stats = build_chunk_graphs(
        chunk_graph_path=Path(args.chunk_graph),
        hub_candidates_path=Path(args.hub_candidates),
        valid_pids=valid_pids,
        allow_missing_hub_candidates=args.allow_missing_hub_candidates,
    )
    print("Graph stats:", json.dumps(graph_stats, indent=2))

    # 4b. 加载 cross-doc chunk 边（可选）
    crossdoc_adj: Dict[str, List[Tuple[str, float]]] = {}
    crossdoc_stats: dict = {}
    if args.crossdoc_edges:
        crossdoc_adj, crossdoc_stats = load_crossdoc_chunk_edges(
            crossdoc_path=Path(args.crossdoc_edges),
            valid_pids=valid_pids,
            weight=args.crossdoc_weight,
            cite_boost_extra=args.crossdoc_cite_boost,
        )
        print("Crossdoc stats:", json.dumps(crossdoc_stats, indent=2))

    # 5. 计算各 method 的指标
    baseline_metrics = compute_metrics(ranking_raw, qrels)
    results = {"dense_baseline": {"metrics": baseline_metrics}}
    print(f"\n{'Config':<35} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'R@100':>8} {'MRR':>8}")
    print("-" * 75)

    def show(name: str, m: dict) -> None:
        print(f"{name:<35} {m.get('recall@1',0):>8.4f} {m.get('recall@5',0):>8.4f} "
              f"{m.get('recall@10',0):>8.4f} {m.get('recall@100',0):>8.4f} {m.get('mrr',0):>8.4f}")

    show("dense_baseline", baseline_metrics)

    # merge crossdoc into exp_adj if present (separate dict for ablation tracking)
    exp_adj_with_crossdoc: Dict[str, List[Tuple[str, float]]] = {}
    if crossdoc_adj:
        merged: Dict[str, Dict[str, float]] = defaultdict(dict)
        for pid, nbrs in exp_adj.items():
            for nb, w in nbrs:
                merged[pid][nb] = max(merged[pid].get(nb, 0.0), w)
        for pid, nbrs in crossdoc_adj.items():
            for nb, w in nbrs:
                merged[pid][nb] = max(merged[pid].get(nb, 0.0), w)
        exp_adj_with_crossdoc = {k: sorted(v.items(), key=lambda x: x[1], reverse=True)
                                  for k, v in merged.items()}

    configs = [
        # (name, use_exp, use_seq, method, use_crossdoc)
        ("seq_only / static_prior",            False, True,  "static",   False),
        ("seq_only / static+neighbor",         False, True,  "neighbor", False),
        ("exp_only / static_prior",            True,  False, "static",   False),
        ("exp_only / static+neighbor",         True,  False, "neighbor", False),
        ("seq+exp / static_prior",             True,  True,  "static",   False),
        ("seq+exp / static+neighbor",          True,  True,  "neighbor", False),
    ]
    if crossdoc_adj:
        configs += [
            ("crossdoc_only / static_prior",       False, False, "static",   True),
            ("exp+crossdoc / static_prior",        True,  False, "static",   True),
            ("exp+crossdoc / static+neighbor",     True,  False, "neighbor", True),
            ("seq+exp+crossdoc / static_prior",    True,  True,  "static",   True),
            ("seq+exp+crossdoc / static+neighbor", True,  True,  "neighbor", True),
        ]

    for name, use_exp, use_seq, method, use_crossdoc in configs:
        active_exp = exp_adj_with_crossdoc if use_crossdoc else exp_adj
        if method == "static":
            ranked = rerank_static_prior(
                ranking_raw, seq_adj, active_exp, dense_scores,
                args.static_weight, use_exp or use_crossdoc, use_seq)
        else:
            ranked = rerank_static_plus_neighbor(
                ranking_raw, seq_adj, active_exp, dense_scores,
                args.static_weight, args.neighbor_decay, args.boost_cap,
                use_exp or use_crossdoc, use_seq)
        m = compute_metrics(ranked, qrels)
        results[name] = {"metrics": m}
        show(name, m)

    print()

    # 6. 写结果
    report = {
        "graph_stats":   graph_stats,
        "configs":       {"static_weight": args.static_weight, "neighbor_decay": args.neighbor_decay},
        "results":       results,
    }
    report_path = out_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report → {report_path}")


if __name__ == "__main__":
    main()
