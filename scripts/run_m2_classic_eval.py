#!/usr/bin/env python3
"""M2 classic retrieval evaluation (stdlib only, conflict-minimized single script).

Default inputs are all under data/m2 + data/.
Use with: python scripts/run_m2_classic_eval.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
METHODS = ["bm25", "bm25f", "lm_dirichlet", "prf", "bm25_title_boost", "oracle_proximity", "hits", "rrf"]


class Element:
    def __init__(self, eid: str, doc_id: str, fields: Dict[str, str]):
        self.eid = eid
        self.doc_id = doc_id
        self.fields = fields


def tok(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        qid = str(r.get("query_id") or "")
        q = str(r.get("query") or "")
        key = qid or f"txt::{q}"
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def load_elements(path: Path) -> List[Element]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    docs = obj.get("documents", {}) or {}
    out: List[Element] = []
    for doc_id, d in docs.items():
        for eid, e in (d.get("elements") or {}).items():
            context = " ".join([str(e.get("context_before") or ""), str(e.get("context_after") or "")]).strip()
            fields = {
                "caption": str(e.get("caption") or ""),
                "content": str(e.get("content") or ""),
                "enriched_title": str(e.get("enriched_title") or ""),
                "enriched_content": str(e.get("enriched_content") or ""),
                "context": context,
            }
            if any(v.strip() for v in fields.values()):
                out.append(Element(str(eid), str(doc_id), fields))
    return out


class Index:
    def __init__(self, elems: List[Element]):
        self.elems = elems
        self.N = len(elems)
        self.field_names = ["caption", "content", "enriched_title", "enriched_content", "context"]

        self.tokens = []
        self.tf = []
        self.dl = []
        self.df = defaultdict(int)
        self.cf = defaultdict(int)

        self.field_tokens: Dict[str, List[List[str]]] = {k: [] for k in self.field_names}

        for e in elems:
            full = "\n".join(e.fields.values())
            t = tok(full)
            self.tokens.append(t)
            tf = Counter(t)
            self.tf.append(tf)
            self.dl.append(len(t))
            for x, c in tf.items():
                self.cf[x] += c
            for x in tf.keys():
                self.df[x] += 1
            for k in self.field_names:
                self.field_tokens[k].append(tok(e.fields.get(k, "")))

        self.avgdl = sum(self.dl) / max(1, self.N)
        self.field_avgdl = {k: (sum(len(v) for v in vv) / max(1, self.N)) for k, vv in self.field_tokens.items()}
        self.total_terms = sum(self.dl)

    def idf(self, term: str) -> float:
        d = self.df.get(term, 0)
        return math.log(1.0 + (self.N - d + 0.5) / (d + 0.5))


def _norm_eid(eid: str) -> str:
    """Normalize element ID format differences (e.g. fig→figure, tbl→table)."""
    eid = eid.replace("_fig_", "_figure_").replace("_tbl_", "_table_").replace("_eq_", "_formula_")
    return eid


def gt_ids(q: Dict[str, Any]) -> Set[str]:
    out = set()
    for s in (q.get("required_evidence_spans") or []):
        if isinstance(s, dict):
            eid = str(s.get("element_id") or "").strip()
            if eid:
                out.add(_norm_eid(eid))
    for eid in (q.get("element_ids") or []):
        eid = str(eid).strip()
        if eid:
            out.add(_norm_eid(eid))
    return out


def gt_gains(q: Dict[str, Any]) -> Dict[str, float]:
    g = {}
    for s in (q.get("required_evidence_spans") or []):
        if not isinstance(s, dict):
            continue
        eid = _norm_eid(str(s.get("element_id") or "").strip())
        et = str(s.get("evidence_type") or "").lower().strip()
        if not eid:
            continue
        gain = 2.0 if et in {"observation", "result"} else 1.0
        g[eid] = max(g.get(eid, 0.0), gain)
    return g


def score_bm25(ix: Index, q: List[str]) -> List[float]:
    out = [0.0] * ix.N
    qs = set(q)
    for i in range(ix.N):
        s = 0.0
        for t in qs:
            f = ix.tf[i].get(t, 0)
            if f <= 0:
                continue
            denom = f + 1.5 * (1 - 0.75 + 0.75 * ix.dl[i] / max(ix.avgdl, 1e-9))
            s += ix.idf(t) * (f * 2.5) / max(denom, 1e-9)
        out[i] = s
    return out


def score_bm25f(ix: Index, q: List[str]) -> List[float]:
    w = {"caption": 3.0, "enriched_content": 2.0, "context": 0.5, "enriched_title": 2.0, "content": 1.0}
    out = [0.0] * ix.N
    qs = set(q)
    for i in range(ix.N):
        s = 0.0
        for f in ix.field_names:
            ft = Counter(ix.field_tokens[f][i])
            dl = len(ix.field_tokens[f][i])
            avg = ix.field_avgdl[f]
            fs = 0.0
            for t in qs:
                c = ft.get(t, 0)
                if c <= 0:
                    continue
                denom = c + 1.2 * (1 - 0.75 + 0.75 * dl / max(avg, 1e-9))
                fs += ix.idf(t) * (c * 2.2) / max(denom, 1e-9)
            s += w.get(f, 1.0) * fs
        out[i] = s
    return out


def score_lm(ix: Index, q: List[str]) -> List[float]:
    out = [0.0] * ix.N
    mu = 2000.0
    for i in range(ix.N):
        s = 0.0
        for t in q:
            pbg = ix.cf.get(t, 0) / max(1, ix.total_terms)
            pbg = max(pbg, 1e-12)
            s += math.log((ix.tf[i].get(t, 0) + mu * pbg) / (ix.dl[i] + mu))
        out[i] = s
    return out



def score_prf(ix: Index, q: List[str], fb_docs: int, exp_terms: int) -> List[float]:
    base = score_bm25(ix, q)
    seeds = sorted(range(ix.N), key=lambda i: base[i], reverse=True)[:fb_docs]

    agg = defaultdict(float)
    qs = set(q)
    for i in seeds:
        for t, c in ix.tf[i].items():
            if t not in qs:
                agg[t] += c * ix.idf(t)

    exp = [t for t, _ in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:exp_terms]]

    return score_bm25(ix, q + exp)


def build_graph(path: Path) -> Dict[str, Set[str]]:
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    g = defaultdict(set)
    for p in (obj.get("pairs") or []):
        a = str(p.get("element_a_id") or "").strip()
        b = str(p.get("element_b_id") or "").strip()
        if a and b:
            g[a].add(b)
            g[b].add(a)
    for ab in (obj.get("adjacent_bridge_adjacency") or []):
        left = [str(x).strip() for x in (ab.get("elements_i") or []) if str(x).strip()]
        right = [str(x).strip() for x in (ab.get("elements_j") or []) if str(x).strip()]
        for a in left:
            for b in right:
                g[a].add(b)
                g[b].add(a)
    return dict(g)


def minmax(x: Sequence[float]) -> List[float]:
    if not x:
        return []
    lo, hi = min(x), max(x)
    if hi - lo < 1e-12:
        return [0.0] * len(x)
    return [(v - lo) / (hi - lo) for v in x]



def score_hits(
    ix: Index,
    q: List[str],
    g: Dict[str, Set[str]],
    eid2i: Dict[str, int],
    seeds_n: int,
    alpha: float,
) -> List[float]:
    bm = minmax(score_bm25(ix, q))
    seeds = sorted(range(ix.N), key=lambda i: bm[i], reverse=True)[:seeds_n]

    active = set(ix.elems[i].eid for i in seeds)
    for n in list(active):
        active.update(g.get(n, set()))
    if not active:
        return bm
    hub = {n: 1.0 for n in active}
    auth = {n: 1.0 for n in active}
    for _ in range(20):
        na = {v: sum(hub.get(u, 0.0) for u in g.get(v, set()) if u in active) for v in active}
        z = math.sqrt(sum(v * v for v in na.values())) or 1.0
        na = {k: v / z for k, v in na.items()}
        nh = {v: sum(na.get(u, 0.0) for u in g.get(v, set()) if u in active) for v in active}
        z = math.sqrt(sum(v * v for v in nh.values())) or 1.0
        nh = {k: v / z for k, v in nh.items()}
        auth, hub = na, nh
    out = bm[:]
    for eid, v in auth.items():
        i = eid2i.get(eid)
        if i is not None:

            out[i] += alpha * v

    return out


def rank_ids(ix: Index, s: Sequence[float], k: int) -> List[str]:
    idx = sorted(range(len(s)), key=lambda i: s[i], reverse=True)[:k]
    return [ix.elems[i].eid for i in idx]


def recall_at(retr: List[str], gt: Set[str], k: int) -> float:
    return (len(set(retr[:k]) & gt) / len(gt)) if gt else 0.0


def rr(retr: List[str], gt: Set[str]) -> float:
    for i, eid in enumerate(retr, 1):
        if eid in gt:
            return 1.0 / i
    return 0.0


def ndcg10(retr: List[str], gains: Dict[str, float]) -> float:
    def dcg(ids):
        s = 0.0
        for i, eid in enumerate(ids[:10], 1):
            g = gains.get(eid, 0.0)
            if g > 0:
                s += g / math.log2(i + 1)
        return s
    ideal = [eid for eid, _ in sorted(gains.items(), key=lambda kv: kv[1], reverse=True)]
    z = dcg(ideal)
    return dcg(retr) / z if z > 1e-12 else 0.0



def run_method(
    method: str,
    qs: List[Dict[str, Any]],
    ix: Index,
    graph: Dict[str, Set[str]],
    top_k: int,
    prf_fb_docs: int,
    prf_exp_terms: int,
    title_match_threshold: float,
    title_boost: float,
    hits_seeds: int,
    hits_alpha: float,
    rrf_k: int,
) -> Dict[str, Any]:

    eid2i = {e.eid: i for i, e in enumerate(ix.elems)}
    sums = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0, 20: 0.0}
    mrr = cov10 = n10 = 0.0
    n = 0
    for q in qs:
        text = str(q.get("query") or "").strip()
        gt = gt_ids(q)
        if not text or not gt:
            continue
        qt = tok(text)
        bm = score_bm25(ix, qt)
        if method == "bm25":
            s = bm
        elif method == "bm25f":
            s = score_bm25f(ix, qt)
        elif method == "lm_dirichlet":
            s = score_lm(ix, qt)
        elif method == "prf":

            s = score_prf(ix, qt, fb_docs=prf_fb_docs, exp_terms=prf_exp_terms)
        elif method == "bm25_title_boost":
            qset = set(qt)
            s = bm[:]
            for i, e in enumerate(ix.elems):
                tset = set(tok(e.fields["caption"]) + tok(e.fields["enriched_title"]))
                if not qset:
                    continue
                hit_ratio = len(qset & tset) / len(qset)
                if hit_ratio >= title_match_threshold:
                    s[i] += title_boost

        elif method in ("proximity", "oracle_proximity"):
            # NOTE: This is an ORACLE method — it uses ground-truth element IDs
            # to identify relevant documents. Results are an upper-bound analysis
            # tool, NOT a fair retrieval baseline. Do not compare directly with
            # non-oracle methods without noting this caveat.
            docs = {ix.elems[eid2i[e]].doc_id for e in gt if e in eid2i}
            s = bm[:]
            for i, e in enumerate(ix.elems):
                if e.doc_id in docs:
                    s[i] += 0.35
        elif method == "hits":

            s = score_hits(ix, qt, graph, eid2i, seeds_n=hits_seeds, alpha=hits_alpha)
        elif method == "rrf":
            hits = score_hits(ix, qt, graph, eid2i, seeds_n=hits_seeds, alpha=hits_alpha)

            br = sorted(range(ix.N), key=lambda i: bm[i], reverse=True)
            hr = sorted(range(ix.N), key=lambda i: hits[i], reverse=True)
            bp = {i: r for r, i in enumerate(br, 1)}
            hp = {i: r for r, i in enumerate(hr, 1)}

            s = [1 / (rrf_k + bp[i]) + 1 / (rrf_k + hp[i]) for i in range(ix.N)]

        else:
            raise ValueError(method)

        retr = rank_ids(ix, s, top_k)
        for k in sums:
            sums[k] += recall_at(retr, gt, k)
        mrr += rr(retr, gt)
        cov10 += recall_at(retr, gt, 10)
        n10 += ndcg10(retr, gt_gains(q))
        n += 1

    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "recall@1": round(sums[1] / n, 4),
        "recall@3": round(sums[3] / n, 4),
        "recall@5": round(sums[5] / n, 4),
        "recall@10": round(sums[10] / n, 4),
        "recall@20": round(sums[20] / n, 4),
        "mrr": round(mrr / n, 4),
        "coverage@10": round(cov10 / n, 4),
        "ndcg@10": round(n10 / n, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="M2 classic retrieval eval (stdlib)")
    ap.add_argument("--q1", type=Path, default=Path("data/m2/l2_production_2026-03-26_section_enriched_pass.jsonl"))
    ap.add_argument("--q2", type=Path, default=Path("data/m2/l3_production_2026-03-26_section_enriched_pass.jsonl"))
    ap.add_argument("--q3", type=Path, default=Path("data/m2/level1_single_element.jsonl"))
    ap.add_argument("--elements", type=Path, default=Path("data/multimodal_elements.json"))
    ap.add_argument("--hub-candidates", type=Path, default=Path("data/m2/hub_candidates_enriched_keyword_boost_full_2026-03-24.json"))
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--top-k", type=int, default=20)

    ap.add_argument("--prf-fb-docs", type=int, default=5)
    ap.add_argument("--prf-exp-terms", type=int, default=8)
    ap.add_argument("--title-match-threshold", type=float, default=0.6)
    ap.add_argument("--title-boost", type=float, default=1.5)
    ap.add_argument("--hits-seeds", type=int, default=100)
    ap.add_argument("--hits-alpha", type=float, default=0.4)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--split-levels", action="store_true", help="Also evaluate q1/q2/q3 separately")
    ap.add_argument("--output", type=Path, default=Path("data/m2/classic_eval_m2.json"))
    args = ap.parse_args()

    q1_rows = load_jsonl(args.q1)
    q2_rows = load_jsonl(args.q2)
    q3_rows = load_jsonl(args.q3)
    queries = dedupe(q1_rows + q2_rows + q3_rows)

    elems = load_elements(args.elements)
    ix = Index(elems)
    graph = build_graph(args.hub_candidates)

    metrics = {}
    for m in args.methods:
        print(f"[m2-eval] {m}")

        metrics[m] = run_method(
            m,
            queries,
            ix,
            graph,
            args.top_k,
            prf_fb_docs=args.prf_fb_docs,
            prf_exp_terms=args.prf_exp_terms,
            title_match_threshold=args.title_match_threshold,
            title_boost=args.title_boost,
            hits_seeds=args.hits_seeds,
            hits_alpha=args.hits_alpha,
            rrf_k=args.rrf_k,
        )

    level_metrics: Dict[str, Dict[str, Any]] = {}
    if args.split_levels:
        for level, rows in [("l2", q1_rows), ("l3", q2_rows), ("l1", q3_rows)]:
            level_metrics[level] = {}
            for m in args.methods:
                level_metrics[level][m] = run_method(
                    m,
                    dedupe(rows),
                    ix,
                    graph,
                    args.top_k,
                    prf_fb_docs=args.prf_fb_docs,
                    prf_exp_terms=args.prf_exp_terms,
                    title_match_threshold=args.title_match_threshold,
                    title_boost=args.title_boost,
                    hits_seeds=args.hits_seeds,
                    hits_alpha=args.hits_alpha,
                    rrf_k=args.rrf_k,
                )


    report = {
        "config": {
            "q1": str(args.q1), "q2": str(args.q2), "q3": str(args.q3),
            "elements": str(args.elements), "hub_candidates": str(args.hub_candidates),
            "top_k": args.top_k, "methods": args.methods,
            "n_queries": len(queries), "n_elements": len(elems),

            "prf_fb_docs": args.prf_fb_docs,
            "prf_exp_terms": args.prf_exp_terms,
            "title_match_threshold": args.title_match_threshold,
            "title_boost": args.title_boost,
            "hits_seeds": args.hits_seeds,
            "hits_alpha": args.hits_alpha,
            "rrf_k": args.rrf_k,
            "split_levels": args.split_levels,
        },
        "metrics": metrics,
        "level_metrics": level_metrics,

    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[m2-eval] wrote: {args.output}")


if __name__ == "__main__":
    main()
