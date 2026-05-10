#!/usr/bin/env python3
"""Build M4query_smoke50 — modality-balanced 50-query subset of M4query_v1.

Mentor 录音 60 C6 原话："50 query × 4 类型：10 文本/10 图/10 表/10 公式"。
Reality check 2026-05-10: M4query_v1 qrel modality dist =
  figure 218 / formula 138 / table 117 / text 0
M4query_v1 has no text-evidence queries, so we sample 17 figure / 17 formula /
16 table by primary qrel modality (rank-1 qrel's element_type).

Output:
  data/03_queries/M4query_smoke50/queries.jsonl
  data/03_queries/M4query_smoke50/qrels.jsonl
  data/03_queries/M4query_smoke50/manifest.md
"""
from __future__ import annotations
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data/03_queries/M4query_v1"
OUT_DIR = ROOT / "data/03_queries/M4query_smoke50"
ELEMENTS_FILE = SRC_DIR / "graphs/multimodal_elements.json"

QUOTAS = {"figure": 17, "formula": 17, "table": 16}
SEED = 20260510


def load_elem_types() -> dict[str, str]:
    data = json.load(open(ELEMENTS_FILE))
    out = {}
    for doc in data["documents"].values():
        for eid, e in doc["elements"].items():
            out[eid] = e["element_type"]
    return out


def main() -> None:
    queries = [json.loads(l) for l in open(SRC_DIR / "queries.jsonl")]
    qrels_by_q: dict[str, list[dict]] = defaultdict(list)
    for line in open(SRC_DIR / "qrels.jsonl"):
        r = json.loads(line)
        qrels_by_q[r["query_id"]].append(r)

    elem_types = load_elem_types()

    # Bucket queries by primary qrel modality (rank-1 qrel)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for q in queries:
        rels = qrels_by_q.get(q["query_id"], [])
        if not rels:
            continue
        primary = elem_types.get(rels[0]["passage_id"], "unknown")
        buckets[primary].append(q)

    rng = random.Random(SEED)
    selected: list[dict] = []
    bucket_stats: dict[str, dict] = {}
    for modality, quota in QUOTAS.items():
        pool = buckets.get(modality, [])
        if len(pool) < quota:
            print(f"[WARN] {modality}: pool={len(pool)} < quota={quota}; taking all")
            pick = list(pool)
        else:
            pick = rng.sample(pool, quota)
        selected.extend(pick)
        bucket_stats[modality] = {
            "pool": len(pool),
            "quota": quota,
            "picked": len(pick),
            "level_dist": dict(Counter(q["query_id"].split("_")[0] for q in pick)),
        }

    selected_ids = {q["query_id"] for q in selected}

    # Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "queries.jsonl", "w") as f:
        for q in selected:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    qrel_count = 0
    with open(OUT_DIR / "qrels.jsonl", "w") as f:
        for qid in selected_ids:
            for r in qrels_by_q[qid]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                qrel_count += 1

    # Manifest
    manifest = [
        "# M4query_smoke50 manifest",
        "",
        f"Built: 2026-05-10  seed={SEED}",
        f"Source: M4query_v1 ({len(queries)} queries / {sum(len(v) for v in qrels_by_q.values())} qrels)",
        "",
        "## Sampling rule",
        "Stratified by **primary qrel modality** (rank-1 qrel's element_type).",
        "",
        "## Mentor C6 spec vs M4query_v1 reality",
        "",
        "| Spec (mentor 录音 60) | This sample | Why |",
        "|---|---|---|",
        "| 10 text | 0 | M4query_v1 has 0 text qrels; text not realizable on this dataset |",
        f"| 10 figure | {bucket_stats['figure']['picked']} | reweighted to fill text quota |",
        f"| 10 formula | {bucket_stats['formula']['picked']} | reweighted |",
        f"| 10 table | {bucket_stats['table']['picked']} | reweighted |",
        "",
        "5/3 BGE pilot top-1 modality showing 348/473 = text was reranker error,",
        "not ground truth. M4query_v1 qrels are 100% from {figure, formula, table}.",
        "",
        "## Bucket stats",
        "",
        "| Modality | Pool size | Quota | Picked | Query level dist |",
        "|---|---:|---:|---:|---|",
    ]
    for m in ("figure", "formula", "table"):
        s = bucket_stats[m]
        manifest.append(
            f"| {m} | {s['pool']} | {s['quota']} | {s['picked']} | {s['level_dist']} |"
        )
    manifest.extend([
        "",
        f"Total queries: {len(selected)}  qrels: {qrel_count}",
        "",
        "## Output files",
        f"- `{OUT_DIR.relative_to(ROOT)}/queries.jsonl`",
        f"- `{OUT_DIR.relative_to(ROOT)}/qrels.jsonl`",
    ])
    (OUT_DIR / "manifest.md").write_text("\n".join(manifest))

    print(f"\nWrote {len(selected)} queries / {qrel_count} qrels to {OUT_DIR}")
    for m in ("figure", "formula", "table"):
        s = bucket_stats[m]
        print(f"  {m}: {s['picked']}/{s['quota']} (pool={s['pool']})")


if __name__ == "__main__":
    main()
