#!/usr/bin/env python3
"""Validate whether ZH query translation degrades retrieval against the EN corpus.

Schema notes (handles both):
  - delivery_v1: positive = list of {element_id, span_text, ...},
                 hard_negatives = list of {element_id, span_text, ...}
  - triplet_v2 : positive = {text, text_short, ...},
                 negatives = list of {text, ...}

Setup:
  - Corpus  = union of every triplet's positive + hard_negative passages
              (in English — corpus is never translated, per user spec).
  - Per query, gold = set of positive element_ids (or positive texts for v2).
  - BM25 retrieval with three query variants:
      EN     : original English (baseline)
      ZH     : Mandarin Chinese (cross-lingual: BM25 token overlap ≈ 0
               except for verbatim-preserved proper nouns)
      ZH→EN  : ZH round-tripped back to English (measures translation fidelity)
"""

import argparse, json, re
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|[一-鿿]")


def tokenize(text: str) -> list[str]:
    """English: lowercase word tokens. Chinese: per-char unigrams."""
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]


def passage_units(t: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (positives, negatives) as lists of (uid, text) tuples."""
    pos_field = t.get("positive")
    if isinstance(pos_field, list):  # delivery_v1
        pos = [(p.get("element_id") or f"pos_{i}", p.get("span_text", "")) for i, p in enumerate(pos_field)]
        neg = [(n.get("element_id") or f"neg_{i}", n.get("span_text", "")) for i, n in enumerate(t.get("hard_negatives", []) or [])]
    elif isinstance(pos_field, dict):  # triplet_v2
        pos = [(pos_field.get("element_id") or "pos_0", pos_field.get("text", ""))]
        neg = [(n.get("element_id") or f"neg_{i}", n.get("text", "")) for i, n in enumerate(t.get("negatives", []) or [])]
    else:
        return [], []
    pos = [(u, x) for u, x in pos if x]
    neg = [(u, x) for u, x in neg if x]
    return pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/05_eval/zh_query_retrieval_report.json")
    args = ap.parse_args()

    triplets = [json.loads(l) for l in Path(args.input).read_text().splitlines() if l.strip()]
    print(f"Loaded {len(triplets)} translated triplets")

    # ── Build corpus over unique (element_id) passages ───────────────────
    uid_to_idx: dict[str, int] = {}
    corpus_texts: list[str] = []
    corpus_uids: list[str] = []

    def _add(uid: str, text: str) -> int:
        if uid in uid_to_idx:
            return uid_to_idx[uid]
        uid_to_idx[uid] = len(corpus_texts)
        corpus_texts.append(text)
        corpus_uids.append(uid)
        return uid_to_idx[uid]

    gold_per_q: list[set[int]] = []
    for t in triplets:
        pos, neg = passage_units(t)
        gold = set()
        for uid, txt in pos:
            gold.add(_add(uid, txt))
        for uid, txt in neg:
            _add(uid, txt)
        gold_per_q.append(gold)

    print(f"Corpus: {len(corpus_texts)} unique passages")
    tokenized = [tokenize(p) for p in corpus_texts]
    bm25 = BM25Okapi(tokenized)

    def eval_variant(label: str, field: str) -> dict:
        ranks_first_gold = []
        ranks_all_gold = []
        skipped = 0
        for t, gold in zip(triplets, gold_per_q):
            q = (t.get(field) or "").strip()
            if not q or not gold:
                skipped += 1
                continue
            q_tok = tokenize(q)
            if not q_tok:
                skipped += 1
                continue
            scores = bm25.get_scores(q_tok)
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            # Rank of first positive found
            best = min((order.index(g) + 1) for g in gold if g < len(order))
            ranks_first_gold.append(best)
            # Average rank of all positives (lower = better)
            all_ranks = [order.index(g) + 1 for g in gold]
            ranks_all_gold.append(sum(all_ranks) / len(all_ranks))
        n = len(ranks_first_gold)
        if n == 0:
            return {"label": label, "n": 0}
        r1 = sum(1 for r in ranks_first_gold if r <= 1) / n
        r5 = sum(1 for r in ranks_first_gold if r <= 5) / n
        r10 = sum(1 for r in ranks_first_gold if r <= 10) / n
        r50 = sum(1 for r in ranks_first_gold if r <= 50) / n
        mrr = sum(1.0 / r for r in ranks_first_gold) / n
        median = sorted(ranks_first_gold)[n // 2]
        return {
            "label": label,
            "n": n,
            "skipped": skipped,
            "R@1": round(r1, 4),
            "R@5": round(r5, 4),
            "R@10": round(r10, 4),
            "R@50": round(r50, 4),
            "MRR": round(mrr, 4),
            "median_first_gold_rank": median,
            "mean_avg_gold_rank": round(sum(ranks_all_gold) / n, 2),
        }

    variants = [
        eval_variant("EN (original)", "query_en"),
        eval_variant("ZH (cross-lingual)", "query_zh"),
        eval_variant("ZH→EN (round-trip)", "query_zh2en"),
    ]

    report = {
        "input": args.input,
        "corpus_size": len(corpus_texts),
        "n_triplets": len(triplets),
        "variants": variants,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    cols = ["n", "R@1", "R@5", "R@10", "R@50", "MRR", "median_first_gold_rank"]
    print(f"\n{'variant':22} " + " ".join(f"{c:>12}" for c in cols))
    print("-" * 110)
    for v in variants:
        row = [v.get(c, "—") for c in cols]
        print(f"{v['label']:22} " + " ".join(f"{str(x):>12}" for x in row))
    print(f"\nReport → {out_path}")


if __name__ == "__main__":
    main()
