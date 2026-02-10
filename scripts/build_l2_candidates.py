#!/usr/bin/env python3
"""Build L2 candidate doc pairs from triaged L1 using inverted index."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

METHOD_PATTERNS = [
    (r"\b(BERT|RoBERTa|GPT(?:-[0-9]+)?|Transformer|ResNet|VGG|LSTM|CNN|RNN|GNN|ViT)\b", re.IGNORECASE),
    # Acronym-style entities should remain case-sensitive to avoid matching common words.
    (r"\b([A-Z]{3,}(?:-[A-Z0-9]+)?)\b", 0),
]
DATASET_PATTERNS = [
    (r"\b(ImageNet|COCO|MNIST|CIFAR(?:-10|-100)?|SQuAD|GLUE|SuperGLUE)\b", re.IGNORECASE),
]
METRIC_PATTERNS = [
    (r"\b(accuracy|f1|bleu|rouge|mrr|ndcg|auc|map|recall|precision)\b", re.IGNORECASE),
]
BLACKLIST = {
    "state of the art", "future work", "deep learning", "machine learning",
    "neural network", "conclusion", "introduction", "results", "discussion",
}
STOP_ACRONYMS = {"FIG", "TABLE", "API", "PDF", "CNN", "RNN"}
STOPWORDS = {
    "about", "across", "all", "and", "are", "between", "both", "can", "does", "each",
    "from", "have", "into", "more", "most", "only", "over", "same", "than", "their",
    "these", "those", "this", "using", "with", "without", "which",
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def extract_entities(text: str) -> Set[str]:
    ents: Set[str] = set()
    for pat, flags in METHOD_PATTERNS + DATASET_PATTERNS + METRIC_PATTERNS:
        for m in re.findall(pat, text, flags=flags):
            ent = m if isinstance(m, str) else m[0]
            e = norm(ent)
            if len(e) < 3 or e in BLACKLIST:
                continue
            if e in STOPWORDS:
                continue
            if e.upper() in STOP_ACRONYMS:
                continue
            ents.add(e)
    return ents


def load_records(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build L2 candidate doc pairs")
    ap.add_argument("--input", default="data/l1_triage_v3.jsonl")
    ap.add_argument("--output", default="data/l2_candidate_pairs_v1.json")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--min-class", default="A", choices=["A", "B", "C"])
    args = ap.parse_args()

    rows = load_records(Path(args.input))
    class_rank = {"A": 3, "B": 2, "C": 1}
    min_rank = class_rank[args.min_class]
    rows = [r for r in rows if class_rank.get(r.get("triage_class", "C"), 1) >= min_rank]

    # entity -> docs
    ent_docs: Dict[str, Set[str]] = defaultdict(set)
    ent_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in rows:
        text = " ".join([r.get("query", ""), r.get("text_evidence", ""), r.get("answer", "")])
        entities = extract_entities(text)
        doc = r.get("doc_id")
        if not doc:
            continue
        for e in entities:
            ent_docs[e].add(doc)
            if len(ent_examples[e]) < 3:
                ent_examples[e].append({
                    "doc_id": doc,
                    "query_id": r.get("query_id"),
                    "query": r.get("query", ""),
                    "visual_anchor": r.get("visual_anchor", ""),
                    "text_evidence": r.get("text_evidence", ""),
                })

    pair_scores: Counter = Counter()
    pair_entities: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for ent, docs in ent_docs.items():
        if len(docs) < 2:
            continue
        for a, b in combinations(sorted(docs), 2):
            pair = (a, b)
            pair_scores[pair] += 1
            pair_entities[pair].add(ent)

    ranked = pair_scores.most_common(args.topk)
    output = {
        "meta": {
            "input": args.input,
            "total_rows": len(rows),
            "topk": args.topk,
            "min_class": args.min_class,
            "num_pairs": len(ranked),
        },
        "pairs": [],
    }

    for (doc_a, doc_b), score in ranked:
        ents = sorted(pair_entities[(doc_a, doc_b)])
        output["pairs"].append({
            "doc_a": doc_a,
            "doc_b": doc_b,
            "shared_entity_count": score,
            "shared_entities": ents[:20],
            "evidence_examples": [
                *[x for x in ent_examples.get(ents[0], []) if x["doc_id"] == doc_a][:1],
                *[x for x in ent_examples.get(ents[0], []) if x["doc_id"] == doc_b][:1],
            ] if ents else [],
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(output['pairs'])} candidate pairs to {out}")


if __name__ == "__main__":
    main()
