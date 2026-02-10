#!/usr/bin/env python3
"""Build L2 cross-document candidate pairs from triaged L1 using an inverted index."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Regex/entity rules are intentionally lightweight for MVP speed.
METHOD_PATTERNS = [
    (r"\b(BERT|RoBERTa|GPT(?:-[0-9]+)?|Transformer|ResNet|VGG|LSTM|CNN|RNN|GNN|ViT|FAISS|BM25)\b", re.IGNORECASE),
    # Acronym-style entities remain case-sensitive to avoid matching common words.
    (r"\b([A-Z]{3,}(?:-[A-Z0-9]+)?)\b", 0),
]
DATASET_PATTERNS = [
    (r"\b(ImageNet|COCO|MNIST|CIFAR(?:-10|-100)?|SQuAD|GLUE|SuperGLUE|MS\s*MARCO|BEIR)\b", re.IGNORECASE),
]
METRIC_PATTERNS = [
    (r"\b(accuracy|f1|bleu|rouge|mrr|ndcg|auc|map|recall|precision|rmse|mae)\b", re.IGNORECASE),
]

BLACKLIST = {
    "state of the art", "future work", "deep learning", "machine learning",
    "neural network", "conclusion", "introduction", "results", "discussion",
    "proposed method", "experimental results",
    # Document structure phrases that leak through regex
    "in figure", "in table", "in section", "shown in", "seen in",
    "figure", "table", "section", "appendix", "equation",
}
STOP_ACRONYMS = {"FIG", "TABLE", "API", "PDF", "RHS", "LHS", "SEC", "EQN", "REF", "APP"}

# Generic ML entities — valid concepts but too broad to be meaningful bridges.
# Pairs linked ONLY by these are weak/random. Require at least 1 non-generic entity.
GENERIC_ENTITIES = {
    "accuracy", "parity", "fairness", "precision", "recall",
    "f1", "auc", "loss", "error", "bias", "variance",
    "training", "testing", "validation", "classification",
    "regression", "prediction", "optimization", "performance",
}
STOPWORDS = {
    "about", "across", "all", "and", "are", "between", "both", "can", "does", "each",
    "from", "have", "into", "more", "most", "only", "over", "same", "than", "their",
    "these", "those", "this", "using", "with", "without", "which", "approach", "method",
}

# Drop bridge entities that are too common across docs.
MAX_DOC_FRACTION = 0.35

VISUAL_WORDS = {
    "red", "blue", "green", "black", "gray", "grey", "orange", "purple", "yellow",
    "dashed", "dotted", "solid", "curve", "line", "bar", "point", "scatter", "cluster",
    "peak", "valley", "slope", "trend", "axis", "subplot", "panel", "node", "edge", "arrow",
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def extract_entities(text: str) -> Set[str]:
    entities: Set[str] = set()
    for pattern, flags in METHOD_PATTERNS + DATASET_PATTERNS + METRIC_PATTERNS:
        for m in re.findall(pattern, text, flags=flags):
            ent = m if isinstance(m, str) else m[0]
            e = norm(ent)
            if len(e) < 3 or e in BLACKLIST or e in STOPWORDS:
                continue
            if e.upper() in STOP_ACRONYMS:
                continue
            entities.add(e)
    return entities


def load_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def visual_score(entry: Dict[str, Any]) -> int:
    words = set(re.findall(r"\b\w+\b", entry.get("visual_anchor", "").lower()))
    return len(words & VISUAL_WORDS)


def score_pair(pair: Dict[str, Any]) -> float:
    score = 0.0

    # Shared entity richness — only count non-generic.
    for ent in pair["shared_entities"]:
        if ent in GENERIC_ENTITIES:
            score += 0.5  # weak signal
        else:
            score += 3.0  # strong signal
        if len(ent) > 6:
            score += 0.5
        if " " in ent:
            score += 1.5  # multi-word = more specific

    # Visual evidence quality.
    score += pair.get("doc_a_visual_score", 0) * 0.5
    score += pair.get("doc_b_visual_score", 0) * 0.5

    # Encourage figure diversity.
    if pair.get("doc_a_figure_type") != pair.get("doc_b_figure_type"):
        score += 2.0

    return max(score, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build L2 candidate doc pairs")
    ap.add_argument("--input", default="data/l1_triage_v3.jsonl")
    ap.add_argument("--output", default="data/l2_candidate_pairs_v1.json")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--min-class", default="A", choices=["A", "B", "C"])
    # Backward-compatible alias used by another branch.
    ap.add_argument("--min-grade", choices=["A", "B"], help=argparse.SUPPRESS)
    args = ap.parse_args()

    min_class = args.min_class
    if args.min_grade:
        min_class = args.min_grade

    rows = load_records(Path(args.input))

    # Compat: support both triage_class (current) and triage (alt branch).
    class_rank = {"A": 3, "B": 2, "C": 1}
    min_rank = class_rank[min_class]
    rows = [
        r for r in rows
        if class_rank.get(r.get("triage_class", r.get("triage", "C")), 1) >= min_rank
    ]

    ent_docs: Dict[str, Set[str]] = defaultdict(set)
    ent_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    doc_entries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in rows:
        doc = r.get("doc_id")
        if not doc:
            continue
        doc_entries[doc].append(r)

        text = " ".join([
            r.get("query", ""), r.get("text_evidence", ""), r.get("answer", ""),
            r.get("visual_anchor", ""), r.get("caption", ""),
        ])
        for e in extract_entities(text):
            ent_docs[e].add(doc)
            if len(ent_examples[e]) < 3:
                ent_examples[e].append({
                    "doc_id": doc,
                    "query_id": r.get("query_id"),
                    "query": r.get("query", ""),
                    "visual_anchor": r.get("visual_anchor", ""),
                    "text_evidence": r.get("text_evidence", ""),
                })

    total_docs = len({r.get("doc_id") for r in rows if r.get("doc_id")})
    max_doc_count = max(2, int(total_docs * MAX_DOC_FRACTION))

    pair_entity_counter: Counter = Counter()
    pair_entities: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    dropped_common = 0

    for ent, docs in ent_docs.items():
        if len(docs) < 2:
            continue
        if len(docs) > max_doc_count:
            dropped_common += 1
            continue
        for a, b in combinations(sorted(docs), 2):
            pair = (a, b)
            pair_entity_counter[pair] += 1
            pair_entities[pair].add(ent)

    raw_ranked = pair_entity_counter.most_common(max(args.topk * 5, args.topk))

    pairs = []
    for (doc_a, doc_b), _ in raw_ranked:
        shared = sorted(pair_entities[(doc_a, doc_b)])
        if not shared:
            continue

        # Require at least 1 non-generic entity for meaningful bridging.
        specific = [e for e in shared if e not in GENERIC_ENTITIES]
        if not specific:
            continue

        best_a = max(doc_entries[doc_a], key=visual_score) if doc_entries[doc_a] else {}
        best_b = max(doc_entries[doc_b], key=visual_score) if doc_entries[doc_b] else {}

        pair = {
            "doc_a": doc_a,
            "doc_b": doc_b,
            "shared_entity_count": len(shared),
            "shared_entities": shared[:20],
            # Rich metadata from best L1 entry per doc (needed by generate_l2_queries.py)
            "doc_a_query_id": best_a.get("query_id", ""),
            "doc_a_query": best_a.get("query", ""),
            "doc_a_answer": best_a.get("answer", ""),
            "doc_a_visual_anchor": best_a.get("visual_anchor", ""),
            "doc_a_text_evidence": best_a.get("text_evidence", ""),
            "doc_a_figure_id": best_a.get("figure_id", ""),
            "doc_a_figure_type": best_a.get("figure_type", "unknown"),
            "doc_a_image_path": best_a.get("image_path", ""),
            "doc_a_caption": best_a.get("caption", ""),
            "doc_a_visual_score": visual_score(best_a) if best_a else 0,
            "doc_b_query_id": best_b.get("query_id", ""),
            "doc_b_query": best_b.get("query", ""),
            "doc_b_answer": best_b.get("answer", ""),
            "doc_b_visual_anchor": best_b.get("visual_anchor", ""),
            "doc_b_text_evidence": best_b.get("text_evidence", ""),
            "doc_b_figure_id": best_b.get("figure_id", ""),
            "doc_b_figure_type": best_b.get("figure_type", "unknown"),
            "doc_b_image_path": best_b.get("image_path", ""),
            "doc_b_caption": best_b.get("caption", ""),
            "doc_b_visual_score": visual_score(best_b) if best_b else 0,
        }
        pair["score"] = score_pair(pair)
        pairs.append(pair)

    pairs.sort(key=lambda p: p["score"], reverse=True)
    pairs = pairs[: args.topk]

    output = {
        "meta": {
            "input": args.input,
            "total_rows": len(rows),
            "topk": args.topk,
            "min_class": min_class,
            "num_pairs": len(pairs),
            "total_docs": total_docs,
            "dropped_common_entities": dropped_common,
            "max_doc_count": max_doc_count,
        },
        "pairs": pairs,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(output['pairs'])} candidate pairs to {out}")


if __name__ == "__main__":
    main()