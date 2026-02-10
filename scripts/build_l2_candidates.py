#!/usr/bin/env python3
"""
Build L2 cross-document candidate pairs from triaged L1 queries.

Strategy:
  1. From A-class L1 queries, extract named entities (methods, datasets, metrics,
     concepts) using lightweight NLP — NOT regex on raw text, but on already-clean
     L1 query/answer/text_evidence fields.
  2. Build inverted index: entity → [(doc_id, figure_id, query_id, ...)]
  3. Find cross-document pairs: entities that appear in 2+ documents.
  4. Score and rank pairs by: entity specificity, modality diversity, anchor quality.
  5. Output top-K candidate pairs for L2 generation.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

# ── Entity extraction (lightweight, no spaCy dependency) ──────────────────

# Common scientific method/model names (will also catch from text dynamically)
METHOD_PATTERNS = [
    # Specific models/methods
    r'\b(BERT|GPT|RoBERTa|XLNet|ALBERT|DistilBERT|T5|BART|LLaMA)\b',
    r'\b(ResNet|VGG|InceptionNet|EfficientNet|ViT|CLIP|DALL-E)\b',
    r'\b(Adam|SGD|AdaGrad|RMSprop)\b',
    r'\b(SVM|KNN|Random Forest|Naive Bayes|Logistic Regression)\b',
    r'\b(GAN|VAE|Autoencoder|Transformer)\b',
    r'\b(LSTM|GRU|RNN|CNN|MLP|DNN)\b',
    r'\b(PCA|t-SNE|UMAP|LDA)\b',
    r'\b(BM25|TF-IDF|FAISS)\b',
    r'\b(RAG|DPR|ANCE|ColBERT)\b',
]

DATASET_PATTERNS = [
    r'\b(SQuAD|GLUE|SuperGLUE|MNLI|SST-\d|MRPC|QQP|QNLI|RTE)\b',
    r'\b(ImageNet|CIFAR-\d+|MNIST|COCO|VOC)\b',
    r'\b(BEIR|MS MARCO|Natural Questions|TriviaQA)\b',
    r'\b(CoQA|HotpotQA|MultiRC)\b',
]

METRIC_PATTERNS = [
    r'\b(accuracy|precision|recall|F1|F-score|AUC|ROC)\b',
    r'\b(BLEU|ROUGE|METEOR|BERTScore|perplexity)\b',
    r'\b(Recall@\d+|MRR|NDCG|MAP)\b',
    r'\b(fairness|parity|equalized odds|disparate impact)\b',
]

# Broader concept extraction: capitalized multi-word phrases (likely method/concept names)
CAPITALIZED_PHRASE = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
# Acronyms (2-6 uppercase letters, possibly with digits)
ACRONYM = re.compile(r'\b([A-Z][A-Z0-9]{1,5})\b')

# Common stopword acronyms to exclude
STOP_ACRONYMS = {
    "AND", "THE", "FOR", "NOT", "BUT", "NOR", "YET", "ARE", "WAS",
    "HAS", "HAD", "CAN", "MAY", "RHS", "LHS", "IFF", "QED", "PDF",
    "API", "URL", "HTML", "JSON", "CSS", "XML", "SQL", "IDE",
    "LET", "SET", "MAP", "USE", "ALL", "ANY", "NEW", "OLD",
    "TWO", "ONE", "WE", "IT", "IN", "ON", "TO", "IS", "OR", "IF",
    "AT", "AS", "BY", "SO", "DO", "UP", "AN", "OF", "NO",
}


def extract_entities(text: str) -> set[str]:
    """Extract candidate entities from clean L1 text fields."""
    entities = set()

    # Pattern-based extraction
    for patterns in [METHOD_PATTERNS, DATASET_PATTERNS, METRIC_PATTERNS]:
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.add(match.group(0).strip())

    # Capitalized phrases (e.g., "Statistical Parity", "Affirmative Action")
    for match in CAPITALIZED_PHRASE.finditer(text):
        phrase = match.group(0).strip()
        if len(phrase) > 5 and len(phrase.split()) <= 4:
            entities.add(phrase)

    # Acronyms
    for match in ACRONYM.finditer(text):
        acr = match.group(0)
        if acr not in STOP_ACRONYMS and len(acr) >= 2:
            entities.add(acr)

    return entities


def score_pair(pair_info: dict) -> float:
    """Score a candidate cross-document pair for L2 generation potential."""
    score = 0.0

    # More shared entities = stronger link
    score += len(pair_info["shared_entities"]) * 2.0

    # Entity specificity bonus (longer/rarer entities are better bridges)
    for ent in pair_info["shared_entities"]:
        if len(ent) > 6:
            score += 1.0
        if " " in ent:  # multi-word = more specific
            score += 1.5

    # Modality diversity bonus
    fig_types = {pair_info["doc_a_figure_type"], pair_info["doc_b_figure_type"]}
    if len(fig_types) > 1:
        score += 3.0  # different figure types = richer cross-modal

    # A-class query quality bonus
    score += pair_info["doc_a_visual_score"] * 0.5
    score += pair_info["doc_b_visual_score"] * 0.5

    # Penalty for very common entities (less discriminative)
    for ent in pair_info["shared_entities"]:
        if ent.lower() in {"accuracy", "precision", "recall", "f1", "method",
                           "model", "approach", "algorithm", "baseline"}:
            score -= 1.0

    return max(score, 0.0)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build L2 cross-document candidate pairs from A-class L1 queries")
    parser.add_argument("--input", default="data/l1_triage_v3.jsonl",
                        help="Triaged L1 JSONL")
    parser.add_argument("--output", default="data/l2_candidate_pairs_v1.json",
                        help="Output candidate pairs JSON")
    parser.add_argument("--topk", type=int, default=100,
                        help="Max candidate pairs to output")
    parser.add_argument("--min-grade", default="A",
                        choices=["A", "B"],
                        help="Minimum triage grade to include")
    args = parser.parse_args()

    # Load triaged entries
    entries = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    # Filter by grade
    allowed_grades = {"A"} if args.min_grade == "A" else {"A", "B"}
    filtered = [e for e in entries if e.get("triage", "C") in allowed_grades]
    print(f"Loaded {len(entries)} entries, {len(filtered)} pass grade >= {args.min_grade}")

    # Build entity → document index
    entity_index = defaultdict(list)  # entity → [(doc_id, entry)]
    doc_entities = defaultdict(set)   # doc_id → {entities}

    for entry in filtered:
        doc_id = entry["doc_id"]
        # Extract entities from all clean text fields
        text_blob = " ".join([
            entry.get("query", ""),
            entry.get("answer", ""),
            entry.get("text_evidence", ""),
            entry.get("visual_anchor", ""),
            entry.get("caption", ""),
        ])
        ents = extract_entities(text_blob)
        for ent in ents:
            entity_index[ent].append((doc_id, entry))
            doc_entities[doc_id].add(ent)

    # Find cross-document entities (appear in 2+ docs)
    cross_doc_entities = {}
    for ent, occurrences in entity_index.items():
        doc_ids = set(doc_id for doc_id, _ in occurrences)
        if len(doc_ids) >= 2:
            cross_doc_entities[ent] = {
                "entity": ent,
                "doc_count": len(doc_ids),
                "doc_ids": sorted(doc_ids),
                "total_mentions": len(occurrences),
            }

    print(f"\nEntity index: {len(entity_index)} unique entities")
    print(f"Cross-document entities: {len(cross_doc_entities)}")
    print(f"\nTop 20 cross-doc entities:")
    for ent, info in sorted(cross_doc_entities.items(),
                            key=lambda x: x[1]["doc_count"], reverse=True)[:20]:
        print(f"  {ent:<30} {info['doc_count']} docs, {info['total_mentions']} mentions")

    # Build candidate document pairs
    # For each cross-doc entity, generate all document pairs
    doc_pair_entities = defaultdict(set)  # (doc_a, doc_b) → {shared entities}
    for ent, info in cross_doc_entities.items():
        for doc_a, doc_b in combinations(sorted(info["doc_ids"]), 2):
            doc_pair_entities[(doc_a, doc_b)].add(ent)

    print(f"\nCandidate document pairs: {len(doc_pair_entities)}")

    # For each pair, pick best representative L1 queries and score
    # Index: doc_id → [entries]
    doc_entries = defaultdict(list)
    for entry in filtered:
        doc_entries[entry["doc_id"]].append(entry)

    VISUAL_WORDS_SET = {
        "red", "blue", "green", "black", "gray", "grey", "orange", "purple",
        "yellow", "dashed", "dotted", "solid", "curve", "line", "bar",
        "top", "bottom", "left", "right", "upper", "lower", "peak", "valley",
        "spike", "slope", "trend", "axis", "subplot", "panel", "shaded",
        "highlighted", "node", "arrow", "box", "circle", "edge",
    }

    def visual_score(entry):
        anchor = entry.get("visual_anchor", "").lower()
        words = set(re.findall(r'\b\w+\b', anchor))
        return len(words & VISUAL_WORDS_SET)

    candidate_pairs = []
    for (doc_a, doc_b), shared_ents in doc_pair_entities.items():
        # Pick best query from each doc (highest visual score)
        best_a = max(doc_entries[doc_a], key=visual_score)
        best_b = max(doc_entries[doc_b], key=visual_score)

        pair_info = {
            "doc_a": doc_a,
            "doc_b": doc_b,
            "shared_entities": sorted(shared_ents),
            "shared_entity_count": len(shared_ents),
            "doc_a_query_id": best_a["query_id"],
            "doc_a_query": best_a["query"],
            "doc_a_answer": best_a["answer"],
            "doc_a_visual_anchor": best_a.get("visual_anchor", ""),
            "doc_a_text_evidence": best_a.get("text_evidence", ""),
            "doc_a_figure_id": best_a["figure_id"],
            "doc_a_figure_type": best_a.get("figure_type", "unknown"),
            "doc_a_image_path": best_a.get("image_path", ""),
            "doc_a_caption": best_a.get("caption", ""),
            "doc_a_visual_score": visual_score(best_a),
            "doc_b_query_id": best_b["query_id"],
            "doc_b_query": best_b["query"],
            "doc_b_answer": best_b["answer"],
            "doc_b_visual_anchor": best_b.get("visual_anchor", ""),
            "doc_b_text_evidence": best_b.get("text_evidence", ""),
            "doc_b_figure_id": best_b["figure_id"],
            "doc_b_figure_type": best_b.get("figure_type", "unknown"),
            "doc_b_image_path": best_b.get("image_path", ""),
            "doc_b_caption": best_b.get("caption", ""),
            "doc_b_visual_score": visual_score(best_b),
        }
        pair_info["score"] = score_pair(pair_info)
        candidate_pairs.append(pair_info)

    # Sort by score, take top-K
    candidate_pairs.sort(key=lambda x: x["score"], reverse=True)
    top_pairs = candidate_pairs[:args.topk]

    # Save
    output = {
        "metadata": {
            "total_l1_entries": len(entries),
            "filtered_entries": len(filtered),
            "unique_entities": len(entity_index),
            "cross_doc_entities": len(cross_doc_entities),
            "total_doc_pairs": len(doc_pair_entities),
            "output_pairs": len(top_pairs),
        },
        "pairs": top_pairs,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*50}")
    print(f"L2 Candidate Pairs")
    print(f"{'='*50}")
    print(f"Output: {args.output}")
    print(f"Top {len(top_pairs)} pairs (by score)")
    print()
    for i, p in enumerate(top_pairs[:10]):
        print(f"  #{i+1}: {p['doc_a']} × {p['doc_b']}")
        print(f"       shared: {', '.join(p['shared_entities'][:5])}"
              f"{'...' if len(p['shared_entities']) > 5 else ''}")
        print(f"       score: {p['score']:.1f}  "
              f"fig_types: {p['doc_a_figure_type']}/{p['doc_b_figure_type']}")
        print()


if __name__ == "__main__":
    main()
