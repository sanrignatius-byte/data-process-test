#!/usr/bin/env python3
"""build_formula_caption_corpus.py
=================================
F-formula caption injection variant: take the existing v1_enriched corpus
and append NL context (from element.context_before, last 300 chars) to each
formula passage. Other modalities unchanged.

Mentor C5 envisioned multi-granularity enrichment; this is the cheapest version
specific to formulas — no LLM API needed (uses already-extracted mineru
context_before paragraph text).

Tests claim:C11 — if dense encoder ceiling on formula is encoder-bound, adding
NL context within the same encoder should still hit ~0.56. If R@10 lifts,
then context-augmented encoding can move the needle.

Inputs:
  data/05_eval/dense_retrieval/rebuilt_20260417/augmented/corpus_v1_enriched.jsonl
  data/03_queries/M4query_v1/graphs/multimodal_elements.json

Output:
  data/05_eval/dense_retrieval/rebuilt_20260417/augmented/corpus_v1_enriched_formula_caption.jsonl
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


def normalize_for_match(s: str) -> set[str]:
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"\b((?:[A-Za-z] ){2,}[A-Za-z])\b",
               lambda m: m.group(1).replace(" ", ""), s)
    return set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", s) if len(t) >= 1)


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/05_eval/dense_retrieval/rebuilt_20260417/augmented/corpus_v1_enriched.jsonl")
    p.add_argument("--elements", default="data/03_queries/M4query_v1/graphs/multimodal_elements.json")
    p.add_argument("--output", default="data/05_eval/dense_retrieval/rebuilt_20260417/augmented/corpus_v1_enriched_formula_caption.jsonl")
    p.add_argument("--context-chars", type=int, default=300, help="Trailing chars of context_before to inject")
    p.add_argument("--separator", default=" | Context: ")
    p.add_argument("--min-jaccard", type=float, default=0.30)
    args = p.parse_args()

    em = json.load(open(args.elements))
    # Index elements by id and by content tokens (per doc)
    elem_by_id: dict[str, dict] = {}
    elem_content_tokens_by_doc: dict[str, list[tuple[str, set]]] = defaultdict(list)
    for did, doc in em["documents"].items():
        for eid, e in doc["elements"].items():
            elem_by_id[eid] = e
            if e.get("element_type") == "formula":
                tok = normalize_for_match(e.get("content", "") or "")
                if tok:
                    elem_content_tokens_by_doc[did].append((eid, tok))

    stats = {
        "total_passages": 0,
        "formula_passages": 0,
        "matched_by_id": 0,
        "matched_by_content": 0,
        "no_match": 0,
        "no_context_to_inject": 0,
        "ctx_chars_added": 0,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out_f:
        for line in open(args.corpus):
            p_obj = json.loads(line)
            stats["total_passages"] += 1
            if p_obj.get("type") != "formula":
                out_f.write(line)
                continue
            stats["formula_passages"] += 1
            pid = p_obj["passage_id"]
            did = p_obj["doc_id"]
            # 1) try direct lookup
            elem = elem_by_id.get(pid)
            method = "id" if elem and elem.get("element_type") == "formula" else None
            if not method:
                # 2) content match within doc
                pass_tok = normalize_for_match(p_obj["text"])
                best_eid, best_j = None, 0.0
                for eid, etok in elem_content_tokens_by_doc.get(did, []):
                    j = jaccard(pass_tok, etok)
                    if j > best_j:
                        best_j, best_eid = j, eid
                if best_eid and best_j >= args.min_jaccard:
                    elem = elem_by_id[best_eid]
                    method = "content"
            if method == "id":
                stats["matched_by_id"] += 1
            elif method == "content":
                stats["matched_by_content"] += 1
            else:
                stats["no_match"] += 1
                out_f.write(line)
                continue

            ctx = (elem.get("context_before") or "").strip()
            if not ctx:
                stats["no_context_to_inject"] += 1
                out_f.write(line)
                continue
            ctx_snippet = ctx[-args.context_chars:].strip()
            ctx_snippet = re.sub(r"\s+", " ", ctx_snippet)
            new_text = p_obj["text"].rstrip() + args.separator + ctx_snippet
            stats["ctx_chars_added"] += len(ctx_snippet) + len(args.separator)
            new_obj = dict(p_obj, text=new_text, formula_caption_method=method)
            out_f.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

    print(f"Wrote {args.output}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if stats["formula_passages"]:
        avg_added = stats["ctx_chars_added"] / max(1, stats["matched_by_id"] + stats["matched_by_content"])
        print(f"  avg ctx chars per matched formula passage: {avg_added:.0f}")


if __name__ == "__main__":
    main()
