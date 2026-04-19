#!/usr/bin/env python3
"""Graph-guided evidence discovery for C-Pool universal queries.

Uses document graph structure (section hierarchy, reference edges, element types,
referring_paragraphs, chunk containment) to predict which corpus passages should
be evidence for each C-Pool query template, then compares with dense retrieval.

Core idea: "利用图结构，去定向发现对query有用的evidence"
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Section classification heuristics ─────────────────────────────────────────

SECTION_CATEGORY_MAP = {
    "summary": ["abstract", "introduction", "conclusion", "summary", "overview"],
    "motivation": ["introduction", "motivation", "background", "related work",
                   "problem", "preliminary", "preliminaries"],
    "method": ["method", "approach", "model", "algorithm", "framework",
               "architecture", "proposed", "system", "design", "technique",
               "formulation", "learning"],
    "results": ["result", "experiment", "evaluation", "analysis", "ablation",
                "comparison", "performance", "discussion", "empirical"],
    "cross_doc": ["related work", "comparison", "baseline", "prior work",
                  "literature", "survey", "previous"],
}


def classify_section(section_name: str) -> Set[str]:
    s = section_name.lower().strip()
    cats = set()
    for cat, keywords in SECTION_CATEGORY_MAP.items():
        if any(kw in s for kw in keywords):
            cats.add(cat)
    return cats


def _extract_section_name(context: str, label: str) -> Optional[str]:
    m = re.search(r'\\(?:sub)*section\{([^}]+)\}', context)
    if m:
        return m.group(1).strip()
    if label.startswith("sec:"):
        return label[4:].replace("_", " ")
    return label


def build_element_section_map(
    ref_graph: dict, elements_graph: dict, chunk_graph: dict,
    corpus: Dict[str, dict] = None,
) -> Dict[str, List[str]]:
    """Build element_id → [section_names] using chunk graph cross-referencing.

    Three methods (tried in order):
    1. "Figure X" / "Table X" text pattern found in a chunk → inherit chunk's sections
    2. Element's referring_paragraphs text overlap with chunk text
    3. Element's context_before text overlap with chunk text
    Also: chunk IDs get their section_titles directly.
    """
    pid_to_sections: Dict[str, Set[str]] = defaultdict(set)

    # Direct: chunk IDs get section_titles
    for doc_id, doc in chunk_graph.get("documents", {}).items():
        for cid, chunk in doc.get("nodes", {}).items():
            for st in chunk.get("section_titles", []):
                pid_to_sections[cid].add(st)

    if corpus is None:
        return {k: sorted(v) for k, v in pid_to_sections.items()}

    # For element passages: map via chunk text matching
    doc_ids = set(p["doc_id"] for p in corpus.values())
    for doc_id in doc_ids:
        cdoc = chunk_graph.get("documents", {}).get(doc_id, {})
        chunks = cdoc.get("nodes", {})
        edoc = elements_graph.get("documents", {}).get(doc_id, {})
        elems = edoc.get("elements", {})
        if not chunks:
            continue

        ordered_chunks = sorted(chunks.values(), key=lambda c: c.get("chunk_idx", 0))
        doc_pids = {pid: p for pid, p in corpus.items() if p["doc_id"] == doc_id}

        for pid in doc_pids:
            if pid in pid_to_sections:
                continue  # already a chunk
            found = set()

            # Method 1: "Figure X" text in chunk
            parts = pid.split("_")
            try:
                num = int(parts[-1])
                etype = parts[-2]
            except (ValueError, IndexError):
                num = None
                etype = None

            if num is not None and etype:
                for chunk in ordered_chunks:
                    ctext = chunk.get("text", "").lower()
                    patterns = [f"{etype} {num}", f"{etype}{num}", f"{etype}. {num}"]
                    if any(pat in ctext for pat in patterns):
                        for s in chunk.get("section_titles", []):
                            found.add(s)
                        break

            # Method 2: referring_paragraphs overlap
            if pid in elems and not found:
                refs = elems[pid].get("referring_paragraphs", [])
                for ref_text in refs[:3]:
                    snippet = ref_text[:100].lower()
                    for chunk in ordered_chunks:
                        ctext = chunk.get("text", "").lower()
                        if len(snippet) >= 30 and snippet[:50] in ctext:
                            for s in chunk.get("section_titles", []):
                                found.add(s)
                            break
                    if found:
                        break

            # Method 3: context_before overlap
            if pid in elems and not found:
                ctx = (elems[pid].get("context_before", "") or "")[:100].lower()
                if ctx and len(ctx) > 20:
                    for chunk in ordered_chunks:
                        ctext = chunk.get("text", "").lower()
                        if ctx[:50] in ctext:
                            for s in chunk.get("section_titles", []):
                                found.add(s)
                            break

            if found:
                pid_to_sections[pid] = found

    return {k: sorted(v) for k, v in pid_to_sections.items()}


def discover_evidence(
    doc_id: str,
    category: str,
    doc_passages: Dict[str, dict],
    pid_sections: Dict[str, List[str]],
    elements_graph: dict,
    ref_graph: dict,
    chunk_graph: dict,
) -> List[Tuple[str, float, List[str]]]:
    """Graph-guided evidence scoring for one doc × one C-Pool category."""
    candidates = []

    elem_doc = elements_graph.get("documents", {}).get(doc_id, {})
    elements = elem_doc.get("elements", {})
    ref_doc = ref_graph.get("documents", {}).get(doc_id, {})
    ref_edges = ref_doc.get("edges", [])
    chunk_doc = chunk_graph.get("documents", {}).get(doc_id, {})
    chunks = chunk_doc.get("nodes", {})

    # Ref in-degree
    ref_in_degree = Counter()
    for edge in ref_edges:
        ref_in_degree[edge.get("target_label", "")] += 1

    for pid, passage in doc_passages.items():
        score = 0.0
        reasons = []
        p_type = passage.get("type", "")

        # Signal 1: Section match
        sections = pid_sections.get(pid, [])
        for sec in sections:
            if category in classify_section(sec):
                score += 1.5
                reasons.append(f"section:{sec[:40]}")
                break

        # Signal 2: Element type fit
        type_scores = {
            "summary": {"figure": 0.3, "table": 0.2, "text": 0.4},
            "motivation": {"figure": 0.4, "text": 0.3, "formula": 0.1},
            "method": {"formula": 0.5, "figure": 0.4, "table": 0.2},
            "results": {"table": 0.6, "figure": 0.5, "formula": 0.1},
            "cross_doc": {"table": 0.4, "figure": 0.3},
        }
        ts = type_scores.get(category, {}).get(p_type, 0)
        if ts > 0:
            score += ts
            reasons.append(f"type:{p_type}={ts}")

        # Signal 3: Referring paragraphs (element importance in doc)
        if pid in elements:
            rc = len(elements[pid].get("referring_paragraphs", []))
            if rc >= 8:
                score += 0.6; reasons.append(f"refs:{rc}")
            elif rc >= 4:
                score += 0.3; reasons.append(f"refs:{rc}")
            elif rc >= 1:
                score += 0.1; reasons.append(f"refs:{rc}")

        # Signal 4: Positional (overview figs/tables)
        try:
            elem_num = int(pid.split("_")[-1])
        except (ValueError, IndexError):
            elem_num = 999

        if category == "summary" and p_type in ("figure", "table") and elem_num <= 2:
            score += 0.4; reasons.append(f"overview:{p_type}_{elem_num}")
        elif category == "motivation" and p_type == "figure" and elem_num <= 3:
            score += 0.3; reasons.append(f"early:{elem_num}")

        # Signal 5: Ref in-degree
        if pid in elements:
            label = elements[pid].get("label", "")
            indeg = ref_in_degree.get(label, 0)
            if indeg >= 3:
                score += 0.4; reasons.append(f"indeg:{indeg}")
            elif indeg >= 1:
                score += 0.15; reasons.append(f"indeg:{indeg}")

        # Signal 6: Chunk section-level
        if pid in chunks:
            sl = chunks[pid].get("section_level", 99)
            if category in ("summary", "motivation") and sl <= 1:
                score += 0.2; reasons.append(f"top_sec:{sl}")

        if score > 0:
            candidates.append((pid, score, reasons))

    candidates.sort(key=lambda x: -x[1])
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpool", default="data/03_queries/c_pool_universal_queries.json")
    ap.add_argument("--corpus", default="data/05_eval/dense_retrieval/augmented_v2/corpus_v1_enriched.jsonl")
    ap.add_argument("--ranking", default="data/05_eval/cpool_retrieval/ranking_cpool_v1_enriched_qwen06b.jsonl")
    ap.add_argument("--chunk-graph", default="data/01_graphs/chunk_virtual_nodes_v2.json")
    ap.add_argument("--elements-graph", default="data/01_graphs/multimodal_elements.json")
    ap.add_argument("--ref-graph", default="data/01_graphs/latex_reference_graph.json")
    ap.add_argument("--output", default="data/05_eval/cpool_retrieval/graph_evidence_analysis.json")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    print("Loading data...")
    cpool = json.load(open(args.cpool))["queries"]

    corpus = {}
    for line in open(args.corpus):
        d = json.loads(line)
        corpus[d["passage_id"]] = d

    chunk_graph = json.load(open(args.chunk_graph))
    elements_graph = json.load(open(args.elements_graph))
    ref_graph = json.load(open(args.ref_graph))

    # Load ranking if available
    rankings = {}
    rp = Path(args.ranking)
    if rp.is_file():
        for line in open(rp):
            d = json.loads(line)
            tk = d["top_k"]
            if tk and isinstance(tk[0], list):
                tk = [x[0] for x in tk]
            rankings[d["query_id"]] = tk
        print(f"Rankings: {len(rankings)} queries")

    # Build section map
    print("Building element→section map...")
    pid_sections = build_element_section_map(ref_graph, elements_graph, chunk_graph, corpus)
    corpus_with_sec = sum(1 for pid in corpus if pid in pid_sections)
    print(f"  Corpus passages with section info: {corpus_with_sec}/{len(corpus)} ({corpus_with_sec/len(corpus):.1%})")

    # Group corpus by doc
    by_doc = defaultdict(dict)
    for pid, p in corpus.items():
        by_doc[p["doc_id"]][pid] = p
    doc_ids = sorted(by_doc.keys())
    print(f"  Docs: {len(doc_ids)}, C-Pool queries: {len(cpool)}")

    # Run
    all_results = []
    cat_stats = defaultdict(lambda: {
        "n": 0, "graph_preds": 0, "retr_top": 0,
        "overlap": 0, "graph_only": 0, "retr_only": 0,
        "graph_types": Counter(), "overlap_types": Counter(),
    })

    for q in cpool:
        qid = q["id"]
        cat = q["category"]
        retr_pids = rankings.get(qid, [])[:args.top_k]

        # Graph predictions across all docs
        graph_preds = []
        for doc_id in doc_ids:
            preds = discover_evidence(
                doc_id, cat, by_doc[doc_id],
                pid_sections, elements_graph, ref_graph, chunk_graph
            )
            for pid, score, reasons in preds[:5]:
                graph_preds.append((pid, score, reasons))

        graph_preds.sort(key=lambda x: -x[1])
        graph_set = set(p[0] for p in graph_preds[:args.top_k])
        retr_set = set(retr_pids)

        overlap = graph_set & retr_set
        graph_only = graph_set - retr_set
        retr_only = retr_set - graph_set

        s = cat_stats[cat]
        s["n"] += 1
        s["graph_preds"] += len(graph_set)
        s["retr_top"] += len(retr_set)
        s["overlap"] += len(overlap)
        s["graph_only"] += len(graph_only)
        s["retr_only"] += len(retr_only)
        for pid in graph_set:
            s["graph_types"][corpus[pid]["type"]] += 1
        for pid in overlap:
            s["overlap_types"][corpus[pid]["type"]] += 1

        all_results.append({
            "query_id": qid, "query": q["query"], "category": cat,
            "graph_top5": [
                {"pid": p[0], "score": round(p[1], 2), "reasons": p[2],
                 "type": corpus[p[0]]["type"], "doc_id": corpus[p[0]]["doc_id"],
                 "text": corpus[p[0]].get("text", "")[:150]}
                for p in graph_preds[:5]
            ],
            "retrieval_top5": [
                {"pid": pid, "rank": i+1, "type": corpus.get(pid, {}).get("type", "?"),
                 "doc_id": corpus.get(pid, {}).get("doc_id", "?"),
                 "text": corpus.get(pid, {}).get("text", "")[:150]}
                for i, pid in enumerate(retr_pids[:5])
            ],
            "overlap_count": len(overlap),
            "overlap_pids": sorted(overlap),
        })

    # Print report
    print("\n" + "=" * 90)
    print("GRAPH-GUIDED EVIDENCE DISCOVERY vs DENSE RETRIEVAL")
    print("=" * 90)

    if rankings:
        print(f"\n{'Category':<12} {'N':>3} {'AvgGraph':>8} {'AvgRetr':>8} {'AvgOvlp':>8} {'GrOnly':>7} {'RtOnly':>7} {'Prec':>7}")
        print("-" * 70)
        for cat in ["summary", "motivation", "method", "results", "cross_doc"]:
            s = cat_stats.get(cat)
            if not s or s["n"] == 0:
                continue
            n = s["n"]
            prec = s["overlap"] / max(s["graph_preds"], 1)
            print(f"{cat:<12} {n:>3} {s['graph_preds']/n:>8.1f} {s['retr_top']/n:>8.1f} {s['overlap']/n:>8.2f} {s['graph_only']/n:>7.1f} {s['retr_only']/n:>7.1f} {prec:>7.1%}")

        to = sum(s["overlap"] for s in cat_stats.values())
        tg = sum(s["graph_preds"] for s in cat_stats.values())
        tr = sum(s["retr_top"] for s in cat_stats.values())
        print(f"\n  Graph→Retrieval precision: {to}/{tg} = {to/max(tg,1):.1%}")
        print(f"  Retrieval→Graph recall:    {to}/{tr} = {to/max(tr,1):.1%}")
    else:
        print("\n  [No ranking — graph predictions only]")

    print("\n📊 Graph prediction type distribution:")
    for cat in ["summary", "motivation", "method", "results", "cross_doc"]:
        s = cat_stats.get(cat)
        if s and s["n"]:
            print(f"  {cat}: {dict(s['graph_types'])}")

    print("\n🔍 Examples per category:")
    shown = set()
    for r in all_results:
        cat = r["category"]
        if cat in shown:
            continue
        shown.add(cat)
        print(f"\n  [{cat}] \"{r['query']}\"")
        for g in r["graph_top5"][:3]:
            print(f"    📌 {g['pid']} (score={g['score']}) {g['reasons']}")
            print(f"       {g['text'][:100]}")
        if r["retrieval_top5"]:
            print(f"  Dense top-3:")
            for rt in r["retrieval_top5"][:3]:
                print(f"    🔍 rank={rt['rank']} {rt['pid']} ({rt['type']})")
                print(f"       {rt['text'][:100]}")

    # Save
    output_data = {
        "per_query": all_results,
        "per_category": {
            cat: {
                "num_queries": s["n"],
                "avg_graph_preds": round(s["graph_preds"] / s["n"], 1),
                "avg_overlap": round(s["overlap"] / s["n"], 2) if rankings else None,
                "pred_types": dict(s["graph_types"]),
                "overlap_types": dict(s["overlap_types"]),
            }
            for cat, s in cat_stats.items() if s["n"] > 0
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(output_data, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
