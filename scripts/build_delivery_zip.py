#!/usr/bin/env python3
"""
Build complete M4query training dataset for contrastive learning & embedding model training.

Output structure:
  M4query_v1/
    train_triplets.jsonl       – query + positive passages + hard negatives
    queries.jsonl              – full query metadata
    corpus.jsonl               – all document passages (retrieval corpus)
    qrels.jsonl                – query-passage relevance labels
    README.md / stats.json
    documents/<doc_id>/mineru/ – MinerU parsed results
    documents/<doc_id>/latex/  – LaTeX source files
    graphs/                    – filtered document graphs
    candidates/                – enriched pair candidates
"""

import json, os, sys, shutil, random, collections
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
QUERY_DIR = ROOT / "data" / "03_queries"
DELIVERY_JSONL = QUERY_DIR / "delivery_v1_2026-04-13.jsonl"
PACK_DIR = ROOT / "data" / "03_queries" / "M4query_v1"
ZIP_OUT = ROOT / "data" / "03_queries" / "M4query_delivery_v1"

MINERU_BASE = ROOT / "data" / "00_raw" / "mineru_output"
LATEX_BASE = ROOT / "data" / "01_graphs" / "latex_sections_rebuild_2026-03-24" / "extracted"

GRAPH_FILES = {
    "data/01_graphs/pruned_graph_v2.json":        "pruned_graph.json",
    "data/01_graphs/hub_scores_v2.json":          "hub_scores.json",
    "data/01_graphs/multimodal_elements_v2.json": "multimodal_elements.json",
}
ENRICHED_FILES = {
    "data/02_enriched/hub_candidates_enriched_v4_intra_doc.json": "hub_candidates_intra_doc.json",
    "data/02_enriched/m2_diverse_candidates_intra_doc.json":      "m2_diverse_candidates_intra_doc.json",
}

random.seed(42)


def load_doc_elements(doc_ids):
    """Load all elements per doc from multimodal_elements + mineru structure."""
    elem_path = ROOT / "data" / "01_graphs" / "multimodal_elements_v2.json"
    with open(elem_path) as f:
        raw = json.load(f)

    doc_elements = {}
    docs_data = raw.get("documents", {})
    for did in doc_ids:
        elems = []
        if did in docs_data:
            doc_entry = docs_data[did]
            raw_elems = doc_entry.get("elements", doc_entry.get("nodes", []))
            if isinstance(raw_elems, dict):
                raw_elems = list(raw_elems.values())
            elems.extend(raw_elems)

        struct_path = MINERU_BASE / did / "structure.json"
        if struct_path.exists():
            with open(struct_path) as f:
                struct = json.load(f)
            seen_ids = {e.get("element_id", "") for e in elems}
            for me in struct.get("elements", []):
                eid = me.get("element_id", "")
                if eid and eid not in seen_ids:
                    elems.append(me)
                    seen_ids.add(eid)

        doc_elements[did] = elems
    return doc_elements


def element_to_passage(elem):
    """Convert a raw element dict to a text passage string for embedding."""
    etype = elem.get("type", "text")
    content = elem.get("content", elem.get("text", ""))
    caption = ""
    if isinstance(elem.get("metadata"), dict):
        caption = elem["metadata"].get("caption", "")

    if etype in ("figure", "image", "table"):
        parts = []
        if caption:
            parts.append(f"[{etype.upper()}] {caption}")
        if content and content != caption:
            parts.append(content)
        return " ".join(parts) if parts else f"[{etype.upper()}]"
    elif etype == "formula":
        return f"[FORMULA] {content}" if content else "[FORMULA]"
    else:
        return content or ""


def build_positive_passages(query):
    """Extract positive passages from a query's evidence fields."""
    passages = []
    text_ev = query.get("text_evidence", "")
    if text_ev and len(text_ev.strip()) > 20:
        passages.append(text_ev.strip())

    va = query.get("visual_anchors", [])
    if isinstance(va, list):
        for anchor in va:
            if isinstance(anchor, dict):
                desc = anchor.get("anchor", "")
                eid = anchor.get("element_id", "")
                if desc:
                    passages.append(f"[VISUAL: {eid}] {desc}")
            elif isinstance(anchor, str) and anchor:
                passages.append(anchor)

    spans = query.get("required_evidence_spans", [])
    if isinstance(spans, list):
        for span in spans:
            if isinstance(span, dict):
                s = span.get("span", "")
                if s and len(s) > 20:
                    passages.append(s)
    return passages


def build_negative_passages(query, doc_elements, all_doc_ids, n_intra=3, n_cross=2):
    """Build hard negative passages."""
    neg_passages = []
    neg_eids = []
    doc_id = query.get("doc_id", "")
    pos_eids = set(query.get("element_ids", []))

    if doc_id in doc_elements:
        pool = doc_elements[doc_id]
        candidates = []
        for e in pool:
            eid = e.get("element_id", e.get("id", ""))
            if eid not in pos_eids:
                passage = element_to_passage(e)
                if len(passage) > 30:
                    candidates.append((eid, passage))
        sampled = random.sample(candidates, min(n_intra, len(candidates)))
        for eid, passage in sampled:
            neg_passages.append(passage)
            neg_eids.append(eid)

    other_docs = [d for d in all_doc_ids if d != doc_id and d in doc_elements]
    if other_docs:
        cross_sampled = random.sample(other_docs, min(n_cross, len(other_docs)))
        for rand_doc in cross_sampled:
            pool = doc_elements[rand_doc]
            viable = [(e.get("element_id", e.get("id", "")), element_to_passage(e))
                      for e in pool if len(element_to_passage(e)) > 30]
            if viable:
                eid, passage = random.choice(viable)
                neg_passages.append(passage)
                neg_eids.append(eid)
    return neg_passages, neg_eids


def build_corpus(doc_elements):
    """Build a flat corpus of all passages."""
    corpus = []
    seen = set()
    for did, elems in sorted(doc_elements.items()):
        for e in elems:
            eid = e.get("element_id", e.get("id", ""))
            if not eid or eid in seen:
                continue
            seen.add(eid)
            passage = element_to_passage(e)
            if len(passage) < 10:
                continue
            corpus.append({
                "passage_id": eid,
                "doc_id": did,
                "type": e.get("type", "text"),
                "text": passage,
                "image_path": e.get("image_path", None),
            })
    return corpus


def filter_graph_by_docs(src_path, doc_ids):
    with open(src_path) as f:
        raw = json.load(f)
    # Dict-keyed by document (pruned_graph, multimodal_elements)
    if "documents" in raw and isinstance(raw["documents"], dict):
        raw["documents"] = {k: v for k, v in raw["documents"].items() if k in doc_ids}
        if "metadata" in raw and isinstance(raw["metadata"], dict):
            raw["metadata"]["papers_processed"] = len(raw["documents"])
            raw["metadata"]["filtered_for_delivery"] = True
    # List-keyed with doc_id field (hub_scores)
    for list_key in ("all_hubs", "top_hubs"):
        if list_key in raw and isinstance(raw[list_key], list):
            raw[list_key] = [h for h in raw[list_key] if h.get("doc_id") in doc_ids]
    return raw


def main():
    print("=" * 60)
    print("M4query Complete Training Dataset Builder")
    print("=" * 60)

    if PACK_DIR.exists():
        shutil.rmtree(PACK_DIR)
    PACK_DIR.mkdir(parents=True)

    # 1. Load queries
    print("\n[1/7] Loading delivery queries...")
    queries = []
    doc_ids = set()
    with open(DELIVERY_JSONL) as f:
        for line in f:
            q = json.loads(line.strip())
            queries.append(q)
            doc_ids.add(q.get("doc_id", ""))
    print(f"  {len(queries)} queries, {len(doc_ids)} docs")

    # 2. Load document elements
    print("\n[2/7] Loading document elements...")
    doc_elements = load_doc_elements(doc_ids)
    total_elems = sum(len(v) for v in doc_elements.values())
    print(f"  {total_elems} elements across {len(doc_elements)} docs")

    # 3. Build corpus
    print("\n[3/7] Building passage corpus...")
    corpus = build_corpus(doc_elements)
    corpus_out = PACK_DIR / "corpus.jsonl"
    with open(corpus_out, "w") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"  {len(corpus)} passages -> {corpus_out.name}")

    # 4. Build triplets + qrels
    print("\n[4/7] Building training triplets & qrels...")
    all_doc_list = list(doc_ids)
    triplets = []
    qrels = []

    for q in queries:
        pos_passages = build_positive_passages(q)
        neg_passages, neg_eids = build_negative_passages(
            q, doc_elements, all_doc_list, n_intra=3, n_cross=2
        )
        img_paths = [
            p.replace("data/mineru_output/", "documents/")
            for p in q.get("image_paths", [])
        ]
        triplet = {
            "query": q["query"],
            "pos": pos_passages,
            "neg": neg_passages,
            "query_id": q["query_id"],
            "doc_id": q["doc_id"],
            "pos_element_ids": q.get("element_ids", []),
            "neg_element_ids": neg_eids,
            "pos_image_paths": img_paths,
            "hop_distance": q.get("hop_distance", 0),
            "query_type": q.get("pair_type", ""),
            "query_style": q.get("query_style", ""),
        }
        triplets.append(triplet)

        for eid in q.get("element_ids", []):
            qrels.append({"query_id": q["query_id"], "passage_id": eid, "relevance": 1})
        for eid in neg_eids:
            qrels.append({"query_id": q["query_id"], "passage_id": eid, "relevance": 0})

    triplets_out = PACK_DIR / "train_triplets.jsonl"
    with open(triplets_out, "w") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    avg_pos = sum(len(t["pos"]) for t in triplets) / len(triplets)
    avg_neg = sum(len(t["neg"]) for t in triplets) / len(triplets)
    print(f"  {len(triplets)} triplets (avg {avg_pos:.1f} pos, {avg_neg:.1f} neg)")

    qrels_out = PACK_DIR / "qrels.jsonl"
    with open(qrels_out, "w") as f:
        for qr in qrels:
            f.write(json.dumps(qr, ensure_ascii=False) + "\n")
    print(f"  {len(qrels)} qrels")

    # Write full queries
    queries_out = PACK_DIR / "queries.jsonl"
    with open(queries_out, "w") as f:
        for q in queries:
            if "image_paths" in q:
                q["image_paths"] = [
                    p.replace("data/mineru_output/", "documents/")
                    for p in q["image_paths"]
                ]
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"  {len(queries)} queries with full metadata")

    # 5. Copy raw document data: MinerU + LaTeX
    print("\n[5/7] Copying document data (MinerU + LaTeX)...")
    docs_dir = PACK_DIR / "documents"
    copied_mineru = 0
    copied_latex = 0
    for did in sorted(doc_ids):
        doc_out = docs_dir / did
        mineru_src = MINERU_BASE / did
        if mineru_src.is_dir():
            shutil.copytree(mineru_src, doc_out / "mineru")
            copied_mineru += 1
        latex_src = LATEX_BASE / did
        if latex_src.is_dir():
            shutil.copytree(latex_src, doc_out / "latex")
            copied_latex += 1
    print(f"  MinerU: {copied_mineru}/{len(doc_ids)} docs")
    print(f"  LaTeX:  {copied_latex}/{len(doc_ids)} docs")

    # 6. Filter graph files
    print("\n[6/7] Filtering graph files...")
    graphs_dir = PACK_DIR / "graphs"
    graphs_dir.mkdir()
    for src_rel, dst_name in GRAPH_FILES.items():
        src_path = ROOT / src_rel
        if src_path.exists():
            filtered = filter_graph_by_docs(src_path, doc_ids)
            dst_path = graphs_dir / dst_name
            with open(dst_path, "w") as f:
                json.dump(filtered, f, ensure_ascii=False)
            src_sz = src_path.stat().st_size / 1024 / 1024
            dst_sz = dst_path.stat().st_size / 1024 / 1024
            print(f"  {dst_name}: {src_sz:.1f}MB -> {dst_sz:.1f}MB")

    cand_dir = PACK_DIR / "candidates"
    cand_dir.mkdir()
    for src_rel, dst_name in ENRICHED_FILES.items():
        src_path = ROOT / src_rel
        if src_path.exists():
            shutil.copy2(src_path, cand_dir / dst_name)
            print(f"  {dst_name}: {src_path.stat().st_size/1024/1024:.1f}MB")

    # 7. Stats + README
    print("\n[7/7] Writing stats & README...")
    hop_dist = collections.Counter()
    style_dist = collections.Counter()
    ptype_dist = collections.Counter()
    doc_dist = collections.Counter()
    for q in queries:
        hop_dist[q.get("hop_distance", "?")] += 1
        style_dist[q.get("query_style", "?")] += 1
        ptype_dist[q.get("pair_type", "?")] += 1
        doc_dist[q.get("doc_id", "?")] += 1

    stats = {
        "version": "v1",
        "created_at": datetime.now().isoformat(),
        "total_queries": len(queries),
        "total_corpus_passages": len(corpus),
        "total_triplets": len(triplets),
        "total_qrels": len(qrels),
        "unique_documents": len(doc_ids),
        "documents_with_mineru": copied_mineru,
        "documents_with_latex": copied_latex,
        "avg_pos_per_query": round(avg_pos, 2),
        "avg_neg_per_query": round(avg_neg, 2),
        "hop_distribution": {str(k): v for k, v in sorted(hop_dist.items())},
        "style_distribution": dict(sorted(style_dist.items())),
        "pair_type_distribution": dict(sorted(ptype_dist.items())),
        "top_10_docs": {k: v for k, v in doc_dist.most_common(10)},
    }
    with open(PACK_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    with open(PACK_DIR / "README.md", "w") as f:
        f.write(f"""# M4query v1 — Multi-hop Cross-modal QA Training Dataset

## Overview

Complete training dataset for contrastive learning and embedding model fine-tuning.

- **{len(queries)}** LLM-QC-verified queries (2-5 hop, cross-modal)
- **{len(corpus)}** passage corpus entries
- **{len(doc_ids)}** source documents with full MinerU parse + LaTeX source
- **{avg_pos:.1f}** avg positive passages, **{avg_neg:.1f}** avg hard negatives per query

## Structure

```
M4query_v1/
├── train_triplets.jsonl    # query + pos[] + neg[] (embedding training)
├── queries.jsonl           # full query metadata
├── corpus.jsonl            # all document passages
├── qrels.jsonl             # relevance judgments
├── documents/<arxiv_id>/
│   ├── mineru/             # MinerU: structure.json, images/, formulas/
│   └── latex/              # LaTeX .tex source files
├── graphs/                 # document knowledge graphs (filtered)
└── candidates/             # enriched pair candidates
```

## train_triplets.jsonl schema

```json
{{
  "query": "...",
  "pos": ["positive passage 1", "positive passage 2"],
  "neg": ["intra-doc hard neg", "cross-doc neg"],
  "query_id": "l1_de_1104.3913_0073",
  "doc_id": "1104.3913",
  "pos_element_ids": ["..._formula_2", "..._figure_2"],
  "neg_element_ids": ["..._text_12", "..._fig_3"],
  "pos_image_paths": ["documents/1104.3913/mineru/..."],
  "hop_distance": 2,
  "query_type": "figure+formula",
  "query_style": "academic"
}}
```

## Negative sampling

- **Intra-doc hard neg** (3/query): same document, non-evidence elements
- **Cross-doc neg** (2/query): random elements from other documents

## Statistics

| Metric | Value |
|--------|-------|
| Queries | {len(queries)} |
| Corpus passages | {len(corpus)} |
| Documents | {len(doc_ids)} |
| Hop distribution | {dict(sorted(hop_dist.items()))} |
| Style | {dict(sorted(style_dist.items()))} |

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
""")

    # Create zip
    print("\nCreating zip archive...")
    zip_path = shutil.make_archive(str(ZIP_OUT), "zip", PACK_DIR.parent, PACK_DIR.name)
    zip_sz = os.path.getsize(zip_path) / 1024 / 1024
    print(f"\n{'='*60}")
    print(f"✅ DELIVERY COMPLETE")
    print(f"{'='*60}")
    print(f"  Zip:       {zip_path}")
    print(f"  Size:      {zip_sz:.0f} MB")
    print(f"  Queries:   {len(queries)}")
    print(f"  Corpus:    {len(corpus)} passages")
    print(f"  Triplets:  {len(triplets)}")
    print(f"  Documents: {copied_mineru} MinerU + {copied_latex} LaTeX")
    return zip_path


if __name__ == "__main__":
    main()
