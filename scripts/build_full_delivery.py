#!/usr/bin/env python3
"""
Full delivery pipeline:
  1. Download LaTeX sources for 53 delivery docs (arXiv e-print)
  2. Build corpus.jsonl from MinerU parsed elements
  3. Build train_triplets.jsonl (query + pos + neg) for contrastive learning
  4. Build qrels.jsonl (query-passage relevance)
  5. Package everything into zip
  6. Upload to ModelScope: IgnatiusMao/M4query_test

Usage (standalone or via SLURM):
    python scripts/build_full_delivery.py
"""

import json, os, sys, shutil, random, collections, hashlib
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUERY_DIR    = ROOT / "data" / "03_queries"
DELIVERY_SRC = QUERY_DIR / "delivery_v1_2026-04-13.jsonl"
DOC_IDS_FILE = QUERY_DIR / "_delivery_doc_ids.txt"
MINERU_BASE  = ROOT / "data" / "00_raw" / "mineru_output"
LATEX_OUTPUT = ROOT / "data" / "00_raw" / "latex_sources_delivery"
GRAPH_DIR    = ROOT / "data" / "01_graphs"
ENRICHED_DIR = ROOT / "data" / "02_enriched"

PACK_NAME = "M4query_v1"
PACK_DIR  = QUERY_DIR / PACK_NAME
ZIP_OUT   = QUERY_DIR / "M4query_delivery_v1"  # .zip appended by make_archive

random.seed(42)

# ── ModelScope config ─────────────────────────────────────────────────
MS_TOKEN   = "ms-4fcf0dbb-239b-4707-82a7-18f1c64f6dcb"
MS_REPO    = "IgnatiusMao/M4query_test"


# ══════════════════════════════════════════════════════════════════════
# Step 1: Download LaTeX sources
# ══════════════════════════════════════════════════════════════════════

def step1_download_latex():
    """Download LaTeX sources for all 53 docs via existing script."""
    print("\n" + "=" * 60)
    print("STEP 1: Download LaTeX sources")
    print("=" * 60)

    # Check which docs already have latex
    doc_ids = [l.strip() for l in open(DOC_IDS_FILE) if l.strip()]

    latex_ext = LATEX_OUTPUT / "extracted"
    already = [d for d in doc_ids if (latex_ext / d).is_dir()]
    missing = [d for d in doc_ids if d not in already]
    print(f"  Already have: {len(already)}, need to download: {len(missing)}")

    if not missing:
        print("  All LaTeX sources present, skipping download.")
        return

    # Also check other latex directories and copy if available
    alt_dirs = [
        ROOT / "data" / "00_raw" / "latex_sources" / "extracted",
        ROOT / "data" / "00_raw" / "latex_sources_all",
        ROOT / "data" / "00_raw" / "latex_sources_batch2" / "extracted",
    ]
    copied = 0
    still_missing = []
    latex_ext.mkdir(parents=True, exist_ok=True)
    for did in missing:
        found = False
        for alt in alt_dirs:
            src = alt / did
            if src.is_dir():
                dst = latex_ext / did
                shutil.copytree(src, dst)
                copied += 1
                found = True
                break
        if not found:
            still_missing.append(did)

    if copied:
        print(f"  Copied {copied} from existing latex dirs")

    if still_missing:
        print(f"  Downloading {len(still_missing)} from arXiv...")
        # Write missing IDs to temp file
        tmp_ids = QUERY_DIR / "_tmp_missing_latex_ids.txt"
        with open(tmp_ids, "w") as f:
            for d in still_missing:
                f.write(d + "\n")

        # Call the download script
        import subprocess
        cmd = [
            sys.executable, str(ROOT / "scripts" / "download_latex_sources.py"),
            "--id-file", str(tmp_ids),
            "--output", str(LATEX_OUTPUT),
            "--delay", "3",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(result.stdout[-500:] if result.stdout else "")
        if result.returncode != 0:
            print(f"  WARNING: download exit code {result.returncode}")
            if result.stderr:
                print(result.stderr[-300:])
        tmp_ids.unlink(missing_ok=True)

    # Final count
    final = [d for d in doc_ids if (latex_ext / d).is_dir()]
    print(f"  Final LaTeX coverage: {len(final)}/{len(doc_ids)}")
    return final


# ══════════════════════════════════════════════════════════════════════
# Step 2: Build corpus (all passage chunks from MinerU)
# ══════════════════════════════════════════════════════════════════════

def step2_build_corpus(doc_ids):
    """
    Build corpus.jsonl: one passage per MinerU element.
    Each passage has: passage_id, doc_id, text, type, metadata.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Build corpus from MinerU elements")
    print("=" * 60)

    corpus = []
    for did in sorted(doc_ids):
        struct_path = MINERU_BASE / did / "structure.json"
        if not struct_path.exists():
            continue
        with open(struct_path) as f:
            doc = json.load(f)
        elements = doc.get("elements", [])
        for elem in elements:
            eid = elem.get("element_id", elem.get("id", ""))
            text = elem.get("content", elem.get("text", ""))
            if not text or len(text.strip()) < 10:
                continue
            passage = {
                "passage_id": f"{did}_{eid}" if eid else f"{did}_{hashlib.md5(text[:100].encode()).hexdigest()[:8]}",
                "doc_id": did,
                "text": text,
                "type": elem.get("type", "unknown"),
                "section": elem.get("section", ""),
                "page": elem.get("page_idx", elem.get("page", -1)),
            }
            # Add image path if multimodal
            img = elem.get("image_path", elem.get("img_path", ""))
            if img and img != "None":
                passage["image_path"] = img
            corpus.append(passage)

    print(f"  Total passages: {len(corpus)} from {len(doc_ids)} docs")
    return corpus


# ══════════════════════════════════════════════════════════════════════
# Step 3: Build training triplets + neg samples
# ══════════════════════════════════════════════════════════════════════

def step3_build_triplets(queries, corpus):
    """
    Build contrastive training triplets:
      { query_id, query, positive: [{passage_id, text, ...}], negative: [{passage_id, text, ...}] }
    
    Neg strategy:
      - intra-doc hard neg: same doc, not in evidence (BM25-like)
      - cross-doc neg: random passage from different doc
    """
    print("\n" + "=" * 60)
    print("STEP 3: Build training triplets")
    print("=" * 60)

    # Index corpus by doc_id
    doc_passages = collections.defaultdict(list)
    passage_lookup = {}
    for p in corpus:
        doc_passages[p["doc_id"]].append(p)
        passage_lookup[p["passage_id"]] = p

    all_doc_ids = list(doc_passages.keys())
    triplets = []
    qrels = []  # query-passage relevance pairs

    for q in queries:
        qid = q["query_id"]
        doc_id = q.get("doc_id", "")
        pos_eids = set(q.get("element_ids", []))
        text_evidence = q.get("text_evidence", "")

        # --- Positive passages ---
        positives = []
        for eid in pos_eids:
            pid = f"{doc_id}_{eid}"
            if pid in passage_lookup:
                positives.append({
                    "passage_id": pid,
                    "text": passage_lookup[pid]["text"][:1000],
                    "type": passage_lookup[pid].get("type", ""),
                })
        # If element_ids didn't match corpus, use text_evidence as synthetic positive
        if not positives and text_evidence:
            positives.append({
                "passage_id": f"{doc_id}_evidence_{hashlib.md5(text_evidence[:50].encode()).hexdigest()[:8]}",
                "text": text_evidence[:1000],
                "type": "text_evidence",
            })

        # --- Negative passages ---
        negatives = []

        # Intra-doc hard negatives (same doc, different element)
        intra_pool = [p for p in doc_passages.get(doc_id, [])
                      if p["passage_id"] not in {pp["passage_id"] for pp in positives}]
        intra_sampled = random.sample(intra_pool, min(2, len(intra_pool)))
        for p in intra_sampled:
            negatives.append({
                "passage_id": p["passage_id"],
                "text": p["text"][:1000],
                "type": p.get("type", ""),
                "neg_source": "intra_doc",
            })

        # Cross-doc negative
        other_docs = [d for d in all_doc_ids if d != doc_id]
        if other_docs:
            rand_doc = random.choice(other_docs)
            pool = doc_passages[rand_doc]
            if pool:
                p = random.choice(pool)
                negatives.append({
                    "passage_id": p["passage_id"],
                    "text": p["text"][:1000],
                    "type": p.get("type", ""),
                    "neg_source": "cross_doc",
                })

        triplet = {
            "query_id": qid,
            "query": q["query"],
            "doc_id": doc_id,
            "hop_distance": q.get("hop_distance"),
            "query_style": q.get("query_style"),
            "positive": positives,
            "negative": negatives,
        }
        triplets.append(triplet)

        # qrels: relevance labels
        for pp in positives:
            qrels.append({
                "query_id": qid,
                "passage_id": pp["passage_id"],
                "relevance": 1,
            })
        for nn in negatives:
            qrels.append({
                "query_id": qid,
                "passage_id": nn["passage_id"],
                "relevance": 0,
            })

    pos_counts = [len(t["positive"]) for t in triplets]
    neg_counts = [len(t["negative"]) for t in triplets]
    print(f"  Triplets: {len(triplets)}")
    print(f"  Avg positives/query: {sum(pos_counts)/len(pos_counts):.1f}")
    print(f"  Avg negatives/query: {sum(neg_counts)/len(neg_counts):.1f}")
    print(f"  Qrels: {len(qrels)}")

    return triplets, qrels


# ══════════════════════════════════════════════════════════════════════
# Step 4: Package everything
# ══════════════════════════════════════════════════════════════════════

def step4_package(queries, corpus, triplets, qrels, doc_ids, latex_docs):
    """
    Package structure:
      M4query_v1/
        README.md
        queries.jsonl          – full queries with QC
        corpus.jsonl           – all passages
        train_triplets.jsonl   – contrastive training data
        qrels.jsonl            – relevance labels
        stats.json
        documents/
          {doc_id}/
            mineru/            – MinerU parsed output
            latex/             – LaTeX source (if available)
        graphs/
          pruned_graph.json
          hub_scores.json
          multimodal_elements.json
        candidates/
          hub_candidates_intra_doc.json
          m2_diverse_candidates_intra_doc.json
    """
    print("\n" + "=" * 60)
    print("STEP 4: Package delivery")
    print("=" * 60)

    # Clean
    if PACK_DIR.exists():
        shutil.rmtree(PACK_DIR)
    PACK_DIR.mkdir(parents=True)

    # ── Write JSONL files ──
    def write_jsonl(path, data):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  Written: {path.name} ({len(data)} records, {path.stat().st_size/1024/1024:.1f}MB)")

    # Queries: fix image paths to be relative to package
    final_queries = []
    for q in queries:
        qc = dict(q)
        if "image_paths" in qc:
            qc["image_paths"] = [
                p.replace("data/mineru_output/", "documents/").replace("/hybrid_auto/", "/mineru/hybrid_auto/")
                for p in qc["image_paths"]
            ]
        # Remove internal provenance field
        qc.pop("_source_batch", None)
        final_queries.append(qc)

    write_jsonl(PACK_DIR / "queries.jsonl", final_queries)
    write_jsonl(PACK_DIR / "corpus.jsonl", corpus)
    write_jsonl(PACK_DIR / "train_triplets.jsonl", triplets)
    write_jsonl(PACK_DIR / "qrels.jsonl", qrels)

    # ── Copy documents ──
    print("\n  Copying document data...")
    docs_dir = PACK_DIR / "documents"
    latex_ext = LATEX_OUTPUT / "extracted"
    copied_mineru = 0
    copied_latex = 0
    for did in sorted(doc_ids):
        doc_out = docs_dir / did

        # MinerU output
        mineru_src = MINERU_BASE / did
        if mineru_src.is_dir():
            shutil.copytree(mineru_src, doc_out / "mineru")
            copied_mineru += 1

        # LaTeX source
        latex_src = latex_ext / did
        if latex_src.is_dir():
            shutil.copytree(latex_src, doc_out / "latex")
            copied_latex += 1
        else:
            # Try other latex directories
            for alt in [
                ROOT / "data" / "00_raw" / "latex_sources" / "extracted" / did,
                ROOT / "data" / "00_raw" / "latex_sources_all" / did,
                ROOT / "data" / "00_raw" / "latex_sources_batch2" / "extracted" / did,
            ]:
                if alt.is_dir():
                    shutil.copytree(alt, doc_out / "latex")
                    copied_latex += 1
                    break

    print(f"  MinerU: {copied_mineru}/{len(doc_ids)}")
    print(f"  LaTeX:  {copied_latex}/{len(doc_ids)}")

    # ── Filter & copy graphs ──
    print("\n  Filtering graph files...")
    graphs_dir = PACK_DIR / "graphs"
    graphs_dir.mkdir()

    graph_files = {
        "pruned_graph_v2.json":          "pruned_graph.json",
        "multimodal_elements_v2.json":   "multimodal_elements.json",
    }
    doc_set = set(doc_ids)
    for src_name, dst_name in graph_files.items():
        src_path = GRAPH_DIR / src_name
        if not src_path.exists():
            continue
        with open(src_path) as f:
            raw = json.load(f)
        if "documents" in raw and isinstance(raw["documents"], dict):
            raw["documents"] = {k: v for k, v in raw["documents"].items() if k in doc_set}
            if "metadata" in raw:
                raw["metadata"]["filtered_for_delivery"] = True
                raw["metadata"]["papers_processed"] = len(raw["documents"])
        dst_path = graphs_dir / dst_name
        with open(dst_path, "w") as f:
            json.dump(raw, f, ensure_ascii=False)
        print(f"  {dst_name}: {src_path.stat().st_size/1024/1024:.1f}MB -> {dst_path.stat().st_size/1024/1024:.1f}MB")

    # hub_scores needs special filtering (list-based, not doc-keyed)
    hub_src = GRAPH_DIR / "hub_scores_v2.json"
    if hub_src.exists():
        with open(hub_src) as f:
            hub = json.load(f)
        # Filter all list/dict fields by doc_id
        if isinstance(hub, dict):
            filtered_hub = {}
            for key, val in hub.items():
                if isinstance(val, list):
                    filtered_hub[key] = [
                        item for item in val
                        if isinstance(item, dict) and item.get("doc_id", "") in doc_set
                    ]
                    # If no doc_id filtering worked, check if items lack doc_id
                    if not filtered_hub[key] and val:
                        filtered_hub[key] = val  # keep as-is
                elif isinstance(val, dict):
                    # Try filtering by key
                    sub = {k: v for k, v in val.items() if k in doc_set}
                    filtered_hub[key] = sub if sub else val
                else:
                    filtered_hub[key] = val
            hub = filtered_hub
        dst_path = graphs_dir / "hub_scores.json"
        with open(dst_path, "w") as f:
            json.dump(hub, f, ensure_ascii=False)
        print(f"  hub_scores.json: {hub_src.stat().st_size/1024/1024:.1f}MB -> {dst_path.stat().st_size/1024/1024:.1f}MB")

    # ── Copy enriched candidates ──
    cand_dir = PACK_DIR / "candidates"
    cand_dir.mkdir()
    for src_name, dst_name in [
        ("hub_candidates_enriched_v4_intra_doc.json", "hub_candidates_intra_doc.json"),
        ("m2_diverse_candidates_intra_doc.json",      "m2_diverse_candidates_intra_doc.json"),
    ]:
        src = ENRICHED_DIR / src_name
        if src.exists():
            shutil.copy2(src, cand_dir / dst_name)
            print(f"  {dst_name}: {src.stat().st_size/1024/1024:.1f}MB")

    # ── Stats ──
    hop_dist = collections.Counter()
    style_dist = collections.Counter()
    doc_dist = collections.Counter()
    for q in final_queries:
        hop_dist[q.get("hop_distance", "?")] += 1
        style_dist[q.get("query_style", "?")] += 1
        doc_dist[q.get("doc_id", "?")] += 1

    stats = {
        "version": "v1",
        "created_at": datetime.now().isoformat(),
        "total_queries": len(final_queries),
        "total_corpus_passages": len(corpus),
        "total_triplets": len(triplets),
        "total_qrels": len(qrels),
        "unique_documents": len(doc_ids),
        "documents_with_mineru": copied_mineru,
        "documents_with_latex": copied_latex,
        "hop_distribution": {str(k): v for k, v in sorted(hop_dist.items())},
        "style_distribution": dict(sorted(style_dist.items())),
        "top_10_docs": {k: v for k, v in doc_dist.most_common(10)},
    }
    with open(PACK_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # ── README ──
    readme = f"""# M4query v1 — Multi-hop Cross-modal QA Dataset

Multi-hop, cross-modal question-answer dataset built from academic papers,
designed for contrastive learning and embedding model training.

## Dataset Structure

```
M4query_v1/
├── queries.jsonl              # {len(final_queries)} QC-passed queries
├── corpus.jsonl               # {len(corpus)} passage chunks (MinerU elements)
├── train_triplets.jsonl       # {len(triplets)} contrastive triplets (query→pos/neg)
├── qrels.jsonl                # {len(qrels)} query-passage relevance labels
├── stats.json                 # Dataset statistics
├── documents/                 # Source documents ({len(doc_ids)} papers)
│   └── {{doc_id}}/
│       ├── mineru/            # MinerU parsed: structure.json, images, formulas
│       └── latex/             # LaTeX source (when available)
├── graphs/                    # Document knowledge graphs
│   ├── pruned_graph.json
│   ├── hub_scores.json
│   └── multimodal_elements.json
└── candidates/                # Enriched pair candidates
    ├── hub_candidates_intra_doc.json
    └── m2_diverse_candidates_intra_doc.json
```

## Training Data Format

### `train_triplets.jsonl` (for contrastive learning / embedding training)

Each line:
```json
{{
  "query_id": "l3_de_1511.00830_0000",
  "query": "...",
  "doc_id": "1511.00830",
  "hop_distance": 3,
  "positive": [
    {{"passage_id": "...", "text": "...", "type": "paragraph"}}
  ],
  "negative": [
    {{"passage_id": "...", "text": "...", "type": "table", "neg_source": "intra_doc"}},
    {{"passage_id": "...", "text": "...", "type": "paragraph", "neg_source": "cross_doc"}}
  ]
}}
```

### `corpus.jsonl` (passage pool for retrieval)

Each line:
```json
{{
  "passage_id": "1511.00830_elem_042",
  "doc_id": "1511.00830",
  "text": "...",
  "type": "paragraph|table|figure|equation",
  "section": "3.2 Method",
  "page": 5
}}
```

### `qrels.jsonl` (relevance labels)

Each line:
```json
{{
  "query_id": "l3_de_1511.00830_0000",
  "passage_id": "1511.00830_elem_042",
  "relevance": 1
}}
```

## Statistics

- Queries: {len(final_queries)}
- Corpus passages: {len(corpus)}
- Training triplets: {len(triplets)}
- Unique documents: {len(doc_ids)}
- Hop distribution: {dict(sorted(hop_dist.items()))}
- Style: {dict(sorted(style_dist.items()))}
- MinerU coverage: {copied_mineru}/{len(doc_ids)}
- LaTeX coverage: {copied_latex}/{len(doc_ids)}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
    with open(PACK_DIR / "README.md", "w") as f:
        f.write(readme)

    # ── Create zip ──
    print("\n  Creating zip archive...")
    zip_path = shutil.make_archive(str(ZIP_OUT), "zip", PACK_DIR.parent, PACK_DIR.name)
    zip_sz = os.path.getsize(zip_path) / 1024 / 1024
    print(f"\n  ✅ Zip: {zip_path} ({zip_sz:.1f} MB)")
    return zip_path


# ══════════════════════════════════════════════════════════════════════
# Step 5: Upload to ModelScope
# ══════════════════════════════════════════════════════════════════════

def step5_upload(zip_path):
    print("\n" + "=" * 60)
    print("STEP 5: Upload to ModelScope")
    print("=" * 60)

    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(MS_TOKEN)
    print(f"  Logged in, uploading to {MS_REPO}...")

    # Upload the whole folder for better browsability
    print(f"  Uploading folder {PACK_DIR} ...")
    api.upload_folder(
        repo_id=MS_REPO,
        folder_path=str(PACK_DIR),
        path_in_repo="",
        commit_message=f"Delivery v1: 473 queries + corpus + triplets + docs ({datetime.now().strftime('%Y-%m-%d')})",
        repo_type="dataset",
    )
    print(f"  ✅ Uploaded folder to {MS_REPO}")

    # Also upload the zip as a single file
    zip_name = os.path.basename(zip_path)
    api.upload_file(
        repo_id=MS_REPO,
        path_or_fileobj=zip_path,
        path_in_repo=zip_name,
        commit_message=f"Add {zip_name}",
        repo_type="dataset",
    )
    print(f"  ✅ Uploaded {zip_name}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(f"M4query Full Delivery Pipeline")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Load doc IDs
    doc_ids = [l.strip() for l in open(DOC_IDS_FILE) if l.strip()]
    print(f"Target docs: {len(doc_ids)}")

    # Load queries
    queries = []
    with open(DELIVERY_SRC) as f:
        for line in f:
            queries.append(json.loads(line.strip()))
    print(f"Queries: {len(queries)}")

    # Step 1: LaTeX
    latex_docs = step1_download_latex()

    # Step 2: Corpus
    corpus = step2_build_corpus(doc_ids)

    # Step 3: Triplets
    triplets, qrels = step3_build_triplets(queries, corpus)

    # Step 4: Package
    zip_path = step4_package(queries, corpus, triplets, qrels, doc_ids, latex_docs)

    # Step 5: Upload
    step5_upload(zip_path)

    print("\n" + "=" * 60)
    print(f"DONE: {datetime.now().isoformat()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
