#!/usr/bin/env python3
"""
Full delivery pipeline:
  1. Download LaTeX sources for delivery docs (arXiv e-print)
  2. Build corpus.jsonl from graph + MinerU elements
  3. Build train_triplets.jsonl (query + pos + neg) for contrastive learning
  4. Build qrels.jsonl (query-passage relevance)
  5. Package everything into zip
  6. Optionally upload to ModelScope

Usage (standalone or via SLURM):
    python scripts/build_full_delivery.py --skip-upload
"""

import argparse
import json
import os
import sys
import shutil
import random
import collections
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
GRAPH_ELEMENTS = GRAPH_DIR / "multimodal_elements_v2.json"

PACK_NAME = "M4query_v1"
PACK_DIR  = QUERY_DIR / PACK_NAME
ZIP_OUT   = QUERY_DIR / "M4query_delivery_v1"  # .zip appended by make_archive

random.seed(42)

# ── ModelScope config ─────────────────────────────────────────────────
MS_TOKEN   = "ms-4fcf0dbb-239b-4707-82a7-18f1c64f6dcb"
MS_REPO    = "IgnatiusMao/M4query_test"


def normalize_element_id(element_id):
    """Canonicalize common element-id aliases so qrels and corpus share one namespace."""
    eid = str(element_id or "").strip()
    if not eid:
        return ""
    return eid.replace("_fig_", "_figure_").replace("_equation_", "_formula_")


def infer_doc_id_from_element_id(element_id):
    eid = normalize_element_id(element_id)
    return eid.split("_", 1)[0] if "_" in eid else eid


def infer_element_type(element_id, raw_type=None):
    raw = str(raw_type or "").strip().lower()
    if raw in {"figure", "image"}:
        return "figure"
    if raw in {"table"}:
        return "table"
    if raw in {"formula", "equation"}:
        return "formula"
    if raw in {"text", "paragraph", "section"}:
        return "text"

    eid = normalize_element_id(element_id)
    if "_figure_" in eid:
        return "figure"
    if "_table_" in eid:
        return "table"
    if "_formula_" in eid:
        return "formula"
    return "text"


def extract_caption(elem):
    meta = elem.get("metadata")
    if isinstance(meta, dict):
        val = meta.get("caption")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def extract_image_path(elem):
    for key in ("image_path", "img_path"):
        val = elem.get(key)
        if isinstance(val, str) and val.strip() and val != "None":
            return val.strip()
    return ""


def merge_element_records(existing, raw_elem, doc_id):
    """Merge graph and MinerU element records keyed by canonical element_id."""
    raw_id = raw_elem.get("element_id", raw_elem.get("id", ""))
    eid = normalize_element_id(raw_id)
    if not eid:
        return existing

    content = str(raw_elem.get("content", raw_elem.get("text", "")) or "").strip()
    candidate = {
        "element_id": eid,
        "doc_id": doc_id or infer_doc_id_from_element_id(eid),
        "type": infer_element_type(eid, raw_elem.get("type")),
        "content": content,
        "caption": extract_caption(raw_elem),
        "section": str(raw_elem.get("section", "") or "").strip(),
        "page": raw_elem.get("page_idx", raw_elem.get("page", -1)),
        "image_path": extract_image_path(raw_elem),
    }

    if existing is None:
        return candidate

    if len(candidate["content"]) > len(existing.get("content", "")):
        existing["content"] = candidate["content"]
    if len(candidate["caption"]) > len(existing.get("caption", "")):
        existing["caption"] = candidate["caption"]
    if candidate["section"] and not existing.get("section"):
        existing["section"] = candidate["section"]
    if existing.get("page", -1) in (-1, None) and candidate["page"] not in (-1, None):
        existing["page"] = candidate["page"]
    if candidate["image_path"] and not existing.get("image_path"):
        existing["image_path"] = candidate["image_path"]
    if existing.get("type") == "text" and candidate["type"] != "text":
        existing["type"] = candidate["type"]
    return existing


def collect_target_doc_ids(base_doc_ids, queries):
    """Include query-owning docs plus any cross-doc positive evidence docs."""
    doc_ids = {d for d in base_doc_ids if d}
    for q in queries:
        if q.get("doc_id"):
            doc_ids.add(q["doc_id"])
        for eid in q.get("element_ids", []):
            doc_ids.add(infer_doc_id_from_element_id(eid))
    return sorted(d for d in doc_ids if d)


def load_doc_elements(doc_ids):
    """Load and merge graph elements with MinerU structure by canonical element_id."""
    with open(GRAPH_ELEMENTS) as f:
        raw = json.load(f)

    doc_elements = {}
    docs_data = raw.get("documents", {})
    for did in doc_ids:
        merged = {}

        doc_entry = docs_data.get(did, {})
        raw_elems = doc_entry.get("elements", doc_entry.get("nodes", []))
        if isinstance(raw_elems, dict):
            raw_elems = list(raw_elems.values())
        for elem in raw_elems:
            existing = merged.get(normalize_element_id(elem.get("element_id", elem.get("id", ""))))
            merged[normalize_element_id(elem.get("element_id", elem.get("id", "")))] = merge_element_records(existing, elem, did)

        struct_path = MINERU_BASE / did / "structure.json"
        if struct_path.exists():
            with open(struct_path) as f:
                struct = json.load(f)
            for elem in struct.get("elements", []):
                norm_id = normalize_element_id(elem.get("element_id", elem.get("id", "")))
                existing = merged.get(norm_id)
                merged[norm_id] = merge_element_records(existing, elem, did)

        doc_elements[did] = [elem for eid, elem in sorted(merged.items()) if eid]
    return doc_elements


def build_evidence_fallbacks(queries):
    """Aggregate short query-side evidence descriptions to rescue empty multimodal elements."""
    fallback_parts = collections.defaultdict(list)
    for q in queries:
        for span in q.get("required_evidence_spans", []) or []:
            if not isinstance(span, dict):
                continue
            eid = normalize_element_id(span.get("element_id", ""))
            text = str(span.get("span", "") or "").strip()
            if eid and text:
                fallback_parts[eid].append(text)
        for anchor in q.get("visual_anchors", []) or []:
            if not isinstance(anchor, dict):
                continue
            eid = normalize_element_id(anchor.get("element_id", ""))
            text = str(anchor.get("anchor", "") or "").strip()
            if eid and text:
                fallback_parts[eid].append(text)

    fallbacks = {}
    for eid, parts in fallback_parts.items():
        dedup = []
        seen = set()
        for part in parts:
            key = part.lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(part)
        fallbacks[eid] = " ".join(dedup[:2]).strip()
    return fallbacks


def element_to_passage(elem, fallback_text=""):
    """Build a retrieval passage, preferring real element text and falling back to query-side evidence when needed."""
    eid = normalize_element_id(elem.get("element_id", ""))
    etype = infer_element_type(eid, elem.get("type"))
    content = str(elem.get("content", "") or "").strip()
    caption = str(elem.get("caption", "") or "").strip()
    fallback_text = str(fallback_text or "").strip()

    parts = []
    if etype in {"figure", "table"}:
        if caption:
            parts.append(f"[{etype.upper()}] {caption}")
        if content and content != caption:
            parts.append(content)
        if fallback_text:
            parts.append(fallback_text)
        if not parts and elem.get("image_path"):
            parts.append(f"[{etype.upper()}] {Path(elem['image_path']).name}")
    elif etype == "formula":
        if content:
            parts.append(f"[FORMULA] {content}")
        if fallback_text:
            parts.append(fallback_text)
    else:
        if content:
            parts.append(content)
        if fallback_text and fallback_text not in content:
            parts.append(fallback_text)

    if not parts:
        parts.append(f"[{etype.upper()}] {eid}")
    return " ".join(" ".join(parts).split())


# ══════════════════════════════════════════════════════════════════════
# Step 1: Download LaTeX sources
# ══════════════════════════════════════════════════════════════════════

def step1_download_latex(doc_ids):
    """Download LaTeX sources for the docs we need to ship."""
    print("\n" + "=" * 60)
    print("STEP 1: Download LaTeX sources")
    print("=" * 60)

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

def step2_build_corpus(doc_ids, queries):
    """Build corpus.jsonl from merged graph + MinerU elements, guaranteeing positive coverage."""
    print("\n" + "=" * 60)
    print("STEP 2: Build corpus from graph + MinerU elements")
    print("=" * 60)

    doc_elements = load_doc_elements(doc_ids)
    fallback_texts = build_evidence_fallbacks(queries)
    positive_ids = {
        normalize_element_id(eid)
        for q in queries
        for eid in q.get("element_ids", [])
        if normalize_element_id(eid)
    }

    corpus = []
    passage_lookup = {}
    type_counts = collections.Counter()

    for did in sorted(doc_ids):
        for elem in doc_elements.get(did, []):
            eid = normalize_element_id(elem.get("element_id", ""))
            if not eid or eid in passage_lookup:
                continue
            text = element_to_passage(elem, fallback_texts.get(eid, "") if eid in positive_ids else "")
            if len(text) < 10 and eid not in positive_ids:
                continue
            passage = {
                "passage_id": eid,
                "doc_id": elem.get("doc_id", did),
                "text": text,
                "type": infer_element_type(eid, elem.get("type")),
                "section": elem.get("section", ""),
                "page": elem.get("page", -1),
            }
            if elem.get("image_path"):
                passage["image_path"] = elem["image_path"]
            corpus.append(passage)
            passage_lookup[eid] = passage
            type_counts[passage["type"]] += 1

    supplemental = 0
    for eid in sorted(positive_ids):
        if eid in passage_lookup:
            continue
        text = fallback_texts.get(eid, "")
        if not text:
            text = f"[{infer_element_type(eid).upper()}] {eid}"
        passage = {
            "passage_id": eid,
            "doc_id": infer_doc_id_from_element_id(eid),
            "text": text,
            "type": infer_element_type(eid),
            "section": "",
            "page": -1,
            "source": "query_fallback",
        }
        corpus.append(passage)
        passage_lookup[eid] = passage
        type_counts[passage["type"]] += 1
        supplemental += 1

    corpus.sort(key=lambda x: x["passage_id"])
    print(f"  Total passages: {len(corpus)} from {len(doc_ids)} docs")
    print(f"  Type breakdown: {dict(sorted(type_counts.items()))}")
    print(f"  Supplemental positive passages: {supplemental}")
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
    qrels = []  # standard positive-only qrels
    unresolved = 0

    for q in queries:
        qid = q["query_id"]
        doc_id = q.get("doc_id", "")
        pos_eids = [normalize_element_id(eid) for eid in q.get("element_ids", []) if normalize_element_id(eid)]

        # --- Positive passages ---
        positives = []
        for eid in pos_eids:
            if eid in passage_lookup:
                positives.append({
                    "passage_id": eid,
                    "text": passage_lookup[eid]["text"][:1000],
                    "type": passage_lookup[eid].get("type", ""),
                })
        if len(positives) != len(pos_eids):
            unresolved += 1

        # --- Negative passages ---
        negatives = []
        positive_ids = {pp["passage_id"] for pp in positives}

        # Intra-doc hard negatives (same doc, different element)
        intra_pool = [p for p in doc_passages.get(doc_id, [])
                      if p["passage_id"] not in positive_ids]
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
                candidates = [p for p in pool if p["passage_id"] not in positive_ids]
                p = random.choice(candidates or pool)
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

    pos_counts = [len(t["positive"]) for t in triplets]
    neg_counts = [len(t["negative"]) for t in triplets]
    print(f"  Triplets: {len(triplets)}")
    print(f"  Avg positives/query: {sum(pos_counts)/len(pos_counts):.1f}")
    print(f"  Avg negatives/query: {sum(neg_counts)/len(neg_counts):.1f}")
    print(f"  Qrels: {len(qrels)}")
    print(f"  Queries with unresolved positives: {unresolved}")

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-upload", action="store_true",
                    help="Build and zip locally without uploading to ModelScope.")
    args = ap.parse_args()

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

    target_doc_ids = collect_target_doc_ids(doc_ids, queries)
    extra_docs = [d for d in target_doc_ids if d not in doc_ids]
    if extra_docs:
        print(f"Expanded docs for cross-doc positives: +{len(extra_docs)} -> {extra_docs}")
    print(f"Final docs to package: {len(target_doc_ids)}")

    # Step 1: LaTeX
    latex_docs = step1_download_latex(target_doc_ids)

    # Step 2: Corpus
    corpus = step2_build_corpus(target_doc_ids, queries)

    # Step 3: Triplets
    triplets, qrels = step3_build_triplets(queries, corpus)

    # Step 4: Package
    zip_path = step4_package(queries, corpus, triplets, qrels, target_doc_ids, latex_docs)

    # Step 5: Upload
    if args.skip_upload:
        print("\nSkipping upload (--skip-upload).")
    else:
        step5_upload(zip_path)

    print("\n" + "=" * 60)
    print(f"DONE: {datetime.now().isoformat()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
