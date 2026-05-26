#!/usr/bin/env python3
"""Package noncs2000 final delivery: sweep + retry + real_user → unified dataset.

Produces delivery directory with:
  - corpus.jsonl        : figure/table/formula/section/chunk passages
  - corpus.jsonl.gz     : gzip compressed version
  - train_triplets.jsonl: per-query 3-4 positives + 5 hard_neg + 5 random_neg
  - images/             : all referenced images, copied from MinerU output

Image handling (per M4query_v2_clean experience):
  - Rewrite image_path from MinerU paths to images/<doc_id>/<hash>.jpg
  - Copy images physically into images/
  - Merge table_screenshot into table (same image, single record)
  - Remove bare figures/tables (no caption/description, no image file)
  - Clean orphan images after copy

Usage:
    python scripts/package_noncs2000_final.py \
        --output data/03_queries/M4query_noncs2000_final \
        --dry-run   # check first
    python scripts/package_noncs2000_final.py \
        --output data/03_queries/M4query_noncs2000_final
"""

from __future__ import annotations
import argparse
import collections
import gzip
import hashlib
import json
import os
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
QUERY_DIR = ROOT / "data" / "03_queries"
GRAPH_DIR = ROOT / "data" / "01_graphs"
ENRICHED_DIR = ROOT / "data" / "02_enriched"
RAW_DIR = ROOT / "data" / "00_raw"

# ── Input sources ────────────────────────────────────────────────────
SWEEP_FILES = [
    QUERY_DIR / "noncs2000_sweep_l3_acad_final_pass.jsonl",
    QUERY_DIR / "noncs2000_sweep_l3_acad_persona_final_pass.jsonl",
    QUERY_DIR / "noncs2000_sweep_l3_mixed_persona_final_pass.jsonl",
]

RETRY_FILES = []
for tag in ["acad", "acad_persona", "mixed_persona"]:
    for s in [0, 1]:
        RETRY_FILES.append(QUERY_DIR / f"noncs2000_retry_{tag}_s{s}_pass.jsonl")

REALUSER_FILES = [QUERY_DIR / f"noncs2000_realuser_s{s}_pass.jsonl" for s in range(6)]

ELEMENTS_PATH = ENRICHED_DIR / "noncs2000_elements_enriched_2111.json"
SECTIONS_PATH = ENRICHED_DIR / "noncs2000_section_nodes_enriched_2111.json"
CHUNKS_PATH = GRAPH_DIR / "noncs2000_hierarchical_chunks_2111_enriched.json"

# MinerU image sources (noncs2000 papers use noncs1000_mineru_output)
MINERU_IMAGE_ROOTS = [
    RAW_DIR / "noncs1000_mineru_output",
    RAW_DIR / "noncs2000_mineru_output",
]

# ── Constants ─────────────────────────────────────────────────────────
HARD_NEG_TARGET = 5
RANDOM_NEG_TARGET = 5
VISUAL_TYPES = {"figure", "table", "formula"}
TEXT_TYPES = {"section", "chunk", "paragraph"}

random.seed(42)


# ── Helpers ───────────────────────────────────────────────────────────
def normalize_eid(eid: str) -> str:
    return (eid or "").replace("_fig_", "_figure_").replace("_equation_", "_formula_")


def doc_of(eid: str) -> str:
    eid = normalize_eid(eid)
    return eid.split("_", 1)[0] if "_" in eid else eid


def infer_type(eid: str) -> str:
    eid = normalize_eid(eid)
    for t in ["figure", "table", "formula", "section", "chunk", "paragraph"]:
        if f"_{t}_" in eid:
            return t
    return "section"


# ── Step 1: Merge queries ────────────────────────────────────────────
def load_queries(files: List[Path]) -> List[Dict]:
    all_q = []
    seen = set()
    for path in files:
        if not path.exists():
            continue
        for line in open(path, encoding="utf-8"):
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = o.get("query_id") or f"{o.get('pair_id','')}::{o.get('query','')[:40]}"
            if qid in seen:
                continue
            seen.add(qid)
            all_q.append(o)
    return all_q


# ── Step 2: Corpus builder ────────────────────────────────────────────
def build_corpus(doc_filter: Set[str]) -> Tuple[Dict[str, Dict], Dict[str, str], Dict[str, str]]:
    """Build corpus dict. Returns (corpus, image_map, table_screenshot_map).

    image_map: {new_rel_path: source_abs_path}
    table_screenshot_map: {table_passage_id: screenshot_src_path}
    """
    corpus: Dict[str, Dict] = {}
    img_map: Dict[str, str] = {}          # new_path → source_abs_path
    ts_map: Dict[str, str] = {}           # table_pid → screenshot src

    # ── Figure / Table / Formula ──
    print("[corpus] loading elements …")
    elem_data = json.loads(ELEMENTS_PATH.read_text(encoding="utf-8"))
    docs = elem_data.get("documents", {})
    n_vis = 0
    for doc_id, doc in docs.items():
        if doc_id not in doc_filter:
            continue
        for eid, el in (doc.get("elements") or {}).items():
            eid_n = normalize_eid(eid)
            t = el.get("element_type") or infer_type(eid_n)
            if t not in VISUAL_TYPES:
                continue
            caption = (el.get("caption") or "").strip()
            content = (el.get("content") or "").strip()
            desc = (el.get("enriched_content") or "").strip()
            img_src = (el.get("image_path") or "").strip()

            # Rewrite image path
            new_img = ""
            if img_src:
                # Resolve to absolute, then hash
                for root_dir in MINERU_IMAGE_ROOTS:
                    # img_src is like: data/00_raw/noncs1000_mineru_output/0704.0212/vlm/images/<hash>.jpg
                    src_path = Path(img_src)
                    if not src_path.is_absolute():
                        src_path = ROOT / img_src
                    if src_path.exists():
                        img_hash = src_path.name  # <hash>.jpg
                        new_img = f"images/{doc_id}/{img_hash}"
                        img_map[new_img] = str(src_path)
                        break

            # Handle table_screenshot → merge into table
            if t == "table" and img_src:
                ts_map[eid_n] = img_src

            corpus[eid_n] = {
                "passage_id": eid_n,
                "type": t,
                "text": content[:2000],
                "caption": caption,
                "image_path": new_img,
                "description": desc,
            }
            n_vis += 1
    print(f"  visual passages: {n_vis}")
    del elem_data

    # ── Section ──
    print("[corpus] loading section nodes …")
    sec_data = json.loads(SECTIONS_PATH.read_text(encoding="utf-8"))
    n_sec = 0
    for sec in sec_data.get("sections", []):
        doc_id = sec.get("doc_id", "")
        if doc_id not in doc_filter:
            continue
        sid = sec.get("section_id") or ""
        idx = sid.rsplit("::", 1)[-1] if "::" in sid else sid
        pid_n = f"{doc_id}_section_{idx}"
        text = (sec.get("section_text") or "").strip() or (sec.get("enriched_content") or "").strip()
        if not text:
            continue
        corpus[pid_n] = {
            "passage_id": pid_n,
            "type": "section",
            "text": text[:3000],
            "caption": (sec.get("section_title") or "").strip(),
            "image_path": "",
            "description": (sec.get("enriched_content") or "").strip()[:1500],
        }
        n_sec += 1
    print(f"  sections: {n_sec}")
    del sec_data

    # ── Chunks ──
    print("[corpus] loading hierarchical chunks …")
    chunk_data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunks_by_doc = chunk_data.get("documents", chunk_data)
    n_chunk = 0
    for doc_id, doc in chunks_by_doc.items():
        if doc_id not in doc_filter:
            continue
        for ch in (doc.get("fine_chunks") or doc.get("chunks") or []):
            cid = ch.get("chunk_id") or ch.get("id") or ""
            if not cid:
                continue
            cid_n = cid if cid.startswith(doc_id) else f"{doc_id}_chunk_{cid}"
            text = (ch.get("text") or ch.get("content") or "").strip()
            if not text:
                continue
            corpus[cid_n] = {
                "passage_id": cid_n,
                "type": "chunk",
                "text": text[:4000],
                "caption": "",
                "image_path": "",
                "description": (ch.get("enriched_description") or "").strip(),
            }
            n_chunk += 1
    print(f"  chunks: {n_chunk}")
    del chunk_data

    return corpus, img_map, ts_map


# ── Step 3: Clean corpus (per M4query_v2_clean experience) ───────────
def clean_corpus(
    corpus: Dict[str, Dict],
    img_map: Dict[str, str],
    queries: List[Dict],
) -> Dict[str, Dict]:
    """Remove bare visual passages and passages with broken image references."""
    removed = 0
    cleaned = {}

    # Build set of passage_ids referenced by queries
    pos_ids = set()
    for q in queries:
        for eid in (q.get("element_ids") or []):
            pos_ids.add(normalize_eid(eid))

    for pid, p in corpus.items():
        t = p.get("type", "")

        # ── Bare figure/table: no caption AND no description AND no image
        if t in ("figure", "table"):
            has_text = bool(p.get("caption") or p.get("description"))
            has_img = bool(p.get("image_path"))
            if not has_text and not has_img:
                removed += 1
                continue
            # Image path exists but file not in img_map → broken reference
            if p.get("image_path") and p["image_path"] not in img_map:
                # Downgrade to text-only if it has content
                if p.get("caption") or p.get("description"):
                    p["image_path"] = ""
                else:
                    removed += 1
                    continue

        # ── Table screenshot merging: already handled via single table record
        cleaned[pid] = p

    print(f"[clean] removed {removed} bare/broken passages, kept {len(cleaned)}")
    return cleaned


# ── Step 4: Backfill descriptions ─────────────────────────────────────
def backfill_descriptions(corpus: Dict[str, Dict], queries: List[Dict]):
    """Backfill missing descriptions from query evidence spans."""
    # Build query_id → evidence_text map
    evidence_map: Dict[str, str] = {}
    for q in queries:
        for span in (q.get("required_evidence_spans") or []):
            if isinstance(span, dict):
                eid = normalize_eid(span.get("element_id", ""))
                text = span.get("span", "")
                if eid and text:
                    evidence_map[eid] = text

    filled = 0
    for pid, p in corpus.items():
        if p.get("type") in VISUAL_TYPES and not p.get("description"):
            if pid in evidence_map and evidence_map[pid] != p.get("text", ""):
                p["description"] = evidence_map[pid][:1500]
                filled += 1
    print(f"[backfill] filled {filled} missing descriptions from evidence spans")


# ── Step 5: Copy images ───────────────────────────────────────────────
def copy_images(img_map: Dict[str, str], output_dir: Path) -> Set[str]:
    """Copy images from source to output/images/. Returns set of copied paths."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    copied = set()

    for rel_path, src in img_map.items():
        dst = images_dir / rel_path   # rel_path = "images/<doc_id>/<hash>.jpg"
        # Actually rel_path already starts with "images/", so use it relative to output_dir
        dst = output_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try:
                shutil.copy2(src, dst)
            except OSError as e:
                print(f"  WARN: copy failed {src} → {dst}: {e}")
                continue
        copied.add(rel_path)
    print(f"[images] copied {len(copied)} images")

    # ── Clean orphan images (in images/ but not in img_map)
    orphans = 0
    for img_file in images_dir.rglob("*.jpg"):
        rel = str(img_file.relative_to(output_dir))
        if rel not in img_map:
            img_file.unlink()
            orphans += 1
    if orphans:
        print(f"[images] cleaned {orphans} orphan images")
    return copied


# ── Step 6: Build triplets ────────────────────────────────────────────
def build_triplets(
    queries: List[Dict],
    corpus: Dict[str, Dict],
    output_dir: Path,
) -> List[Dict]:
    """Build train triplets with negative rebalancing."""
    # Index corpus by type for sampling
    type_index: Dict[str, List[str]] = collections.defaultdict(list)
    for pid, p in corpus.items():
        type_index[p.get("type", "?")].append(pid)

    triplets = []
    n_pos_3 = 0
    n_pos_4 = 0

    for q in queries:
        query_id = q["query_id"]
        query_text = q.get("query", "")

        # Gather positives
        pos: List[str] = []
        seen_pos = set()
        for eid in (q.get("element_ids") or []):
            eid_n = normalize_eid(eid)
            if eid_n in corpus and eid_n not in seen_pos:
                pos.append(eid_n)
                seen_pos.add(eid_n)

        # Add bridge paragraph if available
        bridge_id = q.get("bridge_passage_id") or ""
        if bridge_id and bridge_id in corpus and bridge_id not in seen_pos:
            pos.append(bridge_id)
            seen_pos.add(bridge_id)

        # Add chunk positives
        elem_to_chunks = getattr(build_triplets, "_elem_to_chunks", {})
        for eid in pos[:]:
            for cid in elem_to_chunks.get(eid, []):
                if cid in corpus and cid not in seen_pos:
                    pos.append(cid)
                    seen_pos.add(cid)

        if len(pos) < 3:
            continue  # skip queries without enough positives

        # Trim to 4 max
        pos = pos[:4]
        if len(pos) == 3:
            n_pos_3 += 1
        else:
            n_pos_4 += 1

        # ── Negative sampling ──
        pos_docs = {doc_of(pid) for pid in pos}
        pos_types = {corpus[pid]["type"] for pid in pos}

        # Hard negatives: same doc, non-positive
        hard_neg: List[str] = []
        for pid, p in corpus.items():
            if pid in seen_pos:
                continue
            if doc_of(pid) in pos_docs:
                hard_neg.append(pid)
        random.shuffle(hard_neg)
        hard_neg = hard_neg[:HARD_NEG_TARGET]

        # Fallback: random negatives
        while len(hard_neg) < HARD_NEG_TARGET:
            pid = random.choice(list(corpus.keys()))
            if pid not in seen_pos and pid not in hard_neg:
                hard_neg.append(pid)
        hard_neg = hard_neg[:HARD_NEG_TARGET]

        # Random negatives (different doc)
        random_neg: List[str] = []
        for pid, p in corpus.items():
            if pid in seen_pos or pid in hard_neg:
                continue
            if doc_of(pid) not in pos_docs:
                random_neg.append(pid)
        random.shuffle(random_neg)
        random_neg = random_neg[:RANDOM_NEG_TARGET]

        while len(random_neg) < RANDOM_NEG_TARGET:
            pid = random.choice(list(corpus.keys()))
            if pid not in seen_pos and pid not in hard_neg and pid not in random_neg:
                random_neg.append(pid)
        random_neg = random_neg[:RANDOM_NEG_TARGET]

        triplets.append({
            "query_id": query_id,
            "query": query_text,
            "positive_passages": pos,
            "hard_negative_passages": hard_neg,
            "random_negative_passages": random_neg,
        })

    print(f"[triplets] {len(triplets)} triplets ({n_pos_3}×3pos, {n_pos_4}×4pos)")
    return triplets


# Precompute elem→chunk index
def _build_elem_chunk_index(corpus: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Build mapping from element_id → chunk passage_ids."""
    idx = collections.defaultdict(list)
    chunk_data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunks_by_doc = chunk_data.get("documents", chunk_data)
    for doc_id, doc in chunks_by_doc.items():
        for ch in (doc.get("fine_chunks") or doc.get("chunks") or []):
            cid = ch.get("chunk_id") or ch.get("id") or ""
            if not cid:
                continue
            cid_n = cid if cid.startswith(doc_id) else f"{doc_id}_chunk_{cid}"
            if cid_n not in corpus:
                continue
            for eid in (ch.get("element_ids") or []):
                idx[normalize_eid(eid)].append(cid_n)
    print(f"[index] elem→chunk: {len(idx)} elements covered")
    return idx


# ── Step 7: Write output ──────────────────────────────────────────────
def write_delivery(
    output_dir: Path,
    corpus: Dict[str, Dict],
    triplets: List[Dict],
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # corpus.jsonl
    corpus_path = output_dir / "corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for pid, p in sorted(corpus.items()):
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    corpus_size_mb = corpus_path.stat().st_size / (1024 * 1024)
    print(f"[write] corpus.jsonl: {len(corpus)} passages, {corpus_size_mb:.1f} MB")

    # corpus.jsonl.gz (manual gzip after)
    gz_path = output_dir / "corpus.jsonl.gz"
    with open(corpus_path, "rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    gz_size_mb = gz_path.stat().st_size / (1024 * 1024)
    print(f"[write] corpus.jsonl.gz: {gz_size_mb:.1f} MB")

    # train_triplets.jsonl
    triplet_path = output_dir / "train_triplets.jsonl"
    with open(triplet_path, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    triplet_size_mb = triplet_path.stat().st_size / (1024 * 1024)
    print(f"[write] train_triplets.jsonl: {len(triplets)} triplets, {triplet_size_mb:.1f} MB")

    # README.md
    type_counts = collections.Counter(p.get("type", "?") for p in corpus.values())
    pos_dist = collections.Counter()
    for t in triplets:
        pos_dist[len(t["positive_passages"])] += 1

    readme = f"""# M4query_noncs2000_final

noncs2000 full delivery: sweep + retry + real_user merged, deduplicated.

Built {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}

## Files

| Path | Count | Purpose |
| --- | ---: | --- |
| `corpus.jsonl.gz` | {len(corpus):,} | figure/table/formula/section/chunk passages |
| `train_triplets.jsonl` | {len(triplets):,} | 3-4 positive + {HARD_NEG_TARGET} hard_neg + {RANDOM_NEG_TARGET} random_neg |
| `images/` | {sum(1 for _ in (output_dir / 'images').rglob('*.jpg') if (output_dir / 'images').exists())} | All referenced images |

## corpus type distribution

| type | count |
| --- | ---: |
"""
    for t, c in type_counts.most_common():
        readme += f"| `{t}` | {c:,} |\n"

    readme += f"""
## positive count distribution

| count | query count |
| --- | ---: |
"""
    for n, c in sorted(pos_dist.items()):
        readme += f"| {n} | {c:,} |\n"

    readme += f"""
## negative type distribution

| type | ratio |
| --- | ---: |
"""
    neg_type_counts = collections.Counter()
    neg_total = 0
    for t in triplets:
        for pid in t["hard_negative_passages"] + t["random_negative_passages"]:
            if pid in corpus:
                neg_type_counts[corpus[pid].get("type", "?")] += 1
                neg_total += 1
    for tp, c in neg_type_counts.most_common():
        readme += f"| `{tp}` | {c / neg_total * 100:.1f}% |\n"

    readme += f"""
## Generation script

`scripts/package_noncs2000_final.py`
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"[write] README.md")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=QUERY_DIR / "M4query_noncs2000_final",
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check inputs without building")
    args = parser.parse_args()

    # Step 1: Load + merge queries
    print("=" * 60)
    print("Step 1: Merge queries")
    print("=" * 60)
    all_files = SWEEP_FILES + RETRY_FILES + REALUSER_FILES
    queries = load_queries(all_files)
    print(f"  Total unique pass queries: {len(queries)}")

    if args.dry_run:
        # Show source breakdown
        sweep_n = sum(1 for q in queries if any(
            q.get("query_id", "") in
            set(json.loads(line).get("query_id", "") for line in open(f, encoding="utf-8")
                if f.exists())
            for f in SWEEP_FILES
        ))
        print(f"  Dry run complete. Output would go to: {args.output}")
        return

    # Step 2: Build corpus
    print("\n" + "=" * 60)
    print("Step 2: Build corpus")
    print("=" * 60)
    doc_set = {doc_of(q.get("element_ids", [""])[0]) for q in queries if q.get("element_ids")}
    corpus, img_map, ts_map = build_corpus(doc_set)

    # Step 3: Clean corpus
    print("\n" + "=" * 60)
    print("Step 3: Clean corpus")
    print("=" * 60)
    corpus = clean_corpus(corpus, img_map, queries)

    # Step 4: Backfill descriptions
    print("\n" + "=" * 60)
    print("Step 4: Backfill descriptions")
    print("=" * 60)
    backfill_descriptions(corpus, queries)

    # Step 5: Copy images
    print("\n" + "=" * 60)
    print("Step 5: Copy images")
    print("=" * 60)
    copy_images(img_map, args.output)

    # Step 6: Build triplets
    print("\n" + "=" * 60)
    print("Step 6: Build triplets")
    print("=" * 60)
    build_triplets._elem_to_chunks = _build_elem_chunk_index(corpus)
    triplets = build_triplets(queries, corpus, args.output)

    # Step 7: Write delivery
    print("\n" + "=" * 60)
    print("Step 7: Write delivery")
    print("=" * 60)
    write_delivery(args.output, corpus, triplets)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
