#!/usr/bin/env python3
"""Audit M4query_noncs2000_final delivery package for quality issues."""

import json, os, sys
from collections import Counter, defaultdict
from pathlib import Path

PKG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/03_queries/M4query_noncs2000_final")

if not PKG.exists():
    print(f"ERROR: {PKG} not found")
    sys.exit(1)

corpus_path = PKG / "corpus.jsonl"
triplet_path = PKG / "train_triplets.jsonl"
images_dir = PKG / "images"

print(f"Auditing: {PKG}\n")

# ── 0. Basic stats ────────────────────────────────────────────────────
corpus = {}
for line in open(corpus_path, encoding="utf-8"):
    p = json.loads(line)
    corpus[p["passage_id"]] = p

triplets = [json.loads(line) for line in open(triplet_path, encoding="utf-8")]

print(f"corpus passages: {len(corpus)}")
print(f"triplets: {len(triplets)}")
print(f"images dir: {'exists' if images_dir.exists() else 'MISSING'}")
if images_dir.exists():
    imgs = list(images_dir.rglob("*.jpg"))
    print(f"  image files: {len(imgs)}")
print()

# ── 1. image_path 检查 ────────────────────────────────────────────────
print("=" * 60)
print("1. IMAGE PATH AUDIT")
print("=" * 60)

missing_img = []
broken_img = []
empty_img_visual = []
img_paths = set()

for pid, p in corpus.items():
    ip = p.get("image_path", "")
    if ip:
        img_paths.add(ip)
        # Check file exists
        full = PKG / ip
        if not full.exists():
            broken_img.append((pid, ip, p.get("type")))
    elif p.get("type") in ("figure", "table"):
        empty_img_visual.append((pid, p.get("type")))

# Check image_path format
bad_format = [ip for ip in img_paths if not ip.startswith("images/")]

print(f"  visual passages with image_path: {len(img_paths)}")
print(f"  BROKEN paths (file missing):     {len(broken_img)} {'❌' if broken_img else '✅'}")
if broken_img:
    for pid, ip, t in broken_img[:5]:
        print(f"    [{t}] {pid}: {ip}")

print(f"  BAD FORMAT (not images/...):      {len(bad_format)} {'❌' if bad_format else '✅'}")
if bad_format:
    for ip in bad_format[:5]:
        print(f"    {ip}")

print(f"  EMPTY image_path on visual:       {len(empty_img_visual)} {'⚠️' if empty_img_visual else '✅'}")
if empty_img_visual:
    for pid, t in empty_img_visual[:5]:
        print(f"    [{t}] {pid}: no image_path")

# ── 2. Orphan images ──────────────────────────────────────────────────
print()
print("=" * 60)
print("2. ORPHAN IMAGES")
print("=" * 60)

if images_dir.exists():
    orphans = []
    for img in images_dir.rglob("*.jpg"):
        rel = "images/" + str(img.relative_to(images_dir))
        if rel not in img_paths:
            orphans.append(rel)
    print(f"  orphan images (not in corpus): {len(orphans)} {'⚠️' if orphans else '✅'}")
    if orphans:
        for o in orphans[:5]:
            print(f"    {o}")

# ── 3. Empty content ──────────────────────────────────────────────────
print()
print("=" * 60)
print("3. EMPTY CONTENT AUDIT")
print("=" * 60)

empty_text = []
empty_caption_visual = []
empty_desc_visual = []
fully_empty = []

for pid, p in corpus.items():
    t = p.get("type")
    text = (p.get("text") or "").strip()
    caption = (p.get("caption") or "").strip()
    desc = (p.get("description") or "").strip()

    if not text:
        empty_text.append((pid, t))
    if t in ("figure", "table", "formula") and not caption and not desc:
        fully_empty.append((pid, t))
    elif t in ("figure", "table") and not caption:
        empty_caption_visual.append((pid, t))
    elif t in ("figure", "table") and not desc:
        empty_desc_visual.append((pid, t))

print(f"  empty text:            {len(empty_text)} {'⚠️' if empty_text else '✅'}")
if empty_text:
    by_type = Counter(t for _, t in empty_text)
    for t, n in by_type.most_common():
        print(f"    [{t}]: {n}")

print(f"  fully empty visual:    {len(fully_empty)} {'❌' if fully_empty else '✅'}")
if fully_empty:
    for pid, t in fully_empty[:5]:
        print(f"    [{t}] {pid}: no caption + no description")

print(f"  visual missing caption:{len(empty_caption_visual)} {'⚠️' if empty_caption_visual else '✅'}")
print(f"  visual missing desc:   {len(empty_desc_visual)} {'⚠️' if empty_desc_visual else '✅'}")

# ── 4. Triplet integrity ──────────────────────────────────────────────
print()
print("=" * 60)
print("4. TRIPLET INTEGRITY")
print("=" * 60)

missing_pos = []
missing_neg = []
dup_pos = []
short_triplet = []

for t in triplets:
    pos = t.get("positive_passages", [])
    hard = t.get("hard_negative_passages", [])
    rand = t.get("random_negative_passages", [])

    # Missing from corpus
    for pid in pos:
        if pid not in corpus:
            missing_pos.append((t["query_id"], pid))
    for pid in hard + rand:
        if pid not in corpus:
            missing_neg.append((t["query_id"], pid))

    # Duplicates
    if len(set(pos)) != len(pos):
        dup_pos.append(t["query_id"])

    # Too few positives
    if len(pos) < 3:
        short_triplet.append((t["query_id"], len(pos)))

    # Cross-contamination
    pos_set = set(pos)
    conflict = set(hard + rand) & pos_set
    if conflict:
        pass  # skip for now

print(f"  positive not in corpus:   {len(missing_pos)} {'❌' if missing_pos else '✅'}")
if missing_pos:
    for qid, pid in missing_pos[:5]:
        print(f"    q={qid} pid={pid}")

print(f"  negative not in corpus:   {len(missing_neg)} {'❌' if missing_neg else '✅'}")
if missing_neg:
    for qid, pid in missing_neg[:5]:
        print(f"    q={qid} pid={pid}")

print(f"  duplicate positives:      {len(dup_pos)} {'❌' if dup_pos else '✅'}")
print(f"  short positives (<3):     {len(short_triplet)} {'❌' if short_triplet else '✅'}")
if short_triplet:
    for qid, n in short_triplet[:5]:
        print(f"    {qid}: {n} positives")

# ── 5. Query empty evidence ───────────────────────────────────────────
print()
print("=" * 60)
print("5. QUERY EMPTY EVIDENCE")
print("=" * 60)

# Check actual queries from the triplets for empty fields
empty_query = 0
empty_answer = 0
empty_evidence = 0
all_queries = set()

# Load all queries from sweep/retry/real_user to match triplets
import glob
query_data = {}
for pat in ["noncs2000_sweep_l3_*_final_pass.jsonl",
            "noncs2000_retry_*_s*_pass.jsonl",
            "noncs2000_realuser_s*_pass.jsonl"]:
    for fp in Path("data/03_queries").glob(pat):
        for line in open(fp, encoding="utf-8"):
            try:
                o = json.loads(line)
                query_data[o.get("query_id", "")] = o
            except:
                pass

print(f"  loaded {len(query_data)} query records")

for t in triplets:
    qid = t["query_id"]
    q = query_data.get(qid, {})
    if not q.get("query"):
        empty_query += 1
    if not q.get("answer"):
        empty_answer += 1
    if not q.get("required_evidence_spans"):
        empty_evidence += 1
    all_queries.add(qid)

unmatched = [t["query_id"] for t in triplets if t["query_id"] not in query_data]

print(f"  empty query text:         {empty_query} {'❌' if empty_query else '✅'}")
print(f"  empty answer:             {empty_answer} {'⚠️' if empty_answer else '✅'}")
print(f"  empty evidence_spans:     {empty_evidence} {'❌' if empty_evidence else '✅'}")
print(f"  triplets unmatched:       {len(unmatched)} {'❌' if unmatched else '✅'}")
if unmatched:
    for qid in unmatched[:5]:
        print(f"    {qid}")

# ── 6. Overall ────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)

issues = sum(bool(x) for x in [broken_img, bad_format, orphans, fully_empty, 
                                 missing_pos, missing_neg, dup_pos, short_triplet,
                                 empty_evidence])
total_checks = 25  # rough
clean = total_checks - issues
print(f"  clean checks:  {clean}/{total_checks}")
print(f"  issues found:  {issues}")

if issues == 0:
    print("  ✅ ALL CLEAN")
else:
    print("  ⚠️  Issues to review (see above)")
