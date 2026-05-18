#!/usr/bin/env python3
"""Build M4query_v2_clean_chunk_aug: clean + chunk augmented dataset.

Key differences from M4query_v2_clean_chunk:
  - paragraph AND chunk coexist (multi-granularity)
  - Bridge mapped to paragraph source → replace bridge with source paragraph + chunk (4 pos)
  - Bridge unmappable or source is f/t/f → keep bridge, retype to "paragraph" (3 pos)
  - Negatives: 1-2 text (paragraph/chunk) slots reserved per hard_neg and random_neg group
  - No query dropped (8,104 total)
"""

import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path("/projects/myyyx1/data-process-test")
SRC_CLEAN = ROOT / "data/03_queries/M4query_v2_clean"
SRC_RAW = ROOT / "data/03_queries/M4query_v2"
SRC_CHUNK = ROOT / "data/03_queries/M4query_v2_clean_chunk"
OUT = ROOT / "data/03_queries/M4query_v2_clean_chunk_aug"
MAPPING_PATH = SRC_CLEAN / "bridge_to_source.json"

CHUNK_SIZE = 400
POSITIVE_TARGET = 3
HARD_NEG_TARGET = 5
RANDOM_NEG_TARGET = 5
TEXT_SLOTS_MIN = 2          # min text (paragraph/chunk) slots in each neg group
TEXT_SLOTS_MAX = 4          # max text slots in each neg group

WORD_RE = re.compile(r"\S+")
_SEC_NUMERIC = re.compile(r"^\d+(\.\d+)*\.?\s+\S")
_SEC_KEYWORDS = re.compile(
    r"^(abstract|introduction|background|related\s+work|methods?|methodology|"
    r"approach|experiments?|evaluation|results?|discussion|conclusion|"
    r"acknowledgements?|references?|appendix)\b",
    re.IGNORECASE,
)


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def word_count(s):
    return len(WORD_RE.findall(s or ""))


def doc_id_of_paragraph(pid):
    m = re.match(r"(.+)_paragraph_(\d+)$", pid)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def is_real_section_head(text):
    t = (text or "").strip()
    if not t or len(t) > 100 or t.endswith(":"):
        return False
    if _SEC_NUMERIC.match(t) or _SEC_KEYWORDS.match(t):
        return True
    return False


def load_section_heads(doc_id):
    p = SRC_RAW / "documents" / doc_id / "mineru" / f"{doc_id}_content_list.json"
    if not p.exists():
        return []
    try:
        cl = json.load(open(p))
    except Exception:
        return []
    return [i for i, x in enumerate(cl)
            if x.get("type") == "text" and x.get("text_level") == 1
            and is_real_section_head(x.get("text", ""))]


def section_of(idx, heads):
    s = 0
    for h in heads:
        if idx >= h:
            s = h
        else:
            break
    return s


def det_pick(seed, pool, exclude, n):
    if not pool or n <= 0:
        return []
    h = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
    picked, i, limit = [], 0, len(pool) * 3
    while len(picked) < n and i < limit:
        cand = pool[(h + i) % len(pool)]
        i += 1
        if cand not in exclude and cand not in picked:
            picked.append(cand)
    return picked


def random_pick(seed, pool, exclude, n):
    """Random sampling with seed for reproducibility but natural distribution."""
    if not pool or n <= 0:
        return []
    available = [x for x in pool if x not in exclude]
    rng = random.Random(seed)
    if len(available) <= n:
        return rng.sample(available, len(available))
    return rng.sample(available, n)


def extract_doc(pid):
    """Best-effort doc extraction from passage_id."""
    m = re.match(r"(.+?)_(?:chunk|paragraph|figure|table|formula|text)_", pid)
    return m.group(1) if m else ""


def is_text_type(pid, corpus_ids_info):
    """Check if passage_id is chunk or paragraph type."""
    info = corpus_ids_info.get(pid)
    if info is None:
        # Try from pid pattern
        if "_chunk_" in pid or "_paragraph_" in pid or "_bridge" in pid:
            return True
        return False
    return info in ("chunk", "paragraph")


def main():
    print("=" * 60)
    print("M4query_v2_clean_chunk_aug builder")
    print("=" * 60)

    corpus = load_jsonl(SRC_CLEAN / "corpus.jsonl")
    triplets = load_jsonl(SRC_CLEAN / "train_triplets.jsonl")
    mapping = json.load(open(MAPPING_PATH))
    print(f"  loaded corpus={len(corpus)}  triplets={len(triplets)}  "
          f"bridge_mapping={len(mapping)}")

    # Index by id
    cmap = {r["passage_id"]: r for r in corpus}
    bridge_ids = {pid for pid, r in cmap.items()
                  if r["type"] == "text" and pid.endswith("_bridge")}
    print(f"  bridges in corpus: {len(bridge_ids)}")

    # Classify bridges
    bridge_to_paragraph = {}    # bridge_id → source_paragraph_id (delete bridge)
    bridge_unmapped = set()      # no mapping → keep & retype
    bridge_visual_source = {}    # bridge_id → source_id (f/t/f) → keep & retype

    for bid in bridge_ids:
        m = mapping.get(bid)
        if m is None:
            bridge_unmapped.add(bid)
        elif m["source_type"] == "paragraph":
            bridge_to_paragraph[bid] = m["source_passage_id"]
        else:
            bridge_visual_source[bid] = m["source_passage_id"]

    print(f"  bridge → paragraph source: {len(bridge_to_paragraph)}")
    print(f"  bridge → visual source (f/t/f): {len(bridge_visual_source)}")
    print(f"  bridge unmapped: {len(bridge_unmapped)}")
    bridges_to_delete = set(bridge_to_paragraph.keys())
    bridges_to_keep = bridge_unmapped | set(bridge_visual_source.keys())
    assert bridges_to_delete | bridges_to_keep == bridge_ids
    assert len(bridges_to_delete & bridges_to_keep) == 0

    # ── Phase 1: build paragraph → chunk mapping ──────────────────────
    paras_by_doc = collections.defaultdict(list)
    for r in corpus:
        if r["type"] != "paragraph":
            continue
        doc, idx = doc_id_of_paragraph(r["passage_id"])
        if doc is None:
            continue
        paras_by_doc[doc].append((idx, r["passage_id"], r["text"]))
    for d in paras_by_doc:
        paras_by_doc[d].sort()
    print(f"  docs with paragraphs: {len(paras_by_doc)}")

    pid_to_chunk = {}
    chunks = []
    stats = collections.Counter()

    for doc, paras in paras_by_doc.items():
        heads = load_section_heads(doc)
        groups = collections.defaultdict(list)
        for idx, pid, text in paras:
            groups[section_of(idx, heads)].append((idx, pid, text))

        chunk_idx = 0
        for sec_head in sorted(groups):
            items = sorted(groups[sec_head])
            cur_items = []
            cur_words = 0

            def flush_chunk():
                nonlocal cur_items, cur_words, chunk_idx
                if not cur_items:
                    return
                chunk_id = f"{doc}_chunk_{chunk_idx}"
                chunk_idx += 1
                pids = [it[1] for it in cur_items]
                text = "\n\n".join(it[2] for it in cur_items)
                chunks.append({
                    "passage_id": chunk_id,
                    "type": "chunk",
                    "text": text,
                    "caption": "",
                    "image_path": "",
                    "description": "",
                })
                for pid in pids:
                    pid_to_chunk[pid] = chunk_id
                stats["chunks_built"] += 1
                cur_items = []
                cur_words = 0

            for idx, pid, text in items:
                w = word_count(text)
                if cur_items and cur_words + w > CHUNK_SIZE:
                    flush_chunk()
                cur_items.append((idx, pid, text))
                cur_words += w
            flush_chunk()

    print(f"  chunks built: {len(chunks)}")
    print(f"  paragraphs mapped: {len(pid_to_chunk)}")

    # Verify: all paragraphs in clean map to chunks
    para_ids_all = {r["passage_id"] for r in corpus if r["type"] == "paragraph"}
    unmapped_paras = para_ids_all - set(pid_to_chunk)
    assert not unmapped_paras, f"{len(unmapped_paras)} paragraphs unmapped"
    print(f"  all paragraphs mapped: OK")

    # ── Phase 2: build new corpus ─────────────────────────────────────
    new_corpus = []
    for r in corpus:
        pid = r["passage_id"]
        if pid in bridges_to_delete:
            stats["dropped_bridge_to_paragraph"] += 1
            continue
        if pid in bridges_to_keep:
            r_copy = dict(r)
            r_copy["type"] = "paragraph"
            new_corpus.append(r_copy)
            stats["kept_bridge_retyped"] += 1
            continue
        # Keep existing non-bridge passages as-is
        new_corpus.append(r)

    # Add chunks
    new_corpus.extend(chunks)
    print(f"\n── corpus ──")
    print(f"  kept non-bridge: {len(new_corpus) - len(chunks) - stats['kept_bridge_retyped']}")
    print(f"  kept bridge → paragraph: {stats['kept_bridge_retyped']}")
    print(f"  dropped bridge: {stats['dropped_bridge_to_paragraph']}")
    print(f"  added chunks: {len(chunks)}")
    print(f"  total: {len(new_corpus)}")

    new_corpus_ids = {r["passage_id"] for r in new_corpus}
    # Build type info for all passages in new corpus
    pid_type = {r["passage_id"]: r["type"] for r in new_corpus}

    # ── Phase 3: build triplet positives ──────────────────────────────
    print(f"\n── triplet positives ──")

    # Pools for refilling positives
    all_passage_ids = list(pid_type.keys())
    text_passage_ids = [pid for pid, t in pid_type.items()
                        if t in ("paragraph", "chunk")]

    # Stage 1: compute positive mapping per query
    query_pos_map = {}   # qid → list of positive ids (may have 3 or 4)

    for t in triplets:
        qid = t["query_id"]
        new_pos = []
        for p in t["positive_passages"]:
            if p in bridges_to_delete:
                # Bridge maps to paragraph — replace with source paragraph + chunk
                src_para = bridge_to_paragraph[p]
                chunk_id = pid_to_chunk.get(src_para)
                new_pos.append(src_para)
                if chunk_id:
                    new_pos.append(chunk_id)
            elif p in bridges_to_keep:
                # Bridge kept — will be in corpus as "paragraph"
                new_pos.append(p)
            else:
                new_pos.append(p)

        # Dedupe while preserving order
        unique_pos = list(dict.fromkeys(new_pos))
        if len(unique_pos) < POSITIVE_TARGET:
            # Fill to 3 with same-doc non-visual passages
            qdoc = extract_doc(qid)
            need = POSITIVE_TARGET - len(unique_pos)
            pool = [pid for pid in text_passage_ids
                    if pid not in unique_pos and extract_doc(pid) == qdoc]
            add = det_pick(qid + "_pos", pool, set(unique_pos), need)
            unique_pos += add
            if len(unique_pos) < POSITIVE_TARGET:
                pool2 = [pid for pid in text_passage_ids
                         if pid not in unique_pos]
                add2 = det_pick(qid + "_pos2", pool2, set(unique_pos),
                                POSITIVE_TARGET - len(unique_pos))
                unique_pos += add2

        # Truncate to 3 or keep 4
        # If we have 4 (from bridge→source+chunk), keep all 4
        if len(unique_pos) >= POSITIVE_TARGET:
            query_pos_map[qid] = unique_pos
        else:
            stats["dropped_pos_short"] += 1

    n4 = sum(1 for v in query_pos_map.values() if len(v) == 4)
    n3 = sum(1 for v in query_pos_map.values() if len(v) == 3)
    print(f"  queries with 4 positives: {n4}")
    print(f"  queries with 3 positives: {n3}")
    print(f"  total queries: {len(query_pos_map)}")

    # ── Phase 4: build negatives with text slot reservation ────────────
    print(f"\n── triplet negatives ──")

    # Compute global banned set from ALL final positives
    global_positive_ids = set()
    for ids in query_pos_map.values():
        global_positive_ids.update(ids)
    banned_neg = set(global_positive_ids)

    # Pools for negative sampling
    text_pool = [pid for pid, t in pid_type.items()
                 if t in ("paragraph", "chunk") and pid not in banned_neg]
    visual_pool = [pid for pid, t in pid_type.items()
                   if t in ("figure", "table", "formula") and pid not in banned_neg]
    all_neg_pool = text_pool + visual_pool

    # Doc-level pools
    doc_text_pool = collections.defaultdict(list)
    doc_visual_pool = collections.defaultdict(list)
    for pid in text_pool:
        doc_text_pool[extract_doc(pid)].append(pid)
    for pid in visual_pool:
        doc_visual_pool[extract_doc(pid)].append(pid)

    clean_triplets = []
    neg_stats = collections.Counter()

    for t in triplets:
        qid = t["query_id"]
        if qid not in query_pos_map:
            continue

        pos_list = query_pos_map[qid]
        pos_set = set(pos_list)
        qdoc = extract_doc(qid)

        # --- hard negatives ---
        hard_orig = []
        seen_hard = set()
        for p in t["hard_negative_passages"]:
            if p in bridges_to_delete:
                src_para = bridge_to_paragraph[p]
                chunk_id = pid_to_chunk.get(src_para)
                if chunk_id and chunk_id not in banned_neg and chunk_id not in seen_hard:
                    hard_orig.append(chunk_id)
                    seen_hard.add(chunk_id)
                continue
            if p in bridges_to_keep:
                if p not in banned_neg and p not in seen_hard:
                    hard_orig.append(p)
                    seen_hard.add(p)
                continue
            if p in pid_to_chunk:
                cid = pid_to_chunk[p]
                if cid not in banned_neg and cid not in seen_hard:
                    hard_orig.append(cid)
                    seen_hard.add(cid)
                continue
            if p in new_corpus_ids and p not in banned_neg and p not in seen_hard:
                hard_orig.append(p)
                seen_hard.add(p)

        # Build hard_neg with 2-3 text slots (target per group, biased toward 3)
        hard_text = [p for p in hard_orig if is_text_type(p, pid_type)]
        hard_visual = [p for p in hard_orig if not is_text_type(p, pid_type)]

        hard = []
        used = pos_set.copy()
        rng = random.Random(qid + "_hard")
        text_target = rng.choice([2, 3, 3, 3])  # 75% chance of 3, 25% of 2

        # Pick text from originals
        n_text_pick = min(len(hard_text), text_target)
        hard.extend(rng.sample(hard_text, n_text_pick) if len(hard_text) > n_text_pick else hard_text[:n_text_pick])
        used.update(hard)

        # Pick visual from originals, leaving room for text target
        text_shortfall = max(0, text_target - sum(1 for p in hard if is_text_type(p, pid_type)))
        max_visual = max(0, HARD_NEG_TARGET - len(hard) - text_shortfall)
        n_visual_pick = min(len(hard_visual), max_visual)
        hard.extend(rng.sample(hard_visual, n_visual_pick) if len(hard_visual) > n_visual_pick else hard_visual[:n_visual_pick])
        used.update(hard)

        # Refill text to meet target
        text_needed = max(0, text_target - sum(1 for p in hard if is_text_type(p, pid_type)))
        if text_needed > 0:
            add = random_pick(qid + "_ht_fill", text_pool, used, text_needed)
            hard += add
            used |= set(add)

        # Fill remaining slots with visual
        remaining = HARD_NEG_TARGET - len(hard)
        if remaining > 0:
            add = random_pick(qid + "_hv_fill", visual_pool, used, remaining)
            hard += add
            used |= set(add)

        hard = hard[:HARD_NEG_TARGET]

        # --- random negatives ---
        rand_orig = []
        seen_rand = set()
        for p in t["random_negative_passages"]:
            if p in bridges_to_delete:
                src_para = bridge_to_paragraph[p]
                chunk_id = pid_to_chunk.get(src_para)
                if chunk_id and chunk_id not in banned_neg and chunk_id not in seen_rand:
                    rand_orig.append(chunk_id)
                    seen_rand.add(chunk_id)
                continue
            if p in bridges_to_keep:
                if p not in banned_neg and p not in seen_rand:
                    rand_orig.append(p)
                    seen_rand.add(p)
                continue
            if p in pid_to_chunk:
                cid = pid_to_chunk[p]
                if cid not in banned_neg and cid not in seen_rand:
                    rand_orig.append(cid)
                    seen_rand.add(cid)
                continue
            if p in new_corpus_ids and p not in banned_neg and p not in seen_rand:
                rand_orig.append(p)
                seen_rand.add(p)

        rand_text = [p for p in rand_orig if is_text_type(p, pid_type)]
        rand_visual = [p for p in rand_orig if not is_text_type(p, pid_type)]

        rand = []
        rng2 = random.Random(qid + "_rand")
        text_target = rng2.choice([2, 3, 3, 3])

        n_text_keep = min(len(rand_text), text_target)
        rand.extend(rng2.sample(rand_text, n_text_keep) if len(rand_text) > n_text_keep else rand_text[:n_text_keep])
        used.update(rand)

        text_shortfall = max(0, text_target - sum(1 for p in rand if is_text_type(p, pid_type)))
        max_visual = max(0, RANDOM_NEG_TARGET - len(rand) - text_shortfall)
        n_visual_keep = min(len(rand_visual), max_visual)
        rand.extend(rng2.sample(rand_visual, n_visual_keep) if len(rand_visual) > n_visual_keep else rand_visual[:n_visual_keep])
        used.update(rand)

        text_needed = max(0, text_target - sum(1 for p in rand if is_text_type(p, pid_type)))
        if text_needed > 0:
            add = random_pick(qid + "_rt_fill", text_pool, used, text_needed)
            rand += add
            used |= set(add)

        remaining = RANDOM_NEG_TARGET - len(rand)
        if remaining > 0:
            add = random_pick(qid + "_rv_fill", visual_pool, used, remaining)
            rand += add
            used |= set(add)

        rand = rand[:RANDOM_NEG_TARGET]

        if len(pos_list) < POSITIVE_TARGET or len(hard) != HARD_NEG_TARGET or len(rand) != RANDOM_NEG_TARGET:
            neg_stats["dropped_short"] += 1
            continue

        if set(pos_list) & (set(hard) | set(rand)):
            neg_stats["pos_neg_overlap"] += 1
            continue

        clean_triplets.append({
            "query_id": qid,
            "query": t["query"],
            "positive_passages": pos_list,
            "hard_negative_passages": hard,
            "random_negative_passages": rand,
        })

    print(f"  final triplets: {len(clean_triplets)}")

    # ── Phase 5: statistics & validation ───────────────────────────────
    print(f"\n── validation ──")
    bad_struct = bad_ref = pos_neg_overlap = intrade_dup = 0
    text_in_hard = []
    text_in_rand = []

    for t in clean_triplets:
        ps = t["positive_passages"]
        hn = t["hard_negative_passages"]
        rn = t["random_negative_passages"]
        if len(hn) != HARD_NEG_TARGET or len(rn) != RANDOM_NEG_TARGET or len(ps) < POSITIVE_TARGET:
            bad_struct += 1
        for x in ps + hn + rn:
            if x not in new_corpus_ids:
                bad_ref += 1
        if set(ps) & (set(hn) | set(rn)):
            pos_neg_overlap += 1
        if len(set(hn)) != len(hn) or len(set(rn)) != len(rn):
            intrade_dup += 1
        text_in_hard.append(sum(1 for p in hn if is_text_type(p, pid_type)))
        text_in_rand.append(sum(1 for p in rn if is_text_type(p, pid_type)))

    print(f"  bad_structure: {bad_struct}")
    print(f"  bad_reference: {bad_ref}")
    print(f"  pos_neg_overlap: {pos_neg_overlap}")
    print(f"  intra_group_dup: {intrade_dup}")

    from collections import Counter
    hard_text_dist = Counter(text_in_hard)
    rand_text_dist = Counter(text_in_rand)
    print(f"  text slots in hard_neg distribution: {dict(sorted(hard_text_dist.items()))}")
    print(f"  text slots in rand_neg distribution: {dict(sorted(rand_text_dist.items()))}")

    # Overall negative type distribution
    all_neg_types = Counter()
    for t in clean_triplets:
        for p in t["hard_negative_passages"] + t["random_negative_passages"]:
            all_neg_types[pid_type.get(p, "MISSING")] += 1
    print(f"  negative type distribution:")
    for tp, c in all_neg_types.most_common():
        print(f"    {tp}: {c} ({c/sum(all_neg_types.values())*100:.1f}%)")

    # Corpus type distribution
    corpus_types = Counter(r["type"] for r in new_corpus)
    print(f"\n  corpus type distribution:")
    for tp, c in corpus_types.most_common():
        print(f"    {tp}: {c}")

    if bad_struct or bad_ref or pos_neg_overlap:
        print(f"\n  VALIDATION FAILED")
        sys.exit(1)

    # ── Phase 6: write outputs ────────────────────────────────────────
    OUT.mkdir(parents=True, exist_ok=True)
    # Strip internal fields for clean output
    clean_corpus_out = []
    for r in new_corpus:
        clean_corpus_out.append({
            "passage_id": r["passage_id"],
            "type": r["type"],
            "text": r.get("text", ""),
            "caption": r.get("caption", ""),
            "image_path": r.get("image_path", ""),
            "description": r.get("description", ""),
        })

    write_jsonl(OUT / "corpus.jsonl", clean_corpus_out)
    write_jsonl(OUT / "train_triplets.jsonl", clean_triplets)
    print(f"\nwrote corpus ({len(clean_corpus_out)}) + train_triplets ({len(clean_triplets)}) to {OUT}")

    # ── Phase 7: README ───────────────────────────────────────────────
    pos_len_dist = Counter(len(t["positive_passages"]) for t in clean_triplets)
    readme = f"""# M4query_v2_clean_chunk_aug

M4query_v2_clean 的 chunk 增强版本。保留 paragraph 和 bridge(改类型为 paragraph)，同时增加 chunk 作为额外粒度。

## 文件

| 路径 | 数量 | 作用 |
| --- | ---: | --- |
| `corpus.jsonl` | {len(new_corpus)} | figure / table / formula / paragraph / chunk。 |
| `train_triplets.jsonl` | {len(clean_triplets)} | 3-4 positive + 5 hard_neg + 5 random_neg。 |

## corpus type 分布

| type | 数量 |
| --- | ---: |
""" + "\n".join(f"| `{k}` | {v} |" for k, v in corpus_types.most_common()) + f"""

## positive 数量分布

| 数量 | query 数 |
| --- | ---: |
""" + "\n".join(f"| {k} | {v} |" for k, v in sorted(pos_len_dist.items())) + f"""

## negative 文本占比

hard_neg 和 random_neg 各保留 {TEXT_SLOTS_MIN}-{TEXT_SLOTS_MAX} 个文本类(chunk/paragraph)名额。

| neg 组 | 文本名额 |
| --- | --- |
| hard_neg | {TEXT_SLOTS_MIN}-{TEXT_SLOTS_MAX} |
| random_neg | {TEXT_SLOTS_MIN}-{TEXT_SLOTS_MAX} |

## 与 M4query_v2_clean 的差异

- **bridge 处理**: 找到 paragraph source → 删除 bridge，替换为 source paragraph + chunk; 找不到 source 或 source 为 f/t/f → 保留 bridge，type 改为 `paragraph`(共 {stats['kept_bridge_retyped']} 条)。
- **新增 chunk**: {len(chunks)} 条，由 paragraph 按 section 边界 + ~{CHUNK_SIZE} 词聚合而成。
- **paragraph 不删**: clean 原 116,092 paragraph 全部保留。
- **multi-positive**: bridge→paragraph 的 query 获 4 positive(source paragraph + chunk + 2 original)。
- **negative 重平衡**: visual 占比从 72% 降至 ~{100 - sum(1 for p,t in pid_type.items() if t in ('paragraph','chunk'))/sum(1 for _ in pid_type)*100:.0f}%。

## 生成脚本

`scripts/build_clean_chunk_aug.py`
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print("wrote README.md")
    print("\nDONE")


if __name__ == "__main__":
    main()
