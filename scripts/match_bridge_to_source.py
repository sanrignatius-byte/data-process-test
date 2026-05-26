#!/usr/bin/env python3
"""Map each `_bridge` passage in M4query_v2_clean back to its source passage
(paragraph / figure / table / formula) in the same document.

Pipeline:
  1. Literal 3-key substring match on normalized text across all fields
     (text / caption / description) of every non-bridge passage in the doc.
  2. TF-IDF cosine similarity for whatever literal match missed.
  3. Drop the bridge_id if even the best TF-IDF score is below threshold.

Output: data/03_queries/M4query_v2_clean/bridge_to_source.json
    { bridge_passage_id: {source_passage_id, source_type, source_field, method, score} }
"""

import json
import re
import collections
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path("/projects/myyyx1/data-process-test")
CORPUS = ROOT / "data/03_queries/M4query_v2_clean/corpus.jsonl"
OUT = ROOT / "data/03_queries/M4query_v2_clean/bridge_to_source.json"

TFIDF_THRESHOLD = 0.35   # below this we drop


def norm(s: str) -> str:
    s = re.sub(r"~?\[[^\]]*\]", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = s.replace("$", " ")
    s = re.sub(r"[^a-zA-Z0-9]+", "", s).lower()
    return s


def norm_words(s: str) -> str:
    """Lighter normalization for TF-IDF (keep word boundaries)."""
    s = re.sub(r"~?\[[^\]]*\]", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = s.replace("$", " ").replace("{", " ").replace("}", " ")
    s = re.sub(r"[^a-zA-Z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def doc_id_of(pid: str) -> str:
    m = re.match(r"l3_de_(.+?)_\d+_bridge$", pid)
    if m:
        return m.group(1)
    m = re.match(r"(.+?)_(?:figure|table|formula|paragraph)_", pid)
    if m:
        return m.group(1)
    return pid.rsplit("_", 1)[0]


def literal_match(nbr: str, sources):
    """3-key substring match in any field of any source passage.

    sources: list of (passage_id, type, field, norm_value)
    returns (passage_id, type, field) or None
    """
    if len(nbr) < 30:
        return None
    keys = [nbr[:60]]
    if len(nbr) > 60:
        keys.append(nbr[len(nbr)//2 - 30:len(nbr)//2 + 30])
        keys.append(nbr[-60:])
    keys = [k for k in keys if k and len(k) >= 30]
    for pid, ty, fld, nv in sources:
        if any(k in nv for k in keys):
            return (pid, ty, fld)
    return None


def tfidf_match(bridge_text: str, candidates):
    """Cosine similarity over TF-IDF. Returns (idx, score) of best candidate.

    candidates: list of strings (already normalized to words).
    """
    if not candidates:
        return None
    texts = [norm_words(bridge_text)] + candidates
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0,
                              token_pattern=r"\b\w+\b")
        mat = vec.fit_transform(texts)
    except ValueError:
        return None
    sims = cosine_similarity(mat[0:1], mat[1:])[0]
    if len(sims) == 0:
        return None
    best = int(sims.argmax())
    return (best, float(sims[best]))


def main():
    print("loading corpus...")
    rows = [json.loads(l) for l in open(CORPUS)]
    print(f"  total passages: {len(rows)}")

    # Index sources by doc (everything that's NOT a _bridge)
    sources_by_doc = collections.defaultdict(list)
    candidates_by_doc = collections.defaultdict(list)   # for TF-IDF: (pid, ty, fld, words_text)
    for r in rows:
        if r["passage_id"].endswith("_bridge"):
            continue
        doc = doc_id_of(r["passage_id"])
        for fld in ("text", "caption", "description"):
            v = r.get(fld, "")
            if not v:
                continue
            nv = norm(v)
            if len(nv) >= 20:
                sources_by_doc[doc].append((r["passage_id"], r["type"], fld, nv))
            cw = norm_words(v)
            if len(cw) >= 15:
                candidates_by_doc[doc].append((r["passage_id"], r["type"], fld, cw))

    bridges = [r for r in rows if r["passage_id"].endswith("_bridge")]
    print(f"  bridges to match: {len(bridges)}")
    print(f"  source docs covered: {len(sources_by_doc)}")

    # Phase 1: literal 3-key match
    t0 = time.time()
    mapping = {}
    lit_hit = 0
    need_tfidf = []
    for br in bridges:
        bpid = br["passage_id"]
        doc = doc_id_of(bpid)
        nbr = norm(br["text"])
        srcs = sources_by_doc.get(doc, [])
        hit = literal_match(nbr, srcs)
        if hit:
            mapping[bpid] = {
                "source_passage_id": hit[0],
                "source_type": hit[1],
                "source_field": hit[2],
                "method": "literal",
                "score": 1.0,
            }
            lit_hit += 1
        else:
            need_tfidf.append(br)
    print(f"\nphase 1 (literal): {lit_hit}/{len(bridges)} "
          f"({100*lit_hit//len(bridges)}%) in {time.time()-t0:.1f}s")
    print(f"  remaining for TF-IDF: {len(need_tfidf)}")

    # Phase 2: TF-IDF cosine on the remainder
    t0 = time.time()
    tf_hit = 0
    tf_drop = 0
    score_dist = []
    miss_samples = []
    for br in need_tfidf:
        bpid = br["passage_id"]
        doc = doc_id_of(bpid)
        cands = candidates_by_doc.get(doc, [])
        if not cands:
            tf_drop += 1
            continue
        cand_texts = [c[3] for c in cands]
        res = tfidf_match(br["text"], cand_texts)
        if res is None:
            tf_drop += 1
            continue
        idx, score = res
        score_dist.append(score)
        if score >= TFIDF_THRESHOLD:
            pid, ty, fld, _ = cands[idx]
            mapping[bpid] = {
                "source_passage_id": pid,
                "source_type": ty,
                "source_field": fld,
                "method": "tfidf",
                "score": round(score, 4),
            }
            tf_hit += 1
        else:
            tf_drop += 1
            if len(miss_samples) < 5:
                miss_samples.append((bpid, score, br["text"][:80]))

    print(f"\nphase 2 (TF-IDF, threshold {TFIDF_THRESHOLD}): "
          f"{tf_hit}/{len(need_tfidf)} matched, {tf_drop} below threshold "
          f"in {time.time()-t0:.1f}s")
    if score_dist:
        score_dist.sort()
        n = len(score_dist)
        print(f"  TF-IDF score distribution: min={score_dist[0]:.3f} "
              f"p25={score_dist[n//4]:.3f} med={score_dist[n//2]:.3f} "
              f"p75={score_dist[3*n//4]:.3f} max={score_dist[-1]:.3f}")
    if miss_samples:
        print("  below-threshold samples:")
        for s in miss_samples:
            print(f"    {s[0]} score={s[1]:.3f} bridge={s[2]!r}")

    # Summary
    total = lit_hit + tf_hit
    print(f"\n===== FINAL =====")
    print(f"  total matched: {total}/{len(bridges)} "
          f"({100*total//len(bridges)}%)")
    print(f"  unmatched (will drop the query): {len(bridges)-total}")

    # Breakdown by source type
    type_count = collections.Counter(v["source_type"] for v in mapping.values())
    method_count = collections.Counter(v["method"] for v in mapping.values())
    print(f"  source type distribution: {dict(type_count)}")
    print(f"  method distribution: {dict(method_count)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
