#!/usr/bin/env python3
"""Method C 打包(pilot)：把 query 的 positive 从 element 扩到 chunk(elem→chunk 映射)。

对齐 M4query_v2_clean_chunk_aug 的思路(简化版，用于 pilot 验证)：
  - positive = 证据 element + 含该 element 的 chunk(s) + 桥接段落(若有)
  - corpus 含 element / chunk / paragraph 三种粒度
  - elem→chunk 映射来自 hierarchical chunks 的 chunk.element_ids(chunk↔element 边)
"""
import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="L3 pass queries jsonl")
    ap.add_argument("--chunks", required=True, help="hierarchical chunks (enriched) json")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    chunks = json.load(open(a.chunks, encoding="utf-8"))
    # element_id → [chunk_id]（细 chunk 的 chunk↔element 边）
    elem2chunk = defaultdict(list)
    chunk_text = {}
    for doc in chunks["documents"].values():
        for c in doc.get("chunks", []):
            cid = c["chunk_id"]
            chunk_text[cid] = c.get("text_augmented") or c["text"]
            for eid in c.get("element_ids", []):
                elem2chunk[eid].append(cid)

    rows = [json.loads(l) for l in open(a.queries, encoding="utf-8")]
    out = []
    n_with_chunk = total_chunk_pos = 0
    for r in rows:
        eids = r.get("element_ids", []) or []
        pos_chunks = []
        for eid in eids:
            for cid in elem2chunk.get(eid, []):
                if cid not in pos_chunks:
                    pos_chunks.append(cid)
        rec = {
            "query_id": r.get("query_id"),
            "query": r.get("query"),
            "difficulty_level": r.get("difficulty_level"),
            "positive_elements": eids,                       # 原 element 正例
            "positive_chunks": pos_chunks,                   # elem→chunk 映射的 chunk 正例
            "bridge_quality": r.get("bridge_quality"),       # 桥(L3 才有)
            "n_positive": len(eids) + len(pos_chunks),
        }
        out.append(rec)
        if pos_chunks:
            n_with_chunk += 1
            total_chunk_pos += len(pos_chunks)

    meta = {
        "queries": len(rows),
        "queries_with_chunk_positive": n_with_chunk,
        "total_chunk_positives": total_chunk_pos,
        "elem2chunk_index_size": len(elem2chunk),
    }
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "records": out}, f, ensure_ascii=False, indent=2)
    print("Done:", json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
