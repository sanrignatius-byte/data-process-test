#!/usr/bin/env python3
"""从 MinerU markdown 正文切 paragraph chunk，注入引用到的 multimodal element_ids。

chunk 来源是 MinerU 的干净正文（含公式/表格文本），按 ~target 词贪心聚合段落成 chunk。
element_ids 用 best-effort 正则匹配正文中的 "Figure N / Table N / Eq. N" → {doc}_{type}_{n}，
仅当该 id 在 multimodal_elements 中存在时才关联（不保证位置级精确，供图增强检索用）。
"""
import argparse
import glob
import json
import os
import re

WORD_RE = re.compile(r"\S+")
REF_RE = re.compile(r"\b(figure|fig|table|tab|equation|eq)\b\.?\s*\(?(\d+)\)?", re.I)
TYPE_MAP = {"figure": "figure", "fig": "figure", "table": "table",
            "tab": "table", "equation": "formula", "eq": "formula"}
IMG_LINE = re.compile(r"^!\[\]\(")


def word_count(t: str) -> int:
    return len(WORD_RE.findall(t))


def split_chunks(md: str, target: int, min_words: int):
    paras = [p.strip() for p in md.split("\n\n") if p.strip()]
    chunks, cur, cw = [], [], 0
    for p in paras:
        if IMG_LINE.match(p):           # 跳过纯图片行
            continue
        cur.append(p)
        cw += word_count(p)
        if cw >= target:
            chunks.append("\n\n".join(cur))
            cur, cw = [], 0
    if cur:
        if cw >= min_words or not chunks:
            chunks.append("\n\n".join(cur))
        else:
            chunks[-1] += "\n\n" + "\n\n".join(cur)   # 残余并入上一块
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mineru-dir", required=True)
    ap.add_argument("--elements", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--chunk-size", type=int, default=400)
    ap.add_argument("--min-words", type=int, default=20)
    a = ap.parse_args()

    el = json.load(open(a.elements, encoding="utf-8"))
    doc_elems = {d: set(v.get("elements", {}).keys())
                 for d, v in el.get("documents", {}).items()}

    out = {"metadata": {"source": a.mineru_dir, "target_words": a.chunk_size,
                        "min_words": a.min_words, "script": "scripts/build_md_chunks.py"},
           "documents": {}, "stats": {}}
    tot_docs = tot_chunks = tot_words = tot_elem = docs_wc = 0

    for md_path in sorted(glob.glob(f"{a.mineru_dir}/*/vlm/*.md")):
        doc_id = os.path.basename(os.path.dirname(os.path.dirname(md_path)))
        md = open(md_path, encoding="utf-8").read()
        chunks = split_chunks(md, a.chunk_size, a.min_words)
        tot_docs += 1
        if not chunks:
            continue
        docs_wc += 1
        elemset = doc_elems.get(doc_id, set())
        clist = []
        for i, ct in enumerate(chunks):
            wc = word_count(ct)
            eids = []
            for m in REF_RE.finditer(ct):
                t = TYPE_MAP.get(m.group(1).lower())
                cand = f"{doc_id}_{t}_{m.group(2)}"
                if t and cand in elemset and cand not in eids:
                    eids.append(cand)
            clist.append({"chunk_id": f"{doc_id}_chunk_{i}", "chunk_idx": i,
                          "text": ct, "word_count": wc, "element_ids": eids})
            tot_chunks += 1
            tot_words += wc
            tot_elem += len(eids)
        out["documents"][doc_id] = {"chunks": clist}

    out["stats"] = {"total_docs": tot_docs, "docs_with_chunks": docs_wc,
                    "total_chunks": tot_chunks, "total_words": tot_words,
                    "avg_words_per_chunk": round(tot_words / max(tot_chunks, 1), 1),
                    "total_elements_assigned": tot_elem}
    json.dump(out, open(a.output, "w", encoding="utf-8"), ensure_ascii=False)
    print("Done:", json.dumps(out["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
