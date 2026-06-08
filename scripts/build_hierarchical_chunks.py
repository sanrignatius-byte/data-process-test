#!/usr/bin/env python3
"""Doc-Researcher 风格的粗+细两级 chunk。

从 MinerU markdown 正文按 # 标题切「节级 coarse 单元」，节内再按 ~target 词切
「fine chunk」（沿用 build_md_chunks 的贪心逻辑），coarse→fine 建 parent→child。

- coarse 节点：整节文本（标题+正文），含 section_number/level（从标题编号推断）、
  child_chunk_ids、聚合的 element_ids。供粗粒度召回。
- fine 节点：节内 400 词块，含 parent_section_id、element_ids。供细粒度定位。

element_ids 用与 build_md_chunks 相同的 best-effort 正则注入（仅保留库中存在的）。
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_md_chunks import word_count, IMG_LINE, REF_RE, TYPE_MAP

HEADING_RE = re.compile(r"^#{1,6}\s+(.*\S)\s*$")
# 节号形如 "3", "3.2", "A.1" —— 取标题开头的编号 token
SECNUM_RE = re.compile(r"^([A-Z]?\d+(?:\.\d+)*)\b")


def parse_section_number(title: str):
    """从标题提取节号和层级；无编号(Abstract/Conclusions等)→ level 1, num=None。"""
    m = SECNUM_RE.match(title)
    if not m:
        return None, 1
    num = m.group(1)
    # 层级 = 编号中的段数 (3 → L1, 3.2 → L2, A.1 → L2)
    level = num.replace("A", "").strip(".").count(".") + 1 if any(c.isdigit() for c in num) else 1
    return num, level


def split_into_sections(md: str):
    """切成 [(title, body_text)]；首个标题前的内容归入 preamble。"""
    lines = md.split("\n")
    sections = []
    cur_title, cur_body = None, []
    for ln in lines:
        h = HEADING_RE.match(ln)
        if h:
            if cur_title is not None or cur_body:
                sections.append((cur_title, "\n".join(cur_body).strip()))
            cur_title, cur_body = h.group(1), []
        else:
            cur_body.append(ln)
    if cur_title is not None or cur_body:
        sections.append((cur_title, "\n".join(cur_body).strip()))
    return sections


def split_chunks_in_section(body: str, target: int, min_words: int):
    """节内按 ~target 词贪心切块（残余并入上一块）。返回 chunk 文本列表。"""
    paras = [p.strip() for p in body.split("\n\n") if p.strip() and not IMG_LINE.match(p.strip())]
    chunks, cur, cw = [], [], 0
    for p in paras:
        cur.append(p)
        cw += word_count(p)
        if cw >= target:
            chunks.append("\n\n".join(cur))
            cur, cw = [], 0
    if cur:
        if cw >= min_words or not chunks:
            chunks.append("\n\n".join(cur))
        else:
            chunks[-1] += "\n\n" + "\n\n".join(cur)
    return chunks


def extract_element_ids(text: str, doc_id: str, elemset: set):
    eids = []
    for m in REF_RE.finditer(text):
        t = TYPE_MAP.get(m.group(1).lower())
        cand = f"{doc_id}_{t}_{m.group(2)}"
        if t and cand in elemset and cand not in eids:
            eids.append(cand)
    return eids


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
                        "min_words": a.min_words, "script": "scripts/build_hierarchical_chunks.py",
                        "granularity": "coarse=section / fine=paragraph"},
           "documents": {}, "stats": {}}
    tot_docs = tot_sec = tot_chunk = tot_words = tot_elem = 0

    for md_path in sorted(glob.glob(f"{a.mineru_dir}/*/vlm/*.md")):
        doc_id = os.path.basename(os.path.dirname(os.path.dirname(md_path)))
        md = open(md_path, encoding="utf-8").read()
        elemset = doc_elems.get(doc_id, set())
        sections, chunks = [], []
        tot_docs += 1
        for si, (title, body) in enumerate(split_into_sections(md)):
            sec_id = f"{doc_id}_sec_{si}"
            num, level = parse_section_number(title or "")
            full_text = ((title + "\n\n") if title else "") + body
            child_ids, sec_eids = [], []
            for ci, ct in enumerate(split_chunks_in_section(body, a.chunk_size, a.min_words)):
                ch_id = f"{sec_id}_chunk_{ci}"
                eids = extract_element_ids(ct, doc_id, elemset)
                wc = word_count(ct)
                chunks.append({"chunk_id": ch_id, "parent_section_id": sec_id,
                               "section_idx": si, "chunk_idx": ci, "text": ct,
                               "word_count": wc, "element_ids": eids})
                child_ids.append(ch_id)
                for e in eids:
                    if e not in sec_eids:
                        sec_eids.append(e)
                tot_chunk += 1
                tot_words += wc
                tot_elem += len(eids)
            # 标题里直接引用的元素也算进 coarse 层
            for e in extract_element_ids(title or "", doc_id, elemset):
                if e not in sec_eids:
                    sec_eids.append(e)
            sections.append({"section_id": sec_id, "section_idx": si,
                             "title": title, "section_number": num, "level": level,
                             "text": full_text, "word_count": word_count(full_text),
                             "child_chunk_ids": child_ids, "element_ids": sec_eids})
            tot_sec += 1
        out["documents"][doc_id] = {"sections": sections, "chunks": chunks}

    out["stats"] = {"total_docs": tot_docs, "total_sections": tot_sec,
                    "total_fine_chunks": tot_chunk, "total_words": tot_words,
                    "avg_chunks_per_section": round(tot_chunk / max(tot_sec, 1), 2),
                    "avg_words_per_chunk": round(tot_words / max(tot_chunk, 1), 1),
                    "total_elements_assigned": tot_elem}
    json.dump(out, open(a.output, "w", encoding="utf-8"), ensure_ascii=False)
    print("Done:", json.dumps(out["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
