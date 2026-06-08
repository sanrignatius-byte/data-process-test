#!/usr/bin/env python3
"""把 element 的 enriched [T]/[M]/[C] 注入粗/细 chunk。

落地 mentor 的两点(见 research-wiki 2026-04-21 chunk 讨论):
  1. chunk↔element 边实体化(chunk.element_ids 已有，这里附上对应 enriched 内容)
  2. enriched_content 进入检索 passage(生成 text_augmented，原 text 保留不动)

对粗(section)和细(chunk)两级都注入。enriched 元素来源是 enrich_elements_modora
的输出(任何带 enriched_title/enriched_content 字段的 multimodal_elements 文件)。
"""
import argparse
import json


def build_enriched_index(elements: dict) -> dict:
    """element_id → {type,title,content,keywords}，仅收已 enriched 的。"""
    idx = {}
    for doc in elements.get("documents", {}).values():
        for eid, e in doc.get("elements", {}).items():
            if e.get("enriched_title") or e.get("enriched_content"):
                idx[eid] = {
                    "element_id": eid,
                    "type": e.get("element_type"),
                    "enriched_title": e.get("enriched_title", ""),
                    "enriched_content": e.get("enriched_content", ""),
                    "keywords": (e.get("enriched_metadata") or {}).get("keywords", []),
                }
    return idx


def render_block(items: list) -> str:
    """把注入元素渲染成可检索文本块。"""
    if not items:
        return ""
    lines = ["\n\n[Referenced multimodal elements]"]
    for it in items:
        kw = ", ".join(it["keywords"]) if it["keywords"] else ""
        head = f"- {it['type']}: {it['enriched_title']}".rstrip()
        lines.append(head)
        if it["enriched_content"]:
            lines.append(f"  {it['enriched_content']}")
        if kw:
            lines.append(f"  keywords: {kw}")
    return "\n".join(lines)


def inject_units(units: list, idx: dict):
    """对一组带 element_ids/text 的单元注入；返回 (注入单元数, 注入元素总数)。"""
    n_units = n_items = 0
    for u in units:
        items = [idx[e] for e in u.get("element_ids", []) if e in idx]
        u["injected_elements"] = items
        u["n_injected"] = len(items)
        u["text_augmented"] = u["text"] + render_block(items)
        if items:
            n_units += 1
            n_items += len(items)
    return n_units, n_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="hierarchical chunks json")
    ap.add_argument("--enriched", required=True, help="enriched multimodal_elements json")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    chunks = json.load(open(a.chunks, encoding="utf-8"))
    elements = json.load(open(a.enriched, encoding="utf-8"))
    idx = build_enriched_index(elements)

    sec_units = sec_items = fine_units = fine_items = 0
    for doc in chunks["documents"].values():
        su, si = inject_units(doc.get("sections", []), idx)
        fu, fi = inject_units(doc.get("chunks", []), idx)
        sec_units += su; sec_items += si
        fine_units += fu; fine_items += fi

    chunks.setdefault("metadata", {})["enrich_injection"] = {
        "enriched_source": a.enriched,
        "enriched_elements_available": len(idx),
        "sections_with_injection": sec_units,
        "section_injected_total": sec_items,
        "fine_chunks_with_injection": fine_units,
        "fine_injected_total": fine_items,
    }
    json.dump(chunks, open(a.output, "w", encoding="utf-8"), ensure_ascii=False)
    print("Done:", json.dumps(chunks["metadata"]["enrich_injection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
