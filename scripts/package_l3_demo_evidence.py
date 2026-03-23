#!/usr/bin/env python3
"""
将 L3 demo 10 条 query 的 evidence 按 query 打包：
  - 从 multimodal_elements.json 取元素详情（caption, content, image_path, context）
  - 从 l3_reasoning_chain_queries.jsonl 取 required_evidence_spans + reasoning_steps
  - 输出一个 JSON，每条 query 一个包，方便阅读/展示
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

DEMO_FILE   = ROOT / "data/m2/l3_demo_selection_10.json"
ELEMENTS    = ROOT / "data/multimodal_elements.json"
L3_QUERIES  = ROOT / "data/m2/l3_reasoning_chain_queries.jsonl"
OUTPUT      = ROOT / "data/m2/l3_demo_evidence_package.json"


def load_elements(path):
    """element_id -> element dict (handles nested documents structure)"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    if isinstance(data, dict) and "documents" in data:
        for doc_id, doc in data["documents"].items():
            elems = doc.get("elements", {})
            if isinstance(elems, dict):
                out.update(elems)
            elif isinstance(elems, list):
                for e in elems:
                    out[e["element_id"]] = e
    elif isinstance(data, list):
        out = {e["element_id"]: e for e in data}
    return out


def load_l3_queries(path):
    """query_id -> full query record"""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["query_id"]] = rec
    return out


def build_package(demo, elements_db, l3_db):
    packages = []
    for item in demo:
        qid = item["query_id"]
        l3_rec = l3_db.get(qid, {})

        # 收集涉及的 element 详情
        elem_details = {}
        for eid in item.get("element_ids", []):
            e = elements_db.get(eid)
            if e:
                elem_details[eid] = {
                    "element_type": e.get("element_type"),
                    "doc_id":       e.get("doc_id"),
                    "caption":      e.get("caption", ""),
                    "content":      (e.get("content") or "")[:500],   # 截断防过长
                    "image_path":   e.get("image_path", ""),
                    "context_before": (e.get("context_before") or "")[:300],
                    "context_after":  (e.get("context_after") or "")[:300],
                }

        pkg = {
            "query_id":    qid,
            "is_cross_doc": item.get("is_cross_doc", False),
            "pair_type":    item.get("pair_type"),
            "query":        item["query"],
            "answer":       item["answer"],
            # 三步推理链（来自 demo 文件自身）
            "reasoning_steps": item.get("reasoning_steps", []),
            # 原始 query 记录里的 evidence spans（如果有）
            "required_evidence_spans": l3_rec.get("required_evidence_spans", []),
            "text_evidence": l3_rec.get("text_evidence", ""),
            "visual_anchors": l3_rec.get("visual_anchors", []),
            # 元素详情
            "element_details": elem_details,
        }
        packages.append(pkg)
    return packages


def main():
    demo     = json.loads(DEMO_FILE.read_text(encoding="utf-8"))
    elem_db  = load_elements(ELEMENTS)
    l3_db    = load_l3_queries(L3_QUERIES)

    print(f"Demo queries:  {len(demo)}")
    print(f"Elements DB:   {len(elem_db)}")
    print(f"L3 query DB:   {len(l3_db)}")

    packages = build_package(demo, elem_db, l3_db)

    # 统计
    matched = sum(1 for p in packages if p["required_evidence_spans"])
    print(f"Matched L3 records (has evidence_spans): {matched}/{len(packages)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(packages, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nOutput: {OUTPUT}")
    print("Done!")


if __name__ == "__main__":
    main()
