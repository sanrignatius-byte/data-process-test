#!/usr/bin/env python3
"""给 enriched hub candidates 回填 edge_contexts（桥接段落原文）。

候选的 path 含中间桥节点(::p::N 段落 / ::sec::N 章节)，但 enrich_hub_candidates
没把桥文本填进 edge_contexts(全空)。本脚本按 path 上的桥节点回填：
  1. 优先用拓扑候选已 resolve 的 bridge_contexts(node_id→text，文本干净)；
  2. 兜底用 reference graph 的 paragraphs[idx-1].text_snippet(::p::N = 第 N 段，1-based)。
填好后 L2/L3 生成都能拿到桥接原文进 prompt。
"""
import argparse
import json
import re

PNODE = re.compile(r"::p::(\d+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="enriched hub candidates")
    ap.add_argument("--topology", required=True, help="topology candidates (含 bridge_contexts)")
    ap.add_argument("--latex-graph", required=True, help="reference graph (paragraphs 兜底)")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    enr = json.load(open(a.candidates, encoding="utf-8"))
    topo = json.load(open(a.topology, encoding="utf-8"))
    graph = json.load(open(a.latex_graph, encoding="utf-8"))
    gdocs = graph.get("documents", graph)

    # 1) 全局 node_id → text（聚合所有候选已 resolve 的 bridge_contexts）
    node_text = {}
    for c in topo["candidates"]:
        for bc in c.get("bridge_contexts") or []:
            nid, txt = bc.get("node_id"), (bc.get("text") or "").strip()
            if nid and txt and nid not in node_text:
                node_text[nid] = txt
    print(f"bridge_contexts 全局 node 文本: {len(node_text)} 个")

    # 2) 段落兜底：doc → paragraphs 列表(1-based 索引)
    def para_text(doc_id, idx):
        doc = gdocs.get(doc_id)
        if not doc:
            return ""
        paras = (doc.get("metadata") or {}).get("paragraphs") or []
        if 1 <= idx <= len(paras):
            return (paras[idx - 1].get("text_snippet") or "").strip()
        return ""

    pairs = enr["pairs"]
    filled = items = 0
    for p in pairs:
        doc_id = p["doc_id"]
        ecs = []
        for nid in p.get("path", []):
            if "::p::" not in nid and "::sec::" not in nid:
                continue  # 跳过端点元素
            txt = node_text.get(nid)
            if not txt:
                m = PNODE.search(nid)
                if m:
                    txt = para_text(doc_id, int(m.group(1)))
            if txt:
                ecs.append({"node_id": nid, "text": txt})
        p["edge_contexts"] = ecs
        if ecs:
            filled += 1
            items += len(ecs)

    enr.setdefault("metadata", {})["bridge_backfill"] = {
        "pairs_total": len(pairs),
        "pairs_with_edge_contexts": filled,
        "edge_context_items": items,
        "global_bridge_nodes": len(node_text),
    }
    json.dump(enr, open(a.output, "w", encoding="utf-8"), ensure_ascii=False)
    print("Done:", json.dumps(enr["metadata"]["bridge_backfill"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
