#!/usr/bin/env python3
"""从拓扑候选集筛出涉及的 MinerU 元素，导出 enrich_elements_modora 可吃的子集。

候选文件 (latex_hub_multihop_candidates) 的 path_node_ids/hub_node_id 是 LaTeX 节点 id
(形如 "1306.5204::el::fig:histograms")。本脚本复用 enrich_hub_candidates 里已验证的
build_node_element_map 把它们映射回 MinerU element_id，只保留可 enrich 的
(figure/table/formula)，按原 multimodal_elements schema 导出子集。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.enrich_hub_candidates import (  # noqa: E402
    build_mm_index,
    build_node_element_map,
)

ENRICHABLE = {"figure", "table", "formula"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--latex-graph", required=True)
    ap.add_argument("--elements", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cands = json.load(open(args.candidates, encoding="utf-8"))["candidates"]
    latex_data = json.load(open(args.latex_graph, encoding="utf-8"))
    mm_data = json.load(open(args.elements, encoding="utf-8"))

    # latex graph 顶层可能含 metadata；映射函数只认 doc 字典
    latex_docs = latex_data.get("documents", latex_data)

    mm_index = build_mm_index(mm_data)
    node_to_element = build_node_element_map(latex_docs, mm_index, mm_data)

    # 收集候选路径里的 latex 节点 id
    node_ids: set[str] = set()
    for c in cands:
        if c.get("hub_node_id"):
            node_ids.add(c["hub_node_id"])
        for n in c.get("path_node_ids", []):
            node_ids.add(n)

    # 映射到 MinerU element_id
    eids: set[str] = set()
    unmapped = 0
    for n in node_ids:
        eid = node_to_element.get(n)
        if eid:
            eids.add(eid)
        else:
            unmapped += 1

    all_elements = mm_index["all_elements"]
    # 只留可 enrich 的
    keep_eids = {e for e in eids if all_elements.get(e, {}).get("element_type") in ENRICHABLE}

    # 按 doc 重组成原 schema
    out_docs: dict = {}
    type_counter: Counter = Counter()
    for eid in keep_eids:
        el = all_elements[eid]
        doc_id = el["doc_id"]
        d = out_docs.setdefault(doc_id, {"doc_id": doc_id, "elements": {}})
        d["elements"][eid] = el
        type_counter[el["element_type"]] += 1
    for d in out_docs.values():
        d["num_elements"] = len(d["elements"])

    out = {
        "metadata": {
            "source": args.candidates,
            "note": "candidate-only enrich subset (topology hub multihop)",
            "num_candidate_nodes": len(node_ids),
            "num_unmapped_nodes": unmapped,
            "num_enrichable_elements": len(keep_eids),
            "by_type": dict(type_counter),
        },
        "documents": out_docs,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps(out["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
