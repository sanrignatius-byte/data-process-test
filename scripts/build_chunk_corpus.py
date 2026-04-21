#!/usr/bin/env python3
"""
build_chunk_corpus.py
=====================
从 paragraph_chunks JSON + full multimodal_elements graph 构建检索 corpus。

核心设计：
  chunk passage = 段落文本
                + 区间内所有 element 的原始可见文本（caption/content）
                + 可选 enriched_content overlay

这样做的目的有两个：
  1. chunk 不再只看到 selective enrich 子集，避免漏掉大量 element；
  2. retrieval 可以在统一 element 覆盖下，分别比较 raw graph text 和
     enriched overlay 对结果的影响，减少实验口径污染；
  3. 对闭集 retrieval eval，可以通过 enriched coverage guard 强制检查
     “是否真的已经补齐 enrich”，避免把 partial enrich 当成公平实验。

同时输出重映射后的 qrels：gold passage 从 element_id → parent_chunk_id。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GRAPH_ELEMENTS = "data/03_queries/M4query_v1/graphs/multimodal_elements.json"


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] {path}:{ln} bad json: {e}", file=sys.stderr)
    return rows


def write_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── 构建 element 索引 ──────────────────────────────────────────────────────────

def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def build_element_index(
    graph_elements_path: Path,
    enriched_path: Optional[Path],
) -> Tuple[Dict[str, Dict[int, List[dict]]], dict]:
    """
    返回：
      - {doc_id: {position_idx: [element_dict, ...]}}
      - coverage report

    element_dict 包含：
      element_id / caption / content / enriched_content / element_type
    """
    graph_data = json.loads(graph_elements_path.read_bytes())
    enriched_data = {}
    if enriched_path and enriched_path.exists():
        enriched_data = json.loads(enriched_path.read_bytes())

    enriched_lookup: Dict[Tuple[str, str], str] = {}
    for doc_id, doc in enriched_data.get("documents", {}).items():
        for eid, elem in doc.get("elements", {}).items():
            enriched_lookup[(doc_id, eid)] = (elem.get("enriched_content") or "").strip()

    index: Dict[str, Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    total_graph_elements = 0
    total_overlayed = 0
    total_with_raw_text = 0

    for doc_id, doc in graph_data.get("documents", {}).items():
        for eid, elem in doc.get("elements", {}).items():
            pidx = elem.get("position_idx")
            if pidx is None:
                continue

            caption = (elem.get("caption") or "").strip()
            content = (elem.get("content") or "").strip()
            enriched_content = enriched_lookup.get((doc_id, eid), "")

            total_graph_elements += 1
            if caption or content:
                total_with_raw_text += 1
            if enriched_content:
                total_overlayed += 1

            index[doc_id][pidx].append({
                "element_id":       eid,
                "element_type":     elem.get("element_type", elem.get("type", "")),
                "caption":          caption,
                "content":          content,
                "enriched_content": enriched_content,
            })

    coverage = {
        "graph_elements_path": str(graph_elements_path),
        "enriched_overlay_path": str(enriched_path) if enriched_path else None,
        "total_graph_elements": total_graph_elements,
        "elements_with_raw_text": total_with_raw_text,
        "elements_with_enriched_overlay": total_overlayed,
    }
    return index, coverage


# ── 为单个 chunk 组装 passage 文本 ────────────────────────────────────────────

def build_passage_text(para_text: str, elements: List[dict], text_mode: str) -> str:
    """element 内容先，段落文本后，避免前部截断时丢失多模态信号。"""
    elem_parts = []
    for elem in elements:
        etype = elem["element_type"].capitalize() if elem["element_type"] else "Element"
        lines = []
        if elem["caption"]:
            lines.append(elem["caption"])
        if elem["content"]:
            lines.append(elem["content"])
        if text_mode == "graph_plus_enriched" and elem["enriched_content"]:
            lines.append(elem["enriched_content"])
        if lines:
            elem_parts.append(f"[{etype}]: " + " ".join(lines))
    parts = elem_parts + [para_text.strip()]
    return "\n\n".join(parts)


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks",   required=True, help="paragraph_chunks_nXXX.json")
    ap.add_argument("--graph-elements", default=DEFAULT_GRAPH_ELEMENTS,
                    help="完整 multimodal_elements.json（用于全量 element 对齐）")
    ap.add_argument("--enriched", default=None,
                    help="可选 multimodal_elements_enriched.json（仅作为 enriched overlay）")
    ap.add_argument("--element-text-mode",
                    choices=["graph_only", "graph_plus_enriched"],
                    default="graph_plus_enriched",
                    help="chunk 中 element 文本使用原始 graph text，还是叠加 enriched overlay")
    ap.add_argument("--min-enriched-coverage", type=float, default=None,
                    help="若使用 graph_plus_enriched，可要求目标 docs 的 enriched overlay 覆盖率至少达到该比例")
    ap.add_argument("--qrels",    required=True, help="原始 qrels.jsonl（element_id 级别）")
    ap.add_argument("--out-dir",  required=True, help="输出目录")
    ap.add_argument("--gold-docs-only", action="store_true", default=True,
                    help="只输出 qrels 涉及的 doc 的 chunks（默认 True，减少 corpus 噪声）")
    ap.add_argument("--all-docs", action="store_true",
                    help="输出全部 1147 docs 的 chunks（覆盖 --gold-docs-only）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 qrels，提取 gold docs 和 element→(query_id, relevance) 映射
    print("Loading qrels ...", flush=True)
    qrels_rows = load_jsonl(Path(args.qrels))
    gold_docs: Set[str] = set()
    # element_id → [(query_id, relevance), ...]
    elem_to_qrels: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for row in qrels_rows:
        pid = row["passage_id"]
        qid = row["query_id"]
        rel = int(row.get("relevance", 1))
        doc_id = pid.split("_")[0]
        gold_docs.add(doc_id)
        elem_to_qrels[pid].append((qid, rel))
    print(f"  {len(qrels_rows)} qrels, {len(set(r['passage_id'] for r in qrels_rows))} unique passages, {len(gold_docs)} gold docs")

    graph_elements_path = resolve_path(args.graph_elements)
    enriched_path = resolve_path(args.enriched) if args.enriched else None

    # 2. 构建 element 索引
    print("Building element index ...", flush=True)
    elem_index, elem_coverage = build_element_index(graph_elements_path, enriched_path)
    n_index_docs = len(elem_index)
    n_index_elems = sum(len(v) for d in elem_index.values() for v in d.values())
    print(f"  {n_index_docs} docs, {n_index_elems} elements indexed")
    print(f"  Raw-text elements: {elem_coverage['elements_with_raw_text']}")
    print(f"  Enriched overlays: {elem_coverage['elements_with_enriched_overlay']}")

    # 3. 加载 chunk 图，决定要处理哪些 doc
    print("Loading chunk graph ...", flush=True)
    chunk_data = json.loads(Path(args.chunks).read_bytes())
    all_docs = chunk_data.get("documents", {})

    if not args.all_docs:
        # 默认只处理 gold docs（减少 corpus 大小，对 eval 更公平）
        target_docs = {d: all_docs[d] for d in gold_docs if d in all_docs}
        missing_from_chunks = gold_docs - set(all_docs.keys())
        if missing_from_chunks:
            print(f"[warn] {len(missing_from_chunks)} gold docs missing from chunk graph: {missing_from_chunks}")
    else:
        target_docs = all_docs

    print(f"  Processing {len(target_docs)} docs (chunk graph has {len(all_docs)} total)")

    target_total_elements = 0
    target_overlayed_elements = 0
    for doc_id in target_docs:
        for elems in elem_index.get(doc_id, {}).values():
            target_total_elements += len(elems)
            target_overlayed_elements += sum(1 for e in elems if e["enriched_content"])
    target_overlay_coverage = (
        target_overlayed_elements / target_total_elements if target_total_elements else 0.0
    )
    print(
        f"  Target-doc element coverage: {target_total_elements} total, "
        f"{target_overlayed_elements} with enriched overlay "
        f"({target_overlay_coverage:.1%})"
    )

    if (
        args.element_text_mode == "graph_plus_enriched"
        and args.min_enriched_coverage is not None
        and target_overlay_coverage < args.min_enriched_coverage
    ):
        raise SystemExit(
            f"[fatal] enriched overlay coverage on target docs is {target_overlay_coverage:.1%}, "
            f"below required threshold {args.min_enriched_coverage:.1%}. "
            "Backfill enrich first or switch to --element-text-mode graph_only."
        )

    # 4. 遍历 chunk，构建 corpus 行 + element→chunk 反向映射
    corpus_rows: List[dict] = []
    # element_id → chunk_id（一个 element 只属于一个 chunk）
    elem_to_chunk: Dict[str, str] = {}

    stats = {
        "total_chunks": 0,
        "chunks_with_elements": 0,
        "elements_injected": 0,
        "elements_no_text": 0,
    }

    for doc_id, doc in target_docs.items():
        doc_elem_idx = elem_index.get(doc_id, {})  # position_idx → [elem_dict]
        nodes = doc.get("nodes", {})

        for chunk_id, chunk in nodes.items():
            para_indices: List[int] = chunk.get("paragraph_indices", [])
            para_text: str = chunk.get("text", "")

            # 收集该 chunk 区间内所有 element
            chunk_elems: List[dict] = []
            for pidx in para_indices:
                for e in doc_elem_idx.get(pidx, []):
                    chunk_elems.append(e)
                    # 注册反向映射（element_id → chunk_id）
                    elem_to_chunk[e["element_id"]] = chunk_id

            passage_text = build_passage_text(para_text, chunk_elems, args.element_text_mode)

            has_elem_text = any(
                e["caption"] or e["content"] or e["enriched_content"] for e in chunk_elems
            )
            stats["total_chunks"] += 1
            if chunk_elems:
                stats["chunks_with_elements"] += 1
                stats["elements_injected"] += len(chunk_elems)
            no_text_count = sum(
                1
                for e in chunk_elems
                if not e["caption"] and not e["content"] and not e["enriched_content"]
            )
            stats["elements_no_text"] += no_text_count

            corpus_rows.append({
                "passage_id":    chunk_id,
                "doc_id":        doc_id,
                "text":          passage_text,
                "chunk_idx":     chunk.get("chunk_idx", -1),
                "section_title": chunk.get("section_title", ""),
                "word_count":    chunk.get("word_count", 0),
                "num_elements":  len(chunk_elems),
            })

    print(f"  Corpus: {stats['total_chunks']} passages")
    print(f"  Chunks with elements: {stats['chunks_with_elements']} ({stats['chunks_with_elements']/max(1,stats['total_chunks'])*100:.1f}%)")
    print(f"  Elements injected: {stats['elements_injected']}")

    # 5. 重映射 qrels: element_id → chunk_id
    print("Remapping qrels ...", flush=True)
    new_qrels: List[dict] = []
    n_mapped = n_unmapped = 0

    # 用 (query_id, chunk_id) 去重，防止同一 chunk 被多个 element 触发重复计
    seen: Set[Tuple[str, str]] = set()

    for element_id, qrel_list in elem_to_qrels.items():
        chunk_id = elem_to_chunk.get(element_id)
        if chunk_id is None:
            n_unmapped += 1
            continue
        for (qid, rel) in qrel_list:
            key = (qid, chunk_id)
            if key not in seen:
                seen.add(key)
                new_qrels.append({
                    "query_id":    qid,
                    "passage_id":  chunk_id,
                    "relevance":   rel,
                    "source_element_id": element_id,  # 保留来源，方便 debug
                })
            n_mapped += 1

    print(f"  Mapped: {n_mapped}, Unmapped: {n_unmapped}")
    print(f"  New qrels: {len(new_qrels)} rows ({len(set(r['query_id'] for r in new_qrels))} queries covered)")

    # 6. 写出
    corpus_path = out_dir / "corpus.jsonl"
    qrels_path  = out_dir / "qrels.jsonl"
    write_jsonl(corpus_rows, corpus_path)
    write_jsonl(new_qrels,   qrels_path)
    print(f"  corpus → {corpus_path} ({corpus_path.stat().st_size/1024/1024:.1f} MB)")
    print(f"  qrels  → {qrels_path}")

    # 7. 写报告
    report = {
        "chunks_file":         str(args.chunks),
        "graph_elements_file": str(graph_elements_path),
        "enriched_file":       str(enriched_path) if enriched_path else None,
        "element_text_mode":   args.element_text_mode,
        "original_qrels":      str(args.qrels),
        "gold_docs_only":      not args.all_docs,
        "num_docs_processed":  len(target_docs),
        "corpus_stats":        stats,
        "element_coverage":    elem_coverage,
        "target_doc_element_coverage": {
            "total_elements": target_total_elements,
            "elements_with_enriched_overlay": target_overlayed_elements,
            "enriched_overlay_ratio": target_overlay_coverage,
        },
        "qrels_remapping": {
            "original_qrel_rows":     len(qrels_rows),
            "original_gold_elements": len(elem_to_qrels),
            "mapped_elements":        n_mapped,
            "unmapped_elements":      n_unmapped,
            "new_qrel_rows":          len(new_qrels),
            "queries_covered":        len(set(r["query_id"] for r in new_qrels)),
        },
    }
    report_path = out_dir / "build_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"  report → {report_path}")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
