#!/usr/bin/env python3
"""Build enrichment targets for Method C long-chain bundles.

Goal:
  - Select unique multimodal elements that participate in long-chain bundles
  - Select unique non-modal bridge labels (section/appendix/algorithm/...)
  - Aggregate edge-context snippets for each bridge label
  - Export:
      1. element subset file compatible with enrich_elements_modora.py
      2. bridge target manifest for bridge-specific enrichment
      3. summary manifest for audit / downstream use

Typical usage:
    python scripts/build_method_c_long_chain_enrich_targets.py \
      --min-hops 4 \
      --long-chain-candidates data/01_graphs/long_chain_candidates_v2.json \
      --reference-graph data/01_graphs/latex_reference_graph_v2.json \
      --multimodal-elements data/01_graphs/multimodal_elements_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analyze_latex_graph_topology import build_multimodal_index, map_label_to_element
from src.utils.text_utils import normalize_label_type


ELEMENT_TYPES = {"figure", "table", "equation"}


def _clean_text(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_label_ordinals(doc: Dict[str, Any]) -> Dict[str, int]:
    """Build per-label ordinal among modal labels, matching topology logic."""
    labels = doc.get("labels", {}) or {}
    type_counters = {"figure": 0, "table": 0, "formula": 0}
    label_ordinals: Dict[str, int] = {}
    sorted_labels = sorted(
        labels.items(),
        key=lambda kv: (
            int(kv[1].get("line_no", 0)) if isinstance(kv[1].get("line_no"), int) else 0,
            kv[0],
        ),
    )
    for label_key, info in sorted_labels:
        modal = normalize_label_type(str(info.get("label_type", "")))
        if modal not in ELEMENT_TYPES:
            continue
        mm_type = "formula" if modal == "equation" else modal
        label_ordinals[label_key] = type_counters[mm_type]
        type_counters[mm_type] += 1
    return label_ordinals


def _minify_target_ids(ids: Set[str], limit: int = 12) -> List[str]:
    return sorted(ids)[:limit]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--long-chain-candidates", default="data/01_graphs/long_chain_candidates_v2.json")
    ap.add_argument("--reference-graph", default="data/01_graphs/latex_reference_graph_v2.json")
    ap.add_argument("--multimodal-elements", default="data/01_graphs/multimodal_elements_v2.json")
    ap.add_argument("--mineru-output-dir", default="data/00_raw/mineru_output")
    ap.add_argument("--min-hops", type=int, default=4)
    ap.add_argument("--max-candidates", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--element-subset-output",
        default="data/02_enriched/method_c_long_chain_elements_subset_v1.json",
    )
    ap.add_argument(
        "--bridge-target-output",
        default="data/02_enriched/method_c_long_chain_bridge_targets_v1.json",
    )
    ap.add_argument(
        "--manifest-output",
        default="data/02_enriched/method_c_long_chain_enrich_manifest_v1.json",
    )
    args = ap.parse_args()

    lc_path = PROJECT_ROOT / args.long_chain_candidates
    ref_path = PROJECT_ROOT / args.reference_graph
    mm_path = PROJECT_ROOT / args.multimodal_elements
    mineru_dir = PROJECT_ROOT / args.mineru_output_dir

    print("Loading inputs...")
    lc_data = _load_json(lc_path)
    ref_data = _load_json(ref_path)
    mm_data = _load_json(mm_path)

    candidates = [
        c for c in lc_data.get("candidates", [])
        if int(c.get("hop_distance", 0)) >= args.min_hops
    ]
    if args.max_candidates > 0:
        candidates = candidates[:args.max_candidates]
    print(f"  Long-chain candidates after min-hop filter: {len(candidates)}")

    print("Building multimodal index for label -> element mapping...")
    mm_index = build_multimodal_index(mm_data, mineru_output_dir=str(mineru_dir))

    ref_docs = ref_data.get("documents", {})
    label_ordinals_by_doc = {
        doc_id: _build_label_ordinals(doc)
        for doc_id, doc in ref_docs.items()
    }

    element_targets: Dict[str, Dict[str, Any]] = {}
    bridge_targets: Dict[str, Dict[str, Any]] = {}
    unmapped_modal_labels: List[Dict[str, Any]] = []

    selected_candidate_ids: List[str] = []
    docs_covered: Set[str] = set()

    for cand in candidates:
        cand_id = str(cand.get("candidate_id", ""))
        doc_id = str(cand.get("doc_id", ""))
        docs_covered.add(doc_id)
        selected_candidate_ids.append(cand_id)
        doc = ref_docs.get(doc_id, {})
        labels = doc.get("labels", {}) or {}
        ordinals = label_ordinals_by_doc.get(doc_id, {})
        path_label_keys = cand.get("path_label_keys", [])
        path_types = cand.get("path_types", [])
        path_len = len(path_label_keys)

        for idx, (label_key, path_type) in enumerate(zip(path_label_keys, path_types)):
            role = "endpoint" if idx in (0, path_len - 1) else "intermediate"
            label_info = labels.get(label_key, {})
            modal = normalize_label_type(str(label_info.get("label_type", path_type)))
            if modal in ELEMENT_TYPES:
                mapped_eid = map_label_to_element(
                    doc_id=doc_id,
                    label_key=label_key,
                    label_info=label_info,
                    mm_index=mm_index,
                    label_ordinal=ordinals.get(label_key),
                )
                if not mapped_eid:
                    unmapped_modal_labels.append(
                        {
                            "candidate_id": cand_id,
                            "doc_id": doc_id,
                            "label_key": label_key,
                            "path_type": path_type,
                            "role": role,
                        }
                    )
                    continue

                tgt = element_targets.setdefault(
                    mapped_eid,
                    {
                        "element_id": mapped_eid,
                        "doc_id": doc_id,
                        "label_keys": set(),
                        "label_types": Counter(),
                        "chain_count": 0,
                        "endpoint_count": 0,
                        "intermediate_count": 0,
                        "candidate_ids": set(),
                    },
                )
                tgt["label_keys"].add(label_key)
                tgt["label_types"][modal] += 1
                tgt["chain_count"] += 1
                tgt[f"{role}_count"] += 1
                tgt["candidate_ids"].add(cand_id)
            else:
                bridge_id = f"{doc_id}::{label_key}"
                tgt = bridge_targets.setdefault(
                    bridge_id,
                    {
                        "bridge_id": bridge_id,
                        "doc_id": doc_id,
                        "label_key": label_key,
                        "label_type": str(label_info.get("label_type", path_type)),
                        "environment": label_info.get("environment"),
                        "line_no": label_info.get("line_no"),
                        "caption": label_info.get("caption"),
                        "file_path": label_info.get("file_path"),
                        "chain_count": 0,
                        "endpoint_count": 0,
                        "intermediate_count": 0,
                        "candidate_ids": set(),
                        "connected_modalities": set(),
                        "neighbor_labels": set(),
                        "edge_contexts": [],
                    },
                )
                tgt["chain_count"] += 1
                tgt[f"{role}_count"] += 1
                tgt["candidate_ids"].add(cand_id)

    print("Aggregating bridge edge contexts...")
    for bridge_id, bridge in bridge_targets.items():
        doc = ref_docs.get(bridge["doc_id"], {})
        labels = doc.get("labels", {}) or {}
        edge_contexts: List[str] = []
        seen_ctx: Set[str] = set()
        seen_neighbors: Set[str] = set()
        modalities: Set[str] = set()
        target_label = bridge["label_key"]
        for edge in doc.get("edges", []) or []:
            src = str(edge.get("source_label", ""))
            tgt = str(edge.get("target_label", ""))
            if src != target_label and tgt != target_label:
                continue
            if bool(edge.get("is_containment")) or edge.get("ref_text") == "[containment]":
                continue

            ctx = _clean_text(edge.get("context"))
            if len(ctx) >= 20 and ctx not in seen_ctx:
                seen_ctx.add(ctx)
                edge_contexts.append(ctx)

            other = tgt if src == target_label else src
            other_info = labels.get(other, {})
            other_type = normalize_label_type(str(other_info.get("label_type", edge.get("source_type") if src != target_label else edge.get("target_type"))))
            if other_type in ELEMENT_TYPES:
                modalities.add("formula" if other_type == "equation" else other_type)
            if other and other not in seen_neighbors:
                seen_neighbors.add(other)
                bridge["neighbor_labels"].add(
                    (
                        other,
                        str(other_info.get("label_type", "")),
                        str(edge.get("ref_text", "")),
                    )
                )

        bridge["edge_contexts"] = edge_contexts[:8]
        bridge["connected_modalities"] = modalities

    print("Building multimodal element subset...")
    target_element_ids = set(element_targets.keys())
    subset_documents: Dict[str, Dict[str, Any]] = {}
    selected_element_count = 0
    for doc_id, doc in mm_data.get("documents", {}).items():
        elems = doc.get("elements", {}) or {}
        selected = {
            eid: el
            for eid, el in elems.items()
            if eid in target_element_ids
        }
        if not selected:
            continue
        subset_documents[doc_id] = {
            "doc_id": doc_id,
            "num_elements": len(selected),
            "elements": selected,
        }
        selected_element_count += len(selected)

    element_subset = {
        "metadata": {
            "source": str(mm_path),
            "selection": "method_c_long_chain_bundle",
            "min_hops": args.min_hops,
            "selected_candidates": len(candidates),
            "selected_docs": len({doc_id for doc_id in subset_documents}),
            "selected_elements": selected_element_count,
        },
        "documents": subset_documents,
    }

    serializable_elements: List[Dict[str, Any]] = []
    for entry in element_targets.values():
        serializable_elements.append(
            {
                "element_id": entry["element_id"],
                "doc_id": entry["doc_id"],
                "label_keys": sorted(entry["label_keys"]),
                "label_types": dict(entry["label_types"]),
                "chain_count": entry["chain_count"],
                "endpoint_count": entry["endpoint_count"],
                "intermediate_count": entry["intermediate_count"],
                "candidate_ids": _minify_target_ids(entry["candidate_ids"]),
            }
        )

    serializable_bridges: List[Dict[str, Any]] = []
    for entry in bridge_targets.values():
        serializable_bridges.append(
            {
                "bridge_id": entry["bridge_id"],
                "doc_id": entry["doc_id"],
                "label_key": entry["label_key"],
                "label_type": entry["label_type"],
                "environment": entry["environment"],
                "line_no": entry["line_no"],
                "caption": entry["caption"],
                "file_path": entry["file_path"],
                "chain_count": entry["chain_count"],
                "endpoint_count": entry["endpoint_count"],
                "intermediate_count": entry["intermediate_count"],
                "candidate_ids": _minify_target_ids(entry["candidate_ids"]),
                "connected_modalities": sorted(entry["connected_modalities"]),
                "neighbor_labels": [
                    {
                        "label_key": label_key,
                        "label_type": label_type,
                        "ref_text": ref_text,
                    }
                    for label_key, label_type, ref_text in sorted(entry["neighbor_labels"])[:8]
                ],
                "edge_contexts": entry["edge_contexts"],
            }
        )

    serializable_elements.sort(key=lambda x: (-x["chain_count"], x["element_id"]))
    serializable_bridges.sort(key=lambda x: (-x["chain_count"], x["bridge_id"]))

    bridge_type_counts = Counter(x["label_type"] for x in serializable_bridges)
    manifest = {
        "metadata": {
            "selection": "method_c_long_chain_bundle",
            "min_hops": args.min_hops,
            "selected_candidates": len(candidates),
            "selected_candidate_ids_sample": selected_candidate_ids[:20],
            "docs_covered": len(docs_covered),
            "source_long_chain_candidates": str(lc_path),
            "source_reference_graph": str(ref_path),
            "source_multimodal_elements": str(mm_path),
        },
        "summary": {
            "element_targets": len(serializable_elements),
            "bridge_targets": len(serializable_bridges),
            "subset_docs": len(subset_documents),
            "unmapped_modal_labels": len(unmapped_modal_labels),
            "bridge_type_counts": dict(bridge_type_counts),
        },
        "element_targets": serializable_elements,
        "bridge_targets": serializable_bridges,
        "unmapped_modal_labels_sample": unmapped_modal_labels[:200],
    }

    bridge_output = {
        "metadata": manifest["metadata"],
        "summary": {
            "bridge_targets": len(serializable_bridges),
            "bridge_type_counts": dict(bridge_type_counts),
        },
        "bridge_targets": serializable_bridges,
    }

    element_subset_path = PROJECT_ROOT / args.element_subset_output
    bridge_output_path = PROJECT_ROOT / args.bridge_target_output
    manifest_path = PROJECT_ROOT / args.manifest_output
    for path in (element_subset_path, bridge_output_path, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    element_subset_path.write_text(json.dumps(element_subset, ensure_ascii=False, indent=2), encoding="utf-8")
    bridge_output_path.write_text(json.dumps(bridge_output, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Method C target summary ===")
    print(f"  Candidates selected:     {len(candidates)}")
    print(f"  Documents covered:       {len(docs_covered)}")
    print(f"  Unique element targets:  {len(serializable_elements)}")
    print(f"  Unique bridge targets:   {len(serializable_bridges)}")
    print(f"  Unmapped modal labels:   {len(unmapped_modal_labels)}")
    print(f"  Element subset written:  {element_subset_path}")
    print(f"  Bridge manifest written: {bridge_output_path}")
    print(f"  Audit manifest written:  {manifest_path}")


if __name__ == "__main__":
    main()
