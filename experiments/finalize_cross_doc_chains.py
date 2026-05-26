#!/usr/bin/env python3
"""Finalize cross-document chains from entity-bridge chain materials.

This script keeps the natural 3-paper / 2-hop entity-bridge chains, filters
generic visual artifacts, deduplicates reverse/permutation duplicates, and
audits doc_id <-> element_id consistency.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/05_eval/entity_bridge_chains_53_fixed_20260522T0910Z/chains.jsonl"
DEFAULT_OUTPUT = ROOT / "data/05_eval/cross_doc_chains_final_fixed.json"
DEFAULT_OUTPUT_JSONL = ROOT / "data/05_eval/cross_doc_chains_final_fixed.jsonl"
DEFAULT_AUDIT = ROOT / "data/05_eval/cross_doc_chains_final_fixed_audit.json"

MULTIMODAL_TYPES = {"figure", "table", "formula"}

GENERIC_ENTITIES = {
    "legend symbol",
    "square marker",
    "remark marker",
    "typographic glyph",
    "document icon",
    "hollow square",
    "square symbol",
    "marker",
    "icon",
    "glyph",
    "table",
    "figure",
    "plot",
    "scatter plot",
    "histogram",
    "bar chart",
    "line chart",
    "distribution comparison",
    "overlap",
    "point cloud",
    "cluster separation",
    "x-axis",
    "y-axis",
    "black",
    "white",
    "female",
    "male",
    "num faces",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def element_doc_mismatch(element: dict[str, Any]) -> bool:
    eid = element.get("element_id", "")
    doc_id = element.get("doc_id", "")
    if not eid or not doc_id or "_" not in eid:
        return False
    return eid.split("_", 1)[0] != doc_id


def clean_bridge_entities(bridge: dict[str, Any]) -> list[str]:
    return [
        ent for ent in bridge.get("shared_entities", [])
        if ent.strip().lower() not in GENERIC_ENTITIES
    ]


def finalize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    final = []
    seen_element_sets = set()
    audit = {
        "input_chains": len(rows),
        "dropped_non_2hop": 0,
        "dropped_bad_schema": 0,
        "dropped_generic_bridge": 0,
        "dropped_duplicate": 0,
        "input_doc_element_mismatches": [],
        "output_doc_element_mismatches": [],
    }

    for chain in rows:
        mismatches = [
            {
                "chain_id": chain.get("chain_id", ""),
                "doc_id": e.get("doc_id", ""),
                "element_id": e.get("element_id", ""),
            }
            for e in chain.get("elements", [])
            if element_doc_mismatch(e)
        ]
        audit["input_doc_element_mismatches"].extend(mismatches)

        if chain.get("cross_doc_hops") != 2:
            audit["dropped_non_2hop"] += 1
            continue

        elements = chain.get("elements", [])
        bridges = chain.get("bridges", [])
        paper_path = chain.get("paper_path", [])
        if len(elements) < 3 or len(bridges) != 2 or len(paper_path) != 3:
            audit["dropped_bad_schema"] += 1
            continue
        if any((e.get("element_type") not in MULTIMODAL_TYPES) for e in elements):
            audit["dropped_bad_schema"] += 1
            continue
        if any(element_doc_mismatch(e) for e in elements):
            audit["dropped_bad_schema"] += 1
            continue

        cleaned_by_bridge = [clean_bridge_entities(b) for b in bridges]
        if any(len(ents) < 2 for ents in cleaned_by_bridge):
            audit["dropped_generic_bridge"] += 1
            continue

        element_key = tuple(sorted(e["element_id"] for e in elements))
        if element_key in seen_element_sets:
            audit["dropped_duplicate"] += 1
            continue
        seen_element_sets.add(element_key)

        shared_entities = []
        for ents in cleaned_by_bridge:
            for ent in ents:
                if ent not in shared_entities:
                    shared_entities.append(ent)

        out = {
            "chain_id": f"xdoc_eb_fixed_{len(final):04d}",
            "source_chain_id": chain.get("chain_id", ""),
            "papers": list(paper_path),
            "n_papers": len(paper_path),
            "n_elements": len(elements),
            "n_bridges": len(bridges),
            "score": chain.get("total_score", 0),
            "shared_entities": shared_entities,
            "joint_entities": chain.get("joint_entities", []),
            "element_types": [e.get("element_type", "") for e in elements],
            "bridge_types": ["cross_doc_entity" for _ in bridges],
            "elements": elements,
            "bridges": [
                {
                    **bridge,
                    "clean_shared_entities": cleaned_by_bridge[i],
                    "type": "cross_doc_entity",
                }
                for i, bridge in enumerate(bridges)
            ],
        }
        final.append(out)

    for chain in final:
        for element in chain["elements"]:
            if element_doc_mismatch(element):
                audit["output_doc_element_mismatches"].append({
                    "chain_id": chain["chain_id"],
                    "doc_id": element.get("doc_id", ""),
                    "element_id": element.get("element_id", ""),
                })

    audit["output_chains"] = len(final)
    audit["output_papers"] = len({p for c in final for p in c["papers"]})
    audit["output_element_lengths"] = dict(Counter(len(c["elements"]) for c in final))
    audit["output_paper_lengths"] = dict(Counter(len(c["papers"]) for c in final))
    audit["output_element_types"] = dict(Counter(t for c in final for t in c["element_types"]))
    audit["output_bridge_types"] = dict(Counter(t for c in final for t in c["bridge_types"]))
    audit["top_shared_entities"] = Counter(
        ent for c in final for ent in c["shared_entities"]
    ).most_common(30)
    return final, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    chains, audit = finalize(rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"chains": chains, "audit": audit}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_jsonl = Path(args.output_jsonl)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for chain in chains:
            f.write(json.dumps(chain, ensure_ascii=False) + "\n")

    audit_path = Path(args.audit)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"input_chains: {audit['input_chains']}")
    print(f"output_chains: {audit['output_chains']}")
    print(f"output_papers: {audit['output_papers']}")
    print(f"dropped_non_2hop: {audit['dropped_non_2hop']}")
    print(f"dropped_generic_bridge: {audit['dropped_generic_bridge']}")
    print(f"dropped_duplicate: {audit['dropped_duplicate']}")
    print(f"output_doc_element_mismatches: {len(audit['output_doc_element_mismatches'])}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
