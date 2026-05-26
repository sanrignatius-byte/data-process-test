#!/usr/bin/env python3
"""Rebuild entity-bridge chains using only judged-strong pairs.

The current 38 fixed chains were built from all 83 entity-bridge pairs.
Only 21/83 pairs are judged strong. This script:
1. Loads judged pairs (strong only)
2. Loads existing chains and their bridge pairs
3. Keeps chains where BOTH bridges are from strong pairs
4. For chains with one weak bridge, attempts to find alternative strong pairs
5. Outputs improved chain set
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_judged_pairs() -> dict[str, dict[str, Any]]:
    """Load entity-bridge pair judgments, return {pair_id: judgment}."""
    pairs: dict[str, dict[str, Any]] = {}
    path = ROOT / "data/05_eval/entity_bridge_judge_v2/judgments.jsonl"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                j = json.loads(line)
                pairs[j["candidate_id"]] = j
    return pairs


def load_entity_bridge_pairs() -> list[dict[str, Any]]:
    """Load the raw entity-bridge candidate pairs."""
    path = ROOT / "data/05_eval/entity_bridge_candidates_v2/judge_pack.jsonl"
    pairs: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                pairs.append(json.loads(line))
    return pairs


def main():
    # Load judgments
    judged = load_judged_pairs()
    print(f"Loaded {len(judged)} judged pairs")

    strong_ids = {
        cid for cid, j in judged.items()
        if j.get("judgment", {}).get("verdict") == "strong_chain"
    }
    weak_ids = {
        cid for cid, j in judged.items()
        if j.get("judgment", {}).get("verdict") == "weak_but_related"
    }
    print(f"Strong pairs: {len(strong_ids)}")
    print(f"Weak-but-related pairs: {len(weak_ids)}")

    # Load raw pairs to get doc/element info
    raw_pairs = load_entity_bridge_pairs()
    pair_docs: dict[str, tuple[str, str]] = {}  # candidate_id -> (source_doc, target_doc)
    pair_elements: dict[str, tuple[str, str, str, str]] = {}
    for p in raw_pairs:
        cid = p.get("candidate_id", "")
        pair_docs[cid] = (p.get("source_doc", ""), p.get("target_doc", ""))
        pair_elements[cid] = (
            p.get("source_element_id", ""), p.get("source_element_type", ""),
            p.get("target_element_id", ""), p.get("target_element_type", ""),
        )

    # Load existing chains
    chains_path = ROOT / "data/05_eval/cross_doc_chains_final_fixed.json"
    with chains_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    chains = raw.get("chains", raw) if isinstance(raw, dict) else raw

    print(f"\nLoaded {len(chains)} existing chains")

    # Analyze each chain: which pairs are its bridges?
    chain_analysis = []
    for chain in chains:
        papers = chain["papers"]
        cid = chain["chain_id"]
        source_cid = chain.get("source_chain_id", "")

        # A chain has 2 bridges: (paper[0], paper[1]) and (paper[1], paper[2])
        # The source_chain_id encodes the pair info: eb2_{doc0}_{doc1}_{doc2}_{idx}
        # Find the actual pair IDs by looking up in judged pairs
        bridge_pair_ids: list[str] = []
        for pid in judged:
            p_src = pair_docs.get(pid, ("", ""))
            if p_src[0] == papers[0] and p_src[1] == papers[1]:
                bridge_pair_ids.append(pid)
            elif p_src[0] == papers[1] and p_src[1] == papers[2]:
                bridge_pair_ids.append(pid)

        bridge_verdicts = []
        for bpid in bridge_pair_ids:
            j = judged.get(bpid, {})
            v = j.get("judgment", {}).get("verdict", "unknown")
            bridge_verdicts.append((bpid, v))

        chain_analysis.append({
            "chain_id": cid,
            "papers": papers,
            "bridge_pairs": bridge_pair_ids,
            "bridge_verdicts": bridge_verdicts,
            "shared_entities": chain.get("shared_entities", []),
            "element_types": chain.get("element_types", []),
            "score": chain.get("score", 0),
        })

    # Categorize chains
    both_strong = []
    one_strong = []
    none_strong = []
    for ca in chain_analysis:
        bv = ca["bridge_verdicts"]
        strong_count = sum(1 for _, v in bv if v == "strong_chain")
        if strong_count == 2:
            both_strong.append(ca)
        elif strong_count == 1:
            one_strong.append(ca)
        else:
            none_strong.append(ca)

    print(f"\nChains with both bridges strong: {len(both_strong)}")
    print(f"Chains with one bridge strong: {len(one_strong)}")
    print(f"Chains with no bridges strong: {len(none_strong)}")

    # For chains with one strong bridge, try to find alternative strong pairs
    improved = list(both_strong)  # Start with all-strong chains

    for ca in one_strong:
        papers = ca["papers"]
        # Find which bridge is weak and look for alternatives
        for bpid, verdict in ca["bridge_verdicts"]:
            if verdict == "strong_chain":
                continue
            # This bridge is weak — can we find a strong alternative?
            p_src = pair_docs.get(bpid, ("", ""))
            alternatives = []
            for aid in strong_ids:
                alt_src = pair_docs.get(aid, ("", ""))
                if alt_src == p_src:
                    alternatives.append(aid)
            if alternatives:
                ca["_alternative_bridges"] = alternatives
                ca["_improved"] = True
                improved.append(ca)
                break

    print(f"\nPotentially improvable chains (1 strong + alternatives): {len(improved) - len(both_strong)}")
    print(f"Total candidate chains after improvement: {len(improved)}")

    # Build output: improved chain set
    improved_chains = []
    kept_ids = {ca["chain_id"] for ca in improved}

    for ca in chain_analysis:
        if ca["chain_id"] in kept_ids:
            # Find the original chain data
            orig = next((c for c in chains if c["chain_id"] == ca["chain_id"]), None)
            if orig:
                entry = dict(orig)
                entry["bridge_verdicts"] = [
                    {"pair_id": pid, "verdict": v}
                    for pid, v in ca["bridge_verdicts"]
                ]
                entry["all_bridges_strong"] = all(
                    v == "strong_chain" for _, v in ca["bridge_verdicts"]
                )
                entry["_improvement_notes"] = ca.get("_alternative_bridges", [])
                improved_chains.append(entry)

    # Save
    out_dir = ROOT / "data/05_eval/entity_bridge_chains_improved_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "improved_chains.json"
    out_path.write_text(
        json.dumps(improved_chains, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out_jsonl = out_dir / "improved_chains.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for c in improved_chains:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    all_strong_count = sum(1 for c in improved_chains if c.get("all_bridges_strong"))
    summary = {
        "input_chains": len(chains),
        "both_bridges_strong": len(both_strong),
        "one_bridge_strong_with_alternatives": len(improved) - len(both_strong),
        "total_improved_chains": len(improved_chains),
        "all_bridges_strong_count": all_strong_count,
        "strong_pair_pool": len(strong_ids),
        "improved_chains_papers": len({p for c in improved_chains for p in c.get("papers", [])}),
        "output": str(out_dir.relative_to(ROOT)),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSummary: {json.dumps(summary, indent=2)}")
    print(f"Output: {out_dir}")

    # Print the best chains
    print("\n--- Best chains (both bridges strong) ---")
    for ca in both_strong:
        print(f"  {ca['chain_id']}: papers={ca['papers']} entities={ca['shared_entities'][:3]}")

    print("\n--- Improvable chains (one strong + alternatives) ---")
    for ca in one_strong:
        if ca["chain_id"] in kept_ids:
            print(f"  {ca['chain_id']}: papers={ca['papers']} entities={ca['shared_entities'][:3]}")
            print(f"    Weak bridge alternatives: {ca.get('_alternative_bridges', [])}")


if __name__ == "__main__":
    main()
