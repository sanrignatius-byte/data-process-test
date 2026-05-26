#!/usr/bin/env python3
"""BMGQ-style NLI relation typing on graph paths.

Reads enriched pair candidates, extracts graph paths, and classifies each hop's
relation type using LLM. Filters paths that don't form causal chains.

Output: typed_paths.jsonl with relation labels per hop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.llm import call_llm, parse_json, set_company_credentials
from src.utils.token_logger import log_run

DEFAULT_MODEL = "gpt-5.4"

# BMGQ-style relation types
RELATION_TYPES = [
    "observation",    # Direct reading from data/visualization
    "attribution",    # Attributing an effect to a cause
    "explanation",    # Theoretical/mathematical explanation
    "verification",   # Confirming a claim with additional evidence
    "prediction",     # Deriving a prediction from prior steps
    "background",     # Contextual/background information (not part of causal chain)
]

NLI_SYSTEM_PROMPT = """You are a scientific reasoning classifier. Given:
- source_element: the starting element (figure/table/formula) with caption and content
- target_element: the ending element with caption and content
- bridge_context: the paragraph text connecting them

Classify the reasoning relationship from source to target as EXACTLY ONE of:
- observation: source provides a direct empirical finding that the bridge interprets
- attribution: bridge attributes an effect shown in source to a specific cause
- explanation: bridge provides theoretical/mathematical reasoning that explains source
- verification: bridge uses target to verify/validate a claim from source
- prediction: bridge derives a prediction from source that target materializes
- background: source and bridge share context but don't form a reasoning step

Also rate relevance (0.0-1.0) and provide a one-sentence justification.

Return only valid JSON:
{
  "relation": "<one of the six types>",
  "relevance": 0.85,
  "justification": "<one sentence>",
  "is_causal": true/false
}

Causal relations are: observation, attribution, explanation, verification, prediction.
Background is NOT causal."""


def load_candidates(candidates_path: Path) -> List[Dict[str, Any]]:
    with open(candidates_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = data.get("pairs", data.get("candidates", []))
    if not pairs and isinstance(data, list):
        pairs = data
    return pairs


def extract_path_hops(pair: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract individual hops from a pair's path."""
    path = pair.get("path", [])
    hops = []
    for i in range(len(path) - 1):
        src = path[i]
        tgt = path[i + 1]
        # Skip adjacent paragraph nodes (same-doc backbone)
        hops.append({"source": src, "target": tgt, "hop_index": i})
    return hops


def build_hop_context(
    hop: Dict[str, Any],
    pair: Dict[str, Any],
    elements: Dict[str, Dict[str, Any]],
) -> str:
    """Build context for classifying one hop."""
    src_info = elements.get(hop["source"], {})
    tgt_info = elements.get(hop["target"], {})

    parts = []
    for label, info in [("source", src_info), ("target", tgt_info)]:
        etype = info.get("element_type", "unknown")
        caption = info.get("caption", "") or info.get("enriched_content", "") or ""
        content = info.get("content", "") or ""
        parts.append(
            f"{label} ({etype}): caption={caption[:300]}, content={content[:200]}"
        )

    # Add bridge context
    edge_contexts = pair.get("edge_contexts", [])
    bridge_text = ""
    for ec in edge_contexts:
        if isinstance(ec, dict):
            bridge_text += ec.get("context", "") + " "
        else:
            bridge_text += str(ec) + " "
    if bridge_text.strip():
        parts.append(f"bridge_context: {bridge_text[:400]}")

    return "\n".join(parts)


def classify_hop(
    hop_context: str,
    model: str,
    client: Any,
    total_in: int,
    total_out: int,
) -> Tuple[Optional[Dict], int, int]:
    """Classify one hop using LLM."""
    raw, tin, tout = call_llm(
        client=client,
        model=model,
        provider="company",
        prompt=hop_context,
        system_prompt=NLI_SYSTEM_PROMPT,
        max_tokens=200,
        temperature=0.0,
        user_tag="nli_relation_type",
    )
    total_in += tin
    total_out += tout
    parsed = parse_json(raw or "")
    if isinstance(parsed, dict):
        return parsed, total_in, total_out
    return None, total_in, total_out


def is_valid_causal_chain(relations: List[Dict]) -> Tuple[bool, str]:
    """Check if a sequence of relations forms a valid causal chain.

    Valid patterns (BMGQ-inspired):
    - observation → attribution → explanation
    - observation → verification → prediction
    - observation → explanation → prediction
    - observation → attribution → verification → prediction
    """
    if not relations:
        return False, "empty_chain"

    rel_seq = [r.get("relation", "unknown") for r in relations]

    # All must be causal
    for r in relations:
        if not r.get("is_causal", False):
            return False, f"non_causal_hop: {r.get('relation')}"

    # Must start with observation (empirical grounding)
    if rel_seq[0] != "observation":
        return False, f"chain_starts_with_{rel_seq[0]}_not_observation"

    # Must end with explanation, prediction, or verification
    valid_endings = {"explanation", "prediction", "verification"}
    if rel_seq[-1] not in valid_endings:
        return False, f"chain_ends_with_{rel_seq[-1]}"

    # Must have at least one intermediate reasoning step (not all observation)
    if len(set(rel_seq)) < 2:
        return False, "all_same_relation_type"

    return True, f"valid_causal_chain: {' → '.join(rel_seq)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        default="data/02_enriched/hub_candidates_enriched_v4_intra_doc.json",
        help="Path to enriched pair candidates JSON",
    )
    parser.add_argument(
        "--elements",
        default="data/01_graphs/multimodal_elements.json",
        help="Path to multimodal elements JSON",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default=os.environ.get("COMPANY_API_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    # Setup
    from local_api_logger.logger import APILogger
    import local_api_logger.tracker as tracker

    log_dir = ROOT / "api_logs_cannt_delete"
    log_dir.mkdir(parents=True, exist_ok=True)
    tracker._default_tracker = tracker.APITracker(APILogger(str(log_dir)))

    set_company_credentials(
        os.environ.get("COMPANY_API_URL", ""),
        os.environ.get("COMPANY_API_KEY", ""),
    )

    # Output dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = ROOT / f"data/05_eval/relation_typing_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    candidates_path = ROOT / args.candidates
    if not candidates_path.exists():
        print(f"ERROR: candidates file not found: {candidates_path}")
        sys.exit(1)
    pairs = load_candidates(candidates_path)
    print(f"Loaded {len(pairs)} candidate pairs")

    elements_path = ROOT / args.elements
    elements: Dict[str, Dict[str, Any]] = {}
    if elements_path.exists():
        with open(elements_path, "r", encoding="utf-8") as f:
            elem_data = json.load(f)
        docs = elem_data.get("documents", {})
        for doc_id, doc in docs.items():
            for eid, el in doc.get("elements", {}).items():
                elements[eid] = el
        print(f"Loaded {len(elements)} elements")

    if args.limit:
        pairs = pairs[: args.limit]
        print(f"Limited to {len(pairs)} pairs")

    # Process
    typed_paths_path = out_dir / "typed_paths.jsonl"
    summary_path = out_dir / "summary.json"

    total_in = 0
    total_out = 0
    stats = Counter()
    client = None  # company provider doesn't need client

    print(f"\n--- Classifying relations for {len(pairs)} pairs ---")
    for idx, pair in enumerate(pairs, 1):
        pair_id = pair.get("pair_id", f"unknown_{idx}")
        hops = extract_path_hops(pair)

        if len(hops) < 1:
            stats["no_hops"] += 1
            continue

        hop_results = []
        chain_valid = True
        for hop in hops:
            ctx = build_hop_context(hop, pair, elements)
            result, total_in, total_out = classify_hop(
                ctx, args.model, client, total_in, total_out
            )
            if result:
                hop_results.append(result)
            else:
                hop_results.append({"relation": "parse_failed", "is_causal": False})
                chain_valid = False

            time.sleep(args.delay)

        # Validate chain
        valid, reason = is_valid_causal_chain(hop_results)
        if not valid:
            stats[f"invalid_chain:{reason}"] += 1
            stats["chains_invalid"] += 1
        else:
            stats["chains_valid"] += 1

        # Also check HopWeaver constraints
        elem_ids = pair.get("element_ids", [])
        doc_ids = set()
        for eid in elem_ids:
            doc_ids.add(eid.split("::")[0] if "::" in eid else eid)
        has_fact_dist = len(doc_ids) >= 2
        has_shortcut = len(doc_ids) == 1 and len(elem_ids) >= 2

        output = {
            "pair_id": pair_id,
            "hop_count": len(hops),
            "hop_results": hop_results,
            "chain_valid": valid,
            "chain_reason": reason,
            "fact_distribution": has_fact_dist,
            "no_shortcut_violation": not has_shortcut,
            "relation_sequence": [r.get("relation") for r in hop_results],
            "doc_ids": list(doc_ids),
        }

        with typed_paths_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            f.flush()

        if idx % 50 == 0 or idx == len(pairs):
            print(
                f"[{idx:04d}/{len(pairs):04d}] "
                f"valid={stats['chains_valid']} "
                f"invalid={stats['chains_invalid']} "
                f"(tok_in={total_in}, tok_out={total_out})"
            )

    # Summary
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "total_pairs": len(pairs),
        "chains_valid": stats["chains_valid"],
        "chains_invalid": stats["chains_invalid"],
        "valid_rate": round(
            stats["chains_valid"] / max(stats["chains_valid"] + stats["chains_invalid"], 1), 4
        ),
        "failure_reasons": {
            k: v for k, v in stats.items() if k.startswith("invalid_chain:")
        },
        "tokens": {"in": total_in, "out": total_out},
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary written to {summary_path}")
    print(f"Typed paths written to {typed_paths_path}")

    log_run(
        script="type_graph_relations",
        model=f"company:{args.model}",
        purpose=f"BMGQ NLI relation typing on {len(pairs)} graph paths",
        input_tokens=total_in,
        output_tokens=total_out,
        extra={
            "pairs_processed": len(pairs),
            "chains_valid": stats["chains_valid"],
            "output": str(out_dir),
        },
    )


if __name__ == "__main__":
    main()
