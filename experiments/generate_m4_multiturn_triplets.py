#!/usr/bin/env python3
"""Generate M4 multi-turn triplets from enriched material packs.

Each material contains a 3-5 hop chain (element A → bridges → element B).
This generator produces 2-3 turn dialogues where:
  - Each turn advances along one hop of the chain
  - The query tests stateful retrieval: Turn N depends on Turn N-1's result
  - Mix of comprehension (understand relation) and computation (apply/decide)

Output: triplets ready for embedding training.
  Anchor = query text (turn N's question)
  Positive = the element pair + bridge that answers it
  Negatives = to be constructed downstream
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import call_llm, parse_json, set_company_credentials
from src.utils.token_logger import log_run

DEFAULT_MATERIALS = ROOT / "data/05_eval/m4_enriched_materials_latest/m4_material_pack.jsonl"
DEFAULT_OUT_ROOT = ROOT / "data/05_eval"
DEFAULT_MODEL = "gpt-5.4"


def configure_standard_logger() -> Path:
    from local_api_logger.logger import APILogger  # noqa: WPS433
    import local_api_logger.tracker as tracker  # noqa: WPS433

    log_dir = ROOT / "api_logs_cannt_delete"
    log_dir.mkdir(parents=True, exist_ok=True)
    tracker._default_tracker = tracker.APITracker(APILogger(str(log_dir)))
    return log_dir

MULTITURN_PROMPT = """You are converting a multimodal evidence CHAIN into a
multi-turn applied-problem dialogue for training an embedding model.

The embedding model will power a PPT/slide-generation agent. It needs to retrieve
the right multimodal elements across documents when a user asks multi-turn,
stateful questions.

### THE CHAIN

This is a {hop_count}-hop reasoning chain connecting elements across documents:

**Element A** ({type_a}):
{desc_a}

**Bridge 1** — how element A connects to the next step:
{bridge_1}

**Intermediate** ({type_mid}):
{desc_mid}

**Bridge 2** — how the intermediate connects to element B:
{bridge_2}

**Element B** ({type_b}):
{desc_b}

### YOUR TASK

Convert this chain into a {n_turns}-turn applied-problem dialogue.

Each turn maps to one step along the chain:
- Turn 1: user establishes a concrete scenario from Element A + Bridge 1.
  Assistant confirms, computes or interprets, and states a partial result.
- Turn 2: user asks a follow-up that requires the Intermediate + Bridge 2.
  The question depends on Turn 1's result — it cannot be answered standalone.
- [If 3 turns] Turn 3: user asks a final question requiring Element B,
  building on both previous turns.

### TURN STYLE MIX

Make {n_comprehension} turn(s) comprehension-style ("why does this pattern emerge?")
and {n_computation} turn(s) computation/decision-style ("which configuration is
optimal given the constraint from turn 1?").

### CONSTRAINTS

1. Turn 1 user message (20-70 words):
   - Concrete scenario with specific numbers, settings, or conditions from Element A
   - Natural domain language — DO NOT say "figure", "table", "equation", "chart"

2. Turn 1 assistant answer (30-120 words):
   - Apply Element A's content to the scenario
   - State a concrete intermediate result (number, condition, bound, ranking)
   - End signaling readiness: "Given this [result], we can now..."

3. Turn 2 user message (15-50 words):
   - MUST depend on Turn 1's result
   - Requires the Intermediate or Bridge 2 content to answer
   - Asks to compare, select, predict, or explain

4. Turn 2 assistant answer (40-200 words):
   - Step-by-step: (a) what does the intermediate/bridge tell us,
     (b) how does this interact with Turn 1's result,
     (c) concrete conclusion

5. Turn 3 (if applicable, 15-50 word question + 40-150 word answer):
   - Further depends on Turn 2's conclusion
   - Requires Element B

6. Evidence chunks (3-6 items):
   - Each: type, source_doc, label, what it provides
   - Must include Element A, Element B, and bridge-relevant content

7. Search clauses (3-5 items):
   - Realistic queries a researcher would type to find each piece

8. HARD BANS:
   - NO "figure", "table", "equation", "formula", "chart", "graph", "plot",
     "diagram", "panel", "subfigure" in user messages
   - NO template openings: "How does X relate to Y", "What is the relationship"
   - NO bare "this"/"that"/"these"/"those" as first word of a message
   - Questions MUST be answerable, not open-ended research questions

### OUTPUT FORMAT (valid JSON only)

{{
  "history": [
    {{"role": "user", "message": "...turn 1 setup..."}},
    {{"role": "assistant", "message": "...turn 1 answer..."}}
  ],
  "question": "...turn 2 question...",
  "answer_short": "...1-2 sentence final answer...",
  "answer_long": "...step-by-step reasoning for turn 2...",
  "turns": {n_turns},
  "turn_dependencies": [
    {{"turn": 2, "depends_on_turn": 1, "requires_element": "intermediate or bridge_2"}},
    {{"turn": 3, "depends_on_turn": [1, 2], "requires_element": "element_b"}}
  ],
  "chain_coverage": {{
    "element_a_used_in": [1],
    "bridge_1_used_in": [1],
    "intermediate_used_in": [2],
    "bridge_2_used_in": [2],
    "element_b_used_in": [3]
  }},
  "evidence_chunk_list": [
    {{"type": "figure|table|formula|text", "source": "doc_id", "label": "...", "provides": "..."}}
  ],
  "search_clause": ["query 1", "query 2", "query 3"],
  "style_breakdown": {{
    "comprehension_turns": {n_comprehension},
    "computation_turns": {n_computation}
  }}
}}"""


def _describe_element(elem: dict) -> str:
    """Build a natural language description from enriched element data."""
    parts = []
    for k in ("enriched_title", "enriched_content", "caption", "label"):
        v = elem.get(k, "")
        if v and isinstance(v, str) and len(v.strip().split()) >= 3:
            parts.append(str(v).strip())
            if sum(len(p) for p in parts) > 600:
                break
    if not parts:
        parts.append(f"{elem.get('element_type', 'element')}: {elem.get('element_id', '')}")
    return " ".join(parts)[:900]


def _extract_chain(pair: dict, elements_index: dict = None) -> dict:
    """Extract chain structure from a material pair.

    Returns a dict with element_a, bridges, intermediate, element_b info.
    """
    mc = pair.get("method_c", {})
    bridges = mc.get("compressed_bridge_summaries", [])
    chain_ids = mc.get("compressed_chain_ids", [])
    chain_types = mc.get("compressed_chain_types", [])

    ea = pair.get("element_a", {})
    eb = pair.get("element_b", {})

    # The compressed chain has 4 items: [A, intermediate, bridge1_marker, B]
    # or [A, bridge1, B, bridge2, C] for 5-hop
    # Map to our template: A → br1 → intermediate → br2 → B

    result = {
        "element_a": ea,
        "element_b": eb,
        "bridge_texts": bridges,
        "chain_ids": chain_ids,
        "chain_types": chain_types,
    }

    # Determine intermediate from chain
    if len(chain_ids) >= 4:
        # chain_ids[1] is typically the intermediate node
        result["intermediate"] = {
            "element_id": chain_ids[1] if len(chain_ids) > 1 else "",
            "element_type": chain_types[1] if len(chain_types) > 1 else "element",
            "description": bridges[0] if bridges else "connects A to B",
        }
    else:
        result["intermediate"] = {
            "element_id": "",
            "element_type": "bridge",
            "description": bridges[0] if bridges else "",
        }

    return result


def _count_hops(pair: dict) -> int:
    """Determine effective hops from chain structure."""
    chain = pair.get("method_c", {}).get("compressed_chain_ids", [])
    path = pair.get("path", [])
    return max(len(path), len(chain))


def main():
    parser = argparse.ArgumentParser(
        description="Generate M4 multi-turn triplets from enriched material chains.")
    parser.add_argument("--materials", default=str(DEFAULT_MATERIALS))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--api-url", default=os.environ.get("COMPANY_API_URL", ""))
    parser.add_argument("--api-key", default=os.environ.get("COMPANY_API_KEY", ""))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.out_root) / f"m4_multiturn_triplets_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = configure_standard_logger()
    if not args.dry_run and (not args.api_url or not args.api_key):
        raise SystemExit("company API credentials missing; set COMPANY_API_URL and COMPANY_API_KEY")
    set_company_credentials(args.api_url, args.api_key)

    print(f"Output: {out_dir}  |  Model: {args.model}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")

    materials = []
    with open(args.materials) as f:
        for line in f:
            if line.strip():
                materials.append(json.loads(line))
    selected = materials[args.skip:args.skip + args.limit]
    print(f"Loaded {len(materials)} materials, selected {len(selected)}")

    stats = Counter()
    total_in, total_out = 0, 0
    records_path = out_dir / "multiturn_triplets.jsonl"

    def append_record(record: dict) -> None:
        with records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    for i, mat in enumerate(selected):
        mid = mat.get("material_id", f"mat_{i}")
        hops = _count_hops(mat)
        chain = _extract_chain(mat)

        # Determine turns from hop count: 3-hop → 2 turns, 5-hop → 3 turns
        n_turns = 2 if hops <= 4 else 3
        n_comprehension = 1
        n_computation = n_turns - n_comprehension

        ea = chain["element_a"]
        eb = chain["element_b"]
        bridges = chain["bridge_texts"]
        mid_elem = chain["intermediate"]
        pair_type = mat.get("pair_type", f"{ea.get('element_type','?')}+{eb.get('element_type','?')}")

        br1 = bridges[0] if len(bridges) >= 1 else "connected through document structure"
        br2 = bridges[1] if len(bridges) >= 2 else bridges[0] if bridges else "connected through document structure"
        mid_desc = _describe_element(mid_elem) if mid_elem.get("description") else br1

        prompt = MULTITURN_PROMPT.format(
            hop_count=hops,
            n_turns=n_turns,
            n_comprehension=n_comprehension,
            n_computation=n_computation,
            type_a=ea.get("element_type", "element"),
            type_b=eb.get("element_type", "element"),
            type_mid=mid_elem.get("element_type", "bridge"),
            desc_a=_describe_element(ea),
            desc_b=_describe_element(eb),
            desc_mid=mid_desc,
            bridge_1=br1,
            bridge_2=br2,
        )

        prefix = f"[{i+1}/{len(selected)}] {mid} ({hops}-hop → {n_turns}-turn)"
        print(f"{prefix} ", end="", flush=True)

        if args.dry_run:
            print(f"| prompt={len(prompt)} chars")
            append_record({"material_id": mid, "prompt": prompt})
            continue

        raw = None
        gen_in = gen_out = 0
        last_error = ""
        for attempt in range(args.retries + 1):
            try:
                raw, gen_in, gen_out = call_llm(
                    client=None,
                    model=args.model,
                    prompt=prompt,
                    provider="company",
                    system_prompt=(
                        "You convert multimodal evidence chains into multi-turn applied "
                        "problems for embedding model training. Output valid JSON only."
                    ),
                    user_tag="m4_multiturn_v1",
                    temperature=0.8,
                )
                break
            except Exception as exc:  # noqa: BLE001 - keep batch alive on API/network failures
                last_error = repr(exc)
                if attempt < args.retries:
                    print(f"RETRY{attempt + 1} ", end="", flush=True)
                    continue
                stats["api_errors"] += 1
                print("API_ERROR")
                append_record({
                    "material_id": mid,
                    "parse_failed": True,
                    "api_error": last_error,
                    "tokens": {"in": gen_in, "out": gen_out},
                })
                raw = None
        total_in += gen_in
        total_out += gen_out
        stats["api_calls"] += 1
        if raw is None:
            continue

        parsed = parse_json(raw)
        if not parsed:
            stats["parse_failures"] += 1
            print("PARSE_FAIL")
            append_record({"material_id": mid, "raw": raw[:500] if raw else "", "parse_failed": True})
            continue

        # Extract docs from element IDs (material pack element_a.doc_id is unreliable)
        ea_id = ea.get("element_id", "")
        eb_id = eb.get("element_id", "")
        source_doc = ea_id.split("_", 1)[0] if "_" in ea_id else ""
        target_doc = eb_id.split("_", 1)[0] if "_" in eb_id else ""

        record = {
            "triplet_id": mid,
            "material_id": mid,
            "pair_id": mat.get("pair_id", ""),
            "hop_count": hops,
            "n_turns": n_turns,
            "pair_type": pair_type,
            "element_a_id": ea_id,
            "element_b_id": eb_id,
            "element_a_type": ea.get("element_type", ""),
            "element_b_type": eb.get("element_type", ""),
            "source_doc": source_doc,
            "target_doc": target_doc,
            "bridge_texts": bridges,
            "generated": parsed,
            "tokens": {"in": gen_in, "out": gen_out},
        }
        append_record(record)
        stats["generated"] += 1
        stats[f"turns={n_turns}"] += 1

        q = parsed.get("question", "")
        print(f"| {n_turns}t | {len(q)} chars")

    summary = {
        "status": "ok",
        "output_dir": str(out_dir),
        "model": args.model,
        "materials_processed": len(selected),
        "generated": stats.get("generated", 0),
        "parse_failures": stats.get("parse_failures", 0),
        "turn_distribution": {k: v for k, v in stats.items() if k.startswith("turns=")},
        "api_calls": stats.get("api_calls", 0),
        "api_errors": stats.get("api_errors", 0),
        "tokens": {"in": total_in, "out": total_out},
        "files": {
            "local_api_logger": str(log_dir.relative_to(ROOT)),
            "token_db": "logs/token_usage.db",
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log_run(
        script="experiments/generate_m4_multiturn_triplets.py",
        model=f"company:{args.model}",
        purpose="Generate M4 multi-turn triplets from material chains",
        input_tokens=total_in,
        output_tokens=total_out,
        extra={
            "pairs_processed": len(selected),
            "queries_written": stats.get("generated", 0),
            "parse_failures": stats.get("parse_failures", 0),
            "output": str(out_dir),
        },
    )

    latest = Path(args.out_root) / "m4_multiturn_triplets_latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_dir, target_is_directory=True)

    ng = summary["generated"]
    nf = summary["parse_failures"]
    print(f"\nDone. {ng} triplets ({nf} parse failures) | "
          f"tokens: {total_in} in / {total_out} out")
    print(f"Latest: {latest}")


if __name__ == "__main__":
    main()
