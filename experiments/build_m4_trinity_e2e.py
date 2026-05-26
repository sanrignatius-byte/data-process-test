#!/usr/bin/env python3
"""End-to-end M4 trinity pipeline: cross-doc chains → M4 materials → multi-turn sessions.

Takes the best entity-bridge chains (keep + review from chain judge),
generates M4 materials with enriched element data, then projects to
multi-turn sessions following idea:005→006→007.

This is Track A / experimental-lane.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import call_llm, parse_json, set_company_credentials
from src.utils.token_logger import log_run

DEFAULT_MODEL = "gpt-5.4"

MATERIAL_SYSTEM = """You are building structured M4 (multi-hop, multi-modal, multi-document) materials for query generation.

Given a cross-document evidence chain with multiple multimodal elements, produce a structured material pack.

Rules:
- evidence_summary: a 2-3 sentence summary of the chain's evidence path and what scientific question it answers
- bridge_description: a 1-sentence description of how the cross-document bridges connect
- key_entities: list the 3-5 most specific shared entities/concepts
- reasoning_chain: a 4-6 sentence paragraph tracing the reasoning from element A through the bridge to element C, explaining what each element contributes

Return only valid JSON:
{
  "evidence_summary": "...",
  "bridge_description": "...",
  "key_entities": ["...", "..."],
  "reasoning_chain": "..."
}"""

SESSION_SYSTEM = """You are converting a cross-document multi-hop evidence chain into a natural 2-turn research dialogue.

CRITICAL RULES:
1. Turn 1 question: ask about element A's finding, using concrete details from the reasoning chain.
2. Turn 1 answer: explain what element A shows, including specific numbers/terms.
3. Turn 2 question: MUST reference a specific finding from Turn 1's answer. A reader seeing Turn 2 alone must need Turn 1 first.
4. Turn 2 answer: begin by restating the key Turn 1 finding, then add what element C shows, explaining how the bridge connects them.
5. NO meta-language: figure, table, equation, chart, plot, diagram, paper, section, study.
6. Use pronouns and referring expressions naturally ("this finding", "that trend", "given this", "under these conditions").
7. Keep each turn concise (2-3 sentences).

Return only valid JSON:
{
  "turn1_user": "...",
  "turn1_assistant": "...",
  "turn2_user": "...",
  "turn2_assistant": "..."
}"""


def load_enriched() -> dict[str, dict[str, Any]]:
    path = ROOT / "data/02_enriched/multimodal_elements_enriched.json"
    idx: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return idx
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for doc_id, doc in data.get("documents", {}).items():
        for eid, el in doc.get("elements", {}).items():
            # eid is already prefixed: "1104.3913_figure_2"
            idx[eid] = {
                "doc_id": doc_id,
                "element_id": eid,
                "type": el.get("element_type", "?"),
                "title": el.get("enriched_title", "") or "",
                "content": el.get("enriched_content", "") or "",
                "caption": el.get("caption", "") or "",
            }
    return idx


def load_entity_bridge_chains() -> list[dict[str, Any]]:
    """Load chains filtered to keep + review from chain judge."""
    raw = json.loads(
        (ROOT / "data/05_eval/cross_doc_chains_final_fixed.json").read_text("utf-8")
    )
    chains = raw.get("chains", raw) if isinstance(raw, dict) else raw

    # Load production decisions
    keep_ids: set[str] = set()
    review_ids: set[str] = set()
    for label, path in [
        ("keep", "data/05_eval/cross_doc_chain_judge_fixed/keep_chains.jsonl"),
        ("review", "data/05_eval/cross_doc_chain_judge_fixed/review_chains.jsonl"),
    ]:
        p = ROOT / path
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        cid = json.loads(line).get("chain_id", "")
                        if label == "keep":
                            keep_ids.add(cid)
                        else:
                            review_ids.add(cid)

    selected = keep_ids | review_ids
    return [c for c in chains if c["chain_id"] in selected]


def build_material_prompt(chain: dict[str, Any], elem_data: dict[str, Any]) -> str:
    papers = chain.get("papers", [])
    elements = []  # list of (elem_key, type, caption/content)
    for elem_key, info in elem_data.items():
        elements.append({
            "key": elem_key,
            "type": info["type"],
            "title": info.get("title", "") or "",
            "caption": info.get("caption", "") or "",
            "content": (info.get("content", "") or "")[:600],
        })

    elem_desc = "\n".join(
        f"  [{e['type']}] {e['key']}: {e['title'][:200]} | {e['caption'][:200]} | {e['content'][:200]}"
        for e in elements
    )

    return f"""Build a structured M4 material from this cross-document evidence chain.

Papers: {', '.join(papers)}
Shared entities: {', '.join(chain.get('shared_entities', [])[:8])}
Element types: {', '.join(chain.get('element_types', []))}

Elements:
{elem_desc}

Describe the evidence chain, bridge connections, and write a reasoning chain that traces through the elements."""


def build_session_prompt(material: dict[str, Any], chain: dict[str, Any]) -> str:
    return f"""Convert this cross-document evidence chain into a natural 2-turn research dialogue.

Evidence summary: {material.get('evidence_summary', '')}
Bridge: {material.get('bridge_description', '')}
Reasoning chain: {material.get('reasoning_chain', '')}
Key entities: {', '.join(material.get('key_entities', []))}

The two endpoint elements are from different papers, connected by a scientific bridge.
Turn 1 should ask about the first element's finding.
Turn 2 should reference Turn 1's answer and ask about how it connects to the second element's finding."""


def run_turn_dependency_qc(session: dict[str, Any], model: str) -> dict[str, Any]:
    prompt = f"""Test if Turn 2 genuinely depends on Turn 1's answer.

Turn 1 (your answer was erased):
Q: {session.get('turn1_user', '')}

Turn 2:
Q: {session.get('turn2_user', '')}

Can you answer Turn 2 without knowing what you said in Turn 1?
- If Turn 2 references a specific finding from Turn 1's erased answer -> DEPENDENT
- If Turn 2 can be answered from general knowledge -> INDEPENDENT

Return JSON:
{{
  "verdict": "dependent | independent",
  "explanation": "which specific information Turn 2 needs from Turn 1"
}}"""

    raw, tin, tout = call_llm(
        client=None, model=model, provider="company",
        prompt=prompt,
        system_prompt="You are a turn-dependency evaluator. Return valid JSON only.",
        max_tokens=200, temperature=0.0,
        user_tag="m4_trinity_qc",
    )
    parsed = parse_json(raw or "")
    return {
        "verdict": parsed.get("verdict", "parse_failed") if isinstance(parsed, dict) else "parse_failed",
        "explanation": parsed.get("explanation", "") if isinstance(parsed, dict) else "",
        "tokens": {"in": tin, "out": tout},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("COMPANY_API_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-qc", action="store_true")
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = ROOT / f"data/05_eval/m4_trinity_e2e_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    latest = ROOT / "data/05_eval/m4_trinity_e2e_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out_dir.resolve())

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

    # Load data
    enriched = load_enriched()
    print(f"Loaded {len(enriched)} enriched elements")

    chains = load_entity_bridge_chains()
    print(f"Loaded {len(chains)} keep+review chains")

    if args.limit:
        chains = chains[:args.limit]

    total_in = 0
    total_out = 0
    materials: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []

    print(f"\nProcessing {len(chains)} chains...")

    for idx, chain in enumerate(chains, 1):
        cid = chain["chain_id"]

        # Step 1: Gather element data
        elem_data: dict[str, Any] = {}
        chain_elements = chain.get("elements", [])
        for elem_entry in chain_elements:
            if isinstance(elem_entry, dict):
                elem_id = elem_entry.get("element_id", "")
                if elem_id in enriched:
                    elem_data[elem_id] = enriched[elem_id]

        if not elem_data:
            print(f"[{idx:03d}/{len(chains):03d}] {cid} -> SKIP (no element data)")
            continue

        # Step 2: Generate M4 material
        mprompt = build_material_prompt(chain, elem_data)
        raw_m, tin_m, tout_m = call_llm(
            client=None, model=args.model, provider="company",
            prompt=mprompt, system_prompt=MATERIAL_SYSTEM,
            max_tokens=800, temperature=0.0,
            user_tag="m4_trinity_material",
        )
        total_in += tin_m
        total_out += tout_m
        material = parse_json(raw_m or "")

        if not isinstance(material, dict):
            print(f"[{idx:03d}/{len(chains):03d}] {cid} -> MATERIAL_PARSE_FAIL")
            continue

        material["chain_id"] = cid
        material["papers"] = chain.get("papers", [])
        material["element_types"] = chain.get("element_types", [])
        materials.append(material)

        # Step 3: Generate multi-turn session
        sprompt = build_session_prompt(material, chain)
        raw_s, tin_s, tout_s = call_llm(
            client=None, model=args.model, provider="company",
            prompt=sprompt, system_prompt=SESSION_SYSTEM,
            max_tokens=800, temperature=0.4,
            user_tag="m4_trinity_session",
        )
        total_in += tin_s
        total_out += tout_s
        session = parse_json(raw_s or "")

        if not isinstance(session, dict):
            print(f"[{idx:03d}/{len(chains):03d}] {cid} -> SESSION_PARSE_FAIL")
            continue

        # Step 4: Turn-dependency QC
        qc = None
        if not args.skip_qc:
            qc = run_turn_dependency_qc(session, args.model)
            total_in += qc["tokens"]["in"]
            total_out += qc["tokens"]["out"]

        session_row = {
            "session_id": f"trinity_{cid}",
            "chain_id": cid,
            "papers": chain.get("papers", []),
            "element_types": chain.get("element_types", []),
            "shared_entities": chain.get("shared_entities", []),
            "turn1_user": session.get("turn1_user", ""),
            "turn1_assistant": session.get("turn1_assistant", ""),
            "turn2_user": session.get("turn2_user", ""),
            "turn2_assistant": session.get("turn2_assistant", ""),
            "material": material,
            "turn_dependency_qc": qc,
        }
        sessions.append(session_row)

        dep_str = qc["verdict"] if qc else "skipped"
        print(f"[{idx:03d}/{len(chains):03d}] {cid} -> session={session_row['session_id']} dep={dep_str}")

    # Save outputs
    materials_path = out_dir / "materials.jsonl"
    sessions_path = out_dir / "sessions.jsonl"
    for m in materials:
        materials_path.open("a", encoding="utf-8").write(json.dumps(m, ensure_ascii=False) + "\n")
    for s in sessions:
        sessions_path.open("a", encoding="utf-8").write(json.dumps(s, ensure_ascii=False) + "\n")

    dep_count = sum(1 for s in sessions if s.get("turn_dependency_qc", {}).get("verdict") == "dependent")
    ind_count = sum(1 for s in sessions if s.get("turn_dependency_qc", {}).get("verdict") == "independent")

    summary = {
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "input_chains": len(chains),
        "materials_generated": len(materials),
        "sessions_generated": len(sessions),
        "turn_dependency": {
            "total": len(sessions),
            "dependent": dep_count,
            "independent": ind_count,
            "dependency_rate": round(dep_count / len(sessions), 4) if sessions else 0,
        },
        "tokens": {"in": total_in, "out": total_out},
        "files": {
            "materials": str(materials_path.relative_to(ROOT)),
            "sessions": str(sessions_path.relative_to(ROOT)),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\nSummary: {json.dumps(summary, indent=2)}")
    print(f"Output: {out_dir}")

    log_run(
        script="experiments/build_m4_trinity_e2e.py",
        model=f"company:{args.model}",
        purpose="M4 trinity end-to-end: chains → materials → multi-turn sessions",
        input_tokens=total_in,
        output_tokens=total_out,
        extra={"output": str(out_dir.relative_to(ROOT))},
    )


if __name__ == "__main__":
    main()
