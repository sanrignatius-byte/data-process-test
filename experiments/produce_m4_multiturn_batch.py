#!/usr/bin/env python3
"""Batch production: entity-bridge pairs → chains → M4 materials → multi-turn sessions.

Uses judged strong + weak_but_related pairs (74 total) to build chains,
then runs the full material→session pipeline with proper API logging.

All API calls go through call_llm(provider="company") → local_api_logger.
Each run is recorded via log_run() → logs/token_usage.db.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import call_llm, parse_json, set_company_credentials
from src.utils.token_logger import log_run

DEFAULT_MODEL = "gpt-5.4"

MATERIAL_SYSTEM = """You are building structured M4 (multi-hop, multi-modal, multi-document) materials.

Given a cross-document evidence chain, produce a structured material pack.

Rules:
- evidence_summary: 2-3 sentence summary of the chain's evidence path
- bridge_description: 1 sentence on how the cross-document bridges connect
- key_entities: 3-5 most specific shared entities
- reasoning_chain: 4-6 sentence paragraph tracing reasoning from first element through bridge to last element

Return only valid JSON with keys: evidence_summary, bridge_description, key_entities, reasoning_chain"""

SESSION_SYSTEM = """Convert a cross-document evidence chain into a natural 2-turn research dialogue.

CRITICAL:
1. Turn 1: ask about the first element's finding with concrete details
2. Turn 1 answer: explain what it shows with specific numbers/terms
3. Turn 2: MUST reference a specific finding from Turn 1's answer. A reader seeing Turn 2 alone MUST need Turn 1 first.
4. Turn 2 answer: restate key Turn 1 finding, then add new evidence from bridge+second element
5. NO meta-language: figure, table, equation, chart, plot, diagram, paper, section
6. Use pronouns naturally ("this finding", "given that", "under those conditions")
7. Keep each turn 2-3 sentences

Return only valid JSON: {"turn1_user":"...","turn1_assistant":"...","turn2_user":"...","turn2_assistant":"..."}"""


def load_judged_pair_ids() -> list[str]:
    """Load candidate_ids of strong + weak_but_related judged pairs."""
    ids = []
    path = ROOT / "data/05_eval/entity_bridge_judge_v2/judgments.jsonl"
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            v = j.get("judgment", {}).get("verdict", "") if isinstance(j.get("judgment"), dict) else ""
            if v in ("strong_chain", "weak_but_related"):
                ids.append(j["candidate_id"])
    return ids


def load_pair_details() -> dict[str, dict[str, Any]]:
    """Load raw pair details keyed by candidate_id."""
    pairs: dict[str, dict[str, Any]] = {}
    path = ROOT / "data/05_eval/entity_bridge_candidates_v2/judge_pack.jsonl"
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            pairs[p["candidate_id"]] = p
    return pairs


def load_enriched() -> dict[str, dict[str, Any]]:
    """Load enriched element index."""
    idx: dict[str, dict[str, Any]] = {}
    path = ROOT / "data/02_enriched/multimodal_elements_enriched.json"
    if not path.exists():
        return idx
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for doc_id, doc in data.get("documents", {}).items():
        for eid, el in doc.get("elements", {}).items():
            idx[eid] = {
                "doc_id": doc_id,
                "type": el.get("element_type", "?"),
                "title": el.get("enriched_title", "") or "",
                "content": el.get("enriched_content", "") or "",
                "caption": el.get("caption", "") or "",
            }
    return idx


def build_chains_from_pairs(
    judged_ids: list[str],
    pair_details: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build 2-hop chains from judged pairs across 3 papers."""
    # Map pair_id -> (source_doc, target_doc, shared_entities)
    pair_info: dict[str, tuple[str, str, list[str]]] = {}
    for pid in judged_ids:
        if pid in pair_details:
            pd = pair_details[pid]
            pair_info[pid] = (
                pd["source_doc"],
                pd["target_doc"],
                pd.get("_meta", {}).get("shared_entities", []),
            )

    # Build simple adjacency: doc -> set of neighbor docs
    neighbors: dict[str, set[str]] = defaultdict(set)
    # Also keep the pair info per doc pair: (src,tgt) -> [(pid, ents)]
    pair_registry: dict[tuple[str, str], list[tuple[str, list[str]]]] = defaultdict(list)
    for pid, (src, tgt, ents) in pair_info.items():
        neighbors[src].add(tgt)
        neighbors[tgt].add(src)
        pair_registry[(src, tgt)].append((pid, ents))

    # Find chains: a->middle->c where middle connects to both a and c
    chains: list[dict[str, Any]] = []
    seen_triples: set[tuple[str, str, str]] = set()
    for middle, nb in neighbors.items():
        nb_list = sorted(nb)
        for i in range(len(nb_list)):
            for j in range(i + 1, len(nb_list)):
                a, c = nb_list[i], nb_list[j]
                triple = tuple(sorted([a, middle, c]))
                if triple in seen_triples:
                    continue
                seen_triples.add(triple)

                # Find the specific pairs connecting (a,middle) and (middle,c)
                pairs_am = pair_registry.get((a, middle), []) + pair_registry.get((middle, a), [])
                pairs_mc = pair_registry.get((middle, c), []) + pair_registry.get((c, middle), [])

                if not pairs_am or not pairs_mc:
                    continue

                pid1, ents1 = pairs_am[0]
                pid2, ents2 = pairs_mc[0]

                # Collect elements from both pairs
                elements: list[dict[str, Any]] = []
                for pid, src_doc, tgt_doc in [(pid1, a, middle), (pid2, middle, c)]:
                    pd = pair_details.get(pid, {})
                    for role, doc, key in [
                        ("source", src_doc, "source_element_id"),
                        ("target", tgt_doc, "target_element_id"),
                    ]:
                        eid = pd.get(key, "")
                        etype = pd.get(f"{role}_element_type", "?")
                        if eid:
                            elements.append({
                                "element_id": eid,
                                "element_type": etype,
                                "doc_id": doc,
                            })

                # Merge shared entities
                all_ents = list(set(ents1 + ents2))
                chain_id = f"prod_{a}_{middle}_{c}"

                chains.append({
                    "chain_id": chain_id,
                    "papers": [a, middle, c],
                    "n_papers": 3,
                    "n_elements": len(elements),
                    "n_bridges": 2,
                    "shared_entities": all_ents,
                    "element_types": [e["element_type"] for e in elements],
                    "elements": elements,
                    "bridge_pairs": [
                        {"pair_id": pid1, "from_doc": a, "to_doc": middle},
                        {"pair_id": pid2, "from_doc": middle, "to_doc": c},
                    ],
                })

    return chains


def build_material_prompt(chain: dict[str, Any]) -> str:
    papers = chain["papers"]
    elements = chain.get("elements", [])
    elem_desc = "\n".join(
        f"  [{e.get('element_type','?')}] {e.get('element_id','?')} (role: {e.get('role','?')})"
        for e in elements
    )
    return f"""Build a structured M4 material from this cross-document evidence chain.

Papers: {', '.join(papers)}
Shared entities: {', '.join(chain.get('shared_entities', [])[:8])}

Elements:
{elem_desc}

Describe the evidence chain, bridges, key entities, and write a reasoning chain tracing through the elements."""


def build_session_prompt(material: dict[str, Any], chain: dict[str, Any]) -> str:
    return f"""Convert this cross-document evidence chain into a natural 2-turn research dialogue.

Evidence summary: {material.get('evidence_summary', '')}
Bridge: {material.get('bridge_description', '')}
Reasoning chain: {material.get('reasoning_chain', '')}
Key entities: {', '.join(material.get('key_entities', []))}

Turn 1 asks about the first element's finding. Turn 2 references Turn 1's answer and asks about how the bridge connects to the second element."""


def run_turn_dependency_qc(session: dict[str, Any], model: str) -> dict[str, Any]:
    prompt = f"""Test if Turn 2 genuinely depends on Turn 1's answer.

Turn 1 (your answer was erased):
Q: {session.get('turn1_user', '')}

Turn 2:
Q: {session.get('turn2_user', '')}

Can you answer Turn 2 without your erased Turn 1 answer?
- If Turn 2 references a specific finding from Turn 1 → DEPENDENT
- If Turn 2 is answerable from general knowledge → INDEPENDENT

Return JSON: {{"verdict": "dependent | independent", "explanation": "..."}}"""

    raw, tin, tout = call_llm(
        client=None, model=model, provider="company",
        prompt=prompt,
        system_prompt="Turn-dependency evaluator. Return valid JSON only.",
        max_tokens=200, temperature=0.0,
        user_tag="m4_produce_qc",
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

    # Output setup
    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = ROOT / f"data/05_eval/m4_produce_batch_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    latest = ROOT / "data/05_eval/m4_produce_batch_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out_dir.resolve())

    # API logging setup
    from local_api_logger.logger import APILogger
    import local_api_logger.tracker as tracker
    log_dir = ROOT / "api_logs_cannt_delete"
    log_dir.mkdir(parents=True, exist_ok=True)
    tracker._default_tracker = tracker.APITracker(APILogger(str(log_dir)))
    set_company_credentials(
        os.environ.get("COMPANY_API_URL", ""),
        os.environ.get("COMPANY_API_KEY", ""),
    )

    # Step 1: Load data
    print("=== Step 1: Load entity-bridge pairs ===")
    judged_ids = load_judged_pair_ids()
    pair_details = load_pair_details()
    enriched = load_enriched()
    print(f"Usable judged pairs (strong + weak_but_related): {len(judged_ids)}")
    print(f"Pair details: {len(pair_details)}")
    print(f"Enriched elements: {len(enriched)}")

    # Step 2: Build chains
    print("\n=== Step 2: Build chains from pairs ===")
    chains = build_chains_from_pairs(judged_ids, pair_details)
    print(f"Built {len(chains)} unique chains")

    # Filter: only chains where ALL elements have enriched data
    valid_chains = []
    for c in chains:
        all_have_enriched = all(
            e["element_id"] in enriched
            for e in c.get("elements", [])
        )
        if all_have_enriched:
            valid_chains.append(c)

    print(f"Chains with all elements enriched: {len(valid_chains)}")
    if args.limit:
        valid_chains = valid_chains[:args.limit]
        print(f"Limited to {len(valid_chains)} chains")

    # Save chain inventory
    chains_path = out_dir / "chains.json"
    chains_path.write_text(json.dumps(valid_chains, ensure_ascii=False, indent=2), encoding="utf-8")

    # Step 3: Generate materials + sessions
    print(f"\n=== Step 3: Generate materials + sessions ({len(valid_chains)} chains) ===")
    total_in = 0
    total_out = 0
    materials_path = out_dir / "materials.jsonl"
    sessions_path = out_dir / "sessions.jsonl"
    qc_path = out_dir / "qc_results.jsonl"
    sessions: list[dict[str, Any]] = []

    for idx, chain in enumerate(valid_chains, 1):
        cid = chain["chain_id"]

        # Material
        mprompt = build_material_prompt(chain)
        raw_m, tin_m, tout_m = call_llm(
            client=None, model=args.model, provider="company",
            prompt=mprompt, system_prompt=MATERIAL_SYSTEM,
            max_tokens=800, temperature=0.0,
            user_tag="m4_produce_material",
        )
        total_in += tin_m
        total_out += tout_m
        material = parse_json(raw_m or "")

        if not isinstance(material, dict):
            print(f"[{idx:03d}/{len(valid_chains):03d}] {cid} -> MATERIAL_PARSE_FAIL")
            continue

        material["chain_id"] = cid
        material["papers"] = chain["papers"]
        with materials_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(material, ensure_ascii=False) + "\n")

        # Session
        sprompt = build_session_prompt(material, chain)
        raw_s, tin_s, tout_s = call_llm(
            client=None, model=args.model, provider="company",
            prompt=sprompt, system_prompt=SESSION_SYSTEM,
            max_tokens=800, temperature=0.4,
            user_tag="m4_produce_session",
        )
        total_in += tin_s
        total_out += tout_s
        session = parse_json(raw_s or "")

        if not isinstance(session, dict):
            print(f"[{idx:03d}/{len(valid_chains):03d}] {cid} -> SESSION_PARSE_FAIL")
            continue

        # QC
        qc = None
        if not args.skip_qc:
            qc = run_turn_dependency_qc(session, args.model)
            total_in += qc["tokens"]["in"]
            total_out += qc["tokens"]["out"]

        session_row = {
            "session_id": f"produce_{cid}",
            "chain_id": cid,
            "papers": chain["papers"],
            "shared_entities": chain.get("shared_entities", []),
            "turn1_user": session.get("turn1_user", ""),
            "turn1_assistant": session.get("turn1_assistant", ""),
            "turn2_user": session.get("turn2_user", ""),
            "turn2_assistant": session.get("turn2_assistant", ""),
            "turn_dependency_qc": qc,
        }
        sessions.append(session_row)
        with sessions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(session_row, ensure_ascii=False) + "\n")

        if qc:
            qc["session_id"] = session_row["session_id"]
            with qc_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(qc, ensure_ascii=False) + "\n")

        dep = qc["verdict"] if qc else "skipped"
        print(f"[{idx:03d}/{len(valid_chains):03d}] {cid} -> dep={dep} entities={chain.get('shared_entities',[])[:3]}")

    # Summary
    dep_count = sum(1 for s in sessions if s.get("turn_dependency_qc", {}).get("verdict") == "dependent")
    ind_count = sum(1 for s in sessions if s.get("turn_dependency_qc", {}).get("verdict") == "independent")
    papers_covered = len(set(p for s in sessions for p in s.get("papers", [])))

    summary = {
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "input_pairs": len(judged_ids),
        "chains_built": len(chains),
        "chains_valid": len(valid_chains),
        "sessions_generated": len(sessions),
        "papers_covered": papers_covered,
        "turn_dependency": {
            "total": len(sessions),
            "dependent": dep_count,
            "independent": ind_count,
            "dependency_rate": round(dep_count / len(sessions), 4) if sessions else 0,
        },
        "tokens": {"in": total_in, "out": total_out},
        "files": {
            "chains": str(chains_path.relative_to(ROOT)),
            "materials": str(materials_path.relative_to(ROOT)),
            "sessions": str(sessions_path.relative_to(ROOT)),
            "qc": str(qc_path.relative_to(ROOT)),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(
        f"""# M4 Batch Production Summary

- Input pairs (strong + weak_but_related): **{len(judged_ids)}**
- Chains built: **{len(chains)}**
- Chains with all elements enriched: **{len(valid_chains)}**
- Sessions generated: **{len(sessions)}**
- Papers covered: **{papers_covered}**
- Turn-dependency: **{dep_count}** dependent / **{ind_count}** independent ({summary['turn_dependency']['dependency_rate']:.1%})
- Model: `{args.model}`
- Tokens: {total_in:,} in / {total_out:,} out

## Output
- `{chains_path.relative_to(ROOT)}`
- `{materials_path.relative_to(ROOT)}`
- `{sessions_path.relative_to(ROOT)}`
""",
        encoding="utf-8",
    )

    print(f"\n=== Done ===")
    print(f"Chains: {len(chains)} built, {len(valid_chains)} valid")
    print(f"Sessions: {len(sessions)}")
    print(f"Dependency: {dep_count} dependent, {ind_count} independent ({summary['turn_dependency']['dependency_rate']:.1%})")
    print(f"Papers covered: {papers_covered}")
    print(f"Tokens: {total_in:,} in, {total_out:,} out")
    print(f"Output: {out_dir}")

    log_run(
        script="experiments/produce_m4_multiturn_batch.py",
        model=f"company:{args.model}",
        purpose="Batch M4 multi-turn production from entity-bridge pairs",
        input_tokens=total_in,
        output_tokens=total_out,
        extra={
            "input_pairs": len(judged_ids),
            "chains_built": len(chains),
            "chains_valid": len(valid_chains),
            "sessions_generated": len(sessions),
            "dependency_rate": summary["turn_dependency"]["dependency_rate"],
            "output": str(out_dir.relative_to(ROOT)),
        },
    )


if __name__ == "__main__":
    main()
