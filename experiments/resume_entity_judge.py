#!/usr/bin/env python3
"""Resume entity-bridge judge for remaining candidates, then optionally expand pool."""
from __future__ import annotations

import json, os, sys, time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import call_llm, parse_json, set_company_credentials
from src.utils.token_logger import log_run
from experiments.judge_entity_bridge_pack import (
    load_jsonl, build_prompt, validate, write_jsonl,
    summarize, render_markdown, make_out_dir, image_to_b64,
)

EXISTING_DIR = ROOT / "data/05_eval/entity_bridge_judge_20260521T113000Z"
PACK_PATH = ROOT / "data/05_eval/entity_bridge_candidates_latest/judge_pack.jsonl"


def main():
    set_company_credentials(os.environ.get("COMPANY_API_URL", ""),
                           os.environ.get("COMPANY_API_KEY", ""))

    # Load already judged
    judged_ids = set()
    existing_judgments = []
    jpath = EXISTING_DIR / "judgments.jsonl"
    if jpath.exists():
        with open(jpath) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                judged_ids.add(d["candidate_id"])
                existing_judgments.append(d)
    print(f"Already judged: {len(judged_ids)}")

    # Load all candidates
    all_rows = load_jsonl(PACK_PATH)
    print(f"Total entity-bridge candidates: {len(all_rows)}")

    remaining = [r for r in all_rows if r["candidate_id"] not in judged_ids]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All done. Writing final summary.")
        s = summarize(existing_judgments)
        (EXISTING_DIR / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2))
        print(f"Total: {s['total']}, Strong: {s['strong_chain']} ({s['strong_rate']:.1%})")
        print(f"Verdicts: {s['verdict_counts']}")
        return

    judgments_path = jpath
    prompts_path = EXISTING_DIR / "prompts.jsonl"
    responses_path = EXISTING_DIR / "responses.jsonl"
    failures_path = EXISTING_DIR / "failures.jsonl"

    total_in = 0
    total_out = 0
    judgments = list(existing_judgments)
    total = len(all_rows)

    for i, row in enumerate(remaining):
        idx = len(judged_ids) + i + 1
        prompt = build_prompt(row)

        images = [
            image_to_b64(row.get("element_a_image_path", "")),
            image_to_b64(row.get("element_b_image_path", "")),
        ]

        write_jsonl(prompts_path, {
            "candidate_id": row["candidate_id"],
            "target_stratum": row.get("target_stratum"),
            "prompt": prompt,
            "image_count": sum(1 for x in images if x is not None),
        })

        raw = None
        tin = tout = 0
        for attempt in range(3):
            try:
                raw, tin, tout = call_llm(
                    client=None, model="gpt-5.4", provider="company",
                    prompt=prompt, images=images,
                    system_prompt=(
                        "You are a conservative scientific graph-edge judge. "
                        "Return valid JSON only."
                    ),
                    max_tokens=800, temperature=0.2,
                    user_tag="entity_bridge_judge_resume",
                )
                break
            except Exception as e:
                print(f"  [attempt {attempt+1}/3] API error: {e}", flush=True)
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))

        if raw is None:
            out = {
                "candidate_id": row["candidate_id"],
                "judge_index": idx,
                "target_stratum": row.get("target_stratum"),
                "source_doc": row.get("source_doc"),
                "target_doc": row.get("target_doc"),
                "source_element_id": row.get("source_element_id"),
                "target_element_id": row.get("target_element_id"),
                "pair_type": row.get("pair_type"),
                "shared_entities": row.get("_meta", {}).get("shared_entities", []),
                "judgment": None,
                "validation": {"valid": False, "reason": "api_failed"},
                "tokens": {"in": 0, "out": 0},
            }
        else:
            total_in += tin
            total_out += tout
            parsed = parse_json(raw or "")
            validation = validate(parsed)
            out = {
                "candidate_id": row["candidate_id"],
                "judge_index": idx,
                "target_stratum": row.get("target_stratum"),
                "source_doc": row.get("source_doc"),
                "target_doc": row.get("target_doc"),
                "source_element_id": row.get("source_element_id"),
                "target_element_id": row.get("target_element_id"),
                "pair_type": row.get("pair_type"),
                "shared_entities": row.get("_meta", {}).get("shared_entities", []),
                "judgment": parsed if isinstance(parsed, dict) else None,
                "validation": validation,
                "tokens": {"in": tin, "out": tout},
            }
            write_jsonl(responses_path, {
                "candidate_id": row["candidate_id"],
                "raw": raw,
                "tokens": {"in": tin, "out": tout},
            })
            if not validation["valid"]:
                write_jsonl(failures_path, out | {"raw": raw})

        judgments.append(out)
        write_jsonl(judgments_path, out)

        verdict = out["judgment"].get("verdict") if out["judgment"] else "api_failed"
        print(f"[{idx:03d}/{total:03d}] {row.get('source_doc','')}->{row.get('target_doc','')} "
              f"{row['candidate_id'][:50]} -> {verdict}", flush=True)

    s = summarize(judgments)
    (EXISTING_DIR / "summary.json").write_text(
        json.dumps(s, ensure_ascii=False, indent=2))
    (EXISTING_DIR / "summary.md").write_text(render_markdown(s), encoding="utf-8")

    log_run(
        script="experiments/resume_entity_judge.py",
        model="company:gpt-5.4",
        purpose="Resume entity-bridge judge (scale from 30 to 72)",
        input_tokens=total_in, output_tokens=total_out,
        extra={"candidates_judged": len(remaining), "output": str(EXISTING_DIR.relative_to(ROOT))},
    )

    print(f"\n=== Resume Complete ===")
    print(f"Total: {s['total']}, Strong: {s['strong_chain']} ({s['strong_rate']:.1%})")
    print(f"Verdicts: {s['verdict_counts']}")


if __name__ == "__main__":
    main()
