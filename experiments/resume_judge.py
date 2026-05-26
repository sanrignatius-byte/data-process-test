#!/usr/bin/env python3
"""Resume interrupted chunk-bridge judge from existing judgments.jsonl."""
from __future__ import annotations

import json, os, sys, time, traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import call_llm, parse_json, set_company_credentials
from src.utils.token_logger import log_run
from experiments.judge_chunk_bridge_pack import (
    JudgePackBuilder, build_prompt, validate, write_jsonl,
    summarize, render_markdown, image_to_b64,
)

EXISTING_DIR = ROOT / "data/05_eval/chunk_bridge_judge_v2"
CHAINS_DIR = ROOT / "data/05_eval/chunk_bridge_chains_53_20260522T031612Z"


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

    print(f"Already judged: {len(judged_ids)} chains")

    # Load all chains
    builder = JudgePackBuilder(ROOT)
    all_rows = builder.build(limit=0, chains_path=str(CHAINS_DIR / "chains.jsonl"))
    print(f"Total chains: {len(all_rows)}")

    # Filter to remaining
    remaining = [r for r in all_rows if r["candidate_id"] not in judged_ids]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do — all chains judged.")
        return

    # Append-mode paths
    prompts_path = EXISTING_DIR / "prompts.jsonl"
    responses_path = EXISTING_DIR / "responses.jsonl"
    judgments_path = EXISTING_DIR / "judgments.jsonl"  # append
    failures_path = EXISTING_DIR / "failures.jsonl"

    total_in = 0
    total_out = 0
    judgments = list(existing_judgments)  # start from existing

    start_idx = len(judged_ids)
    total = len(all_rows)

    for i, row in enumerate(remaining):
        idx = start_idx + i + 1
        prompt = build_prompt(row)

        images = []
        for img_path in row.get("source_element_images", [])[:1]:
            b64 = image_to_b64(img_path)
            if b64:
                images.append(b64)
        for img_path in row.get("target_element_images", [])[:1]:
            b64 = image_to_b64(img_path)
            if b64:
                images.append(b64)

        write_jsonl(prompts_path, {
            "candidate_id": row["candidate_id"],
            "prompt": prompt,
            "image_count": len(images),
        })

        # Retry loop with error handling
        raw = None
        tin = tout = 0
        for attempt in range(3):
            try:
                raw, tin, tout = call_llm(
                    client=None, model="gpt-5.4", provider="company",
                    prompt=prompt, images=images,
                    system_prompt=(
                        "You are a conservative scientific evidence-chain judge. "
                        "Return valid JSON only."
                    ),
                    max_tokens=800, temperature=0.2,
                    user_tag="chunk_bridge_judge_resume",
                )
                break
            except Exception as e:
                print(f"  [attempt {attempt+1}/3] API error: {e}", flush=True)
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    raw = None
                    tin = tout = 0

        if raw is None:
            print(f"[{idx:03d}/{total:03d}] {row['candidate_id'][:50]} -> API_FAILED", flush=True)
            out = {
                "candidate_id": row["candidate_id"],
                "judge_index": idx,
                "source_doc": row.get("source_doc"),
                "target_doc": row.get("target_doc"),
                "source_element_ids": row.get("source_element_ids", []),
                "target_element_ids": row.get("target_element_ids", []),
                "pair_type": row.get("pair_type"),
                "total_score": row.get("total_score", 0),
                "similarity": row.get("similarity", 0),
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
                "source_doc": row.get("source_doc"),
                "target_doc": row.get("target_doc"),
                "source_element_ids": row.get("source_element_ids", []),
                "target_element_ids": row.get("target_element_ids", []),
                "pair_type": row.get("pair_type"),
                "total_score": row.get("total_score", 0),
                "similarity": row.get("similarity", 0),
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

    # Write final summary
    summary = {
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "gpt-5.4",
        "output_dir": str(EXISTING_DIR.relative_to(ROOT)),
        **summarize(judgments),
    }
    (EXISTING_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (EXISTING_DIR / "summary.md").write_text(render_markdown(summary), encoding="utf-8")

    log_run(
        script="experiments/resume_judge.py",
        model="company:gpt-5.4",
        purpose="Resume chunk-bridge judge (caught up remaining chains)",
        input_tokens=total_in, output_tokens=total_out,
        extra={"chains_judged": len(remaining), "output": str(EXISTING_DIR.relative_to(ROOT))},
    )

    print(f"\n=== Resume Complete ===")
    s = summarize(judgments)
    print(f"Total: {s['total']}, Strong: {s['strong_chain']} ({s['strong_rate']:.1%}), "
          f"Usable: {s['usable_rate']:.1%}")
    print(f"Verdicts: {s['verdict_counts']}")


if __name__ == "__main__":
    main()
