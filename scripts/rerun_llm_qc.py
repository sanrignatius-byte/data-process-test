#!/usr/bin/env python3
"""Re-run LLM QC on existing generated entries (without re-generating queries).

Use case: the ablation logic was buggy in a previous run. This script:
  1. Reads an existing .jsonl (e.g. OLD_BUGGY)
  2. Strips old LLM QC fields from qc_issues / qc_metrics
  3. Re-runs run_llm_qc with the fixed logic
  4. Writes updated entries to a new output file

Usage:
  python scripts/rerun_llm_qc.py \\
    --input  data/03_queries/l3_enriched_v3_2026-04-07_OLD_BUGGY.jsonl \\
    --candidates data/02_enriched/hub_candidates_enriched_v3.json \\
    --output data/03_queries/l3_enriched_v3_2026-04-07_rerun.jsonl \\
    --pass-only \\
    --provider company --model gpt-5.4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── repo root on sys.path ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.qc.llm_judge import run_llm_qc
from src.utils.image_utils import encode_image
from src.api import set_company_credentials, get_company_credentials
from src.utils.token_logger import log_run

# LLM QC fields that must be stripped and re-computed
_LLM_ISSUES  = {"llm_fake_multihop", "llm_answer_hallucination"}
_LLM_METRICS = {"llm_ablation", "llm_grounding", "llm_fake_multihop_warn", "llm_hallucination_warn"}


def strip_llm_qc(entry: dict) -> dict:
    """Remove old LLM QC results from an entry (in-place, returns entry)."""
    entry["qc_issues"] = [i for i in entry.get("qc_issues", []) if i not in _LLM_ISSUES]
    for k in _LLM_METRICS:
        entry.get("qc_metrics", {}).pop(k, None)
    return entry


def build_pair_index(candidates_path: str) -> dict:
    """Return {pair_id: pair_dict} from the candidates JSON."""
    with open(candidates_path) as f:
        raw = json.load(f)
    pairs = raw if isinstance(raw, list) else raw.get("pairs", [])
    return {p["pair_id"]: p for p in pairs}


def main():
    ap = argparse.ArgumentParser(description="Re-run LLM QC on existing generated entries")
    ap.add_argument("--input",      required=True, help="Input .jsonl (OLD_BUGGY)")
    ap.add_argument("--candidates", required=True, help="Candidates JSON (for element_a/element_b)")
    ap.add_argument("--output",     required=True, help="Output .jsonl path")
    ap.add_argument("--pass-only",  action="store_true", help="Also write _pass.jsonl alongside output")
    ap.add_argument("--provider",   default="company", choices=["company", "openai", "anthropic"])
    ap.add_argument("--model",      default=None)
    ap.add_argument("--no-images",  action="store_true", help="Skip image encoding")
    ap.add_argument("--delay",      type=float, default=0.0, help="Seconds to sleep between API calls")
    ap.add_argument("--dry-run",    action="store_true", help="Skip LLM calls (rule QC only)")
    ap.add_argument("--resume",     action="store_true",
                    help="Append to existing output; skip entries whose pair_id+query_id already present")
    ap.add_argument(
        "--company-api-url", default=os.environ.get("COMPANY_API_URL", ""),
    )
    ap.add_argument(
        "--company-api-key", default=os.environ.get("COMPANY_API_KEY", ""),
    )
    args = ap.parse_args()

    # ── model default ────────────────────────────────────────────────────────
    if args.model is None:
        args.model = "gpt-5.4" if args.provider == "company" else (
            "gpt-4o" if args.provider == "openai" else "claude-sonnet-4-5-20250929"
        )

    # ── client init ──────────────────────────────────────────────────────────
    client = None
    if not args.dry_run:
        if args.provider == "company":
            set_company_credentials(args.company_api_url, args.company_api_key)
            url, key = get_company_credentials()
            if not url:
                sys.exit("ERROR: COMPANY_API_URL not set")
            if not key:
                sys.exit("ERROR: COMPANY_API_KEY not set")
            print(f"  Company API: {url}")
        elif args.provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # ── load data ────────────────────────────────────────────────────────────
    print(f"Loading candidates: {args.candidates}")
    pair_index = build_pair_index(args.candidates)
    print(f"  {len(pair_index)} pairs indexed")

    print(f"Loading input: {args.input}")
    with open(args.input) as f:
        entries = [json.loads(l) for l in f if l.strip()]
    print(f"  {len(entries)} entries")

    # ── output paths ─────────────────────────────────────────────────────────
    out_path  = Path(args.output)
    pass_path = out_path.with_name(out_path.stem + "_pass" + out_path.suffix) if args.pass_only else None

    total_in, total_out = 0, 0
    kept = 0
    missing_pair = 0
    skipped_resume = 0

    # ── resume: load already-processed keys ────────────────────────────────
    done_keys: set = set()
    if args.resume and out_path.exists():
        with open(out_path) as _rf:
            for _line in _rf:
                _line = _line.strip()
                if _line:
                    try:
                        _obj = json.loads(_line)
                        done_keys.add((_obj.get("pair_id", ""), _obj.get("query_id", "")))
                    except (json.JSONDecodeError, KeyError):
                        pass
        print(f"  --resume: {len(done_keys)} entries already in {out_path.name}, will skip")

    file_mode = "a" if args.resume else "w"

    print(f"\nRe-running LLM QC ({args.provider}/{args.model}) on {len(entries)} entries...")
    print(f"  Output: {out_path}  (mode={file_mode})")
    if pass_path:
        print(f"  Pass:   {pass_path}")
    print(flush=True)

    with open(out_path, file_mode) as fout, \
         (open(pass_path, file_mode) if pass_path else open(os.devnull, "w")) as fpass:

        for i, entry in enumerate(entries, 1):
            pair_id  = entry.get("pair_id", "?")
            query_id = entry.get("query_id", "")
            pair     = pair_index.get(pair_id)

            # skip already-done entries in resume mode
            if (pair_id, query_id) in done_keys:
                skipped_resume += 1
                continue

            if pair is None:
                print(f"  [{i}/{len(entries)}] {pair_id} — MISSING in candidates, copying as-is")
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                missing_pair += 1
                continue

            # Strip old LLM QC results
            entry = strip_llm_qc(entry)

            # Encode images
            llm_images = []
            if not args.no_images and not args.dry_run:
                llm_images = [
                    encode_image(pair["element_a"].get("image_path")),
                    encode_image(pair["element_b"].get("image_path")),
                ]

            # Re-run LLM QC
            try:
                llm_issues, llm_metrics, llm_in, llm_out = run_llm_qc(
                    obj=entry,
                    pair=pair,
                    client=client,
                    model=args.model,
                    provider=args.provider,
                    images=llm_images,
                    dry_run=args.dry_run,
                )
                total_in  += llm_in
                total_out += llm_out
                entry["qc_issues"].extend(llm_issues)
                entry["qc_metrics"].update(llm_metrics)
            except Exception as e:
                print(f"    WARNING: LLM QC failed for {pair_id}: {e}")
                entry["qc_metrics"]["llm_qc_error"] = str(e)[:200]

            entry["qc_pass"] = len(entry["qc_issues"]) == 0

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()
            if pass_path and entry["qc_pass"]:
                fpass.write(json.dumps(entry, ensure_ascii=False) + "\n")
                fpass.flush()
                kept += 1

            status = "PASS" if entry["qc_pass"] else f"FAIL({','.join(entry['qc_issues'])})"
            print(f"  [{i}/{len(entries)}] {pair_id} — {status}", flush=True)

            if args.delay > 0 and i < len(entries):
                time.sleep(args.delay)

    # ── summary ──────────────────────────────────────────────────────────────
    est_cost = total_in * 3 / 1e6 + total_out * 15 / 1e6
    print(f"\n{'='*60}")
    print(f"Done: {kept}/{len(entries)} pass")
    if skipped_resume:
        print(f"  Skipped (resume): {skipped_resume}")
    if missing_pair:
        print(f"  Missing pair (copied as-is): {missing_pair}")
    print(f"  Tokens: {total_in:,} in / {total_out:,} out  (est. ${est_cost:.3f})")
    print(f"  Output: {out_path}")
    if pass_path:
        print(f"  Pass:   {pass_path}")

    log_run(
        script="rerun_llm_qc",
        model=args.model,
        purpose=f"LLM QC rerun — {Path(args.input).name}",
        input_tokens=total_in,
        output_tokens=total_out,
        extra={
            "provider": args.provider,
            "input": args.input,
            "output": str(out_path),
            "total_entries": len(entries),
            "qc_pass": kept,
            "skipped_resume": skipped_resume,
        },
    )


if __name__ == "__main__":
    main()
