#!/usr/bin/env python3
"""Experimental company-API smoke for new cross-document prompt generation.

Safety rules:
- Experimental lane only: no production ``src/`` changes and no production data
  artifacts.
- All outputs go under a fresh ``data/05_eval/trinity_company_api_smoke_*``
    directory, or ``data/05_eval/trinity_company_api_smoke_latest``.
- API call logs use the fixed repository-standard ``api_logs_cannt_delete`` path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import call_llm, parse_json, set_company_credentials  # noqa: E402
from src.utils.token_logger import log_run  # noqa: E402

from experiments.crossdoc_prompt_dryrun import (  # noqa: E402
    DEFAULT_CANDIDATES,
    build_generic_crossdoc_prompt,
    load_pairs,
)


DEFAULT_MODEL = "gpt-5.4"


def make_out_dir(explicit: str = "") -> Path:
    if explicit:
        out = Path(explicit)
        if not out.is_absolute():
            out = ROOT / out
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = ROOT / f"data/05_eval/trinity_company_api_smoke_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    latest = ROOT / "data/05_eval/trinity_company_api_smoke_latest"
    try:
        if out.resolve() == latest.resolve() if latest.exists() else False:
            return out
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out.resolve())
    except OSError:
        # Symlink is convenience only; never fail the experiment because of it.
        pass
    return out


def configure_standard_logger() -> Path:
    """Use the fixed repository-standard local API logger directory."""
    from local_api_logger.logger import APILogger  # noqa: WPS433
    import local_api_logger.tracker as tracker  # noqa: WPS433

    log_dir = ROOT / "api_logs_cannt_delete"
    log_dir.mkdir(parents=True, exist_ok=True)
    tracker._default_tracker = tracker.APITracker(APILogger(str(log_dir)))
    return log_dir


def compact_prompt_for_company(prompt: str) -> str:
    return (
        prompt
        + "\n\nImportant: Return only valid JSON. Do not wrap it in markdown fences."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental company API smoke for cross-doc prompts")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--company-api-url", default=os.environ.get("COMPANY_API_URL", ""))
    parser.add_argument("--company-api-key", default=os.environ.get("COMPANY_API_KEY", ""))
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = make_out_dir(args.output_dir)
    standard_log_dir = configure_standard_logger()

    summary: dict[str, Any] = {
        "status": "started",
        "output_dir": str(out_dir.relative_to(ROOT)),
        "model": args.model,
        "limit": args.limit,
        "company_api_url_set": bool(args.company_api_url),
        "company_api_key_set": bool(args.company_api_key),
        "notes": [],
    }

    if not args.company_api_url or not args.company_api_key:
        summary["status"] = "blocked_missing_company_credentials"
        summary["notes"].append(
            "COMPANY_API_URL/COMPANY_API_KEY are not set in this shell; no API call was attempted."
        )
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "summary.md").write_text(
            "# Cross-doc Company API Smoke\n\n"
            "Status: **blocked_missing_company_credentials**\n\n"
            "`COMPANY_API_URL` / `COMPANY_API_KEY` are not set in this shell. "
            "No API call was attempted; isolated output directory was created.\n",
            encoding="utf-8",
        )
        print(f"[blocked] missing COMPANY_API_URL/COMPANY_API_KEY; wrote {out_dir / 'summary.md'}")
        return

    set_company_credentials(args.company_api_url, args.company_api_key)

    candidates = Path(args.candidates)
    if not candidates.is_absolute():
        candidates = ROOT / candidates
    pairs = load_pairs(candidates)[: args.limit]

    prompts_path = out_dir / "prompts.jsonl"
    responses_path = out_dir / "responses.jsonl"
    parsed_path = out_dir / "parsed.jsonl"
    failures_path = out_dir / "failures.jsonl"

    total_in = 0
    total_out = 0
    parsed_ok = 0
    failures = 0

    with prompts_path.open("w", encoding="utf-8") as pf, responses_path.open("w", encoding="utf-8") as rf, parsed_path.open("w", encoding="utf-8") as jf, failures_path.open("w", encoding="utf-8") as ff:
        for idx, pair in enumerate(pairs, 1):
            prompt = compact_prompt_for_company(build_generic_crossdoc_prompt(pair))
            prompt_row = {
                "idx": idx,
                "pair_id": pair.get("pair_id"),
                "pair_type": pair.get("pair_type"),
                "path": pair.get("path"),
                "prompt": prompt,
            }
            pf.write(json.dumps(prompt_row, ensure_ascii=False) + "\n")
            try:
                text, in_tok, out_tok = call_llm(
                    None,
                    args.model,
                    prompt,
                    provider="company",
                    system_prompt="You generate grounded cross-document academic retrieval queries.",
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    user_tag="trinity_xdoc_smoke",
                )
                total_in += in_tok
                total_out += out_tok
                response_row = {
                    "idx": idx,
                    "pair_id": pair.get("pair_id"),
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "raw_text": text or "",
                }
                rf.write(json.dumps(response_row, ensure_ascii=False) + "\n")
                parsed = parse_json(text)
                if parsed:
                    parsed_ok += 1
                    parsed["idx"] = idx
                    parsed["pair_id"] = pair.get("pair_id")
                    parsed["pair_type"] = pair.get("pair_type")
                    parsed["path"] = pair.get("path")
                    jf.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                else:
                    failures += 1
                    ff.write(json.dumps({"idx": idx, "pair_id": pair.get("pair_id"), "error": "parse_json_failed", "raw_text": text or ""}, ensure_ascii=False) + "\n")
            except Exception as exc:  # noqa: BLE001 - smoke diagnostic
                failures += 1
                ff.write(json.dumps({"idx": idx, "pair_id": pair.get("pair_id"), "error": repr(exc)}, ensure_ascii=False) + "\n")

    summary.update(
        {
            "status": "completed" if failures == 0 else "completed_with_failures",
            "pairs_attempted": len(pairs),
            "parsed_ok": parsed_ok,
            "failures": failures,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "files": {
                "prompts": str(prompts_path.relative_to(ROOT)),
                "responses": str(responses_path.relative_to(ROOT)),
                "parsed": str(parsed_path.relative_to(ROOT)),
                "failures": str(failures_path.relative_to(ROOT)),
                "local_api_logger": str(standard_log_dir.relative_to(ROOT)),
                "token_db": "logs/token_usage.db",
            },
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(
        "# Cross-doc Company API Smoke\n\n"
        f"- status: **{summary['status']}**\n"
        f"- pairs attempted: **{len(pairs)}**\n"
        f"- parsed_ok: **{parsed_ok}**\n"
        f"- failures: **{failures}**\n"
        f"- tokens: **{total_in} in / {total_out} out**\n"
        f"- local_api_logger: `{summary['files']['local_api_logger']}`\n"
        f"- token_db: `{summary['files']['token_db']}`\n",
        encoding="utf-8",
    )

    log_run(
        script="experiments/crossdoc_company_api_smoke.py",
        model=f"company:{args.model}",
        input_tokens=total_in,
        output_tokens=total_out,
        purpose="Experimental cross-doc same-modality prompt API smoke",
        extra={
            "pairs_processed": len(pairs),
            "queries_written": parsed_ok,
            "parse_failures": failures,
            "output": str(parsed_path.relative_to(ROOT)),
            "output_dir": str(out_dir.relative_to(ROOT)),
        },
    )
    print(f"[ok] status={summary['status']} parsed_ok={parsed_ok}/{len(pairs)} failures={failures}")
    print(f"[ok] wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()