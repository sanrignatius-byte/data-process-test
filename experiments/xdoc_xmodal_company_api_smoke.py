#!/usr/bin/env python3
"""Company-API smoke for PDF-first cross-doc + cross-modal 3-node chains.

Experimental lane only. Writes experiment artifacts under a fresh
``data/05_eval/xdoc_xmodal_company_api_smoke_*`` directory, while API call logs
use the fixed repository-standard ``api_logs_cannt_delete`` path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import call_llm, parse_json, set_company_credentials  # noqa: E402
from src.utils.token_logger import log_run  # noqa: E402
from experiments.xdoc_xmodal_prompt_dryrun import build_prompt, load_candidates  # noqa: E402


DEFAULT_CANDIDATES = ROOT / "data/05_eval/pdf_first_xdoc_xmodal_design_latest/xdoc_xmodal_candidates.json"
DEFAULT_MODEL = "gpt-5.4"


def make_out_dir(explicit: str = "") -> Path:
    if explicit:
        out = Path(explicit)
        if not out.is_absolute():
            out = ROOT / out
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = ROOT / f"data/05_eval/xdoc_xmodal_company_api_smoke_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    latest = ROOT / "data/05_eval/xdoc_xmodal_company_api_smoke_latest"
    try:
        if latest.exists() and latest.resolve() == out.resolve():
            return out
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out.resolve())
    except OSError:
        pass
    return out


def configure_standard_logger() -> Path:
    from local_api_logger.logger import APILogger  # noqa: WPS433
    import local_api_logger.tracker as tracker  # noqa: WPS433

    log_dir = ROOT / "api_logs_cannt_delete"
    log_dir.mkdir(parents=True, exist_ok=True)
    tracker._default_tracker = tracker.APITracker(APILogger(str(log_dir)))
    return log_dir


def min_richness(candidate: dict[str, Any]) -> int:
    meta = candidate.get("pdf_first_metadata", {})
    return min(
        int(meta.get("source_richness", 0) or 0),
        int(meta.get("target_richness", 0) or 0),
        int(meta.get("neighbor_richness", 0) or 0),
    )


def select_stratified(candidates: list[dict[str, Any]], per_base_type: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.get("base_pair_type", "unknown")].append(candidate)
    selected: list[dict[str, Any]] = []
    for base_type in sorted(groups):
        ranked = sorted(groups[base_type], key=min_richness, reverse=True)
        selected.extend(ranked[:per_base_type])
    return selected


def prompt_with_json_instruction(candidate: dict[str, Any]) -> str:
    return (
        build_prompt(candidate)
        + "\n\nReturn only valid JSON. Do not use markdown fences. Keep the query answerable only by using all three elements."
    )


def validate_parsed(obj: dict[str, Any]) -> dict[str, Any]:
    required = ["query", "answer", "reasoning_chain", "required_evidence_spans", "chain_roles", "qc_notes"]
    missing = [key for key in required if key not in obj]
    spans = obj.get("required_evidence_spans")
    return {
        "missing_required_fields": missing,
        "span_count": len(spans) if isinstance(spans, list) else None,
        "has_three_spans": isinstance(spans, list) and len(spans) == 3,
        "has_chain_roles": isinstance(obj.get("chain_roles"), dict),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Company API smoke for xdoc+xmodal chains")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--per-base-type", type=int, default=2)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--company-api-url", default=os.environ.get("COMPANY_API_URL", ""))
    parser.add_argument("--company-api-key", default=os.environ.get("COMPANY_API_KEY", ""))
    parser.add_argument("--max-tokens", type=int, default=1100)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = make_out_dir(args.output_dir)
    standard_log_dir = configure_standard_logger()

    summary: dict[str, Any] = {
        "status": "started",
        "model": args.model,
        "per_base_type": args.per_base_type,
        "output_dir": str(out_dir.relative_to(ROOT)),
        "company_api_url_set": bool(args.company_api_url),
        "company_api_key_set": bool(args.company_api_key),
    }

    if not args.company_api_url or not args.company_api_key:
        summary["status"] = "blocked_missing_company_credentials"
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[blocked] missing company API credentials; wrote {out_dir / 'summary.json'}")
        return

    set_company_credentials(args.company_api_url, args.company_api_key)
    candidates_path = Path(args.candidates)
    if not candidates_path.is_absolute():
        candidates_path = ROOT / candidates_path
    candidates = select_stratified(load_candidates(candidates_path), args.per_base_type)

    prompts_path = out_dir / "prompts.jsonl"
    responses_path = out_dir / "responses.jsonl"
    parsed_path = out_dir / "parsed.jsonl"
    failures_path = out_dir / "failures.jsonl"
    selected_path = out_dir / "selected_candidates.json"
    selected_path.write_text(json.dumps({"candidates": candidates}, ensure_ascii=False, indent=2), encoding="utf-8")

    total_in = total_out = parsed_ok = failures = 0
    validation_counts = defaultdict(int)

    with prompts_path.open("w", encoding="utf-8") as pf, responses_path.open("w", encoding="utf-8") as rf, parsed_path.open("w", encoding="utf-8") as jf, failures_path.open("w", encoding="utf-8") as ff:
        for idx, candidate in enumerate(candidates, 1):
            prompt = prompt_with_json_instruction(candidate)
            pf.write(json.dumps({"idx": idx, "candidate_id": candidate["candidate_id"], "chain_shape": candidate["chain_shape"], "path": candidate["path"], "prompt": prompt}, ensure_ascii=False) + "\n")
            try:
                text, in_tok, out_tok = call_llm(
                    None,
                    args.model,
                    prompt,
                    provider="company",
                    system_prompt="You generate grounded PDF-first cross-document and cross-modal academic retrieval queries.",
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    user_tag="xdoc_xmodal_smoke",
                )
                total_in += in_tok
                total_out += out_tok
                rf.write(json.dumps({"idx": idx, "candidate_id": candidate["candidate_id"], "input_tokens": in_tok, "output_tokens": out_tok, "raw_text": text or ""}, ensure_ascii=False) + "\n")
                parsed = parse_json(text)
                if not parsed:
                    failures += 1
                    ff.write(json.dumps({"idx": idx, "candidate_id": candidate["candidate_id"], "error": "parse_json_failed", "raw_text": text or ""}, ensure_ascii=False) + "\n")
                    continue
                validation = validate_parsed(parsed)
                for key, value in validation.items():
                    if value is True:
                        validation_counts[key] += 1
                parsed.update({
                    "idx": idx,
                    "candidate_id": candidate["candidate_id"],
                    "chain_shape": candidate["chain_shape"],
                    "path": candidate["path"],
                    "validation": validation,
                })
                parsed_ok += 1
                jf.write(json.dumps(parsed, ensure_ascii=False) + "\n")
            except Exception as exc:  # noqa: BLE001 - smoke diagnostic
                failures += 1
                ff.write(json.dumps({"idx": idx, "candidate_id": candidate["candidate_id"], "error": repr(exc)}, ensure_ascii=False) + "\n")

    summary.update({
        "status": "completed" if failures == 0 else "completed_with_failures",
        "candidates_attempted": len(candidates),
        "parsed_ok": parsed_ok,
        "failures": failures,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "validation_counts": dict(validation_counts),
        "files": {
            "selected_candidates": str(selected_path.relative_to(ROOT)),
            "prompts": str(prompts_path.relative_to(ROOT)),
            "responses": str(responses_path.relative_to(ROOT)),
            "parsed": str(parsed_path.relative_to(ROOT)),
            "failures": str(failures_path.relative_to(ROOT)),
            "local_api_logger": str(standard_log_dir.relative_to(ROOT)),
            "token_db": "logs/token_usage.db",
        },
    })
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(
        "# XDoc + XModal Company API Smoke\n\n"
        f"- status: **{summary['status']}**\n"
        f"- candidates attempted: **{len(candidates)}**\n"
        f"- parsed_ok: **{parsed_ok}**\n"
        f"- failures: **{failures}**\n"
        f"- tokens: **{total_in} in / {total_out} out**\n"
        f"- local_api_logger: `{summary['files']['local_api_logger']}`\n"
        f"- token_db: `{summary['files']['token_db']}`\n",
        encoding="utf-8",
    )
    log_run(
        script="experiments/xdoc_xmodal_company_api_smoke.py",
        model=f"company:{args.model}",
        input_tokens=total_in,
        output_tokens=total_out,
        purpose="Experimental PDF-first cross-doc + cross-modal chain generation smoke",
        extra={
            "pairs_processed": len(candidates),
            "queries_written": parsed_ok,
            "parse_failures": failures,
            "output": str(parsed_path.relative_to(ROOT)),
            "output_dir": str(out_dir.relative_to(ROOT)),
        },
    )
    print(f"[ok] status={summary['status']} parsed_ok={parsed_ok}/{len(candidates)} failures={failures}")
    print(f"[ok] wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()