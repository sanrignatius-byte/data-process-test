#!/usr/bin/env python3
"""Judge xdoc element-resolver candidates with the company VLM/LLM path.

This is an experimental-lane judge for
`data/05_eval/xdoc_element_resolver_v1_latest/judge_pack_120.jsonl`.
It answers one question: which resolver strata produce real M4 chains?

All API calls must go through `src.api.call_llm(provider="company")`, which
uses `local_api_logger.wrap_requests_call`; this script also writes the token
total to `logs/token_usage.db` via `log_run()`.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import call_llm, parse_json, set_company_credentials  # noqa: E402
from src.utils.token_logger import log_run  # noqa: E402


DEFAULT_PACK = ROOT / "data/05_eval/xdoc_element_resolver_v1_latest/judge_pack_120.jsonl"
DEFAULT_MODEL = "gpt-5.4"
VERDICTS = {
    "strong_chain",
    "weak_but_related",
    "topic_only",
    "wrong_target",
    "wrong_source",
    "insufficient_context",
}


def configure_standard_logger() -> Path:
    from local_api_logger.logger import APILogger  # noqa: WPS433
    import local_api_logger.tracker as tracker  # noqa: WPS433

    log_dir = ROOT / "api_logs_cannt_delete"
    log_dir.mkdir(parents=True, exist_ok=True)
    tracker._default_tracker = tracker.APITracker(APILogger(str(log_dir)))
    return log_dir


def make_out_dir(explicit: str = "") -> Path:
    if explicit:
        out = Path(explicit)
        if not out.is_absolute():
            out = ROOT / out
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = ROOT / f"data/05_eval/xdoc_resolver_judge_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    latest = ROOT / "data/05_eval/xdoc_resolver_judge_latest"
    try:
        if latest.exists() and latest.resolve() == out.resolve():
            return out
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out.resolve())
    except OSError:
        pass
    return out


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def clip(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def image_to_b64(path: str) -> tuple[str, str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists() or not p.is_file():
        return None
    mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
    return base64.b64encode(p.read_bytes()).decode("ascii"), mime


def build_prompt(row: dict[str, Any]) -> str:
    detail = row.get("target_resolution_detail") or {}
    return f"""You are judging cross-document scientific-document graph candidates.

Two endpoint images may be attached in this order:
1. source element image from the citing/source paper
2. target element image from the cited/target paper

Your task is to decide whether this candidate forms a valid M4 chain:
source element -> citation bridge text -> target element.

Use a conservative standard:
- strong_chain: the bridge specifically discusses or strongly implies the target element, and the source element is a plausible local anchor for why this source paper is making the comparison.
- weak_but_related: source/bridge/target are scientifically related, but the target element is not specifically enough identified.
- topic_only: same broad topic but no real chain.
- wrong_target: the bridge likely points to a different target element.
- wrong_source: the target may be plausible, but the source element is not a plausible local anchor.
- insufficient_context: not enough evidence to decide.

Important evaluation rules:
- For stratum A_hard_title_window, test hard explicit precision. Do not assume it is correct just because the method name says hard.
- For B/C/D strata, treat them as exploratory signals; judge the actual evidence.
- Ignore generic visual similarity. A table/figure shape match alone is not enough.
- If the bridge says "Figure 6" but the caption/content of target Figure 6 is about a different dataset, metric, or claim, mark wrong_target.
- Return only valid JSON.

Required JSON fields:
{{
  "verdict": "strong_chain | weak_but_related | topic_only | wrong_target | wrong_source | insufficient_context",
  "confidence": 0.0,
  "target_supported": true,
  "source_anchor_supported": true,
  "bridge_specificity": "explicit_number | explicit_named_result | semantic_match | broad_topic | none",
  "main_failure": "none | target_mismatch | source_mismatch | bridge_too_generic | visual_only | missing_context | other",
  "rationale": "2-4 concise sentences",
  "evidence": {{
    "bridge_cue": "short cue from citation bridge",
    "target_cue": "short cue from target caption/image/text",
    "source_cue": "short cue from source caption/image/text"
  }}
}}

Candidate:
- candidate_id: {row.get("candidate_id")}
- stratum: {row.get("target_stratum")}
- anchor_reason: {row.get("target_anchor_reason")}
- pair_type: {row.get("pair_type")}
- source_doc: {row.get("source_doc")}
- target_doc: {row.get("target_doc")}
- source_element_id: {row.get("source_element_id")}
- source_element_type: {row.get("source_element_type")}
- target_element_id: {row.get("target_element_id")}
- target_element_type: {row.get("target_element_type")}
- citation_probability: {row.get("citation_probability")}
- citation_fanout: {row.get("citation_fanout")}
- section_title: {row.get("section_title")}
- source_resolution_method: {row.get("source_resolution_method")}
- target_resolution_method: {row.get("target_resolution_method")}
- target_resolution_score: {row.get("target_resolution_score")}
- target_ref_text: {detail.get("ref_text")}
- target_ref_window_before: {clip(detail.get("ref_window_before", ""), 300)}
- target_ref_window_after: {clip(detail.get("ref_window_after", ""), 500)}

Source element caption/content:
{clip(row.get("source_caption_or_content", ""), 900)}

Citation bridge text:
{clip(row.get("citation_bridge_text", ""), 1300)}

Target element caption/content:
{clip(row.get("target_caption_or_content", ""), 900)}

Question for judge:
{row.get("question_for_judge", "")}
"""


def validate(obj: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {"valid": False, "reason": "not_object"}
    missing = [
        key for key in (
            "verdict",
            "confidence",
            "target_supported",
            "source_anchor_supported",
            "bridge_specificity",
            "main_failure",
            "rationale",
            "evidence",
        ) if key not in obj
    ]
    if missing:
        return {"valid": False, "reason": "missing:" + ",".join(missing)}
    if obj.get("verdict") not in VERDICTS:
        return {"valid": False, "reason": "bad_verdict"}
    try:
        conf = float(obj.get("confidence"))
    except (TypeError, ValueError):
        return {"valid": False, "reason": "bad_confidence"}
    if not 0 <= conf <= 1:
        return {"valid": False, "reason": "confidence_out_of_range"}
    return {"valid": True, "reason": "ok"}


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    verdict_counts: Counter[str] = Counter()
    validation_counts: Counter[str] = Counter()
    stratum_counts: dict[str, Counter[str]] = defaultdict(Counter)
    anchor_counts: dict[str, Counter[str]] = defaultdict(Counter)
    method_counts: dict[str, Counter[str]] = defaultdict(Counter)
    pair_type_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in judgments:
        verdict = row.get("judgment", {}).get("verdict", "parse_failed")
        stratum = row.get("target_stratum", "?")
        anchor = row.get("target_anchor_reason", "?")
        method = row.get("target_resolution_method", "?")
        pair_type = row.get("pair_type", "?")
        verdict_counts[verdict] += 1
        validation_counts[row.get("validation", {}).get("reason", "?")] += 1
        stratum_counts[stratum][verdict] += 1
        anchor_counts[anchor][verdict] += 1
        method_counts[method][verdict] += 1
        pair_type_counts[pair_type][verdict] += 1

    strong_total = verdict_counts["strong_chain"]
    total = len(judgments)
    return {
        "total": total,
        "strong_chain": strong_total,
        "strong_rate": round(strong_total / total, 4) if total else 0.0,
        "verdict_counts": dict(verdict_counts),
        "validation_counts": dict(validation_counts),
        "by_stratum": {k: dict(v) for k, v in sorted(stratum_counts.items())},
        "by_anchor_reason": {k: dict(v) for k, v in sorted(anchor_counts.items())},
        "by_target_method": {k: dict(v) for k, v in sorted(method_counts.items())},
        "by_pair_type": {k: dict(v) for k, v in sorted(pair_type_counts.items())},
        "precision_claim_scope": {
            "hard_explicit_precision": (
                "Only A_hard_title_window can support a hard explicit precision claim."
            ),
            "exploratory_strata": [
                "B_edge_title_match",
                "C_soft_fanout_or_single_ref",
                "D_unanchored_explicit",
                "E_overlap_high",
                "F_overlap_low",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default=str(DEFAULT_PACK))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--model", default=os.environ.get("COMPANY_API_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--company-api-url", default=os.environ.get("COMPANY_API_URL", ""))
    parser.add_argument("--company-api-key", default=os.environ.get("COMPANY_API_KEY", ""))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()

    pack_path = Path(args.pack)
    if not pack_path.is_absolute():
        pack_path = ROOT / pack_path
    rows = load_jsonl(pack_path, limit=args.limit)
    out_dir = make_out_dir(args.output_dir)
    log_dir = configure_standard_logger()
    set_company_credentials(args.company_api_url, args.company_api_key)

    prompts_path = out_dir / "prompts.jsonl"
    responses_path = out_dir / "responses.jsonl"
    judgments_path = out_dir / "judgments.jsonl"
    failures_path = out_dir / "failures.jsonl"

    total_in = 0
    total_out = 0
    judgments: list[dict[str, Any]] = []

    print(f"Loaded {len(rows)} judge items")
    print(f"Output: {out_dir}")
    print(f"Model: {args.model}")

    for idx, row in enumerate(rows, 1):
        prompt = build_prompt(row)
        images: list[tuple[str, str] | None] = []
        if not args.no_images:
            images = [
                image_to_b64(row.get("element_a_image_path", "")),
                image_to_b64(row.get("element_b_image_path", "")),
            ]
        write_jsonl(prompts_path, {
            "candidate_id": row.get("candidate_id"),
            "target_stratum": row.get("target_stratum"),
            "prompt": prompt,
            "image_count": sum(1 for x in images if x is not None),
        })

        raw, tin, tout = call_llm(
            client=None,
            model=args.model,
            provider="company",
            prompt=prompt,
            images=images,
            system_prompt=(
                "You are a conservative scientific graph-edge judge. "
                "Return valid JSON only."
            ),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            user_tag="xdoc_resolver_judge",
        )
        total_in += tin
        total_out += tout
        parsed = parse_json(raw or "")
        validation = validate(parsed)

        out = {
            "candidate_id": row.get("candidate_id"),
            "judge_index": row.get("judge_index"),
            "target_stratum": row.get("target_stratum"),
            "target_anchor_reason": row.get("target_anchor_reason"),
            "source_doc": row.get("source_doc"),
            "target_doc": row.get("target_doc"),
            "source_element_id": row.get("source_element_id"),
            "target_element_id": row.get("target_element_id"),
            "pair_type": row.get("pair_type"),
            "source_resolution_method": row.get("source_resolution_method"),
            "target_resolution_method": row.get("target_resolution_method"),
            "target_resolution_score": row.get("target_resolution_score"),
            "citation_probability": row.get("citation_probability"),
            "judgment": parsed if isinstance(parsed, dict) else None,
            "validation": validation,
            "tokens": {"in": tin, "out": tout},
        }
        judgments.append(out)
        write_jsonl(judgments_path, out)
        write_jsonl(responses_path, {
            "candidate_id": row.get("candidate_id"),
            "raw": raw,
            "tokens": {"in": tin, "out": tout},
        })
        if not validation["valid"]:
            write_jsonl(failures_path, out | {"raw": raw})

        verdict = out["judgment"].get("verdict") if out["judgment"] else "parse_failed"
        print(
            f"[{idx:03d}/{len(rows):03d}] {row.get('target_stratum')} "
            f"{row.get('candidate_id')} -> {verdict}"
        )

    summary = {
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "pack": str(pack_path.relative_to(ROOT)),
        "output_dir": str(out_dir.relative_to(ROOT)),
        "company_api_url_set": bool(args.company_api_url),
        "company_api_key_set": bool(args.company_api_key),
        "files": {
            "local_api_logger": str(log_dir.relative_to(ROOT)),
            "token_db": "logs/token_usage.db",
            "prompts": str(prompts_path.relative_to(ROOT)),
            "responses": str(responses_path.relative_to(ROOT)),
            "judgments": str(judgments_path.relative_to(ROOT)),
            "failures": str(failures_path.relative_to(ROOT)),
        },
        "tokens": {"in": total_in, "out": total_out},
        **summarize(judgments),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")

    log_run(
        script="experiments/judge_xdoc_resolver_pack.py",
        model=f"company:{args.model}",
        purpose="Judge xdoc resolver v1 stratified candidate pack",
        input_tokens=total_in,
        output_tokens=total_out,
        extra={
            "pairs_processed": len(rows),
            "qc_pass": summary["strong_chain"],
            "qc_fail": len(rows) - summary["strong_chain"],
            "parse_failures": summary["validation_counts"].get("not_object", 0),
            "output": str(out_dir.relative_to(ROOT)),
            "verdict_counts": summary["verdict_counts"],
            "by_stratum": summary["by_stratum"],
        },
    )
    print(f"Done. strong_chain={summary['strong_chain']}/{summary['total']}")
    print(f"Latest: {ROOT / 'data/05_eval/xdoc_resolver_judge_latest'}")


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# XDoc Resolver Judge Summary",
        "",
        f"- total: **{summary['total']}**",
        f"- strong_chain: **{summary['strong_chain']}** ({summary['strong_rate']:.1%})",
        f"- model: `{summary['model']}`",
        f"- output: `{summary['output_dir']}`",
        "",
        "## Verdict Counts",
        "",
    ]
    for k, v in sorted(summary["verdict_counts"].items()):
        lines.append(f"- `{k}`: {v}")
    lines.extend(["", "## By Stratum", ""])
    for stratum, counts in summary["by_stratum"].items():
        parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines.append(f"- `{stratum}`: {parts}")
    lines.extend([
        "",
        "## Claim Scope",
        "",
        "- Hard explicit precision can only be claimed from `A_hard_title_window`.",
        "- `B/C/D` are exploratory explicit-resolution signals.",
        "- `E/F` are overlap baselines.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
