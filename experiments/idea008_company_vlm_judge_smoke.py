#!/usr/bin/env python3
"""Company-API VLM judge smoke for idea:008 cross-document edges.

Experimental lane only. Uses the repository-standard company API path:
``src.api.call_llm(provider="company")`` -> ``local_api_logger.wrap_requests_call``.
Outputs are isolated under ``data/05_eval/idea008_company_vlm_judge_smoke_*``.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import call_llm, parse_json, set_company_credentials  # noqa: E402
from src.utils.token_logger import log_run  # noqa: E402


DEFAULT_CANDIDATES = ROOT / "data/05_eval/idea008_phase0_judge_pack_latest/phase0_candidates.jsonl"
DEFAULT_IDS = [
    "idea008_phase0_0001",  # clean strong table/table positive control
    "idea008_phase0_0009",  # layout false positive, empty-square artifact
    "idea008_phase0_0013",  # caption-zero causal/fairness graph positive probe
]
DEFAULT_MODEL = "gpt-5.4"


def make_out_dir(explicit: str = "") -> Path:
    if explicit:
        out = Path(explicit)
        if not out.is_absolute():
            out = ROOT / out
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = ROOT / f"data/05_eval/idea008_company_vlm_judge_smoke_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    latest = ROOT / "data/05_eval/idea008_company_vlm_judge_smoke_latest"
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


def load_candidates(path: Path, ids: list[str]) -> list[dict[str, Any]]:
    wanted = set(ids)
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("candidate_id")
            if cid in wanted:
                rows[cid] = row
    missing = [cid for cid in ids if cid not in rows]
    if missing:
        raise SystemExit(f"missing candidate ids: {missing}")
    return [rows[cid] for cid in ids]


def clip_text(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def image_to_b64(path: str) -> tuple[str, str]:
    p = Path(path)
    mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
    return base64.b64encode(p.read_bytes()).decode("ascii"), mime


def element_block(name: str, element: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"{name}:",
            f"- node_id: {element.get('node_id')}",
            f"- doc_id: {element.get('doc_id')}",
            f"- type: {element.get('node_type')}",
            f"- label: {element.get('label')}",
            f"- caption: {clip_text(element.get('caption', ''), 850)}",
            f"- enriched_preview: {clip_text(element.get('enriched_preview', ''), 650)}",
            f"- local_context: {clip_text(element.get('context', ''), 1200)}",
        ]
    )


def build_prompt(candidate: dict[str, Any]) -> str:
    scores = candidate.get("scores", {})
    return f"""You are a strict multimodal judge for scientific-document graph edges.

Two images are attached in order: Element A image, then Element B image.

Decision task:
Decide whether this candidate should be promoted from a CLIP-recall edge to a STRONG
cross-document semantic edge. Promote only when the two elements share a specific
scientific role such as the same dataset+metric, same method diagram, same causal
structure, same experimental comparison, or same reusable conceptual object. Reject
generic visual layout matches such as empty boxes, chart style, table shape, or
histogram shape when the underlying scientific content differs.

Return only valid JSON with exactly these fields:
- verdict: one of ["strong_edge", "weak_related", "visual_layout_only", "unrelated", "insufficient"]
- confidence: number from 0 to 1
- shared_semantics: short phrase naming the shared concept, or "" if none
- evidence_a: one concrete visual/textual cue from Element A
- evidence_b: one concrete visual/textual cue from Element B
- rationale: 2-4 sentences
- failure_mode: one of ["none", "caption_degraded", "layout_false_positive", "generic_caption", "missing_image", "insufficient_context", "other"]

Candidate metadata:
- candidate_id: {candidate.get("candidate_id")}
- existing_support_tier: {candidate.get("support_tier")}
- caption_bucket: {candidate.get("caption_bucket")}
- heuristic_label_hint: {candidate.get("heuristic_label_hint")}
- scores: combined={scores.get("combined_score")} visual={scores.get("visual_score")} caption={scores.get("caption_sim")} context={scores.get("context_sim")} enriched={scores.get("enriched_sim")}

{element_block("Element A", candidate.get("source", {}))}

{element_block("Element B", candidate.get("target", {}))}
"""


def validate_judgment(obj: dict[str, Any] | None) -> dict[str, Any]:
    allowed_verdicts = {"strong_edge", "weak_related", "visual_layout_only", "unrelated", "insufficient"}
    allowed_failures = {
        "none",
        "caption_degraded",
        "layout_false_positive",
        "generic_caption",
        "missing_image",
        "insufficient_context",
        "other",
    }
    if not isinstance(obj, dict):
        return {"valid": False, "reason": "not_object"}
    required = [
        "verdict",
        "confidence",
        "shared_semantics",
        "evidence_a",
        "evidence_b",
        "rationale",
        "failure_mode",
    ]
    missing = [key for key in required if key not in obj]
    if missing:
        return {"valid": False, "reason": f"missing:{','.join(missing)}"}
    if obj.get("verdict") not in allowed_verdicts:
        return {"valid": False, "reason": "bad_verdict"}
    if obj.get("failure_mode") not in allowed_failures:
        return {"valid": False, "reason": "bad_failure_mode"}
    try:
        conf = float(obj.get("confidence"))
    except (TypeError, ValueError):
        return {"valid": False, "reason": "bad_confidence"}
    if conf < 0 or conf > 1:
        return {"valid": False, "reason": "confidence_out_of_range"}
    return {"valid": True, "reason": "ok"}


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Company VLM judge smoke for idea:008")
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--model", default=os.environ.get("COMPANY_API_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--company-api-url", default=os.environ.get("COMPANY_API_URL", ""))
    parser.add_argument("--company-api-key", default=os.environ.get("COMPANY_API_KEY", ""))
    parser.add_argument("--max-tokens", type=int, default=850)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    out_dir = make_out_dir(args.output_dir)
    standard_log_dir = configure_standard_logger()
    summary: dict[str, Any] = {
        "status": "started",
        "model": args.model,
        "candidate_ids": args.ids,
        "output_dir": str(out_dir.relative_to(ROOT)),
        "company_api_url_set": bool(args.company_api_url),
        "company_api_key_set": bool(args.company_api_key),
        "files": {
            "local_api_logger": str(standard_log_dir.relative_to(ROOT)),
            "token_db": "logs/token_usage.db",
        },
    }

    if not args.company_api_url or not args.company_api_key:
        summary["status"] = "blocked_missing_company_credentials"
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[blocked] missing company API credentials; wrote {out_dir / 'summary.json'}")
        return

    candidates_path = Path(args.candidates)
    if not candidates_path.is_absolute():
        candidates_path = ROOT / candidates_path
    candidates = load_candidates(candidates_path, args.ids)
    set_company_credentials(args.company_api_url, args.company_api_key)

    selected_path = out_dir / "selected_candidates.json"
    prompts_path = out_dir / "prompts.jsonl"
    responses_path = out_dir / "responses.jsonl"
    judgments_path = out_dir / "judgments.jsonl"
    failures_path = out_dir / "failures.jsonl"

    selected_path.write_text(
        json.dumps({"candidates": candidates}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_in = total_out = parsed_ok = failures = 0
    verdict_counts: Counter[str] = Counter()
    validation_counts: Counter[str] = Counter()

    for idx, candidate in enumerate(candidates, 1):
        prompt = build_prompt(candidate)
        prompt_row = {
            "idx": idx,
            "candidate_id": candidate["candidate_id"],
            "support_tier": candidate.get("support_tier"),
            "caption_bucket": candidate.get("caption_bucket"),
            "prompt": prompt,
        }
        write_jsonl(prompts_path, prompt_row)

        try:
            images = [
                image_to_b64(candidate["source"]["image_path"]),
                image_to_b64(candidate["target"]["image_path"]),
            ]
            text, in_tok, out_tok = call_llm(
                None,
                args.model,
                prompt,
                images=images,
                provider="company",
                system_prompt=(
                    "You are a conservative VLM judge for scientific graph edge quality. "
                    "Return only valid JSON."
                ),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                user_tag="idea008_vlm_judge_smoke",
            )
            total_in += in_tok
            total_out += out_tok
            write_jsonl(
                responses_path,
                {
                    "idx": idx,
                    "candidate_id": candidate["candidate_id"],
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "raw_text": text or "",
                },
            )
            parsed = parse_json(text)
            validation = validate_judgment(parsed)
            validation_counts[validation["reason"]] += 1
            if not validation["valid"]:
                failures += 1
                write_jsonl(
                    failures_path,
                    {
                        "idx": idx,
                        "candidate_id": candidate["candidate_id"],
                        "error": "invalid_or_unparsed_json",
                        "validation": validation,
                        "raw_text": text or "",
                    },
                )
                continue
            assert parsed is not None
            parsed_ok += 1
            verdict_counts[str(parsed["verdict"])] += 1
            parsed.update(
                {
                    "idx": idx,
                    "candidate_id": candidate["candidate_id"],
                    "support_tier": candidate.get("support_tier"),
                    "caption_bucket": candidate.get("caption_bucket"),
                    "source_id": candidate.get("source_id"),
                    "target_id": candidate.get("target_id"),
                    "validation": validation,
                }
            )
            write_jsonl(judgments_path, parsed)
        except Exception as exc:  # noqa: BLE001 - smoke diagnostic should persist errors
            failures += 1
            write_jsonl(
                failures_path,
                {"idx": idx, "candidate_id": candidate["candidate_id"], "error": repr(exc)},
            )

    status = "completed" if failures == 0 else "completed_with_failures"
    summary.update(
        {
            "status": status,
            "candidates_attempted": len(candidates),
            "parsed_ok": parsed_ok,
            "failures": failures,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "verdict_counts": dict(verdict_counts),
            "validation_counts": dict(validation_counts),
            "files": {
                **summary["files"],
                "selected_candidates": str(selected_path.relative_to(ROOT)),
                "prompts": str(prompts_path.relative_to(ROOT)),
                "responses": str(responses_path.relative_to(ROOT)),
                "judgments": str(judgments_path.relative_to(ROOT)),
                "failures": str(failures_path.relative_to(ROOT)),
            },
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(
        "# idea:008 Company VLM Judge Smoke\n\n"
        f"- status: **{summary['status']}**\n"
        f"- model: `{summary['model']}`\n"
        f"- candidates attempted: **{len(candidates)}**\n"
        f"- parsed_ok: **{parsed_ok}**\n"
        f"- failures: **{failures}**\n"
        f"- verdict counts: `{dict(verdict_counts)}`\n"
        f"- tokens: **{total_in} in / {total_out} out**\n"
        f"- local_api_logger: `{summary['files']['local_api_logger']}`\n"
        f"- token_db: `{summary['files']['token_db']}`\n",
        encoding="utf-8",
    )
    log_run(
        script="experiments/idea008_company_vlm_judge_smoke.py",
        model=f"company:{args.model}",
        input_tokens=total_in,
        output_tokens=total_out,
        purpose="idea:008 company VLM judge smoke for cross-document edge promotion",
        extra={
            "pairs_processed": len(candidates),
            "queries_written": parsed_ok,
            "parse_failures": failures,
            "output": str(judgments_path.relative_to(ROOT)),
            "output_dir": str(out_dir.relative_to(ROOT)),
            "verdict_counts": dict(verdict_counts),
        },
    )
    print(f"[ok] status={status} parsed_ok={parsed_ok}/{len(candidates)} failures={failures}")
    print(f"[ok] verdict_counts={dict(verdict_counts)}")
    print(f"[ok] wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
