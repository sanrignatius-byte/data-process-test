#!/usr/bin/env python3
"""Judge entity-bridge cross-document candidates with the company VLM/LLM path.

Unlike the resolver-v1 judge (which asked "does the citation bridge discuss the target
element?"), this judge asks: "Do these two elements from different papers discuss the
SAME specific research entity (method, dataset, metric, model) such that they form
a valid cross-document evidence chain?"

The bridge is a set of shared enriched keywords — the question is whether the elements
genuinely instantiate those shared concepts in a way that supports a cross-doc chain.
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

VERDICTS = {
    "strong_chain",
    "weak_but_related",
    "topic_only",
    "wrong_target",
    "wrong_source",
    "insufficient_context",
}


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
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
    return s if len(s) <= limit else s[: limit - 3] + "..."


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
    meta = row.get("_meta", {})
    entities = meta.get("shared_entities", [])
    entity_list = "\n".join(f"  - {e}" for e in entities[:10])

    return f"""You are judging whether two elements from DIFFERENT scientific papers form a valid cross-document evidence chain, linked by shared research concepts (entities).

Two endpoint images MAY be attached:
1. source element image from Paper A
2. target element image from Paper B

Your task: determine whether these two elements discuss the SAME specific research entity
(method, dataset, metric, model, task) such that an M4 multi-hop chain can reasonably
pass through them.

Shared entities (extracted from element enriched metadata):
{entity_list}

Use a conservative standard:
- strong_chain: Both elements specifically and concretely discuss the SAME entity. The entity appears as a core subject of both elements (not just mentioned in passing). A researcher would naturally chain these elements together.
- weak_but_related: The elements discuss closely related aspects of the shared entity, but not precisely the same thing. There is a real scientific connection, but the chain is loose.
- topic_only: Both elements are in the same broad research area, but the specific shared entities are superficial or the elements discuss different aspects of them.
- wrong_target: The target element does NOT meaningfully discuss the claimed shared entities. The entity match is spurious (e.g., a homonym or a generic word match).
- wrong_source: The source element does NOT meaningfully discuss the claimed shared entities.
- insufficient_context: Not enough caption/text information to decide.

CRITICAL EVALUATION RULES:
- Do NOT assume two elements are linked just because they share a keyword. A table about "ImageNet accuracy" and a figure about "ImageNet examples" share "ImageNet" but talk about COMPLETELY different things.
- Look for CONCRETE correspondence: same benchmark, same metric, same method variant, same dataset split, same theoretical quantity.
- If the source element is about method performance and the target element is about dataset statistics, they are topic_only at best.
- If both elements report accuracy/F1/BLEU for the same named benchmark or method variant, that is strong_chain.
- If both elements visualize or formalize the same mathematical object (e.g., the same fairness metric, the same causal graph structure), that is strong_chain.
- Return only valid JSON.

Required JSON:
{{
  "verdict": "strong_chain | weak_but_related | topic_only | wrong_target | wrong_source | insufficient_context",
  "confidence": 0.0,
  "target_supported": true,
  "source_anchor_supported": true,
  "bridge_specificity": "exact_entity_match | closely_related_entity | same_broad_topic | spurious_entity_match | homonym | none",
  "main_failure": "none | target_element_does_not_discuss_entity | source_element_does_not_discuss_entity | entity_too_generic | different_aspects_of_same_entity | visual_only | missing_context | other",
  "rationale": "2-4 concise sentences explaining whether both elements discuss the same specific entity and why",
  "evidence": {{
    "entity_cue_source": "how the shared entity appears in the source element",
    "entity_cue_target": "how the shared entity appears in the target element",
    "bridge_quality": "specific_match | broad_overlap | spurious | insufficient"
  }}
}}

Candidate:
- candidate_id: {row.get("candidate_id")}
- source_doc: {row.get("source_doc")}
- target_doc: {row.get("target_doc")}
- source_element_id: {row.get("source_element_id")}
- source_element_type: {row.get("source_element_type")}
- target_element_id: {row.get("target_element_id")}
- target_element_type: {row.get("target_element_type")}
- pair_type: {row.get("pair_type")}
- shared_entities: {entities[:8]}
- entity_bridge_score: {meta.get("entity_bridge_score", 0):.3f}
- citation_link: {meta.get("citation_direction", "none")}

Source element caption/content:
{clip(row.get("source_caption_or_content", ""), 800)}

Source element enriched description:
{clip(row.get("citation_bridge_text", "").split("Source element [")[1].split("Target element [")[0] if "Source element [" in row.get("citation_bridge_text", "") else "", 600)}

Target element caption/content:
{clip(row.get("target_caption_or_content", ""), 800)}

Target element enriched description:
{clip(row.get("citation_bridge_text", "").split("Target element [")[1] if "Target element [" in row.get("citation_bridge_text", "") else "", 600)}

Question: Do element [{row.get("source_element_id")}] and element [{row.get("target_element_id")}] both specifically discuss the shared entity [{entities[0] if entities else 'unknown'}] (and related: {', '.join(entities[1:4]) if len(entities) > 1 else 'none'}) such that they form a semantically meaningful cross-document evidence chain?
"""


def validate(obj: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {"valid": False, "reason": "not_object"}
    missing = [
        key for key in (
            "verdict", "confidence", "target_supported",
            "source_anchor_supported", "bridge_specificity",
            "main_failure", "rationale", "evidence",
        ) if key not in obj
    ]
    if missing:
        return {"valid": False, "reason": "missing:" + ",".join(missing)}
    if obj.get("verdict") not in VERDICTS:
        return {"valid": False, "reason": f"bad_verdict:{obj.get('verdict')}"}
    try:
        conf = float(obj.get("confidence"))
    except (TypeError, ValueError):
        return {"valid": False, "reason": "bad_confidence"}
    if not 0 <= conf <= 1:
        return {"valid": False, "reason": "confidence_oob"}
    return {"valid": True, "reason": "ok"}


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts: Counter[str] = Counter()
    by_stratum: dict[str, Counter[str]] = defaultdict(Counter)
    by_pair_type: dict[str, Counter[str]] = defaultdict(Counter)
    for j in judgments:
        v = j["judgment"].get("verdict", "parse_failed") if isinstance(j.get("judgment"), dict) else "parse_failed"
        verdicts[v] += 1
        s = j.get("target_stratum", "unknown")
        pt = j.get("pair_type", "unknown")
        by_stratum[s][v] += 1
        by_pair_type[pt][v] += 1
    total = len(judgments)
    strong = verdicts.get("strong_chain", 0)
    return {
        "total": total,
        "strong_chain": strong,
        "strong_rate": round(strong / total, 4) if total else 0,
        "verdict_counts": {k: v for k, v in verdicts.most_common()},
        "validation_counts": dict(Counter(
            j.get("validation", {}).get("reason", "unknown") for j in judgments
        )),
        "by_stratum": {s: dict(c) for s, c in sorted(by_stratum.items())},
        "by_pair_type": {pt: dict(c) for pt, c in sorted(by_pair_type.items())},
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Entity-Bridge Judge Summary",
        "",
        f"- total: **{summary['total']}**",
        f"- strong_chain: **{summary['strong_chain']}** ({summary['strong_rate']:.1%})",
        f"- model: `{summary['model']}`",
    ]
    lines.append("\n## Verdict Counts\n")
    for v, c in summary.get("verdict_counts", {}).items():
        lines.append(f"- `{v}`: {c}")
    if summary.get("by_stratum"):
        lines.append("\n## By Stratum\n")
        for s, vc in summary["by_stratum"].items():
            inner = ", ".join(f"{k}={v}" for k, v in vc.items())
            lines.append(f"- `{s}`: {inner}")
    return "\n".join(lines)


def make_out_dir(base: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if base:
        out_dir = ROOT / base
    else:
        out_dir = ROOT / f"data/05_eval/entity_bridge_judge_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Judge entity-bridge cross-document element candidates"
    )
    parser.add_argument(
        "--pack",
        default=str(ROOT / "data/05_eval/entity_bridge_candidates_latest/judge_pack.jsonl"),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--limit", type=int, default=30, help="Max candidates to judge (0=all)")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--company-api-url", default=os.environ.get("COMPANY_API_URL", ""))
    parser.add_argument("--company-api-key", default=os.environ.get("COMPANY_API_KEY", ""))
    args = parser.parse_args()

    pack_path = Path(args.pack)
    if not pack_path.is_absolute():
        pack_path = ROOT / pack_path
    rows = load_jsonl(pack_path, limit=args.limit)
    out_dir = make_out_dir(args.output_dir)
    set_company_credentials(args.company_api_url, args.company_api_key)

    prompts_path = out_dir / "prompts.jsonl"
    responses_path = out_dir / "responses.jsonl"
    judgments_path = out_dir / "judgments.jsonl"
    failures_path = out_dir / "failures.jsonl"

    # Clean for fresh run
    for p in [prompts_path, responses_path, judgments_path, failures_path]:
        if p.exists():
            p.unlink()

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
                "Return valid JSON only. Do not be fooled by keyword overlap — "
                "verify that both elements concretely discuss the same specific entity."
            ),
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            user_tag="entity_bridge_judge",
        )
        total_in += tin
        total_out += tout
        parsed = parse_json(raw or "")
        validation = validate(parsed)

        out = {
            "candidate_id": row.get("candidate_id"),
            "judge_index": row.get("judge_index"),
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
            f"[{idx:03d}/{len(rows):03d}] {row.get('source_doc')}->{row.get('target_doc')} "
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
            "prompts": str(prompts_path.relative_to(ROOT)),
            "responses": str(responses_path.relative_to(ROOT)),
            "judgments": str(judgments_path.relative_to(ROOT)),
            "failures": str(failures_path.relative_to(ROOT)),
        },
        "tokens": {"in": total_in, "out": total_out},
        **summarize(judgments),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")

    log_run(
        script="experiments/judge_entity_bridge_pack.py",
        model=f"company:{args.model}",
        purpose="Judge entity-bridge cross-document element candidates",
        input_tokens=total_in,
        output_tokens=total_out,
        extra={
            "candidates_judged": len(judgments),
            "strong_chain": summary["strong_chain"],
            "output": str(out_dir.relative_to(ROOT)),
        },
    )

    print(f"\n=== Judge Complete ===")
    print(f"Total: {summary['total']}")
    print(f"Strong chain: {summary['strong_chain']} ({summary['strong_rate']:.1%})")
    print(f"Verdicts: {summary['verdict_counts']}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
