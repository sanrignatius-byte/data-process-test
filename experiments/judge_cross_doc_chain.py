#!/usr/bin/env python3
"""Judge fixed cross-document multi-hop chains.

This evaluates whether a 3-paper / 2-bridge entity chain is a meaningful
scientific evidence path, not just a pile of shared keywords.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    "broken_chain",
    "spurious_entity_match",
    "insufficient_context",
}

PRODUCTION_USE = {"keep", "review", "drop"}


def clip(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 3] + "..."


def load_enriched() -> dict[str, dict[str, Any]]:
    path = ROOT / "data/02_enriched/multimodal_elements_enriched.json"
    idx: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for doc_id, doc in data.get("documents", {}).items():
        for eid, el in doc.get("elements", {}).items():
            idx[eid] = {
                "doc_id": doc_id,
                "element_id": eid,
                "element_type": el.get("element_type", ""),
                "caption": el.get("caption", "") or el.get("content", ""),
                "enriched_title": el.get("enriched_title", ""),
                "enriched_content": el.get("enriched_content", ""),
            }
    return idx


def load_chains(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    chains = data["chains"] if isinstance(data, dict) and "chains" in data else data
    if limit:
        chains = chains[:limit]
    return chains


def enrich_element(e: dict[str, Any], enriched_idx: dict[str, dict[str, Any]]) -> dict[str, Any]:
    eid = e.get("element_id", "")
    base = enriched_idx.get(eid, {})
    out = dict(e)
    for key in ("caption", "enriched_title", "enriched_content", "element_type"):
        if not out.get(key) and base.get(key):
            out[key] = base[key]
    out["node_type"] = out.get("node_type") or out.get("element_type") or base.get("element_type", "")
    return out


def format_element(e: dict[str, Any]) -> str:
    return "\n".join([
        f"- doc: {e.get('doc_id', '')}",
        f"  element_id: {e.get('element_id', '')}",
        f"  type: {e.get('node_type') or e.get('element_type', '')}",
        f"  role: {e.get('role', '')}",
        f"  title: {clip(e.get('enriched_title', ''), 180)}",
        f"  caption: {clip(e.get('caption', ''), 260)}",
        f"  enriched_content: {clip(e.get('enriched_content', ''), 420)}",
    ])


def format_bridge(i: int, b: dict[str, Any]) -> str:
    entities = b.get("clean_shared_entities") or b.get("shared_entities", [])
    return "\n".join([
        f"Bridge {i}: {b.get('from_doc', '')} [{b.get('from_element_id', '')}] -> "
        f"{b.get('to_doc', '')} [{b.get('to_element_id', '')}]",
        f"  shared_entities: {entities[:8]}",
        f"  bridge_score: {b.get('bridge_score', b.get('score', 0))}",
        f"  bridge_description: {clip(b.get('bridge_description', ''), 300)}",
    ])


def build_prompt(chain: dict[str, Any], enriched_idx: dict[str, dict[str, Any]]) -> str:
    elems = [enrich_element(e, enriched_idx) for e in chain.get("elements", [])]
    bridges = chain.get("bridges", [])
    element_text = "\n\n".join(format_element(e) for e in elems)
    bridge_text = "\n\n".join(format_bridge(i, b) for i, b in enumerate(bridges, 1))
    all_entities = chain.get("shared_entities", [])
    entity_context = chain.get("entity_context", {})
    if isinstance(entity_context, dict) and entity_context:
        context_lines = []
        for ent, ctx in list(entity_context.items())[:8]:
            context_lines.append(f"- {ent}: {clip(ctx, 700)}")
        entity_context_text = "\n".join(context_lines)
    else:
        entity_context_text = "(not provided)"

    return f"""You are judging whether a multi-hop cross-document scientific chain is useful for generating M4-style training queries.

The chain is expected to be:
- a path across 3 papers,
- with 2 cross-document bridge edges,
- grounded by concrete multimodal elements (figures/tables/formulas),
- connected by specific research entities rather than generic keywords.

Use a conservative production-data standard:
- strong_chain: Every bridge is specifically supported by its two endpoint elements, and the full path forms a coherent scientific chain. This is suitable as production training material.
- weak_but_related: The bridges are scientifically related but loose, or one hop is broad while still meaningful. Suitable only with human review or as weak data.
- topic_only: The papers share a broad area but the chain does not support a specific multi-hop relation.
- broken_chain: At least one bridge endpoint does not discuss the claimed entity, so the path breaks.
- spurious_entity_match: The chain is driven by visual/typographic/artifact terms, generic words, homonyms, or extraction noise.
- insufficient_context: The element metadata is too thin to decide.

Important:
- Judge the two bridge edges separately, then judge the full chain.
- A 3-paper chain is strong only if the middle paper acts as a meaningful relay, not merely because it shares unrelated entities with both sides.
- Shared entities like "linear model" or "outcome" are generic unless the endpoint elements clearly instantiate the same scientific construct.
- Shared entities like a named benchmark, dataset, metric, fairness criterion, causal construct, or method variant can support a strong chain when concretely present.
- Return valid JSON only.

Required JSON:
{{
  "verdict": "strong_chain | weak_but_related | topic_only | broken_chain | spurious_entity_match | insufficient_context",
  "confidence": 0.0,
  "production_use": "keep | review | drop",
  "chain_coherence": "high | medium | low | none",
  "middle_paper_is_relay": true,
  "main_failure": "none | generic_entity | one_bad_bridge | disconnected_middle | typographic_artifact | different_scientific_objects | missing_context | other",
  "bridge_judgments": [
    {{
      "bridge_index": 1,
      "verdict": "strong | weak | topic_only | broken | spurious | insufficient",
      "supported": true,
      "rationale": "one concise sentence"
    }}
  ],
  "rationale": "2-4 concise sentences explaining whether this is a usable cross-document evidence chain.",
  "evidence": {{
    "best_shared_concept": "specific concept that makes the chain work, or none",
    "weakest_link": "which bridge or element limits chain quality",
    "production_note": "brief recommendation for production filtering"
  }}
}}

Candidate:
- chain_id: {chain.get('chain_id', '')}
- source_chain_id: {chain.get('source_chain_id', '')}
- papers: {chain.get('papers', [])}
- element_types: {chain.get('element_types', [])}
- shared_entities: {all_entities[:12]}
- strategy: {chain.get('strategy', '')}

Entity-level context:
{entity_context_text}

Elements:
{element_text}

Cross-document bridges:
{bridge_text}

Question: Is this full chain a meaningful multi-hop cross-document scientific evidence path suitable for production M4-query training data?
"""


def validate(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {"valid": False, "reason": "not_object"}
    required = [
        "verdict", "confidence", "production_use", "chain_coherence",
        "middle_paper_is_relay", "main_failure", "bridge_judgments",
        "rationale", "evidence",
    ]
    missing = [k for k in required if k not in obj]
    if missing:
        return {"valid": False, "reason": "missing:" + ",".join(missing)}
    if obj.get("verdict") not in VERDICTS:
        return {"valid": False, "reason": f"bad_verdict:{obj.get('verdict')}"}
    if obj.get("production_use") not in PRODUCTION_USE:
        return {"valid": False, "reason": f"bad_production_use:{obj.get('production_use')}"}
    try:
        conf = float(obj.get("confidence"))
    except (TypeError, ValueError):
        return {"valid": False, "reason": "bad_confidence"}
    if not 0 <= conf <= 1:
        return {"valid": False, "reason": "confidence_oob"}
    if not isinstance(obj.get("bridge_judgments"), list):
        return {"valid": False, "reason": "bad_bridge_judgments"}
    return {"valid": True, "reason": "ok"}


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts: Counter[str] = Counter()
    prod: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    bridge_verdicts: Counter[str] = Counter()

    for j in judgments:
        jj = j.get("judgment")
        verdict = jj.get("verdict", "parse_failed") if isinstance(jj, dict) else "parse_failed"
        verdicts[verdict] += 1
        if isinstance(jj, dict):
            prod[jj.get("production_use", "unknown")] += 1
            failures[jj.get("main_failure", "unknown")] += 1
            for bj in jj.get("bridge_judgments", []):
                if isinstance(bj, dict):
                    bridge_verdicts[bj.get("verdict", "unknown")] += 1
        by_type["+".join(j.get("element_types", []))][verdict] += 1

    total = len(judgments)
    strong = verdicts.get("strong_chain", 0)
    usable = strong + verdicts.get("weak_but_related", 0)
    keep = prod.get("keep", 0)
    keep_review = keep + prod.get("review", 0)
    return {
        "total": total,
        "strong_chain": strong,
        "strong_rate": round(strong / total, 4) if total else 0,
        "usable_chain": usable,
        "usable_rate": round(usable / total, 4) if total else 0,
        "production_keep": keep,
        "production_keep_rate": round(keep / total, 4) if total else 0,
        "production_keep_or_review": keep_review,
        "production_keep_or_review_rate": round(keep_review / total, 4) if total else 0,
        "verdict_counts": dict(verdicts.most_common()),
        "production_counts": dict(prod.most_common()),
        "main_failure_counts": dict(failures.most_common()),
        "bridge_verdict_counts": dict(bridge_verdicts.most_common()),
        "validation_counts": dict(Counter(j.get("validation", {}).get("reason", "unknown") for j in judgments)),
        "by_element_signature": {k: dict(v) for k, v in sorted(by_type.items())},
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Cross-Document Chain Judge Summary",
        "",
        f"- total: **{summary['total']}**",
        f"- strong_chain: **{summary['strong_chain']}** ({summary['strong_rate']:.1%})",
        f"- usable_chain: **{summary['usable_chain']}** ({summary['usable_rate']:.1%})",
        f"- production keep: **{summary['production_keep']}** ({summary['production_keep_rate']:.1%})",
        f"- production keep/review: **{summary['production_keep_or_review']}** ({summary['production_keep_or_review_rate']:.1%})",
        f"- model: `{summary['model']}`",
        "",
        "## Verdict Counts",
        "",
    ]
    for key, value in summary.get("verdict_counts", {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Production Counts", ""])
    for key, value in summary.get("production_counts", {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Main Failures", ""])
    for key, value in summary.get("main_failure_counts", {}).items():
        lines.append(f"- `{key}`: {value}")
    return "\n".join(lines)


def make_out_dir(base: str) -> Path:
    out_dir = ROOT / base if base else ROOT / (
        "data/05_eval/cross_doc_chain_judge_" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def existing_judgments(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return seen, rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("chain_id") or row.get("candidate_id")
            if cid:
                seen.add(cid)
            rows.append(row)
    return seen, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge fixed cross-document chains")
    parser.add_argument("--input", default="data/05_eval/cross_doc_chains_final_fixed.json")
    parser.add_argument("--output-dir", default="data/05_eval/cross_doc_chain_judge_fixed")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep-between", type=float, default=0.0)
    parser.add_argument("--rate-limit-sleep", type=float, default=90.0)
    parser.add_argument("--company-api-url", default=os.environ.get("COMPANY_API_URL", ""))
    parser.add_argument("--company-api-key", default=os.environ.get("COMPANY_API_KEY", ""))
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    out_dir = make_out_dir(args.output_dir)
    set_company_credentials(args.company_api_url, args.company_api_key)

    prompts_path = out_dir / "prompts.jsonl"
    responses_path = out_dir / "responses.jsonl"
    judgments_path = out_dir / "judgments.jsonl"
    failures_path = out_dir / "failures.jsonl"

    if not args.resume:
        for p in (prompts_path, responses_path, judgments_path, failures_path):
            if p.exists():
                p.unlink()

    already, judgments = existing_judgments(judgments_path) if args.resume else (set(), [])
    chains = load_chains(input_path, limit=args.limit)
    enriched_idx = load_enriched()

    total_in = 0
    total_out = 0
    print(f"Loaded {len(chains)} chains")
    print(f"Already judged: {len(already)}")
    print(f"Output: {out_dir.relative_to(ROOT)}")
    print(f"Model: {args.model}")

    for idx, chain in enumerate(chains, 1):
        cid = chain.get("chain_id", f"chain_{idx:04d}")
        if cid in already:
            continue
        prompt = build_prompt(chain, enriched_idx)
        write_jsonl(prompts_path, {
            "chain_id": cid,
            "prompt": prompt,
            "paper_count": len(chain.get("papers", [])),
            "element_count": len(chain.get("elements", [])),
        })

        raw = None
        tin = tout = 0
        last_error = ""
        for attempt in range(5):
            try:
                raw, tin, tout = call_llm(
                    client=None,
                    model=args.model,
                    provider="company",
                    prompt=prompt,
                    images=[],
                    system_prompt=(
                        "You are a conservative scientific multi-hop chain judge. "
                        "Return valid JSON only."
                    ),
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    user_tag="cross_doc_chain_judge",
                )
                break
            except Exception as exc:  # noqa: BLE001 - batch judge should keep going
                last_error = repr(exc)
                is_rate_limit = "429" in last_error or "Too Many Requests" in last_error
                sleep_s = args.rate_limit_sleep * (attempt + 1) if is_rate_limit else 10 * (attempt + 1)
                print(f"  [attempt {attempt + 1}/5] {cid} API error: {last_error}", flush=True)
                if attempt < 4:
                    print(f"  sleeping {sleep_s:.1f}s before retry", flush=True)
                    time.sleep(sleep_s)

        total_in += tin
        total_out += tout
        if raw is None and last_error:
            raise RuntimeError(
                f"API errors persisted for {cid}: {last_error}. "
                "Stopping without writing api_failed judgments. "
                "Rerun with --resume later."
            )
        parsed = parse_json(raw or "")
        validation = validate(parsed)
        out = {
            "chain_id": cid,
            "judge_index": idx,
            "source_chain_id": chain.get("source_chain_id", ""),
            "papers": chain.get("papers", []),
            "element_ids": [e.get("element_id", "") for e in chain.get("elements", [])],
            "element_types": chain.get("element_types", []),
            "bridge_entities": [b.get("clean_shared_entities") or b.get("shared_entities", []) for b in chain.get("bridges", [])],
            "shared_entities": chain.get("shared_entities", []),
            "judgment": parsed if isinstance(parsed, dict) else None,
            "validation": validation,
            "tokens": {"in": tin, "out": tout},
        }
        if raw is None:
            out["api_error"] = last_error
            out["validation"] = {"valid": False, "reason": "api_failed"}
        judgments.append(out)
        write_jsonl(judgments_path, out)
        write_jsonl(responses_path, {"chain_id": cid, "raw": raw, "tokens": {"in": tin, "out": tout}})
        if not validation["valid"] or raw is None:
            write_jsonl(failures_path, out | {"raw": raw})

        verdict = out["judgment"].get("verdict") if out["judgment"] else "parse_failed"
        prod = out["judgment"].get("production_use") if out["judgment"] else "?"
        print(f"[{idx:03d}/{len(chains):03d}] {cid} -> {verdict} / {prod}", flush=True)
        if args.sleep_between > 0:
            time.sleep(args.sleep_between)

    summary = {
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "input": str(input_path.relative_to(ROOT)),
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
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")

    log_run(
        script="experiments/judge_cross_doc_chain.py",
        model=f"company:{args.model}",
        purpose="Judge fixed 3-paper cross-document chains before production run",
        input_tokens=total_in,
        output_tokens=total_out,
        extra={
            "chains_judged": len(judgments),
            "strong_chain": summary["strong_chain"],
            "output": str(out_dir.relative_to(ROOT)),
        },
    )

    print("\n=== Judge Complete ===")
    print(f"Total: {summary['total']}")
    print(f"Strong: {summary['strong_chain']} ({summary['strong_rate']:.1%})")
    print(f"Usable: {summary['usable_chain']} ({summary['usable_rate']:.1%})")
    print(f"Production keep: {summary['production_keep']} ({summary['production_keep_rate']:.1%})")
    print(f"Verdicts: {summary['verdict_counts']}")
    print(f"Production: {summary['production_counts']}")
    print(f"Failures: {summary['main_failure_counts']}")


if __name__ == "__main__":
    main()
