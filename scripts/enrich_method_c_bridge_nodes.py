#!/usr/bin/env python3
"""Enrich Method C bridge nodes with structured semantic summaries.

Bridge nodes come from long-chain bundles and typically correspond to
section / appendix / algorithm / theorem-like structural nodes.  Unlike
multimodal elements, they are enriched from reference-graph metadata and
edge-context snippets rather than MinerU images.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import call_llm, parse_json, set_company_credentials
from src.utils.token_logger import log_run


SYSTEM_PROMPT = (
    "You are a scientific document analyst. "
    "Summarize bridge nodes from long-chain academic graph paths into structured JSON. "
    "Output valid JSON only, no markdown fences, no extra text."
)

PROMPT_TEMPLATE = """Analyze this bridge node from an academic long-chain graph.

## Bridge metadata
Document ID: {doc_id}
Bridge ID: {bridge_id}
LaTeX label: {label_key}
Label type: {label_type}
Environment: {environment}
Line number: {line_no}
Source file: {file_path}
Caption/title hint: {caption}
Long-chain occurrences: {chain_count}
As endpoint count: {endpoint_count}
As intermediate bridge count: {intermediate_count}
Connected modalities: {connected_modalities}

## Neighbor labels
{neighbor_labels}

## Bridge contexts from reference edges
{edge_contexts}

## Instructions
Produce a structured description with exactly three fields:
1. **title**: A concise descriptive title (5-15 words) for the semantic role of this bridge node.
2. **metadata**: A JSON object with:
   - "bridge_type": one of ["section_bridge", "appendix_bridge", "algorithm_bridge", "theorem_bridge", "definition_bridge", "structural_bridge", "other"]
   - "semantic_role": one of ["problem_setup", "method_transition", "derivation_link", "experiment_setup", "result_interpretation", "cross_modal_bridge", "supporting_context", "other"]
   - "keywords": list of 4-8 keywords
   - "query_intents": list of 2-5 likely query intents this bridge supports
   - "connected_modalities": list copied / normalized from the inputs
3. **content**: A 2-4 sentence semantic summary.
   Explain what conceptual role this bridge plays in connecting evidence across the long chain.
   Use the edge contexts to infer whether it introduces assumptions, transitions, derivations, experimental setup, or result interpretation.
   Avoid meta-language like "this bridge shows"; summarize the actual connective function.

## Output format (JSON only)
{{
  "title": "...",
  "metadata": {{
    "bridge_type": "...",
    "semantic_role": "...",
    "keywords": ["...", "..."],
    "query_intents": ["...", "..."],
    "connected_modalities": ["...", "..."]
  }},
  "content": "..."
}}"""


def call_api(client: Any, model: str, prompt: str, provider: str) -> Tuple[Optional[str], int, int]:
    return call_llm(
        client, model, prompt,
        provider=provider,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=768,
        temperature=0.2,
        user_tag="method_c_bridge_enrichment",
    )


def validate_enrichment(result: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    if not (result.get("title") or "").strip():
        issues.append("missing_title")
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        issues.append("missing_metadata")
        metadata = {}
    if not isinstance(metadata.get("keywords"), list) or not metadata.get("keywords"):
        issues.append("missing_keywords")
    if not isinstance(metadata.get("query_intents"), list) or not metadata.get("query_intents"):
        issues.append("missing_query_intents")
    if not isinstance(metadata.get("connected_modalities"), list):
        issues.append("missing_connected_modalities")
    if not (result.get("content") or "").strip():
        issues.append("missing_content")
    return issues


def build_prompt(row: Dict[str, Any]) -> str:
    edge_contexts = row.get("edge_contexts", []) or []
    neighbors = row.get("neighbor_labels", []) or []

    edge_text = "\n".join(
        f"- {ctx[:500]}" for ctx in edge_contexts[:8]
    ) if edge_contexts else "(no bridge contexts found)"
    neighbor_text = "\n".join(
        f"- {n.get('label_key')} [{n.get('label_type')}] via {n.get('ref_text')}"
        for n in neighbors[:8]
    ) if neighbors else "(no neighbor labels found)"

    return PROMPT_TEMPLATE.format(
        doc_id=row.get("doc_id", ""),
        bridge_id=row.get("bridge_id", ""),
        label_key=row.get("label_key", ""),
        label_type=row.get("label_type", ""),
        environment=row.get("environment", ""),
        line_no=row.get("line_no", ""),
        file_path=row.get("file_path", ""),
        caption=(row.get("caption") or "")[:300] or "(none)",
        chain_count=row.get("chain_count", 0),
        endpoint_count=row.get("endpoint_count", 0),
        intermediate_count=row.get("intermediate_count", 0),
        connected_modalities=", ".join(row.get("connected_modalities", []) or []) or "(none)",
        neighbor_labels=neighbor_text,
        edge_contexts=edge_text,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/02_enriched/method_c_long_chain_bridge_targets_v1.json")
    ap.add_argument("--output", default="data/02_enriched/method_c_long_chain_bridges_enriched_v1.json")
    ap.add_argument("--provider", choices=["anthropic", "openai", "company"], default="company")
    ap.add_argument("--model", default=None)
    ap.add_argument("--company-api-url", default=os.environ.get("COMPANY_API_URL", ""))
    ap.add_argument("--company-api-key", default=os.environ.get("COMPANY_API_KEY", ""))
    ap.add_argument("--delay", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--incremental", action="store_true")
    ap.add_argument("--flush-every", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.model is None:
        if args.provider == "anthropic":
            args.model = "claude-sonnet-4-5-20250929"
        elif args.provider == "openai":
            args.model = "gpt-4o"
        else:
            args.model = "gpt-5.4"

    in_path = PROJECT_ROOT / args.input
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(in_path.read_text(encoding="utf-8"))
    rows = list(data.get("bridge_targets", []))
    if args.limit > 0:
        rows = rows[:args.limit]

    client: Any = None
    if not args.dry_run:
        if args.provider == "anthropic":
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise SystemExit("ANTHROPIC_API_KEY not set")
            client = anthropic.Anthropic(api_key=api_key)
        elif args.provider == "openai":
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise SystemExit("OPENAI_API_KEY not set")
            client = OpenAI(api_key=api_key)
        else:
            if not args.company_api_url or not args.company_api_key:
                raise SystemExit("COMPANY_API_URL / COMPANY_API_KEY not set")
            set_company_credentials(args.company_api_url, args.company_api_key)

    print(f"Loaded bridge targets: {len(rows)}")

    total_in_tok = 0
    total_out_tok = 0
    processed = 0
    failed = 0
    results: List[Dict[str, Any]] = []
    done_ids: set[str] = set()

    if args.incremental and out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            prev_rows = existing.get("bridges", [])
            results.extend(prev_rows)
            done_ids = {r["bridge_id"] for r in prev_rows if "bridge_id" in r}
            prev_meta = existing.get("metadata", {})
            total_in_tok = int(prev_meta.get("input_tokens", 0))
            total_out_tok = int(prev_meta.get("output_tokens", 0))
            processed = len(prev_rows)
            print(f"Incremental: loaded {len(done_ids)} done bridge targets")
        except Exception as exc:
            print(f"WARNING: failed to load existing bridge output: {exc}")

    def _flush(reason: str = "") -> None:
        snapshot = {
            "metadata": {
                "source": str(in_path),
                "model": f"{args.provider}:{args.model}",
                "total_requested": len(rows),
                "processed": processed,
                "failed": failed,
                "input_tokens": total_in_tok,
                "output_tokens": total_out_tok,
                "partial": True,
            },
            "bridges": results,
        }
        out_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
        if reason:
            print(f"  [flush] {reason} — {len(results)} bridges saved")

    new_count = 0
    for idx, row in enumerate(rows, start=1):
        if row["bridge_id"] in done_ids:
            continue
        prompt = build_prompt(row)

        if args.dry_run:
            print(f"[{idx}/{len(rows)}] {row['bridge_id']} — DRY RUN")
            if idx <= 2:
                print(prompt[:600] + "...")
            results.append(
                {
                    **row,
                    "enriched_title": f"[DRY RUN] {row['label_key']}",
                    "enriched_metadata": {"keywords": [], "query_intents": [], "connected_modalities": row.get("connected_modalities", [])},
                    "enriched_content": "[DRY RUN]",
                    "enrichment_issues": [],
                }
            )
            processed += 1
            new_count += 1
            continue

        try:
            text, in_tok, out_tok = call_api(client, args.model, prompt, args.provider)
            total_in_tok += in_tok
            total_out_tok += out_tok
            parsed = parse_json(text)
            if parsed is None:
                print(f"[{idx}/{len(rows)}] {row['bridge_id']} — PARSE FAIL")
                failed += 1
                new_count += 1
                if args.flush_every > 0 and new_count % args.flush_every == 0:
                    _flush(f"after {new_count} new items")
                continue

            issues = validate_enrichment(parsed)
            results.append(
                {
                    **row,
                    "enriched_title": parsed.get("title", ""),
                    "enriched_metadata": parsed.get("metadata", {}),
                    "enriched_content": parsed.get("content", ""),
                    "enrichment_issues": issues,
                }
            )
            processed += 1
            new_count += 1
            if issues:
                print(f"[{idx}/{len(rows)}] {row['bridge_id']} — OK with issues: {issues}")
            else:
                print(f"[{idx}/{len(rows)}] {row['bridge_id']} — OK")
        except Exception as exc:
            print(f"[{idx}/{len(rows)}] {row['bridge_id']} — ERROR: {exc}")
            failed += 1
            new_count += 1

        if args.flush_every > 0 and new_count % args.flush_every == 0:
            _flush(f"after {new_count} new items")

        if args.delay > 0 and idx < len(rows):
            time.sleep(args.delay)

    output = {
        "metadata": {
            "source": str(in_path),
            "model": f"{args.provider}:{args.model}",
            "total_requested": len(rows),
            "processed": processed,
            "failed": failed,
            "input_tokens": total_in_tok,
            "output_tokens": total_out_tok,
        },
        "bridges": results,
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote bridge enrichments: {out_path} ({new_count} new this run)")

    if not args.dry_run:
        log_run(
            script="enrich_method_c_bridge_nodes",
            model=f"{args.provider}:{args.model}",
            purpose=f"Method C bridge enrichment — {processed} enriched, {failed} failed",
            input_tokens=total_in_tok,
            output_tokens=total_out_tok,
            extra={
                "pairs_processed": processed,
                "parse_failures": failed,
                "output": str(out_path),
            },
        )


if __name__ == "__main__":
    main()
