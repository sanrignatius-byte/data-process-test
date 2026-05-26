#!/usr/bin/env python3
"""Generate retrieval-style M4 queries for embedding triplet training.

Unlike Method C (research-comprehension "Why would X support Y?") and
multiturn-app (scenario-based "Which X should they choose?"), this generator
produces queries that match what a PPT/RAG agent would actually issue:
descriptions of what multimodal elements to find and how they connect.

Each material produces 2-3 query variants for triplet diversity.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api import call_llm, parse_json, set_company_credentials  # noqa: E402

DEFAULT_MATERIALS = ROOT / "data/05_eval/m4_enriched_materials_latest/m4_material_pack.jsonl"
DEFAULT_OUT_ROOT = ROOT / "data/05_eval"
DEFAULT_MODEL = "gpt-5.4"

RETRIEVAL_PROMPT = """You are converting multimodal academic evidence pairs into
retrieval-style queries for training an embedding model. The embedding model will
be used by a PPT/slide-generation agent that needs to find relevant cross-document
multimodal element pairs.

### WHAT A RETRIEVAL QUERY IS

A retrieval query describes WHAT ELEMENTS the user wants to find and HOW THEY
CONNECT. It is NOT a comprehension question or an applied problem.

Good: "Architecture diagram comparing self-attention variants across transformer
papers, paired with a FLOPs-vs-accuracy benchmark table from an efficient-attention
survey"
Bad: "Why would replacing standard attention with sparse attention improve
throughput while maintaining perplexity?"

The query should sound like what a researcher would type into a semantic search
engine when building a cross-document comparison slide.

### INPUT MATERIAL

**Element A** ({type_a}):
Title: {title_a}
Content: {content_a}

**Element B** ({type_b}):
Title: {title_b}
Content: {content_b}

**Bridge — how these elements connect across documents:**
{bridge_text}

### CONSTRAINTS

1. Generate {n_variants} distinct retrieval query variants for this material.
2. Each query is 15-60 words, in natural English.
3. Describe what to FIND, not what to UNDERSTAND.
4. Include cues about element modality (figure/table/formula) when relevant to
   disambiguation, but don't over-specify (leave room for embedding generalization).
5. Each variant should emphasize a DIFFERENT retrieval angle:
   - One: the bridge concept (what connects the two elements)
   - One: the contrast or comparison between the two elements
   - One (if 3 variants): the specific domain, method, or task context
6. Avoid template openings: no "Find a...", no "Locate the...", no "Search for..."
   as the first 3 words of every query. Vary the phrasing.
7. Keep the query self-contained — it should be parseable without the original
   paper context.

### OUTPUT FORMAT (valid JSON only)

{{
  "queries": [
    "retrieval query variant 1",
    "retrieval query variant 2",
    "retrieval query variant 3"
  ],
  "triplet_metadata": {{
    "modality_pair": "{pair_type}",
    "retrieval_difficulty": "easy|medium|hard",
    "requires_cross_doc": true,
    "domain_keywords": ["keyword1", "keyword2", "keyword3"]
  }}
}}"""


def _elem_content(elem: dict) -> str:
    """Build a rich natural language description of an element."""
    parts = []
    # Enriched fields come first — they're the most NL
    for k in ("enriched_title", "enriched_content"):
        v = elem.get(k, "")
        if v and isinstance(v, str) and len(v.strip().split()) >= 3:
            parts.append(str(v).strip())
    # Then caption
    caption = elem.get("caption", "") or elem.get("label", "")
    if caption and isinstance(caption, str) and len(caption.strip().split()) >= 2:
        parts.append(str(caption).strip())
    # context as last resort
    ctx = elem.get("context_before", "") or ""
    if ctx and len(parts) < 2:
        parts.append(str(ctx)[:300])
    return " ".join(parts)[:800] or f"{elem.get('element_type', 'element')} from the document"


def _bridge_text(pair: dict) -> str:
    mc = pair.get("method_c", {})
    bridges = mc.get("compressed_bridge_summaries", [])
    if bridges:
        return " ".join(str(b)[:400] for b in bridges)[:900]
    hub = pair.get("hub_metadata", {})
    summary = hub.get("hub_semantic_summary", "")
    if summary:
        return str(summary)[:500]
    return "These elements are connected through cross-document citation and structural bridges."


def build_retrieval_prompt(pair: dict, n_variants: int = 3) -> str:
    ea = pair.get("element_a", {})
    eb = pair.get("element_b", {})
    return RETRIEVAL_PROMPT.format(
        type_a=ea.get("element_type", "element"),
        type_b=eb.get("element_type", "element"),
        title_a=(ea.get("enriched_title") or ea.get("label") or ea.get("element_type", "")),
        title_b=(eb.get("enriched_title") or eb.get("label") or eb.get("element_type", "")),
        content_a=_elem_content(ea),
        content_b=_elem_content(eb),
        bridge_text=_bridge_text(pair),
        pair_type=pair.get("pair_type", "unknown"),
        n_variants=n_variants,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate retrieval-style M4 queries for embedding triplet training.")
    parser.add_argument("--materials", default=str(DEFAULT_MATERIALS))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--variants", type=int, default=3,
                       help="query variants per material")
    parser.add_argument("--api-url", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / f"m4_retrieval_queries_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.api_url and args.api_key:
        set_company_credentials(args.api_url, args.api_key)

    print(f"Output: {out_dir}")
    print(f"Model: {args.model} | Variants per material: {args.variants}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")

    materials = []
    with open(args.materials) as f:
        for line in f:
            if line.strip():
                materials.append(json.loads(line))
    selected = materials[args.skip:args.skip + args.limit]
    print(f"Loaded {len(materials)} materials, selected {len(selected)}")

    results = []
    stats = Counter()
    total_in, total_out = 0, 0

    for i, mat in enumerate(selected):
        mid = mat.get("material_id", f"mat_{i}")
        print(f"\n[{i+1}/{len(selected)}] {mid} ", end="", flush=True)

        prompt = build_retrieval_prompt(mat, args.variants)

        if args.dry_run:
            print(f"| prompt={len(prompt)} chars")
            results.append({"material_id": mid, "prompt": prompt})
            continue

        raw, gen_in, gen_out = call_llm(
            client=None,
            model=args.model,
            prompt=prompt,
            provider="company",
            system_prompt=(
                "You convert multimodal academic evidence into retrieval-style "
                "queries for embedding model training. Output valid JSON only."
            ),
            user_tag="m4_retrieval_v1",
            temperature=0.8,
        )
        total_in += gen_in
        total_out += gen_out
        stats["api_calls"] += 1

        parsed = parse_json(raw)
        if not parsed or "queries" not in parsed:
            stats["parse_failures"] += 1
            print("PARSE FAIL")
            results.append({
                "material_id": mid,
                "raw": raw[:500] if raw else "",
                "parse_failed": True,
            })
            continue

        queries = parsed.get("queries", [])
        meta = parsed.get("triplet_metadata", {})
        stats["generated"] += 1

        # Build triplet-ready records
        for qidx, query_text in enumerate(queries):
            record = {
                "query_id": f"{mid}_r{qidx}",
                "material_id": mid,
                "pair_id": mat.get("pair_id", ""),
                "query": query_text,
                "query_variant_index": qidx,
                "positive_element_a_id": mat.get("element_a", {}).get("element_id", ""),
                "positive_element_b_id": mat.get("element_b", {}).get("element_id", ""),
                "positive_pair_type": mat.get("pair_type", ""),
                "element_a_type": mat.get("element_a", {}).get("element_type", ""),
                "element_b_type": mat.get("element_b", {}).get("element_type", ""),
                "source_doc": mat.get("source_doc", ""),
                "target_doc": mat.get("target_doc", ""),
                "bridge_text": _bridge_text(mat),
                "difficulty": meta.get("retrieval_difficulty", "medium"),
                "domain_keywords": meta.get("domain_keywords", []),
                "requires_cross_doc": meta.get("requires_cross_doc", True),
            }
            results.append(record)

        n_queries = len(queries)
        stats[f"queries_generated"] += n_queries
        print(f"| {n_queries} variants")

        # Write incrementally
        with open(out_dir / "retrieval_queries.jsonl", "a") as f:
            for r in results[-n_queries:]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    summary = {
        "status": "ok",
        "mode": "dry_run" if args.dry_run else "live",
        "output_dir": str(out_dir),
        "source_materials": str(args.materials),
        "model": args.model,
        "materials_processed": len(selected),
        "api_calls": stats.get("api_calls", 0),
        "generated": stats.get("generated", 0),
        "parse_failures": stats.get("parse_failures", 0),
        "total_queries": stats.get("queries_generated", 0),
        "variants_per_material": args.variants,
        "tokens": {"in": total_in, "out": total_out},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Latest symlink
    latest = Path(args.out_root) / "m4_retrieval_queries_latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_dir, target_is_directory=True)

    nq = summary["total_queries"]
    nfail = summary["parse_failures"]
    print(f"\nDone. {nq} retrieval queries from {summary['generated']} materials "
          f"({nfail} parse failures)")
    print(f"Tokens: {total_in} in / {total_out} out")
    print(f"Latest: {latest}")
    print(f"Output: {out_dir}/retrieval_queries.jsonl")


if __name__ == "__main__":
    main()
