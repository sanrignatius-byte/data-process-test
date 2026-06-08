#!/usr/bin/env python3
"""
Lightweight enrichment for Huawei product doc graph.

Two enrichment passes:
  Phase A — Section summaries: generates 2-3 sentence summaries for sections
  Phase B — Table enrichment: generates [T]/[M]/[C] descriptions for tables

Output format compatible with pair injection (inject_pair_enrichments.py)
and downstream query generation.

Usage:
  python scripts/enrich_huawei_graph.py \
    --graph data/01_graphs/huawei_multimodal_elements.json \
    --output data/02_enriched/huawei_enriched.json \
    --phase section,table \
    --limit 20
"""

import argparse, json, os, re, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai

ROOT = Path(__file__).resolve().parents[1]

# ── Phase A: Section summary prompt ──
SECTION_SYSTEM = "You are a technical documentation analyst. Summarize sections concisely."

SECTION_PROMPT = """Summarize this section from a Huawei product/patent document.

## Section Title
{section_label}

## Section Content (first 1500 chars)
{section_content}

## Instructions
Write a concise 2-3 sentence summary covering:
1. What this section is about (main topic)
2. Key technical details or steps mentioned
3. The role this section plays in the document

Output JSON only:
{{"summary": "2-3 sentence summary"}}"""

# ── Phase B: Table enrichment prompt ──
TABLE_SYSTEM = "You are a technical documentation analyst. Describe tables concisely."

TABLE_PROMPT = """Analyze this table from a Huawei product/patent document.

## Table Caption
{table_caption}

## Table Content (first 1500 chars)
{table_content}

## Instructions
Produce exactly three fields:
1. **title**: Concise title (5-15 words) capturing what the table shows
2. **metadata**: JSON with "table_type" and "keywords" (3-7 keywords)
3. **content**: 2-3 sentence description of the table's content and key values

Output JSON only:
{{"title": "...", "metadata": {{"table_type": "config|comparison|specs|data|overview|other", "keywords": ["k1","k2","k3"]}}, "content": "2-3 sentence description"}}"""


def call_api(
    client: openai.OpenAI,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int = 500,
) -> Tuple[Optional[str], int, int]:
    """Call API and return (text, in_tokens, out_tokens)."""
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        text = r.choices[0].message.content or ""
        in_tok = r.usage.prompt_tokens if r.usage else 0
        out_tok = r.usage.completion_tokens if r.usage else 0
        return text, in_tok, out_tok
    except Exception as e:
        return None, 0, 0


def parse_json_response(text: str) -> Optional[dict]:
    """Parse JSON from LLM response."""
    text = (text or "").strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def enrich_sections(
    graph: dict,
    client: openai.OpenAI,
    model: str,
    limit: int = 0,
    delay: float = 0.3,
    dry_run: bool = False,
) -> dict[str, dict]:
    """Enrich section elements with 2-3 sentence summaries."""
    sections = {}
    for did, doc in graph.get("documents", {}).items():
        for eid, el in doc.get("elements", {}).items():
            if el.get("element_type") == "section":
                sections[eid] = {**el, "doc_id": did}

    total = len(sections)
    if limit:
        sections = dict(list(sections.items())[:limit])
        total = len(sections)

    print(f"  Phase A (section summaries): {total} sections")
    if dry_run:
        print(f"    [DRY RUN] Would enrich {total} sections")
        return {}

    enriched = {}
    ok = fail = 0
    total_in = total_out = 0

    for i, (eid, el) in enumerate(sections.items()):
        label = (el.get("label") or el.get("content") or "")[:200]
        content = (el.get("content") or "")[:1500]
        if len(content) < 30:
            continue

        prompt = SECTION_PROMPT.format(section_label=label, section_content=content)

        if dry_run:
            continue

        text, in_tok, out_tok = call_api(client, model, SECTION_SYSTEM, prompt)
        total_in += in_tok
        total_out += out_tok

        obj = parse_json_response(text)
        if obj and obj.get("summary"):
            enriched[eid] = {
                "element_id": eid,
                "enriched_content": obj["summary"],
                "doc_id": el.get("doc_id", ""),
            }
            ok += 1
            if ok % 5 == 0:
                print(f"    [{i+1}/{total}] sections enriched, {ok} ok")
        else:
            fail += 1

        time.sleep(delay)

    print(f"    Sections: {ok} ok, {fail} fail, {total_in} in / {total_out} out tokens")
    return enriched


def enrich_tables(
    graph: dict,
    client: openai.OpenAI,
    model: str,
    limit: int = 0,
    delay: float = 0.3,
    dry_run: bool = False,
) -> dict[str, dict]:
    """Enrich table elements with [T]/[M]/[C]."""
    tables = {}
    for did, doc in graph.get("documents", {}).items():
        for eid, el in doc.get("elements", {}).items():
            if el.get("element_type") == "table":
                tables[eid] = {**el, "doc_id": did}

    total = len(tables)
    if limit:
        tables = dict(list(tables.items())[:limit])
        total = len(tables)

    print(f"  Phase B (table enrichment): {total} tables")
    if dry_run:
        print(f"    [DRY RUN] Would enrich {total} tables")
        return {}

    enriched = {}
    ok = fail = 0
    total_in = total_out = 0

    for i, (eid, el) in enumerate(tables.items()):
        caption = (el.get("caption") or el.get("label") or "")[:300]
        content = (el.get("content") or "")[:1500]
        if len(content) < 50:
            continue

        prompt = TABLE_PROMPT.format(table_caption=caption, table_content=content)

        text, in_tok, out_tok = call_api(client, model, TABLE_SYSTEM, prompt, max_tokens=600)
        total_in += in_tok
        total_out += out_tok

        obj = parse_json_response(text)
        if obj and (obj.get("title") or obj.get("content")):
            enriched[eid] = {
                "element_id": eid,
                "enriched_title": obj.get("title", ""),
                "enriched_metadata": obj.get("metadata", {}),
                "enriched_content": obj.get("content", ""),
                "doc_id": el.get("doc_id", ""),
            }
            ok += 1
        else:
            fail += 1
            if fail <= 3:
                print(f"    [fail] {eid[:50]}: {text[:80]}")

        if ok % 5 == 0 and ok > 0:
            print(f"    [{ok}/{total}] tables enriched")

        time.sleep(delay)

    print(f"    Tables: {ok} ok, {fail} fail, {total_in} in / {total_out} out tokens")
    return enriched


def save_enriched(sections: dict, tables: dict, output_path: Path, graph_path: str):
    """Save enriched data in overlay format compatible with inject_pair_enrichments."""
    enriched_elements = {}
    for eid, data in sections.items():
        enriched_elements[eid] = {
            "element_id": eid,
            "enriched_content": data.get("enriched_content", ""),
        }
    for eid, data in tables.items():
        if eid in enriched_elements:
            enriched_elements[eid].update({
                "enriched_title": data.get("enriched_title", ""),
                "enriched_metadata": data.get("enriched_metadata", {}),
            })
            enriched_elements[eid]["enriched_content"] = (
                enriched_elements[eid].get("enriched_content", "") + "; " +
                data.get("enriched_content", "")
            ).strip("; ")
        else:
            enriched_elements[eid] = {
                "element_id": eid,
                "enriched_title": data.get("enriched_title", ""),
                "enriched_metadata": data.get("enriched_metadata", {}),
                "enriched_content": data.get("enriched_content", ""),
            }

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_graph": graph_path,
        "sections_enriched": len(sections),
        "tables_enriched": len(tables),
        "total_enriched_elements": len(enriched_elements),
        "enriched_elements": enriched_elements,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n  Enriched output: {output_path}")
    print(f"  Sections enriched: {len(sections)}")
    print(f"  Tables enriched: {len(tables)}")


def main():
    ap = argparse.ArgumentParser(description="Enrich Huawei product doc graph elements")
    ap.add_argument("--graph", default="data/01_graphs/huawei_multimodal_elements.json")
    ap.add_argument("--output", default="data/02_enriched/huawei_enriched.json")
    ap.add_argument("--phase", default="section,table", help="Comma-separated: section,table")
    ap.add_argument("--limit", type=int, default=0, help="Limit elements per phase")
    ap.add_argument("--delay", type=float, default=0.3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    phases = [p.strip() for p in args.phase.split(",")]

    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))

    api_url = os.environ.get("COMPANY_API_URL", "")
    api_key = os.environ.get("COMPANY_API_KEY", "")
    if not args.dry_run and (not api_url or not api_key):
        print("ERROR: COMPANY_API_URL and COMPANY_API_KEY must be set in environment")
        sys.exit(1)

    client = openai.OpenAI(
        base_url=api_url.rsplit("/v1", 1)[0] + "/v1",
        api_key=api_key,
    ) if not args.dry_run else None
    model = "gpt-5.4"

    sections_enriched = {}
    tables_enriched = {}

    if "section" in phases:
        print("\n" + "=" * 50)
        print("  PHASE A: Section Summaries")
        print("=" * 50)
        sections_enriched = enrich_sections(
            graph, client, model, limit=args.limit, delay=args.delay, dry_run=args.dry_run,
        )

    if "table" in phases:
        print("\n" + "=" * 50)
        print("  PHASE B: Table [T]/[M]/[C] Enrichment")
        print("=" * 50)
        tables_enriched = enrich_tables(
            graph, client, model, limit=args.limit, delay=args.delay, dry_run=args.dry_run,
        )

    if not args.dry_run:
        save_enriched(sections_enriched, tables_enriched, Path(args.output), args.graph)

    print("\nDone.")


if __name__ == "__main__":
    main()
