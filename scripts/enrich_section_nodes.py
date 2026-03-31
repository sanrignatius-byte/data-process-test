#!/usr/bin/env python3
"""Enrich section/subsection nodes with structured semantic summaries.

Builds a section-level text corpus from LaTeX reference-graph metadata and
uses an LLM to produce structured summaries for section/subsection nodes.

This is intended for generic/universal query retrieval, where figure/table
element enrichment is not enough because many queries target section-level
content such as "introduction", "architecture", "summary", or "results".
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
from src.utils.token_logger import log_run


SYSTEM_PROMPT = (
    "You are a scientific document analyst. "
    "Summarize section-level nodes from academic papers into structured JSON. "
    "Output valid JSON only, no markdown fences, no extra text."
)

PROMPT_TEMPLATE = """Analyze this section-level node from a scientific paper.

## Node metadata
Document ID: {doc_id}
Node ID: {section_id}
Node type: {node_type}
Section level: {section_level}
Section title: {section_title}
Source file: {file_path}
Line range: {line_start}-{line_end}

## Local section text
{section_text}

## Instructions
Produce a structured description with exactly three fields:
1. **title**: A concise descriptive title (5-15 words) capturing the semantic role of this section.
   Do NOT just repeat the raw section title if it is too generic.
2. **metadata**: A JSON object with:
   - "section_type": one of ["introduction", "related_work", "method", "architecture", "training", "experiment", "results", "ablation", "analysis", "discussion", "conclusion", "appendix", "background", "other"]
   - "keywords": list of 4-8 keywords
   - "query_intents": list of 2-5 likely generic query intents this section can answer
   - "scope": brief phrase describing what evidence this section mostly contains
3. **content**: A comprehensive 2-4 sentence semantic summary.
   Include the core contribution of this section, the main claims or content, and what kinds of information it is useful for retrieving.
   Avoid meta-language like "this section says"; summarize the actual substance.

## Output format (JSON only)
{{
  "title": "...",
  "metadata": {{
    "section_type": "...",
    "keywords": ["...", "..."],
    "query_intents": ["...", "..."],
    "scope": "..."
  }},
  "content": "..."
}}"""


_COMPANY_API_URL: str = ""
_COMPANY_API_KEY: str = ""


def _collect_company_stream(stream_generator: Any) -> Tuple[str, int, int]:
    content_parts: List[str] = []
    in_tok, out_tok = 0, 0

    for line in stream_generator:
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        line = line.strip() if isinstance(line, str) else str(line).strip()
        if not line or not line.startswith("data: "):
            continue
        data = line[6:].strip()
        if data == "[DONE]":
            continue
        try:
            parsed = json.loads(data)
            if parsed.get("usage"):
                in_tok = int(parsed["usage"].get("prompt_tokens", 0) or 0)
                out_tok = int(parsed["usage"].get("completion_tokens", 0) or 0)
            choices = parsed.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {}) or {}
            chunk = delta.get("content")
            if isinstance(chunk, str):
                content_parts.append(chunk)
            elif isinstance(chunk, list):
                for part in chunk:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content_parts.append(str(part.get("text", "")))
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    return "".join(content_parts), in_tok, out_tok


def call_api(client: Any, model: str, prompt: str, provider: str) -> Tuple[Optional[str], int, int]:
    if provider == "openai":
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=768,
            temperature=0.2,
        )
        msg = r.choices[0].message.content if r.choices else ""
        text = str(msg or "")
        in_tok = int(getattr(getattr(r, "usage", None), "prompt_tokens", 0) or 0)
        out_tok = int(getattr(getattr(r, "usage", None), "completion_tokens", 0) or 0)
        return text, in_tok, out_tok

    if provider == "company":
        from local_api_logger import wrap_requests_call

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_COMPANY_API_KEY}",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 768,
            "temperature": 0.2,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        stream = wrap_requests_call(
            model=model,
            url=_COMPANY_API_URL,
            headers=headers,
            payload=payload,
            user="section_enrichment",
            verify=False,
        )
        return _collect_company_stream(stream)

    # anthropic
    r = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        max_tokens=768,
        temperature=0.2,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    text = r.content[0].text.strip() if r.content else ""
    in_tok = int(getattr(getattr(r, "usage", None), "input_tokens", 0) or 0)
    out_tok = int(getattr(getattr(r, "usage", None), "output_tokens", 0) or 0)
    return text, in_tok, out_tok


def extract_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def validate_enrichment(result: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    if not (result.get("title") or "").strip():
        issues.append("missing_title")
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        issues.append("missing_metadata")
        metadata = {}
    keywords = metadata.get("keywords")
    if not isinstance(keywords, list) or not keywords:
        issues.append("missing_keywords")
    intents = metadata.get("query_intents")
    if not isinstance(intents, list) or not intents:
        issues.append("missing_query_intents")
    if not (result.get("content") or "").strip():
        issues.append("missing_content")
    return issues


def build_section_text(section: Dict[str, Any], paragraphs: List[Dict[str, Any]], max_chars: int) -> str:
    start = int(section.get("line_no_start") or 0)
    end = int(section.get("line_no_end") or 0)
    path = str(section.get("file_path") or "")

    matches: List[str] = []
    for para in paragraphs:
        p_start = int(para.get("line_no_start") or 0)
        p_end = int(para.get("line_no_end") or 0)
        p_path = str(para.get("file_path") or "")
        if path and p_path and p_path != path:
            continue
        if start and end and p_start >= start and p_end <= end:
            snippet = (para.get("text_snippet") or "").strip()
            if snippet:
                matches.append(snippet)

    if not matches:
        return "(no paragraph snippets found inside section range)"

    joined = "\n\n".join(matches)
    return joined[:max_chars]


def load_sections(ref_graph_path: Path, levels: List[str], max_chars: int) -> List[Dict[str, Any]]:
    raw = json.loads(ref_graph_path.read_text(encoding="utf-8"))
    docs = raw.get("documents", raw) if isinstance(raw, dict) else raw

    rows: List[Dict[str, Any]] = []
    for doc_id, doc in docs.items():
        if not isinstance(doc, dict):
            continue
        metadata = doc.get("metadata", {}) or {}
        sections = metadata.get("sections", []) or []
        paragraphs = metadata.get("paragraphs", []) or []

        for idx, sec in enumerate(sections, start=1):
            command = str(sec.get("command", "section")).lower()
            node_type = command if command in {"section", "subsection", "subsubsection"} else "section"
            if node_type not in levels:
                continue
            section_text = build_section_text(sec, paragraphs, max_chars=max_chars)
            rows.append(
                {
                    "section_id": f"{doc_id}::sec::{idx:04d}",
                    "doc_id": doc_id,
                    "node_type": node_type,
                    "section_level": sec.get("level"),
                    "section_title": sec.get("title", ""),
                    "file_path": sec.get("file_path", ""),
                    "line_no_start": sec.get("line_no_start"),
                    "line_no_end": sec.get("line_no_end"),
                    "section_text": section_text,
                }
            )
    return rows


def build_prompt(row: Dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        doc_id=row["doc_id"],
        section_id=row["section_id"],
        node_type=row["node_type"],
        section_level=row.get("section_level"),
        section_title=row.get("section_title", ""),
        file_path=row.get("file_path", ""),
        line_start=row.get("line_no_start"),
        line_end=row.get("line_no_end"),
        section_text=row.get("section_text", ""),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich section/subsection nodes with semantic summaries")
    ap.add_argument("--reference-graph", default="data/latex_reference_graph.json")
    ap.add_argument("--output", default="data/section_nodes_enriched.json")
    ap.add_argument("--provider", choices=["anthropic", "openai", "company"], default="company")
    ap.add_argument("--model", default=None)
    ap.add_argument("--company-api-url", default=os.environ.get("COMPANY_API_URL", ""))
    ap.add_argument("--company-api-key", default=os.environ.get("COMPANY_API_KEY", ""))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--levels", default="section,subsection,subsubsection")
    ap.add_argument("--incremental", action="store_true",
                     help="Skip sections already present in the output file and append new results")
    ap.add_argument("--flush-every", type=int, default=10,
                     help="Flush partial results to disk every N sections (default: 10)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.model is None:
        if args.provider == "anthropic":
            args.model = "claude-sonnet-4-5-20250929"
        elif args.provider == "openai":
            args.model = "gpt-4o"
        else:
            args.model = "gpt-5.4"

    global _COMPANY_API_URL, _COMPANY_API_KEY
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
            _COMPANY_API_URL = args.company_api_url
            _COMPANY_API_KEY = args.company_api_key
            if not _COMPANY_API_URL or not _COMPANY_API_KEY:
                raise SystemExit("COMPANY_API_URL / COMPANY_API_KEY not set")

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    ref_graph_path = Path(args.reference_graph)
    if not ref_graph_path.is_absolute():
        ref_graph_path = PROJECT_ROOT / ref_graph_path
    rows = load_sections(ref_graph_path, levels=levels, max_chars=args.max_chars)
    if args.limit > 0:
        rows = rows[:args.limit]

    print(f"Loaded section nodes: {len(rows)}")

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- incremental: load existing results and skip already-processed sections ---
    total_in_tok = 0
    total_out_tok = 0
    processed = 0
    failed = 0
    results: List[Dict[str, Any]] = []
    done_ids: set = set()

    if args.incremental and out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            prev_sections = existing.get("sections", [])
            results.extend(prev_sections)
            done_ids = {s["section_id"] for s in prev_sections if "section_id" in s}
            prev_meta = existing.get("metadata", {})
            total_in_tok = int(prev_meta.get("input_tokens", 0))
            total_out_tok = int(prev_meta.get("output_tokens", 0))
            processed = len(prev_sections)
            print(f"Incremental: loaded {len(done_ids)} already-enriched sections, resuming...")
        except Exception as exc:
            print(f"WARNING: could not load existing output for incremental mode: {exc}")

    def _flush(reason: str = "") -> None:
        """Write partial results to disk so progress is never lost."""
        snapshot = {
            "metadata": {
                "source_reference_graph": str(ref_graph_path),
                "levels": levels,
                "model": f"{args.provider}:{args.model}",
                "total_requested": len(rows),
                "processed": processed,
                "failed": failed,
                "input_tokens": total_in_tok,
                "output_tokens": total_out_tok,
                "partial": True,
            },
            "sections": results,
        }
        out_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
        if reason:
            print(f"  [flush] {reason} — {len(results)} sections saved")

    new_count = 0
    for idx, row in enumerate(rows, start=1):
        if row["section_id"] in done_ids:
            continue

        prompt = build_prompt(row)

        if args.dry_run:
            print(f"[{idx}/{len(rows)}] {row['section_id']} ({row['node_type']}) — DRY RUN")
            if idx <= 2:
                print(prompt[:500] + "...")
            results.append(
                {
                    **row,
                    "enriched_title": f"[DRY RUN] {row['section_title']}",
                    "enriched_metadata": {"keywords": [], "query_intents": []},
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
            parsed = extract_json(text)
            if parsed is None:
                print(f"[{idx}/{len(rows)}] {row['section_id']} — PARSE FAIL")
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
                print(f"[{idx}/{len(rows)}] {row['section_id']} — OK with issues: {issues}")
            else:
                print(f"[{idx}/{len(rows)}] {row['section_id']} — OK")
        except Exception as exc:
            print(f"[{idx}/{len(rows)}] {row['section_id']} — ERROR: {exc}")
            failed += 1
            new_count += 1

        # periodic flush to disk
        if args.flush_every > 0 and new_count % args.flush_every == 0:
            _flush(f"after {new_count} new items")

        if args.delay > 0 and idx < len(rows):
            time.sleep(args.delay)

    # --- final write (mark partial=False) ---
    output = {
        "metadata": {
            "source_reference_graph": str(ref_graph_path),
            "levels": levels,
            "model": f"{args.provider}:{args.model}",
            "total_requested": len(rows),
            "processed": processed,
            "failed": failed,
            "input_tokens": total_in_tok,
            "output_tokens": total_out_tok,
        },
        "sections": results,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote section enrichments: {out_path} ({new_count} new this run)")

    if not args.dry_run:
        log_run(
            script="enrich_section_nodes",
            model=f"{args.provider}:{args.model}",
            purpose=f"Section-level enrichment — {processed} enriched, {failed} failed",
            input_tokens=total_in_tok,
            output_tokens=total_out_tok,
            extra={
                "levels": levels,
                "processed": processed,
                "failed": failed,
                "output": str(out_path),
            },
        )


if __name__ == "__main__":
    main()
