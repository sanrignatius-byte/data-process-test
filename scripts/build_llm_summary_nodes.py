#!/usr/bin/env python3
"""
build_llm_summary_nodes.py
===========================
Use LLM API to generate high-quality summary virtual nodes for sections
and chunks, then build ``summary_of`` edges into the document graph.

This replaces the heuristic "first N sentences" strategy in
``build_summary_nodes.py`` with actual LLM summarization, producing
richer, more semantically meaningful summary text that can significantly
improve downstream retrieval.

Pipeline:
  1. Read chunk_virtual_nodes.json (already built by build_chunk_virtual_nodes.py)
  2. For each doc (optionally filtered by --doc-ids / --doc-id-file):
     a. Summarize each section  → section_summary node
     b. Summarize each chunk    → chunk_summary node
  3. Build edges:  summary_of (summary → section/chunk)
  4. Save to data/01_graphs/llm_summary_virtual_nodes.json

Decoupled design:
  - This script ONLY adds summary nodes + summary_of edges.
  - Chunk nodes + containment edges are the responsibility of
    build_chunk_virtual_nodes.py.
  - Downstream scripts (eval_graph_retrieval, etc.) merge layers at
    read-time so each module stays independent.

Usage:
    # Summarize all 53 old docs
    python scripts/build_llm_summary_nodes.py \\
        --doc-id-file data/doc_lists/old_53_docs.txt

    # Dry run on 2 docs
    python scripts/build_llm_summary_nodes.py \\
        --doc-ids 1511.00830,1802.08139 --dry-run

    # Incremental resume (skip already-processed docs)
    python scripts/build_llm_summary_nodes.py \\
        --doc-id-file data/doc_lists/old_53_docs.txt --incremental
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.token_logger import log_run  # noqa: E402
from src.api import call_llm, parse_json, set_company_credentials  # noqa: E402

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_CHUNK_INPUT = "data/01_graphs/chunk_virtual_nodes.json"
DEFAULT_OUTPUT = "data/01_graphs/llm_summary_virtual_nodes.json"
DEFAULT_MAX_CHARS = 3000  # max chars of source text sent to LLM
DEFAULT_MAX_SUMMARY_WORDS = 60

# ── Prompts ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a scientific document analyst. "
    "Produce concise, information-dense summaries of document sections and text chunks. "
    "Output valid JSON only, no markdown fences, no extra text."
)

SECTION_SUMMARY_PROMPT = """Summarize this section from a scientific paper.

## Section metadata
Document ID: {doc_id}
Section title: {section_title}
Section level: {section_level}
Number of paragraphs: {num_paragraphs}
Total words: {total_words}

## Section text (may be truncated)
{section_text}

## Instructions
Produce a JSON with exactly these fields:
1. **summary**: A concise 2-4 sentence summary (max {max_words} words) capturing the core content.
   Focus on: key claims, methods, results, or definitions introduced.
   Avoid meta-language ("this section discusses..."); state the substance directly.
2. **keywords**: List of 4-8 domain-specific keywords/keyphrases from this section.
3. **section_type**: One of ["introduction", "related_work", "method", "architecture",
   "training", "experiment", "results", "ablation", "analysis", "discussion",
   "conclusion", "appendix", "background", "abstract", "other"]
4. **retrieval_hints**: List of 2-4 short phrases describing what kinds of queries
   this section can help answer (e.g., "comparison of model architectures",
   "training hyperparameters", "dataset statistics").

## Output format (JSON only):
{{
  "summary": "...",
  "keywords": ["...", "..."],
  "section_type": "...",
  "retrieval_hints": ["...", "..."]
}}"""

CHUNK_SUMMARY_PROMPT = """Summarize this text chunk from a scientific paper.

## Chunk metadata
Document ID: {doc_id}
Chunk index: {chunk_idx}
Section: {section_title}
Word count: {word_count}

## Chunk text
{chunk_text}

## Instructions
Produce a JSON with exactly these fields:
1. **summary**: A concise 1-3 sentence summary (max {max_words} words) capturing the key information.
   Be specific: include method names, metric values, dataset names when present.
   Avoid meta-language; state the substance directly.
2. **keywords**: List of 3-6 domain-specific keywords/keyphrases.
3. **retrieval_hints**: List of 1-3 short phrases describing what queries this chunk answers.

## Output format (JSON only):
{{
  "summary": "...",
  "keywords": ["...", "..."],
  "retrieval_hints": ["...", "..."]
}}"""


# ── LLM call wrapper ───────────────────────────────────────────────────────

def call_summary_api(
    client: Any,
    model: str,
    prompt: str,
    provider: str,
    max_tokens: int = 512,
) -> Tuple[Optional[str], int, int]:
    """Thin wrapper around src.api.call_llm for summary generation."""
    return call_llm(
        client, model, prompt,
        provider=provider,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=max_tokens,
        temperature=0.2,
        user_tag="llm_summary_nodes",
    )


def validate_section_summary(result: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    if not (result.get("summary") or "").strip():
        issues.append("missing_summary")
    if not isinstance(result.get("keywords"), list) or not result["keywords"]:
        issues.append("missing_keywords")
    if not result.get("section_type"):
        issues.append("missing_section_type")
    return issues


def validate_chunk_summary(result: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    if not (result.get("summary") or "").strip():
        issues.append("missing_summary")
    if not isinstance(result.get("keywords"), list) or not result["keywords"]:
        issues.append("missing_keywords")
    return issues


# ── Section text aggregation ────────────────────────────────────────────────

def aggregate_section_text(
    doc_data: Dict[str, Any],
    section_title: str,
    max_chars: int,
) -> Tuple[str, int, int]:
    """Collect paragraph/chunk texts belonging to a section, truncated to max_chars.

    Prefers paragraph_nodes when available; falls back to chunk nodes.
    Returns (text, num_paragraphs_or_chunks, total_words).
    """
    # ── Try paragraph_nodes first ──
    para_nodes = doc_data.get("paragraph_nodes", {}) or {}
    paras = []
    for para_id, para in para_nodes.items():
        if para.get("section_title") == section_title:
            paras.append(para)

    if paras:
        paras.sort(key=lambda p: p.get("para_idx", 0))
        total_words = sum(p.get("word_count", 0) for p in paras)
        texts = []
        char_count = 0
        for p in paras:
            t = p.get("text", "").strip()
            if char_count + len(t) > max_chars:
                remaining = max_chars - char_count
                if remaining > 100:
                    texts.append(t[:remaining] + "…")
                break
            texts.append(t)
            char_count += len(t)
        return "\n\n".join(texts), len(paras), total_words

    # ── Fallback: collect ALL chunks belonging to this section ──
    chunk_nodes = doc_data.get("nodes", {}) or {}
    matching_chunks = []
    for chunk_id, chunk in chunk_nodes.items():
        # A chunk may span multiple sections (section_titles list)
        chunk_sections = chunk.get("section_titles", [chunk.get("section_title", "")])
        if section_title in chunk_sections:
            matching_chunks.append(chunk)

    if not matching_chunks:
        return "", 0, 0

    matching_chunks.sort(key=lambda c: c.get("chunk_idx", 0))
    total_words = sum(c.get("word_count", 0) for c in matching_chunks)
    texts = []
    char_count = 0
    for c in matching_chunks:
        t = c.get("text", "").strip()
        if char_count + len(t) > max_chars:
            remaining = max_chars - char_count
            if remaining > 100:
                texts.append(t[:remaining] + "…")
            break
        texts.append(t)
        char_count += len(t)

    return "\n\n".join(texts), len(matching_chunks), total_words


# ── Core processing ────────────────────────────────────────────────────────

def process_doc(
    doc_id: str,
    doc_data: Dict[str, Any],
    client: Any,
    model: str,
    provider: str,
    *,
    max_chars: int,
    max_summary_words: int,
    delay: float,
    dry_run: bool,
) -> Tuple[Dict, Dict, List, int, int, int, int]:
    """Process one document: generate section + chunk summaries.

    Returns: (section_summaries, chunk_summaries, edges,
              total_in_tok, total_out_tok, processed, failed)
    """
    section_summaries: Dict[str, dict] = OrderedDict()
    chunk_summaries: Dict[str, dict] = OrderedDict()
    edges: List[dict] = []
    total_in_tok = 0
    total_out_tok = 0
    processed = 0
    failed = 0

    # ── Discover unique sections from paragraph nodes ──
    seen_sections: Dict[str, int] = OrderedDict()  # title → section_level
    para_nodes = doc_data.get("paragraph_nodes", {}) or {}
    for para_id, para in para_nodes.items():
        sec_title = para.get("section_title", "")
        if sec_title and sec_title not in seen_sections:
            seen_sections[sec_title] = para.get("section_level", 0)

    # If no paragraph_nodes, try chunks
    if not seen_sections:
        for chunk_id, chunk in (doc_data.get("nodes", {}) or {}).items():
            sec_title = chunk.get("section_title", "")
            if sec_title and sec_title not in seen_sections:
                seen_sections[sec_title] = chunk.get("section_level", 0)

    # ── Section summaries ──
    for sec_idx, (sec_title, sec_level) in enumerate(seen_sections.items()):
        section_text, num_paras, total_words = aggregate_section_text(
            doc_data, sec_title, max_chars
        )
        if not section_text.strip() or total_words < 20:
            continue

        prompt = SECTION_SUMMARY_PROMPT.format(
            doc_id=doc_id,
            section_title=sec_title,
            section_level=sec_level,
            num_paragraphs=num_paras,
            total_words=total_words,
            section_text=section_text,
            max_words=max_summary_words,
        )

        summary_id = f"{doc_id}_secsum_{sec_idx}"

        if dry_run:
            section_summaries[summary_id] = {
                "summary_id": summary_id,
                "doc_id": doc_id,
                "scope": "section",
                "source_section": sec_title,
                "method": "dry_run",
                "text": f"[DRY RUN] {sec_title}",
            }
            edges.append({
                "source": summary_id,
                "target": sec_title,
                "source_type": "summary",
                "target_type": "section",
                "relation": "summary_of",
            })
            processed += 1
            continue

        try:
            raw_text, in_tok, out_tok = call_summary_api(
                client, model, prompt, provider
            )
            total_in_tok += in_tok
            total_out_tok += out_tok
            parsed = parse_json(raw_text)

            if parsed is None:
                print(f"  [{doc_id}] section '{sec_title}' — PARSE FAIL")
                failed += 1
                if delay > 0:
                    time.sleep(delay)
                continue

            issues = validate_section_summary(parsed)
            summary_text = parsed.get("summary", "")

            section_summaries[summary_id] = {
                "summary_id": summary_id,
                "doc_id": doc_id,
                "scope": "section",
                "source_section": sec_title,
                "section_level": sec_level,
                "method": "llm",
                "text": summary_text,
                "keywords": parsed.get("keywords", []),
                "section_type": parsed.get("section_type", "other"),
                "retrieval_hints": parsed.get("retrieval_hints", []),
                "issues": issues,
            }
            edges.append({
                "source": summary_id,
                "target": sec_title,
                "source_type": "summary",
                "target_type": "section",
                "relation": "summary_of",
            })
            processed += 1

            status = "OK" if not issues else f"OK with issues: {issues}"
            print(f"  [{doc_id}] section '{sec_title}' — {status}")

        except Exception as exc:
            print(f"  [{doc_id}] section '{sec_title}' — ERROR: {exc}")
            failed += 1

        if delay > 0:
            time.sleep(delay)

    # ── Chunk summaries ──
    chunk_nodes = doc_data.get("nodes", {}) or {}
    for chunk_id, chunk in chunk_nodes.items():
        chunk_text = chunk.get("text", "").strip()
        if not chunk_text or chunk.get("word_count", 0) < 20:
            continue

        # Truncate for prompt
        if len(chunk_text) > max_chars:
            chunk_text = chunk_text[:max_chars] + "…"

        prompt = CHUNK_SUMMARY_PROMPT.format(
            doc_id=doc_id,
            chunk_idx=chunk.get("chunk_idx", "?"),
            section_title=chunk.get("section_title", ""),
            word_count=chunk.get("word_count", 0),
            chunk_text=chunk_text,
            max_words=max_summary_words,
        )

        summary_id = f"{chunk_id}_llmsum"

        if dry_run:
            chunk_summaries[summary_id] = {
                "summary_id": summary_id,
                "doc_id": doc_id,
                "scope": "chunk",
                "source_chunk": chunk_id,
                "method": "dry_run",
                "text": f"[DRY RUN] chunk {chunk.get('chunk_idx', '?')}",
            }
            edges.append({
                "source": summary_id,
                "target": chunk_id,
                "source_type": "summary",
                "target_type": "chunk",
                "relation": "summary_of",
            })
            processed += 1
            continue

        try:
            raw_text, in_tok, out_tok = call_summary_api(
                client, model, prompt, provider
            )
            total_in_tok += in_tok
            total_out_tok += out_tok
            parsed = parse_json(raw_text)

            if parsed is None:
                print(f"  [{doc_id}] {chunk_id} — PARSE FAIL")
                failed += 1
                if delay > 0:
                    time.sleep(delay)
                continue

            issues = validate_chunk_summary(parsed)
            summary_text = parsed.get("summary", "")

            chunk_summaries[summary_id] = {
                "summary_id": summary_id,
                "doc_id": doc_id,
                "scope": "chunk",
                "source_chunk": chunk_id,
                "section_title": chunk.get("section_title", ""),
                "method": "llm",
                "text": summary_text,
                "keywords": parsed.get("keywords", []),
                "retrieval_hints": parsed.get("retrieval_hints", []),
                "issues": issues,
            }
            edges.append({
                "source": summary_id,
                "target": chunk_id,
                "source_type": "summary",
                "target_type": "chunk",
                "relation": "summary_of",
            })
            processed += 1

        except Exception as exc:
            print(f"  [{doc_id}] {chunk_id} — ERROR: {exc}")
            failed += 1

        if delay > 0:
            time.sleep(delay)

    return (section_summaries, chunk_summaries, edges,
            total_in_tok, total_out_tok, processed, failed)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build LLM-generated summary virtual nodes for sections and chunks"
    )
    ap.add_argument("--input", default=DEFAULT_CHUNK_INPUT,
                    help=f"chunk_virtual_nodes.json path (default: {DEFAULT_CHUNK_INPUT})")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help=f"Output path (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--doc-ids", default=None,
                    help="Comma-separated doc IDs to process (default: all)")
    ap.add_argument("--doc-id-file", default=None,
                    help="File with one doc_id per line")
    ap.add_argument("--provider", choices=["anthropic", "openai", "company"],
                    default="company")
    ap.add_argument("--model", default=None,
                    help="Model name (default: auto per provider)")
    ap.add_argument("--company-api-url",
                    default=os.environ.get("COMPANY_API_URL", ""))
    ap.add_argument("--company-api-key",
                    default=os.environ.get("COMPANY_API_KEY", ""))
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                    help=f"Max chars of source text per prompt (default: {DEFAULT_MAX_CHARS})")
    ap.add_argument("--max-summary-words", type=int, default=DEFAULT_MAX_SUMMARY_WORDS,
                    help=f"Max words per summary (default: {DEFAULT_MAX_SUMMARY_WORDS})")
    ap.add_argument("--delay", type=float, default=0.3,
                    help="Delay between API calls in seconds (default: 0.3)")
    ap.add_argument("--limit-docs", type=int, default=0,
                    help="Process only first N docs (debug)")
    ap.add_argument("--flush-every", type=int, default=5,
                    help="Flush results to disk every N docs (default: 5)")
    ap.add_argument("--incremental", action="store_true",
                    help="Skip already-processed docs in the output file")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip API calls, produce placeholder summaries")
    args = ap.parse_args()

    # ── Model defaults ──
    if args.model is None:
        if args.provider == "anthropic":
            args.model = "claude-sonnet-4-5-20250929"
        elif args.provider == "openai":
            args.model = "gpt-4o"
        else:
            args.model = "gpt-5.4"

    # ── API client setup ──
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
            url = args.company_api_url
            key = args.company_api_key
            if not url or not key:
                raise SystemExit(
                    "COMPANY_API_URL / COMPANY_API_KEY not set. "
                    "Run: set -a && source .env && set +a"
                )
            set_company_credentials(url, key)

    # ── Load chunk virtual nodes ──
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}\n"
                         "  Run scripts/build_chunk_virtual_nodes.py first.")

    print(f"Loading chunk virtual nodes from {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        chunk_data = json.load(f)
    all_docs = chunk_data.get("documents", {})
    print(f"  Total docs in file: {len(all_docs)}")

    # ── Filter doc IDs ──
    target_doc_ids: Optional[List[str]] = None
    if args.doc_id_file:
        p = Path(args.doc_id_file)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        target_doc_ids = [
            line.strip() for line in p.read_text().strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        print(f"  Filtering to {len(target_doc_ids)} docs from {args.doc_id_file}")
    elif args.doc_ids:
        target_doc_ids = [d.strip() for d in args.doc_ids.split(",") if d.strip()]
        print(f"  Filtering to {len(target_doc_ids)} docs from --doc-ids")

    if target_doc_ids is not None:
        docs_to_process = OrderedDict(
            (did, all_docs[did]) for did in target_doc_ids if did in all_docs
        )
        missing = [did for did in target_doc_ids if did not in all_docs]
        if missing:
            print(f"  WARNING: {len(missing)} doc IDs not found in chunk data: {missing[:5]}...")
    else:
        docs_to_process = OrderedDict(all_docs)

    if args.limit_docs > 0:
        keys = list(docs_to_process.keys())[:args.limit_docs]
        docs_to_process = OrderedDict((k, docs_to_process[k]) for k in keys)
        print(f"  Limited to {len(docs_to_process)} docs (--limit-docs)")

    # ── Output path & incremental resume ──
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_results: Dict[str, Any] = {}
    done_doc_ids: set = set()
    cumulative_in_tok = 0
    cumulative_out_tok = 0

    if args.incremental and out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            for did, doc_result in existing.get("documents", {}).items():
                existing_results[did] = doc_result
                done_doc_ids.add(did)
            prev_meta = existing.get("metadata", {})
            cumulative_in_tok = int(prev_meta.get("total_input_tokens", 0))
            cumulative_out_tok = int(prev_meta.get("total_output_tokens", 0))
            print(f"  Incremental: loaded {len(done_doc_ids)} already-processed docs, resuming...")
        except Exception as exc:
            print(f"  WARNING: could not load existing output for incremental mode: {exc}")

    # ── Process ──
    print(f"\n{'='*60}")
    print(f"Building LLM summary nodes")
    print(f"  Provider: {args.provider}")
    print(f"  Model:    {args.model}")
    print(f"  Docs:     {len(docs_to_process)}")
    print(f"  Dry run:  {args.dry_run}")
    print(f"{'='*60}\n")

    grand_total_in_tok = cumulative_in_tok
    grand_total_out_tok = cumulative_out_tok
    grand_processed = 0
    grand_failed = 0
    results: Dict[str, Any] = dict(existing_results)
    doc_count = 0

    def _flush(reason: str = "") -> None:
        snapshot = _build_output(
            results, args, grand_total_in_tok, grand_total_out_tok,
            grand_processed, grand_failed, partial=True,
        )
        out_path.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if reason:
            print(f"  [flush] {reason} — {len(results)} docs saved")

    for doc_id, doc_data in docs_to_process.items():
        if doc_id in done_doc_ids:
            continue

        doc_count += 1
        num_chunks = doc_data.get("num_chunks", len(doc_data.get("nodes", {})))
        print(f"\n[{doc_count}/{len(docs_to_process)}] {doc_id} "
              f"({num_chunks} chunks)")

        sec_sums, chunk_sums, edges, in_tok, out_tok, proc, fail = process_doc(
            doc_id, doc_data, client, args.model, args.provider,
            max_chars=args.max_chars,
            max_summary_words=args.max_summary_words,
            delay=args.delay,
            dry_run=args.dry_run,
        )

        results[doc_id] = {
            "doc_id": doc_id,
            "section_summaries": sec_sums,
            "chunk_summaries": chunk_sums,
            "edges": edges,
            "num_section_summaries": len(sec_sums),
            "num_chunk_summaries": len(chunk_sums),
            "num_edges": len(edges),
        }

        grand_total_in_tok += in_tok
        grand_total_out_tok += out_tok
        grand_processed += proc
        grand_failed += fail

        print(f"  → {len(sec_sums)} section sums, {len(chunk_sums)} chunk sums, "
              f"{in_tok}+{out_tok} tokens, {fail} failures")

        # Periodic flush
        if args.flush_every > 0 and doc_count % args.flush_every == 0:
            _flush(f"after {doc_count} docs")

    # ── Final write ──
    output = _build_output(
        results, args, grand_total_in_tok, grand_total_out_tok,
        grand_processed, grand_failed, partial=False,
    )
    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Stats ──
    total_sec = sum(
        d.get("num_section_summaries", 0) for d in results.values()
    )
    total_chunk = sum(
        d.get("num_chunk_summaries", 0) for d in results.values()
    )
    total_edges = sum(d.get("num_edges", 0) for d in results.values())

    print(f"\n{'='*60}")
    print(f"── LLM Summary Nodes Complete ──")
    print(f"  Docs processed:        {len(results)}")
    print(f"  Section summaries:     {total_sec}")
    print(f"  Chunk summaries:       {total_chunk}")
    print(f"  summary_of edges:      {total_edges}")
    print(f"  Total input tokens:    {grand_total_in_tok:,}")
    print(f"  Total output tokens:   {grand_total_out_tok:,}")
    print(f"  Failures:              {grand_failed}")
    size_mb = out_path.stat().st_size / (1024 * 1024) if out_path.exists() else 0
    print(f"  Saved to {out_path} ({size_mb:.1f} MB)")
    print(f"{'='*60}")

    # ── Iron Rule 1: log token usage ──
    if not args.dry_run:
        log_run(
            script="build_llm_summary_nodes",
            model=f"{args.provider}:{args.model}",
            purpose=(
                f"LLM summary virtual nodes — "
                f"{total_sec} section sums + {total_chunk} chunk sums "
                f"across {len(results)} docs"
            ),
            input_tokens=grand_total_in_tok,
            output_tokens=grand_total_out_tok,
            extra={
                "docs_processed": len(results),
                "section_summaries": total_sec,
                "chunk_summaries": total_chunk,
                "failed": grand_failed,
                "output": str(out_path),
            },
        )


def _build_output(
    results: Dict[str, Any],
    args: argparse.Namespace,
    total_in_tok: int,
    total_out_tok: int,
    processed: int,
    failed: int,
    partial: bool,
) -> Dict[str, Any]:
    total_sec = sum(
        d.get("num_section_summaries", 0) for d in results.values()
    )
    total_chunk = sum(
        d.get("num_chunk_summaries", 0) for d in results.values()
    )
    total_edges = sum(d.get("num_edges", 0) for d in results.values())

    return {
        "metadata": {
            "source_file": args.input,
            "script": "scripts/build_llm_summary_nodes.py",
            "strategy": "llm",
            "model": f"{args.provider}:{args.model}",
            "max_chars": args.max_chars,
            "max_summary_words": args.max_summary_words,
            "node_types": ["summary"],
            "edge_types": ["summary_of"],
            "total_input_tokens": total_in_tok,
            "total_output_tokens": total_out_tok,
            "processed": processed,
            "failed": failed,
            "partial": partial,
        },
        "documents": results,
        "stats": {
            "total_docs": len(results),
            "total_section_summaries": total_sec,
            "total_chunk_summaries": total_chunk,
            "total_edges": total_edges,
        },
    }


if __name__ == "__main__":
    main()
