#!/usr/bin/env python3
"""Enrich multimodal elements with structured [T]/[M]/[C] descriptions.

Inspired by MoDora's CCTree enrichment approach: for each non-text element
(figure/table/formula), generate a structured description with three fields:
  [T] Title  — concise descriptive title
  [M] Metadata — keywords, element subtype, key variables/metrics
  [C] Content — comprehensive semantic description

This enrichment layer improves downstream query generation by providing
richer, more searchable text representations of multimodal elements.

Usage:
    python scripts/enrich_elements_modora.py \
        --input data/multimodal_elements.json \
        --output data/multimodal_elements_enriched.json \
        --provider anthropic \
        --model claude-sonnet-4-5-20250929 \
        --delay 0.3 \
        [--limit 10] \
        [--no-images] \
        [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.token_logger import log_run

# ── Truncation limits for prompt context (chars) ──
MAX_CAPTION_CHARS = 400
MAX_CONTENT_CHARS = 1200
MAX_CONTEXT_CHARS = 300
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB

# ──────────────────────────────────────────────────────────────
# Enrichment prompts per modality (inspired by MoDora [T]/[M]/[C])
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a scientific document analyst. "
    "Produce structured descriptions of document elements. "
    "Output valid JSON only, no markdown fences, no extra text."
)

PROMPT_FIGURE = """Analyze this figure from a scientific paper.

## Context
Caption: {caption}
Surrounding text: {context}

## Instructions
Produce a structured description with exactly three fields:
1. **title**: A concise descriptive title (5-15 words) that captures what this figure shows.
   Do NOT just repeat the caption. Synthesize the visual content and purpose.
2. **metadata**: A JSON object with:
   - "figure_type": one of ["line_plot", "bar_chart", "scatter_plot", "heatmap", "confusion_matrix", "architecture_diagram", "flow_chart", "model_diagram", "example_visualization", "distribution_plot", "comparison_plot", "other"]
   - "keywords": list of 3-7 keywords (methods, metrics, datasets visible)
   - "axes": brief description of what axes represent (if applicable, else null)
   - "num_series": number of data series/groups shown (if applicable, else null)
3. **content**: A comprehensive 2-4 sentence description of what the figure shows.
   Include: main trends/patterns, notable comparisons, key takeaways.
   Be specific about values and relationships visible in the figure.
   Do NOT use meta-language like "the figure shows" — describe the content directly.

## Output format (JSON only):
{{
  "title": "...",
  "metadata": {{
    "figure_type": "...",
    "keywords": ["...", "..."],
    "axes": "..." or null,
    "num_series": N or null
  }},
  "content": "..."
}}"""

PROMPT_TABLE = """Analyze this table from a scientific paper.

## Context
Caption: {caption}
Raw content (HTML/markdown): {raw_content}
Surrounding text: {context}

## Instructions
Produce a structured description with exactly three fields:
1. **title**: A concise descriptive title (5-15 words) summarizing what this table presents.
   Do NOT just repeat the caption.
2. **metadata**: A JSON object with:
   - "table_type": one of ["results_comparison", "ablation_study", "dataset_statistics", "hyperparameters", "feature_comparison", "summary_statistics", "configuration", "other"]
   - "keywords": list of 3-7 keywords (methods, metrics, datasets)
   - "columns": list of column header names
   - "num_rows": approximate number of data rows
   - "best_values": list of notable best/highlighted values if identifiable, else []
3. **content**: A comprehensive 2-4 sentence description of the table's content.
   Include: what is being compared, key findings, which method/config performs best.
   Reference specific values where possible.
   Do NOT use meta-language like "the table shows" — describe the content directly.

## Output format (JSON only):
{{
  "title": "...",
  "metadata": {{
    "table_type": "...",
    "keywords": ["...", "..."],
    "columns": ["...", "..."],
    "num_rows": N,
    "best_values": ["..."] or []
  }},
  "content": "..."
}}"""

PROMPT_FORMULA = """Analyze this mathematical formula/equation from a scientific paper.

## Context
Caption/label: {caption}
LaTeX source: {raw_content}
Surrounding text: {context}

## Instructions
Produce a structured description with exactly three fields:
1. **title**: A concise descriptive title (5-15 words) naming what this formula defines or computes.
2. **metadata**: A JSON object with:
   - "formula_type": one of ["loss_function", "objective", "constraint", "definition", "update_rule", "probability", "metric", "bound", "decomposition", "other"]
   - "keywords": list of 3-7 keywords (concepts, variables, methods)
   - "variables": dict mapping variable names to their roles/meanings (up to 8 main variables)
   - "domain": the mathematical domain (e.g., "optimization", "probability", "linear_algebra", "statistics")
3. **content**: A 2-4 sentence plain-English explanation of what this formula does.
   Include: what it computes, what each major term contributes, how it connects to the paper's method.
   Avoid simply restating the LaTeX. Explain the semantic meaning.

## Output format (JSON only):
{{
  "title": "...",
  "metadata": {{
    "formula_type": "...",
    "keywords": ["...", "..."],
    "variables": {{"x": "input feature", "y": "label"}},
    "domain": "..."
  }},
  "content": "..."
}}"""


# ──────────────────────────────────────────────────────────────
# Image utilities
# ──────────────────────────────────────────────────────────────

# Known path prefixes from different environments that should be
# re-rooted to PROJECT_ROOT when running elsewhere.
_KNOWN_PREFIXES = [
    "/home/d00855555/query_myx/data-process-test/",
    "/projects/_hdd/myyyx1/data-process-test/",
    "/projects/myyyx1/data-process-test/",
]


# 跨环境路径适配：集群绝对路径 → 本地相对路径，四级 fallback 策略
def resolve_image_path(raw_path: str) -> Optional[Path]:
    """Try to resolve an image path across different environments.

    Handles four cases:
    1. Path exists as-is (original environment) → use directly.
    2. Path starts with a known cluster prefix → strip prefix, re-root to
       PROJECT_ROOT.
    3. Path contains '/data/mineru_output/' → extract the suffix after this
       marker and resolve relative to PROJECT_ROOT/data/mineru_output/.
    4. Generic fallback: find 'data/' in the path and re-root to PROJECT_ROOT.
    """
    if not raw_path:
        return None

    p = Path(raw_path)
    if p.exists():
        return p

    # Normalise backslashes (Windows paths stored in JSON)
    normed = raw_path.replace("\\", "/")

    # Strategy 1: strip known cluster prefix → re-root to PROJECT_ROOT
    for prefix in _KNOWN_PREFIXES:
        if normed.startswith(prefix):
            relative = normed[len(prefix):]
            candidate = PROJECT_ROOT / relative
            if candidate.exists():
                return candidate

    # Strategy 2: extract suffix after '/data/mineru_output/'
    parts = normed.split("/data/mineru_output/")
    if len(parts) == 2:
        candidate = PROJECT_ROOT / "data" / "mineru_output" / parts[1]
        if candidate.exists():
            return candidate

    # Strategy 3: generic – find '/data/' and re-root everything after it
    idx = normed.find("/data/")
    if idx >= 0:
        relative = normed[idx + 1:]  # keep 'data/...'
        candidate = PROJECT_ROOT / relative
        if candidate.exists():
            return candidate

    # All strategies exhausted — log for debugging cross-environment issues
    print(f"  [resolve_image_path] MISS: {raw_path!r}", file=sys.stderr)
    return None


# 图片 → base64，跳过 >5MB 的超大图
def load_image_b64(image_path: str) -> Optional[Tuple[str, str]]:
    """Load image as base64 string. Returns (b64_data, mime_type) or None."""
    resolved = resolve_image_path(image_path)
    if resolved is None:
        return None

    mime, _ = mimetypes.guess_type(str(resolved))
    if not mime:
        mime = "image/jpeg"

    try:
        with open(resolved, "rb") as f:
            data = f.read()
        if len(data) > MAX_IMAGE_BYTES:
            return None
        return base64.b64encode(data).decode("ascii"), mime
    except (IOError, OSError):
        return None


# ──────────────────────────────────────────────────────────────
# API call (reuses patterns from generate_multihop_l1_queries.py)
# ──────────────────────────────────────────────────────────────

_COMPANY_API_URL: str = ""
_COMPANY_API_KEY: str = ""


# 逐行解析 SSE 流，累计 content 片段 + 末尾 usage token 统计
def _collect_company_stream(stream_generator) -> Tuple[str, int, int]:
    """Collect content and token usage from company API SSE stream."""
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
            if "usage" in parsed and parsed["usage"]:
                in_tok = parsed["usage"].get("prompt_tokens", 0)
                out_tok = parsed["usage"].get("completion_tokens", 0)
            choices = parsed.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if c := delta.get("content"):
                content_parts.append(c)
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    return "".join(content_parts), in_tok, out_tok


# 三路分发：anthropic / openai / company(yunwu.ai SSE)
def call_api(
    client: Any,
    model: str,
    prompt: str,
    image: Optional[Tuple[str, str]],
    provider: str = "anthropic",
) -> Tuple[Optional[str], int, int]:
    """Call provider API. Returns (text, input_tokens, output_tokens)."""
    if provider == "openai":
        content: List[Dict[str, Any]] = []
        if image is not None:
            b64, mime = image
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})

        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        msg = r.choices[0].message.content if r.choices else ""
        text = str(msg or "")
        in_tok = int(getattr(getattr(r, "usage", None), "prompt_tokens", 0) or 0)
        out_tok = int(getattr(getattr(r, "usage", None), "completion_tokens", 0) or 0)
        return text, in_tok, out_tok

    if provider == "company":
        from local_api_logger import wrap_requests_call

        user_content: List[Dict[str, Any]] = []
        if image is not None:
            b64, mime = image
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        user_content.append({"type": "text", "text": prompt})

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_COMPANY_API_KEY}",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 512,
            "temperature": 0.2,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        stream = wrap_requests_call(
            model=model,
            url=_COMPANY_API_URL,
            headers=headers,
            payload=payload,
            user="element_enrichment",
            verify=False,
        )
        text, in_tok, out_tok = _collect_company_stream(stream)
        return text, in_tok, out_tok

    # Default: anthropic
    content_aa: List[Dict[str, Any]] = []
    if image is not None:
        b64, mime = image
        content_aa.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": b64},
        })
    content_aa.append({"type": "text", "text": prompt})

    r = client.messages.create(
        model=model,
        system=SYSTEM_PROMPT,
        max_tokens=512,
        temperature=0.2,
        messages=[{"role": "user", "content": content_aa}],
    )
    return (
        r.content[0].text,
        r.usage.input_tokens,
        r.usage.output_tokens,
    )


# ──────────────────────────────────────────────────────────────
# JSON extraction
# ──────────────────────────────────────────────────────────────

# 从 LLM 返回的混杂文本中提取首个合法 JSON 对象（括号配对法）
def extract_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract first valid JSON object from text."""
    if not text:
        return None
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    # Find first { ... }
    for start_idx, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start_idx, len(text)):
            c = text[i]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start_idx:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


# ──────────────────────────────────────────────────────────────
# Prompt building
# ──────────────────────────────────────────────────────────────

# 按 figure/table/formula 三类选择对应 prompt 模板并填充上下文
def build_element_prompt(element: Dict[str, Any]) -> str:
    """Build the enrichment prompt for a single element."""
    etype = element["element_type"]
    caption = (element.get("caption", "") or "")[:MAX_CAPTION_CHARS]
    raw_content = (element.get("content", "") or "")[:MAX_CONTENT_CHARS]
    ctx_before = (element.get("context_before", "") or "")[:MAX_CONTEXT_CHARS]
    ctx_after = (element.get("context_after", "") or "")[:MAX_CONTEXT_CHARS]
    context = f"{ctx_before} ... {ctx_after}".strip() if (ctx_before or ctx_after) else "(no context)"

    if etype == "figure":
        return PROMPT_FIGURE.format(caption=caption, context=context)
    elif etype == "table":
        return PROMPT_TABLE.format(
            caption=caption,
            raw_content=raw_content[:800],
            context=context,
        )
    elif etype == "formula":
        return PROMPT_FORMULA.format(
            caption=caption,
            raw_content=raw_content,
            context=context,
        )
    else:
        return PROMPT_FIGURE.format(caption=caption, context=context)


# ──────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────

# 校验 [T]/[M]/[C] 三字段是否齐全且非空
def validate_enrichment(result: Dict[str, Any], etype: str) -> List[str]:
    """Validate enrichment result structure. Returns list of issues."""
    issues = []
    if not isinstance(result.get("title"), str) or len(result["title"]) < 3:
        issues.append("missing_or_short_title")
    if not isinstance(result.get("content"), str) or len(result["content"]) < 20:
        issues.append("missing_or_short_content")
    if not isinstance(result.get("metadata"), dict):
        issues.append("missing_metadata")
    else:
        meta = result["metadata"]
        if etype == "figure" and "figure_type" not in meta:
            issues.append("missing_figure_type")
        if etype == "table" and "table_type" not in meta:
            issues.append("missing_table_type")
        if etype == "formula" and "formula_type" not in meta:
            issues.append("missing_formula_type")
        if not isinstance(meta.get("keywords"), list) or len(meta["keywords"]) < 1:
            issues.append("missing_keywords")
    return issues


# ──────────────────────────────────────────────────────────────
# Main processing
# ──────────────────────────────────────────────────────────────

# 批量 enrichment 主循环：增量跳过 + API 调用 + 校验 + token 累计
def process_elements(
    mm_data: Dict[str, Any],
    client: Any,
    model: str,
    provider: str,
    delay: float,
    limit: int,
    no_images: bool,
    dry_run: bool,
    existing_enriched: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Process all elements and return enrichment results.

    Returns dict: element_id → {title, metadata, content, validation_issues}
    """
    results: Dict[str, Dict[str, Any]] = {}
    total_in_tok = 0
    total_out_tok = 0
    processed = 0
    skipped_existing = 0
    skipped_no_prompt = 0
    failed = 0

    # Collect all elements
    all_elements: List[Dict[str, Any]] = []
    for doc_key, doc in mm_data["documents"].items():
        elements = doc.get("elements", {})
        if isinstance(elements, dict):
            for el in elements.values():
                all_elements.append(el)
        else:
            for el in elements:
                all_elements.append(el)

    if limit > 0:
        all_elements = all_elements[:limit]

    total = len(all_elements)
    print(f"\nProcessing {total} elements...")

    for idx, el in enumerate(all_elements):
        eid = el["element_id"]
        etype = el["element_type"]

        # Skip if already enriched
        if existing_enriched and eid in existing_enriched:
            skipped_existing += 1
            results[eid] = existing_enriched[eid]
            continue

        # Skip non-enrichable types (sections etc.)
        if etype not in ("figure", "table", "formula"):
            skipped_no_prompt += 1
            continue

        prompt = build_element_prompt(el)

        if dry_run:
            print(f"  [{idx+1}/{total}] {eid} ({etype}) — DRY RUN")
            if idx < 2:
                print(f"    Prompt preview ({len(prompt)} chars):")
                print(f"    {prompt[:200]}...")
            results[eid] = {
                "title": f"[DRY RUN] {eid}",
                "metadata": {"keywords": []},
                "content": "[DRY RUN]",
                "validation_issues": [],
            }
            continue

        # Load image
        image = None
        if not no_images and etype == "figure":
            img_path = el.get("image_path", "") or ""
            if img_path:
                image = load_image_b64(img_path)

        try:
            text, in_tok, out_tok = call_api(client, model, prompt, image, provider)
            total_in_tok += in_tok
            total_out_tok += out_tok

            parsed = extract_json(text)
            if parsed is None:
                # Show first 200 chars of LLM output for debugging prompt/format issues
                snippet = (text or "")[:200].replace("\n", " ")
                print(f"  [{idx+1}/{total}] {eid} — PARSE FAIL | raw: {snippet!r}")
                failed += 1
                continue

            issues = validate_enrichment(parsed, etype)
            parsed["validation_issues"] = issues
            results[eid] = parsed
            processed += 1

            status = "PASS" if not issues else f"WARN({','.join(issues)})"
            print(f"  [{idx+1}/{total}] {eid} ({etype}) — {status}"
                  f"  [tok: {in_tok}+{out_tok}]")

        except Exception as exc:
            print(f"  [{idx+1}/{total}] {eid} — ERROR: {exc}")
            failed += 1

        if delay > 0:
            time.sleep(delay)

    print(f"\n=== Enrichment Summary ===")
    print(f"  Total elements: {total}")
    print(f"  Processed (API): {processed}")
    print(f"  Skipped (existing): {skipped_existing}")
    print(f"  Skipped (not enrichable): {skipped_no_prompt}")
    print(f"  Failed: {failed}")
    print(f"  Total tokens: {total_in_tok} in + {total_out_tok} out")
    if total_in_tok > 0:
        cost_est = (total_in_tok * 3 + total_out_tok * 15) / 1_000_000
        print(f"  Estimated cost: ${cost_est:.2f} (Sonnet pricing)")

    return results, total_in_tok, total_out_tok, processed, failed


# 将 enriched_title/metadata/content 回写到原始元素上（不覆盖原字段）
def merge_enrichments(
    mm_data: Dict[str, Any],
    enrichments: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge enrichment results back into multimodal_elements structure.

    Adds enriched_title, enriched_metadata, enriched_content fields
    to each element (without overwriting original fields).
    """
    enriched_data = json.loads(json.dumps(mm_data))  # deep copy

    merged_count = 0
    for doc_key, doc in enriched_data["documents"].items():
        elements = doc.get("elements", {})
        if isinstance(elements, dict):
            for eid, el in elements.items():
                if eid in enrichments:
                    enr = enrichments[eid]
                    el["enriched_title"] = enr.get("title", "")
                    el["enriched_metadata"] = enr.get("metadata", {})
                    el["enriched_content"] = enr.get("content", "")
                    el["enrichment_issues"] = enr.get("validation_issues", [])
                    merged_count += 1

    # Update top-level metadata
    enriched_data["metadata"]["enrichment"] = {
        "method": "modora_tmc",
        "total_enriched": merged_count,
        "total_elements": sum(
            doc["num_elements"] for doc in enriched_data["documents"].values()
        ),
    }

    print(f"  Merged enrichments for {merged_count} elements")
    return enriched_data


# 入口：加载 → enrichment → 合并 → 写出 → log_run（铁律）
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/multimodal_elements.json",
                    help="Input multimodal elements file")
    ap.add_argument("--output", default="data/multimodal_elements_enriched.json",
                    help="Output enriched elements file")
    ap.add_argument("--provider", choices=["anthropic", "openai", "company"],
                    default="company", help="API provider")
    ap.add_argument("--model", default=None,
                    help="Model name (default: auto per provider)")
    ap.add_argument("--delay", type=float, default=0.3,
                    help="Delay between API calls in seconds")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit number of elements to process (0=all)")
    ap.add_argument("--no-images", action="store_true",
                    help="Skip sending images to API (text-only enrichment)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build prompts but don't call API")
    ap.add_argument("--incremental", action="store_true",
                    help="Load existing output and only enrich missing elements")
    # Provider-specific args
    ap.add_argument("--company-api-key", default=None)
    ap.add_argument("--company-api-url", default=None)

    args = ap.parse_args()

    # Resolve model
    if args.model is None:
        if args.provider == "anthropic":
            args.model = "claude-sonnet-4-5-20250929"
        elif args.provider == "openai":
            args.model = "gpt-4o"
        else:
            args.model = "claude-sonnet-4-20250514"

    # Load input
    print("Loading data...")
    with open(args.input, encoding="utf-8") as f:
        mm_data = json.load(f)
    n_docs = len(mm_data["documents"])
    n_elements = sum(
        doc["num_elements"] for doc in mm_data["documents"].values()
    )
    print(f"  Documents: {n_docs}, Elements: {n_elements}")

    # Load existing enrichments for incremental mode
    existing_enriched: Optional[Dict[str, Dict]] = None
    if args.incremental and Path(args.output).exists():
        print(f"  Loading existing enrichments from {args.output}...")
        with open(args.output, encoding="utf-8") as f:
            existing_data = json.load(f)
        existing_enriched = {}
        for doc in existing_data["documents"].values():
            elements = doc.get("elements", {})
            if isinstance(elements, dict):
                for eid, el in elements.items():
                    if "enriched_title" in el:
                        existing_enriched[eid] = {
                            "title": el["enriched_title"],
                            "metadata": el.get("enriched_metadata", {}),
                            "content": el.get("enriched_content", ""),
                            "validation_issues": el.get("enrichment_issues", []),
                        }
        print(f"  Found {len(existing_enriched)} existing enrichments")

    # Initialize client
    global _COMPANY_API_URL, _COMPANY_API_KEY
    client = None

    if not args.dry_run:
        if args.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
        elif args.provider == "openai":
            from openai import OpenAI
            client = OpenAI()
        elif args.provider == "company":
            _COMPANY_API_KEY = (
                args.company_api_key
                or os.environ.get("COMPANY_API_KEY", "")
            )
            _COMPANY_API_URL = (
                args.company_api_url
                or os.environ.get("COMPANY_API_URL", "https://yunwu.ai/v1/chat/completions")
            )
            if not _COMPANY_API_KEY:
                print("ERROR: COMPANY_API_KEY not set", file=sys.stderr)
                sys.exit(1)

    # Process
    enrichments, total_in_tok, total_out_tok, processed, failed = process_elements(
        mm_data=mm_data,
        client=client,
        model=args.model,
        provider=args.provider,
        delay=args.delay,
        limit=args.limit,
        no_images=args.no_images,
        dry_run=args.dry_run,
        existing_enriched=existing_enriched,
    )

    # Merge and write
    enriched_data = merge_enrichments(mm_data, enrichments)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, indent=2, ensure_ascii=False)
    print(f"\nWritten to {args.output}")

    # Iron Rule: log token usage
    log_run(
        script="enrich_elements_modora",
        model=f"{args.provider}:{args.model}",
        purpose=f"MoDora [T]/[M]/[C] element enrichment — {processed} enriched, {failed} failed",
        input_tokens=total_in_tok,
        output_tokens=total_out_tok,
        extra={
            "pairs_processed": processed,
            "parse_failures": failed,
            "output": str(args.output),
        },
    )


if __name__ == "__main__":
    main()
