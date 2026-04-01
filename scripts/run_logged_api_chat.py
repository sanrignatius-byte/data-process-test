#!/usr/bin/env python3
"""Run company chat-completions API with local_api_logger integration.

This script is intended as the project entrypoint for API-based query generation
experiments with auditable token/cost logs.

Prerequisite:
  - `local_api_logger/` package exists in project root (as you mentioned).

Examples:
  # Stream mode (default)
  python scripts/run_logged_api_chat.py \
    --api-url https://yunwu.ai/v1/chat/completions \
    --model claude-sonnet-4-20250514 \
    --prompt "数从1到10" \
    --user data \
    --api-key "$YUNWU_API_KEY" \
    --insecure

  # Non-stream mode
  python scripts/run_logged_api_chat.py \
    --api-url https://yunwu.ai/v1/chat/completions \
    --model claude-sonnet-4-20250514 \
    --prompt "总结这篇论文的方法" \
    --no-stream
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Iterable, Tuple


def collect_stream_data(stream_generator: Iterable[str], verbose_errors: bool = False) -> Tuple[str, str]:
    """Collect reasoning/content from SSE lines emitted by wrap_requests_call."""
    reasoning_parts = []
    content_parts = []

    for raw in stream_generator:
        line = (raw or "").strip()
        if not line or not line.startswith("data: "):
            continue

        data = line[6:].strip()
        if data == "[DONE]":
            continue

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            if verbose_errors:
                print(f"[WARN] JSON parse failed: {exc}; sample={data[:120]}", file=sys.stderr)
            continue

        choices = parsed.get("choices") or []
        if not choices:
            continue

        delta = (choices[0] or {}).get("delta") or {}

        reasoning = delta.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            reasoning_parts.append(reasoning)

        content = delta.get("content")
        if isinstance(content, str) and content:
            content_parts.append(content)

    return "".join(reasoning_parts), "".join(content_parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run chat completion with local_api_logger")
    p.add_argument("--api-url", default="https://yunwu.ai/v1/chat/completions")
    p.add_argument("--model", default="claude-sonnet-4-20250514")
    p.add_argument("--prompt", required=True)
    p.add_argument("--user", default="query_gen")
    p.add_argument("--api-key", default=os.environ.get("YUNWU_API_KEY", ""))
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--stream", action="store_true", default=True)
    p.add_argument("--no-stream", dest="stream", action="store_false")
    p.add_argument("--insecure", action="store_true", help="Disable SSL verify")
    p.add_argument("--verbose-errors", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.api_key:
        print("[ERROR] Missing API key. Use --api-key or export YUNWU_API_KEY.", file=sys.stderr)
        return 2

    try:
        from local_api_logger import print_recent_calls, print_stats_summary, wrap_requests_call
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to import local_api_logger: {exc}", file=sys.stderr)
        print("        Ensure local_api_logger/ exists in project root.", file=sys.stderr)
        return 2

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": args.stream,
    }

    print("\nSending request with local_api_logger wrapper...")
    print("Note: wrapper auto-adds stream_options for usage stats when needed.\n")

    try:
        wrapped = wrap_requests_call(
            model=args.model,
            url=args.api_url,
            headers=headers,
            payload=payload,
            user=args.user,
            verify=not args.insecure,
        )

        if args.stream:
            reasoning, content = collect_stream_data(wrapped, verbose_errors=args.verbose_errors)
            print("-" * 80)
            print("✓ Streaming response finished")
            if reasoning:
                print("\n[Reasoning]\n" + reasoning)
            print("\n[Content]\n" + content)
        else:
            # Non-stream mode: wrapper returns response-like JSON object in current SDK setup.
            print("-" * 80)
            print("✓ Non-stream response finished")
            print(json.dumps(wrapped, ensure_ascii=False, indent=2)[:4000])

        print("✓ Call logged automatically with token stats")

    except Exception as exc:  # noqa: BLE001
        print(f"✗ Request failed: {exc}", file=sys.stderr)
        return 1

    time.sleep(0.2)
    print("\n" + "=" * 80)
    print("Recent call")
    print("=" * 80)
    print_recent_calls(model=args.model, limit=1)

    time.sleep(0.2)
    print("\n" + "=" * 80)
    print("Stats summary")
    print("=" * 80)
    print_stats_summary(model=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
