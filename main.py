#!/usr/bin/env python3
"""Company API demo & connectivity test.

Usage:
    # Set API key first
    export COMPANY_API_KEY="sk-your-key-here"

    # Run demo
    python main.py

    # Or specify key inline
    python main.py --api-key "sk-your-key-here"

    # Use a different model
    python main.py --model "claude-sonnet-4-20250514"
"""

from __future__ import annotations

import argparse
import json
import os
import time

from local_api_logger import wrap_requests_call, print_recent_calls


# ── Default config ──────────────────────────────────────────
DEFAULT_API_URL = "https://yunwu.ai/v1/chat/completions"
DEFAULT_MODEL = "claude-sonnet-4-20250514"


def collect_stream_data(stream_generator):
    """Collect reasoning and content from SSE stream generator."""
    reasoning_parts, content_parts = [], []
    usage_info = {}

    for line in stream_generator:
        line = line.strip() if isinstance(line, str) else line
        if not line or not line.startswith("data: "):
            continue

        data = line[6:].strip()
        if data == "[DONE]":
            continue

        try:
            parsed = json.loads(data)

            # Extract token usage from the final chunk
            if "usage" in parsed and parsed["usage"]:
                usage_info = parsed["usage"]

            if not parsed.get("choices") or len(parsed["choices"]) == 0:
                continue

            delta = parsed["choices"][0].get("delta", {})

            if reasoning := delta.get("reasoning_content"):
                reasoning_parts.append(reasoning)

            if content := delta.get("content"):
                content_parts.append(content)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Parse error: {e}, data: {data[:100]}")
            continue

    return "".join(reasoning_parts), "".join(content_parts), usage_info


def main():
    ap = argparse.ArgumentParser(description="Company API demo")
    ap.add_argument("--api-url", default=os.environ.get("COMPANY_API_URL", DEFAULT_API_URL))
    ap.add_argument("--api-key", default=os.environ.get("COMPANY_API_KEY", ""))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default="Count from 1 to 10", help="Test prompt")
    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: API key not set. Use --api-key or export COMPANY_API_KEY=...")
        return

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": True,
    }

    print(f"\nAPI URL: {args.api_url}")
    print(f"Model:   {args.model}")
    print(f"Prompt:  {args.prompt}")
    print("\nSending streaming request...")
    print("Note: wrapper auto-adds stream_options for token stats\n")

    try:
        stream = wrap_requests_call(
            model=args.model,
            url=args.api_url,
            headers=headers,
            payload=payload,
            user="demo",
            verify=False,
        )

        reasoning, content, usage = collect_stream_data(stream)

        print("-" * 80)
        print("\nStreaming response complete!")
        if reasoning:
            print(f"\n[Reasoning]\n{reasoning}")
        print(f"\n[Content]\n{content}")
        if usage:
            print(f"\n[Usage] input={usage.get('prompt_tokens', '?')}, "
                  f"output={usage.get('completion_tokens', '?')}, "
                  f"total={usage.get('total_tokens', '?')}")
        print("\nLog recorded automatically.")

    except Exception as e:
        print(f"Request failed: {e}")
        import traceback
        traceback.print_exc()

    # Show stats
    time.sleep(0.2)
    print("\n" + "=" * 80)
    print("Recent call log")
    print("=" * 80)
    print_recent_calls(model=args.model, limit=1)


if __name__ == "__main__":
    main()

    time.sleep(0.5)
    from local_api_logger import print_stats_summary
    print_stats_summary()
