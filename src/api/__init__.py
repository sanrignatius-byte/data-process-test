"""Unified LLM API client + JSON extraction.

Consolidates call_api / _collect_company_stream / parse_json previously
duplicated across 4+ scripts.  Preserves the global mutable
``_COMPANY_API_URL`` / ``_COMPANY_API_KEY`` injection pattern (user's key
changes monthly and needs runtime injection).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# ── Global company API config (set from CLI args in each script's main) ──────
_COMPANY_API_URL: str = ""
_COMPANY_API_KEY: str = ""


def set_company_credentials(url: str, key: str) -> None:
    """Set the company API URL and key at runtime."""
    global _COMPANY_API_URL, _COMPANY_API_KEY
    _COMPANY_API_URL = url
    _COMPANY_API_KEY = key


def get_company_credentials() -> Tuple[str, str]:
    """Return the current ``(url, key)`` pair."""
    return _COMPANY_API_URL, _COMPANY_API_KEY


# ── SSE stream collector ─────────────────────────────────────────────────────

def collect_company_stream(stream_generator) -> Tuple[str, int, int]:
    """Collect content and token usage from a company API SSE stream.

    Returns ``(text, input_tokens, output_tokens)``.
    """
    content_parts: List[str] = []
    in_tok, out_tok = 0, 0
    line_count = 0
    debug_lines: List[str] = []

    for line in stream_generator:
        line_count += 1
        raw_repr = repr(line)[:200] if line_count <= 5 else None
        if raw_repr:
            debug_lines.append(raw_repr)

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

    text = "".join(content_parts)
    if not text and debug_lines:
        print(f"\n  [DEBUG] stream had {line_count} lines, first 5:")
        for dl in debug_lines:
            print(f"    {dl}")
    if in_tok == 0 and out_tok == 0 and content_parts:
        print(
            "  [WARN] No token usage found in stream response. "
            "Ensure stream_options.include_usage is enabled."
        )
    return text, in_tok, out_tok


# ── Unified LLM call ─────────────────────────────────────────────────────────

def call_llm(
    client: Any,
    model: str,
    prompt: str,
    *,
    images: Optional[List[Optional[Tuple[str, str]]]] = None,
    provider: str = "anthropic",
    system_prompt: str = "",
    max_tokens: int = 1536,
    temperature: float = 0.4,
    user_tag: str = "pipeline",
) -> Tuple[Optional[str], int, int]:
    """Call an LLM provider.  Returns ``(text, input_tokens, output_tokens)``.

    Parameters
    ----------
    client
        An initialized API client (``anthropic.Anthropic``, ``openai.OpenAI``,
        or *None* when provider is ``"company"``).
    model
        Model name (e.g. ``"claude-sonnet-4-5-20250929"``).
    prompt
        User-message text.
    images
        Optional list of ``(base64_data, mime_type)`` tuples.  ``None`` entries
        are silently skipped.
    provider
        One of ``"anthropic"`` | ``"openai"`` | ``"company"``.
    system_prompt
        System message prepended to the conversation.
    max_tokens
        Maximum generation tokens.
    temperature
        Sampling temperature.
    user_tag
        Identifier for the ``local_api_logger`` ``user`` field.
    """
    imgs = [i for i in (images or []) if i is not None]

    # ── OpenAI path ──────────────────────────────────────────────────────────
    if provider == "openai":
        content: List[Dict[str, Any]] = []
        for b64, mime in imgs:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        content.append({"type": "text", "text": prompt})

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        r = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        msg = r.choices[0].message.content if r.choices else ""
        if isinstance(msg, list):
            text = "".join(
                part.get("text", "") for part in msg if isinstance(part, dict)
            )
        else:
            text = str(msg or "")
        in_tok = int(getattr(getattr(r, "usage", None), "prompt_tokens", 0) or 0)
        out_tok = int(getattr(getattr(r, "usage", None), "completion_tokens", 0) or 0)
        return text, in_tok, out_tok

    # ── Company path (yunwu.ai / OpenAI-compatible SSE) ──────────────────────
    if provider == "company":
        from local_api_logger import wrap_requests_call

        user_content: List[Dict[str, Any]] = []
        for b64, mime in imgs:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        user_content.append({"type": "text", "text": prompt})

        messages_c: List[Dict[str, Any]] = []
        if system_prompt:
            messages_c.append({"role": "system", "content": system_prompt})
        messages_c.append({"role": "user", "content": user_content})

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_COMPANY_API_KEY}",
        }
        payload = {
            "model": model,
            "messages": messages_c,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        stream = wrap_requests_call(
            model=model,
            url=_COMPANY_API_URL,
            headers=headers,
            payload=payload,
            user=user_tag,
            verify=False,
        )
        text, in_tok, out_tok = collect_company_stream(stream)
        return text, in_tok, out_tok

    # ── Anthropic path (default) ─────────────────────────────────────────────
    content_aa: List[Dict[str, Any]] = []
    for b64, mime in imgs:
        content_aa.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": b64},
        })
    content_aa.append({"type": "text", "text": prompt})

    r = client.messages.create(
        model=model,
        system=system_prompt or "",
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": content_aa}],
    )
    return (
        r.content[0].text,
        r.usage.input_tokens,
        r.usage.output_tokens,
    )


# ── JSON extraction from LLM output ──────────────────────────────────────────

def extract_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Extract first valid JSON object from (possibly noisy) LLM output.

    Strips markdown fences, then uses a brace-depth scanner to find the first
    balanced ``{ … }`` substring and parses it.
    """
    if not text:
        return None
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)

    for start_idx, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start_idx, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start_idx:i + 1])
                    except json.JSONDecodeError:
                        break
        break  # first candidate failed — give up
    return None


def parse_json(txt: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM output — tries ``json.loads`` first, falls back to extraction."""
    if not txt:
        return None
    cleaned = txt.strip()
    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return extract_json(txt)
