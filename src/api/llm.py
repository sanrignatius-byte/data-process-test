"""统一 LLM 调用层 —— 三条路全在这儿。

之前 4+ 个脚本各写各的 call_api / SSE 解析 / JSON 提取，重复代码满天飞。
现在全收进这一个文件，三条 provider 路径：
  - anthropic：直连 Claude API（默认）
  - openai：走 OpenAI SDK
  - company：走公司代理 yunwu.ai（SSE 流式 + local_api_logger 自动记账）

全局变量 _COMPANY_API_URL / _COMPANY_API_KEY 是故意设计的 ——
key 每月换一次，需要运行时注入，不要重构成 config 对象。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# ── Global company API config (set from CLI args in each script's main) ──────
_COMPANY_API_URL: str = ""
_COMPANY_API_KEY: str = ""


def set_company_credentials(url: str, key: str) -> None:
    """运行时设置公司 API 的 URL 和 Key —— 每个脚本的 main() 里调一次就行。"""
    global _COMPANY_API_URL, _COMPANY_API_KEY
    _COMPANY_API_URL = url
    _COMPANY_API_KEY = key


def get_company_credentials() -> Tuple[str, str]:
    """拿当前的 (url, key)，debug 用或者给子模块传参。"""
    return _COMPANY_API_URL, _COMPANY_API_KEY


# ── SSE stream collector ─────────────────────────────────────────────────────

def collect_company_stream(stream_generator) -> Tuple[str, int, int]:
    """从公司 SSE 流里一行一行拼出完整回复 + token 用量。

    公司 API 是 OpenAI 兼容的流式接口，每行 ``data: {...}`` 格式。
    我们逐行解析 delta.content 拼文本，最后一条带 usage 字段取 token 数。
    返回 ``(完整文本, input_tokens, output_tokens)``。
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
    """一把梭调 LLM —— 不管你用哪家，接口都一样。

    三条路径内部自动路由：
      - anthropic: client.messages.create()，图片用 base64 source
      - openai:    client.chat.completions.create()，图片用 image_url
      - company:   wrap_requests_call() + SSE 流，走公司代理

    返回 ``(生成的文本, input_tokens, output_tokens)``。
    images 参数传 ``[(base64_data, mime_type), ...]``，None 会被自动跳过。
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
            timeout=300,
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
    """从 LLM 的乱七八糟输出里抠出第一个合法 JSON。

    LLM 经常在 JSON 外面包一层 markdown 代码块，或者前后加废话。
    这里先剥掉 ```json 围栏，然后用花括号深度扫描找第一个平衡的 {...}。
    比 json.loads 直接上要鲁棒得多。
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
    """解析 LLM 输出的 JSON —— 先试快路径 json.loads，失败了走 extract_json 兜底。"""
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
