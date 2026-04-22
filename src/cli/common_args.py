"""LLM 脚本通用 CLI 参数 —— 消除 10+ 个 ``generate_*``/``enrich_*`` 脚本里重复的 argparse。

典型用法::

    import argparse
    from src.cli import add_llm_args, build_llm_client
    from src.api import set_company_credentials

    ap = argparse.ArgumentParser()
    add_llm_args(ap)
    ap.add_argument("--candidates", required=True)  # 脚本自己的参数
    args = ap.parse_args()

    client = build_llm_client(args)  # 返回 provider 对应的 client，或 None (company)

**铁律**：company provider 的实际 HTTP 调用在 :mod:`src.api.llm` 里走
``local_api_logger.wrap_requests_call``。本模块不直接发起请求。
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Optional


def add_llm_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """在现有 parser 上注册一组统一的 LLM 相关 CLI 参数。

    注册的参数：
      --provider {anthropic,openai,company}  默认 company
      --model MODEL                          默认 claude-sonnet-4-5-20250929
      --api-key KEY                          默认读 ANTHROPIC_API_KEY / OPENAI_API_KEY
      --company-api-url URL                  默认读 COMPANY_API_URL 环境变量
      --company-api-key KEY                  默认读 COMPANY_API_KEY 环境变量
      --delay SECONDS                        每次调用之间的 sleep，默认 0.3
      --limit N                              处理前 N 条 candidate（调试用）
      --dry-run                              不调 API，只打 prompt

    **凭据优先级**：CLI 参数 > 环境变量。**安全提醒**：CLI 传 key 会出现在
    ``ps`` / shell history 里，生产环境优先用 ``.env`` + 环境变量（参考
    CLAUDE.md 铁律 3）。

    返回同一个 parser 以支持链式。
    """
    parser.add_argument(
        "--provider",
        default="company",
        choices=["anthropic", "openai", "company"],
        help="LLM provider (default: company → yunwu.ai via local_api_logger)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-5-20250929",
        help="Model ID (provider-specific)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for anthropic/openai (falls back to env vars)",
    )
    parser.add_argument(
        "--company-api-url",
        default=None,
        help="Override COMPANY_API_URL env var",
    )
    parser.add_argument(
        "--company-api-key",
        default=None,
        help="Override COMPANY_API_KEY env var",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Sleep between API calls (default: 0.3s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N items (debug)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't call API, just print prompts",
    )
    return parser


def build_llm_client(args: argparse.Namespace) -> Optional[Any]:
    """根据 ``args.provider`` 构造对应的 SDK client，并配置 company 凭据。

    - anthropic → ``anthropic.Anthropic(api_key=...)``
    - openai    → ``openai.OpenAI(api_key=...)``
    - company   → 返回 ``None``（company 路径不需要 SDK client，直接走
                  ``local_api_logger.wrap_requests_call``；本函数会顺便
                  调 :func:`src.api.set_company_credentials` 把 URL/key 注入全局）

    dry-run 时返回 ``None``，脚本应自行处理。
    """
    if getattr(args, "dry_run", False):
        return None

    provider = getattr(args, "provider", "company")

    if provider == "anthropic":
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "provider=anthropic requires `pip install anthropic`"
            ) from e
        key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "anthropic provider needs --api-key or ANTHROPIC_API_KEY env var"
            )
        return anthropic.Anthropic(api_key=key)

    if provider == "openai":
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "provider=openai requires `pip install openai`"
            ) from e
        key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "openai provider needs --api-key or OPENAI_API_KEY env var"
            )
        return OpenAI(api_key=key)

    if provider == "company":
        # Company 路径不需要 SDK client — src.api.llm.call_llm() 会直接走
        # local_api_logger.wrap_requests_call()。我们在这里把 URL/key 注入到
        # src.api 的全局变量，好让后续的 call_llm 拿到。
        url = args.company_api_url or os.environ.get("COMPANY_API_URL", "")
        key = args.company_api_key or os.environ.get("COMPANY_API_KEY", "")
        if not url or not key:
            raise RuntimeError(
                "company provider needs --company-api-url/--company-api-key "
                "or COMPANY_API_URL/COMPANY_API_KEY env vars"
            )
        from src.api import set_company_credentials
        set_company_credentials(url, key)
        return None

    raise ValueError(f"unknown provider: {provider!r}")


__all__ = ["add_llm_args", "build_llm_client"]
