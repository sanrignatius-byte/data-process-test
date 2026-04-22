"""LLM 调用层入口 —— 所有实现都在 :mod:`src.api.llm`。

这个包的 __init__ 只做 re-export，让 ``from src.api import call_llm, parse_json``
这种既有写法继续工作，同时也支持 ``from src.api.llm import call_llm`` 的新写法。

**铁律（不准僭越）**：company provider 的 HTTP 请求必须全部走
``local_api_logger.wrap_requests_call``。禁止在本包或任何脚本里直接 ``requests.post``
调 ``yunwu.ai`` endpoint —— 那会绕过 token 记账。
"""

from src.api.llm import (
    call_llm,
    collect_company_stream,
    extract_json,
    get_company_credentials,
    parse_json,
    set_company_credentials,
)

__all__ = [
    "call_llm",
    "collect_company_stream",
    "extract_json",
    "get_company_credentials",
    "parse_json",
    "set_company_credentials",
]
