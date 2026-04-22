"""统一 CLI 基元 —— Phase 2 产物。

目前只提供 :mod:`src.cli.common_args` 一个 helper，把 ``--provider / --model /
--api-key / --company-api-url / --delay / --limit / --dry-run`` 这一套到处重复
注册的参数收口到一个地方。

**铁律提醒**：company provider 的实际 HTTP 调用仍然走 :mod:`src.api.llm` →
``local_api_logger.wrap_requests_call``，本模块只负责参数解析和 client 构建，
绝不直接发起请求。
"""

from src.cli.common_args import add_llm_args, build_llm_client

__all__ = ["add_llm_args", "build_llm_client"]
