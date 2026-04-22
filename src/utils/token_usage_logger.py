"""Per-API-call JSONL 审计日志 —— 仅作为 :mod:`src.utils.token_logger` 的补充。

**铁律 1（CLAUDE.md）**：任何调用 LLM API 的脚本**必须**在结束时调用
:func:`src.utils.token_logger.log_run` 做最终 per-run 记账（SQLite，管理层报表）。
本模块提供的是**可选的 per-call 细粒度 JSONL 审计**，用于：

- 调试单次 API 调用的 prompt / response / token 分布
- 追溯某条 query 的生成来自哪次 API 请求

重复信息是故意的：``log_run()`` 聚合整个脚本运行的总 token，而 ``TokenUsageLogger``
逐条记录每次调用。**两者必须同时使用**，不能用本模块替代 ``log_run()``。

如果你只是想合规，直接用 ``log_run()`` 就够了，不需要引本模块。
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional


class TokenUsageLogger:
    """Thread-safe JSONL logger for LLM token usage auditing.

    **这不是 token_logger.log_run() 的替代品** —— 只是 per-call 细粒度补充。
    脚本仍然必须在结束时调用 ``log_run()`` 做最终记账。
    """

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]

    def log(
        self,
        *,
        provider: str,
        model: str,
        operation: str,
        prompt: str,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        response_chars: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "operation": operation,
            "prompt_hash": self._hash_text(prompt),
            "prompt_chars": len(prompt or ""),
            "response_chars": response_chars,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": (
                (input_tokens or 0) + (output_tokens or 0)
                if (input_tokens is not None or output_tokens is not None)
                else None
            ),
            "metadata": metadata or {},
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.log_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
